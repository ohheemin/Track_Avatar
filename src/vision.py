#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

sys.path.insert(0, "/home/ohheemin/.local/lib/python3.10/site-packages")

import time, math
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
from collections import deque

COLOR_W, COLOR_H, FPS = 848, 480, 30
DEPTH_W, DEPTH_H      = 848, 480

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_pinky","right_pinky",
    "left_index","right_index",
    "left_thumb","right_thumb",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
IDX_L_ELBOW    = 13
IDX_R_ELBOW    = 14
IDX_L_WRIST    = 15
IDX_R_WRIST    = 16
IDX_L_HIP      = 23
IDX_R_HIP      = 24

IDX_H_WRIST      = 0
IDX_H_PINKY_MCP  = 17
IDX_H_INDEX_MCP  = 5
MID_FINGER_MCP   = 9


# hip 아래 관절 인덱스 (표시 제외)
IDX_HIP_AND_BELOW = set(range(23, 33))

HAND_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

ARM_LEFT  = [IDX_L_SHOULDER, IDX_L_ELBOW, IDX_L_WRIST]
ARM_RIGHT = [IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST]
ARM_ALL   = list(set(ARM_LEFT + ARM_RIGHT))

# hip(23,24)까지만 표시
DISPLAY_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]

COLOR_JOINT   = (0,   255, 100)
COLOR_TEXT    = (255, 255,   0)
COLOR_TEXT_BG = (0,   0,     0)
COLOR_SKEL    = (100, 200, 255)
COLOR_LH_CV   = (255, 180,  50)
COLOR_RH_CV   = (50,  180, 255)

# ═══ 노이즈 필터 파라미터 ═════════════════════════════════════════════════════
MOVING_AVG_N  = 5
DEADBAND_DEG  = 1.0
DEADBAND_RAD  = math.radians(DEADBAND_DEG)
KF_Q          = 1e-3
KF_R          = 1e-1


class KalmanFilter1D:
    """1차원 칼만 필터 (등속 모델, state = [angle, angular_velocity])."""
    def __init__(self, q: float = KF_Q, r: float = KF_R):
        dt = 1.0 / FPS
        self.F = np.array([[1.0, dt ], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[q, 0.0], [0.0, q]])
        self.R = np.array([[r]])
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0
        self.initialized = False

    def update(self, z: float) -> float:
        if not self.initialized:
            self.x[0, 0] = z
            self.initialized = True
            return z
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        S      = self.H @ P_pred @ self.H.T + self.R
        K      = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (np.array([[z]]) - self.H @ x_pred)
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        return float(self.x[0, 0])


class JointFilter:
    def __init__(self, n_joints: int = 6,
                 window: int = MOVING_AVG_N,
                 deadband: float = DEADBAND_RAD):
        self.n        = n_joints
        self.deadband = deadband
        self.kf       = [KalmanFilter1D() for _ in range(n_joints)]
        self.buffers  = [deque(maxlen=window) for _ in range(n_joints)]
        self.last_pub = [0.0] * n_joints

    def update(self, thetas: list) -> list:
        filtered = []
        for i, th in enumerate(thetas):
            kf_out = self.kf[i].update(th)
            self.buffers[i].append(kf_out)
            avg = float(np.mean(self.buffers[i]))
            if abs(avg - self.last_pub[i]) < self.deadband:
                filtered.append(self.last_pub[i])
            else:
                self.last_pub[i] = avg
                filtered.append(avg)
        return filtered


# ═══ ROS2 노드 ═══════════════════════════════════════════════════════════════
class VisionPublisher(Node):
    
    JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]

    def __init__(self):
        super().__init__("vision_publisher")
        self._joint_pub = self.create_publisher(JointState, "/robot/joint_states", 10)
        self._rhand_pub = self.create_publisher(Bool, "/hand_open/right", 10)
        self.get_logger().info(" Ready to publish /robot/joint_states ")
        #self.frame_idx = None

    def publish_joint_angles(self, thetas: list, frame_idx: int = 0):
        # #frame_idx = None
        # msg = JointState()
        # msg.header.stamp    = self.get_clock().now().to_msg()
        # msg.header.frame_id = "base_link"
        # msg.name            = self.JOINT_NAMES
        # msg.position        = [float(t) for t in thetas]
        # msg.effort          = [float(frame_idx)]*len(thetas)
        # self._joint_pub.publish(msg)

        msg = JointState()
        msg.header.stamp    = self.get_clock().now().to_msg()  
        msg.header.frame_id = "base_link"
        msg.name            = self.JOINT_NAMES
        msg.position        = [float(t) for t in thetas]
        msg.effort          = [float(frame_idx)] * len(thetas) 
        self._joint_pub.publish(msg)

    def publish_hand(self, side: str, is_open: bool):
        if side == "right":
            msg = Bool()
            msg.data = bool(is_open)
            self._rhand_pub.publish(msg)

def rotation_x(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[1,  0,   0,  0],
                     [0,  ct, -st, 0],
                     [0,  st,  ct, 0],
                     [0,  0,   0,  1]], dtype=float)

def rotation_y(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ ct, 0, st, 0],
                     [  0, 1,  0, 0],
                     [-st, 0, ct, 0],
                     [  0, 0,  0, 1]], dtype=float)

def rotation_z(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, 0],
                     [st,  ct, 0, 0],
                     [ 0,   0, 1, 0],
                     [ 0,   0, 0, 1]], dtype=float)

def dh_parameter(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ ct, -st*ca,  st*sa, a*ct],
        [ st,  ct*ca, -ct*sa, a*st],
        [  0,     sa,     ca,    d],
        [  0,      0,      0,    1]
    ], dtype=float)

def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _build_global_frame(pose_lms):
    if pose_lms is None:
        return None, None, None

    lms = pose_lms.landmark
    def _mp(i): return np.array([lms[i].x, lms[i].y, lms[i].z])

    ls = _mp(IDX_L_SHOULDER); rs = _mp(IDX_R_SHOULDER)
    lh = _mp(IDX_L_HIP);      rh = _mp(IDX_R_HIP)

    sho_mid = (ls + rs) / 2
    hip_mid = (lh + rh) / 2

    X_g = _safe_unit(rs - ls)
    Z_g = _safe_unit(hip_mid - sho_mid)
    Z_g = _safe_unit(Z_g - float(np.dot(Z_g, X_g)) * X_g)
    Y_g = _safe_unit(np.cross(Z_g, X_g))

    return X_g, Y_g, Z_g


def build_transform_matrix(arm_coords: dict, pose_lms) -> np.ndarray:
    X_g, Y_g, Z_g = _build_global_frame(pose_lms)
    if X_g is None:
        return np.eye(4)

    sh = arm_coords.get(IDX_R_SHOULDER)
    p  = np.array(sh, dtype=float) if sh is not None else np.zeros(3)
    n, o, a = X_g, Y_g, Z_g

    return np.array([
        [n[0], o[0], a[0], p[0]],
        [n[1], o[1], a[1], p[1]],
        [n[2], o[2], a[2], p[2]],
        [0.0,  0.0,  0.0,  1.0 ],
    ])


def cam_to_global(arm_coords: dict, global_matrix: np.ndarray, joint_index: int):
    gl_joint = arm_coords.get(joint_index)
    if gl_joint is None:
        return None
    inv_GM   = np.linalg.inv(global_matrix)
    pt_col_v = np.array([[gl_joint[0]],
                         [gl_joint[1]],
                         [gl_joint[2]],
                         [1.0]], dtype=float)
    v = inv_GM @ pt_col_v
    return np.array([v[0,0], v[1,0], v[2,0]])


def cam_to_global_hand(rhand_pts: list, global_matrix: np.ndarray, joint_index: int):
    gl_joint = rhand_pts[joint_index]
    if gl_joint is None:
        return None
    inv_GM   = np.linalg.inv(global_matrix)
    pt_col_v = np.array([[gl_joint[0]],
                         [gl_joint[1]],
                         [gl_joint[2]],
                         [1.0]], dtype=float)
    v = inv_GM @ pt_col_v
    return np.array([v[0,0], v[1,0], v[2,0]])


################################################################
def arm_joint_angles(arm_coords: dict, global_matrix, joint=None):
    if joint is None:
        print("Joint # is Needed!!!")
        return 0.0
    if joint < 1 or joint > 6:
        print("Wrong Joint # is Inserted!!!")
        return 0.0

    el = arm_coords.get(IDX_R_ELBOW)
    if el is None:
        print("Elbow coordinates are missing!!!")
        return 0.0

    inv_GM    = np.linalg.inv(global_matrix)
    arm_col_v = np.array([[el[0]], [el[1]], [el[2]], [1.0]], dtype=float)
    local_v   = inv_GM @ arm_col_v
    x_arm = local_v[0, 0]
    y_arm = local_v[1, 0]
    z_arm = local_v[2, 0]

    if joint == 1:
        return np.arctan2(-y_arm, z_arm)
    elif joint == 2:
        r = math.sqrt(x_arm**2 + y_arm**2 + z_arm**2)
        if r < 1e-6:
            return 0.0
        angle_2 = np.arccos(math.sqrt(y_arm**2 + z_arm**2) / r)
        if x_arm < 0:
            angle_2 = -angle_2
        return angle_2

    return 0.0
################################################################


def compute_dh_joint_angles(arm_coords: dict, pose_lms, rhand_pts: list):
    """각도 6개와 (el_vec_cam, wr_vec_cam) 튜플을 함께 반환."""
    nan6     = [0.0] * 6
    nan_vecs = (None, None)

    X_g, Y_g, Z_g = _build_global_frame(pose_lms)
    if X_g is None:
        return nan6, nan_vecs

    T      = build_transform_matrix(arm_coords, pose_lms)
    theta1 = arm_joint_angles(arm_coords, T, joint=1)
    theta2 = arm_joint_angles(arm_coords, T, joint=2)

    noa_frame = rotation_x(theta1) @ rotation_y(theta2)
    n_vec = noa_frame[:3, 0]
    o_vec = noa_frame[:3, 1]

    sh = cam_to_global(arm_coords, T, IDX_R_SHOULDER)
    el = cam_to_global(arm_coords, T, IDX_R_ELBOW)
    wr = cam_to_global(arm_coords, T, IDX_R_WRIST)
    if sh is None or el is None or wr is None:
        return nan6, nan_vecs

    # θ3: 팔꿈치 굴곡 평면 방향
    l_arm  = wr - el
    h_arm  = el
    el_vec = _safe_unit(np.cross(h_arm, l_arm))
    j3_rot_def = np.dot(el_vec, o_vec)
    if j3_rot_def > 0:
        theta3 = np.arccos(np.clip(np.dot(el_vec, n_vec), -1.0, 1.0))
    else:
        theta3 = -np.arccos(np.clip(np.dot(el_vec, n_vec), -1.0, 1.0))

    # θ4: 팔꿈치 굴곡 크기
    l_unit = _safe_unit(l_arm)
    h_unit = _safe_unit(h_arm)
    theta4 = np.arccos(np.clip(np.dot(l_unit, h_unit), -1.0, 1.0))

    # θ5, θ6: 오른손이 감지된 경우에만 계산
    theta5 = 0.0
    theta6 = 0.0
    wr_vec = None
    wr_vec_prev = None

    if rhand_pts and len(rhand_pts) >= 21:
        pk = cam_to_global_hand(rhand_pts, T, IDX_H_PINKY_MCP)
        id = cam_to_global_hand(rhand_pts, T, IDX_H_INDEX_MCP)
        wh = cam_to_global_hand(rhand_pts, T, IDX_H_WRIST)
        md = cam_to_global_hand(rhand_pts, T, MID_FINGER_MCP)

        if pk is not None and id is not None and wh is not None and md is not None:
            p_vec  = pk - wh
            i_vec  = id - wh
            m_vec  = md - wh

            m_unit = _safe_unit(m_vec)
            theta_abs_6 = np.arccos(np.clip(np.dot(m_unit, l_unit), -1.0, 1.0))

            if wr_vec == None:
                wr_vec = _safe_unit(np.cross(p_vec, i_vec))
                wr_vec_prev = wr_vec
            else:
                if wr_vec[0]*wr_vec_prev[0] + wr_vec[1]*wr_vec_prev[1] + wr_vec[2]*wr_vec_prev[2] > 0:
                    wr_vec_prev = wr_vec
                    wr_vec = _safe_unit(np.cross(p_vec, i_vec))
                else:
                    wr_vec = wr_vec_prev

            j6_rot_def = np.dot(wr_vec, l_unit)
            if j6_rot_def > 0:
                theta6 = - theta_abs_6
            else:
                theta6 = theta_abs_6

            threshold = np.cos(np.radians(105))

            wr_axis = _safe_unit(np.cross(m_unit, wr_vec))
            th5_vec = _safe_unit(wr_vec * np.cos(-theta6) + np.cross(wr_axis, wr_vec) * np.sin(-theta6))
            theta_abs_5 = np.arccos(np.clip(np.dot(th5_vec, el_vec), -1.0, 1.0))
            th5_axis = _safe_unit(np.cross(el_vec, th5_vec))
            j5_rot_def = np.dot(th5_axis, l_unit)

            if j6_rot_def > threshold:
                if j5_rot_def > 0:
                    theta5 = theta_abs_5
                else:
                    theta5 = - theta_abs_5
            else:
                if th5_vec[2] > 0:
                    theta5 = theta5 + 0.03
                else:
                    theta5 = theta5 - 0.03

    # 글로벌 좌표계 → 카메라 좌표계로 역변환 (회전만)
    R_mat      = T[:3, :3]
    el_vec_cam = R_mat @ el_vec if el_vec is not None else None
    wr_vec_cam = R_mat @ wr_vec if wr_vec is not None else None

    return [theta1, theta2, theta3, theta4, 0.3 * theta5, -theta6], (el_vec_cam, wr_vec_cam)


def _project(pt3d, intrinsics):
    x, y, z = pt3d
    if z <= 1e-4:
        return None
    px = int(intrinsics.fx * x / z + intrinsics.ppx)
    py = int(intrinsics.fy * y / z + intrinsics.ppy)
    return (px, py)


def draw_body_axes(disp, arm_coords, pose_lms, intrinsics, w, h,
                   el_vec_cam=None, wr_vec_cam=None):
    X_g, Y_g, Z_g = _build_global_frame(pose_lms)
    if X_g is None:
        return

    AXIS_INFO = [
        ("X(n)", X_g, (80,  80,  255)),
        ("Y(o)", Y_g, (80,  255, 80 )),
        ("Z(a)", Z_g, (255, 80,  80 )),
    ]
    AXIS_LEN = 0.15
    lms = pose_lms.landmark

    def lm_px(idx):
        lm = lms[idx]
        return (int(lm.x * w), int(lm.y * h))

    ls_px  = lm_px(IDX_L_SHOULDER); rs_px = lm_px(IDX_R_SHOULDER)
    lh_px  = lm_px(IDX_L_HIP);      rh_px = lm_px(IDX_R_HIP)
    sho_px = ((ls_px[0]+rs_px[0])//2, (ls_px[1]+rs_px[1])//2)
    hip_px = ((lh_px[0]+rh_px[0])//2, (lh_px[1]+rh_px[1])//2)

    cv2.circle(disp, sho_px, 7, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(disp, sho_px, 9, (255, 255, 255), 1,  cv2.LINE_AA)
    draw_text_with_bg(disp, "Sho.Mid", (sho_px[0]+8, sho_px[1]-10),
                      font_scale=0.38, text_color=(0, 255, 255))
    cv2.circle(disp, hip_px, 7, (255, 128, 0), -1, cv2.LINE_AA)
    cv2.circle(disp, hip_px, 9, (255, 255, 255), 1, cv2.LINE_AA)
    draw_text_with_bg(disp, "Hip.Mid", (hip_px[0]+8, hip_px[1]-10),
                      font_scale=0.38, text_color=(255, 128, 0))
    cv2.line(disp, sho_px, hip_px, (200, 200, 200), 1, cv2.LINE_AA)

    origin_3d = arm_coords.get(IDX_R_SHOULDER)
    if origin_3d is not None:
        origin_3d_arr = np.array(origin_3d)
        o2d = _project(origin_3d_arr, intrinsics)
        if o2d is None:
            return

        # ── 기존 X/Y/Z 축 ──────────────────────────────────────────────────
        for name, vec, col in AXIS_INFO:
            end_3d = origin_3d_arr + vec * AXIS_LEN
            e2d    = _project(end_3d, intrinsics)
            if e2d is None:
                continue
            cv2.arrowedLine(disp, o2d, e2d, col, 2, cv2.LINE_AA, tipLength=0.25)
            draw_text_with_bg(disp, name, (e2d[0]+4, e2d[1]+4),
                              font_scale=0.5, thickness=1,
                              text_color=col, bg_color=(0,0,0))

        # ── el_vec: 팔꿈치 원점, 청록색 ────────────────────────────────────
        if el_vec_cam is not None:
            el_origin = arm_coords.get(IDX_R_ELBOW)
            if el_origin is not None:
                el_origin_arr = np.array(el_origin)
                el_o2d = _project(el_origin_arr, intrinsics)
                el_end = el_origin_arr + el_vec_cam * AXIS_LEN
                el_e2d = _project(el_end, intrinsics)
                if el_o2d is not None and el_e2d is not None:
                    cv2.arrowedLine(disp, el_o2d, el_e2d,
                                    (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.25)
                    draw_text_with_bg(disp, "el_vec",
                                      (el_e2d[0]+4, el_e2d[1]+4),
                                      font_scale=0.45, text_color=(0, 220, 255))

        # ── wr_vec: 손목 원점, 보라색 ──────────────────────────────────────
        if wr_vec_cam is not None:
            wr_origin = arm_coords.get(IDX_R_WRIST)
            if wr_origin is not None:
                wr_origin_arr = np.array(wr_origin)
                wr_o2d = _project(wr_origin_arr, intrinsics)
                wr_end = wr_origin_arr + wr_vec_cam * AXIS_LEN
                wr_e2d = _project(wr_end, intrinsics)
                if wr_o2d is not None and wr_e2d is not None:
                    cv2.arrowedLine(disp, wr_o2d, wr_e2d,
                                    (255, 100, 220), 2, cv2.LINE_AA, tipLength=0.25)
                    draw_text_with_bg(disp, "wr_vec",
                                      (wr_e2d[0]+4, wr_e2d[1]+4),
                                      font_scale=0.45, text_color=(255, 100, 220))

        cv2.circle(disp, o2d, 5, (255, 255, 255), -1, cv2.LINE_AA)
        draw_text_with_bg(disp, "R.Sh", (o2d[0]+8, o2d[1]-10),
                          font_scale=0.40, text_color=(255, 255, 255))

    else:
        # 폴백: 뎁스 없을 때 픽셀 근사
        rs_lm  = lms[IDX_R_SHOULDER]
        ox, oy = int(rs_lm.x * w), int(rs_lm.y * h)
        SCALE  = 80
        def to_screen(vec3):
            sx = int(ox + vec3[0] * SCALE - vec3[2] * SCALE * 0.3)
            sy = int(oy + vec3[1] * SCALE - vec3[2] * SCALE * 0.3)
            return (sx, sy)
        for name, vec, col in AXIS_INFO:
            ep = to_screen(vec)
            cv2.arrowedLine(disp, (ox, oy), ep, col, 2, cv2.LINE_AA, tipLength=0.25)
            draw_text_with_bg(disp, name, (ep[0]+4, ep[1]+4),
                              font_scale=0.5, text_color=col)
        cv2.circle(disp, (ox, oy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        draw_text_with_bg(disp, "R.Sh", (ox+8, oy-10),
                          font_scale=0.40, text_color=(255, 255, 255))


FINGER_TIP_PIP     = [(8,6),(12,10),(16,14),(20,18)]
FIST_FINGER_THRESH = 3

def is_hand_open(hand_lms, hand_pts_3d=None) -> bool:
    if hand_lms is None:
        return False
    lm = hand_lms.landmark

    def _count_folded(pts):
        wrist  = np.array(pts[0])
        folded = valid = 0
        for ti, pi in FINGER_TIP_PIP:
            t, p = pts[ti], pts[pi]
            if t is None or p is None:
                continue
            valid += 1
            if np.linalg.norm(np.array(t)-wrist) <= np.linalg.norm(np.array(p)-wrist):
                folded += 1
        return folded, valid

    if hand_pts_3d and len(hand_pts_3d) == 21 and hand_pts_3d[0] is not None:
        f, v = _count_folded(hand_pts_3d)
        if v >= 3:
            return f < FIST_FINGER_THRESH

    norm_pts = [np.array([lm[i].x, lm[i].y, lm[i].z]) for i in range(21)]
    f, _ = _count_folded(norm_pts)
    return f < FIST_FINGER_THRESH


def now_ms():
    return time.time() * 1000.0

def get_3d_point(depth_frame, intrinsics, px, py):
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if px < 0 or py < 0 or px >= dw or py >= dh:
        return None
    d = depth_frame.get_distance(int(px), int(py))
    if d <= 0.1 or d > 6.0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], d)

def collect_hand_3d(hand_lms, depth_frame, intrinsics, img_w, img_h):
    return [get_3d_point(depth_frame, intrinsics,
                         int(lm.x*img_w), int(lm.y*img_h))
            for lm in hand_lms.landmark]

def draw_text_with_bg(img, text, pos, font_scale=0.38, thickness=1,
                      text_color=COLOR_TEXT, bg_color=COLOR_TEXT_BG, alpha=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    ov = img.copy()
    cv2.rectangle(ov,(x-2,y-th-2),(x+tw+2,y+bl+2),bg_color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
    cv2.putText(img,text,(x,y),font,font_scale,text_color,thickness,cv2.LINE_AA)

def draw_hand_on_frame(disp, hand_lms, img_w, img_h, cv_color, depth_frame, intrinsics):
    pts_px = [(int(lm.x*img_w), int(lm.y*img_h)) for lm in hand_lms.landmark]
    for a,b in HAND_CONNECTIONS:
        cv2.line(disp, pts_px[a], pts_px[b], cv_color, 1, cv2.LINE_AA)
    for i,(px,py) in enumerate(pts_px):
        pt3d = get_3d_point(depth_frame, intrinsics, px, py)
        r = 4 if i in {0,4,8,12,16,20} else 2
        cv2.circle(disp,(px,py),r,cv_color,-1,cv2.LINE_AA)
        if i in {0,4,8,12,16,20} and pt3d:
            draw_text_with_bg(disp,
                f"{HAND_NAMES[i]}({pt3d[0]:+.2f},{pt3d[2]:.2f})",
                (px+4,py+4),font_scale=0.30,text_color=(255,255,255))


def main():
    rclpy.init()
    ros_node   = VisionPublisher()
    jnt_filter = JointFilter()

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  FPS)
    print("RealSense connecting...")
    profile    = pipeline.start(cfg)
    align      = rs.align(rs.stream.color)
    intrinsics = (profile.get_stream(rs.stream.color)
                         .as_video_stream_profile().get_intrinsics())
    print(f"{COLOR_W}x{COLOR_H}@{FPS}fps fx={intrinsics.fx:.1f}")
    print(f"Filter: Kalman(Q={KF_Q},R={KF_R})  MovingAvg(N={MOVING_AVG_N})  Deadband({DEADBAND_DEG:.1f}deg)")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    arm_coords     = {}
    lhand_pts      = []
    rhand_pts      = []
    frame_lat_buf  = deque(maxlen=30)
    lat_frame_ms   = 0.0
    lat_frame_avg  = 0.0
    show_depth     = False
    depth_win_open = False
    frame_idx      = 0

    print("Running — Q/ESC:Quit  S:Save  D:Depth")
    print("ROS2: /robot/joint_states  /hand_open/right")

    try:
        while True:
            rclpy.spin_once(ros_node, timeout_sec=0)

            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            t0           = color_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
            color_img    = np.asanyarray(color_frame.get_data())
            lat_frame_ms = now_ms() - t0
            frame_lat_buf.append(lat_frame_ms)
            lat_frame_avg = sum(frame_lat_buf) / len(frame_lat_buf)

            depth_arr = np.asanyarray(depth_frame.get_data())
            h, w = color_img.shape[:2]
            disp = color_img.copy()

            rgb    = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            cur_arm = {}
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark

                # 스켈레톤: hip 아래 연결선 제외
                for ca, cb in POSE_CONNECTIONS:
                    if ca in IDX_HIP_AND_BELOW or cb in IDX_HIP_AND_BELOW:
                        continue
                    a, b = lms[ca], lms[cb]
                    if a.visibility > 0.3 and b.visibility > 0.3:
                        cv2.line(disp, (int(a.x*w), int(a.y*h)),
                                 (int(b.x*w), int(b.y*h)), COLOR_SKEL, 2, cv2.LINE_AA)

                # 관절 점 & 라벨: hip 이하 제외
                for idx in DISPLAY_INDICES:
                    lm = lms[idx]
                    if lm.visibility < 0.3:
                        continue
                    px, py = int(lm.x*w), int(lm.y*h)
                    pt3d   = get_3d_point(depth_frame, intrinsics, px, py)
                    if idx in ARM_ALL and pt3d is not None:
                        cur_arm[idx] = pt3d
                    cv2.circle(disp, (px,py), 5, COLOR_JOINT, -1, cv2.LINE_AA)
                    cv2.circle(disp, (px,py), 7, (255,255,255), 1, cv2.LINE_AA)
                    name = LANDMARK_NAMES[idx]
                    if pt3d:
                        draw_text_with_bg(disp,
                            f"{name}({pt3d[0]:+.2f},{pt3d[1]:+.2f},{pt3d[2]:.2f})m",
                            (px+8, py+4))
                    else:
                        draw_text_with_bg(disp,
                            f"{name} MP({lm.x:.2f},{lm.y:.2f})", (px+8, py+4))

                # 기타 상체 소점 (hip 아래 제외)
                for idx2, lm2 in enumerate(lms):
                    if idx2 in DISPLAY_INDICES or idx2 in IDX_HIP_AND_BELOW:
                        continue
                    if lm2.visibility < 0.4:
                        continue
                    cv2.circle(disp, (int(lm2.x*w), int(lm2.y*h)),
                               3, (180,180,180), -1, cv2.LINE_AA)

            for idx in ARM_ALL:
                if idx in cur_arm:
                    arm_coords[idx] = cur_arm[idx]

            if result.left_hand_landmarks:
                lhand_pts = collect_hand_3d(result.left_hand_landmarks, depth_frame, intrinsics, w, h)
                draw_hand_on_frame(disp, result.left_hand_landmarks, w, h, COLOR_LH_CV, depth_frame, intrinsics)

            if result.right_hand_landmarks:
                rhand_pts = collect_hand_3d(result.right_hand_landmarks, depth_frame, intrinsics, w, h)
                draw_hand_on_frame(disp, result.right_hand_landmarks, w, h, COLOR_RH_CV, depth_frame, intrinsics)

            # ── 관절각 계산 → 필터 → publish ──────────────────────────────────
            raw_thetas, (el_vec_cam, wr_vec_cam) = compute_dh_joint_angles(
                arm_coords, result.pose_landmarks, rhand_pts)

            kf_thetas       = [jnt_filter.kf[i].x[0, 0] if jnt_filter.kf[i].initialized
                               else raw_thetas[i] for i in range(6)]
            filtered_thetas = jnt_filter.update(raw_thetas)
            ros_node.publish_joint_angles(filtered_thetas, frame_idx)

            T = build_transform_matrix(arm_coords, result.pose_landmarks)

            r_open = is_hand_open(result.right_hand_landmarks,
                                  rhand_pts if result.right_hand_landmarks else None)
            ros_node.publish_hand("right", r_open)
            l_open = is_hand_open(result.left_hand_landmarks,
                                  lhand_pts if result.left_hand_landmarks else None)

            draw_body_axes(disp, arm_coords, result.pose_landmarks, intrinsics, w, h,
                           el_vec_cam=el_vec_cam, wr_vec_cam=wr_vec_cam)

            # ── 오버레이: raw / kalman / pub ──────────────────────────────────
            y_ang  = 70
            labels = ["th1","th2","th3","th4","th5","th6"]

            draw_text_with_bg(disp, "[raw]", (w-230, y_ang),
                              font_scale=0.34, text_color=(180,180,100))
            y_ang += 14
            for lbl, th in zip(labels, raw_thetas):
                draw_text_with_bg(disp, f"{lbl}:{math.degrees(th):+7.2f}",
                                  (w-230, y_ang), font_scale=0.26, text_color=(180,180,130))
                y_ang += 12

            y_ang += 4
            draw_text_with_bg(disp, "[kalman]", (w-230, y_ang),
                              font_scale=0.34, text_color=(100,220,255))
            y_ang += 14
            for lbl, th in zip(labels, kf_thetas):
                draw_text_with_bg(disp, f"{lbl}:{math.degrees(th):+7.2f}",
                                  (w-230, y_ang), font_scale=0.26, text_color=(120,200,255))
                y_ang += 12

            y_ang += 4
            draw_text_with_bg(disp, "[pub] kalman+avg+dead", (w-230, y_ang),
                              font_scale=0.34, text_color=(255,255,100))
            y_ang += 14
            for lbl, th in zip(labels, filtered_thetas):
                draw_text_with_bg(disp, f"{lbl}:{math.degrees(th):+7.2f}",
                                  (w-230, y_ang), font_scale=0.26, text_color=(255,220,180))
                y_ang += 12

            y_mat = y_ang + 6
            draw_text_with_bg(disp, "T (n|o|a|p) 4x4", (w-230, y_mat),
                              font_scale=0.34, text_color=(180, 255, 180))
            y_mat += 14
            for r_i in range(4):
                row_str = " ".join(f"{T[r_i,c_i]:+.3f}" for c_i in range(4))
                draw_text_with_bg(disp, f"[{row_str}]",
                                  (w-230, y_mat), font_scale=0.26, text_color=(200, 255, 200))
                y_mat += 12

            draw_text_with_bg(disp, "[ Axis ]",    (w-120, y_mat+10),
                              font_scale=0.38, text_color=(200,200,200))
            draw_text_with_bg(disp, "X(n): Right", (w-120, y_mat+24),
                              font_scale=0.34, text_color=(80,  80,  255))
            draw_text_with_bg(disp, "Y(o): Back",  (w-120, y_mat+38),
                              font_scale=0.34, text_color=(80,  255, 80))
            draw_text_with_bg(disp, "Z(a): Down",  (w-120, y_mat+52),
                              font_scale=0.34, text_color=(255, 80,  80))
            draw_text_with_bg(disp, "el_vec: Elbow plane nrm", (w-120, y_mat+66),
                              font_scale=0.34, text_color=(0, 220, 255))
            draw_text_with_bg(disp, "wr_vec: Hand palm nrm",  (w-120, y_mat+80),
                              font_scale=0.34, text_color=(255, 100, 220))
            draw_text_with_bg(disp,
                f"Avg N={MOVING_AVG_N}  Dead={DEADBAND_DEG:.1f}deg",
                (w-230, y_mat+96), font_scale=0.32, text_color=(180,180,255))

            draw_text_with_bg(disp,
                f"L-hand: {'open' if l_open else 'fist'}",
                (w-190, h-50), font_scale=0.44,
                text_color=(100,220,255) if l_open else (180,180,180))
            draw_text_with_bg(disp,
                f"R-hand: {'open' if r_open else 'fist'} [pub]",
                (w-190, h-30), font_scale=0.44,
                text_color=(100,255,150) if r_open else (200,100,100))

            draw_text_with_bg(disp,
                f"Frame Lat:{lat_frame_ms:5.1f}ms avg{lat_frame_avg:5.1f}ms",
                (10, h-50), font_scale=0.42, text_color=(100,255,180))
            draw_text_with_bg(disp, f"Frame:{frame_idx}", (10, 22),
                              font_scale=0.45, text_color=(200,255,200))
            draw_text_with_bg(disp, "Q/ESC:Quit S:Save D:Depth", (10, 44),
                              font_scale=0.40, text_color=(200,200,200))

            detected  = "Detected" if result.pose_landmarks else "No Person"
            dot_color = (0,255,0) if result.pose_landmarks else (0,0,255)
            cv2.circle(disp, (w-20, 20), 8, dot_color, -1)
            draw_text_with_bg(disp, detected, (w-110, 24),
                              font_scale=0.40, text_color=(255,255,255))

            for s, lms_flag, col in [
                ("L-hand O" if result.left_hand_landmarks else "L-hand X",
                 result.left_hand_landmarks, (200,200,255)),
                ("R-hand O" if result.right_hand_landmarks else "R-hand X",
                 result.right_hand_landmarks, (255,200,200)),
            ]:
                draw_text_with_bg(disp, s, (w-120, 44 if "L" in s else 62),
                                  font_scale=0.38,
                                  text_color=col if lms_flag else (100,100,100))

            y_off = h - 100
            for arm_name, indices, col in [("LEFT ARM",  ARM_LEFT,  (200,200,255)),
                                            ("RIGHT ARM", ARM_RIGHT, (255,200,200))]:
                draw_text_with_bg(disp, arm_name, (10, y_off),
                                  font_scale=0.38, text_color=col)
                y_off -= 16
                for idx in reversed(indices):
                    pt  = arm_coords.get(idx)
                    jn  = LANDMARK_NAMES[idx].split("_")[-1]
                    txt = (f"  {jn}:({pt[0]:+.2f},{pt[1]:+.2f},{pt[2]:.2f})m"
                           if pt else f"  {jn}: --")
                    draw_text_with_bg(disp, txt, (10, y_off),
                                      font_scale=0.32, text_color=col)
                    y_off -= 13
                y_off -= 5

            if frame_idx % 10 == 0:
                raw_str = " ".join(f"{math.degrees(t):+.1f}" for t in raw_thetas)
                kf_str  = " ".join(f"{math.degrees(t):+.1f}" for t in kf_thetas)
                pub_str = " ".join(f"{math.degrees(t):+.1f}" for t in filtered_thetas)
                print(f"[{frame_idx:05d}] Lat:{lat_frame_ms:.0f}ms")
                print(f"  raw:   [{raw_str}]")
                print(f"  kalman:[{kf_str}]")
                print(f"  pub:   [{pub_str}]")
                print(f"  T=\n{np.round(T, 4)}")

            cv2.imshow("VISION", disp)

            if show_depth:
                dc = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_arr, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("Depth Map", dc)
                depth_win_open = True
            elif depth_win_open:
                try:
                    cv2.destroyWindow("Depth Map")
                except Exception:
                    pass           
                depth_win_open = False

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                fn = f"pose_capture_{frame_idx:04d}.png"
                cv2.imwrite(fn, disp)
                print(f"[SAVE] {fn}")
            elif key == ord("d"):
                show_depth = not show_depth
            frame_idx += 1

    finally:
        pipeline.stop()
        holistic.close()
        cv2.destroyAllWindows()
        ros_node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()