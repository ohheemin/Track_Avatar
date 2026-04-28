import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

sys.path.insert(0, "/home/ohheemin/.local/lib/python3.10/site-packages")

import math, threading
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
from collections import deque

COLOR_W, COLOR_H, FPS = 848, 480, 60
DEPTH_W, DEPTH_H      = 848, 480

PUBLISH_HZ     = 20               # 0.05초마다 publish
PUBLISH_PERIOD = 1.0 / PUBLISH_HZ

IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
IDX_L_ELBOW    = 13
IDX_R_ELBOW    = 14
IDX_L_WRIST    = 15
IDX_R_WRIST    = 16
IDX_L_HIP      = 23
IDX_R_HIP      = 24

IDX_H_WRIST     = 0
IDX_H_PINKY_MCP = 17
IDX_H_INDEX_MCP = 5
MID_FINGER_MCP  = 9

ARM_LEFT  = [IDX_L_SHOULDER, IDX_L_ELBOW, IDX_L_WRIST]
ARM_RIGHT = [IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST]
ARM_ALL   = list(set(ARM_LEFT + ARM_RIGHT))

DISPLAY_INDICES    = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FINGER_TIP_PIP     = [(8,6),(12,10),(16,14),(20,18)]
FIST_FINGER_THRESH = 3

MOVING_AVG_N = 10
DEADBAND_DEG = 1.0
DEADBAND_RAD = math.radians(DEADBAND_DEG)
KF_Q         = 1e-3
KF_R         = 1e-1


# ═══ 칼만 필터 ═══════════════════════════════════════════════════════════════
class KalmanFilter1D:
    def __init__(self, q=KF_Q, r=KF_R):
        dt = 1.0 / FPS
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[q, 0.0], [0.0, q]])
        self.R = np.array([[r]])
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.initialized = False

    def update(self, z):
        if not self.initialized:
            self.x[0, 0] = z
            self.initialized = True
            return z
        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.Q
        S  = self.H @ Pp @ self.H.T + self.R
        K  = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K @ (np.array([[z]]) - self.H @ xp)
        self.P = (np.eye(2) - K @ self.H) @ Pp
        return float(self.x[0, 0])


class JointFilter:
    def __init__(self, n=6, window=MOVING_AVG_N, deadband=DEADBAND_RAD):
        self.deadband = deadband
        self.kf       = [KalmanFilter1D() for _ in range(n)]
        self.bufs     = [deque(maxlen=window) for _ in range(n)]
        self.last_pub = [0.0] * n

    def update(self, thetas):
        out = []
        for i, th in enumerate(thetas):
            kf_v = self.kf[i].update(th)
            self.bufs[i].append(kf_v)
            avg = float(np.mean(self.bufs[i]))
            if abs(avg - self.last_pub[i]) < self.deadband:
                out.append(self.last_pub[i])
            else:
                self.last_pub[i] = avg
                out.append(avg)
        return out


# ═══ ROS2 노드 ═══════════════════════════════════════════════════════════════
class HolisticPublisher(Node):
    JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]

    def __init__(self):
        super().__init__("holistic_publisher")
        self._joint_pub = self.create_publisher(JointState, "/robot/joint_states", 10)
        self._rhand_pub = self.create_publisher(Bool, "/hand_open/right", 10)

        self._lock       = threading.Lock()
        self._thetas     = [0.0] * 6
        self._r_open     = False
        self._data_ready = False

        self.create_timer(PUBLISH_PERIOD, self._timer_cb)
        self.get_logger().info(
            f"HolisticPublisher ready — publish {PUBLISH_HZ}Hz "
            f"({PUBLISH_PERIOD*1000:.0f}ms)")

    def update_state(self, thetas, r_open):
        with self._lock:
            self._thetas     = thetas
            self._r_open     = r_open
            self._data_ready = True

    def _timer_cb(self):
        with self._lock:
            if not self._data_ready:
                return
            thetas = list(self._thetas)
            r_open = self._r_open

        stamp = self.get_clock().now().to_msg()

        jmsg = JointState()
        jmsg.header.stamp    = stamp
        jmsg.header.frame_id = "base_link"
        jmsg.name            = self.JOINT_NAMES
        jmsg.position        = [float(t) for t in thetas]
        self._joint_pub.publish(jmsg)

        hmsg      = Bool()
        hmsg.data = bool(r_open)
        self._rhand_pub.publish(hmsg)


# ═══ 회전 행렬 ═══════════════════════════════════════════════════════════════
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


# ═══ 기하 유틸 ═══════════════════════════════════════════════════════════════
def _safe_unit(v):
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

def build_transform_matrix(arm_coords, pose_lms):
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

def cam_to_global(arm_coords, global_matrix, joint_index):
    gl_joint = arm_coords.get(joint_index)
    if gl_joint is None:
        return None
    inv_GM = np.linalg.inv(global_matrix)
    pt     = np.array([[gl_joint[0]], [gl_joint[1]], [gl_joint[2]], [1.0]], dtype=float)
    v      = inv_GM @ pt
    return np.array([v[0,0], v[1,0], v[2,0]])

def cam_to_global_hand(rhand_pts, global_matrix, joint_index):
    gl_joint = rhand_pts[joint_index]
    if gl_joint is None:
        return None
    inv_GM = np.linalg.inv(global_matrix)
    pt     = np.array([[gl_joint[0]], [gl_joint[1]], [gl_joint[2]], [1.0]], dtype=float)
    v      = inv_GM @ pt
    return np.array([v[0,0], v[1,0], v[2,0]])

def arm_joint_angles(arm_coords, global_matrix, joint=None):
    if joint is None or joint < 1 or joint > 6:
        return 0.0
    el = arm_coords.get(IDX_R_ELBOW)
    if el is None:
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


# ═══ 조인트 각도 계산 (프레임 간 상태 유지가 필요한 변수는 클래스로 관리) ═══
class JointAngleEstimator:
    """
    wr_vec_prev, theta5 누적값을 프레임 간에 유지해야 하므로
    함수가 아닌 클래스로 상태를 보존합니다.
    """
    def __init__(self):
        self.wr_vec_prev = None   # 프레임 간 wr_vec 플리핑 방지용
        self.theta5_acc  = 0.0   # theta5 누적값 (j6_rot_def <= threshold 분기)

    def compute(self, arm_coords, pose_lms, rhand_pts):
        nan6 = [0.0] * 6

        X_g, Y_g, Z_g = _build_global_frame(pose_lms)
        if X_g is None:
            return nan6

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
            return nan6

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

        # θ5, θ6
        theta5 = self.theta5_acc   # 이전 프레임 누적값 유지
        theta6 = 0.0

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

                # ── wr_vec 플리핑 방지 (프레임 간 상태 유지) ─────────────
                new_wr_vec = _safe_unit(np.cross(p_vec, i_vec))
                if self.wr_vec_prev is None:
                    # 첫 프레임
                    wr_vec           = new_wr_vec
                    self.wr_vec_prev = new_wr_vec
                else:
                    dot = float(np.dot(new_wr_vec, self.wr_vec_prev))
                    if dot > 0:
                        # 방향 일관 → 업데이트
                        self.wr_vec_prev = new_wr_vec
                        wr_vec           = new_wr_vec
                    else:
                        # 방향 반전 → 이전 값 유지
                        wr_vec = self.wr_vec_prev

                j6_rot_def = np.dot(wr_vec, l_unit)
                theta6     = -theta_abs_6 if j6_rot_def > 0 else theta_abs_6

                threshold   = np.cos(np.radians(105))
                wr_axis     = _safe_unit(np.cross(m_unit, wr_vec))
                th5_vec     = _safe_unit(wr_vec * np.cos(-theta6) +
                                         np.cross(wr_axis, wr_vec) * np.sin(-theta6))
                theta_abs_5 = np.arccos(np.clip(np.dot(th5_vec, el_vec), -1.0, 1.0))
                th5_axis    = _safe_unit(np.cross(el_vec, th5_vec))
                j5_rot_def  = np.dot(th5_axis, l_unit)


                theta5 = theta_abs_5 if j5_rot_def > 0 else -theta_abs_5


                self.theta5_acc = theta5   # 다음 프레임을 위해 저장

        return [theta1, theta2, theta3, theta4, theta5, -theta6]


# ═══ 뎁스 / 손 유틸 ══════════════════════════════════════════════════════════
def get_3d_point(depth_frame, intrinsics, px, py):
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if not (0 <= px < dw and 0 <= py < dh):
        return None
    d = depth_frame.get_distance(int(px), int(py))
    if not (0.1 < d < 6.0):
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], d)

def collect_hand_3d(hand_lms, depth_frame, intrinsics, w, h):
    return [get_3d_point(depth_frame, intrinsics,
                         int(lm.x*w), int(lm.y*h))
            for lm in hand_lms.landmark]

def is_hand_open(hand_lms, hand_pts_3d=None):
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


# ═══ 메인 ════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    ros_node    = HolisticPublisher()
    jnt_filter  = JointFilter()
    angle_estim = JointAngleEstimator()   # 프레임 간 상태 보존

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  FPS)
    print("RealSense connecting...")
    profile    = pipeline.start(cfg)
    align      = rs.align(rs.stream.color)
    intrinsics = (profile.get_stream(rs.stream.color)
                         .as_video_stream_profile().get_intrinsics())
    print(f"Camera {COLOR_W}x{COLOR_H}@{FPS}fps  |  "
          f"Publish {PUBLISH_HZ}Hz ({PUBLISH_PERIOD*1000:.0f}ms)")
    print(f"Filter: Kalman(Q={KF_Q},R={KF_R})  "
          f"MovingAvg(N={MOVING_AVG_N})  Deadband({DEADBAND_DEG:.1f}deg)")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    arm_coords = {}
    rhand_pts  = []
    lhand_pts  = []
    frame_idx  = 0

    print("Running headless — Ctrl+C to quit")
    print("Topics: /robot/joint_states  /hand_open/right")

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            h, w      = color_img.shape[:2]
            result    = holistic.process(
                cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

            # ── 포즈 관절 3D ──────────────────────────────────────────────
            cur_arm = {}
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                for idx in DISPLAY_INDICES:
                    lm = lms[idx]
                    if lm.visibility < 0.3:
                        continue
                    pt3d = get_3d_point(depth_frame, intrinsics,
                                        int(lm.x*w), int(lm.y*h))
                    if idx in ARM_ALL and pt3d is not None:
                        cur_arm[idx] = pt3d
            for idx in ARM_ALL:
                if idx in cur_arm:
                    arm_coords[idx] = cur_arm[idx]

            # ── 손 3D ─────────────────────────────────────────────────────
            if result.right_hand_landmarks:
                rhand_pts = collect_hand_3d(
                    result.right_hand_landmarks, depth_frame, intrinsics, w, h)
            if result.left_hand_landmarks:
                lhand_pts = collect_hand_3d(
                    result.left_hand_landmarks, depth_frame, intrinsics, w, h)

            # ── 각도 계산 + 필터 ──────────────────────────────────────────
            raw_thetas      = angle_estim.compute(
                arm_coords, result.pose_landmarks, rhand_pts)
            filtered_thetas = jnt_filter.update(raw_thetas)
            r_open          = is_hand_open(
                result.right_hand_landmarks,
                rhand_pts if result.right_hand_landmarks else None)

            # ── 공유 버퍼 업데이트 (타이머가 0.05초마다 읽어서 publish) ───
            ros_node.update_state(filtered_thetas, r_open)

            if frame_idx % 100 == 0:
                s = " ".join(f"{math.degrees(t):+.1f}" for t in filtered_thetas)
                print(f"[{frame_idx:06d}] [{s}]  R:{'open' if r_open else 'fist'}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.stop()
        holistic.close()
        ros_node.destroy_node()
        rclpy.shutdown()
        print("Done.")

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

sys.path.insert(0, "/home/ohheemin/.local/lib/python3.10/site-packages")

import math, threading
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
from collections import deque

COLOR_W, COLOR_H, FPS = 848, 480, 60
DEPTH_W, DEPTH_H      = 848, 480

PUBLISH_HZ     = 20               # 0.05초마다 publish
PUBLISH_PERIOD = 1.0 / PUBLISH_HZ

IDX_L_SHOULDER = 11
IDX_R_SHOULDER = 12
IDX_L_ELBOW    = 13
IDX_R_ELBOW    = 14
IDX_L_WRIST    = 15
IDX_R_WRIST    = 16
IDX_L_HIP      = 23
IDX_R_HIP      = 24

IDX_H_WRIST     = 0
IDX_H_PINKY_MCP = 17
IDX_H_INDEX_MCP = 5
MID_FINGER_MCP  = 9

ARM_LEFT  = [IDX_L_SHOULDER, IDX_L_ELBOW, IDX_L_WRIST]
ARM_RIGHT = [IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST]
ARM_ALL   = list(set(ARM_LEFT + ARM_RIGHT))

DISPLAY_INDICES    = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FINGER_TIP_PIP     = [(8,6),(12,10),(16,14),(20,18)]
FIST_FINGER_THRESH = 3

MOVING_AVG_N = 3
DEADBAND_DEG = 1.0
DEADBAND_RAD = math.radians(DEADBAND_DEG)
KF_Q         = 1e-3
KF_R         = 1e-1


# ═══ 칼만 필터 ═══════════════════════════════════════════════════════════════
class KalmanFilter1D:
    def __init__(self, q=KF_Q, r=KF_R):
        dt = 1.0 / FPS
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[q, 0.0], [0.0, q]])
        self.R = np.array([[r]])
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.initialized = False

    def update(self, z):
        if not self.initialized:
            self.x[0, 0] = z
            self.initialized = True
            return z
        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.Q
        S  = self.H @ Pp @ self.H.T + self.R
        K  = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K @ (np.array([[z]]) - self.H @ xp)
        self.P = (np.eye(2) - K @ self.H) @ Pp
        return float(self.x[0, 0])


class JointFilter:
    def __init__(self, n=6, window=MOVING_AVG_N, deadband=DEADBAND_RAD):
        self.deadband = deadband
        self.kf       = [KalmanFilter1D() for _ in range(n)]
        self.bufs     = [deque(maxlen=window) for _ in range(n)]
        self.last_pub = [0.0] * n

    def update(self, thetas):
        out = []
        for i, th in enumerate(thetas):
            kf_v = self.kf[i].update(th)
            self.bufs[i].append(kf_v)
            avg = float(np.mean(self.bufs[i]))
            if abs(avg - self.last_pub[i]) < self.deadband:
                out.append(self.last_pub[i])
            else:
                self.last_pub[i] = avg
                out.append(avg)
        return out


# ═══ ROS2 노드 ═══════════════════════════════════════════════════════════════
class HolisticPublisher(Node):
    JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]

    def __init__(self):
        super().__init__("holistic_publisher")
        self._joint_pub = self.create_publisher(JointState, "/robot/joint_states", 10)
        self._rhand_pub = self.create_publisher(Bool, "/hand_open/right", 10)

        self._lock       = threading.Lock()
        self._thetas     = [0.0] * 6
        self._r_open     = False
        self._data_ready = False

        self.create_timer(PUBLISH_PERIOD, self._timer_cb)
        self.get_logger().info(
            f"HolisticPublisher ready — publish {PUBLISH_HZ}Hz "
            f"({PUBLISH_PERIOD*1000:.0f}ms)")

    def update_state(self, thetas, r_open):
        with self._lock:
            self._thetas     = thetas
            self._r_open     = r_open
            self._data_ready = True

    def _timer_cb(self):
        with self._lock:
            if not self._data_ready:
                return
            thetas = list(self._thetas)
            r_open = self._r_open

        stamp = self.get_clock().now().to_msg()

        jmsg = JointState()
        jmsg.header.stamp    = stamp
        jmsg.header.frame_id = "base_link"
        jmsg.name            = self.JOINT_NAMES
        jmsg.position        = [float(t) for t in thetas]
        self._joint_pub.publish(jmsg)

        hmsg      = Bool()
        hmsg.data = bool(r_open)
        self._rhand_pub.publish(hmsg)


# ═══ 회전 행렬 ═══════════════════════════════════════════════════════════════
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


# ═══ 기하 유틸 ═══════════════════════════════════════════════════════════════
def _safe_unit(v):
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

def build_transform_matrix(arm_coords, pose_lms):
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

def cam_to_global(arm_coords, global_matrix, joint_index):
    gl_joint = arm_coords.get(joint_index)
    if gl_joint is None:
        return None
    inv_GM = np.linalg.inv(global_matrix)
    pt     = np.array([[gl_joint[0]], [gl_joint[1]], [gl_joint[2]], [1.0]], dtype=float)
    v      = inv_GM @ pt
    return np.array([v[0,0], v[1,0], v[2,0]])

def cam_to_global_hand(rhand_pts, global_matrix, joint_index):
    gl_joint = rhand_pts[joint_index]
    if gl_joint is None:
        return None
    inv_GM = np.linalg.inv(global_matrix)
    pt     = np.array([[gl_joint[0]], [gl_joint[1]], [gl_joint[2]], [1.0]], dtype=float)
    v      = inv_GM @ pt
    return np.array([v[0,0], v[1,0], v[2,0]])

def arm_joint_angles(arm_coords, global_matrix, joint=None):
    if joint is None or joint < 1 or joint > 6:
        return 0.0
    el = arm_coords.get(IDX_R_ELBOW)
    if el is None:
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


# ═══ 조인트 각도 계산 (프레임 간 상태 유지가 필요한 변수는 클래스로 관리) ═══
class JointAngleEstimator:
    """
    wr_vec_prev, theta5 누적값을 프레임 간에 유지해야 하므로
    함수가 아닌 클래스로 상태를 보존합니다.
    """
    def __init__(self):
        self.wr_vec_prev = None   # 프레임 간 wr_vec 플리핑 방지용
        self.theta5_acc  = 0.0   # theta5 누적값 (j6_rot_def <= threshold 분기)

    def compute(self, arm_coords, pose_lms, rhand_pts):
        nan6 = [0.0] * 6

        X_g, Y_g, Z_g = _build_global_frame(pose_lms)
        if X_g is None:
            return nan6

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
            return nan6

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

        # θ5, θ6
        theta5 = self.theta5_acc   # 이전 프레임 누적값 유지
        theta6 = 0.0

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

                # ── wr_vec 플리핑 방지 (프레임 간 상태 유지) ─────────────
                new_wr_vec = _safe_unit(np.cross(p_vec, i_vec))
                if self.wr_vec_prev is None:
                    # 첫 프레임
                    wr_vec           = new_wr_vec
                    self.wr_vec_prev = new_wr_vec
                else:
                    dot = float(np.dot(new_wr_vec, self.wr_vec_prev))
                    if dot > 0:
                        # 방향 일관 → 업데이트
                        self.wr_vec_prev = new_wr_vec
                        wr_vec           = new_wr_vec
                    else:
                        # 방향 반전 → 이전 값 유지
                        wr_vec = self.wr_vec_prev

                j6_rot_def = np.dot(wr_vec, l_unit)
                theta6     = -theta_abs_6 if j6_rot_def > 0 else theta_abs_6

                threshold   = np.cos(np.radians(105))
                wr_axis     = _safe_unit(np.cross(m_unit, wr_vec))
                th5_vec     = _safe_unit(wr_vec * np.cos(-theta6) +
                                         np.cross(wr_axis, wr_vec) * np.sin(-theta6))
                theta_abs_5 = np.arccos(np.clip(np.dot(th5_vec, el_vec), -1.0, 1.0))
                th5_axis    = _safe_unit(np.cross(el_vec, th5_vec))
                j5_rot_def  = np.dot(th5_axis, l_unit)

                if j6_rot_def > threshold:
                    theta5 = theta_abs_5 if j5_rot_def > 0 else -theta_abs_5
                else:
                    # ── theta5 누적 (프레임 간 상태 유지) ────────────────
                    if th5_vec[2] > 0:
                        theta5 = theta5 + 0.03
                    else:
                        theta5 = theta5 - 0.03

                self.theta5_acc = theta5   # 다음 프레임을 위해 저장

        return [theta1, theta2, theta3, theta4, theta5, -theta6]


# ═══ 뎁스 / 손 유틸 ══════════════════════════════════════════════════════════
def get_3d_point(depth_frame, intrinsics, px, py):
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if not (0 <= px < dw and 0 <= py < dh):
        return None
    d = depth_frame.get_distance(int(px), int(py))
    if not (0.1 < d < 6.0):
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], d)

def collect_hand_3d(hand_lms, depth_frame, intrinsics, w, h):
    return [get_3d_point(depth_frame, intrinsics,
                         int(lm.x*w), int(lm.y*h))
            for lm in hand_lms.landmark]

def is_hand_open(hand_lms, hand_pts_3d=None):
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


# ═══ 메인 ════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    ros_node    = HolisticPublisher()
    jnt_filter  = JointFilter()
    angle_estim = JointAngleEstimator()   # 프레임 간 상태 보존

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  FPS)
    print("RealSense connecting...")
    profile    = pipeline.start(cfg)
    align      = rs.align(rs.stream.color)
    intrinsics = (profile.get_stream(rs.stream.color)
                         .as_video_stream_profile().get_intrinsics())
    print(f"Camera {COLOR_W}x{COLOR_H}@{FPS}fps  |  "
          f"Publish {PUBLISH_HZ}Hz ({PUBLISH_PERIOD*1000:.0f}ms)")
    print(f"Filter: Kalman(Q={KF_Q},R={KF_R})  "
          f"MovingAvg(N={MOVING_AVG_N})  Deadband({DEADBAND_DEG:.1f}deg)")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    arm_coords = {}
    rhand_pts  = []
    lhand_pts  = []
    frame_idx  = 0

    print("Running headless — Ctrl+C to quit")
    print("Topics: /robot/joint_states  /hand_open/right")

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            h, w      = color_img.shape[:2]
            result    = holistic.process(
                cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

            # ── 포즈 관절 3D ──────────────────────────────────────────────
            cur_arm = {}
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                for idx in DISPLAY_INDICES:
                    lm = lms[idx]
                    if lm.visibility < 0.3:
                        continue
                    pt3d = get_3d_point(depth_frame, intrinsics,
                                        int(lm.x*w), int(lm.y*h))
                    if idx in ARM_ALL and pt3d is not None:
                        cur_arm[idx] = pt3d
            for idx in ARM_ALL:
                if idx in cur_arm:
                    arm_coords[idx] = cur_arm[idx]

            # ── 손 3D ─────────────────────────────────────────────────────
            if result.right_hand_landmarks:
                rhand_pts = collect_hand_3d(
                    result.right_hand_landmarks, depth_frame, intrinsics, w, h)
            if result.left_hand_landmarks:
                lhand_pts = collect_hand_3d(
                    result.left_hand_landmarks, depth_frame, intrinsics, w, h)

            # ── 각도 계산 + 필터 ──────────────────────────────────────────
            raw_thetas      = angle_estim.compute(
                arm_coords, result.pose_landmarks, rhand_pts)
            filtered_thetas = jnt_filter.update(raw_thetas)
            r_open          = is_hand_open(
                result.right_hand_landmarks,
                rhand_pts if result.right_hand_landmarks else None)

            # ── 공유 버퍼 업데이트 (타이머가 0.05초마다 읽어서 publish) ───
            ros_node.update_state(filtered_thetas, r_open)

            if frame_idx % 100 == 0:
                s = " ".join(f"{math.degrees(t):+.1f}" for t in filtered_thetas)
                print(f"[{frame_idx:06d}] [{s}]  R:{'open' if r_open else 'fist'}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pipeline.stop()
        holistic.close()
        ros_node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
