"""
    D435 카메라 + MediaPipe Pose: 관절 3D 좌표 실시간 시각화
    뎁스카메라로부터 RGBD 이미지를 받아온 후, 스켈레톤 모델로부터 
    관절 인덱싱하고 3d 좌표 추출함

"""

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

COLOR_W, COLOR_H, FPS = 848, 480, 30
DEPTH_W, DEPTH_H      = 848, 480

# 표시할 관절 이름 
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# 주요 관절만 3D 좌표를 화면에 표시 
DISPLAY_INDICES = [
    0,   # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
]

# 색상 (BGR)
COLOR_JOINT   = (0,   255, 100)
COLOR_TEXT    = (255, 255,   0)
COLOR_TEXT_BG = (0,   0,     0)
COLOR_SKEL    = (100, 200, 255)

# MediaPipe Pose 연결 
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def draw_text_with_bg(img, text, pos, font_scale=0.38, thickness=1,
                       text_color=COLOR_TEXT, bg_color=COLOR_TEXT_BG, alpha=0.55):
    """
        반투명 배경이 있는 텍스트 그리기.
        배경 위에 스켈레톤 결과 화면이 그려지게 됨
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def get_3d_point(depth_frame, intrinsics, px, py):
    """
    픽셀 좌표 (px, py)에서 RealSense depth를 이용해
    카메라 좌표계 3D 점 (X, Y, Z) 반환.
    유효하지 않으면 None 반환.
    """
    if px < 0 or py < 0 or px >= depth_frame.get_width() or py >= depth_frame.get_height():
        return None
    depth_val = depth_frame.get_distance(int(px), int(py))
    if depth_val <= 0.1 or depth_val > 6.0:   # 10cm ~ 6m 유효 범위
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_val)
    return point   # [X, Y, Z] 미터

# 메인 파이프라인
def main():
 
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)

    print("[INFO] RealSense D435 연결 중...")
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    color_profile   = profile.get_stream(rs.stream.color)
    intrinsics       = color_profile.as_video_stream_profile().get_intrinsics()

    print(f"해상도: {COLOR_W}x{COLOR_H} @ {FPS}fps")
    print(f"내부 파라미터: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
          f"cx={intrinsics.ppx:.1f}, cy={intrinsics.ppy:.1f}")

    mp_pose    = mp.solutions.pose
    pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,        
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    show_depth = False
    frame_idx  = 0
    print("실행 중임")

    try:
        while True:
            frames       = pipeline.wait_for_frames()
            aligned      = align.process(frames)
            color_frame  = aligned.get_color_frame()
            depth_frame  = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())   
            depth_arr = np.asanyarray(depth_frame.get_data())   

            h, w = color_img.shape[:2]
            disp  = color_img.copy()

            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb_img)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                for conn in POSE_CONNECTIONS:
                    idx_a, idx_b = conn
                    a = lms[idx_a]
                    b = lms[idx_b]
                    if a.visibility > 0.3 and b.visibility > 0.3:
                        px_a = int(a.x * w), int(a.y * h)
                        px_b = int(b.x * w), int(b.y * h)
                        cv2.line(disp, px_a, px_b, COLOR_SKEL, 2, cv2.LINE_AA)

                for idx in DISPLAY_INDICES:
                    lm = lms[idx]
                    if lm.visibility < 0.3:
                        continue

                    px = int(lm.x * w)
                    py = int(lm.y * h)

                    pt3d = get_3d_point(depth_frame, intrinsics, px, py)

                    cv2.circle(disp, (px, py), 5, COLOR_JOINT, -1, cv2.LINE_AA)
                    cv2.circle(disp, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)

                    name = LANDMARK_NAMES[idx]
                    if pt3d is not None:
                        x3, y3, z3 = pt3d
                        label = f"{name} ({x3:+.2f},{y3:+.2f},{z3:.2f})m"
                    else:
                    
                        label = f"{name} MP({lm.x:.2f},{lm.y:.2f},{lm.z:.2f})"

                    draw_text_with_bg(disp, label, (px + 8, py + 4))

                for idx, lm in enumerate(lms):
                    if idx in DISPLAY_INDICES:
                        continue
                    if lm.visibility < 0.4:
                        continue
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(disp, (px, py), 3, (180, 180, 180), -1, cv2.LINE_AA)

            draw_text_with_bg(disp, f"Frame: {frame_idx}", (10, 22),
                              font_scale=0.45, text_color=(200, 255, 200))
            draw_text_with_bg(disp, "Q/ESC:Quit  S:Save  D:Depth", (10, 44),
                              font_scale=0.40, text_color=(200, 200, 200))

            detected = "Detected" if results.pose_landmarks else "No Person"
            dot_color = (0, 255, 0) if results.pose_landmarks else (0, 0, 255)
            cv2.circle(disp, (w - 20, 20), 8, dot_color, -1)
            draw_text_with_bg(disp, detected, (w - 110, 24),
                              font_scale=0.40, text_color=(255, 255, 255))

            cv2.imshow("D435 + MediaPipe Pose | 3D Joints", disp)

            
            if show_depth:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_arr, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.imshow("Depth Map", depth_colormap)
            else:
                try:
                    cv2.destroyWindow("Depth Map")
                except:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):          
                break
            elif key == ord('s'):              
                fname = f"pose_capture_{frame_idx:04d}.png"
                cv2.imwrite(fname, disp)
                print(f"[SAVE] {fname}")
            elif key == ord('d'):              
                show_depth = not show_depth

            frame_idx += 1

    finally:
        pipeline.stop()
        pose_model.close()
        cv2.destroyAllWindows()
        print("종료 완료.")


if __name__ == "__main__":
    main()