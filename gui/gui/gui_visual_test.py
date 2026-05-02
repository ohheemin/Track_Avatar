import customtkinter as ctk
import time
import random
import os
import cv2
from PIL import Image, ImageTk
import subprocess
import threading
import signal

from collections import deque

# --- ROS2 Python 라이브러리 임포트 ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

# --- ROS2 프로세스 추적용 전역 변수 및 함수 ---
_launch_process_lock = threading.Lock()
_launch_process_by_key = {}

WORKSPACE_SETUP = os.path.expanduser("~/track_ws/install/setup.bash")

def _launch_ros2_launch(setup_script, package_name, launch_file, extra_args=None, label="ROS2", dedup_key=None):
    launch_cmd_parts = ["ros2", "launch", package_name, launch_file]
    if extra_args:
        launch_cmd_parts.extend(extra_args)
    source_chain = f"source {setup_script}" if os.path.exists(setup_script) else "echo 'No workspace setup found'"
    bash_command = f"source /opt/ros/humble/setup.bash && {source_chain} && {' '.join(launch_cmd_parts)}"
    try:
        with _launch_process_lock:
            if dedup_key:
                existing = _launch_process_by_key.get(dedup_key)
                if existing is not None and existing.poll() is None:
                    return True
            proc = subprocess.Popen(["bash", "-c", bash_command], preexec_fn=os.setsid)
            if dedup_key:
                _launch_process_by_key[dedup_key] = proc
        return True
    except Exception as e:
        print(f"[ERROR] Failed to launch {label}: {e}")
        return False
    
def _launch_ros2_run(setup_script, package_name, executable_name, extra_args=None, label="ROS2 Run", dedup_key=None):
    run_cmd_parts = ["ros2", "run", package_name, executable_name]
    if extra_args:
        run_cmd_parts.extend(extra_args)
    source_chain = f"source {setup_script}" if os.path.exists(setup_script) else "echo 'No workspace setup found'"
    bash_command = f"source /opt/ros/humble/setup.bash && {source_chain} && {' '.join(run_cmd_parts)}"
    
    print(f"[DEBUG] {label} run command: {bash_command}")
    try:
        with _launch_process_lock:
            if dedup_key:
                existing = _launch_process_by_key.get(dedup_key)
                if existing is not None and existing.poll() is None:
                    return True
            proc = subprocess.Popen(["bash", "-c", bash_command], preexec_fn=os.setsid)
            if dedup_key:
                _launch_process_by_key[dedup_key] = proc
        return True
    except Exception as e:
        print(f"[ERROR] Failed to run {label}: {e}")
        return False

def _cleanup_processes_by_key(dedup_key):
    with _launch_process_lock:
        proc = _launch_process_by_key.get(dedup_key)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            _launch_process_by_key[dedup_key] = None

# --- CustomTkinter 테마 설정 ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class DataBuffer:
    def __init__(self, max_rows=100, num_elements=1):
        self.buffer = deque(maxlen=max_rows)
        self.num_elements = num_elements

    def add_data(self, new_list):
        if len(new_list) == self.num_elements:
            self.buffer.append(new_list)

    def get_matrix(self):
        return np.array(self.buffer)

class TeleopMonitorNode(Node):
    def __init__(self, gui_app):
        super().__init__('teleop_gui_monitor')
        self.gui_app = gui_app
        self.current_sub = self.create_subscription(Float64, '/motor_1/current', self.current_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.clock_sub = self.create_subscription(Float64, '/calculated_clock', self.clock_callback, 10)

    def current_callback(self, msg):
        self.gui_app.root.after(0, self.gui_app.update_payload_current, msg.data)

    def joint_callback(self, msg):
        try:
            joints_str = ", ".join([f"{val:.2f}" for val in msg.position[:3]])
            gripper_val = msg.position[-1] if len(msg.position) > 0 else 0.0
            gripper_state = "OPEN" if gripper_val > 0.5 else "CLOSED"
            display_text = f"Joints(deg): [{joints_str}]\nGripper: {gripper_state}"
            self.gui_app.root.after(0, self.gui_app.update_robot_states, display_text)
        except Exception: pass

    # latency
    def clock_callback(self, msg):
        # self.gui_app.latest_latency_ms = msg.data
        # self.save_target.add_data(self.gui_app.latest_latency_ms.copy())
        self.gui_app.root.after(0, self.gui_app.update_latency_buffer, msg.data)

class VisualTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Robot Teleoperation Panel")
        self.root.geometry("1600x950")
        
        self.max_latency_trials = 5
        self.latency_trial = 0
        self.pnp_time_left = 120
        self.pnp_timer_running = False
        self.cap = None
        self.video_running = False
        self.realsense_online = False

        self.latest_latency_ms = 0.0

        self.latency_buffer = deque(maxlen=100)
        self.current_avg_latency = 0.0

        self.create_widgets()
        # self.root.after(100, self.open_camera)

        try:
            rclpy.init(args=None)
            self.ros_node = TeleopMonitorNode(self)
            self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
            self.ros_thread.start()
        except Exception as e:
            self.log_message(f"Failed to init ROS2 Node: {e}")
            
        self.save_latency = DataBuffer(max_rows=100, num_elements=1)

    def create_widgets(self):
        ROOT_BG = "#121212"
        PANEL_BG = "#1E1E1E"
        ACCENT_BLUE = "#4FC3F7"

        # title_label = ctk.CTkLabel(self.root, text="3D Reconstruction based - Teleoperation System", 
        #                            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold"), 
        #                            text_color=ACCENT_BLUE, bg_color=ROOT_BG)
        # title_label.pack(pady=(15, 10))

        content_wrapper = ctk.CTkFrame(self.root, fg_color=ROOT_BG, corner_radius=0)
        content_wrapper.pack(fill="both", expand=True, padx=20, pady=5)
        
        content_wrapper.rowconfigure(0, weight=1)
        content_wrapper.columnconfigure(0, weight=1) 
        content_wrapper.columnconfigure(1, weight=2) 
        content_wrapper.columnconfigure(2, weight=1) 

        # ==========================================
        # [왼쪽 컬럼]: 패키지 제어 및 로봇 스테이터스
        # ==========================================
        left_col = ctk.CTkFrame(content_wrapper, fg_color=ROOT_BG, corner_radius=0)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # RealSense
        rs_frame = ctk.CTkFrame(left_col, corner_radius=25, fg_color=PANEL_BG, border_width=2, border_color="#333333")
        rs_frame.pack(fill="both", expand=True, pady=(0, 10))
        rs_inner = ctk.CTkFrame(rs_frame, fg_color="transparent")
        rs_inner.pack(expand=True)
        ctk.CTkLabel(rs_inner, text="RealSense Controller", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.lbl_rs_status = ctk.CTkLabel(rs_inner, text="[ D435i Offline ]", text_color="#FF5252", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_rs_status.pack(pady=5)
        self.btn_realsense = ctk.CTkButton(rs_inner, text="Start RealSense", height=40, corner_radius=12, fg_color="#0277BD", border_width=2, border_color="#0277BD", command=self.toggle_realsense)
        self.btn_realsense.pack(fill="x", padx=30)

        # Robot Planning
        robot_frame = ctk.CTkFrame(left_col, corner_radius=25, fg_color=PANEL_BG, border_width=2, border_color="#333333")
        robot_frame.pack(fill="both", expand=True, pady=(5, 5))
        robot_inner = ctk.CTkFrame(robot_frame, fg_color="transparent")
        robot_inner.pack(expand=True)
        ctk.CTkLabel(robot_inner, text="Robot Planning", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.lbl_robot_status = ctk.CTkLabel(robot_inner, text="[ Planning Offline ]", text_color="#FF5252", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_robot_status.pack(pady=5)
        self.btn_planning = ctk.CTkButton(robot_inner, text="Start Planning", height=40, corner_radius=12, fg_color="#0277BD", border_width=2, border_color="#0277BD", command=self.mock_start_planning)
        self.btn_planning.pack(fill="x", padx=30)

        # Status Monitor
        status_frame = ctk.CTkFrame(left_col, corner_radius=25, fg_color=PANEL_BG, border_width=2, border_color="#333333")
        status_frame.pack(fill="both", expand=True, pady=(10, 0))
        status_inner = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_inner.pack(expand=True, fill="x")
        ctk.CTkLabel(status_inner, text="Robot State", font=ctk.CTkFont(size=16, weight="bold"), text_color="#AEEA00").pack(pady=10)
        self.lbl_robot_states = ctk.CTkLabel(status_inner, text="Joints(deg): --\nGripper: --", font=ctk.CTkFont(family="Consolas", size=14))
        self.lbl_robot_states.pack(pady=5)
        self.lbl_motor_current = ctk.CTkLabel(status_inner, text="Current: 0.0 mA", font=ctk.CTkFont(family="Consolas", size=14))
        self.lbl_motor_current.pack(pady=5)

        # ==========================================
        # [가운데 컬럼]: 카메라 및 Pick & Place
        # ==========================================
        center_col = ctk.CTkFrame(content_wrapper, fg_color=ROOT_BG)
        center_col.grid(row=0, column=1, sticky="nsew", padx=10)

        cam_frame = ctk.CTkFrame(center_col, corner_radius=25, fg_color=PANEL_BG, border_width=2, border_color="#333333")
        cam_frame.pack(fill="both", expand=True, pady=(0, 10))
        # ctk.CTkLabel(cam_frame, text="Live Vision Feed (STCOM Biz)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 0))
        self.lbl_video = ctk.CTkLabel(cam_frame, text="Initializing Camera...", fg_color="black", corner_radius=15, bg_color=PANEL_BG)
        self.lbl_video.pack(expand=True, fill="both", padx=15, pady=15)

        pnp_frame = ctk.CTkFrame(center_col, corner_radius=25, fg_color=PANEL_BG, border_width=2, border_color="#333333")
        pnp_frame.pack(fill="x", pady=(10, 0), ipady=15)
        ctk.CTkLabel(pnp_frame, text="Pick & Place Test", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.lbl_timer = ctk.CTkLabel(pnp_frame, text="02:00", text_color="#00E5FF", font=ctk.CTkFont(family="Consolas", size=28, weight="bold"))
        self.lbl_timer.pack(pady=5)
        
        # PnP 제어 버튼 프레임 (O/X 제거 후 Start/Reset 배치)
        pnp_btn_f = ctk.CTkFrame(pnp_frame, fg_color="transparent")
        pnp_btn_f.pack(pady=5)
        
        self.btn_pnp_start = ctk.CTkButton(pnp_btn_f, text="Start Timer", width=140, height=45, corner_radius=12, fg_color="#0277BD", command=lambda: self.start_trial("pnp"))
        self.btn_pnp_start.pack(side="left", padx=10)
        
        self.btn_pnp_reset = ctk.CTkButton(pnp_btn_f, text="Reset", width=100, height=45, corner_radius=12, fg_color="#455A64", command=self.reset_pnp)
        self.btn_pnp_reset.pack(side="left", padx=10)

        # ==========================================
        # [오른쪽 컬럼]: Payload 및 Latency
        # ==========================================
        right_col = ctk.CTkFrame(content_wrapper, fg_color=ROOT_BG)
        right_col.grid(row=0, column=2, sticky="nsew", padx=(15, 0))

        # Payload Test (수동 버튼 제거)
        payload_frame = ctk.CTkFrame(right_col, corner_radius=25, fg_color=PANEL_BG, border_width=2, border_color="#333333")
        payload_frame.pack(fill="both", expand=True, pady=(0, 10))
        p_inner = ctk.CTkFrame(payload_frame, fg_color="transparent")
        p_inner.pack(expand=True)
        ctk.CTkLabel(p_inner, text="Payload Test (80g)", font=ctk.CTkFont(size=16, weight="bold"), text_color="#FFB300").pack(pady=15)
        
        self.btn_payload_start = ctk.CTkButton(p_inner, text="Execute Payload Test", height=45, corner_radius=12, fg_color="#0277BD", command=lambda: self.start_trial("payload"))
        self.btn_payload_start.pack(fill="x", padx=30, pady=10)

        # Latency (기존 유지)
        lat_frame = ctk.CTkFrame(right_col, corner_radius=25, fg_color=PANEL_BG, border_width=2, border_color="#333333")
        lat_frame.pack(fill="both", expand=True, pady=(10, 0))
        l_inner = ctk.CTkFrame(lat_frame, fg_color="transparent")
        l_inner.pack(expand=True)
        ctk.CTkLabel(l_inner, text="Latency Check (ms)", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        # latency 평균
        self.lbl_realtime_latency = ctk.CTkLabel(l_inner, text="Real-time Avg: -- ms", text_color="#AEEA00", font=ctk.CTkFont(family="Consolas", size=14, weight="bold"))
        self.lbl_realtime_latency.pack(pady=2)

        self.btn_lat_start = ctk.CTkButton(l_inner, text="Measure", corner_radius=12, height=40, fg_color="#0277BD", command=lambda: self.start_trial("latency"))
        self.btn_lat_start.pack(fill="x", padx=30, pady=10)
        self.lat_inds = []
        l_ind_f = ctk.CTkFrame(l_inner, fg_color="transparent")
        l_ind_f.pack()
        for i in range(5):
            lbl = ctk.CTkLabel(l_ind_f, text="--", font=ctk.CTkFont(family="Consolas", size=12), width=40, fg_color="#212121", corner_radius=8)
            lbl.grid(row=0, column=i, padx=2)
            self.lat_inds.append(lbl)

        # ---------------------------------------------------------
        # [하단 영역]: 비상 정지 및 로그
        # ---------------------------------------------------------
        bottom_wrapper = ctk.CTkFrame(self.root, fg_color=ROOT_BG)
        bottom_wrapper.pack(fill="x", padx=20, pady=(0, 10))
        self.log_area = ctk.CTkTextbox(bottom_wrapper, height=120, corner_radius=15, fg_color="#000000", text_color="#00FF00", font=ctk.CTkFont(family="Consolas", size=12))
        self.log_area.pack(side="left", fill="both", expand=True, padx=(0, 20))
        self.btn_stop = ctk.CTkButton(bottom_wrapper, text="EMERGENCY\nSTOP", width=120, height=120, corner_radius=60, fg_color="#B71C1C", hover_color="#D32F2F", font=ctk.CTkFont(size=14, weight="bold"), border_width=4, border_color="#FFCDD2", command=self.mock_stop_all)
        self.btn_stop.pack(side="right")

    def log_message(self, msg):
        self.log_area.configure(state="normal")
        self.log_area.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_area.see("end")
        self.log_area.configure(state="disabled")

    def update_payload_current(self, current_val):
        self.lbl_motor_current.configure(text=f"Current: {current_val:.2f} mA")

    def update_robot_states(self, state_str):
        self.lbl_robot_states.configure(text=state_str)

    def open_camera(self):
        if not self.video_running:
            self.cap = cv2.VideoCapture(4) 
            if not self.cap.isOpened():
                self.lbl_video.configure(text="Camera Error", text_color="red")
                return
            self.video_running = True
            self.update_video_frame()

    def update_video_frame(self):
        try:
            if self.video_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (800, 450))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 450))
                    self.lbl_video.configure(image=imgtk, text="")
                self.root.after(30, self.update_video_frame)
        except Exception: pass

    def toggle_realsense(self):
        if not self.realsense_online:
            if _launch_ros2_run(WORKSPACE_SETUP, "avatar", "vision_nk.py", [], "RealSense", "realsense_key"):
                self.realsense_online = True
                self.lbl_rs_status.configure(text="[ D435i Online ]", text_color="#4CAF50")
                self.btn_realsense.configure(text="Stop RealSense", fg_color="#D81B60")
        else:
            _cleanup_processes_by_key("realsense_key")
            self.realsense_online = False
            self.lbl_rs_status.configure(text="[ D435i Offline ]", text_color="#FF5252")
            self.btn_realsense.configure(text="Start RealSense", fg_color="#0277BD")

    def mock_start_planning(self):
        if _launch_ros2_run(WORKSPACE_SETUP, "avatar", "dxl_subscriber.py", None, "RobotPlanning", "planning_key"):
            self.lbl_robot_status.configure(text="[ Planning Online ]", text_color="#0277BD")
            self.btn_planning.configure(state="disabled")

    def start_trial(self, t_type):
        if t_type == "payload":
            self.log_message("Payload Test (80g) Executed.")
        elif t_type == "pnp":
            self.log_message("Pick & Place Timer Started.")
            self.btn_pnp_start.configure(state="disabled")
            self.pnp_time_left = 120
            self.pnp_timer_running = True
            self.update_timer()
        elif t_type == "latency":
            if self.latency_trial < self.max_latency_trials:
                self.btn_lat_start.configure(state="disabled")
                self.root.after(500, self.record_latency)

    def reset_pnp(self):
        """타이머를 초기 상태(02:00)로 리셋하는 함수"""
        self.pnp_timer_running = False
        self.pnp_time_left = 120
        self.lbl_timer.configure(text="02:00", text_color="#00E5FF")
        self.btn_pnp_start.configure(state="normal")
        self.log_message("PnP Timer Reset.")

    def update_latency_buffer(self, new_val):
        self.latency_buffer.append(new_val)
        # 버퍼에 쌓인 데이터의 평균 계산
        self.current_avg_latency = sum(self.latency_buffer) / len(self.latency_buffer)
        # UI에 실시간 표시
        self.lbl_realtime_latency.configure(text=f"Real-time Avg: {int(self.current_avg_latency)} ms")
        
        # (선택) 기존의 save_target 로직이 필요하다면 여기서 사용 가능합니다.
        # self.save_latency.add_data([self.current_avg_latency])

    def record_latency(self):
        # 데이터가 아직 안 들어왔을 경우 방어 코드
        if not self.latency_buffer:
            self.log_message("Warning: No latency data received yet.")
            return

        val = int(self.current_avg_latency)

        self.lat_inds[self.latency_trial].configure(text=f"{val}")
        self.latency_trial += 1

        if self.latency_trial < self.max_latency_trials:
            self.btn_lat_start.configure(state="normal")
        else:
            self.btn_lat_start.configure(text="Done", state="disabled")
        
        self.log_message(f"Latency Recorded: {val} ms (Averaged)")

    def update_timer(self):
        if self.pnp_timer_running and self.pnp_time_left > 0:
            mins, secs = divmod(self.pnp_time_left, 60)
            self.lbl_timer.configure(text=f"{mins:02d}:{secs:02d}")
            self.pnp_time_left -= 1
            self.root.after(1000, self.update_timer)
        elif self.pnp_timer_running:
            self.lbl_timer.configure(text="00:00", text_color="#FF5252")
            self.log_message("PnP Time Over.")

    def mock_stop_all(self):
        _cleanup_processes_by_key("realsense_key")
        _cleanup_processes_by_key("planning_key")
        self.realsense_online = False
        self.pnp_timer_running = False
        self.lbl_rs_status.configure(text="[ D435i Offline ]", text_color="#FF5252")
        self.lbl_robot_status.configure(text="[ Planning Offline ]", text_color="#FF5252")
        self.btn_realsense.configure(text="Start RealSense", fg_color="#0277BD")
        self.btn_planning.configure(state="normal")
        self.log_message("Emergency Stop Initiated.")

def main(args=None):
    root = ctk.CTk()
    app = VisualTestGUI(root)
    def on_closing():
        if app.cap: app.cap.release()
        _cleanup_processes_by_key("realsense_key")
        _cleanup_processes_by_key("planning_key")
        os._exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()