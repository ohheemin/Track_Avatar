import customtkinter as ctk
import time
import random
import os
import cv2
from PIL import Image, ImageTk
import subprocess
import threading
import signal

# --- ROS2 프로세스 추적용 전역 변수 및 함수 ---
_launch_process_lock = threading.Lock()
_launch_process_by_key = {}

# 현재 작업 공간(Workspace)의 setup.bash 경로 (사용자 환경에 맞게 자동 지정)
WORKSPACE_SETUP = os.path.expanduser("~/track_ws/install/setup.bash")

def _launch_ros2_launch(setup_script, package_name, launch_file, extra_args=None, label="ROS2", dedup_key=None):
    """ROS2 패키지를 백그라운드에서 안전하게 실행하는 함수"""
    launch_cmd_parts = ["ros2", "launch", package_name, launch_file]
    if extra_args:
        launch_cmd_parts.extend(extra_args)
        
    source_chain = f"source {setup_script}" if os.path.exists(setup_script) else "echo 'No workspace setup found'"
    bash_command = f"source /opt/ros/humble/setup.bash && {source_chain} && {' '.join(launch_cmd_parts)}"
    
    print(f"[DEBUG] {label} launch command: {bash_command}")
    
    try:
        with _launch_process_lock:
            # 중복 실행 방지
            if dedup_key:
                existing = _launch_process_by_key.get(dedup_key)
                if existing is not None and existing.poll() is None:
                    print(f"[INFO] {label} is already running. Duplicate launch skipped.")
                    return True
            
            # 프로세스 그룹(Process Group) 리더로 실행하여 하위 노드까지 한 번에 제어
            proc = subprocess.Popen(["bash", "-c", bash_command], preexec_fn=os.setsid)
            if dedup_key:
                _launch_process_by_key[dedup_key] = proc
                
        print(f"[INFO] {label} launcher started")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to launch {label}: {e}")
        return False

def _cleanup_processes_by_key(dedup_key):
    """실행 중인 ROS2 프로세스 그룹을 안전하게 강제 종료"""
    with _launch_process_lock:
        proc = _launch_process_by_key.get(dedup_key)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT) # Ctrl+C 와 동일
                proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL) # 반응 없으면 강제 Kill
                except Exception:
                    pass
            _launch_process_by_key[dedup_key] = None

# --- CustomTkinter 전역 테마 설정 ---
ctk.set_appearance_mode("Dark")        # 다크 모드
ctk.set_default_color_theme("blue")    # 파란색 계열 포인트 컬러

class VisualTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Robot Teleoperation Panel")
        self.root.geometry("1600x950")
        
        # 상태 변수
        self.max_trials = 5
        self.payload_trial = 0
        self.pnp_trial = 0
        self.latency_trial = 0
        self.pnp_time_left = 120
        self.pnp_timer_running = False
        
        # 카메라 관련 변수
        self.cap = None
        self.video_running = False
        self.realsense_online = False

        self.create_widgets()

        # 프로그램 시작 시 웹캠 자동 실행 (딜레이 100ms)
        self.root.after(100, self.open_camera)

    def create_widgets(self):
        # 전체 배경색 정의
        ROOT_BG = "#1E1E1E"
        PANEL_BG = "#1E1E1E"

        title_label = ctk.CTkLabel(self.root, text="3D Reconstruction based - Teleoperation System", 
                                   font=ctk.CTkFont(family="Helvetica", size=24, weight="bold"), 
                                   text_color="#007BFF", bg_color=ROOT_BG)
        title_label.pack(pady=(15, 10))

        # 메인 래퍼
        main_wrapper = ctk.CTkFrame(self.root, fg_color=ROOT_BG, corner_radius=50)
        main_wrapper.pack(fill="both", expand=True, padx=20, pady=5)

        # ---------------------------------------------------------
        # [구역 1]
        # ---------------------------------------------------------
        top_frame = ctk.CTkFrame(main_wrapper, fg_color=ROOT_BG, corner_radius=0)
        top_frame.pack(fill="x", pady=(0, 15))
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=3)

        sys_frame = ctk.CTkFrame(top_frame, fg_color=ROOT_BG, corner_radius=0)
        sys_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # 1. RealSense 프레임
        rs_frame = ctk.CTkFrame(sys_frame, corner_radius=25, fg_color=PANEL_BG, bg_color=ROOT_BG)
        rs_frame.pack(fill="x", pady=(0, 15), ipadx=10, ipady=10)
        ctk.CTkLabel(rs_frame, text="RealSense Controller", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        self.lbl_rs_status = ctk.CTkLabel(rs_frame, text="[ D435i Offline ]", text_color="#FF5252", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_rs_status.pack(pady=(0, 10))
        
        self.btn_realsense = ctk.CTkButton(rs_frame, text="Start RealSense", font=ctk.CTkFont(size=14, weight="bold"),
                                           height=40, corner_radius=20, fg_color="#00838F", hover_color="#006064",
                                           bg_color=PANEL_BG, command=self.toggle_realsense)
        self.btn_realsense.pack(fill="x", padx=20, pady=(0, 10))

        # 2. Robot Planning 프레임
        robot_frame = ctk.CTkFrame(sys_frame, corner_radius=25, fg_color=PANEL_BG, bg_color=ROOT_BG)
        robot_frame.pack(fill="x", ipadx=10, ipady=10)
        ctk.CTkLabel(robot_frame, text="Robot Planning", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        self.lbl_robot_status = ctk.CTkLabel(robot_frame, text="[ Planning Offline ]", text_color="#007BFF", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_robot_status.pack(pady=(0, 10))
        
        self.btn_planning = ctk.CTkButton(robot_frame, text="Start Planning", font=ctk.CTkFont(size=14, weight="bold"),
                                          height=40, corner_radius=20, fg_color="#2E7D32", hover_color="#1B5E20",
                                          bg_color=PANEL_BG, command=self.mock_start_planning)
        self.btn_planning.pack(fill="x", padx=20, pady=(0, 10))

        # 3. 실시간 카메라 뷰 (우측)
        cam_frame = ctk.CTkFrame(top_frame, corner_radius=25, fg_color=PANEL_BG, bg_color=ROOT_BG)
        cam_frame.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(cam_frame, text="Live Vision Feed (STCOM Biz)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 0))
        
        self.lbl_video = ctk.CTkLabel(cam_frame, text="Initializing Camera...", fg_color="black", corner_radius=15, bg_color=PANEL_BG)
        self.lbl_video.pack(expand=True, fill="both", padx=15, pady=15)

        # ---------------------------------------------------------
        # [구역 2] 성능 평가 태스크
        # ---------------------------------------------------------
        eval_frame = ctk.CTkFrame(main_wrapper, corner_radius=25, fg_color=PANEL_BG, bg_color=ROOT_BG)
        eval_frame.pack(fill="x", pady=(0, 15), ipadx=10, ipady=15)
        for i in range(3): eval_frame.columnconfigure(i, weight=1)

        # === 2-1. 가반하중 (Payload Test) ===
        p_col = ctk.CTkFrame(eval_frame, fg_color=PANEL_BG, corner_radius=0)
        p_col.grid(row=0, column=0, padx=10)
        ctk.CTkLabel(p_col, text="Payload Test", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        self.btn_payload_start = ctk.CTkButton(p_col, text="Start Trial 1", corner_radius=15, height=35, bg_color=PANEL_BG, command=lambda: self.start_trial("payload"))
        self.btn_payload_start.pack(fill="x", pady=5)
        
        # 성공/실패 수동 버튼 프레임 추가
        pf_frame1 = ctk.CTkFrame(p_col, fg_color="transparent")
        pf_frame1.pack(pady=5)
        self.btn_pl_succ = ctk.CTkButton(pf_frame1, text="Success", width=70, corner_radius=8, fg_color="#388E3C", hover_color="#2E7D32", state="disabled", command=lambda: self.record_result("payload", True))
        self.btn_pl_succ.pack(side="left", padx=5)
        self.btn_pl_fail = ctk.CTkButton(pf_frame1, text="Fail", width=70, corner_radius=8, fg_color="#D32F2F", hover_color="#B71C1C", state="disabled", command=lambda: self.record_result("payload", False))
        self.btn_pl_fail.pack(side="left", padx=5)
        
        self.pl_inds = self.create_indicator_row(p_col, PANEL_BG)

        # === 2-2. 픽앤플레이스 (Pick & Place Test) ===
        pnp_col = ctk.CTkFrame(eval_frame, fg_color=PANEL_BG, corner_radius=0)
        pnp_col.grid(row=0, column=1, padx=10)
        ctk.CTkLabel(pnp_col, text="Pick & Place Test", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        self.lbl_timer = ctk.CTkLabel(pnp_col, text="02:00", text_color="#00E5FF", font=ctk.CTkFont(family="Consolas", size=24, weight="bold"))
        self.lbl_timer.pack()
        self.btn_pnp_start = ctk.CTkButton(pnp_col, text="Start PnP Timer", corner_radius=15, height=35, bg_color=PANEL_BG, command=lambda: self.start_trial("pnp"))
        self.btn_pnp_start.pack(fill="x", pady=5)
        
        # 성공/실패 수동 버튼 프레임 추가
        pf_frame2 = ctk.CTkFrame(pnp_col, fg_color="transparent")
        pf_frame2.pack(pady=5)
        self.btn_pnp_succ = ctk.CTkButton(pf_frame2, text="Success", width=70, corner_radius=8, fg_color="#007BFF", hover_color="#007BFF", state="disabled", command=lambda: self.record_result("pnp", True))
        self.btn_pnp_succ.pack(side="left", padx=5)
        self.btn_pnp_fail = ctk.CTkButton(pf_frame2, text="Fail", width=70, corner_radius=8, fg_color="#007BFF", hover_color="#007BFF", state="disabled", command=lambda: self.record_result("pnp", False))
        self.btn_pnp_fail.pack(side="left", padx=5)
        
        self.pnp_inds = self.create_indicator_row(pnp_col, PANEL_BG)

        # === 2-3. 레이턴시 ===
        # (이 부분은 기존 코드 그대로 유지)
        l_col = ctk.CTkFrame(eval_frame, fg_color=PANEL_BG, corner_radius=0)
        l_col.grid(row=0, column=2, padx=10)
        ctk.CTkLabel(l_col, text="Latency Check (ms)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        self.btn_lat_start = ctk.CTkButton(l_col, text="Measure", corner_radius=15, height=35, bg_color=PANEL_BG, command=lambda: self.start_trial("latency"))
        self.btn_lat_start.pack(fill="x", pady=5)
        
        l_ind_f = ctk.CTkFrame(l_col, fg_color=PANEL_BG, corner_radius=0)
        l_ind_f.pack(pady=10)
        self.lat_inds = []
        for i in range(5):
            lbl = ctk.CTkLabel(l_ind_f, text="--", font=ctk.CTkFont(family="Consolas", size=12), fg_color="#212121", corner_radius=8, bg_color=PANEL_BG, width=40)
            lbl.grid(row=0, column=i, padx=3)
            self.lat_inds.append(lbl)

        # ---------------------------------------------------------
        # [구역 3] 로그 및 비상 정지
        # ---------------------------------------------------------
        self.btn_stop = ctk.CTkButton(main_wrapper, text="[ EMERGENCY STOP ]", font=ctk.CTkFont(size=16, weight="bold"), 
                                      height=50, corner_radius=25, fg_color="#007BFF", hover_color="#007BFF",
                                      bg_color=ROOT_BG, command=self.mock_stop_all)
        self.btn_stop.pack(fill="x", pady=(0, 10))

        self.log_area = ctk.CTkTextbox(main_wrapper, height=150, corner_radius=15, fg_color="#000000", text_color="#007BFF", bg_color=ROOT_BG, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_area.pack(fill="both", expand=True)
        self.log_message("System Ready. ROS2 Launch Backend Integrated.")

    def create_indicator_row(self, parent, bg_col):
        f = ctk.CTkFrame(parent, fg_color=bg_col, corner_radius=0)
        f.pack(pady=5)
        inds = []
        for _ in range(5):
            lbl = ctk.CTkLabel(f, text="[-]", font=ctk.CTkFont(family="Consolas", size=16), text_color="gray", bg_color=bg_col)
            lbl.pack(side="left", padx=5)
            inds.append(lbl)
        return inds

    def log_message(self, msg):
        self.log_area.configure(state="normal")
        self.log_area.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_area.see("end")
        self.log_area.configure(state="disabled")

    # --- 카메라 로직 ---
    def open_camera(self):
        if not self.video_running:
            self.cap = cv2.VideoCapture(2) 
            if not self.cap.isOpened():
                self.log_message("ERROR: Failed to load STCOM Biz WebCam on /dev/video4.")
                self.lbl_video.configure(text="Camera Error", text_color="red")
                return

            self.video_running = True
            self.log_message("STCOM Biz WebCam loaded automatically.")
            self.update_video_frame()

    def update_video_frame(self):
        try:
            if self.video_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    height, width, _ = frame.shape
                    zoom_factor = 1.5 
                    new_width = int(width / zoom_factor)
                    new_height = int(height / zoom_factor)
                    start_x = (width - new_width) // 2
                    start_y = (height - new_height) // 2
                    cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]
                    
                    final_frame = cv2.resize(cropped_frame, (960, 540))
                    frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.flip(frame_rgb, 1)

                    img = Image.fromarray(frame_rgb)
                    imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(960, 540))
                    
                    self.lbl_video.configure(image=imgtk, text="")
                    
                self.root.after(30, self.update_video_frame)
        except Exception as e:
            pass

    # --- 1. RealSense ROS2 연동 ---
    def toggle_realsense(self):
        if not self.realsense_online:
            self.lbl_rs_status.configure(text="[ Starting D435i... ]", text_color="orange")
            self.log_message("Executing ROS2 launch for RealSense...")
            
            success = _launch_ros2_launch(
                setup_script=WORKSPACE_SETUP,
                package_name="realsense2_camera",
                launch_file="rs_launch.py",
                extra_args=["align_depth.enable:=true"],
                label="RealSense",
                dedup_key="realsense_key"
            )
            
            if success:
                self.realsense_online = True
                self.lbl_rs_status.configure(text="[ D435i Online ]", text_color="#4CAF50")
                self.btn_realsense.configure(text="Stop RealSense", fg_color="#D81B60", hover_color="#C2185B")
            else:
                self.lbl_rs_status.configure(text="[ Launch Error ]", text_color="red")
        else:
            self.log_message("Terminating RealSense nodes...")
            _cleanup_processes_by_key("realsense_key")
            
            self.realsense_online = False
            self.lbl_rs_status.configure(text="[ D435i Offline ]", text_color="#007BFF")
            self.btn_realsense.configure(text="Start RealSense", fg_color="#007BFF", hover_color="#007BFF")

    # --- 2. 로봇 플래닝 ROS2 연동 ---
    def mock_start_planning(self):
        self.lbl_robot_status.configure(text="[준수코드 키는중]", text_color="orange")
        self.log_message("Executing Robot Planning & HW Interface...")
        
        # 중요: 팀원들이 만든 플래닝 패키지명과 런치 파일명으로 변경해 주세요.
        success = _launch_ros2_launch(
            setup_script=WORKSPACE_SETUP,
            package_name="my_planning_package",    # <-- 여기에 패키지명 입력
            launch_file="robot_planning.launch.py",# <-- 여기에 런치 파일명 입력
            label="RobotPlanning",
            dedup_key="planning_key"
        )
        
        if success:
            self.lbl_robot_status.configure(text="[준수코드 실행]", text_color="#007BFF")
            self.btn_planning.configure(state="disabled")
            self.log_message("Robot planning nodes launched successfully.")

    # --- 평가 항목 (기존 유지) ---
    def start_trial(self, t_type):
        self.log_message(f"Starting {t_type} evaluation trial...")

    # --- 3. 비상 정지 / 일괄 종료 ---
    def mock_stop_all(self):
        self.log_message("!!! EMERGENCY STOP !!! Terminating all ROS2 processes...")
        
        _cleanup_processes_by_key("realsense_key")
        _cleanup_processes_by_key("planning_key")
        
        self.realsense_online = False
        self.lbl_rs_status.configure(text="[ D435i Offline ]", text_color="#007BFF")
        self.lbl_robot_status.configure(text="[ 준수코드 Offline ]", text_color="#007BFF")
        self.btn_realsense.configure(text="Start RealSense", fg_color="#007BFF", hover_color="#007BFF")
        self.btn_planning.configure(state="normal")
        self.log_message("All systems halted.")

if __name__ == "__main__":
    root = ctk.CTk()
    app = VisualTestGUI(root)
    
    def on_closing():
        if app.cap: app.cap.release()
        _cleanup_processes_by_key("realsense_key")
        _cleanup_processes_by_key("planning_key")
        os._exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()