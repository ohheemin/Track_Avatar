import os
import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext
import signal
import sys
import atexit

# --- 1. ROS2 Launch 엔진 (이전 단계에서 추출 및 최적화) ---
_launch_process_lock = threading.Lock()
_launch_process_by_key = {}

def _unique_existing_paths(paths):
    out = []
    seen = set()
    for p in paths:
        if not p or p in seen:
            continue
        p = os.path.expanduser(p) # '~' 경로 처리
        if os.path.exists(p):
            out.append(p)
            seen.add(p)
    return out

def _launch_ros2_launch(setup_script, package_name, launch_file, extra_args=None, label="ROS2", dedup_key=None, log_callback=None):
    setup_scripts = setup_script if isinstance(setup_script, (list, tuple)) else [setup_script]
    setup_scripts = _unique_existing_paths(setup_scripts)
    
    if not setup_scripts:
        msg = f"[WARNING] {label} setup script not found."
        if log_callback: log_callback(msg)
        return False
        
    launch_cmd_parts = ["ros2", "launch", package_name, launch_file]
    if extra_args:
        launch_cmd_parts.extend(extra_args)
        
    source_chain = " && ".join([f"source {s}" for s in setup_scripts])
    bash_command = f"source /opt/ros/humble/setup.bash && {source_chain} && {' '.join(launch_cmd_parts)}"
    
    try:
        with _launch_process_lock:
            if dedup_key:
                existing = _launch_process_by_key.get(dedup_key)
                if existing is not None and existing.poll() is None:
                    msg = f"[INFO] {label} is already running. Skipped."
                    if log_callback: log_callback(msg)
                    return True
            
            proc = subprocess.Popen(["bash", "-c", bash_command], cwd=os.getcwd())
            if dedup_key:
                _launch_process_by_key[dedup_key] = proc
                
        msg = f"[SUCCESS] {label} node started."
        if log_callback: log_callback(msg)
        return True
    except Exception as e:
        msg = f"[ERROR] Failed to launch {label}: {e}"
        if log_callback: log_callback(msg)
        return False

def _cleanup_all_processes():
    """실행된 모든 백그라운드 프로세스를 종료합니다."""
    with _launch_process_lock:
        for key, proc in _launch_process_by_key.items():
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    # 추가로 확실히 죽여야 할 프로세스 이름이 있다면 여기에 pkill 추가
    subprocess.run(["pkill", "-f", "realsense2_camera_node"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "move_group"], stderr=subprocess.DEVNULL)

# --- 2. Tkinter GUI 클래스 ---
class TeleopGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision-Arm Teleoperation Control Panel")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # 사용자 ROS2 워크스페이스 설정 (실제 환경에 맞게 수정 필요)
        self.setup_bash = "~/track_ws/install/setup.bash"

        # 평가 진행 상태 변수들
        self.max_trials = 5
        self.payload_trial = 0
        self.pnp_trial = 0
        self.latency_trial = 0

        # PnP 타이머
        self.pnp_time_left = 120
        self.pnp_timer_running = False

        self.create_widgets()
        
        # GUI 창 닫기 버튼(X)을 누를 때 이벤트 연결
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        title_label = tk.Label(self.root, text="D435i - Dynamixel Teleoperation System", font=("Helvetica", 18, "bold"), pady=10)
        title_label.pack(fill=tk.X)

        main_wrapper = tk.Frame(self.root, padx=20, pady=10)
        main_wrapper.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------
        # [구역 1] 시스템 제어
        # ---------------------------------------------------------
        sys_frame = tk.Frame(main_wrapper)
        sys_frame.pack(fill=tk.X, pady=(0, 20))
        sys_frame.columnconfigure(0, weight=1)
        sys_frame.columnconfigure(1, weight=1)

        vision_frame = tk.LabelFrame(sys_frame, text="1. System Initialization", font=("Helvetica", 12, "bold"), padx=10, pady=10)
        vision_frame.grid(row=0, column=0, padx=10, sticky="nsew")
        self.lbl_vision_status = tk.Label(vision_frame, text="[ Offline ]", fg="red", font=("Helvetica", 12, "bold"))
        self.lbl_vision_status.pack(pady=(0, 10))
        self.btn_vision = tk.Button(vision_frame, text="Start Vision (D435i)", font=("Helvetica", 11), height=2, bg="#E1F5FE", command=self.mock_start_vision)
        self.btn_vision.pack(fill=tk.X, padx=20)

        robot_frame = tk.LabelFrame(sys_frame, text="2. Robot Control", font=("Helvetica", 12, "bold"), padx=10, pady=10)
        robot_frame.grid(row=0, column=1, padx=10, sticky="nsew")
        self.lbl_robot_status = tk.Label(robot_frame, text="[ Offline ]", fg="red", font=("Helvetica", 12, "bold"))
        self.lbl_robot_status.pack(pady=(0, 10))
        self.btn_planning = tk.Button(robot_frame, text="Start Planning", font=("Helvetica", 11), height=2, bg="#E8F5E9", command=self.mock_start_planning)
        self.btn_planning.pack(fill=tk.X, padx=20)

        # ---------------------------------------------------------
        # [구역 2] 성능 평가 태스크
        # ---------------------------------------------------------
        eval_frame = tk.LabelFrame(main_wrapper, text="3. Evaluation Tasks (5 Trials Each)", font=("Helvetica", 12, "bold"), padx=10, pady=15)
        eval_frame.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        for i in range(3):
            eval_frame.columnconfigure(i, weight=1)

        # === 2-1. 가반하중 평가 ===
        payload_col = tk.Frame(eval_frame)
        payload_col.grid(row=0, column=0, padx=10, sticky="nsew")
        tk.Label(payload_col, text="[ Payload Test ]", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.btn_payload_start = tk.Button(payload_col, text="Start Trial 1", font=("Helvetica", 10), bg="#FFF9C4", command=lambda: self.start_trial("payload"))
        self.btn_payload_start.pack(fill=tk.X, pady=5)
        
        pf_frame1 = tk.Frame(payload_col)
        pf_frame1.pack(pady=5)
        self.btn_pl_succ = tk.Button(pf_frame1, text="Success", bg="lightgreen", state="disabled", command=lambda: self.record_result("payload", True))
        self.btn_pl_succ.pack(side=tk.LEFT, padx=5)
        self.btn_pl_fail = tk.Button(pf_frame1, text="Fail", bg="salmon", state="disabled", command=lambda: self.record_result("payload", False))
        self.btn_pl_fail.pack(side=tk.LEFT, padx=5)
        self.pl_inds = self.create_indicator_row(payload_col, "[-]")

        # === 2-2. 픽앤플레이스 ===
        pnp_col = tk.Frame(eval_frame)
        pnp_col.grid(row=0, column=1, padx=10, sticky="nsew")
        tk.Label(pnp_col, text="[ Pick & Place Test ]", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.lbl_timer = tk.Label(pnp_col, text="02:00", font=("Consolas", 16, "bold"), fg="blue")
        self.lbl_timer.pack()
        self.btn_pnp_start = tk.Button(pnp_col, text="Start Timer (Trial 1)", font=("Helvetica", 10), bg="#FFE0B2", command=lambda: self.start_trial("pnp"))
        self.btn_pnp_start.pack(fill=tk.X, pady=5)
        
        pf_frame2 = tk.Frame(pnp_col)
        pf_frame2.pack(pady=5)
        self.btn_pnp_succ = tk.Button(pf_frame2, text="Success", bg="lightgreen", state="disabled", command=lambda: self.record_result("pnp", True))
        self.btn_pnp_succ.pack(side=tk.LEFT, padx=5)
        self.btn_pnp_fail = tk.Button(pf_frame2, text="Fail", bg="salmon", state="disabled", command=lambda: self.record_result("pnp", False))
        self.btn_pnp_fail.pack(side=tk.LEFT, padx=5)
        self.pnp_inds = self.create_indicator_row(pnp_col, "[-]")

        # === 2-3. 레이턴시 ===
        lat_col = tk.Frame(eval_frame)
        lat_col.grid(row=0, column=2, padx=10, sticky="nsew")
        tk.Label(lat_col, text="[ Latency Check ]", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.btn_lat_start = tk.Button(lat_col, text="Get Topic Value (Trial 1)", font=("Helvetica", 10), bg="#D1C4E9", command=lambda: self.start_trial("latency"))
        self.btn_lat_start.pack(fill=tk.X, pady=5)
        
        lat_ind_frame = tk.Frame(lat_col)
        lat_ind_frame.pack(pady=15)
        self.lat_inds = []
        for i in range(5):
            lbl = tk.Label(lat_ind_frame, text="-- ms", font=("Consolas", 10), width=6, relief="groove")
            lbl.grid(row=0, column=i, padx=2)
            self.lat_inds.append(lbl)

        # ---------------------------------------------------------
        # [구역 3] 로그 및 정지
        # ---------------------------------------------------------
        self.btn_stop = tk.Button(main_wrapper, text="[ STOP ALL / E-STOP ]", font=("Helvetica", 14, "bold"), height=2, bg="#FFCDD2", fg="red", command=self.mock_stop_all)
        self.btn_stop.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.log_area = scrolledtext.ScrolledText(main_wrapper, wrap=tk.WORD, font=("Consolas", 10), state='disabled', bg="#F5F5F5")
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=10)
        self.log_message("System Initialized. Safe Mode Active. Waiting for user input...")

    def create_indicator_row(self, parent, default_text):
        frame = tk.Frame(parent)
        frame.pack(pady=10)
        inds = []
        for _ in range(5):
            lbl = tk.Label(frame, text=default_text, font=("Consolas", 14, "bold"))
            lbl.pack(side=tk.LEFT, padx=5)
            inds.append(lbl)
        return inds

    def log_message(self, message):
        self.log_area.config(state='normal')
        current_time = time.strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{current_time}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def start_trial(self, task_type):
        if task_type == "payload":
            if self.payload_trial < self.max_trials:
                self.log_message(f"Payload Test: Started Trial {self.payload_trial + 1}")
                self.btn_payload_start.config(state="disabled")
                self.btn_pl_succ.config(state="normal")
                self.btn_pl_fail.config(state="normal")
            
        elif task_type == "pnp":
            if self.pnp_trial < self.max_trials:
                self.log_message(f"Pick & Place: Timer Started for Trial {self.pnp_trial + 1}")
                self.btn_pnp_start.config(state="disabled")
                self.btn_pnp_succ.config(state="normal")
                self.btn_pnp_fail.config(state="normal")
                self.pnp_time_left = 120
                self.pnp_timer_running = True
                self.update_timer()
                
        elif task_type == "latency":
            if self.latency_trial < self.max_trials:
                self.btn_lat_start.config(state="disabled")
                self.log_message(f"Latency Check: Waiting for topic data (Trial {self.latency_trial + 1})...")
                self.root.after(500, self.record_latency)

    def record_result(self, task_type, is_success):
        # 이모지 대신 명확한 텍스트(컬러) 기호 사용
        symbol = "[O]" if is_success else "[X]"
        fg_color = "green" if is_success else "red"
        status_text = "Success" if is_success else "FAIL"
        
        if task_type == "payload":
            self.pl_inds[self.payload_trial].config(text=symbol, fg=fg_color)
            self.log_message(f"Payload Test: Trial {self.payload_trial + 1} -> {status_text}")
            self.payload_trial += 1
            self.btn_pl_succ.config(state="disabled")
            self.btn_pl_fail.config(state="disabled")
            if self.payload_trial < self.max_trials:
                self.btn_payload_start.config(state="normal", text=f"Start Trial {self.payload_trial + 1}")
            else:
                self.btn_payload_start.config(text="Complete", state="disabled")

        elif task_type == "pnp":
            self.pnp_timer_running = False
            self.pnp_inds[self.pnp_trial].config(text=symbol, fg=fg_color)
            self.log_message(f"Pick & Place: Trial {self.pnp_trial + 1} -> {status_text} (Time Left: {self.lbl_timer.cget('text')})")
            self.pnp_trial += 1
            self.btn_pnp_succ.config(state="disabled")
            self.btn_pnp_fail.config(state="disabled")
            self.lbl_timer.config(text="02:00", fg="blue")
            if self.pnp_trial < self.max_trials:
                self.btn_pnp_start.config(state="normal", text=f"Start Timer (Trial {self.pnp_trial + 1})")
            else:
                self.btn_pnp_start.config(text="Complete", state="disabled")

    def record_latency(self):
        simulated_latency = round(random.uniform(15.2, 85.7), 1) 
        self.lat_inds[self.latency_trial].config(text=f"{simulated_latency}")
        self.log_message(f"Latency Check: Trial {self.latency_trial + 1} -> Received {simulated_latency} ms")
        self.latency_trial += 1
        
        if self.latency_trial < self.max_trials:
            self.btn_lat_start.config(state="normal", text=f"Get Topic Value (Trial {self.latency_trial + 1})")
        else:
            self.btn_lat_start.config(text="Complete", state="disabled")

    def update_timer(self):
        if self.pnp_timer_running and self.pnp_time_left > 0:
            mins, secs = divmod(self.pnp_time_left, 60)
            self.lbl_timer.config(text=f"{mins:02d}:{secs:02d}")
            if self.pnp_time_left <= 10:
                self.lbl_timer.config(fg="red")
            self.pnp_time_left -= 1
            self.root.after(1000, self.update_timer)
        elif self.pnp_timer_running and self.pnp_time_left == 0:
            self.lbl_timer.config(text="00:00")
            self.log_message("Pick & Place: Time Over!")
            self.record_result("pnp", False)

    # ------------------- 비전 런치파일 실행 함수 -----------------------
    def start_vision(self):
        self.btn_vision.config(state="disabled")
        self.lbl_vision_status.config(text="[ Starting... ]", fg="orange")
        self.log_message(">>> Launching Vision Package...")

        # 비전 패키지 ros2 launch 실행
        success = _launch_ros2_launch(
            setup_script=self.setup_bash,          # __init__에 정의된 setup.bash 경로
            package_name="avatar",      # 예: 실제 비전 패키지 이름
            launch_file="vision.py",            # 예: 실제 런치 파일 이름
            extra_args=["align_depth:=true"],      # 추가 파라미터 (없으면 None)
            label="Vision",
            dedup_key="vision_node",
            log_callback=self.log_message
        )

        if success:
            self.lbl_vision_status.config(text="[ Online ]", fg="green")
            self.btn_vision.config(state="normal") # 필요시 켜둔 상태로 비활성화 유지해도 됨
        else:
            self.lbl_vision_status.config(text="[ Error ]", fg="red")
            self.btn_vision.config(state="normal")

    # ------------------- 플래닝 런치파일 실행 함수 -----------------------
    def start_planning(self):
        self.btn_planning.config(state="disabled")
        self.lbl_robot_status.config(text="[ Starting... ]", fg="orange")
        self.log_message(">>> Launching Planning Package...")

        # 플래닝 패키지(MoveIt2) ros2 launch 실행
        success = _launch_ros2_launch(
            setup_script=self.setup_bash,
            package_name="my_robot_moveit_config", # 예: 실제 로봇 패키지 이름
            launch_file="demo.launch.py",          # 예: 실제 런치 파일 이름
            label="Planning",
            dedup_key="planning_node",
            log_callback=self.log_message
        )

        if success:
            self.lbl_robot_status.config(text="[ Online ]", fg="green")
            self.btn_planning.config(state="normal")
        else:
            self.lbl_robot_status.config(text="[ Error ]", fg="red")
            self.btn_planning.config(state="normal")

    # ---------------------- 스탑 버튼 ------------------------------
    def stop_all(self):
        self.log_message("!!! EMERGENCY STOP INITIATED !!!")
        self.pnp_timer_running = False
        
        # 1. 시각적 상태 업데이트
        self.lbl_vision_status.config(text="[ Offline ]", fg="red")
        self.lbl_robot_status.config(text="[ Offline ]", fg="red")
        self.btn_vision.config(state="normal")
        self.btn_planning.config(state="normal")
        
        # 2. 실제 프로세스 강제 종료 (위에서 정의된 함수 호출)
        _cleanup_all_processes()
        self.log_message("All ROS 2 nodes and processes halted safely.")

# --- 3. 메인 실행 부 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TeleopGUI(root)
    # 이전에 성공했던 가장 안전한 방식의 종료 함수
    def on_closing():
        print("\n[INFO] 창이 닫혔습니다. 프로그램을 완전히 종료합니다.")
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()