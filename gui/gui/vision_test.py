import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import random
import os
import cv2
from PIL import Image, ImageTk

# --- 다크 모드 색상 팔레트 ---
BG_COLOR = "#121212"        # 전체 배경 (매우 어두운 회색/검정)
FRAME_BG = "#1E1E1E"        # 프레임 배경
FG_COLOR = "#FFFFFF"        # 기본 글자색 (흰색)
BTN_BG = "#333333"          # 기본 버튼 배경
BTN_FG = "#FFFFFF"          # 기본 버튼 글자색
ACCENT_COLOR = "#4FC3F7"    # 강조 색상 (하늘색)

class VisualTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Teleoperation Control Panel (Dark Mode & Vision)")
        self.root.geometry("1200x950")  # 카메라 화면을 위해 세로 크기 약간 확장
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)

        # 평가 진행 상태 변수들
        self.max_trials = 5
        self.payload_trial = 0
        self.pnp_trial = 0
        self.latency_trial = 0

        # PnP 타이머
        self.pnp_time_left = 120
        self.pnp_timer_running = False
        
        # OpenCV 웹캠 제어 변수
        self.cap = None
        self.video_running = False

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="D435i - Dynamixel Teleoperation System", 
                               font=("Helvetica", 18, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR, pady=10)
        title_label.pack(fill=tk.X)

        main_wrapper = tk.Frame(self.root, padx=20, pady=10, bg=BG_COLOR)
        main_wrapper.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------
        # [구역 1] 시스템 제어 & 카메라 뷰 (상단)
        # ---------------------------------------------------------
        top_frame = tk.Frame(main_wrapper, bg=BG_COLOR)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=2) # 카메라 화면 쪽에 더 많은 공간 할당

        # 1-1. 시스템 상태 컨트롤 (좌측)
        sys_frame = tk.Frame(top_frame, bg=BG_COLOR)
        sys_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        vision_frame = tk.LabelFrame(sys_frame, text="1. System Initialization", font=("Helvetica", 12, "bold"), 
                                     bg=FRAME_BG, fg=FG_COLOR, padx=10, pady=10)
        vision_frame.pack(fill=tk.X, pady=(0, 10))
        self.lbl_vision_status = tk.Label(vision_frame, text="[ Offline ]", fg="#FF5252", bg=FRAME_BG, font=("Helvetica", 12, "bold"))
        self.lbl_vision_status.pack(pady=(0, 10))
        self.btn_vision = tk.Button(vision_frame, text="Start Vision (STCOM Biz)", font=("Helvetica", 11), height=2, 
                                    bg="#0277BD", fg=FG_COLOR, activebackground="#01579B", activeforeground=FG_COLOR,
                                    command=self.start_vision)
        self.btn_vision.pack(fill=tk.X, padx=10)

        robot_frame = tk.LabelFrame(sys_frame, text="2. Robot Control", font=("Helvetica", 12, "bold"), 
                                    bg=FRAME_BG, fg=FG_COLOR, padx=10, pady=10)
        robot_frame.pack(fill=tk.X)
        self.lbl_robot_status = tk.Label(robot_frame, text="[ Offline ]", fg="#FF5252", bg=FRAME_BG, font=("Helvetica", 12, "bold"))
        self.lbl_robot_status.pack(pady=(0, 10))
        self.btn_planning = tk.Button(robot_frame, text="Start Planning (MoveIt2)", font=("Helvetica", 11), height=2, 
                                      bg="#2E7D32", fg=FG_COLOR, activebackground="#1B5E20", activeforeground=FG_COLOR,
                                      command=self.mock_start_planning)
        self.btn_planning.pack(fill=tk.X, padx=10)

        # 1-2. 실시간 카메라 스트리밍 뷰 (우측)
        cam_frame = tk.LabelFrame(top_frame, text="Live Vision Feed", font=("Helvetica", 12, "bold"), 
                                  bg=FRAME_BG, fg=FG_COLOR, padx=10, pady=10)
        cam_frame.grid(row=0, column=1, sticky="nsew")
        
        self.lbl_video = tk.Label(cam_frame, bg="black", text="No Video Feed", fg="gray", font=("Helvetica", 16))
        self.lbl_video.pack(expand=True, fill=tk.BOTH)

        # ---------------------------------------------------------
        # [구역 2] 성능 평가 태스크 (중단)
        # ---------------------------------------------------------
        eval_frame = tk.LabelFrame(main_wrapper, text="3. Evaluation Tasks (5 Trials Each)", font=("Helvetica", 12, "bold"), 
                                   bg=FRAME_BG, fg=FG_COLOR, padx=10, pady=15)
        eval_frame.pack(fill=tk.X, pady=(0, 15), padx=0)
        
        for i in range(3):
            eval_frame.columnconfigure(i, weight=1)

        # === 2-1. 가반하중 평가 ===
        payload_col = tk.Frame(eval_frame, bg=FRAME_BG)
        payload_col.grid(row=0, column=0, padx=10, sticky="nsew")
        tk.Label(payload_col, text="[ Payload Test ]", font=("Helvetica", 12, "bold"), bg=FRAME_BG, fg=FG_COLOR).pack(pady=5)
        self.btn_payload_start = tk.Button(payload_col, text="Start Trial 1", font=("Helvetica", 10), bg=BTN_BG, fg=BTN_FG, command=lambda: self.start_trial("payload"))
        self.btn_payload_start.pack(fill=tk.X, pady=5)
        
        pf_frame1 = tk.Frame(payload_col, bg=FRAME_BG)
        pf_frame1.pack(pady=5)
        self.btn_pl_succ = tk.Button(pf_frame1, text="Success", bg="#388E3C", fg="white", state="disabled", command=lambda: self.record_result("payload", True))
        self.btn_pl_succ.pack(side=tk.LEFT, padx=5)
        self.btn_pl_fail = tk.Button(pf_frame1, text="Fail", bg="#D32F2F", fg="white", state="disabled", command=lambda: self.record_result("payload", False))
        self.btn_pl_fail.pack(side=tk.LEFT, padx=5)
        self.pl_inds = self.create_indicator_row(payload_col, "[-]")

        # === 2-2. 픽앤플레이스 ===
        pnp_col = tk.Frame(eval_frame, bg=FRAME_BG)
        pnp_col.grid(row=0, column=1, padx=10, sticky="nsew")
        tk.Label(pnp_col, text="[ Pick & Place Test ]", font=("Helvetica", 12, "bold"), bg=FRAME_BG, fg=FG_COLOR).pack(pady=5)
        self.lbl_timer = tk.Label(pnp_col, text="02:00", font=("Consolas", 16, "bold"), bg=FRAME_BG, fg=ACCENT_COLOR)
        self.lbl_timer.pack()
        self.btn_pnp_start = tk.Button(pnp_col, text="Start Timer (Trial 1)", font=("Helvetica", 10), bg=BTN_BG, fg=BTN_FG, command=lambda: self.start_trial("pnp"))
        self.btn_pnp_start.pack(fill=tk.X, pady=5)
        
        pf_frame2 = tk.Frame(pnp_col, bg=FRAME_BG)
        pf_frame2.pack(pady=5)
        self.btn_pnp_succ = tk.Button(pf_frame2, text="Success", bg="#388E3C", fg="white", state="disabled", command=lambda: self.record_result("pnp", True))
        self.btn_pnp_succ.pack(side=tk.LEFT, padx=5)
        self.btn_pnp_fail = tk.Button(pf_frame2, text="Fail", bg="#D32F2F", fg="white", state="disabled", command=lambda: self.record_result("pnp", False))
        self.btn_pnp_fail.pack(side=tk.LEFT, padx=5)
        self.pnp_inds = self.create_indicator_row(pnp_col, "[-]")

        # === 2-3. 레이턴시 ===
        lat_col = tk.Frame(eval_frame, bg=FRAME_BG)
        lat_col.grid(row=0, column=2, padx=10, sticky="nsew")
        tk.Label(lat_col, text="[ Latency Check ]", font=("Helvetica", 12, "bold"), bg=FRAME_BG, fg=FG_COLOR).pack(pady=5)
        self.btn_lat_start = tk.Button(lat_col, text="Get Topic Value (Trial 1)", font=("Helvetica", 10), bg=BTN_BG, fg=BTN_FG, command=lambda: self.start_trial("latency"))
        self.btn_lat_start.pack(fill=tk.X, pady=5)
        
        lat_ind_frame = tk.Frame(lat_col, bg=FRAME_BG)
        lat_ind_frame.pack(pady=15)
        self.lat_inds = []
        for i in range(5):
            lbl = tk.Label(lat_ind_frame, text="-- ms", font=("Consolas", 10), width=6, bg="#2C2C2C", fg=FG_COLOR, relief="groove")
            lbl.grid(row=0, column=i, padx=2)
            self.lat_inds.append(lbl)

        # ---------------------------------------------------------
        # [구역 3] 로그 및 정지 (하단)
        # ---------------------------------------------------------
        self.btn_stop = tk.Button(main_wrapper, text="[ STOP ALL / E-STOP ]", font=("Helvetica", 14, "bold"), 
                                  height=2, bg="#B71C1C", fg="white", activebackground="#D32F2F", activeforeground="white",
                                  command=self.mock_stop_all)
        self.btn_stop.pack(fill=tk.X, padx=0, pady=(0, 10))

        # 로그창 다크 모드 설정
        self.log_area = scrolledtext.ScrolledText(main_wrapper, wrap=tk.WORD, font=("Consolas", 10), 
                                                  bg="#000000", fg="#00FF00", insertbackground="white")
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=0)
        self.log_area.config(state='disabled')
        self.log_message("System Initialized. Dark Mode & Safe Mode Active.")

    def create_indicator_row(self, parent, default_text):
        frame = tk.Frame(parent, bg=FRAME_BG)
        frame.pack(pady=10)
        inds = []
        for _ in range(5):
            lbl = tk.Label(frame, text=default_text, font=("Consolas", 14, "bold"), bg=FRAME_BG, fg="gray")
            lbl.pack(side=tk.LEFT, padx=5)
            inds.append(lbl)
        return inds

    def log_message(self, message):
        self.log_area.config(state='normal')
        current_time = time.strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{current_time}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    # --- 카메라 연동 로직 ---
    def start_vision(self):
        self.btn_vision.config(state="disabled")
        self.lbl_vision_status.config(text="[ Starting... ]", fg="orange")
        self.log_message("Initializing Camera on /dev/video4...")
        
        # 카메라 시작을 별도 메서드로 분리
        self.root.after(500, self.open_camera)

    def open_camera(self):
        if not self.video_running:
            self.cap = cv2.VideoCapture(5) # STCOM Biz 인덱스
            # 해상도를 창 크기에 맞게 조절
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
            if not self.cap.isOpened():
                self.log_message("ERROR: Cannot open camera on index 4.")
                self.lbl_vision_status.config(text="[ Error ]", fg="red")
                self.btn_vision.config(state="normal")
                return

            self.video_running = True
            self.lbl_vision_status.config(text="[ Online ]", fg="#4CAF50") # 밝은 초록색
            self.log_message("Camera Online. Streaming started.")
            self.update_video_frame()

    def update_video_frame(self):
        try:
            # 변수명 오류 수정: self.webcam_running -> self.video_running
            if self.video_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # 1. 원본 해상도 파악
                    height, width, _ = frame.shape
                    
                    # 2. 디지털 줌 (크롭 영역 계산)
                    zoom_factor = 2.0 # 1.5배 줌
                    new_width = int(width / zoom_factor)
                    new_height = int(height / zoom_factor)
                    
                    start_x = (width - new_width) // 2
                    start_y = (height - new_height) // 2
                    
                    # 3. 중앙 자르기 (줌-인 효과)
                    cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]
                    
                    # 4. GUI 화면 표시 크기 확대 (960x540)
                    gui_width = 1500
                    gui_height = 1000
                    final_frame = cv2.resize(cropped_frame, (gui_width, gui_height))

                    # 5. 색상 변환 및 미러링
                    frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    # frame_rgb = cv2.flip(frame_rgb, 1)

                    # 6. Tkinter 이미지 업데이트
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.lbl_video.imgtk = imgtk
                    self.lbl_video.configure(image=imgtk, text="")
                    
            # 30ms 뒤에 다시 자기 자신을 호출 (루프 유지)
            self.root.after(30, self.update_video_frame)

        except Exception as e:
            # 만약 에러가 나면 터미널에 원인을 출력하고 루프를 정지하지 않음
            print(f"[ERROR] 영상 업데이트 실패: {e}")

    def stop_camera(self):
        self.video_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.lbl_video.configure(image='', text="Video Stopped", fg="gray")

    # --- 기존 제어 로직 ---
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
        symbol = "[O]" if is_success else "[X]"
        fg_color = "#4CAF50" if is_success else "#FF5252"
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
            self.lbl_timer.config(text="02:00", fg=ACCENT_COLOR)
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
                self.lbl_timer.config(fg="#FF5252") # 10초 남으면 빨간색
            self.pnp_time_left -= 1
            self.root.after(1000, self.update_timer)
        elif self.pnp_timer_running and self.pnp_time_left == 0:
            self.lbl_timer.config(text="00:00")
            self.log_message("Pick & Place: Time Over!")
            self.record_result("pnp", False)

    def mock_start_planning(self):
        self.btn_planning.config(state="disabled")
        self.lbl_robot_status.config(text="[ Starting... ]", fg="orange")
        self.log_message("Starting Planning Package...")
        self.root.after(2000, lambda: [self.lbl_robot_status.config(text="[ Online ]", fg="#4CAF50"), self.btn_planning.config(state="normal")])

    def mock_stop_all(self):
        self.log_message("!!! EMERGENCY STOP INITIATED !!!")
        self.pnp_timer_running = False
        self.stop_camera() # E-Stop 시 카메라도 정지 및 자원 반환
        self.lbl_vision_status.config(text="[ Offline ]", fg="#FF5252")
        self.lbl_robot_status.config(text="[ Offline ]", fg="#FF5252")
        self.btn_vision.config(state="normal")
        self.log_message("Systems halted. Hardware resources released.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualTestGUI(root)

    def on_closing():
        print("\n[INFO] Closing window. Releasing resources...")
        app.stop_camera() # 창이 닫힐 때 카메라 자원을 확실히 반환
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()