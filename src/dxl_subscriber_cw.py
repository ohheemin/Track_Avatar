#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32
from sensor_msgs.msg import JointState
import dynamixel_sdk as dxl
import numpy as np
import math
import time
from collections import deque

class DataBuffer:
    def __init__(self, max_rows=200, num_elements=7):
        self.buffer = deque(maxlen=max_rows)
        self.num_elements = num_elements

    def add_data(self, new_list):
        if len(new_list) == self.num_elements:
            self.buffer.append(new_list)

    def get_matrix(self):
        return np.array(self.buffer)

# --- 다이나믹셀 제어 테이블 주소 ---
ADDR_OPERATING_MODE         = 11
ADDR_TORQUE_ENABLE          = 64
ADDR_GOAL_POSITION          = 116
ADDR_PROFILE_VELOCITY       = 112
ADDR_PROFILE_ACCELERATION   = 108
ADDR_PWM_LIMIT              = 36
ADDR_PRESENT_POSITION       = 132
ADDR_GOAL_PWM               = 100
ADDR_POSITION_D_GAIN        = 80
ADDR_POSITION_I_GAIN        = 82
ADDR_POSITION_P_GAIN        = 84  
ADDR_FEEDFORWARD_2ND_GAIN   = 88  
ADDR_FEEDFORWARD_1ST_GAIN   = 90  

# --- 튜닝 및 한계 설정 변수 ---
LIMIT_VELOCITY = 200
LIMIT_ACCELERATION = 50
D_GAIN_VALUE     = 1000
I_GAIN_VALUE     = 200
P_GAIN_VALUE     = 1000  
FF1_GAIN_VALUE   = 150  
DEADBAND_STEP    = 3 

ff1_gain = 400
ff2_gain = 0

PROTOCOL_VERSION        = 2.0
BAUDRATE                = 1000000        
DEVICENAME              = '/dev/ttyUSB0' 
TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0
OP_MODE_PWM             = 16 

# [간소화] 라디안 대신 DXL 스텝 자체로 궤적을 바로 계산하여 연산량 감소
def jerk_limited_trajectory_dxl(start_pos, target_pos, total_time, segments):
    trajectory = []
    distance = target_pos - start_pos
    for i in range(segments + 1):
        tau = i / segments
        pos = start_pos + distance * (3 * (tau**2) - 2 * (tau**3))
        trajectory.append(int(pos))
    return trajectory

class DxlHardwareController(Node):
    def __init__(self):
        super().__init__('dxl_hardware_controller')

        self.portHandler = dxl.PortHandler(DEVICENAME)
        self.packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            self.get_logger().error(f'포트 열기 실패: {DEVICENAME}')
            return
        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error('보드레이트 설정 실패')
            return
            
        self.dxl_ids = [1, 2, 3, 4, 5, 6, 7]
        self.dxl_x_lim = [1, 3, 5]
        self.dxl_o_lim = [2, 4]
        self.dxl_6_lim = [6]

        self.last_sent_pos = {dxl_id: -1 for dxl_id in self.dxl_ids}
        self.hand_prev_pos = -1 # [수정] 그리퍼 이전 상태 저장용 변수 초기화

        # =============ema=============
        self.alpha = 0.15
        self.filtered_pos = {}

        # --- SyncRead 초기화 (1~6번 모터 위치 동시 읽기) ---
        self.groupSyncRead = dxl.GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_POSITION, 4)
        for i in range(1, 7):
            self.groupSyncRead.addParam(i)

        # ------------syncwrite 초기하 ----------
        self.groupSyncWrite = dxl.GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, 4)

        # --- 모터 설정 ---
        for dxl_id in self.dxl_ids:
            if dxl_id != 7: # ---------------------vel이랑 Accel 값 수정--------------------------
                self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_OPERATING_MODE, 3)
                self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_PROFILE_ACCELERATION, LIMIT_ACCELERATION)
                self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_PROFILE_VELOCITY, LIMIT_VELOCITY)
                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_POSITION_D_GAIN, D_GAIN_VALUE)
                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_POSITION_I_GAIN, I_GAIN_VALUE)
                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_POSITION_P_GAIN, P_GAIN_VALUE)

                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_FEEDFORWARD_1ST_GAIN, ff1_gain)
                self.packetHandler.write2ByteTxRx(self.portHandler, dxl_id, ADDR_FEEDFORWARD_2ND_GAIN, ff2_gain)

        # --- 그리퍼 설정 ---
        self.packetHandler.write4ByteTxRx(self.portHandler, 7, ADDR_PROFILE_ACCELERATION, 0)
        self.packetHandler.write4ByteTxRx(self.portHandler, 7, ADDR_PROFILE_VELOCITY, 0)
        self.packetHandler.write1ByteTxRx(self.portHandler, 7, ADDR_OPERATING_MODE, OP_MODE_PWM)
        self.packetHandler.write2ByteTxRx(self.portHandler, 7, ADDR_PWM_LIMIT, 650)

        # --- 토크 ON ---
        for dxl_id in self.dxl_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        self.joint_name_to_id = {'joint_1': 1, 'joint_2': 2, 'joint_3': 3, 'joint_4': 4, 'joint_5': 5, 'joint_6': 6}

        #self.subscription = self.create_subscription(JointState, '/dynamixel_controller/joint_cmds', self.joint_state_callback, 10)
        self.subscription = self.create_subscription(JointState, '/robot/joint_states', self.joint_state_callback, 10)
        self.hand_sub = self.create_subscription(Bool, '/hand_open/right', self.hand_state_callback, 10)
        self.index_pub = self.create_publisher(Int32, "/index", 10)
        self.timer = self.create_timer(0.01, self.index_callback)
        
        self.flag = 1
        self.target_pos_array = [0] * 7 # [수정] 빈 리스트가 아닌 크기 7 배열로 사전 할당 (IndexError 방지)
        self.save_target = DataBuffer(max_rows=200, num_elements=7)

    def rad_to_dxl(self, rad):
        step = int((rad / (2.0 * math.pi)) * 4096.0) + 2048
        return max(0, min(4095, step)) 
    
    def rad_to_dxl_lim(self, rad):
        step = int((rad / (2.0 * math.pi)) * 4096.0) + 2048
        return max(2048, min(3298, step)) 
    
    def rad_to_dxl_6(self, rad):
        step = int((rad / (2.0 * math.pi)) * 4096.0) + 2048
        return max(1536, min(3072, step)) 

    def joint_state_callback(self, msg):
        if self.flag == 1:
            second = 3
            num_steps = second * 100        
            target_angle_matrix = np.zeros((6, num_steps)) # ID 1~6 행렬 생성

            # [최적화] SyncRead로 현재 위치를 한 번에 싹 다 가져옴
            self.groupSyncRead.txRxPacket()

            for i, name in enumerate(msg.name):
                if name in self.joint_name_to_id:
                    dxl_id = self.joint_name_to_id[name]
                    target_rad = msg.position[i]
                    
                    # 변환 한계치 적용
                    if dxl_id in self.dxl_x_lim: target_pos = self.rad_to_dxl(target_rad)
                    elif dxl_id in self.dxl_o_lim: target_pos = self.rad_to_dxl_lim(target_rad)
                    elif dxl_id in self.dxl_6_lim: target_pos = self.rad_to_dxl_6(target_rad)
                    else: target_pos = self.rad_to_dxl(target_rad)

                    # 현재 위치 읽기
                    if self.groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, 4):
                        init_pos = self.groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
                    else:
                        init_pos = 2048 # 통신 실패 시 기본값

                    # DXL 스텝 값으로 바로 궤적 생성
                    traj = jerk_limited_trajectory_dxl(init_pos, target_pos, second, num_steps - 1)
                    # 메시지 순서에 상관없이 dxl_id를 기준으로 행렬에 삽입 (버그 방지)
                    target_angle_matrix[dxl_id - 1, :] = traj

            # 궤적 전송 루프
            self.get_logger().info("초기 궤적 이동 시작...")
            for step_idx in range(num_steps):
                self.groupSyncWrite.clearParam()
                for id_idx in range(6):
                    dxl_id = id_idx + 1
                    target_cmd = int(target_angle_matrix[id_idx, step_idx])
                    
                    param_goal_position = [
                        dxl.DXL_LOBYTE(dxl.DXL_LOWORD(target_cmd)), 
                        dxl.DXL_HIBYTE(dxl.DXL_LOWORD(target_cmd)), 
                        dxl.DXL_LOBYTE(dxl.DXL_HIWORD(target_cmd)), 
                        dxl.DXL_HIBYTE(dxl.DXL_HIWORD(target_cmd))
                    ]
                    self.groupSyncWrite.addParam(dxl_id, param_goal_position)

                self.groupSyncWrite.txPacket()
                time.sleep(0.01) # 100Hz 주기
            
            self.get_logger().info("초기 이동 완료!")
            self.flag = 2         
        
        elif self.flag == 2:
            self.groupSyncWrite.clearParam()
            for i, name in enumerate(msg.name):
                if name in self.joint_name_to_id:
                    dxl_id = self.joint_name_to_id[name]
                    target_rad = msg.position[i]
                    
                    # =======ema=========
                    if dxl_id in self.dxl_x_lim: raw_target = self.rad_to_dxl(target_rad)
                    elif dxl_id in self.dxl_o_lim: raw_target = self.rad_to_dxl_lim(target_rad)
                    elif dxl_id in self.dxl_6_lim: raw_target = self.rad_to_dxl_6(target_rad)
                    else: raw_target = self.rad_to_dxl(target_rad)

                    if dxl_id not in self.filtered_pos:
                        self.filtered_pos[dxl_id] = raw_target
                    else:
                        self.filtered_pos[dxl_id] = (self.alpha * raw_target) + ((1.0 - self.alpha) * self.filtered_pos[dxl_id])

                    target_pos = int(self.filtered_pos[dxl_id])

                    self.target_pos_array[dxl_id - 1] = target_pos

                    if dxl_id != 7:
                        if abs(target_pos - self.last_sent_pos[dxl_id]) <= DEADBAND_STEP:
                            target_to_send = self.last_sent_pos[dxl_id]
                        else:
                            target_to_send = target_pos
                            self.last_sent_pos[dxl_id] = target_to_send
                        # self.last_sent_pos[dxl_id] = target_pos

                        param_goal_position = [
                            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(target_to_send)), 
                            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(target_to_send)), 
                            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(target_to_send)), 
                            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(target_to_send))
                        ]
                        self.groupSyncWrite.addParam(dxl_id, param_goal_position)

            self.groupSyncWrite.txPacket()
            
            # [수정] JointState의 effort가 비어있을 경우를 대비한 안전 장치
            if len(msg.effort) > 0:
                self.target_pos_array[6] = int(msg.effort[0])
            else:
                self.target_pos_array[6] = 0
                
            self.save_target.add_data(self.target_pos_array.copy())
            self.get_logger().info(f'Add Array: {self.target_pos_array.copy()}') 
                
    def hand_state_callback(self, msg):
        hand_state = msg.data
        dxl_present_position, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, 7, ADDR_PRESENT_POSITION)

        # [수정] 클래스 멤버 변수를 활용해 이전 상태와 비교 로직 정상화
        if hand_state == False: # 닫기
            if dxl_present_position > 700:
                if self.hand_prev_pos == dxl_present_position:
                    goal_pwm = -600
                else:    
                    goal_pwm = - min(((650 - 250) * (dxl_present_position + 2048) / (2048 * 2) + 250), 650)
            else:
                goal_pwm = -250

        else: # 열기
            if dxl_present_position > 2048:
                goal_pwm = 0
            elif self.hand_prev_pos == dxl_present_position:
                goal_pwm = 600
            else:
                goal_pwm = 450

        self.packetHandler.write2ByteTxRx(self.portHandler, 7, ADDR_GOAL_PWM, int(goal_pwm))
        self.hand_prev_pos = dxl_present_position # 상태 갱신
        #self.get_logger().info(f'Gripper Position: {dxl_present_position}')   

    def index_callback(self):
        if self.flag != 2:
            return
        
        # [수정] 괄호 추가 (메서드 실행)
        joint_n_id = self.save_target.get_matrix()
        if joint_n_id.shape[0] == 0: 
            return # 데이터가 아직 없으면 종료

        position = np.zeros(6)

        # [최적화] SyncRead로 6개 데이터를 한 번에 가져옴
        if self.groupSyncRead.txRxPacket() != dxl.COMM_SUCCESS:
            return
            
        for i in range(1, 7):
            if self.groupSyncRead.isAvailable(i, ADDR_PRESENT_POSITION, 4):
                position[i - 1] = self.groupSyncRead.getData(i, ADDR_PRESENT_POSITION, 4)
            else:
                return # 데이터 누락 시 오작동 방지

        clearance_half = 11
        diff = np.abs(joint_n_id[:, :6] - position)
        
        # [최적화] 이중 for문 대신 NumPy의 초고속 벡터 비교
        matches = np.all(diff <= clearance_half, axis=1)
        match_indices = np.where(matches)[0]

        if match_indices.size > 0:
            msg = Int32()
            # ROS 2 메시지 통신을 위해 순수 int형으로 형변환
            msg.data = int(joint_n_id[match_indices[0], 6])
            self.index_pub.publish(msg)
            self.get_logger().info(f'Index: {msg.data}') 

    def destroy_node(self):
        for dxl_id in self.dxl_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.portHandler.closePort()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DxlHardwareController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()