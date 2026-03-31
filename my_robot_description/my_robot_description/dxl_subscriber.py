import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
import dynamixel_sdk as dxl  # 다이나믹셀 SDK
import math

# --- 다이나믹셀 제어 테이블 주소 (X-시리즈 공통) ---
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_POSITION      = 116
ADDR_PROFILE_VELOCITY   = 112
ADDR_PROFILE_ACCELERATION = 108

LIMIT_VELOCITY = 30
LIMIT_ACCELERATION = 40

# --- 프로토콜 및 통신 설정 ---
PROTOCOL_VERSION        = 2.0
BAUDRATE                = 1000000        # 다이나믹셀 위자드에서 설정한 통신 속도에 맞추세요 (보통 57600 또는 1000000)
DEVICENAME              = '/dev/ttyUSB0' # U2D2 또는 통신 장치가 연결된 포트

TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0

class DxlHardwareController(Node):
    def __init__(self):
        super().__init__('dxl_hardware_controller')

        # 1. 다이나믹셀 포트 및 패킷 핸들러 초기화
        self.portHandler = dxl.PortHandler(DEVICENAME)
        self.packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

        # 포트 열기 및 보드레이트 설정
        if not self.portHandler.openPort():
            self.get_logger().error(f'포트를 열 수 없습니다: {DEVICENAME}. 권한(chmod)을 확인하세요.')
            return
        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error('보드레이트 설정 실패')
            return
            
        self.get_logger().info('다이나믹셀 포트 연결 성공!')

        # 2. 모터 ID 리스트 (1: XM430, 2~6: XL430)
        self.dxl_ids = [1, 2, 3, 4, 5, 6, 7]

        for dxl_id in self.dxl_ids:
            # 가속도 설정 (4 Byte)
            self.packetHandler.write4ByteTxRx(
                self.portHandler, dxl_id, ADDR_PROFILE_ACCELERATION, LIMIT_ACCELERATION)
            
            # 속도 설정 (4 Byte)
            self.packetHandler.write4ByteTxRx(
                self.portHandler, dxl_id, ADDR_PROFILE_VELOCITY, LIMIT_VELOCITY)
            
            self.get_logger().info(f'ID {dxl_id} 속도({LIMIT_VELOCITY}) 및 가속도({LIMIT_ACCELERATION}) 제한 설정 완료')

        # 모든 모터 토크 켜기
        for dxl_id in self.dxl_ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != dxl.COMM_SUCCESS:
                self.get_logger().error(f'ID {dxl_id} 토크 켜기 실패: {self.packetHandler.getTxRxResult(dxl_comm_result)}')
            else:
                self.get_logger().info(f'ID {dxl_id} 토크 ON')

        # 3. URDF 관절 이름과 하드웨어 ID 매핑
        # URDF에 정의된 joint 이름과 실제 다이나믹셀 ID를 매칭해줍니다. 
        # (이름은 이전 URDF 기준 예시입니다. 실제 하드웨어 배치에 맞게 ID를 수정하세요)
        self.joint_name_to_id = {
            'joint_1': 1, # XM430
            'joint_2': 2, # XL430
            'joint_3': 3, # XL430
            'joint_4': 4, # XL430
            'joint_5': 5, # XL430
            'joint_6': 6  # XL430
        }

        # 4. ROS 2 Subscriber 설정
        self.subscription = self.create_subscription(
            JointState,
            '/robot/joint_states',
            self.joint_state_callback,
            10)
        
        self.hand_sub = self.create_subscription(
            Bool,
            '/hand_open/right',
            self.hand_state_callback,
            10)

    def rad_to_dxl(self, rad):
        """ 라디안 각도를 다이나믹셀 X-시리즈 스텝 값(0~4095)으로 변환 """
        # X-시리즈 분해능: 1바퀴(2*PI) = 4096 스텝
        # 0도 (0 라디안) = 2048 스텝 (중앙)
        step = int((rad / (2.0 * math.pi)) * 4096.0) + 2048
        
        # 기계적 한계를 넘지 않도록 0 ~ 4095로 제한 (필요시 관절별 한계치로 수정)
        return max(2048, min(4095, step)) # Warning!!! Negative Number is not calculated!!!

    def joint_state_callback(self, msg):
        # 수신된 관절 배열을 순회하며 모터에 명령 전송
        for i, name in enumerate(msg.name):
            if name in self.joint_name_to_id:
                dxl_id = self.joint_name_to_id[name]
                target_rad = msg.position[i]
                
                # 라디안 -> 스텝 변환
                target_pos = self.rad_to_dxl(target_rad)

                # 다이나믹셀에 목표 위치 쓰기 (Goal Position은 4바이트 데이터)
                self.packetHandler.write4ByteTxRx(
                    self.portHandler, dxl_id, ADDR_GOAL_POSITION, target_pos)
                
    def hand_state_callback(self, msg):
        hand_state = msg.data
        if hand_state == True:
            self.packetHandler.write4ByteTxRx(self.portHandler, 7, ADDR_GOAL_POSITION, 2048)
        else:
            self.packetHandler.write4ByteTxRx(self.portHandler, 7, ADDR_GOAL_POSITION, 1200)

    def destroy_node(self):
        # 프로그램 종료 시 안전을 위해 모든 모터 토크 해제
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