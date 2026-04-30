#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

class JerkLimitedFilterNode(Node):
    def __init__(self):
        super().__init__('jerk_limited_filter_node')

        # 1. 제어 주기 설정 (예: 100Hz = 0.01초)
        self.control_period = 0.05
        
        # 2. Ruckig 관련 변수
        self.dof = 0                  # 관절 개수 (첫 메시지 수신 시 초기화)
        self.otg = None               # Online Trajectory Generator (Ruckig)
        self.inp = None               # Ruckig 입력 파라미터 (현재 상태, 목표 상태, 한계치)
        self.out = None               # Ruckig 출력 파라미터 (다음 상태)
        
        self.target_positions = []    # 카메라에서 들어오는 실시간 목표 위치
        self.is_initialized = False

        # 3. 통신 인터페이스
        self.raw_sub = self.create_subscription(
            JointState,
            '/robot/joint_states',
            self.vision_callback,
            10
        )
        self.filtered_pub = self.create_publisher(
            JointState,
            '/dynamixel_controller/joint_cmds',
            10
        )

        # 4. 100Hz 제어 루프 타이머 생성
        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.index = np.zeros(6)
        
        self.get_logger().info("Ruckig 저크 제한 궤적 노드가 시작되었습니다.")

    def init_ruckig(self, initial_positions):
        """첫 데이터가 들어왔을 때 Ruckig 초기화 및 하드웨어 한계 설정"""
        self.dof = len(initial_positions)
        self.otg = Ruckig(self.dof, self.control_period)
        self.inp = InputParameter(self.dof)
        self.out = OutputParameter(self.dof)

        # 현재 상태를 초기 위치로 설정
        self.inp.current_position = initial_positions
        self.inp.current_velocity = [0.0] * self.dof
        self.inp.current_acceleration = [0.0] * self.dof

        # XL430 모터 및 기구부의 기계적 한계치 설정 (튜닝 필수)
        # rad/s, rad/s^2, rad/s^3 단위
        self.inp.max_velocity = [4.0] * self.dof      # 최대 속도
        self.inp.max_acceleration = [2.0] * self.dof  # 최대 가속도
        self.inp.max_jerk = [8.0] * self.dof         # 최대 저크 (이 값이 작을수록 부드러움)
        
        self.is_initialized = True
        self.get_logger().info(f"Ruckig 초기화 완료: {self.dof} DoF")

    def vision_callback(self, msg):
        """카메라로부터 목표 각도가 들어오면 타겟 변수만 업데이트 (비동기)"""
        if not self.is_initialized:
            self.init_ruckig(msg.position)
            self.joint_names = msg.name

        self.index = msg.effort    
        # 목표 위치 갱신
        self.target_positions = list(msg.position)

    def control_loop(self):
        """100Hz로 지속적으로 돌아가는 실시간 궤적 생성 루프"""
        if not self.is_initialized:
            return

        # 1. Ruckig에 카메라 목표 위치 입력
        self.inp.target_position = self.target_positions
        self.inp.target_velocity = [0.0] * self.dof       # 최종 도달 시 멈춤
        self.inp.target_acceleration = [0.0] * self.dof   # 최종 도달 시 멈춤

        # 2. 다음 0.01초(10ms) 뒤의 위치 계산 (업데이트)
        res = self.otg.update(self.inp, self.out)


        # 결과가 정상적으로 계산되었으면
        if res == Result.Working or res == Result.Finished:
            # 3. 계산된 '다음 위치'를 모터 명령으로 발행
            filtered_msg = JointState()

            filtered_msg.header.stamp = self.get_clock().now().to_msg()
            filtered_msg.name = self.joint_names
            filtered_msg.position = self.out.new_position

            filtered_msg.effort = self.index
            self.get_logger().info(f'Index: {filtered_msg.effort}')
            
            self.filtered_pub.publish(filtered_msg)

            # 4. (매우 중요) 이번 루프의 '출력(결과)'을 다음 루프의 '시작점'으로 덮어씌움
            # 이를 통해 속도와 가속도가 절대 끊기지 않고 부드럽게 이어집니다.
            self.inp.current_position = self.out.new_position
            self.inp.current_velocity = self.out.new_velocity
            self.inp.current_acceleration = self.out.new_acceleration
        else:
            self.get_logger().warn("궤적 계산 실패! 목표 위치가 너무 멀거나 한계치가 빡빡합니다.")

def main(args=None):
    rclpy.init(args=args)
    node = JerkLimitedFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()