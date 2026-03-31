import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher_ = self.create_publisher(JointState, '/my_robot/joint_states', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = JointState()
        
        # 1. 핵심! 현재 시간을 찍어줍니다. (RViz가 무시하지 않도록)
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # 2. 관절 이름 입력
        msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        # 3. 원하는 각도 입력 (라디안 단위: 0.0 ~ 3.14)
        msg.position = [0.5, 0.2, 0.0, -0.5, 0.0, 0.0] 
        
        # 퍼블리시!
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    robot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()