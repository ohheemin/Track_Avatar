from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

def generate_launch_description():

    package_name = 'my_robot_description'
    package_share_directory = get_package_share_directory(package_name)

    urdf_file = Path(package_share_directory) / "urdf" / "humanoid.urdf"
    
    rviz_config_file = Path(package_share_directory) / "rviz" / "view_success.rviz"

    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    return LaunchDescription([

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            remappings=[('/joint_states', '/my_robot/joint_states')]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', str(rviz_config_file)],
            output='screen'
        )
    ])
