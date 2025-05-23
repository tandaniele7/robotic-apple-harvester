from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Include the gen3_lite.launch.py file with servo:=false argument
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([
        #         PathJoinSubstitution([
        #             FindPackageShare('pic4kinova'),
        #             'launch',
        #             'gen3_lite.launch.py'
        #         ])
        #     ]),
        #     launch_arguments={'servo': 'false'}.items()
        # ),
        
        # Launch the move_manipulator node
        Node(
            package='kinova_cmd_pose',
            executable='move_manipulator',
            name='move_manipulator'
        ),
        
        # Launch the starting_position node
        Node(
            package='kinova_cmd_pose',
            executable='starting_position',
            name='starting_position'
        ),
        
        # Launch the apple_to_box node
        Node(
            package='kinova_cmd_pose',
            executable='apple_to_box',
            name='apple_to_box'
        )
    ])