"""Bring up the armbench tabletop simulation (Gazebo Harmonic + ros2_control).

Arguments
---------
headless   true (default) runs ``gz sim -s`` (server only); false also opens the GUI.
world      Absolute path to a world SDF. Default: the template world with no cubes.
           Generate a seeded world with ``armbench scene --seed N --out /tmp/scene.sdf``.
rtf        Target real-time factor passed to Gazebo (0 = run as fast as possible).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

MAX_CUBES = 6
ARM_CONTROLLERS = ["joint_state_broadcaster", "joint_trajectory_controller", "gripper_controller"]
ENERGY_CONTROLLERS = ["energy_state_broadcaster"]


def launch_setup(context):  # noqa: ANN001, ANN201
    headless = LaunchConfiguration("headless")
    world = LaunchConfiguration("world")

    controllers_file = PathJoinSubstitution(
        [FindPackageShare("armbench_bringup"), "config", "controllers.yaml"]
    )
    description_file = PathJoinSubstitution(
        [FindPackageShare("armbench_description"), "urdf", "armbench_ur5e.urdf.xacro"]
    )
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            description_file,
            " name:=armbench_ur5e",
            " simulation_controllers:=",
            controllers_file,
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    gz_args_headless = " -s -r -v 2 "
    gz_args_gui = " -r -v 2 "
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={
            "gz_args": [
                gz_args_headless if headless.perform(context) == "true" else gz_args_gui,
                world,
            ]
        }.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string",
            robot_description_content,
            "-name",
            "armbench_ur5e",
            "-allow_renaming",
            "true",
            "-z",
            "0.0",
        ],
    )

    spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-manager", "/controller_manager"],
            output="screen",
        )
        for name in [*ARM_CONTROLLERS, *ENERGY_CONTROLLERS]
    ]

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # Ground-truth cube poses (gz PosePublisher on each cube model; absent cubes are
            # simply never published).
            *[
                f"/model/cube_{i}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose"
                for i in range(MAX_CUBES)
            ],
        ],
        parameters=[{"use_sim_time": True}],
    )

    return [robot_state_publisher, gz_sim, spawn_robot, bridge, *spawners]


def generate_launch_description() -> LaunchDescription:
    default_world = PathJoinSubstitution(
        [FindPackageShare("armbench_description"), "worlds", "tabletop.sdf"]
    )
    declared = [
        DeclareLaunchArgument("headless", default_value="true", description="Server only."),
        DeclareLaunchArgument("world", default_value=default_world, description="World SDF."),
    ]
    return LaunchDescription([*declared, OpaqueFunction(function=launch_setup)])
