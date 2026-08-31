"""Launch the configurable YOLO detector node."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_detector(context: LaunchContext):
    parameters = [LaunchConfiguration("config_file").perform(context)]

    optional_overrides = {
        "model_path": LaunchConfiguration("model_path").perform(context),
        "source_mode": LaunchConfiguration("source_mode").perform(context),
        "image_topic": LaunchConfiguration("image_topic").perform(context),
        "camera_source": LaunchConfiguration("camera_source").perform(context),
    }
    parameters.extend(
        {name: value}
        for name, value in optional_overrides.items()
        if value != ""
    )

    return [
        Node(
            package="yolo_detector",
            executable="yolo_detector_node",
            name="yolo_detector",
            output="screen",
            parameters=parameters,
        )
    ]


def generate_launch_description() -> LaunchDescription:
    default_config = (
        get_package_share_directory("yolo_detector") + "/config/detector.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Absolute path to the detector YAML configuration.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Optional model_path override.",
            ),
            DeclareLaunchArgument(
                "source_mode",
                default_value="",
                description="Optional source_mode override: topic or camera.",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="",
                description="Optional image_topic override.",
            ),
            DeclareLaunchArgument(
                "camera_source",
                default_value="",
                description="Optional camera index or pipeline override.",
            ),
            OpaqueFunction(function=_launch_detector),
        ]
    )
