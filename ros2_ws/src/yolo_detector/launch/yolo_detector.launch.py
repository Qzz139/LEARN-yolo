# ROS 2 launch 入口：加载 YAML 参数、应用非空覆盖项，并可预加载 libgomp。
"""Launch the configurable YOLO detector node."""

import os
import tempfile
from pathlib import Path

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_libgomp_path() -> str:
    for candidate in (
        "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
        "/lib/aarch64-linux-gnu/libgomp.so.1",
    ):
        if Path(candidate).is_file():
            return candidate
    return ""


def _launch_detector(context: LaunchContext):
    parameters = [LaunchConfiguration("config_file").perform(context)]

    optional_overrides = {
        "model_path": LaunchConfiguration("model_path").perform(context),
        "source_mode": LaunchConfiguration("source_mode").perform(context),
        "image_topic": LaunchConfiguration("image_topic").perform(context),
        "camera_source": LaunchConfiguration("camera_source").perform(context),
        "image_source": LaunchConfiguration("image_source").perform(context),
    }
    # 空启动参数表示保留配置文件值，不覆盖为一个空字符串。
    overrides = {
        name: value
        for name, value in optional_overrides.items()
        if value != ""
    }
    if overrides:
        # Foxy writes parameter dictionaries under the /** wildcard, but its
        # parameter parser does not reliably apply that scope. Write an exact
        # fully-qualified node scope instead.
        # 临时文件使用精确节点作用域；启动进程稍后读取，因此退出 with 后仍保留。
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="yolo_detector_params_",
            suffix=".yaml",
            delete=False,
        ) as parameter_file:
            yaml.safe_dump(
                {"/yolo_detector": {"ros__parameters": overrides}},
                parameter_file,
                default_flow_style=False,
            )
            parameters.append(parameter_file.name)

    node_options = {}
    libgomp_path = LaunchConfiguration("libgomp_path").perform(context).strip()
    if libgomp_path:
        if not Path(libgomp_path).is_file():
            raise FileNotFoundError(
                f"libgomp preload was not found: {libgomp_path}"
            )
        # 保留调用环境已有的预加载库，将 libgomp 放到最前面。
        existing_preload = os.environ.get("LD_PRELOAD", "")
        preload_value = libgomp_path
        if existing_preload:
            preload_value += os.pathsep + existing_preload
        node_options["additional_env"] = {"LD_PRELOAD": preload_value}

    return [
        Node(
            package="yolo_detector",
            executable="yolo_detector_node",
            name="yolo_detector",
            output="screen",
            parameters=parameters,
            **node_options,
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
            DeclareLaunchArgument(
                "image_source",
                default_value="",
                description="Optional static image path for image mode.",
            ),
            DeclareLaunchArgument(
                "libgomp_path",
                default_value=_default_libgomp_path(),
                description=(
                    "Optional libgomp path to preload before starting the node."
                ),
            ),
            OpaqueFunction(function=_launch_detector),
        ]
    )
