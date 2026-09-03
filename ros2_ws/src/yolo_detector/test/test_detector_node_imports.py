"""Import-level regression tests for the ROS detector node."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeDetection2DArray:
    pass


class DummyMessage:
    pass


def fake_module(name: str, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


class DetectorNodeImportTests(unittest.TestCase):
    def test_detection_array_publisher_type_is_imported(self):
        node_type = type("Node", (), {})
        modules = {
            "cv_bridge": fake_module("cv_bridge", CvBridge=DummyMessage),
            "geometry_msgs": fake_module("geometry_msgs"),
            "geometry_msgs.msg": fake_module(
                "geometry_msgs.msg", Pose2D=DummyMessage
            ),
            "rclpy": fake_module("rclpy"),
            "rclpy.node": fake_module("rclpy.node", Node=node_type),
            "rclpy.qos": fake_module(
                "rclpy.qos", qos_profile_sensor_data=object()
            ),
            "sensor_msgs": fake_module("sensor_msgs"),
            "sensor_msgs.msg": fake_module(
                "sensor_msgs.msg", Image=DummyMessage
            ),
            "std_msgs": fake_module("std_msgs"),
            "std_msgs.msg": fake_module(
                "std_msgs.msg", Float32=DummyMessage, Header=DummyMessage
            ),
            "std_srvs": fake_module("std_srvs"),
            "std_srvs.srv": fake_module(
                "std_srvs.srv", SetBool=DummyMessage, Trigger=DummyMessage
            ),
            "vision_msgs": fake_module("vision_msgs"),
            "vision_msgs.msg": fake_module(
                "vision_msgs.msg",
                Detection2D=DummyMessage,
                Detection2DArray=FakeDetection2DArray,
                ObjectHypothesisWithPose=DummyMessage,
            ),
        }

        module_name = "yolo_detector.detector_node"
        ros_messages_name = "yolo_detector.ros_messages"
        sys.modules.pop(module_name, None)
        sys.modules.pop(ros_messages_name, None)
        try:
            with mock.patch.dict(sys.modules, modules):
                detector_node = importlib.import_module(module_name)
                self.assertIs(
                    detector_node.Detection2DArray, FakeDetection2DArray
                )
        finally:
            sys.modules.pop(module_name, None)
            sys.modules.pop(ros_messages_name, None)


if __name__ == "__main__":
    unittest.main()
