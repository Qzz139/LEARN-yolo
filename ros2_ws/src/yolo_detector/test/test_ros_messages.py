"""Compatibility tests for ROS detection message construction."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeHeader:
    pass


class FakeImage:
    def __init__(self):
        self.encoding = ""
        self.header = None


class FakeCenter:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


class FakeBoundingBox:
    def __init__(self):
        self.center = FakeCenter()
        self.size_x = 0.0
        self.size_y = 0.0


class FakeDetection:
    def __init__(self):
        self.header = None
        self.bbox = FakeBoundingBox()
        self.results = []


class FakeDetectionArray:
    def __init__(self):
        self.header = None
        self.detections = []


class FoxyHypothesis:
    def __init__(self):
        self.id = ""
        self.score = 0.0


def install_fake_ros_messages():
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.Image = FakeImage
    sensor = types.ModuleType("sensor_msgs")
    sensor.msg = sensor_msg

    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Header = FakeHeader
    std = types.ModuleType("std_msgs")
    std.msg = std_msg

    vision_msg = types.ModuleType("vision_msgs.msg")
    vision_msg.Detection2D = FakeDetection
    vision_msg.Detection2DArray = FakeDetectionArray
    vision_msg.ObjectHypothesisWithPose = FoxyHypothesis
    vision = types.ModuleType("vision_msgs")
    vision.msg = vision_msg

    sys.modules["sensor_msgs"] = sensor
    sys.modules["sensor_msgs.msg"] = sensor_msg
    sys.modules["std_msgs"] = std
    sys.modules["std_msgs.msg"] = std_msg
    sys.modules["vision_msgs"] = vision
    sys.modules["vision_msgs.msg"] = vision_msg


install_fake_ros_messages()

fake_cv2 = types.ModuleType("cv2")
fake_cv2.COLOR_RGB2BGR = 1
fake_cv2.COLOR_GRAY2BGR = 2
fake_cv2.COLOR_BGRA2BGR = 3
fake_cv2.COLOR_RGBA2BGR = 4
fake_cv2.cvtColor = lambda frame, _conversion: frame
sys.modules["cv2"] = fake_cv2

from yolo_detector import ros_messages


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10.0, 20.0, 30.0, 60.0]])
        self.conf = FakeTensor([0.8])
        self.cls = FakeTensor([1.0])

    def __len__(self):
        return len(self.xyxy.values)


class FakeBridge:
    def cv2_to_imgmsg(self, frame, encoding):
        self.frame = frame
        self.requested_encoding = encoding
        message = FakeImage()
        message.encoding = "8UC3"
        return message


class RosMessageTests(unittest.TestCase):
    def make_result(self):
        return SimpleNamespace(boxes=FakeBoxes(), names={1: "model-name"})

    def test_foxy_detection_schema_is_preserved(self):
        header = FakeHeader()

        message = ros_messages.build_detections_message(
            self.make_result(), header, ["keyboard", "monitor", "mouse"]
        )

        detection = message.detections[0]
        self.assertIs(message.header, header)
        self.assertEqual(detection.bbox.center.x, 20.0)
        self.assertEqual(detection.bbox.center.y, 40.0)
        self.assertEqual(detection.bbox.size_x, 20.0)
        self.assertEqual(detection.bbox.size_y, 40.0)
        self.assertEqual(detection.results[0].id, "monitor")
        self.assertAlmostEqual(detection.results[0].score, 0.8)

    def test_newer_hypothesis_schema_is_preserved(self):
        original = ros_messages.ObjectHypothesisWithPose

        class ModernHypothesis:
            def __init__(self):
                self.hypothesis = SimpleNamespace(class_id="", score=0.0)

        ros_messages.ObjectHypothesisWithPose = ModernHypothesis
        try:
            message = ros_messages.build_detections_message(
                self.make_result(), FakeHeader(), ["keyboard", "monitor", "mouse"]
            )
        finally:
            ros_messages.ObjectHypothesisWithPose = original

        hypothesis = message.detections[0].results[0].hypothesis
        self.assertEqual(hypothesis.class_id, "monitor")
        self.assertAlmostEqual(hypothesis.score, 0.8)

    def test_bgr8_bridge_workaround_is_preserved(self):
        bridge = FakeBridge()
        header = FakeHeader()
        frame = object()

        message = ros_messages.bgr8_image_message(bridge, frame, header)

        self.assertIs(bridge.frame, frame)
        self.assertEqual(bridge.requested_encoding, "passthrough")
        self.assertEqual(message.encoding, "bgr8")
        self.assertIs(message.header, header)


if __name__ == "__main__":
    unittest.main()
