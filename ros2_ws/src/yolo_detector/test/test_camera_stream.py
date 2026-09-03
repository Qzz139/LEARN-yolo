"""Unit tests for USB camera reconnect behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_detector.camera_stream import RecoveringCamera


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class FakeLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


class FakeCapture:
    def __init__(self, opened=True, reads=()):
        self.opened = opened
        self.reads = list(reads)
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.reads.pop(0) if self.reads else (False, None)

    def release(self):
        self.released = True


class FakeCv2:
    CAP_GSTREAMER = 1800

    def __init__(self, captures):
        self.captures = list(captures)
        self.calls = []

    def VideoCapture(self, source, *backend):
        self.calls.append((source, backend))
        return self.captures.pop(0)


class RecoveringCameraTests(unittest.TestCase):
    def test_numeric_source_is_converted_and_frame_is_returned(self):
        frame = object()
        cv2 = FakeCv2([FakeCapture(reads=[(True, frame)])])
        camera = RecoveringCamera(
            "1", "auto", 2.0, 3, cv2, FakeLogger(), FakeClock()
        )

        self.assertIs(camera.read(), frame)
        self.assertEqual(cv2.calls, [(1, ())])

    def test_stable_device_path_remains_a_string(self):
        path = "/dev/v4l/by-id/example-video-index0"
        cv2 = FakeCv2([FakeCapture()])
        camera = RecoveringCamera(
            path, "auto", 2.0, 3, cv2, FakeLogger(), FakeClock()
        )

        self.assertEqual(camera.source, path)
        self.assertEqual(cv2.calls, [(path, ())])

    def test_failed_reads_close_then_reconnect_after_interval(self):
        clock = FakeClock()
        lost = FakeCapture(reads=[(False, None), (False, None)])
        recovered_frame = object()
        recovered = FakeCapture(reads=[(True, recovered_frame)])
        cv2 = FakeCv2([lost, recovered])
        logger = FakeLogger()
        camera = RecoveringCamera(
            "/dev/camera", "auto", 2.0, 2, cv2, logger, clock
        )

        self.assertIsNone(camera.read())
        self.assertIsNone(camera.read())
        self.assertTrue(lost.released)

        clock.value += 1.0
        self.assertIsNone(camera.read())
        self.assertEqual(len(cv2.calls), 1)

        clock.value += 1.0
        self.assertIsNone(camera.read())
        self.assertEqual(len(cv2.calls), 2)
        self.assertIs(camera.read(), recovered_frame)
        self.assertTrue(
            any("Reconnected" in message for message in logger.info_messages)
        )

    def test_gstreamer_backend_is_forwarded(self):
        cv2 = FakeCv2([FakeCapture()])
        RecoveringCamera(
            "pipeline", "gstreamer", 2.0, 3, cv2, FakeLogger(), FakeClock()
        )

        self.assertEqual(cv2.calls, [("pipeline", (cv2.CAP_GSTREAMER,))])


if __name__ == "__main__":
    unittest.main()
