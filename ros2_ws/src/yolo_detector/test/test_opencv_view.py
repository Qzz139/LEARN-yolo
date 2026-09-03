"""Unit tests for the optional OpenCV detector window."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_detector.opencv_view import OpenCvViewer


class FakeCv2:
    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 1

    def __init__(self, *, key: int = -1, visible: float = 1.0):
        self.key = key
        self.visible = visible
        self.calls = []

    def namedWindow(self, name, mode):
        self.calls.append(("namedWindow", name, mode))

    def imshow(self, name, frame):
        self.calls.append(("imshow", name, frame))

    def waitKey(self, delay):
        self.calls.append(("waitKey", delay))
        return self.key

    def getWindowProperty(self, name, property_id):
        self.calls.append(("getWindowProperty", name, property_id))
        return self.visible

    def destroyWindow(self, name):
        self.calls.append(("destroyWindow", name))


class OpenCvViewerTests(unittest.TestCase):
    def test_disabled_viewer_does_not_touch_opencv(self):
        cv2 = FakeCv2()
        viewer = OpenCvViewer(enabled=False, cv2_module=cv2)

        self.assertTrue(viewer.show(object()))
        self.assertEqual(cv2.calls, [])

    def test_window_is_created_once_and_frames_are_shown(self):
        cv2 = FakeCv2()
        viewer = OpenCvViewer(enabled=True, cv2_module=cv2)

        self.assertTrue(viewer.show("first"))
        self.assertTrue(viewer.show("second"))

        named_calls = [call for call in cv2.calls if call[0] == "namedWindow"]
        shown_frames = [call[2] for call in cv2.calls if call[0] == "imshow"]
        self.assertEqual(len(named_calls), 1)
        self.assertEqual(shown_frames, ["first", "second"])

    def test_q_closes_window_and_requests_shutdown(self):
        cv2 = FakeCv2(key=ord("q"))
        viewer = OpenCvViewer(enabled=True, cv2_module=cv2)

        self.assertFalse(viewer.show("frame"))
        self.assertFalse(viewer.enabled)
        self.assertIn(("destroyWindow", "YOLO Detector"), cv2.calls)

    def test_window_close_requests_shutdown(self):
        cv2 = FakeCv2(visible=0.0)
        viewer = OpenCvViewer(enabled=True, cv2_module=cv2)

        self.assertFalse(viewer.show("frame"))
        self.assertFalse(viewer.enabled)


if __name__ == "__main__":
    unittest.main()
