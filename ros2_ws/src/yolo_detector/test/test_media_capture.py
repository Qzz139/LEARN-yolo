"""Unit tests for annotated snapshot and recording state."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_detector.media_capture import MediaCapture


class FakeFrame:
    shape = (480, 640, 3)

    def copy(self):
        return FakeFrame()


class FakeWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class FakeCv2:
    def __init__(self, writer_opened=True):
        self.writer_opened = writer_opened
        self.writers = []

    @staticmethod
    def VideoWriter_fourcc(*codec):
        return "".join(codec)

    def VideoWriter(self, path, fourcc, fps, size):
        writer = FakeWriter(self.writer_opened)
        writer.path = path
        writer.fourcc = fourcc
        writer.fps = fps
        writer.size = size
        self.writers.append(writer)
        return writer

    @staticmethod
    def imwrite(path, frame):
        Path(path).write_bytes(b"jpeg")
        return True


class MediaCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.cv2 = FakeCv2()
        self.capture = MediaCapture(self.output_dir, 10.0, "mp4v", self.cv2)

    def tearDown(self):
        self.capture.close()
        self.temp_dir.cleanup()

    def test_snapshot_requires_a_frame(self):
        with self.assertRaisesRegex(RuntimeError, "No annotated frame"):
            self.capture.save_snapshot()

    def test_snapshot_is_written_after_a_frame(self):
        self.capture.process_frame(FakeFrame())

        path = self.capture.save_snapshot()

        self.assertEqual(path.parent, self.output_dir)
        self.assertEqual(path.suffix, ".jpg")
        self.assertTrue(path.is_file())

    def test_recording_opens_on_next_frame_and_stops_cleanly(self):
        self.capture.request_recording()

        path = self.capture.process_frame(FakeFrame())

        self.assertIsNotNone(path)
        self.assertTrue(self.capture.is_recording)
        self.assertEqual(self.cv2.writers[0].size, (640, 480))
        self.assertEqual(len(self.cv2.writers[0].frames), 1)

        stopped_path = self.capture.stop_recording()

        self.assertEqual(stopped_path, path)
        self.assertFalse(self.capture.recording_requested)
        self.assertFalse(self.capture.is_recording)
        self.assertTrue(self.cv2.writers[0].released)

    def test_bad_codec_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            MediaCapture(self.output_dir, 10.0, "bad", self.cv2)


if __name__ == "__main__":
    unittest.main()
