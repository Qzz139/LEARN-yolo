"""Save annotated snapshots and recordings without depending on ROS."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class MediaCapture:
    """Own the latest annotated frame and an optional OpenCV video writer."""

    def __init__(
        self,
        output_dir: Path,
        recording_fps: float,
        recording_codec: str,
        cv2_module: Any,
    ) -> None:
        codec = recording_codec.strip()
        if len(codec) != 4:
            raise ValueError("recording_codec must contain exactly four characters.")
        if recording_fps <= 0.0:
            raise ValueError("recording_fps must be greater than zero.")

        self.output_dir = output_dir.expanduser()
        self.recording_fps = float(recording_fps)
        self.recording_codec = codec
        self._cv2 = cv2_module
        self._latest_frame: Optional[Any] = None
        self._writer: Optional[Any] = None
        self._recording_path: Optional[Path] = None
        self.recording_requested = False

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def recording_path(self) -> Optional[Path]:
        return self._recording_path

    def request_recording(self) -> None:
        """Arm recording; the writer opens when the next frame gives its size."""

        self.recording_requested = True

    def process_frame(self, annotated_frame: Any) -> Optional[Path]:
        """Remember a frame and write it when recording is armed."""

        self._latest_frame = annotated_frame.copy()
        if not self.recording_requested:
            return None
        if self._writer is None:
            self._open_writer(annotated_frame)
        self._writer.write(annotated_frame)
        return self._recording_path

    def save_snapshot(self) -> Path:
        if self._latest_frame is None:
            raise RuntimeError("No annotated frame is available yet.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"snapshot-{self._timestamp()}.jpg"
        if not self._cv2.imwrite(str(path), self._latest_frame):
            raise RuntimeError(f"OpenCV could not write snapshot: {path}")
        return path

    def stop_recording(self) -> Optional[Path]:
        path = self._recording_path
        if self._writer is not None:
            self._writer.release()
        self._writer = None
        self._recording_path = None
        self.recording_requested = False
        return path

    def close(self) -> None:
        self.stop_recording()
        self._latest_frame = None

    def _open_writer(self, frame: Any) -> None:
        if len(frame.shape) < 2:
            self.recording_requested = False
            raise ValueError("Annotated frame has no height and width.")

        height, width = int(frame.shape[0]), int(frame.shape[1])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"recording-{self._timestamp()}.mp4"
        fourcc = self._cv2.VideoWriter_fourcc(*self.recording_codec)
        writer = self._cv2.VideoWriter(
            str(path),
            fourcc,
            self.recording_fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            self.recording_requested = False
            raise RuntimeError(
                "OpenCV could not start video recording with codec "
                f"{self.recording_codec!r}: {path}"
            )

        self._writer = writer
        self._recording_path = path

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S-%f")
