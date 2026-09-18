# 相机恢复组件：累计读取失败后关闭设备，并按时间间隔尝试重连。
"""Recover an OpenCV camera stream after USB disconnects."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional


class RecoveringCamera:
    """Open, read, close, and periodically reconnect one camera source."""

    def __init__(
        self,
        source_text: str,
        backend: str,
        reconnect_interval: float,
        read_failure_threshold: int,
        cv2_module: Any,
        logger: Any,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if backend not in {"auto", "gstreamer"}:
            raise ValueError(
                "camera_backend must be 'auto' or 'gstreamer', "
                f"but received {backend!r}."
            )
        if reconnect_interval <= 0.0:
            raise ValueError("camera_reconnect_interval must be greater than zero.")
        if read_failure_threshold <= 0:
            raise ValueError(
                "camera_read_failure_threshold must be greater than zero."
            )

        self.source_text = source_text.strip()
        # 纯数字输入作为相机索引，设备路径或 GStreamer 管道保留为字符串。
        self.source: Any = (
            int(self.source_text)
            if self.source_text.lstrip("-").isdigit()
            else self.source_text
        )
        self.backend = backend
        self.reconnect_interval = float(reconnect_interval)
        self.read_failure_threshold = int(read_failure_threshold)
        self._cv2 = cv2_module
        self._logger = logger
        self._clock = clock
        self._capture: Optional[Any] = None
        self._read_failures = 0
        self._last_reconnect_attempt = 0.0
        self._connect(initial=True)

    def read(self) -> Optional[Any]:
        """Return one frame, or None while disconnected/reconnecting."""

        if self._capture is None:
            self._connect()
            return None

        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            ok, frame = False, None
            self._logger.warning(f"Camera read raised an error: {exc}")

        if ok and frame is not None:
            self._read_failures = 0
            return frame

        # 容忍短暂丢帧，只有连续失败达到阈值才释放设备并进入重连状态。
        self._read_failures += 1
        if self._read_failures >= self.read_failure_threshold:
            self._logger.warning(
                "Camera stream was lost; closing it and starting "
                "automatic reconnect attempts."
            )
            self._capture.release()
            self._capture = None
            self._read_failures = 0
            self._last_reconnect_attempt = self._clock()
        return None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _create_capture(self) -> Any:
        if self.backend == "gstreamer":
            return self._cv2.VideoCapture(self.source, self._cv2.CAP_GSTREAMER)
        return self._cv2.VideoCapture(self.source)

    def _connect(self, initial: bool = False) -> bool:
        # 采用可注入的单调时钟控制重试频率，测试时可推进时间而无需真实等待。
        attempted_at = self._clock()
        if (
            not initial
            and attempted_at - self._last_reconnect_attempt
            < self.reconnect_interval
        ):
            return False
        self._last_reconnect_attempt = attempted_at

        try:
            capture = self._create_capture()
        except Exception as exc:
            self._capture = None
            self._logger.warning(
                f"Camera {self.source_text!r} could not be opened: {exc}; "
                f"retrying in {self.reconnect_interval:.1f}s."
            )
            return False

        if not capture.isOpened():
            capture.release()
            self._capture = None
            self._logger.warning(
                f"Camera {self.source_text!r} is unavailable; "
                f"retrying in {self.reconnect_interval:.1f}s."
            )
            return False

        self._capture = capture
        self._read_failures = 0
        action = "Opened" if initial else "Reconnected"
        self._logger.info(f"{action} camera source {self.source_text!r}.")
        return True
