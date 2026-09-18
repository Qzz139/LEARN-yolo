# 窗口组件：延迟创建 OpenCV 窗口，并把按键或关闭按钮转换为退出信号。
"""Lifecycle-safe OpenCV display for the detector node."""

from __future__ import annotations

from typing import Any


class OpenCvViewer:
    """Show annotated frames and report when the user requests shutdown."""

    def __init__(
        self,
        *,
        enabled: bool,
        cv2_module: Any,
        window_name: str = "YOLO Detector",
    ) -> None:
        self._enabled = enabled
        self._cv2 = cv2_module
        self._window_name = window_name
        self._opened = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def show(self, frame: Any) -> bool:
        """Display one frame; return false after Q, Escape, or window close."""
        if not self._enabled:
            return True

        try:
            if not self._opened:
                self._cv2.namedWindow(
                    self._window_name,
                    self._cv2.WINDOW_NORMAL,
                )
                self._opened = True

            self._cv2.imshow(self._window_name, frame)
            # 短暂处理 GUI 事件，既刷新窗口也读取按键，避免阻塞节点。
            key = self._cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self.close()
                return False

            if hasattr(self._cv2, "getWindowProperty"):
                visible = self._cv2.getWindowProperty(
                    self._window_name,
                    self._cv2.WND_PROP_VISIBLE,
                )
                if visible < 1:
                    self.close()
                    return False
        except Exception as exc:
            self.close()
            raise RuntimeError(f"OpenCV window failed: {exc}") from exc

        return True

    def close(self) -> None:
        """Close the window and disable future display calls."""
        # 关闭后禁止后续帧重建窗口，重复清理也不会再次销毁同一窗口。
        self._enabled = False
        if not self._opened:
            return
        try:
            self._cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        self._opened = False
