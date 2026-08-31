#!/usr/bin/env python3
"""Preview the trained YOLO model locally with OpenCV DNN.

The exported ONNX model already includes post-processing and returns rows in
the form ``[x1, y1, x2, y2, confidence, class_id]``.  This program deliberately
does not depend on ROS, PyTorch, Ultralytics, or onnxruntime.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by the launcher
    raise SystemExit(
        "Missing local preview dependencies. Install them with:\n"
        "  python3 -m pip install -r apps/local_preview/requirements.txt"
    ) from exc


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "weights/yolo26n_baseline_v1/best.onnx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/local_preview"
DEFAULT_LABELS = ("keyboard", "monitor", "mouse")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
WINDOW_NAME = "YOLO Local Preview"

# BGR colors chosen to stay distinct on both light and dark scenes.
COLORS = (
    (40, 210, 255),   # keyboard: amber
    (255, 120, 80),   # monitor: blue
    (90, 220, 90),    # mouse: green
)


class PreviewError(RuntimeError):
    """Actionable error that should be shown without a Python traceback."""


@dataclass(frozen=True)
class Detection:
    """One detection in original-image coordinates."""

    class_id: int
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


@dataclass(frozen=True)
class InferenceResult:
    """Detections plus timing for one frame."""

    detections: Sequence[Detection]
    inference_ms: float


def _is_git_lfs_pointer(path: Path) -> bool:
    """Return True when a checkout contains a Git LFS pointer, not the model."""

    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise PreviewError(f"{description} does not exist: {path}")
    if _is_git_lfs_pointer(path):
        raise PreviewError(
            f"{description} is only a Git LFS pointer: {path}\n"
            "Run `git lfs pull`, then try again."
        )


def _letterbox(
    image: "np.ndarray", size: int
) -> Tuple["np.ndarray", float, int, int]:
    """Resize without distortion and pad to a square model input."""

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise PreviewError("Input frame has an invalid size.")

    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized
    return canvas, scale, pad_x, pad_y


class OnnxDetector:
    """Small OpenCV-DNN wrapper for the exported end-to-end YOLO model."""

    def __init__(
        self,
        model_path: Path,
        labels: Sequence[str],
        confidence: float,
        iou_threshold: float,
        input_size: int = 640,
    ) -> None:
        _require_file(model_path, "ONNX model")
        if not labels:
            raise PreviewError("At least one class label is required.")
        if not 0.0 <= confidence <= 1.0:
            raise PreviewError("Confidence must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise PreviewError("IoU threshold must be between 0 and 1.")

        self.model_path = model_path
        self.labels = tuple(labels)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        try:
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
        except cv2.error as exc:
            raise PreviewError(f"OpenCV could not load the ONNX model: {exc}") from exc

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, frame: "np.ndarray") -> InferenceResult:
        model_input, scale, pad_x, pad_y = _letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(
            model_input,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        started = time.perf_counter()
        raw_output = self.net.forward()
        inference_ms = (time.perf_counter() - started) * 1000.0

        rows = np.asarray(raw_output).reshape(-1, raw_output.shape[-1])
        if rows.shape[1] < 6:
            raise PreviewError(
                "Unexpected ONNX output shape "
                f"{tuple(raw_output.shape)}; expected rows of six values."
            )

        height, width = frame.shape[:2]
        detections: List[Detection] = []
        for row in rows:
            confidence = float(row[4])
            if not math.isfinite(confidence) or confidence < self.confidence:
                continue

            class_id = int(round(float(row[5])))
            label = (
                self.labels[class_id]
                if 0 <= class_id < len(self.labels)
                else f"class_{class_id}"
            )

            x1 = int(round((float(row[0]) - pad_x) / scale))
            y1 = int(round((float(row[1]) - pad_y) / scale))
            x2 = int(round((float(row[2]) - pad_x) / scale))
            y2 = int(round((float(row[3]) - pad_y) / scale))
            x1 = min(max(x1, 0), max(width - 1, 0))
            y1 = min(max(y1, 0), max(height - 1, 0))
            x2 = min(max(x2, 0), max(width - 1, 0))
            y2 = min(max(y2, 0), max(height - 1, 0))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                )
            )

        detections = _class_aware_nms(detections, self.iou_threshold)
        return InferenceResult(detections=detections, inference_ms=inference_ms)


def _box_iou(first: Detection, second: Detection) -> float:
    first_x1, first_y1, first_x2, first_y2 = first.box
    second_x1, second_y1, second_x2, second_y2 = second.box
    intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _class_aware_nms(
    detections: Iterable[Detection], iou_threshold: float
) -> Sequence[Detection]:
    """Suppress duplicate boxes while never mixing different classes."""

    kept: List[Detection] = []
    for candidate in sorted(
        detections, key=lambda item: item.confidence, reverse=True
    ):
        overlaps = (
            existing.class_id == candidate.class_id
            and _box_iou(existing, candidate) > iou_threshold
            for existing in kept
        )
        if not any(overlaps):
            kept.append(candidate)
    return kept


def _draw_label(
    image: "np.ndarray",
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    x, y = origin
    y = max(y, text_height + baseline + 4)
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 5),
        (x + text_width + 8, y),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - baseline - 2),
        font,
        font_scale,
        (15, 15, 15),
        thickness,
        lineType=cv2.LINE_AA,
    )


def annotate(
    frame: "np.ndarray",
    result: InferenceResult,
    display_fps: Optional[float] = None,
    confidence: Optional[float] = None,
) -> "np.ndarray":
    """Draw detections and a compact status panel."""

    output = frame.copy()
    height, width = output.shape[:2]
    scale = max(0.45, min(width, height) / 900.0)
    line_width = max(2, int(round(scale * 3)))
    font_scale = max(0.45, scale * 0.65)

    for detection in result.detections:
        x1, y1, x2, y2 = detection.box
        color = COLORS[detection.class_id % len(COLORS)]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width)
        _draw_label(
            output,
            f"{detection.label} {detection.confidence:.0%}",
            (x1, y1),
            color,
            font_scale,
            max(1, line_width // 2),
        )

    status_parts = [
        f"objects {len(result.detections)}",
        f"inference {result.inference_ms:.1f} ms",
    ]
    if display_fps is not None and display_fps > 0:
        status_parts.append(f"stream {display_fps:.1f} FPS")
    if confidence is not None:
        status_parts.append(f"conf {confidence:.2f}")
    status = "  |  ".join(status_parts)
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_scale = max(0.48, font_scale * 0.9)
    panel_thickness = max(1, line_width // 2)
    (status_width, status_height), baseline = cv2.getTextSize(
        status, font, panel_scale, panel_thickness
    )
    cv2.rectangle(
        output,
        (0, 0),
        (min(width, status_width + 24), status_height + baseline + 18),
        (20, 24, 30),
        thickness=-1,
    )
    cv2.putText(
        output,
        status,
        (12, status_height + 8),
        font,
        panel_scale,
        (245, 245, 245),
        panel_thickness,
        lineType=cv2.LINE_AA,
    )
    return output


def _detection_summary(detections: Sequence[Detection]) -> str:
    if not detections:
        return "no detections"
    return ", ".join(
        f"{item.label} {item.confidence:.1%}" for item in detections
    )


def _write_image(path: Path, image: "np.ndarray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise PreviewError(f"Could not save image: {path}")


def _image_output_path(source: Path, output_dir: Path) -> Path:
    return output_dir / f"{source.stem}_detected.jpg"


def _video_output_path(source_name: str, output_dir: Path) -> Path:
    safe_stem = Path(source_name).stem if source_name else "camera"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return output_dir / f"{safe_stem}_detected_{timestamp}.mp4"


def _photo_paths(output_dir: Path) -> Tuple[Path, Path]:
    """Return collision-resistant paths for an original/detected photo pair."""

    now = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    milliseconds = int((now % 1.0) * 1000)
    stem = f"photo_{timestamp}-{milliseconds:03d}"
    photo_dir = output_dir / "photos"
    return (
        photo_dir / f"{stem}_original.jpg",
        photo_dir / f"{stem}_detected.jpg",
    )


def save_photo_pair(
    original: "np.ndarray", detected: "np.ndarray", output_dir: Path
) -> Tuple[Path, Path]:
    """Save both the untouched camera frame and its detection rendering."""

    original_path, detected_path = _photo_paths(output_dir)
    _write_image(original_path, original)
    _write_image(detected_path, detected)
    return original_path, detected_path


def _draw_stream_help(image: "np.ndarray") -> None:
    """Draw interaction hints on the display copy, not on saved media."""

    text = "C / ENTER / CLICK: PHOTO   SPACE: PAUSE   +/-: CONF   Q: QUIT"
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(width, height) / 1700.0)
    thickness = max(1, int(round(font_scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    panel_height = text_height + baseline + 16
    cv2.rectangle(
        image,
        (0, max(0, height - panel_height)),
        (min(width, text_width + 24), height),
        (20, 24, 30),
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (12, height - baseline - 7),
        font,
        font_scale,
        (245, 245, 245),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_photo_notice(image: "np.ndarray") -> None:
    """Show a short shutter-style confirmation on the live preview."""

    text = "PHOTO SAVED"
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(width, height) / 650.0)
    thickness = max(2, int(round(font_scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    left = max(0, (width - text_width) // 2 - 18)
    top = max(0, (height - text_height) // 2 - 18)
    right = min(width - 1, left + text_width + 36)
    bottom = min(height - 1, top + text_height + baseline + 36)
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    cv2.putText(
        image,
        text,
        (left + 18, top + text_height + 12),
        font,
        font_scale,
        (90, 230, 120),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _request_photo_on_click(event, _x, _y, _flags, state) -> None:
    if event == cv2.EVENT_LBUTTONUP:
        state["requested"] = True


def _show_still(
    image: "np.ndarray", save_callback, initial_saved_path: Optional[Path] = None
) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, image)
    if initial_saved_path is not None:
        print(f"Saved: {initial_saved_path}")
    print("Controls: Q/Esc close, S save result")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            saved_path = save_callback()
            print(f"Saved: {saved_path}")
    cv2.destroyWindow(WINDOW_NAME)


def run_image(
    detector: OnnxDetector,
    source: Path,
    output_dir: Path,
    show: bool,
    save: bool,
) -> None:
    _require_file(source, "Input image")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise PreviewError(f"OpenCV could not read image: {source}")

    result = detector.predict(image)
    rendered = annotate(image, result, confidence=detector.confidence)
    output_path = _image_output_path(source, output_dir)
    saved_path: Optional[Path] = None
    if save:
        _write_image(output_path, rendered)
        saved_path = output_path

    print(
        f"Image: {source}\n"
        f"Result: {_detection_summary(result.detections)}\n"
        f"Inference: {result.inference_ms:.1f} ms"
    )
    if show:
        _show_still(
            rendered,
            save_callback=lambda: (_write_image(output_path, rendered) or output_path),
            initial_saved_path=saved_path,
        )
    elif saved_path is not None:
        print(f"Saved: {saved_path}")


def run_image_directory(
    detector: OnnxDetector,
    source: Path,
    output_dir: Path,
) -> None:
    candidates = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    real_images = [path for path in candidates if not _is_git_lfs_pointer(path)]
    if not real_images:
        raise PreviewError(
            f"No readable images found in {source}. "
            "If this is a Git LFS checkout, run `git lfs pull` first."
        )

    print(f"Processing {len(real_images)} images from {source}")
    total_detections = 0
    for index, path in enumerate(real_images, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[{index}/{len(real_images)}] skipped unreadable image: {path}")
            continue
        result = detector.predict(image)
        total_detections += len(result.detections)
        rendered = annotate(image, result, confidence=detector.confidence)
        target = _image_output_path(path, output_dir)
        _write_image(target, rendered)
        print(
            f"[{index}/{len(real_images)}] {path.name}: "
            f"{_detection_summary(result.detections)}"
        )
    print(f"Saved batch results to {output_dir} ({total_detections} detections)")


def _open_capture(source: Union[int, Path]) -> "cv2.VideoCapture":
    if isinstance(source, int) and sys.platform == "darwin":
        capture = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        capture.release()
        if isinstance(source, int):
            raise PreviewError(
                f"Could not open camera {source}. On macOS, allow camera access "
                "for Terminal or Codex in System Settings > Privacy & Security > Camera."
            )
        raise PreviewError(f"Could not open video: {source}")
    return capture


def run_stream(
    detector: OnnxDetector,
    source: Union[int, Path],
    output_dir: Path,
    show: bool,
    save: bool,
    max_frames: Optional[int],
    camera_width: int,
    camera_height: int,
) -> None:
    if isinstance(source, Path):
        _require_file(source, "Input video")
    elif not show and max_frames is None:
        raise PreviewError("Camera with --no-show also requires --max-frames.")

    capture = _open_capture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    input_fps = capture.get(cv2.CAP_PROP_FPS)
    writer = None
    output_path = _video_output_path(
        str(source) if isinstance(source, Path) else f"camera_{source}", output_dir
    )
    frame_index = 0
    fps_ema: Optional[float] = None
    paused = False
    last_rendered: Optional["np.ndarray"] = None
    last_original: Optional["np.ndarray"] = None
    photo_notice_until = 0.0
    mouse_state = {"requested": False}

    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, _request_photo_on_click, mouse_state)
        print(
            "Controls: C/Enter/click take photo, Q/Esc quit, Space pause, "
            "+/- change confidence"
        )

    try:
        while True:
            if not paused:
                frame_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break

                result = detector.predict(frame)
                last_original = frame.copy()
                elapsed = max(time.perf_counter() - frame_started, 1e-9)
                instant_fps = 1.0 / elapsed
                fps_ema = (
                    instant_fps
                    if fps_ema is None
                    else 0.15 * instant_fps + 0.85 * fps_ema
                )
                last_rendered = annotate(
                    frame,
                    result,
                    display_fps=fps_ema,
                    confidence=detector.confidence,
                )
                frame_index += 1

                if save:
                    if writer is None:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        height, width = last_rendered.shape[:2]
                        target_fps = (
                            input_fps
                            if math.isfinite(input_fps) and input_fps > 0
                            else min(max(fps_ema or 20.0, 5.0), 30.0)
                        )
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            target_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise PreviewError(f"Could not create video: {output_path}")
                    writer.write(last_rendered)

                if frame_index == 1 or frame_index % 30 == 0:
                    print(
                        f"Frame {frame_index}: "
                        f"{_detection_summary(result.detections)}; "
                        f"{result.inference_ms:.1f} ms"
                    )

                if max_frames is not None and frame_index >= max_frames:
                    break

            if not show:
                continue
            if last_rendered is not None:
                display_frame = last_rendered.copy()
                _draw_stream_help(display_frame)
                if time.monotonic() < photo_notice_until:
                    _draw_photo_notice(display_frame)
                cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(20 if paused else 1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("+"), ord("=")):
                detector.confidence = min(0.95, detector.confidence + 0.05)
                print(f"Confidence: {detector.confidence:.2f}")
            elif key in (ord("-"), ord("_")):
                detector.confidence = max(0.05, detector.confidence - 0.05)
                print(f"Confidence: {detector.confidence:.2f}")

            photo_requested = key in (
                10,
                13,
                ord("c"),
                ord("s"),
            ) or bool(mouse_state["requested"])
            mouse_state["requested"] = False
            if photo_requested and last_original is not None and last_rendered is not None:
                original_path, detected_path = save_photo_pair(
                    last_original, last_rendered, output_dir
                )
                photo_notice_until = time.monotonic() + 1.2
                print("Photo saved:")
                print(f"  Original: {original_path}")
                print(f"  Detected: {detected_path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyWindow(WINDOW_NAME)

    if save and writer is not None:
        print(f"Saved video: {output_path}")
    print(f"Processed {frame_index} frames")


def _parse_labels(value: str) -> Sequence[str]:
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not labels:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated label.")
    return labels


def _parse_source(value: str) -> Union[int, Path]:
    stripped = value.strip()
    if stripped.isdigit() or (
        stripped.startswith("-") and stripped[1:].isdigit()
    ):
        return int(stripped)
    return Path(stripped).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview the trained keyboard/monitor/mouse detector using OpenCV DNN."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, image, video, or directory of images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Exported end-to-end ONNX model.",
    )
    parser.add_argument(
        "--labels",
        type=_parse_labels,
        default=DEFAULT_LABELS,
        help="Comma-separated class names in model order.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence shown.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold used to suppress duplicate boxes.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated image/video. Directory input is always saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results and snapshots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an OpenCV window (useful for batch checks).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop a video/camera after this many frames.",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise PreviewError("--max-frames must be positive.")

    source = _parse_source(args.source)
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    detector = OnnxDetector(
        model_path=model_path,
        labels=args.labels,
        confidence=args.conf,
        iou_threshold=args.iou,
    )

    print(f"Model: {model_path}")
    print(f"Classes: {', '.join(detector.labels)}")
    print("Backend: OpenCV DNN / CPU")
    if isinstance(source, int):
        print(f"Source: camera {source}")
        run_stream(
            detector=detector,
            source=source,
            output_dir=output_dir,
            show=not args.no_show,
            save=args.save,
            max_frames=args.max_frames,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        )
    elif source.is_dir():
        print(f"Source: image directory {source}")
        run_image_directory(detector, source, output_dir)
    elif source.suffix.lower() in IMAGE_SUFFIXES:
        print(f"Source: image {source}")
        run_image(
            detector=detector,
            source=source,
            output_dir=output_dir,
            show=not args.no_show,
            save=args.save,
        )
    elif source.suffix.lower() in VIDEO_SUFFIXES:
        print(f"Source: video {source}")
        run_stream(
            detector=detector,
            source=source,
            output_dir=output_dir,
            show=not args.no_show,
            save=args.save,
            max_frames=args.max_frames,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        )
    else:
        raise PreviewError(
            f"Unsupported source: {source}\n"
            "Use a camera index, image, video, or directory of images."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreviewError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
