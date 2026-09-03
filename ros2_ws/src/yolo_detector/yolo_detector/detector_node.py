"""ROS 2 node that runs Ultralytics YOLO on camera images."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header
from std_srvs.srv import SetBool, Trigger
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .media_capture import MediaCapture


class YoloDetectorNode(Node):
    """Receive camera frames, run YOLO, and publish standard ROS 2 messages."""

    def __init__(self) -> None:
        super().__init__("yolo_detector")

        self._declare_parameters()

        self._bridge = CvBridge()
        self._model = self._load_model()
        self._class_names = list(self.get_parameter("class_names").value)
        self._frame_count = 0
        self._last_completed_at: Optional[float] = None
        self._fps_ema: Optional[float] = None

        capture_output_text = str(
            self.get_parameter("capture_output_dir").value
        ).strip()
        capture_output_dir = (
            Path(capture_output_text).expanduser()
            if capture_output_text
            else Path.home() / "yolo_captures"
        )
        self._media_capture = MediaCapture(
            output_dir=capture_output_dir,
            recording_fps=float(self.get_parameter("recording_fps").value),
            recording_codec=str(self.get_parameter("recording_codec").value),
            cv2_module=cv2,
        )
        if bool(self.get_parameter("record_on_start").value):
            self._media_capture.request_recording()

        detections_topic = str(self.get_parameter("detections_topic").value)
        fps_topic = str(self.get_parameter("fps_topic").value)
        annotated_topic = str(self.get_parameter("annotated_image_topic").value)
        self._publish_annotated = bool(
            self.get_parameter("publish_annotated_image").value
        )

        self._detections_publisher = self.create_publisher(
            Detection2DArray, detections_topic, 10
        )
        self._fps_publisher = self.create_publisher(Float32, fps_topic, 10)
        self._annotated_publisher = None
        if self._publish_annotated:
            self._annotated_publisher = self.create_publisher(
                Image, annotated_topic, qos_profile_sensor_data
            )

        self._snapshot_service = self.create_service(
            Trigger, "/yolo/save_snapshot", self._save_snapshot
        )
        self._recording_service = self.create_service(
            SetBool, "/yolo/set_recording", self._set_recording
        )

        self._capture: Optional[cv2.VideoCapture] = None
        self._input_timer = None
        self._static_frame = None
        self._image_subscription = None

        source_mode = str(self.get_parameter("source_mode").value).lower()
        if source_mode == "topic":
            self._start_topic_input()
        elif source_mode == "camera":
            self._start_camera_input()
        elif source_mode == "image":
            self._start_image_input()
        else:
            raise ValueError(
                "Parameter 'source_mode' must be 'topic', 'camera', or 'image', "
                f"but received {source_mode!r}."
            )

        self.get_logger().info(
            f"YOLO detector ready: input={source_mode}, "
            f"detections={detections_topic}, fps={fps_topic}"
        )
        self.get_logger().info(
            "Capture controls: snapshot=/yolo/save_snapshot, "
            "recording=/yolo/set_recording"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model_path", "")
        self.declare_parameter("source_mode", "topic")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_source", "0")
        self.declare_parameter("camera_backend", "auto")
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        self.declare_parameter("camera_reconnect_interval", 2.0)
        self.declare_parameter("camera_read_failure_threshold", 3)
        self.declare_parameter("image_source", "")
        self.declare_parameter("image_fps", 10.0)

        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("annotated_image_topic", "/yolo/annotated_image")
        self.declare_parameter("fps_topic", "/yolo/fps")
        self.declare_parameter("publish_annotated_image", True)

        self.declare_parameter("capture_output_dir", "")
        self.declare_parameter("record_on_start", False)
        self.declare_parameter("recording_fps", 10.0)
        self.declare_parameter("recording_codec", "mp4v")

        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("device", "")
        self.declare_parameter("class_names", ["keyboard", "monitor", "mouse"])
        self.declare_parameter("log_every_n_frames", 30)

    def _load_model(self) -> Any:
        model_path_value = str(self.get_parameter("model_path").value).strip()
        if not model_path_value:
            raise RuntimeError(
                "Parameter 'model_path' is empty. Set it to an absolute .pt, "
                ".onnx, or Jetson-built .engine model path."
            )

        model_path = Path(model_path_value).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model was not found: {model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Could not import Ultralytics or one of its dependencies. "
                f"Original import error: {exc}"
            ) from exc

        self.get_logger().info(f"Loading YOLO model: {model_path}")
        return YOLO(str(model_path), task="detect")

    def _start_topic_input(self) -> None:
        image_topic = str(self.get_parameter("image_topic").value)
        self._image_subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")

    def _start_camera_input(self) -> None:
        self._camera_source_text = str(
            self.get_parameter("camera_source").value
        ).strip()
        self._camera_backend = str(
            self.get_parameter("camera_backend").value
        ).lower()
        self._camera_source: Any = (
            int(self._camera_source_text)
            if self._camera_source_text.lstrip("-").isdigit()
            else self._camera_source_text
        )
        if self._camera_backend not in {"auto", "gstreamer"}:
            raise ValueError(
                "Parameter 'camera_backend' must be 'auto' or 'gstreamer', "
                f"but received {self._camera_backend!r}."
            )

        self._camera_reconnect_interval = float(
            self.get_parameter("camera_reconnect_interval").value
        )
        self._camera_read_failure_threshold = int(
            self.get_parameter("camera_read_failure_threshold").value
        )
        if self._camera_reconnect_interval <= 0.0:
            raise ValueError(
                "Parameter 'camera_reconnect_interval' must be greater than zero."
            )
        if self._camera_read_failure_threshold <= 0:
            raise ValueError(
                "Parameter 'camera_read_failure_threshold' must be greater than zero."
            )

        requested_fps = float(self.get_parameter("camera_fps").value)
        if requested_fps <= 0.0:
            raise ValueError("Parameter 'camera_fps' must be greater than zero.")

        self._camera_read_failures = 0
        self._last_camera_reconnect_attempt = 0.0
        self._connect_camera(initial=True)
        self._input_timer = self.create_timer(
            1.0 / requested_fps, self._camera_callback
        )
        self.get_logger().info(
            f"Reading camera source {self._camera_source_text!r} at up to "
            f"{requested_fps:.1f} FPS; reconnect interval="
            f"{self._camera_reconnect_interval:.1f}s"
        )

    def _create_camera_capture(self) -> cv2.VideoCapture:
        if self._camera_backend == "gstreamer":
            return cv2.VideoCapture(self._camera_source, cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(self._camera_source)

    def _connect_camera(self, initial: bool = False) -> bool:
        attempted_at = time.monotonic()
        if (
            not initial
            and attempted_at - self._last_camera_reconnect_attempt
            < self._camera_reconnect_interval
        ):
            return False
        self._last_camera_reconnect_attempt = attempted_at

        try:
            capture = self._create_camera_capture()
        except Exception as exc:
            self._capture = None
            self.get_logger().warning(
                f"Camera {self._camera_source_text!r} could not be opened: "
                f"{exc}; retrying in {self._camera_reconnect_interval:.1f}s."
            )
            return False

        if not capture.isOpened():
            capture.release()
            self._capture = None
            self.get_logger().warning(
                f"Camera {self._camera_source_text!r} is unavailable; "
                f"retrying in {self._camera_reconnect_interval:.1f}s."
            )
            return False

        self._capture = capture
        self._camera_read_failures = 0
        if initial:
            self.get_logger().info(
                f"Opened camera source {self._camera_source_text!r}."
            )
        else:
            self.get_logger().info(
                f"Camera {self._camera_source_text!r} reconnected."
            )
        return True

    def _start_image_input(self) -> None:
        image_source = Path(
            str(self.get_parameter("image_source").value).strip()
        ).expanduser()
        if not image_source.is_file():
            raise FileNotFoundError(
                f"Static image source was not found: {image_source}"
            )

        frame = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not decode static image: {image_source}")

        requested_fps = float(self.get_parameter("image_fps").value)
        if requested_fps <= 0.0:
            raise ValueError("Parameter 'image_fps' must be greater than zero.")

        self._static_frame = frame
        self._input_timer = self.create_timer(
            1.0 / requested_fps, self._static_image_callback
        )
        self.get_logger().info(
            f"Repeating static image {str(image_source)!r} at up to "
            f"{requested_fps:.1f} FPS"
        )

    def _image_callback(self, image_message: Image) -> None:
        try:
            frame = self._image_message_to_bgr(image_message)
            self._process_frame(frame, image_message.header)
        except Exception as exc:  # Keep the ROS node alive after a bad frame.
            self.get_logger().error(f"Failed to process ROS image: {exc}")

    def _camera_callback(self) -> None:
        if self._capture is None:
            self._connect_camera()
            return

        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            ok, frame = False, None
            self.get_logger().warning(f"Camera read raised an error: {exc}")

        if not ok or frame is None:
            self._camera_read_failures += 1
            if (
                self._camera_read_failures
                >= self._camera_read_failure_threshold
            ):
                self.get_logger().warning(
                    "Camera stream was lost; closing it and starting "
                    "automatic reconnect attempts."
                )
                self._capture.release()
                self._capture = None
                self._camera_read_failures = 0
                self._last_camera_reconnect_attempt = time.monotonic()
            return

        self._camera_read_failures = 0
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = str(self.get_parameter("camera_frame_id").value)

        try:
            self._process_frame(frame, header)
        except Exception as exc:  # Keep the ROS node alive after a bad frame.
            self.get_logger().error(f"Failed to process camera frame: {exc}")

    def _static_image_callback(self) -> None:
        if self._static_frame is None:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = str(self.get_parameter("camera_frame_id").value)

        try:
            self._process_frame(self._static_frame.copy(), header)
        except Exception as exc:  # Keep the ROS node alive after a bad frame.
            self.get_logger().error(f"Failed to process static image: {exc}")

    def _image_message_to_bgr(self, image_message: Image) -> Any:
        """Convert common ROS image encodings without OpenCV 4/5 ABI mixing."""

        frame = self._bridge.imgmsg_to_cv2(
            image_message, desired_encoding="passthrough"
        )
        encoding = image_message.encoding.lower()

        if encoding in {"bgr8", "8uc3"}:
            return frame
        if encoding == "rgb8":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if encoding in {"mono8", "8uc1"}:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        raise ValueError(
            f"Unsupported ROS image encoding {image_message.encoding!r}; "
            "expected bgr8, rgb8, mono8, bgra8, or rgba8."
        )

    def _bgr8_image_message(self, frame: Any, header: Header) -> Image:
        """Create bgr8 safely when Foxy cv_bridge meets OpenCV 5.

        Foxy cv_bridge validates explicit encodings using OpenCV 4 numeric type
        IDs. OpenCV 5 changed those IDs. The passthrough conversion serializes
        the same uint8 BGR bytes without that obsolete numeric-ID comparison.
        """

        message = self._bridge.cv2_to_imgmsg(frame, encoding="passthrough")
        if message.encoding != "8UC3":
            raise ValueError(
                "Annotated frame must be an 8-bit, three-channel BGR image, "
                f"but cv_bridge reported {message.encoding!r}."
            )
        message.encoding = "bgr8"
        message.header = header
        return message

    def _process_frame(self, frame: Any, header: Header) -> None:
        predict_options = {
            "source": frame,
            "imgsz": int(self.get_parameter("imgsz").value),
            "conf": float(self.get_parameter("conf_threshold").value),
            "iou": float(self.get_parameter("iou_threshold").value),
            "verbose": False,
        }
        device = str(self.get_parameter("device").value).strip()
        if device:
            predict_options["device"] = device

        result = self._model.predict(**predict_options)[0]

        detections_message = self._build_detections_message(result, header)
        self._detections_publisher.publish(detections_message)

        needs_annotated_frame = (
            self._annotated_publisher is not None
            or self._media_capture.recording_requested
        )
        if needs_annotated_frame:
            annotated_frame = result.plot()
            was_recording = self._media_capture.is_recording
            try:
                recording_path = self._media_capture.process_frame(
                    annotated_frame
                )
                if recording_path is not None and not was_recording:
                    self.get_logger().info(
                        f"Video recording started: {recording_path}"
                    )
            except Exception as exc:
                self._media_capture.stop_recording()
                self.get_logger().error(f"Video recording failed: {exc}")

            if self._annotated_publisher is not None:
                annotated_message = self._bgr8_image_message(
                    annotated_frame, header
                )
                self._annotated_publisher.publish(annotated_message)

        self._publish_measured_fps()

    def _save_snapshot(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        try:
            path = self._media_capture.save_snapshot()
            response.success = True
            response.message = str(path)
            self.get_logger().info(f"Snapshot saved: {path}")
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"Snapshot failed: {exc}")
        return response

    def _set_recording(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        if request.data:
            if self._media_capture.recording_requested:
                response.success = True
                response.message = "Recording is already active or armed."
            else:
                self._media_capture.request_recording()
                response.success = True
                response.message = "Recording will start on the next frame."
                self.get_logger().info(response.message)
            return response

        path = self._media_capture.stop_recording()
        response.success = True
        if path is None:
            response.message = "Recording was not active."
        else:
            response.message = f"Recording saved: {path}"
            self.get_logger().info(response.message)
        return response

    def _build_detections_message(
        self, result: Any, header: Header
    ) -> Detection2DArray:
        output = Detection2DArray()
        output.header = header

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return output

        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        model_names = result.names

        for xyxy, confidence, class_id_value in zip(
            coordinates, confidences, class_ids
        ):
            x_min, y_min, x_max, y_max = (float(value) for value in xyxy)
            class_id = int(class_id_value)

            detection = Detection2D()
            detection.header = header
            self._set_bbox(
                detection,
                center_x=(x_min + x_max) / 2.0,
                center_y=(y_min + y_max) / 2.0,
                width=max(0.0, x_max - x_min),
                height=max(0.0, y_max - y_min),
            )

            hypothesis = ObjectHypothesisWithPose()
            self._set_hypothesis(
                hypothesis,
                class_name=self._resolve_class_name(class_id, model_names),
                confidence=float(confidence),
            )
            detection.results.append(hypothesis)
            output.detections.append(detection)

        return output

    @staticmethod
    def _set_bbox(
        detection: Detection2D,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> None:
        """Fill BoundingBox2D on both ROS 2 Humble and Jazzy schemas."""

        center = detection.bbox.center
        if hasattr(center, "position"):
            center.position.x = center_x
            center.position.y = center_y
        else:
            center.x = center_x
            center.y = center_y
        center.theta = 0.0
        detection.bbox.size_x = width
        detection.bbox.size_y = height

    @staticmethod
    def _set_hypothesis(
        hypothesis: ObjectHypothesisWithPose,
        class_name: str,
        confidence: float,
    ) -> None:
        """Fill both Foxy and newer vision_msgs hypothesis schemas."""

        if hasattr(hypothesis, "hypothesis"):
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = confidence
            return
        if hasattr(hypothesis, "id") and hasattr(hypothesis, "score"):
            hypothesis.id = class_name
            hypothesis.score = confidence
            return
        raise TypeError("Unsupported vision_msgs ObjectHypothesisWithPose schema")

    def _resolve_class_name(
        self,
        class_id: int,
        model_names: Mapping[int, str] | Sequence[str],
    ) -> str:
        if 0 <= class_id < len(self._class_names):
            return str(self._class_names[class_id])
        if isinstance(model_names, Mapping):
            return str(model_names.get(class_id, class_id))
        if 0 <= class_id < len(model_names):
            return str(model_names[class_id])
        return str(class_id)

    def _publish_measured_fps(self) -> None:
        completed_at = time.perf_counter()
        if self._last_completed_at is None:
            self._last_completed_at = completed_at
            return

        frame_interval = completed_at - self._last_completed_at
        self._last_completed_at = completed_at
        if frame_interval <= 0.0:
            return

        instantaneous_fps = 1.0 / frame_interval
        if self._fps_ema is None:
            self._fps_ema = instantaneous_fps
        else:
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * instantaneous_fps

        fps_message = Float32()
        fps_message.data = float(self._fps_ema)
        self._fps_publisher.publish(fps_message)

        self._frame_count += 1
        log_every = int(self.get_parameter("log_every_n_frames").value)
        if log_every > 0 and self._frame_count % log_every == 0:
            self.get_logger().info(
                f"Output FPS (exponential moving average): {self._fps_ema:.2f}"
            )

    def destroy_node(self) -> bool:
        recording_path = self._media_capture.stop_recording()
        if recording_path is not None:
            self.get_logger().info(f"Recording saved: {recording_path}")
        self._media_capture.close()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[YoloDetectorNode] = None
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
