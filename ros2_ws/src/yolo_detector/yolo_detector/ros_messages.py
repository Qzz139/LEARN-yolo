"""ROS image and detection message compatibility helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def image_message_to_bgr(bridge: Any, image_message: Image) -> Any:
    """Convert common ROS image encodings without OpenCV 4/5 ABI mixing."""

    frame = bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")
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


def bgr8_image_message(bridge: Any, frame: Any, header: Header) -> Image:
    """Create bgr8 safely when Foxy cv_bridge meets OpenCV 5."""

    message = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
    if message.encoding != "8UC3":
        raise ValueError(
            "Annotated frame must be an 8-bit, three-channel BGR image, "
            f"but cv_bridge reported {message.encoding!r}."
        )
    message.encoding = "bgr8"
    message.header = header
    return message


def build_detections_message(
    result: Any,
    header: Header,
    class_names: Sequence[str],
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
        _set_bbox(
            detection,
            center_x=(x_min + x_max) / 2.0,
            center_y=(y_min + y_max) / 2.0,
            width=max(0.0, x_max - x_min),
            height=max(0.0, y_max - y_min),
        )

        hypothesis = ObjectHypothesisWithPose()
        _set_hypothesis(
            hypothesis,
            class_name=_resolve_class_name(
                class_id, model_names, class_names
            ),
            confidence=float(confidence),
        )
        detection.results.append(hypothesis)
        output.detections.append(detection)

    return output


def _set_bbox(
    detection: Detection2D,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> None:
    """Fill BoundingBox2D on both ROS 2 Foxy and newer schemas."""

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


def _set_hypothesis(
    hypothesis: ObjectHypothesisWithPose,
    class_name: str,
    confidence: float,
) -> None:
    """Fill ObjectHypothesisWithPose on Foxy and newer schemas."""

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
    class_id: int,
    model_names: Mapping[int, str] | Sequence[str],
    class_names: Sequence[str],
) -> str:
    # Model metadata is authoritative and automatically follows retrained models.
    if isinstance(model_names, Mapping):
        if class_id in model_names:
            return str(model_names[class_id])
    if 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    if 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)
