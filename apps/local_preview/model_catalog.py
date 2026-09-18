# 模型目录解析：统一源码运行和打包运行时的路径、类别与模型选择规则。
"""Resolve versioned detector models for source and packaged executions."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


MANIFEST_ENV = "YOLO_PREVIEW_MANIFEST"
MODEL_ENV = "YOLO_PREVIEW_MODEL"
MODEL_ID_ENV = "YOLO_PREVIEW_MODEL_ID"
DEFAULT_LABELS = ("keyboard", "monitor", "mouse")


class ModelCatalogError(RuntimeError):
    """Raised when the model manifest or a requested entry is invalid."""


@dataclass(frozen=True)
class ModelSelection:
    """Resolved model settings used by the inference application."""

    model_id: str
    model_path: Path
    labels: Tuple[str, ...]
    input_size: int
    status: str


def runtime_root() -> Path:
    """Return the project root in source mode or package root when frozen."""

    # 冻结程序以可执行文件所在目录为根；源码运行则回溯到仓库根目录。
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    configured = os.environ.get(MANIFEST_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return runtime_root() / "weights" / "manifest.json"


def _load_manifest(path: Optional[Path] = None) -> Tuple[Path, Dict[str, object]]:
    manifest_path = (
        path.expanduser().resolve() if path is not None else default_manifest_path()
    )
    if not manifest_path.is_file():
        raise ModelCatalogError(f"Model manifest does not exist: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelCatalogError(
            f"Could not read model manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ModelCatalogError(
            f"Model manifest must contain a 'models' object: {manifest_path}"
        )
    return manifest_path, data


def _labels_from_manifest(
    data: Mapping[str, object],
    model: Optional[Mapping[str, object]] = None,
) -> Tuple[str, ...]:
    """Resolve model-specific labels, falling back to legacy global labels."""

    # 优先使用单个模型的类别表，兼容旧版清单中的全局类别表。
    raw_labels = model.get("classes") if model and "classes" in model else None
    raw_labels = raw_labels if raw_labels is not None else data.get("classes", DEFAULT_LABELS)
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ModelCatalogError("Model manifest 'classes' must be a non-empty list.")
    labels = tuple(str(label).strip() for label in raw_labels)
    if any(not label for label in labels):
        raise ModelCatalogError("Model manifest contains an empty class name.")
    return labels


def _resolve_artifact(manifest_path: Path, artifact: str) -> Path:
    artifact_path = Path(artifact).expanduser()
    # 清单中的相对模型路径以清单所在目录为基准，与启动工作目录无关。
    if not artifact_path.is_absolute():
        artifact_path = manifest_path.parent / artifact_path
    return artifact_path.resolve()


def resolve_model(
    *,
    manifest_path: Optional[Path] = None,
    explicit_model: Optional[Path] = None,
    explicit_model_id: Optional[str] = None,
) -> ModelSelection:
    """Resolve CLI, environment, then active-manifest model selection."""

    # 选择优先级：显式路径、环境路径、显式模型 ID、环境 ID、清单默认 ID。
    environment_model = os.environ.get(MODEL_ENV, "").strip()
    requested_path = explicit_model
    if requested_path is None and environment_model:
        requested_path = Path(environment_model)

    try:
        resolved_manifest_path, data = _load_manifest(manifest_path)
        labels = _labels_from_manifest(data)
        default_input_size = int(data.get("input_size", 640))
    except ModelCatalogError:
        # 指定独立模型路径时，即使清单不可用，也允许采用默认标签继续解析。
        if requested_path is None:
            raise
        labels = DEFAULT_LABELS
        default_input_size = 640
        return ModelSelection(
            model_id="custom",
            model_path=requested_path.expanduser().resolve(),
            labels=labels,
            input_size=default_input_size,
            status="custom",
        )

    if requested_path is not None:
        return ModelSelection(
            model_id="custom",
            model_path=requested_path.expanduser().resolve(),
            labels=labels,
            input_size=default_input_size,
            status="custom",
        )

    environment_model_id = os.environ.get(MODEL_ID_ENV, "").strip()
    selected_id = (
        (explicit_model_id or "").strip()
        or environment_model_id
        or str(data.get("active_model", "")).strip()
    )
    if not selected_id:
        raise ModelCatalogError("No model ID was selected and 'active_model' is empty.")

    models = data["models"]
    assert isinstance(models, dict)
    entry = models.get(selected_id)
    if not isinstance(entry, dict):
        available = ", ".join(str(model_id) for model_id in models)
        raise ModelCatalogError(
            f"Unknown model ID '{selected_id}'. Available models: {available}"
        )
    labels = _labels_from_manifest(data, entry)

    # 只有具有 ONNX 产物的条目才能用于桌面预览，计划训练的条目会明确报错。
    artifacts = entry.get("artifacts", {})
    artifact = artifacts.get("onnx") if isinstance(artifacts, dict) else None
    if not isinstance(artifact, str) or not artifact.strip():
        status = str(entry.get("status", "unknown"))
        raise ModelCatalogError(
            f"Model '{selected_id}' has no ONNX artifact yet (status: {status})."
        )

    input_size = int(entry.get("input_size", default_input_size))
    if input_size <= 0:
        raise ModelCatalogError(f"Model '{selected_id}' has an invalid input size.")
    return ModelSelection(
        model_id=selected_id,
        model_path=_resolve_artifact(resolved_manifest_path, artifact),
        labels=labels,
        input_size=input_size,
        status=str(entry.get("status", "unknown")),
    )


def catalog_lines(manifest_path: Optional[Path] = None) -> Sequence[str]:
    """Return human-readable candidate rows for ``--list-models``."""

    resolved_manifest_path, data = _load_manifest(manifest_path)
    active_model = str(data.get("active_model", ""))
    models = data["models"]
    assert isinstance(models, dict)
    lines = []
    for model_id, raw_entry in models.items():
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        artifacts = entry.get("artifacts", {})
        artifact = artifacts.get("onnx") if isinstance(artifacts, dict) else None
        artifact_state = "not exported"
        if isinstance(artifact, str) and artifact.strip():
            artifact_state = (
                "ready"
                if _resolve_artifact(resolved_manifest_path, artifact).is_file()
                else "missing file"
            )
        marker = "*" if model_id == active_model else " "
        family = str(entry.get("family", model_id))
        status = str(entry.get("status", "unknown"))
        lines.append(
            f"{marker} {model_id}: {family}; status={status}; ONNX={artifact_state}"
        )
    return tuple(lines)


def iter_packaged_onnx(manifest_path: Optional[Path] = None) -> Iterable[Path]:
    """Yield every existing ONNX artifact recorded in the manifest."""

    resolved_manifest_path, data = _load_manifest(manifest_path)
    models = data["models"]
    assert isinstance(models, dict)
    for raw_entry in models.values():
        if not isinstance(raw_entry, dict):
            continue
        artifacts = raw_entry.get("artifacts", {})
        artifact = artifacts.get("onnx") if isinstance(artifacts, dict) else None
        if isinstance(artifact, str) and artifact.strip():
            path = _resolve_artifact(resolved_manifest_path, artifact)
            if path.is_file():
                yield path
