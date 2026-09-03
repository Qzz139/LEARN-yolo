#!/usr/bin/env python3
"""Audit a YOLO detection dataset before starting an expensive training run."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to the dataset data.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("data.yaml must contain a mapping")
    return data


def class_names(config: dict[str, Any]) -> list[str]:
    raw_names = config.get("names")
    if isinstance(raw_names, list):
        names = [str(value).strip() for value in raw_names]
    elif isinstance(raw_names, dict):
        try:
            ids = sorted(int(key) for key in raw_names)
        except (TypeError, ValueError) as exc:
            raise ValueError("class IDs in names must be integers") from exc
        if ids != list(range(len(ids))):
            raise ValueError("class IDs in names must be contiguous and start at zero")
        names = [str(raw_names.get(index, raw_names.get(str(index)))).strip() for index in ids]
    else:
        raise ValueError("names must be a list or mapping")
    if not names or any(not name for name in names):
        raise ValueError("names must contain at least one non-empty class name")
    if len(set(names)) != len(names):
        raise ValueError("class names must be unique")
    configured_count = config.get("nc")
    if configured_count is not None and int(configured_count) != len(names):
        raise ValueError(f"nc={configured_count} does not match {len(names)} names")
    return names


def resolve_split(config_path: Path, config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"split {key!r} must be a non-empty path string")
    root_value = config.get("path")
    root = config_path.parent
    if isinstance(root_value, str) and root_value.strip():
        configured_root = Path(root_value).expanduser()
        root = configured_root if configured_root.is_absolute() else root / configured_root
    split = Path(value).expanduser()
    return split.resolve() if split.is_absolute() else (root / split).resolve()


def label_path_for(image_path: Path, image_root: Path, label_root: Path) -> Path:
    relative = image_path.relative_to(image_root)
    return (label_root / relative).with_suffix(".txt")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config_path = args.data.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    try:
        config = load_config(config_path)
        names = class_names(config)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    class_boxes: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    split_rows: list[tuple[str, int, int, int, int]] = []
    digests: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for split_name in ("train", "val", "test"):
        try:
            image_root = resolve_split(config_path, config, split_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not image_root.is_dir():
            errors.append(f"{split_name}: image directory does not exist: {image_root}")
            continue

        if image_root.name.lower() != "images":
            warnings.append(
                f"{split_name}: expected an images directory, received {image_root}"
            )
        label_root = image_root.parent / "labels"
        if not label_root.is_dir():
            errors.append(f"{split_name}: label directory does not exist: {label_root}")
            continue

        images = sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        label_files = sorted(label_root.rglob("*.txt"))
        expected_labels = {
            label_path_for(image_path, image_root, label_root) for image_path in images
        }
        missing_labels = sorted(path for path in expected_labels if not path.is_file())
        orphan_labels = sorted(path for path in label_files if path not in expected_labels)
        for path in missing_labels[:20]:
            errors.append(f"{split_name}: missing label for {path.relative_to(label_root)}")
        if len(missing_labels) > 20:
            errors.append(f"{split_name}: {len(missing_labels) - 20} more labels are missing")
        for path in orphan_labels[:20]:
            errors.append(f"{split_name}: orphan label {path.relative_to(label_root)}")
        if len(orphan_labels) > 20:
            errors.append(f"{split_name}: {len(orphan_labels) - 20} more labels are orphaned")

        empty_labels = 0
        split_boxes = 0
        size_counts: Counter[tuple[int, int]] = Counter()
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    size_counts[image.size] += 1
            except Exception as exc:
                errors.append(f"{split_name}: unreadable image {image_path.name}: {exc}")
                continue

            digests[file_digest(image_path)].append((split_name, image_path))
            label_path = label_path_for(image_path, image_root, label_root)
            if not label_path.is_file():
                continue
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                errors.append(f"{split_name}: unreadable label {label_path.name}: {exc}")
                continue
            lines = [line.strip() for line in lines if line.strip()]
            if not lines:
                empty_labels += 1
                continue

            image_classes: set[int] = set()
            for line_number, line in enumerate(lines, 1):
                fields = line.split()
                location = f"{label_path.name}:{line_number}"
                if len(fields) != 5:
                    errors.append(f"{split_name}: {location} has {len(fields)} fields, expected 5")
                    continue
                try:
                    class_value, *coordinates = (float(field) for field in fields)
                except ValueError:
                    errors.append(f"{split_name}: {location} contains a non-numeric value")
                    continue
                class_id = int(class_value)
                if class_value != class_id or not 0 <= class_id < len(names):
                    errors.append(f"{split_name}: {location} has invalid class ID {class_value}")
                    continue
                if not all(math.isfinite(value) for value in coordinates):
                    errors.append(f"{split_name}: {location} contains a non-finite coordinate")
                    continue
                x_center, y_center, width, height = coordinates
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                    errors.append(f"{split_name}: {location} has a center outside [0, 1]")
                    continue
                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    errors.append(f"{split_name}: {location} has an invalid box size")
                    continue
                if (
                    x_center - width / 2 < -1e-6
                    or y_center - height / 2 < -1e-6
                    or x_center + width / 2 > 1.0 + 1e-6
                    or y_center + height / 2 > 1.0 + 1e-6
                ):
                    warnings.append(f"{split_name}: {location} extends outside the image")
                class_boxes[class_id] += 1
                image_classes.add(class_id)
                split_boxes += 1
            for class_id in image_classes:
                class_images[class_id] += 1

        if len(size_counts) > 1:
            common = ", ".join(
                f"{width}x{height} ({count})"
                for (width, height), count in size_counts.most_common(5)
            )
            warnings.append(f"{split_name}: mixed image sizes: {common}")
        split_rows.append(
            (split_name, len(images), len(label_files), split_boxes, empty_labels)
        )

    for duplicate_paths in digests.values():
        duplicate_splits = {split for split, _ in duplicate_paths}
        if len(duplicate_splits) > 1:
            joined = ", ".join(
                f"{split}:{path.name}" for split, path in duplicate_paths
            )
            errors.append(f"identical image appears across splits: {joined}")

    print(f"Dataset: {config_path}")
    print(f"Classes ({len(names)}): {', '.join(names)}")
    print("\nSplits:")
    print(f"{'split':<8} {'images':>8} {'labels':>8} {'boxes':>8} {'empty':>8}")
    for split_name, images, labels, boxes, empty in split_rows:
        print(f"{split_name:<8} {images:>8} {labels:>8} {boxes:>8} {empty:>8}")

    print("\nClass distribution:")
    print(f"{'id':>3} {'class':<20} {'boxes':>8} {'images':>8}")
    for class_id, name in enumerate(names):
        print(
            f"{class_id:>3} {name:<20} "
            f"{class_boxes[class_id]:>8} {class_images[class_id]:>8}"
        )
        if class_boxes[class_id] == 0:
            errors.append(f"class {class_id} ({name}) has no boxes")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings[:50]:
            print(f"- {warning}")
        if len(warnings) > 50:
            print(f"- ... {len(warnings) - 50} more warnings")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"- ... {len(errors) - 50} more errors")
        return 1

    print("\nAudit passed: no blocking dataset errors found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
