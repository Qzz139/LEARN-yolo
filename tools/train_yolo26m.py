#!/usr/bin/env python3
"""Reproducibly train YOLO26m on the project's current tabletop dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets/tabletop_v1/data.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "yolo26m.pt",
    )
    parser.add_argument("--name", default="yolo26m_tabletop_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted run from its last.pt checkpoint.",
    )
    return parser.parse_args()


def existing_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    data_path = existing_file(args.data, "Dataset configuration")
    project_path = PROJECT_ROOT / "runs/train"

    if args.resume:
        checkpoint = existing_file(args.resume, "Resume checkpoint")
        YOLO(str(checkpoint), task="detect").train(resume=True)
        return

    model_path = existing_file(args.model, "Pretrained model")
    model = YOLO(str(model_path), task="detect")
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        exist_ok=False,
        plots=True,
        seed=args.seed,
        deterministic=True,
    )
    save_dir = Path(result.save_dir).resolve()
    print(f"Training artifacts: {save_dir}")
    print(f"Best checkpoint: {save_dir / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
