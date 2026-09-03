#!/usr/bin/env python3
"""Cross-platform entry point with manifest-based model selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from model_catalog import ModelCatalogError, catalog_lines, resolve_model


def _launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--list-models", action="store_true")
    return parser


def _print_help() -> None:
    import main as preview

    parser = preview.build_parser()
    for action in parser._actions:
        if action.dest == "model":
            action.default = None
            action.help = "Explicit ONNX path (overrides the model catalog)."
        elif action.dest == "labels":
            action.default = None
            action.help = "Override comma-separated class names in model order."
    print(parser.format_help().rstrip())
    print("\nModel catalog options:")
    print("  --model-id ID       select an entry from weights/manifest.json")
    print("  --manifest PATH     use a different model manifest")
    print("  --list-models       list candidates and exit")
    print("\nSelection order:")
    print("  --model > YOLO_PREVIEW_MODEL > --model-id >")
    print("  YOLO_PREVIEW_MODEL_ID > manifest active_model")


def main(argv: Optional[Sequence[str]] = None) -> int:
    original_args = list(argv if argv is not None else sys.argv[1:])
    if "--help" in original_args or "-h" in original_args:
        _print_help()
        return 0

    launcher_args, preview_args = _launcher_parser().parse_known_args(original_args)
    if launcher_args.list_models:
        print("Registered models (* = active):")
        for line in catalog_lines(launcher_args.manifest):
            print(line)
        return 0

    selection = resolve_model(
        manifest_path=launcher_args.manifest,
        explicit_model=launcher_args.model,
        explicit_model_id=launcher_args.model_id,
    )
    labels = launcher_args.labels or ",".join(selection.labels)
    forwarded_args = [
        *preview_args,
        "--model",
        str(selection.model_path),
        "--labels",
        labels,
    ]
    if getattr(sys, "frozen", False) and "--output-dir" not in preview_args:
        forwarded_args.extend(
            ["--output-dir", str(Path.home() / "YOLOPreview" / "outputs")]
        )

    print(f"Model ID: {selection.model_id} ({selection.status})")
    import main as preview

    try:
        return preview.main(forwarded_args)
    except preview.PreviewError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModelCatalogError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
