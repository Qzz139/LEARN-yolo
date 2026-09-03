#!/usr/bin/env python3
"""Build a native one-folder desktop package on the current operating system."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_catalog import default_manifest_path, iter_packaged_onnx


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
BUILD_ROOT = PROJECT_ROOT / ".build" / "local_preview"
RELEASE_ROOT = PROJECT_ROOT / "release"
APPLICATION_NAME = "yolo-preview"


def _platform_slug() -> str:
    names = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}
    system = platform.system()
    if system not in names:
        raise RuntimeError(f"Unsupported packaging platform: {system}")
    machine = platform.machine().lower()
    architecture = "arm64" if machine in {"arm64", "aarch64"} else "x86_64"
    return f"{names[system]}-{architecture}"


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.stat().st_size < 1024 and path.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )
    except OSError:
        return False


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _build_bundle() -> Path:
    dist_root = BUILD_ROOT / "dist"
    work_root = BUILD_ROOT / "work"
    spec_root = BUILD_ROOT / "spec"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--name",
        APPLICATION_NAME,
        "--paths",
        str(APP_DIR),
        "--distpath",
        str(dist_root),
        "--workpath",
        str(work_root),
        "--specpath",
        str(spec_root),
        str(APP_DIR / "launcher.py"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    bundle = dist_root / APPLICATION_NAME
    if not bundle.is_dir():
        raise RuntimeError(f"PyInstaller output was not created: {bundle}")
    return bundle


def _copy_runtime_files(bundle: Path, release_dir: Path) -> None:
    if release_dir.parent.resolve() != RELEASE_ROOT.resolve():
        raise RuntimeError(f"Refusing to replace unexpected path: {release_dir}")
    if release_dir.exists():
        shutil.rmtree(release_dir)
    shutil.copytree(bundle, release_dir)

    manifest = default_manifest_path()
    (release_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, release_dir / "weights" / "manifest.json")
    for model_path in iter_packaged_onnx(manifest):
        if _is_lfs_pointer(model_path):
            raise RuntimeError(
                f"Model is still a Git LFS pointer: {model_path}. Run git lfs pull."
            )
        relative_path = model_path.relative_to(manifest.parent)
        destination = release_dir / "weights" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, destination)

    shutil.copy2(APP_DIR / "README.package.md", release_dir / "README.md")
    build_info = {
        "application": APPLICATION_NAME,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "git_revision": _git_revision(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (release_dir / "BUILD-INFO.json").write_text(
        json.dumps(build_info, indent=2) + "\n", encoding="utf-8"
    )


def _smoke_test(release_dir: Path) -> None:
    executable = release_dir / (
        f"{APPLICATION_NAME}.exe"
        if platform.system() == "Windows"
        else APPLICATION_NAME
    )
    subprocess.run([str(executable), "--list-models"], check=True)
    sample_image = PROJECT_ROOT / "bus.jpg"
    if sample_image.is_file():
        subprocess.run(
            [str(executable), "--source", str(sample_image), "--no-show"],
            check=True,
        )


def _archive(release_dir: Path) -> Path:
    base_name = RELEASE_ROOT / release_dir.name
    if platform.system() == "Windows":
        archive_path = shutil.make_archive(
            str(base_name), "zip", root_dir=RELEASE_ROOT, base_dir=release_dir.name
        )
    else:
        archive_path = shutil.make_archive(
            str(base_name), "gztar", root_dir=RELEASE_ROOT, base_dir=release_dir.name
        )
    return Path(archive_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Build without launching the packaged executable.",
    )
    args = parser.parse_args()

    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    platform_slug = _platform_slug()
    release_dir = RELEASE_ROOT / f"{APPLICATION_NAME}-{platform_slug}"
    bundle = _build_bundle()
    _copy_runtime_files(bundle, release_dir)
    if not args.skip_smoke_test:
        _smoke_test(release_dir)
    archive_path = _archive(release_dir)
    print(f"Package directory: {release_dir}")
    print(f"Archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
