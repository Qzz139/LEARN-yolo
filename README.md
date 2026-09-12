**English** | [简体中文](README_cn.md)

# 【LEARN】YOLO

> **Code-only branch:** this branch intentionally omits `datasets/` so the
> source and deployment files can be browsed without dataset contents. Use
> `main` for the complete project, or provide a compatible dataset locally at
> the paths documented below before auditing or retraining.

A summer-term object detection course project using a Jetson Orin NX and a
camera to detect tabletop objects in real time, display bounding boxes, class
labels, and confidence scores, and publish results through ROS 2. The refreshed
dataset and default model cover 11 classes: `book`, `bottle`, `earphone`, `glass`,
`headphone`, `keyboard`, `laptop`, `mobile`, `mouse`, `pen`, and `penstand`.

## Acceptance Criteria

- Detect at least two classes of tabletop objects simultaneously.
- Test 20 objects and achieve a correct recognition rate of at least 80%.
- Achieve a detection speed of at least 5 FPS on the Jetson.
- Save test results, representative failure cases, and result videos.
- Submit the dataset, models, code, running instructions, and lab report.

## Current Model Candidates

The model manifest is stored in [`weights/manifest.json`](weights/manifest.json):

- YOLO26n baseline v1: archived.
- YOLO26s baseline v2: a legacy three-class candidate available as a fallback.
- YOLO26m baseline v1: a legacy three-class model for performance comparison.
- YOLO26m tabletop v1: the current default candidate for the refreshed 11-class dataset.

The new model completed 100 training epochs and achieved mAP50 `0.845` and
mAP50-95 `0.647` on a separate test set. These metrics cannot be directly compared
with those of the older models because their class sets and datasets differ.
Recognition accuracy and FPS still need to be validated on at least 20 physical
objects after connecting a camera to the Jetson.

The application no longer guesses which model to use from `runs/` or file
timestamps. An explicit `--model` path takes priority; otherwise, it uses
`--model-id` or the manifest's `active_model`. Each model can have its own class
list in the manifest, so both the legacy three-class models and the new 11-class
model display the correct labels.

## Desktop Application

The desktop application uses ONNX and OpenCV DNN and does not require ROS,
PyTorch, or Ultralytics.

```text
Windows: apps\local_preview\run.cmd --source 0
macOS:   ./apps/local_preview/run.command --source 0
Linux:   ./apps/local_preview/run.sh --source 0
```

List model candidates:

```text
apps\local_preview\run.cmd --list-models
```

Run with a specific model:

```text
apps\local_preview\run.cmd --model-id yolo26m_tabletop_v1 --source 0
```

See [`apps/local_preview/README.md`](apps/local_preview/README.md) for full instructions.

## ROS 2 and Jetson

The ROS 2 package is located at
[`ros2_ws/src/yolo_detector`](ros2_ws/src/yolo_detector) and publishes:

- `/yolo/detections`
- `/yolo/annotated_image`
- `/yolo/fps`

See [`docs/jetson-environment.md`](docs/jetson-environment.md) for the current
Jetson environment and compatibility notes, and
[`docs/jetson-sync.md`](docs/jetson-sync.md) for the recommended Git sync workflow.

After connecting a USB camera to the Jetson, launch the detector from the project root:

```bash
./deploy/jetson/start_detector.sh
```

See [`deploy/jetson/README.md`](deploy/jetson/README.md) for first-run instructions
and desktop shortcut installation. Runtime settings are centralized in
`deploy/jetson/jetson.env`, so switching models or cameras does not require
editing Python files.

## Data and Version Control

- Current dataset configuration: [`datasets/tabletop_v1/data.yaml`](datasets/tabletop_v1/data.yaml)
- Current dataset documentation: [`datasets/tabletop_v1/README.md`](datasets/tabletop_v1/README.md)
- Audit the dataset: `python tools/audit_yolo_dataset.py datasets/tabletop_v1/data.yaml`
- Reproduce training: `python tools/train_yolo26m.py`
- Model and test evidence: [`weights/yolo26m_tabletop_v1/`](weights/yolo26m_tabletop_v1)
- Older datasets, models, and training records are retained for fallback and comparison.
- Large data and model files are managed with Git LFS.
- `runs/`, `outputs/`, build directories, and regenerable caches are not committed.

The current dataset does not include `monitor`, so the new model does not detect
monitors. If this class is required for the final course submission, add monitor
images and labels and retrain with a 12-class configuration.

## Original Course Requirements

<img width="430" height="462" alt="Course object detection lab requirements" src="https://github.com/user-attachments/assets/4b3c4d1c-1d89-4596-a287-e0139dbf3c88" />
