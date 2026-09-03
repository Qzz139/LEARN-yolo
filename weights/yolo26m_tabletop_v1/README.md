# YOLO26m tabletop v1

This directory contains the deployment artifacts and compact evidence for the
11-class model trained on the refreshed tabletop dataset.

## Training

- Dataset: `datasets/tabletop_v1/data.yaml`
- Images: 1,142 train, 148 validation, 151 test
- Classes: `book`, `bottle`, `earphone`, `glass`, `headphone`, `keyboard`,
  `laptop`, `mobile`, `mouse`, `pen`, `penstand`
- Base checkpoint: `yolo26m.pt`
- Input size: 640
- Batch size: 16
- Requested/completed epochs: 100/100
- Best recorded epoch: 80 (zero-based CSV epoch 79)
- Seed: 42, deterministic training enabled
- Ultralytics: 8.4.135

The best checkpoint produced the following validation metrics:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.802 | 0.761 | 0.806 | 0.633 |

## Independent test split

The preserved `best.pt` was evaluated on 151 test images containing 259
instances:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.841 | 0.813 | 0.845 | 0.647 |
| book | 0.593 | 0.657 | 0.663 | 0.509 |
| bottle | 0.949 | 0.949 | 0.973 | 0.819 |
| earphone | 0.781 | 0.715 | 0.825 | 0.425 |
| glass | 0.903 | 0.818 | 0.810 | 0.656 |
| headphone | 0.902 | 0.857 | 0.893 | 0.720 |
| keyboard | 0.949 | 0.925 | 0.943 | 0.782 |
| laptop | 0.924 | 0.875 | 0.912 | 0.761 |
| mobile | 0.708 | 0.714 | 0.788 | 0.563 |
| mouse | 0.959 | 0.870 | 0.902 | 0.799 |
| pen | 0.642 | 0.615 | 0.606 | 0.295 |
| penstand | 0.945 | 0.946 | 0.981 | 0.784 |

`earphone` and `pen` have the weakest localization metrics and are the first
classes to target when collecting additional hard examples.

## ONNX desktop validation

The exported ONNX model completed the full 151-image test split through
Ultralytics ONNX Runtime on CPU: precision `0.841`, recall `0.796`, mAP50
`0.832`, and mAP50-95 `0.624`, averaging about `100.5 ms/image`. A separate
OpenCV DNN batch smoke test processed all 151 images successfully. The small
metric reduction from PT (`0.647` to `0.624` mAP50-95) should be retained in
deployment reporting rather than mixed with the PT result.

## Artifacts

- `best.pt`: PyTorch checkpoint for Ultralytics and Jetson deployment
- `best.onnx`: simplified opset 18 ONNX, static 640 x 640 input, output shape
  `(1, 300, 6)` for the OpenCV DNN desktop application
- `args.yaml`: complete training arguments
- `results.csv` and `results.png`: per-epoch metrics and training curves
- `confusion_matrix*.png`: validation confusion matrices
- `test_confusion_matrix*.png`: independent test confusion matrices

This model is not directly comparable to the legacy three-class models because
the refreshed dataset changes the task to 11 classes and removes `monitor`.
