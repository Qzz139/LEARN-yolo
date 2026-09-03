# YOLO26m baseline v1

This directory contains the preserved deployment artifacts and compact training
evidence for the first custom YOLO26m candidate.

## Training

- Dataset: `datasets/dataset/data.yaml`
- Classes: `keyboard`, `monitor`, `mouse`
- Input size: 640
- Requested epochs: 100
- Completed epochs: 87 (`0` through `86` in `results.csv`)
- Best recorded epoch: 66
- Best recorded mAP50: 0.89481
- Best recorded mAP50-95: 0.63493

The complete arguments, epoch metrics, and training graph are preserved as
`args.yaml`, `results.csv`, and `results.png`.

## Artifacts

- `best.pt`: custom PyTorch checkpoint for Jetson benchmarking
- `best.onnx`: opset 17, static 640x640 input, simplified ONNX for the
  cross-platform desktop application

Fresh validation of `best.pt` on 18 validation images (28 instances) produced
precision 0.911, recall 0.882, mAP50 0.877, and mAP50-95 0.616. The ONNX export
was also loaded and validated successfully.

YOLO26s baseline v2 remains the default because it currently has stronger
validation metrics. Promote this model only after the two models are compared
with the same live Jetson camera stream and both accuracy and FPS are recorded.
