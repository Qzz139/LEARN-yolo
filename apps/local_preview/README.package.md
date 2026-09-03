# YOLO desktop preview package

This package contains the desktop detector, model manifest, and exported ONNX
models. Python, PyTorch, Ultralytics, and ROS are not required.

Run the executable for your operating system from a terminal:

- Windows: `yolo-preview.exe`
- macOS/Linux: `./yolo-preview`

The default source is camera 0. Use `--help` for all options and
`--list-models` to inspect bundled candidates.

Examples:

```text
yolo-preview.exe --list-models
yolo-preview.exe --source 0
yolo-preview.exe --source path/to/image.jpg --save
```

On macOS or Linux, replace `yolo-preview.exe` with `./yolo-preview`.
Results are written to `YOLOPreview/outputs` under the current user's home
directory. The macOS package is unsigned; the user may need to approve it in
Privacy & Security and grant camera permission on first launch.

YOLO26m is listed as a planned candidate but is not bundled until its trained
ONNX artifact is added to `weights/manifest.json`.
