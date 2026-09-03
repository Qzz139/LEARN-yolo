# 【LEARN】YOLO

夏季学期目标检测课程项目：使用 Jetson Orin NX 和摄像头实时识别
`keyboard`、`monitor`、`mouse`，显示检测框、类别和置信度，并通过 ROS 2
发布结果。

## 验收目标

- 同时识别不少于两类桌面物体；
- 测试 20 个物体，正确识别率不低于 80%；
- Jetson 端检测速度不低于 5 FPS；
- 保存测试结果、典型错误案例和结果视频；
- 提交数据集、模型、程序、运行说明和实验报告。

## 当前模型候选

模型清单保存在 [`weights/manifest.json`](weights/manifest.json)：

- YOLO26n baseline v1：已归档；
- YOLO26s baseline v2：当前默认候选；
- YOLO26m baseline v1：已训练并导出，作为现场性能对比候选；

YOLO26m在当前验证集上的指标低于YOLO26s，因此默认模型暂不切换。Jetson
接入真实摄像头后，将在相同输入和参数下比较两者的正确率与FPS。

程序不会再通过 `runs/` 或文件时间猜测模型。显式 `--model` 路径优先，
否则使用 `--model-id` 或清单中的 `active_model`。

## 桌面程序

桌面程序使用 ONNX 和 OpenCV DNN，不依赖 ROS、PyTorch 或 Ultralytics。

```text
Windows: apps\local_preview\run.cmd --source 0
macOS:   ./apps/local_preview/run.command --source 0
Linux:   ./apps/local_preview/run.sh --source 0
```

查看模型候选：

```text
apps\local_preview\run.cmd --list-models
```

使用指定模型：

```text
apps\local_preview\run.cmd --model-id yolo26s_baseline_v2 --source 0
```

完整说明见 [`apps/local_preview/README.md`](apps/local_preview/README.md)。

## ROS 2与Jetson

ROS 2包位于 [`ros2_ws/src/yolo_detector`](ros2_ws/src/yolo_detector)，输出：

- `/yolo/detections`
- `/yolo/annotated_image`
- `/yolo/fps`

Jetson当前环境和兼容性注意事项见
[`docs/jetson-environment.md`](docs/jetson-environment.md)，推荐的Git同步方式见
[`docs/jetson-sync.md`](docs/jetson-sync.md)。

Jetson 接入 USB 摄像头后，可在项目根目录一键启动：

```bash
./deploy/jetson/start_detector.sh
```

第一次运行和桌面快捷方式安装说明见
[`deploy/jetson/README.md`](deploy/jetson/README.md)。运行参数集中在
`deploy/jetson/jetson.env`，切换模型或摄像头时不需要修改 Python 文件。

## 数据与版本控制

- 数据配置：[`datasets/dataset/data.yaml`](datasets/dataset/data.yaml)
- 数据说明：[`datasets/dataset/README.md`](datasets/dataset/README.md)
- 训练记录：[`record/`](record)
- 正式模型：[`weights/`](weights)
- 大型数据和模型文件由 Git LFS 管理；
- `runs/`、`outputs/`、构建目录和可重建缓存不提交。

## 课程原始要求

<img width="430" height="462" alt="课程目标检测实验要求" src="https://github.com/user-attachments/assets/4b3c4d1c-1d89-4596-a287-e0139dbf3c88" />
