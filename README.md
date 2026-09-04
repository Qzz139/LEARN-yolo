# 【LEARN】YOLO

夏季学期目标检测课程项目：使用 Jetson Orin NX 和摄像头实时识别
桌面物体，显示检测框、类别和置信度，并通过 ROS 2 发布结果。当前刷新后的
数据集和默认模型覆盖 `book`、`bottle`、`earphone`、`glass`、`headphone`、
`keyboard`、`laptop`、`mobile`、`mouse`、`pen`、`penstand` 共 11 类。

## 验收目标

- 同时识别不少于两类桌面物体；
- 测试 20 个物体，正确识别率不低于 80%；
- Jetson 端检测速度不低于 5 FPS；
- 保存测试结果、典型错误案例和结果视频；
- 提交数据集、模型、程序、运行说明和实验报告。

## 当前模型候选

模型清单保存在 [`weights/manifest.json`](weights/manifest.json)：

- YOLO26n baseline v1：已归档；
- YOLO26s baseline v2：旧三分类候选，可回退运行；
- YOLO26m baseline v1：旧三分类性能对比模型；
- YOLO26m tabletop v1：刷新后 11 分类数据集的当前默认候选。

新模型完整训练 100 轮，在独立测试集上取得 mAP50 `0.845`、mAP50-95
`0.647`。由于新旧模型的类别集合和数据集不同，指标不能直接横向比较；
Jetson 接入真实摄像头后仍需完成不少于 20 个实物的正确率和 FPS 验收。

程序不会再通过 `runs/` 或文件时间猜测模型。显式 `--model` 路径优先，
否则使用 `--model-id` 或清单中的 `active_model`。每个模型可在清单中携带
自己的类别列表，因此旧三分类模型和新 11 分类模型都能正确显示标签。

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
apps\local_preview\run.cmd --model-id yolo26m_tabletop_v1 --source 0
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

- 当前数据配置：[`datasets/tabletop_v1/data.yaml`](datasets/tabletop_v1/data.yaml)
- 当前数据说明：[`datasets/tabletop_v1/README.md`](datasets/tabletop_v1/README.md)
- 数据审计：`python tools/audit_yolo_dataset.py datasets/tabletop_v1/data.yaml`
- 复现训练：`python tools/train_yolo26m.py`
- 模型与测试证据：[`weights/yolo26m_tabletop_v1/`](weights/yolo26m_tabletop_v1)
- 旧数据集、旧模型和训练记录保留用于回退与对照；
- 大型数据和模型文件由 Git LFS 管理；
- `runs/`、`outputs/`、构建目录和可重建缓存不提交。



## 课程原始要求

<img width="430" height="462" alt="课程目标检测实验要求" src="https://github.com/user-attachments/assets/4b3c4d1c-1d89-4596-a287-e0139dbf3c88" />
