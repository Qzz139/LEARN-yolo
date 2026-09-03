# Jetson USB 摄像头一键启动

这组脚本将项目已有步骤串成一次启动：加载 ROS 2 Foxy、激活
`~/venvs/yolo_ros2`、检查模型和 USB 摄像头、构建 ROS 2 包，然后启动
YOLO 检测节点并发布：

- `/yolo/detections`
- `/yolo/fps`
- `/yolo/annotated_image`

默认模型是 `weights/yolo26s_baseline_v2/best.pt`，默认摄像头是
`/dev/video0`。所有可调项都集中在 `jetson.env`，不需要修改 Python 文件。

## 第一次使用

在 Jetson 项目根目录运行：

```bash
git pull --ff-only
git lfs pull
chmod +x deploy/jetson/*.sh
./deploy/jetson/start_detector.sh --check-only
```

检查通过后启动：

```bash
./deploy/jetson/start_detector.sh
```

按 `Ctrl+C` 停止。若从其他终端停止：

```bash
./deploy/jetson/stop_detector.sh
```

若 `/dev/video0` 无法读取而 `/dev/video1` 可以：

```bash
./deploy/jetson/start_detector.sh --camera-source 1
```

若要临时测试 YOLO26m：

```bash
./deploy/jetson/start_detector.sh --model-id yolo26m_baseline_v1
```

## 安装桌面快捷方式

在 Jetson 图形桌面中执行一次：

```bash
./deploy/jetson/install_desktop.sh
```

此后可以双击 `LEARN-YOLO` 启动。Ubuntu 首次双击若提示不受信任，右键
快捷方式并选择“允许启动”。要自动打开检测画面，把 `jetson.env` 中的
`OPEN_VIEWER=false` 改成 `OPEN_VIEWER=true`。

运行日志保存在 `outputs/jetson/`，该目录不会提交到 Git。
