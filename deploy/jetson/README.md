# Jetson USB 摄像头一键启动

这组脚本将项目已有步骤串成一次启动：加载 ROS 2 Foxy、激活
`~/venvs/yolo_ros2`、检查模型和 USB 摄像头、构建 ROS 2 包，然后启动
YOLO 检测节点并发布：

- `/yolo/detections`
- `/yolo/fps`
- `/yolo/annotated_image`

默认模型是 `weights/yolo26s_baseline_v2/best.pt`。摄像头默认通过
`/dev/v4l/by-id/` 下的稳定设备标识定位，不依赖会在拔插后变化的
`/dev/video0` 编号。所有可调项都集中在 `jetson.env`，不需要修改 Python 文件。

检测运行时，终端提供三个快捷键：

- `P`：保存当前带检测框画面为 JPEG；
- `R`：开始或停止保存带检测框的 MP4；
- `Q`：退出程序。

照片和视频默认保存在 `outputs/captures/`。也可以在启动时加 `--record`，让
录像随检测自动开始。

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

## USB 拔插保护

启动脚本会优先查找配置的稳定 `by-id` 路径，找不到时再自动选择其他
`*-video-index0` 或 `/dev/video*`，并等待最多 30 秒。程序运行中若连续三帧
读取失败，会关闭旧句柄，此后每 2 秒尝试重新连接；摄像头重新插入后无需重启
程序。等待时间和重连频率可在 `jetson.env` 中调整。

只有在调试时才需要绕过自动选择：

```bash
./deploy/jetson/start_detector.sh --camera-source /dev/video1
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

此后可以双击 `LEARN-YOLO` 启动，桌面快捷方式默认同时打开检测画面。
Ubuntu 首次双击若提示不受信任，右键快捷方式并选择“允许启动”。直接在
终端运行 `start_detector.sh` 时仍默认不打开窗口；需要节点直接显示 OpenCV
窗口时添加 `--view`。窗口中按 `Q`、`Esc` 或点击关闭按钮都会结束检测。

运行日志保存在 `outputs/jetson/`，该目录不会提交到 Git。照片和录像也属于
运行产物，不会自动提交到 Git；需要留档时再有选择地复制到正式成果目录。

## ROS 2 服务控制

没有交互终端时，可从另一终端调用：

```bash
ros2 service call /yolo/save_snapshot std_srvs/srv/Trigger "{}"
ros2 service call /yolo/set_recording std_srvs/srv/SetBool "{data: true}"
ros2 service call /yolo/set_recording std_srvs/srv/SetBool "{data: false}"
```
