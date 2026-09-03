# yolo_detector ROS 2 包

这个包先把完整程序接口跑通，暂不绑定某一台 Jetson、某个 ROS 2 发行版、摄像头驱动或模型格式。相同的节点可以加载 Ultralytics 支持的 `.pt`、`.onnx` 或 `.engine` 模型。

## 数据流

输入有三种模式：

- `topic`（默认）：订阅 `/camera/image_raw`，类型为 `sensor_msgs/msg/Image`。
- `camera`：通过 OpenCV 直接打开摄像头编号、视频地址或 GStreamer 管道。
- `image`：按固定频率重复读取一张图片，用于没有摄像头时验证完整 ROS 2 输出链路。

节点输出：

- `/yolo/detections`：`vision_msgs/msg/Detection2DArray`。每个目标包含类别名称、置信度和像素坐标边界框。
- `/yolo/annotated_image`：`sensor_msgs/msg/Image`。带检测框的图像，可通过参数关闭以减少开销。
- `/yolo/fps`：`std_msgs/msg/Float32`。按实际输出帧间隔计算的平滑 FPS。

类别顺序当前固定为训练数据的 `keyboard`、`monitor`、`mouse`，可在 [`config/detector.yaml`](config/detector.yaml) 中修改。

## 尚待部署时填写的配置

打开 [`config/detector.yaml`](config/detector.yaml)，至少填写：

```yaml
model_path: "/模型的绝对路径/best.pt"
```

以下项目等拿到 Jetson 环境信息后再确定：

- ROS 2 发行版及对应的环境加载命令；
- 最终使用 `.pt`、`.onnx` 还是在目标 Jetson 上生成的 `.engine`；
- USB 摄像头、CSI 摄像头或现有 ROS 相机节点；
- `device`、摄像头管道和性能参数。

TensorRT `.engine` 与构建它的 Jetson/TensorRT 环境相关，应在最终目标设备上生成，不要直接复制 PC 上生成的 engine。

## 构建

以下命令需要在已经安装 ROS 2 的 Ubuntu/Jetson 环境执行，`<ros_distro>` 暂时保留为占位符：

```bash
source /opt/ros/<ros_distro>/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select yolo_detector
source install/setup.bash
```

Python 运行环境还需要 `ultralytics`。Jetson 上的 PyTorch、CUDA 和 TensorRT 必须与 JetPack 匹配，因此部署前再选择安装方式，不应在现阶段盲目覆盖 Jetson 自带或 NVIDIA 提供的版本。

## 运行

当前 Jetson 使用 USB 摄像头时，推荐直接使用仓库的一键启动脚本；脚本会加载
Foxy 和项目虚拟环境、检查 `/dev/video0`、构建包并处理 JetPack 5 所需的
`LD_PRELOAD`：

```bash
./deploy/jetson/start_detector.sh
```

下面保留原始 ROS 2 命令，适合调试节点或使用其他输入模式。

订阅已有 ROS 相机话题：

```bash
ros2 launch yolo_detector yolo_detector.launch.py \
  model_path:=/absolute/path/to/best.pt \
  source_mode:=topic \
  image_topic:=/camera/image_raw
```

直接打开编号为 0 的摄像头：

```bash
ros2 launch yolo_detector yolo_detector.launch.py \
  model_path:=/absolute/path/to/best.pt \
  source_mode:=camera \
  camera_source:=0
```

没有摄像头时，可先用仓库中的图片做端到端验证：

```bash
ros2 launch yolo_detector yolo_detector.launch.py \
  model_path:=/absolute/path/to/best.pt \
  source_mode:=image \
  image_source:=/absolute/path/to/image.jpg
```

ROS 2 Foxy 的 `cv_bridge` 按 OpenCV 4 的图像类型编号编译。节点已对
OpenCV 5 做兼容处理，无须为了图像消息转换而替换 Jetson 当前的 OpenCV。

在 ARM64 Jetson 上，启动文件会在文件存在时自动预加载
`/usr/lib/aarch64-linux-gnu/libgomp.so.1`，避免 PyTorch 在 ROS 依赖加载后出现
`cannot allocate memory in static TLS block`。其他平台不会自动设置该变量，也可通过
`libgomp_path` 启动参数显式覆盖或禁用。

也可以直接编辑 `config/detector.yaml`，然后只执行：

```bash
ros2 launch yolo_detector yolo_detector.launch.py
```

## 验证输出

```bash
timeout -s INT 5s ros2 topic echo /yolo/detections
ros2 topic hz /yolo/detections
ros2 topic echo /yolo/fps
```

若需要提高最终 FPS，可先将 `publish_annotated_image` 设为 `false`，再比较 `/yolo/fps`；这避免把绘制检测框的开销混入最低性能版本。
