# Jetson环境基线

记录日期：2026-09-02。该文件是后续部署、复现和实验报告的环境依据。

## 硬件和系统

| 项目 | 当前值 |
| --- | --- |
| 设备 | NVIDIA Jetson Orin NX Developer Kit |
| 内存 | 16 GiB |
| 架构 | ARM64 / aarch64 |
| 根分区 | 116 GB；约78 GB可用 |
| Swap | 7.6 GB |
| Ubuntu | 20.04.6 LTS Focal |
| L4T | R35.6.5 |
| JetPack路线 | JetPack 5.1.6维护线 |

## GPU计算环境

| 组件 | 当前值 |
| --- | --- |
| CUDA Toolkit | 11.4.19；NVCC 11.4.315 |
| cuDNN | 8.6.0.166 |
| TensorRT | 8.5.2.2；Python绑定已安装 |
| PyTorch | 2.1.0a0+41361538.nv23.06 |
| Torchvision | 0.16.1 |
| PyTorch CUDA | 11.4；CUDA可用和矩阵计算已通过 |

Torchvision能够导入，但CUDA NMS尚未单独验证。加载模型前应先完成该测试。

## Python和ROS

| 项目 | 当前值 |
| --- | --- |
| 系统Python | 3.8.10，`/usr/bin/python3` |
| 项目环境 | `~/venvs/yolo_ros2`，使用`--system-site-packages` |
| pip | 24.3.1 |
| NumPy | 预期1.24.4，待复核 |
| 系统OpenCV | 4.2.0 |
| 虚拟环境OpenCV | 5.0.0 |
| Ultralytics | 8.4.135 |
| ROS 2 | Foxy |
| ROS 1 | Noetic仍安装，但不再默认加载 |

ROS 2依赖 `rclpy`、`sensor_msgs`、`std_msgs`、`vision_msgs`、`cv_bridge`和
`colcon`均已安装。

## 已识别的兼容性门槛

1. ROS 2 Foxy和`cv_bridge`来自Ubuntu系统环境，通常绑定系统Python、NumPy和
   OpenCV 4.2。虚拟环境中的pip OpenCV 5.0可能覆盖系统`cv2`并产生ABI冲突。
   ROS节点部署前必须执行图像转换测试；若冲突，ROS环境改用系统OpenCV。
2. TensorRT engine必须在该Jetson上、使用TensorRT 8.5.2.2生成，不从Windows或
   其他Jetson复制。
3. 桌面Linux x86-64程序包不适用于Jetson aarch64，也不替代ROS节点。
4. ROS 2 Foxy已经停止上游维护，因此保持现有系统依赖稳定，不进行无目的的大版本
   升级。

## 未完成事项

- 复核NumPy实际版本；
- 验证Torchvision CUDA NMS；
- 传入模型并完成单图GPU推理；
- 构建ROS 2包；
- 确认摄像头连接方式；当前没有`/dev/video*`；
- 生成TensorRT FP16 engine；
- 实测ROS检测话题、20目标正确率和持续FPS。

Docker 26.1.3和NVIDIA容器组件已经安装，但GPU容器尚未验证。当前优先采用原生
ROS环境，只有依赖无法稳定共存时才切换到容器方案。
