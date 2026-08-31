# 【LEARN】yolo

YOLO 学习项目。

## ROS 2 检测程序

ROS 2 Python 包位于 [`ros2_ws/src/yolo_detector`](ros2_ws/src/yolo_detector)。它支持订阅 ROS 图像话题或直接读取摄像头，并输出标准检测消息、标注图像和实时 FPS。

Jetson、ROS 2 发行版、摄像头来源和最终模型路径暂不写死，部署时通过参数配置。详细说明见 [`ros2_ws/src/yolo_detector/README.md`](ros2_ws/src/yolo_detector/README.md)。
