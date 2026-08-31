# 【LEARN】yolo

YOLO 学习项目。

## 本机查看模型效果

独立预览应用位于 [`apps/local_preview`](apps/local_preview)，使用训练后的 ONNX 模型
完成摄像头、拍照、图片、视频及批量检测，不依赖 ROS 或 PyTorch。

```bash
./apps/local_preview/run.command
```

完整操作、快捷键及独立环境安装方式见
[`apps/local_preview/README.md`](apps/local_preview/README.md)。

## ROS 2 检测程序

ROS 2 Python 包位于 [`ros2_ws/src/yolo_detector`](ros2_ws/src/yolo_detector)。它支持订阅 ROS 图像话题或直接读取摄像头，并输出标准检测消息、标注图像和实时 FPS。

Jetson、ROS 2 发行版、摄像头来源和最终模型路径暂不写死，部署时通过参数配置。详细说明见 [`ros2_ws/src/yolo_detector/README.md`](ros2_ws/src/yolo_detector/README.md)。
