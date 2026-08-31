# YOLO 本机预览应用

独立运行训练后的 `best.onnx`，使用 OpenCV DNN 完成推理，不依赖 ROS、PyTorch、
Ultralytics 或 onnxruntime。

## 目录结构

```text
apps/local_preview/
├── main.py          # 图片、视频、摄像头与拍照逻辑
├── run.command      # macOS 双击/终端启动器
├── requirements.txt # 最小本机依赖
└── README.md        # 本说明
```

模型仍统一保存在项目级目录：
[`weights/yolo26n_baseline_v1/best.onnx`](../../weights/yolo26n_baseline_v1/best.onnx)。
运行结果统一写入项目级 `outputs/local_preview/`，不会混入应用源码目录。

## 启动摄像头

macOS 可以直接双击 `run.command`，也可以在项目根目录执行：

```bash
./apps/local_preview/run.command
```

首次启动时，需要允许 Terminal 或 Codex 使用摄像头。

摄像头窗口快捷键：

- `C`、回车、`S` 或鼠标左键：拍照；
- 空格：暂停/继续；
- `+`/`-`：调整置信度阈值；
- `Q`/`Esc`：退出。

每次拍照会同时保存无检测框原图和带检测框图片：

```text
outputs/local_preview/photos/photo_时间_original.jpg
outputs/local_preview/photos/photo_时间_detected.jpg
```

## 图片、视频与批量检测

以下命令从项目根目录执行：

```bash
# 检测一张图片并保存
./apps/local_preview/run.command \
  --source datasets/dataset/images/test/mouse_keyboard_001.jpg \
  --save

# 检测视频并保存标注视频
./apps/local_preview/run.command --source /absolute/path/input.mp4 --save

# 批量处理目录中的真实图片
./apps/local_preview/run.command \
  --source datasets/dataset/images/test \
  --no-show

# 无窗口验证
./apps/local_preview/run.command \
  --source datasets/dataset/images/test/keyboard_001.jpg \
  --no-show --save
```

常用参数可通过下面的命令查看：

```bash
./apps/local_preview/run.command --help
```

## 独立环境

当前电脑的启动器会自动找到已有的 OpenCV Python。如果换到其他电脑，可在应用目录
创建隔离环境：

```bash
cd apps/local_preview
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run.command
```

不要在 macOS 上安装项目根目录的 `requirements-lock.txt`；它记录的是 CUDA 训练环境。
