# YOLO桌面预览程序

同一套 Python/ONNX 程序支持 Windows、macOS 和 Linux。源码模式需要
Python、NumPy 和 OpenCV；PyInstaller 原生包不要求目标电脑安装 Python。

## 源码启动

在项目根目录执行：

```text
# Windows cmd或PowerShell
apps\local_preview\run.cmd --source 0

# macOS
./apps/local_preview/run.command --source 0

# Linux
./apps/local_preview/run.sh --source 0
```

若找不到依赖，在 `apps/local_preview` 创建独立环境：

```text
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
```

## 模型选择

[`weights/manifest.json`](../../weights/manifest.json) 是唯一模型清单。目前：

| ID | 模型 | 状态 |
| --- | --- | --- |
| `yolo26n_baseline_v1` | YOLO26n | 已归档，可运行 |
| `yolo26s_baseline_v2` | YOLO26s | 当前默认候选，可运行 |
| `yolo26m_candidate` | YOLO26m | 待训练，尚无ONNX模型 |

```text
# 查看模型
apps\local_preview\run.cmd --list-models

# 指定已导出的候选
apps\local_preview\run.cmd --model-id yolo26n_baseline_v1 --source 0

# 指定任意ONNX文件；其优先级最高
apps\local_preview\run.cmd --model path\to\best.onnx --source 0
```

选择优先级为：

```text
--model
YOLO_PREVIEW_MODEL
--model-id
YOLO_PREVIEW_MODEL_ID
manifest.active_model
```

## 输入与输出

```text
# 图片
apps\local_preview\run.cmd --source path\to\image.jpg --save

# 视频
apps\local_preview\run.cmd --source path\to\video.mp4 --save

# 图片目录
apps\local_preview\run.cmd --source path\to\images --no-show

# 摄像头前100帧，无窗口
apps\local_preview\run.cmd --source 0 --max-frames 100 --no-show
```

摄像头窗口快捷键：

- `C`、回车、`S`或鼠标左键：保存原图和标注图；
- 空格：暂停或继续；
- `+`/`-`：调节置信度；
- `Q`/`Esc`：退出。

源码模式输出到项目 `outputs/local_preview/`。打包程序输出到当前用户目录下的
`YOLOPreview/outputs/`。

## 构建原生包

PyInstaller不是跨平台编译器，每个操作系统必须在本系统构建。项目提供统一命令：

```text
python -m pip install -r apps/local_preview/requirements-build.txt
python apps/local_preview/package.py
```

构建程序会：

1. 生成one-folder原生程序；
2. 附带模型清单和当前已有的ONNX模型；
3. 执行模型清单及单图推理冒烟测试；
4. 在 `release/` 生成ZIP或TAR.GZ归档。

GitHub Actions工作流 `.github/workflows/build-desktop.yml` 会分别构建：

- Windows x86-64；
- macOS ARM64；
- Linux x86-64。

Jetson ARM64不使用这里的桌面Linux包；它使用ROS 2源码和在Jetson本机生成的
TensorRT engine。
