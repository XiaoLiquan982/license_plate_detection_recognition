# License Plate Detection Recognition

基于 YOLOv5 检测和车牌识别模型的车牌检测识别项目，支持图片/视频命令行推理，并提供 PyQt5 图形界面用于图片识别、结果展示和历史记录管理。

## 功能特性

- 车牌检测、车牌号码识别和车牌颜色识别
- 支持单张图片、图片目录和视频推理
- PyQt5 桌面 GUI，支持选择图片、查看识别结果、历史记录查询/导出/删除
- 内置示例图片和预训练权重
- 支持 PyInstaller 打包，配置文件为 `PlateGUI.spec`

## 项目结构

```text
.
├── detect_plate.py              # 命令行检测识别入口
├── gui_app.py                   # PyQt5 图形界面入口
├── plate_recognition/           # 车牌识别相关代码
├── models/                      # YOLO 模型结构
├── utils/                       # 检测、绘图、工具函数
├── weights/                     # 检测和识别模型权重
├── imgs/                        # 示例图片
├── fonts/                       # 车牌绘制字体
├── requirements.txt             # Python 依赖
└── PlateGUI.spec                # PyInstaller 打包配置
```

## 环境准备

建议使用 Python 3.8 或 3.9，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 GPU，请根据本机 CUDA 版本安装合适的 PyTorch。

## 模型权重

项目默认读取以下权重文件：

- `weights/plate_detect.pt`
- `weights/plate_rec_color.pth`

当前仓库已包含上述权重文件。如果需要下载 YOLOv5 官方权重，可运行：

```bash
bash weights/download_weights.sh
```

## 命令行使用

识别 `imgs/` 目录中的示例图片：

```bash
python detect_plate.py --image_path imgs --output result
```

指定模型、图片路径和输出目录：

```bash
python detect_plate.py ^
  --detect_model weights/plate_detect.pt ^
  --rec_model weights/plate_rec_color.pth ^
  --image_path imgs ^
  --output result ^
  --is_color True
```

视频识别：

```bash
python detect_plate.py --video path/to/video.mp4 --output result
```

## GUI 使用

启动桌面应用：

```bash
python gui_app.py
```

GUI 默认使用 `weights/plate_detect.pt` 和 `weights/plate_rec_color.pth`，识别历史会写入项目目录下的 `history.csv`。

## 打包

安装 PyInstaller 后可使用已有 spec 文件打包：

```bash
pyinstaller PlateGUI.spec
```

生成的 `build/` 和 `dist/` 为构建产物，不建议提交到 Git 仓库。

## 注意事项

- 首次运行前请确认模型权重文件存在。
- Windows 环境下如遇中文显示问题，请确认字体文件 `fonts/platech.ttf` 存在。
- 大文件和构建产物建议通过 Release 或其他文件分发方式管理，不直接提交 `dist/`。
