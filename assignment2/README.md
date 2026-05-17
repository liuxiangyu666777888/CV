# Assignment 2

任务二：场景目标检测与视频多目标跟踪。

本目录覆盖 4 个部分：

1. 使用 VisDrone 数据集训练检测模型
2. 将连续图像序列导出为测试视频
3. 对视频执行检测 + 跟踪并导出结果
4. 进行遮挡 / ID 跳变分析与越线计数

## 目录

```text
assignment2/
├─ configs/
├─ datasets/
├─ outputs/
├─ scripts/
├─ README.md
└─ requirements.txt
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 1. 转换 VisDrone 为 YOLO 格式

```bash
python scripts/prepare_visdrone_yolo.py --source ../data/archive --target datasets/visdrone_yolo
```

会生成：

- `datasets/visdrone_yolo/images/{train,val,test}`
- `datasets/visdrone_yolo/labels/{train,val,test}`
- `datasets/visdrone_yolo/data.yaml`

## 2. 训练检测模型

```bash
python scripts/train_detector.py --data datasets/visdrone_yolo/data.yaml --model yolo11n.pt --epochs 30 --imgsz 960
```

如果你想直接使用已有权重：

```bash
python scripts/train_detector.py --data datasets/visdrone_yolo/data.yaml --model ../data/archive/yolov9_finetuned.pt --resume-only
```

## 3. 从图像序列生成测试视频

```bash
python scripts/build_test_video.py --images-root ../data/archive/VisDrone2019-DET-test-dev/images --sequence 9999979_00000 --fps 15 --output outputs/videos/seq_9999979.mp4
```

## 4. 视频跟踪与越线计数

```bash
python scripts/run_tracking.py --model ../data/archive/yolov9_finetuned.pt --source outputs/videos/seq_9999979.mp4 --line 640 0 640 720 --output-dir outputs/tracking/seq_9999979
```

输出包括：

- 标注视频
- `tracks.csv`
- `summary.json`
- 越线计数结果

## 5. 遮挡 / ID 跳变分析

```bash
python scripts/analyze_occlusion.py --video outputs/videos/seq_9999979.mp4 --tracks outputs/tracking/seq_9999979/tracks.csv --start-frame 10 --num-frames 4 --output-dir outputs/analysis/seq_9999979
```

会生成：

- 4 帧可视化图
- 拼图 `contact_sheet.jpg`
- `analysis.md`

## VisDrone 类别映射

当前实现保留 10 个检测类别，忽略 `ignored regions(0)` 和 `others(11)`：

1. pedestrian
2. people
3. bicycle
4. car
5. van
6. truck
7. tricycle
8. awning-tricycle
9. bus
10. motor
