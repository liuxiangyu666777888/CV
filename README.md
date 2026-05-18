# 计算机视觉作业

复旦大学计算机视觉课程作业，涵盖图像分类、目标检测与跟踪、语义分割三个任务。

## 目录结构

```
├── HW2_report.md              # 完整实验报告
├── assignment1/                # 任务一：宠物品种分类
│   ├── configs/                # 实验配置文件
│   ├── src/                    # 训练、评估、模型代码
│   ├── outputs/                # 实验结果（CSV、混淆矩阵、曲线图）
│   └── fig/                    # 报告插图
├── assignment2/                # 任务二：VisDrone 目标检测与跟踪
│   ├── scripts/                # 检测训练、跟踪、分析脚本
│   ├── configs/                # 类别配置文件
│   ├── outputs/                # 跟踪结果、遮挡分析、越线计数
│   └── runs/                   # YOLO 训练日志
├── assignment3/                # 任务三：U-Net 语义分割
│   ├── models/                 # U-Net 架构实现
│   ├── losses/                 # Dice Loss / Combined Loss
│   ├── outputs/                # 模型权重与评估可视化
│   └── fig/                    # 报告插图
└── .gitignore
```

## 任务一：基于 ImageNet 预训练模型的宠物品种识别

在 Oxford-IIIT Pet Dataset（37 类）上微调 ResNet-18，对比预训练/随机初始化、超参数、注意力机制的影响。

- **最佳结果**：ResNet-18 + ImageNet 预训练，Accuracy = 87.43%
- **关键技术**：迁移学习、SE / CBAM 注意力、超参数消融

```bash
cd assignment1
pip install -r requirements.txt
python src/train.py --config configs/baseline_resnet18_pretrained.yaml
```

详见 [assignment1/README.md](assignment1/README.md)

## 任务二：VisDrone 目标检测与视频多目标跟踪

在 VisDrone2019-DET 上训练 YOLO11n 检测器，结合 ByteTrack 完成视频多目标跟踪、遮挡分析和越线计数。

- **检测结果**：mAP50 = 0.2734（600 像素分辨率，未用 Mosaic 增强）
- **跟踪结果**：478 帧视频输出 194 个 Track ID，验证越线计数流水线可行性
- **关键技术**：YOLO11n、ByteTrack、轨迹碎片化分析

```bash
cd assignment2
pip install -r requirements.txt
python scripts/train_detector.py --data datasets/visdrone_yolo/data.yaml --model yolo11n.pt --epochs 30
python scripts/run_tracking.py --source <video> --output-dir outputs/tracking/
```

详见 [assignment2/README.md](assignment2/README.md)

## 任务三：从零实现 U-Net 并比较不同损失函数

从零搭建 U-Net（31M 参数），在 Oxford-IIIT Pet trimap 上完成三分类语义分割（前景/背景/边界），对比 Cross-Entropy、Dice Loss、Combined Loss。

- **最佳结果**：U-Net + Dice Loss，val mIoU = 0.7654
- **核心发现**：类别不平衡场景下 Dice Loss 显著优于 Cross-Entropy（+12.6 pp）
- **关键技术**：Skip Connection、Dice Loss、像素级多分类评估

```bash
cd assignment3
python train.py --loss dice
python evaluate.py --model outputs/unet_dice/best.pt --output-dir outputs/evaluation
```

详见 [assignment3/README.md](assignment3/README.md)

## 环境要求

- Python 3.10+
- PyTorch 2.x
- torchvision
- ultralytics (任务二)
- 详见各子目录下的 `requirements.txt`

## 模型权重

训练好的模型权重文件较大，未包含在仓库中。
