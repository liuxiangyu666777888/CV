# Assignment 3: U-Net Semantic Segmentation

从零搭建 U-Net，在 Oxford-IIIT Pet Dataset 上进行三分类语义分割（前景/背景/边界），对比三种损失函数的 mIoU 表现。

## 目录

```
assignment3/
├── config.py              # 路径、超参数配置
├── data/
│   └── dataset.py         # OxfordPetDataset + 数据增强
├── models/
│   └── unet.py            # U-Net 架构（含 skip connection）
├── losses/
│   └── dice_loss.py       # Dice Loss + Combined Loss
├── train.py               # 训练入口
├── evaluate.py            # 测试评估
└── README.md
```

## 数据集

Oxford-IIIT Pet Dataset：7390 张宠物图片，每张对应一个 trimap 标注（1=前景, 2=背景, 3=边界）。

- 图片目录：`d:/计算机视觉/data/images/images/`
- trimap 目录：`d:/计算机视觉/data/annotations/annotations/trimaps/`
- 训练/验证划分：`trainval.txt`（3680 张，按 80/20 拆分）
- 测试集：`test.txt`（3669 张）

## 环境配置

```bash
pip install torch torchvision pillow numpy swanlab tqdm
```

## 训练

三种损失函数分别训练：

```bash
# Cross-Entropy Loss
python train.py --loss ce

# Dice Loss
python train.py --loss dice

# Combined Loss (CE + Dice)
python train.py --loss combined
```

训练配置（在 config.py 中修改）：
- 输入尺寸：256×256
- Batch size：16
- 学习率：1e-3（CosineAnnealing）
- Epochs：50
- 优化器：AdamW

训练输出保存在 `outputs/unet_{ce,dice,combined}/` 下，包括 `best.pt`、`last.pt` 和 SwanLab 日志。

## 评估

```bash
python evaluate.py --model outputs/unet_combined/best.pt --output-dir outputs/evaluation
```

输出：per-class IoU、mIoU 和可视化对比图。

## SwanLab 可视化

训练过程中自动记录 loss 和 mIoU 曲线，访问 https://swanlab.cn 查看。

## 模型结构

- U-Net (base_filters=64)：4 层编码器 + 4 层解码器 + skip connections
- 输入：3×256×256 RGB
- 输出：3×256×256（3 类 logits）
- 参数量：~31M

## 损失函数

| 损失函数 | 命令 | 说明 |
|----------|------|------|
| CrossEntropy | `--loss ce` | 标准交叉熵 |
| Dice Loss | `--loss dice` | 手动实现，处理类别不平衡 |
| Combined | `--loss combined` | CE + Dice 各 0.5 权重 |
