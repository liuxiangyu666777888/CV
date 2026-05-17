# Assignment 1

Oxford-IIIT Pet Dataset 上的宠物分类微调实验代码，覆盖以下任务：

- `ResNet-18/ResNet-34` baseline
- 预训练与随机初始化消融
- 超参数组合实验
- 注意力模块对比：`SE` 与 `CBAM`

## 目录结构

```text
assignment1/
├─ configs/
├─ outputs/
├─ src/
├─ README.md
└─ requirements.txt
```

## 数据准备

当前代码默认读取：

- `../data/oxford_pet_images/images`
- `../data/oxford_pet_annotations/annotations`

其中：

- 图片来自 `images.tar.gz` 解压后的 `images/`
- 标注来自 `annotations.tar.gz` 解压后的 `annotations/`

## 环境安装

```bash
pip install -r requirements.txt
```

## 运行示例

Baseline:

```bash
python src/train.py --config configs/baseline_resnet18_pretrained.yaml
```

预训练消融：

```bash
python src/train.py --config configs/baseline_resnet18_scratch.yaml
```

注意力模块对比：

```bash
python src/train.py --config configs/resnet18_se.yaml
python src/train.py --config configs/resnet18_cbam.yaml
```

批量运行多个现成配置：

```bash
python src/train.py --config configs/baseline_resnet18_pretrained.yaml --set train.lr=0.001 train.epochs=20
```

快速冒烟测试：

```bash
python src/train.py --config configs/smoke_test.yaml
```

一键超参数分析：

```powershell
.\run_hparam_analysis.ps1
```

汇总实验结果：

```bash
python src/summarize_results.py
```

## 输出文件

每个实验会在 `outputs/<experiment_name>/` 下生成：

- `best.pt`：最优权重
- `history.csv`：训练日志
- `curves.png`：loss / accuracy 曲线
- `predictions.csv`：测试集预测明细
- `confusion_matrix.png`：混淆矩阵
- `summary.json`：实验摘要

## 建议实验表

你可以在报告里对比以下结果：

1. `resnet18 + pretrained` vs `resnet18 + scratch`
2. `resnet18` 不同学习率、epoch、batch size
3. `resnet18 + se` / `resnet18 + cbam` vs baseline
4. `resnet34 + pretrained` vs `resnet18 + pretrained`

## 说明

- 当前实现使用官方 `trainval.txt` 与 `test.txt`
- 类别数为 `37`
- 默认评价指标为 `Accuracy`
- 代码未集成 `wandb/swanlab`，但导出了完整日志和曲线，可直接补进报告
- 现在支持命令行覆盖配置，无需为每组超参数手动复制 yaml
