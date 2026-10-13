# 胎儿心脏超声 Soft ROI Refiner 分割

基于 TransUNet（R50-ViT-B/16）的胎儿心脏超声多结构分割（12 类：背景 + 11 个解剖结构），
并针对四个小目标结构（左/右肺动脉、上腔静脉、气管）提出 Soft ROI Refiner。

详细说明（方法、数据协议、完整用法）见 [README.md](README.md)。

## 方法概要

<p align="center">
  <img src="assets/method_overview.png" width="720">
</p>

实验递进链：官方原始配方 → 修正前景 Dice 损失 → ImageNet-21K 初始化 →
Adjusted TransUNet（384 输入 / 70 epochs，主基线）→ + Soft ROI Refiner（本文方法）。

Refiner 利用 decoder 特征、原图投影与粗概率生成软定位图和残差 logits；残差层零初始化，
仅修改类别 8–11，类别 0–7 旁路。

## 五种子结果（42/123/456/789/1024）

| 阶段 | Dice | IoU | Boundary Dice | Detection % |
|---|---:|---:|---:|---:|
| Official raw | 68.57 ± 2.44 | 57.32 ± 2.12 | 39.25 ± 1.19 | 93.28 |
| Corrected Dice | 69.26 ± 1.16 | 57.83 ± 1.34 | 39.37 ± 1.09 | 94.28 |
| ImageNet-21K | 75.28 ± 0.99 | 63.64 ± 1.09 | 43.02 ± 1.13 | 97.23 |
| Adjusted TransUNet | 79.76 ± 0.50 | 68.94 ± 0.58 | 49.37 ± 0.73 | 98.36 |
| **+ Soft ROI Refiner** | **80.07 ± 0.53** | **69.24 ± 0.63** | **49.47 ± 0.78** | **98.58** |

ROI Refiner 相对调整后基线的逐种子配对差异未达统计显著（Dice Δ +0.32 pp，p = 0.40），
应表述为"平均小幅改善并伴随可解释的召回–精度权衡"，不得声称显著提升。

## 快速开始

```bash
pip install -r requirements.txt
cp configs/paths.example.sh configs/paths.sh   # 修改其中的数据/权重路径

bash scripts/00_download_pretrained.sh
bash scripts/04_run_tests.sh
bash scripts/01_train_progression.sh adjusted  # 主基线（5 种子）
bash scripts/01_train_progression.sh roi       # 本文方法（5 种子）
bash scripts/02_train_baselines.sh             # U-Net 系列基线
bash scripts/03_summarize.sh                   # 汇总论文表格
```

数据格式：`$DATA_ROOT/<case>.npz`（键：`image`、`label`）+ `$LIST_DIR/train|val|test.txt`
（固定划分，不得重新划分）。D10 数据集不随仓库分发。

## 许可

继承 TransUNet 的 Apache-2.0 许可，见 [LICENSE](LICENSE)。
