# 胎儿心脏超声小结构分割的 Soft ROI 精修（ROI-TransUNet 官方实现）

[English](README.md) | [简体中文](README_zh-CN.md)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

基于 [TransUNet](https://github.com/Beckschen/TransUNet)（R50-ViT-B/16）。

| 出版信息 | |
|---|---|
| 标题 | Soft ROI Refinement for Small-Structure Segmentation in Fetal Cardiac Ultrasound |
| 作者 | Rui Wang, Camilla Fazi, Netzahualcoyotl Hernandez-Cruz |
| 会议 | IEEE IECBES 2026 *(已录用待刊)* |
| DOI | <!-- 录用后补充 --> |

## 概述

胎儿心脏超声中，肺动脉、上腔静脉、气管等小血管结构像素少、边界弱，与大腔室共享
同一个分割头时容易被忽略。**ROI-TransUNet** 在 R50-ViT-B/16 TransUNet 上加入一个
**软 ROI 精修分支**：decoder 特征、浅层图像投影和粗概率共同生成软定位图与零初始化的
残差 logits，只作用于四个最弱的结构；共享预测路径完整保留作为回退，推理时无需任何
后处理。

仓库完整复现论文的五阶段实验链——官方原始配方 → 修正前景 Dice 损失 →
ImageNet-21K 初始化 → 调优后的 Adjusted TransUNet → **+ Soft ROI Refiner**——
全部在统一协议下用五个固定训练种子（42/123/456/789/1024）完成。

## 贡献点

- **软定位 + 残差精修**设计：用连续注意力取代硬 ROI 裁剪，残差头零初始化，
  训练起点与基线完全一致。
- **五种子评估**并含训练配置对照，报告 95% t 区间与配对显著性检验，
  不以单种子结果下结论。
- 与 U-Net、Attention U-Net、UNet++、Swin-Unet 的**同协议对比**，
  覆盖重叠、边界与检出三类指标。
- 对方法效果的如实刻画：相对原始 TransUNet 的大部分提升来自调优后的训练配置，
  分支本身带来的是小幅、可解释的召回–精度权衡，而非全面大幅提升。

## 方法

<p align="center">
  <img src="assets/method_overview.png" width="100%">
</p>

- **软定位**：分支用 decoder 特征、投影图像和粗概率为每个弱结构预测 sigmoid
  注意力图——不做硬裁剪，定位错误可平滑退化。
- **零初始化残差头**：弱类别（LPA/RPA/SVC/气管，ID 8–11）各接一个零初始化的
  残差 logit 头；类别 0–7 完全旁路该分支。
- **修正损失**：0.2 CE + 0.8 前景 Dice，修正了原版损失对前景项多乘 11/12 的
  归一化错误。

## 主结果（五种子，冻结测试集）

跨训练种子的均值 ± 标准差（Dice/IoU/Recall/Precision/Boundary Dice 单位 %，
HD95/ASSD 单位像素，仅对检出结构平均）：

| 阶段 | Dice | IoU | Recall | Precision | Boundary Dice | HD95 | ASSD | 检出率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official raw | 68.57 ± 2.44 | 57.32 ± 2.12 | 68.93 ± 2.95 | 74.97 ± 2.00 | 39.25 ± 1.19 | 18.03 | 5.83 | 93.28 |
| Corrected Dice | 69.26 ± 1.16 | 57.83 ± 1.34 | 70.68 ± 1.47 | 74.44 ± 1.27 | 39.37 ± 1.09 | 18.98 | 6.06 | 94.28 |
| ImageNet-21K | 75.28 ± 0.99 | 63.64 ± 1.09 | 75.91 ± 1.87 | 79.51 ± 1.59 | 43.02 ± 1.13 | 12.53 | 4.38 | 97.23 |
| Adjusted TransUNet | 79.76 ± 0.50 | 68.94 ± 0.58 | 79.71 ± 1.05 | 83.19 ± 1.10 | 49.37 ± 0.73 | 10.44 | 3.70 | 98.36 |
| **+ Soft ROI Refiner** | **80.07 ± 0.53** | **69.24 ± 0.63** | **80.64 ± 0.93** | 82.77 ± 1.15 | **49.47 ± 0.78** | 10.61 | 3.73 | **98.58** |

ROI Refiner 相对调整后基线的逐种子配对差异**未达**统计显著（Dice Δ +0.32 pp，
p = 0.40）；因此表格以"含对照的递进链"呈现，而非大幅方法增益的证据。

评价口径：固定 894 图划分（626/89/179 训练/验证/测试）从不重新划分；区域指标只在
GT 中真实存在的 815 个测试 case–class 对上平均，漏检记零分；HD95/ASSD 仅对检出
结构平均，同时报告检出率。

## 定性结果

<p align="center">
  <img src="assets/qualitative_example.png" width="860">
</p>

两个代表性病例：精修分支补全了基线欠分割的结构（行 A），并基本找回基线完全漏掉的
微小血管（行 B）；decoder Grad-CAM 验证分支确实关注目标区域。

## 复现步骤

```bash
# 1. 克隆
git clone https://github.com/Ru1Wang05/Soft_ROI_Refinement.git
cd Soft_ROI_Refinement

# 2. 环境（Python >= 3.8，建议 CUDA）
pip install -r requirements.txt

# 3. 配置本地路径（脚本检测到 configs/paths.sh 即自动读取）
cp configs/paths.example.sh configs/paths.sh    # 设置 DATA_ROOT、LIST_DIR、PRETRAINED

# 4. 下载预训练编码器权重
bash scripts/00_download_pretrained.sh

# 5. 准备数据——每个病例一个 NPZ + 固定划分列表（见下）

# 6. 运行自检（CPU 即可）
bash scripts/04_run_tests.sh

# 7. 训练——五阶段递进链与同协议基线
bash scripts/01_train_progression.sh raw         # 官方 TransUNet 配方
bash scripts/01_train_progression.sh corrected   # + 修正损失
bash scripts/01_train_progression.sh pretrain    # + ImageNet-21k 初始化
bash scripts/01_train_progression.sh adjusted    # Adjusted TransUNet（主基线）
bash scripts/01_train_progression.sh roi         # + Soft ROI Refiner（本文方法）
bash scripts/02_train_baselines.sh               # U-Net / AttUNet / UNet++ / Swin-Unet

# 8. 汇总论文表格
bash scripts/03_summarize.sh
```

各阶段可只跑部分种子，如 `bash scripts/01_train_progression.sh roi 42 123`；
单个基线可用 `bash scripts/02_train_baselines.sh unet`。

### 期望的数据布局

```text
$DATA_ROOT/<case>.npz        键：image（H×W float）、label（H×W int，0–11）
$LIST_DIR/train.txt          固定划分列表，每行一个病例名
$LIST_DIR/val.txt
$LIST_DIR/test.txt
```

所有训练共用一个入口 `train.py --model {transunet, roi_refiner, unet, attunet,
unetpp, swinunet}`；评估走 `evaluate.py`（区域指标）与 `evaluate_boundary.py`
（Boundary Dice、HD95、ASSD、检出率、逐病例标签图）。主实验默认协议：384×384
输入、batch 4、70 epochs、SGD lr 0.01 多项式衰减（power 0.9）、每 2 轮验证、
按验证 Dice 保存 `best_dice.pth`、AMP + 确定性模式。

## 仓库结构

```text
├── networks/            TransUNet 编码器/解码器、Soft ROI Refiner、CNN 基线
├── datasets/            固定划分的 NPZ 数据集与增强
├── losses.py            组合损失（含修正前景 Dice）
├── trainers.py          共享训练循环 + present-case-class 评估
├── train.py             所有模型的统一训练入口
├── evaluate.py          单 checkpoint 区域指标
├── evaluate_boundary.py 边界指标、逐病例 CSV 与标签图
├── summarize.py         冻结结果汇总为论文表格
├── tests/run_checks.py  损失/协议、ROI 通道映射与架构自检
├── scripts/             编号复现入口（00–04）
├── configs/             可移植路径配置模板
├── citations/           BibTeX 引用导出
├── assets/              方法图与定性图
├── CITATION.cff         机器可读引用元数据
└── LICENSE              Apache-2.0（继承自 TransUNet）
```

推理时不需要 `tests/` 与 `configs/`；其余均为复现流水线的一部分。

## 数据与权重

D10 胎儿心脏数据集与训练好的 checkpoint **不随仓库分发**。使用者把预处理后的
NPZ 数据和固定划分列表放在本地，通过 `configs/paths.sh` 指向即可；ImageNet-21k
编码器权重由 `scripts/00_download_pretrained.sh` 下载。

## 引用

如使用本代码，请引用：

```bibtex
@inproceedings{wang2026softroi,
  title     = {Soft ROI Refinement for Small-Structure Segmentation in Fetal Cardiac Ultrasound},
  author    = {Wang, Rui and Fazi, Camilla and Hernandez-Cruz, Netzahualcoyotl},
  booktitle = {IEEE IECBES},
  year      = {2026},
  note      = {To appear}
}
```

机器可读引用见 [CITATION.cff](CITATION.cff)；BibTeX 文件在 [citations/](citations/)。

## 致谢

- [TransUNet](https://github.com/Beckschen/TransUNet)（Apache-2.0）——本项目所基于的骨干实现。
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)——Swin-Unet 基线的训练配方。

## 许可

本项目继承 TransUNet 的 Apache-2.0 许可，见 [LICENSE](LICENSE)。

## 医学免责声明

本软件仅用于科学研究，未获临床使用批准，不能替代专业医疗诊断。
