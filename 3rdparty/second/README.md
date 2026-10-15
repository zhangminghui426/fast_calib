# SECOND 3D 检测器

标定板 3D 目标检测项目, 用于 fast_calib 雷达-相机外参标定。

## 目录结构

```
second/
├── configs/          # 配置文件
│   └── default.py   # 默认配置
├── datasets/         # 数据集
│   └── dataset.py   # BoardDataset
├── preprocess/       # 数据预处理
│   ├── prepare_data.py  # 数据准备
│   └── voxelize.py     # 体素化
├── models/           # 模型定义
│   ├── vfe.py       # Voxel Feature Encoder
│   ├── backbone.py  # SpMiddleFPN
│   ├── rpn.py       # Region Proposal Network
│   └── second.py    # SECOND 主模型
├── utils/            # 工具函数
│   └── pcd_reader.py # PCD 文件读取
├── train.py          # 训练脚本
├── infer_visualize.py # 推理和可视化脚本
└── README.md          # 说明文档
```

## 快速开始

### 1. 准备数据

把 PCD 文件和标注放到 `data/board/pcd/` 和 `data/board/label/` 目录下。

### 2. 训练

```bash
cd 3rdparty/second
python train.py
```

### 3. 推理

```bash
python infer_visualize.py
```

## 模型结构

- VFE: 逐体素特征编码 (MLP + Max Pooling)
- Backbone: 简化版稀疏 backbone (MLP)
- RPN: 分类头 + 回归头

## 配置说明

默认配置在 `configs/default.py` 中, 可以根据需要修改。
