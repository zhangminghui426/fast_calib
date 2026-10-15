"""SpMiddleFPN: 稀疏 3D backbone + 上采样 FPN。"""

import torch
import torch.nn as nn


def scatter_voxel_features(voxel_features, voxel_coords, grid_size):
    """把体素特征散回到稠密网格 (简化实现, CPU 用 numpy 不现实, 这里直接输出稀疏索引)。

    为了避免 spconv 依赖, 这里用简化的 MLP 直接处理体素特征序列,
    不做 3D 卷积。这是为了快速验证流程的简化版本。
    """
    return voxel_features


class SpMiddleFPN(nn.Module):
    """简化版稀疏 backbone: 直接用 MLP 处理体素特征, 不做 3D 卷积。"""

    def __init__(self, in_channels: int, filters=(64, 128, 256), upsample_filters=(128, 128, 128)):
        super().__init__()
        # 简化版: 直接用 MLP 替代稀疏卷积
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, filters[0]),
            nn.ReLU(),
            nn.Linear(filters[0], filters[1]),
            nn.ReLU(),
            nn.Linear(filters[1], filters[2]),
            nn.ReLU(),
        )
        self.out_channels = filters[2]

    def forward(self, voxel_features, voxel_coords, grid_size):
        x = self.mlp(voxel_features)
        return x
