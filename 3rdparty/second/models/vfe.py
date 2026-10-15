"""Voxel Feature Encoder (VFE)。"""

import torch
import torch.nn as nn


class VoxelFeatureEncoder(nn.Module):
    """逐体素特征编码: 对每个体素内的点做逐点 MLP, 再做最大池化。"""

    def __init__(self, in_channels: int, filters=(32, 128)):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in filters:
            layers.append(nn.Linear(c_in, c_out))
            layers.append(nn.ReLU())
            c_in = c_out
        self.mlp = nn.Sequential(*layers)
        self.out_channels = filters[-1]

    def forward(self, voxel_features, voxel_num_points):
        """
        Args:
            voxel_features: (M, T, C) 体素特征。
            voxel_num_points: (M,) 每个体素的点数。

        Returns:
            (M, out_channels) 每个体素的特征。
        """
        M, T, C = voxel_features.shape
        x = voxel_features.view(M * T, C)
        x = self.mlp(x)
        x = x.view(M, T, -1)
        # 用点数 mask 把不足 T 的位置置为 -inf, 避免池化到无效点
        mask = torch.arange(T, device=x.device)[None, :] < voxel_num_points[:, None]
        x = x.masked_fill(~mask[:, :, None], -1e9)
        x = x.max(dim=1).values
        return x
