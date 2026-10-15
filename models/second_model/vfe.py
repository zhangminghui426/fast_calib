"""Voxel Feature Encoding (VFE)。

SECOND 的体素特征编码层:
  1. 逐点 MLP 编码原始特征 (x, y, z, intensity)
  2. 拼接体素内相对坐标
  3. 体素内 max pooling 得到局部聚合特征
  4. 再次 MLP 编码, 输出每个体素的特征向量
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VFELayer(nn.Module):
    """单层 VFE: 逐点编码 + max pooling + 特征拼接。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, voxel_features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        B, M, T, C = voxel_features.shape
        x = voxel_features.reshape(B * M * T, C)
        x = self.linear(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = x.reshape(B, M, T, -1)

        mask = point_mask.unsqueeze(-1)
        x = x * mask + (~mask.bool()) * (-1e6)
        pooled = x.max(dim=2, keepdim=True)[0]
        pooled = pooled.expand(-1, -1, T, -1)

        out = torch.cat([x, pooled], dim=-1)
        return out


class VoxelFeatureEncoder(nn.Module):
    """VFE 堆叠 + 最终 max pooling。"""

    def __init__(self, in_channels: int = 4, num_filters: tuple[int, ...] = (32, 128)):
        super().__init__()
        layers = []
        prev_ch = in_channels + 3
        for out_ch in num_filters:
            layers.append(VFELayer(prev_ch, out_ch))
            prev_ch = out_ch * 2
        self.layers = nn.ModuleList(layers)
        self.out_channels = prev_ch

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_points: torch.Tensor,
        grid_size: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        point_cloud_range: tuple[float, float, float, float, float, float],
    ) -> torch.Tensor:
        B, M, T, C = voxel_features.shape

        point_mask = (torch.arange(T, device=voxel_features.device)[None, None, :]
                      < num_points[:, :, None]).float()

        xmin, ymin, zmin = point_cloud_range[:3]
        vx, vy, vz = voxel_size
        coords_f = voxel_coords.float()
        voxel_center_x = (coords_f[..., 0] + 0.5) * vx + xmin
        voxel_center_y = (coords_f[..., 1] + 0.5) * vy + ymin
        voxel_center_z = (coords_f[..., 2] + 0.5) * vz + zmin
        voxel_center = torch.stack([voxel_center_x, voxel_center_y, voxel_center_z], dim=-1)
        voxel_center = voxel_center.unsqueeze(2).expand(-1, -1, T, -1)

        pts_xyz = voxel_features[..., :3]
        relative_xyz = pts_xyz - voxel_center

        x = torch.cat([voxel_features, relative_xyz], dim=-1)

        for layer in self.layers:
            x = layer(x, point_mask)

        mask = point_mask.unsqueeze(-1)
        x = x * mask + (~mask.bool()) * (-1e6)
        voxel_embedding = x.max(dim=2)[0]
        return voxel_embedding
