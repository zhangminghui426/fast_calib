"""SECOND: 3D 目标检测器。"""

import numpy as np
import torch
import torch.nn as nn

from vfe import VoxelFeatureEncoder
from backbone import SpMiddleFPN
from rpn import RPN


def generate_anchors(grid_size, voxel_size, point_cloud_range, anchor_sizes, anchor_rotations):
    """生成所有位置的 anchors。

    Args:
        grid_size: (Dx, Dy, Dz) 体素网格尺寸。
        voxel_size: (vx, vy, vz) 体素大小。
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max) 点云范围。
        anchor_sizes: list of (w, l, h) anchor 尺寸。
        anchor_rotations: tuple of (rot) anchor 角度。

    Returns:
        anchors: (Dy, Dx, num_rot, 7) 每个网格位置的 anchors [x, y, z, w, l, h, rot]。
    """
    Dx, Dy, Dz = grid_size
    vx, vy, vz = voxel_size
    x_min, y_min, z_min = point_cloud_range[:3]

    # 每个体素中心的坐标
    cx = x_min + (np.arange(Dx) + 0.5) * vx
    cy = y_min + (np.arange(Dy) + 0.5) * vy
    cz = z_min + (np.arange(Dz) + 0.5) * vz

    # 只在 z 中间层生成 anchors (简化)
    z_mid = cz[Dz // 2]

    num_rot = len(anchor_rotations)
    anchors = np.zeros((Dy, Dx, num_rot, 7), dtype=np.float32)

    for i, y in enumerate(cy):
        for j, x in enumerate(cx):
            for k, rot in enumerate(anchor_rotations):
                w, l, h = anchor_sizes[0]  # 简化: 只用第一个尺寸
                anchors[i, j, k] = [x, y, z_mid, w, l, h, rot]

    return anchors


class SECOND(nn.Module):
    """SECOND 检测器: VFE + Backbone + RPN。"""

    def __init__(
        self,
        in_channels: int = 4,
        vfe_filters=(32, 128),
        sparse_filters=(64, 128, 256),
        upsample_filters=(128, 128, 128),
        grid_size=(100, 100, 25),
        num_anchor_per_loc: int = 2,
        num_class: int = 1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.vfe = VoxelFeatureEncoder(in_channels, vfe_filters)
        self.backbone = SpMiddleFPN(self.vfe.out_channels, sparse_filters, upsample_filters)
        self.rpn = RPN(self.backbone.out_channels, num_anchor_per_loc, num_class)

    def forward(self, voxel_features, voxel_coords, voxel_num_points, voxel_size, point_cloud_range):
        """
        Args:
            voxel_features: (M, T, C) 体素特征。
            voxel_coords: (M, 3) 体素坐标。
            voxel_num_points: (M,) 每个体素的点数。
            voxel_size: (vx, vy, vz) 体素大小。
            point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max) 点云范围。

        Returns:
            cls_pred: (num_anchors, num_class) 分类预测。
            loc_pred: (num_anchors, 7) 回归预测。
        """
        # VFE
        voxel_feat = self.vfe(voxel_features, voxel_num_points)  # (M, C)

        # Backbone
        backbone_feat = self.backbone(voxel_feat, voxel_coords, self.grid_size)  # (M, C)

        # RPN
        cls_pred, loc_pred = self.rpn(backbone_feat, self.grid_size)

        return cls_pred, loc_pred
