"""SECOND 检测模型整体封装。

VFE -> SpMiddleFPN -> RPN, 并提供 anchor 生成与解码工具。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vfe import VoxelFeatureEncoder
from backbone import SpMiddleFPN
from rpn import RPN


class SECOND(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        vfe_filters: tuple[int, ...] = (32, 128),
        sparse_filters: tuple[int, ...] = (64, 128, 256),
        upsample_filters: tuple[int, ...] = (128, 128, 128),
        grid_size: tuple[int, int, int] = (200, 200, 25),
        num_anchor_per_loc: int = 2,
        num_class: int = 1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_anchor_per_loc = num_anchor_per_loc
        self.num_class = num_class

        self.vfe = VoxelFeatureEncoder(in_channels=in_channels, num_filters=vfe_filters)
        backbone_in = self.vfe.out_channels
        self.backbone = SpMiddleFPN(
            in_channels=backbone_in,
            sparse_filters=sparse_filters,
            upsample_filters=upsample_filters,
            grid_size=grid_size,
        )
        self.rpn = RPN(
            in_channels=self.backbone.out_channels,
            num_anchor_per_loc=num_anchor_per_loc,
            num_class=num_class,
        )

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_points: torch.Tensor,
        voxel_size: tuple[float, float, float],
        point_cloud_range: tuple[float, float, float, float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = voxel_features.shape[0]
        voxel_embedding = self.vfe(
            voxel_features, voxel_coords, num_points,
            self.grid_size, voxel_size, point_cloud_range,
        )
        bev_feature = self.backbone(voxel_embedding, voxel_coords, B)
        cls_pred, loc_pred = self.rpn(bev_feature)
        return cls_pred, loc_pred


def generate_anchors(
    grid_size: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    point_cloud_range: tuple[float, float, float, float, float, float],
    anchor_sizes: list[tuple[float, float, float]],
    anchor_rotations: tuple[float, ...],
) -> torch.Tensor:
    Dx, Dy, Dz = grid_size
    vx, vy, vz = voxel_size
    xmin, ymin, zmin, _, _, zmax = point_cloud_range

    xs = torch.arange(Dx, dtype=torch.float32) * vx + vx / 2 + xmin
    ys = torch.arange(Dy, dtype=torch.float32) * vy + vy / 2 + ymin
    z_center = (zmin + zmax) / 2

    gy, gx = torch.meshgrid(torch.arange(Dy), torch.arange(Dx), indexing="ij")
    cx = xs[gx]
    cy = ys[gy]
    cz = torch.full_like(cx, z_center)

    anchors_list = []
    for (w, l, h) in anchor_sizes:
        for rot in anchor_rotations:
            a = torch.stack([cx, cy, cz,
                             torch.full_like(cx, w),
                             torch.full_like(cx, l),
                             torch.full_like(cx, h),
                             torch.full_like(cx, rot)], dim=-1)
            anchors_list.append(a)

    anchors = torch.stack(anchors_list, dim=2)
    return anchors
