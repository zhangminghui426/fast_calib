"""稀疏卷积 Backbone (SpMiddleResNetFPN)。

使用 spconv 进行 3D 稀疏卷积编码, 再通过反卷积多尺度上采样,
最终在 BEV 视角拼接出致密特征图。

若 spconv 不可用, 自动 fallback 到致密 3D 卷积 (scatter to dense),
保证训练流程可跑通。
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False


class SparseBackend(nn.Module):
    """稀疏卷积后端, 自动选择 spconv 或致密 fallback。"""

    def __init__(self, in_channels: int, grid_size: tuple[int, int, int]):
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.use_spconv = HAS_SPCONV

    def features_to_dense(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        Dx, Dy, Dz = self.grid_size
        B, M, C = voxel_features.shape
        dense = torch.zeros(B, C, Dz, Dy, Dx, device=voxel_features.device, dtype=voxel_features.dtype)
        for b in range(batch_size):
            ix = voxel_coords[b, :, 0].long()
            iy = voxel_coords[b, :, 1].long()
            iz = voxel_coords[b, :, 2].long()
            feat = voxel_features[b]
            dense[b, :, iz, iy, ix] = feat.t()
        return dense

    def sparse_conv_forward(
        self,
        x: torch.Tensor,
        module: nn.Module,
        indices_key: str = "indices",
    ) -> torch.Tensor:
        return module(x)


class SpMiddleFPN(nn.Module):
    """SECOND 稀疏中间层 + FPN。

    结构:
      - 3 个稀疏卷积下采样 (stride=2)
      - 3 个稀疏反卷积上采样 (stride=2)
      - 在 BEV 维度 (z) 压缩, 拼接多尺度特征
    """

    def __init__(
        self,
        in_channels: int = 128,
        sparse_filters: tuple[int, ...] = (64, 128, 256),
        upsample_filters: tuple[int, ...] = (128, 128, 128),
        grid_size: tuple[int, int, int] = (200, 200, 25),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.use_spconv = HAS_SPCONV and torch.cuda.is_available()

        if self.use_spconv:
            self.down_convs = nn.ModuleList()
            prev_ch = in_channels
            for out_ch in sparse_filters:
                self.down_convs.append(
                    spconv.SparseSequential(
                        spconv.SparseConv3d(prev_ch, out_ch, kernel_size=3,
                                            stride=2, padding=1, indice_key=f"sp{out_ch}"),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                    )
                )
                prev_ch = out_ch

            self.up_convs = nn.ModuleList()
            for i, out_ch in enumerate(upsample_filters):
                in_ch_deconv = sparse_filters[-(i + 1)]
                self.up_convs.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(in_ch_deconv, out_ch, kernel_size=3,
                                                   indice_key=f"sp{sparse_filters[-(i+1)]}"),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                    )
                )
        else:
            self.down_convs = nn.ModuleList()
            prev_ch = in_channels
            for out_ch in sparse_filters:
                self.down_convs.append(nn.Sequential(
                    nn.Conv3d(prev_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(),
                ))
                prev_ch = out_ch

            self.up_convs = nn.ModuleList()
            for i, out_ch in enumerate(upsample_filters):
                in_ch_deconv = sparse_filters[-(i + 1)]
                self.up_convs.append(nn.Sequential(
                    nn.ConvTranspose3d(in_ch_deconv, out_ch, kernel_size=3,
                                        stride=2, padding=1, output_padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(),
                ))

        self.out_channels = sum(upsample_filters)

    def forward(
        self,
        voxel_embedding: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        Dx, Dy, Dz = self.grid_size

        if self.use_spconv:
            return self._forward_spconv(voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz)
        else:
            return self._forward_dense(voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz)

    def _forward_spconv(self, voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz):
        B, M, C = voxel_embedding.shape
        all_indices = []
        all_features = []
        for b in range(B):
            n = voxel_coords[b].shape[0]
            batch_idx = torch.full((n, 1), b, device=voxel_embedding.device, dtype=torch.int32)
            coords = voxel_coords[b].int()
            sp_coords = torch.stack([coords[:, 2], coords[:, 1], coords[:, 0]], dim=1)
            indices = torch.cat([batch_idx, sp_coords], dim=1)
            all_indices.append(indices)
            all_features.append(voxel_embedding[b])
        indices = torch.cat(all_indices, dim=0).int()
        features = torch.cat(all_features, dim=0)

        x = spconv.SparseConvTensor(features, indices, (Dz, Dy, Dx), batch_size)

        downs = []
        for conv in self.down_convs:
            x = conv(x)
            downs.append(x)

        x = self.up_convs[0](downs[-1])
        ups = [x.dense()]
        for i in range(1, len(self.up_convs)):
            x = self.up_convs[i](downs[-(i + 1)])
            ups.append(x.dense())

        bev_list = []
        for up in ups:
            up = nn.functional.interpolate(
                up, size=(Dz, Dy, Dx), mode="trilinear", align_corners=False
            )
            bev = up.sum(dim=2)
            bev_list.append(bev)
        return torch.cat(bev_list, dim=1)

    def _forward_dense(self, voxel_embedding, voxel_coords, batch_size, Dx, Dy, Dz):
        B, M, C = voxel_embedding.shape
        dense = torch.zeros(B, C, Dz, Dy, Dx, device=voxel_embedding.device, dtype=voxel_embedding.dtype)
        for b in range(B):
            ix = voxel_coords[b, :, 0].long()
            iy = voxel_coords[b, :, 1].long()
            iz = voxel_coords[b, :, 2].long()
            dense[b, :, iz, iy, ix] = voxel_embedding[b].t()

        x = dense
        downs = []
        for conv in self.down_convs:
            x = conv(x)
            downs.append(x)

        ups = []
        x = self.up_convs[0](downs[-1])
        ups.append(x)
        for i in range(1, len(self.up_convs)):
            x = self.up_convs[i](downs[-(i + 1)])
            ups.append(x)

        bev_list = []
        for up in ups:
            up = nn.functional.interpolate(
                up, size=(Dz, Dy, Dx), mode="trilinear", align_corners=False
            )
            bev = up.sum(dim=2)
            bev_list.append(bev)
        return torch.cat(bev_list, dim=1)
