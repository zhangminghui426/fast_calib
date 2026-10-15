"""数据集: 从 PCD 文件和标注生成体素化数据。"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess.voxelize import points_to_voxels
from configs.default import (
    voxel_size, point_cloud_range, max_points_per_voxel, max_num_voxels,
    grid_size, anchor_sizes, anchor_rotations,
)
from models.second import generate_anchors


class BoardDataset(Dataset):
    """标定板数据集。"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = self._load_samples()
        self.anchors = generate_anchors(
            grid_size, voxel_size, point_cloud_range,
            anchor_sizes, anchor_rotations,
        )  # (Dy, Dx, num_rot, 7)

    def _load_samples(self):
        """加载所有样本路径。"""
        samples = []
        pcd_dir = os.path.join(self.data_dir, "pcd")
        label_dir = os.path.join(self.data_dir, "label")

        for fname in sorted(os.listdir(pcd_dir)):
            if fname.endswith(".bin"):
                pcd_path = os.path.join(pcd_dir, fname)
                label_path = os.path.join(label_dir, fname.replace(".bin", ".txt"))
                if os.path.exists(label_path):
                    samples.append((pcd_path, label_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pcd_path, label_path = self.samples[idx]

        # 加载点云
        points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)

        # 加载标注 (cx, cy, cz, w, l, h, yaw)
        with open(label_path, "r") as f:
            line = f.readline().strip()
            gt_box = np.array([float(x) for x in line.split()], dtype=np.float32)

        # 体素化
        vf, vc, np_pts = points_to_voxels(
            points, voxel_size, point_cloud_range,
            max_points_per_voxel, max_num_voxels,
        )

        # 生成标签
        cls_label, loc_label = self._make_label(gt_box)

        return {
            "voxel_features": torch.from_numpy(vf),
            "voxel_coords": torch.from_numpy(vc),
            "voxel_num_points": torch.from_numpy(np_pts),
            "cls_label": torch.from_numpy(cls_label),
            "loc_label": torch.from_numpy(loc_label),
        }

    def _make_label(self, gt_box):
        """为所有 anchors 生成分类和回归标签。"""
        Dy, Dx, num_rot, _ = self.anchors.shape
        num_anchors = Dy * Dx * num_rot

        cls_label = np.zeros((num_anchors,), dtype=np.float32)
        loc_label = np.zeros((num_anchors, 7), dtype=np.float32)

        gt_cx, gt_cy, gt_cz, gt_w, gt_l, gt_h, gt_yaw = gt_box

        # 找和 GT 最匹配的 anchor
        anchors_flat = self.anchors.reshape(-1, 7)
        ax, ay, az = anchors_flat[:, 0], anchors_flat[:, 1], anchors_flat[:, 2]
        aw, al, ah = anchors_flat[:, 3], anchors_flat[:, 4], anchors_flat[:, 5]
        ayaw = anchors_flat[:, 6]

        # 简化: 找中心最近的 anchor 作为正样本
        dist = (ax - gt_cx) ** 2 + (ay - gt_cy) ** 2 + (az - gt_cz) ** 2
        best_idx = np.argmin(dist)

        # 设置正样本标签
        cls_label[best_idx] = 1.0

        # 回归标签: (dx, dy, dz, dw, dl, dh, dyaw)
        loc_label[best_idx, 0] = (gt_cx - ax[best_idx]) / aw[best_idx]
        loc_label[best_idx, 1] = (gt_cy - ay[best_idx]) / al[best_idx]
        loc_label[best_idx, 2] = (gt_cz - az[best_idx])
        loc_label[best_idx, 3] = np.log(gt_w / aw[best_idx])
        loc_label[best_idx, 4] = np.log(gt_l / al[best_idx])
        loc_label[best_idx, 5] = np.log(gt_h / ah[best_idx])
        loc_label[best_idx, 6] = gt_yaw - ayaw[best_idx]

        return cls_label, loc_label
