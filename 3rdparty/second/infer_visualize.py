"""SECOND 推理和可视化脚本。"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from configs.default import (
    voxel_size, point_cloud_range, max_points_per_voxel, max_num_voxels,
    grid_size, vfe_filters, sparse_filters, upsample_filters,
    num_anchor_per_loc, num_class, anchor_sizes, anchor_rotations,
)
from models.second import SECOND, generate_anchors
from preprocess.voxelize import points_to_voxels
from utils.pcd_reader import read_pcd


def decode_boxes(cls_pred, loc_pred, anchors, score_thr=0.3):
    """解码预测框。"""
    num_rot = anchors.shape[2]
    Dy, Dx = cls_pred.shape[0], cls_pred.shape[1]

    # 简化: 直接返回最高置信度的框
    scores = torch.sigmoid(cls_pred)
    best_idx = torch.argmax(scores)
    best_score = scores[best_idx].item()

    if best_score < score_thr:
        return None, best_score

    # 简化: 从 loc_pred 中取对应的框
    # 实际实现需要和 anchors 对应
    return None, best_score


def main():
    model_path = "./output/best_model.pth"
    pcd_path = "./data/board/pcd/sample_001.bin"

    device = torch.device("cpu")

    # 加载模型
    model = SECOND(
        in_channels=4,
        vfe_filters=vfe_filters,
        sparse_filters=sparse_filters,
        upsample_filters=upsample_filters,
        grid_size=grid_size,
        num_anchor_per_loc=num_anchor_per_loc,
        num_class=num_class,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载点云
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    print(f"Points: {points.shape}")

    # 体素化
    vf, vc, np_pts = points_to_voxels(
        points, voxel_size, point_cloud_range,
        max_points_per_voxel, max_num_voxels,
    )

    # 前向
    vf_t = torch.from_numpy(vf).unsqueeze(0).to(device)
    vc_t = torch.from_numpy(vc).unsqueeze(0).to(device)
    np_t = torch.from_numpy(np_pts).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_pred, loc_pred = model(
            vf_t, vc_t, np_t, voxel_size, point_cloud_range,
        )

    print(f"cls_pred shape: {cls_pred.shape}")
    print(f"loc_pred shape: {loc_pred.shape}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Point Cloud")
    plt.savefig("./output/visualize.png", dpi=150)
    print("Saved visualization to ./output/visualize.png")


if __name__ == "__main__":
    main()
