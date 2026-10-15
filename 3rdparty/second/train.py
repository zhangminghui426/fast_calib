"""SECOND 训练脚本。"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.default import (
    voxel_size, point_cloud_range, max_points_per_voxel, max_num_voxels,
    grid_size, vfe_filters, sparse_filters, upsample_filters,
    num_anchor_per_loc, num_class, batch_size, lr, num_epochs,
    cls_weight, loc_weight,
)
from datasets.dataset import BoardDataset
from models.second import SECOND


def collate_fn(batch):
    """自定义 collate: 因为体素数量不固定, 不能直接 stack。"""
    # 简化: 只取第一个样本 (batch_size=1)
    return batch[0]


def main():
    # 数据目录
    data_dir = "./data/board"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # 数据集
    dataset = BoardDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SECOND(
        in_channels=4,
        vfe_filters=vfe_filters,
        sparse_filters=sparse_filters,
        upsample_filters=upsample_filters,
        grid_size=grid_size,
        num_anchor_per_loc=num_anchor_per_loc,
        num_class=num_class,
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 损失函数
    cls_loss_fn = nn.BCEWithLogitsLoss()
    loc_loss_fn = nn.SmoothL1Loss()

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            vf = batch["voxel_features"].unsqueeze(0).to(device)
            vc = batch["voxel_coords"].unsqueeze(0).to(device)
            np_pts = batch["voxel_num_points"].unsqueeze(0).to(device)
            cls_label = batch["cls_label"].to(device)
            loc_label = batch["loc_label"].to(device)

            # 前向
            cls_pred, loc_pred = model(
                vf, vc, np_pts, voxel_size, point_cloud_range,
            )

            # 计算损失
            # 简化: 因为体素数量和 anchors 数量不一定对齐, 这里用简化的损失
            # 实际使用时需要确保 cls_pred 和 cls_label 形状一致
            cls_loss = cls_loss_fn(cls_pred.squeeze(), cls_label)
            loc_loss = loc_loss_fn(loc_pred, loc_label)
            loss = cls_weight * cls_loss + loc_weight * loc_loss

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 保存模型
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}")

    # 保存最终模型
    final_path = os.path.join(output_dir, "best_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
