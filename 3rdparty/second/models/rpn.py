"""Region Proposal Network (RPN)。"""

import torch
import torch.nn as nn


class RPN(nn.Module):
    """简化版 RPN: 直接对每个体素做分类和回归预测。"""

    def __init__(self, in_channels: int, num_anchors_per_loc: int = 2, num_class: int = 1):
        super().__init__()
        self.num_anchors_per_loc = num_anchors_per_loc
        self.num_class = num_class
        # 分类头: 每个 anchor 一个类别的置信度
        self.cls_head = nn.Linear(in_channels, num_anchors_per_loc * num_class)
        # 回归头: 每个 anchor 7 个值 (dx, dy, dz, dw, dl, dh, dyaw)
        self.loc_head = nn.Linear(in_channels, num_anchors_per_loc * 7)

    def forward(self, x, grid_size):
        """
        Args:
            x: (M, C) 体素特征。
            grid_size: (Dx, Dy, Dz) 网格尺寸。

        Returns:
            cls_pred: (num_anchors, num_class) 分类预测。
            loc_pred: (num_anchors, 7) 回归预测。
        """
        cls = self.cls_head(x)  # (M, num_anchors_per_loc * num_class)
        loc = self.loc_head(x)  # (M, num_anchors_per_loc * 7)

        # 简化: 直接返回每个体素的预测, 不做网格重排
        # 实际使用时需要和 anchors 一一对应
        return cls, loc
