"""基于 SECOND 的标定板检测器。

实现 CalibrationBoardDetector 接口:
  detect(pc_xyzi) -> (bool, ROI)

输入: 雷达点云 (N, 4) [x, y, z, intensity]
输出: 标定板 3D 框 ROI ((xmin,xmax),(ymin,ymax),(zmin,zmax))

用法:
  detector = SECONDBoardDetector(model_path="path/to/best_model.pth")
  ok, roi = detector.detect(pc)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# 让 second_model 可导入
_THIS_DIR = Path(__file__).resolve().parent
_SECOND_DIR = _THIS_DIR.parent / "models" / "second_model"
if str(_SECOND_DIR) not in sys.path:
    sys.path.insert(0, str(_SECOND_DIR))

from vfe import VoxelFeatureEncoder
from backbone import SpMiddleFPN
from rpn import RPN
from second import SECOND, generate_anchors


# ==================== 硬编码配置 (与训练时一致) ====================
VOXEL_SIZE = (0.1, 0.1, 0.2)
POINT_CLOUD_RANGE = (0.0, -5.0, -3.0, 10.0, 5.0, 2.0)
MAX_POINTS_PER_VOXEL = 10
MAX_NUM_VOXELS = 12000
GRID_SIZE = (100, 100, 25)
VFE_FILTERS = (32, 128)
SPARSE_FILTERS = (64, 128, 256)
UPSAMPLE_FILTERS = (128, 128, 128)
NUM_ANCHOR_PER_LOC = 2
NUM_CLASS = 1
ANCHOR_SIZES = [(0.55, 1.7, 1.18)]
ANCHOR_ROTATIONS = (0.0, 1.5707963)
SCORE_THR = 0.3
ROI_MARGIN = 0.3  # 检测框向外扩展的 padding (m)


def _points_to_voxels(points):
    """硬体素化 (复制自 preprocess/voxelize.py, 保持自包含)。"""
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    vx, vy, vz = VOXEL_SIZE
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        M = 0
        C = points.shape[1]
        return (
            np.zeros((M, MAX_POINTS_PER_VOXEL, C), dtype=np.float32),
            np.zeros((M, 3), dtype=np.int32),
            np.zeros((M,), dtype=np.int32),
        )
    ix = np.floor((pts[:, 0] - x_min) / vx).astype(np.int32)
    iy = np.floor((pts[:, 1] - y_min) / vy).astype(np.int32)
    iz = np.floor((pts[:, 2] - z_min) / vz).astype(np.int32)
    grid_dx = int(round((x_max - x_min) / vx))
    grid_dy = int(round((y_max - y_min) / vy))
    voxel_keys = ix * grid_dy * grid_dx + iy * grid_dx + iz
    order = np.argsort(voxel_keys)
    sorted_keys = voxel_keys[order]
    sorted_pts = pts[order]
    diff = np.diff(sorted_keys)
    split_idx = np.where(diff != 0)[0] + 1
    starts = np.concatenate([[0], split_idx])
    ends = np.concatenate([split_idx, [len(sorted_keys)]])
    num_voxels = len(starts)
    C = pts.shape[1]
    voxel_features = np.zeros((num_voxels, MAX_POINTS_PER_VOXEL, C), dtype=np.float32)
    voxel_coords = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points = np.zeros((num_voxels,), dtype=np.int32)
    for i in range(num_voxels):
        s, e = starts[i], ends[i]
        n = min(e - s, MAX_POINTS_PER_VOXEL)
        voxel_features[i, :n] = sorted_pts[s : s + n]
        num_points[i] = n
        p0 = sorted_pts[s]
        voxel_coords[i, 0] = int(np.floor((p0[0] - x_min) / vx))
        voxel_coords[i, 1] = int(np.floor((p0[1] - y_min) / vy))
        voxel_coords[i, 2] = int(np.floor((p0[2] - z_min) / vz))
    if num_voxels > MAX_NUM_VOXELS:
        voxel_features = voxel_features[:MAX_NUM_VOXELS]
        voxel_coords = voxel_coords[:MAX_NUM_VOXELS]
        num_points = num_points[:MAX_NUM_VOXELS]
    return voxel_features, voxel_coords, num_points


class SECONDBoardDetector:
    """SECOND 标定板检测器, 输出 3D 框 ROI。"""

    def __init__(self, model_path: str, score_thr: float = SCORE_THR):
        self.score_thr = score_thr
        self.device = torch.device("cpu")

        # 构建模型
        self.model = SECOND(
            in_channels=4,
            vfe_filters=VFE_FILTERS,
            sparse_filters=SPARSE_FILTERS,
            upsample_filters=UPSAMPLE_FILTERS,
            grid_size=GRID_SIZE,
            num_anchor_per_loc=NUM_ANCHOR_PER_LOC,
            num_class=NUM_CLASS,
        ).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)

        # ===== 权重调试 =====
        print(f"  [WEIGHT] ckpt keys count: {len(ckpt)}")
        print(f"  [WEIGHT] model state_dict keys count: {len(self.model.state_dict())}")
        ckpt_keys = set(ckpt.keys())
        model_keys = set(self.model.state_dict().keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        print(f"  [WEIGHT] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            print(f"  [WEIGHT] sample missing: {list(missing)[:3]}")
        if len(unexpected) > 0:
            print(f"  [WEIGHT] sample unexpected: {list(unexpected)[:3]}")

        # 检查权重统计
        all_vals = torch.cat([v.flatten() for v in ckpt.values() if v.numel() > 0])
        print(f"  [WEIGHT] overall mean={all_vals.mean():.4f}, std={all_vals.std():.4f}, max={all_vals.max():.4f}, min={all_vals.min():.4f}")

        self.model.load_state_dict(ckpt)
        self.model.eval()

        # 预生成 anchors
        self.anchors = generate_anchors(
            GRID_SIZE, VOXEL_SIZE, POINT_CLOUD_RANGE,
            ANCHOR_SIZES, ANCHOR_ROTATIONS,
        )  # (Dy, Dx, num_rot, 7)

    def _decode(self, cls_pred, loc_pred):
        """解码最高置信度的框, 返回 (cx, cy, cz, w, l, h, yaw, score)。"""
        num_rot = self.anchors.shape[2]
        Dy, Dx = cls_pred.shape[1], cls_pred.shape[2]
        loc = loc_pred.view(num_rot, 7, Dy, Dx).permute(2, 3, 0, 1).contiguous()
        scores = torch.sigmoid(cls_pred.permute(1, 2, 0).contiguous())

        # ===== 调试输出 =====
        print(f"  [DEBUG] cls_pred shape={tuple(cls_pred.shape)}, max={cls_pred.max():.3f}, min={cls_pred.min():.3f}")
        print(f"  [DEBUG] loc_pred shape={tuple(loc_pred.shape)}, max={loc_pred.max():.3f}, min={loc_pred.min():.3f}")
        print(f"  [DEBUG] anchors shape={tuple(self.anchors.shape)}")
        print(f"  [DEBUG] scores max={scores.max():.3f}, min={scores.min():.3f}, >0.3 count={(scores > 0.3).sum().item()}")

        ax, ay, az = self.anchors[..., 0], self.anchors[..., 1], self.anchors[..., 2]
        aw, al, ah = self.anchors[..., 3], self.anchors[..., 4], self.anchors[..., 5]
        ayaw = self.anchors[..., 6]
        dx, dy, dz, dw, dl, dh, dyaw = loc.unbind(dim=-1)

        print(f"  [DEBUG] dx range=[{dx.min():.3f}, {dx.max():.3f}]")
        print(f"  [DEBUG] dy range=[{dy.min():.3f}, {dy.max():.3f}]")
        print(f"  [DEBUG] dz range=[{dz.min():.3f}, {dz.max():.3f}]")
        print(f"  [DEBUG] dw range=[{dw.min():.3f}, {dw.max():.3f}]")
        print(f"  [DEBUG] dl range=[{dl.min():.3f}, {dl.max():.3f}]")
        print(f"  [DEBUG] dh range=[{dh.min():.3f}, {dh.max():.3f}]")

        pred_x = ax + dx * aw
        pred_y = ay + dy * al
        pred_z = az + dz
        # clamp 避免 exp 溢出
        dw = torch.clamp(dw, max=4.0)
        dl = torch.clamp(dl, max=4.0)
        dh = torch.clamp(dh, max=4.0)
        pred_w = aw * torch.exp(dw)
        pred_l = al * torch.exp(dl)
        pred_h = ah * torch.exp(dh)
        pred_yaw = ayaw + dyaw

        print(f"  [DEBUG] pred_x range=[{pred_x.min():.3f}, {pred_x.max():.3f}]")
        print(f"  [DEBUG] pred_y range=[{pred_y.min():.3f}, {pred_y.max():.3f}]")
        print(f"  [DEBUG] pred_z range=[{pred_z.min():.3f}, {pred_z.max():.3f}]")
        print(f"  [DEBUG] pred_w range=[{pred_w.min():.3f}, {pred_w.max():.3f}]")

        boxes = torch.stack([pred_x, pred_y, pred_z, pred_w, pred_l, pred_h, pred_yaw], dim=-1)
        boxes = boxes.cpu().numpy().reshape(-1, 7)
        scores = scores.cpu().numpy().reshape(-1)

        # 过滤不合理的 box: 中心坐标必须在点云范围内, 尺寸不能太大
        x_min_pc, y_min_pc, z_min_pc, x_max_pc, y_max_pc, z_max_pc = POINT_CLOUD_RANGE
        valid_mask = (
            (boxes[:, 0] >= x_min_pc) & (boxes[:, 0] <= x_max_pc)
            & (boxes[:, 1] >= y_min_pc) & (boxes[:, 1] <= y_max_pc)
            & (boxes[:, 2] >= z_min_pc) & (boxes[:, 2] <= z_max_pc)
            & (boxes[:, 3] < 5.0) & (boxes[:, 4] < 5.0) & (boxes[:, 5] < 5.0)
        )
        print(f"  [DEBUG] valid box count: {valid_mask.sum()} / {len(boxes)}")

        mask = (scores > self.score_thr) & valid_mask
        if mask.sum() == 0:
            return None
        boxes, scores = boxes[mask], scores[mask]
        best = np.argmax(scores)
        print(f"  [DEBUG] best score={scores[best]:.3f}, best box={boxes[best]}")
        return boxes[best], scores[best]

    def detect(self, pc_xyzi: np.ndarray):
        """检测标定板, 返回 (ok, ROI)。

        Args:
            pc_xyzi: (N, 4) [x, y, z, intensity]。

        Returns:
            ok: 是否检测到标定板。
            ROI: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) 或 None。
        """
        if pc_xyzi.shape[0] < 100:
            return False, None

        # 只取 x, y, z, intensity 4 维 (mcap 点云可能含 ring 等额外字段)
        pc_xyzi = pc_xyzi[:, :4].copy()

        # intensity 归一化: 训练时用的是 intensity/255.0 (0~1 范围)
        # 推理时如果 intensity 最大值 > 1.0, 说明是原始 0~255 范围, 需要除以 255
        inten = pc_xyzi[:, 3]
        print(f"  [INPUT] intensity range: [{inten.min():.4f}, {inten.max():.4f}], mean={inten.mean():.4f}")
        if inten.max() > 1.5:
            print(f"  [INPUT] intensity max > 1.5, 归一化除以 255.0 (与训练时一致)")
            pc_xyzi[:, 3] = inten / 255.0

        # 体素化
        vf, vc, np_pts = _points_to_voxels(pc_xyzi)
        if vf.shape[0] == 0:
            return False, None

        vf_t = torch.from_numpy(vf).unsqueeze(0).to(self.device)
        vc_t = torch.from_numpy(vc).unsqueeze(0).to(self.device)
        np_t = torch.from_numpy(np_pts).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_pred, loc_pred = self.model(
                vf_t, vc_t, np_t, VOXEL_SIZE, POINT_CLOUD_RANGE,
            )

        result = self._decode(cls_pred[0], loc_pred[0])
        if result is None:
            return False, None

        box, score = result
        cx, cy, cz, w, l, h, yaw = box

        # 从 3D 框生成 ROI (加 margin, 不做 yaw 旋转, 保守取 AABB)
        xmin = max(cx - w / 2 - ROI_MARGIN, POINT_CLOUD_RANGE[0])
        xmax = min(cx + w / 2 + ROI_MARGIN, POINT_CLOUD_RANGE[3])
        ymin = max(cy - l / 2 - ROI_MARGIN, POINT_CLOUD_RANGE[1])
        ymax = min(cy + l / 2 + ROI_MARGIN, POINT_CLOUD_RANGE[4])
        zmin = max(cz - h / 2 - ROI_MARGIN, POINT_CLOUD_RANGE[2])
        zmax = min(cz + h / 2 + ROI_MARGIN, POINT_CLOUD_RANGE[5])

        roi = (
            (float(xmin), float(xmax)),
            (float(ymin), float(ymax)),
            (float(zmin), float(zmax)),
        )
        return True, roi
