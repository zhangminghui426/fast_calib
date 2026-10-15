"""基于 SECOND 的标定板检测器。

实现 CalibrationBoardDetector 接口:
  detect(pc_xyzi) -> (bool, ROI)

输入: 雷达点云 (N, 4) [x, y, z, intensity]
输出: 标定板 3D 框 ROI ((xmin,xmax),(ymin,ymax),(zmin,zmax))
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SECOND_DIR = _THIS_DIR.parent / "models" / "second_model"
if str(_SECOND_DIR) not in sys.path:
    sys.path.insert(0, str(_SECOND_DIR))

from vfe import VoxelFeatureEncoder
from backbone import SpMiddleFPN
from rpn import RPN
from second import SECOND, generate_anchors


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
ROI_MARGIN = 0.3


def _points_to_voxels(points):
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
    def __init__(self, model_path: str, score_thr: float = SCORE_THR):
        self.score_thr = score_thr
        self.device = torch.device("cpu")

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
        self.model.load_state_dict(ckpt)
        self.model.eval()

        self.anchors = generate_anchors(
            GRID_SIZE, VOXEL_SIZE, POINT_CLOUD_RANGE,
            ANCHOR_SIZES, ANCHOR_ROTATIONS,
        )

    def _decode(self, cls_pred, loc_pred):
        num_rot = self.anchors.shape[2]
        Dy, Dx = cls_pred.shape[1], cls_pred.shape[2]
        loc = loc_pred.view(num_rot, 7, Dy, Dx).permute(2, 3, 0, 1).contiguous()
        scores = torch.sigmoid(cls_pred.permute(1, 2, 0).contiguous())

        ax, ay, az = self.anchors[..., 0], self.anchors[..., 1], self.anchors[..., 2]
        aw, al, ah = self.anchors[..., 3], self.anchors[..., 4], self.anchors[..., 5]
        ayaw = self.anchors[..., 6]
        dx, dy, dz, dw, dl, dh, dyaw = loc.unbind(dim=-1)

        pred_x = ax + dx * aw
        pred_y = ay + dy * al
        pred_z = az + dz
        pred_w = aw * torch.exp(dw)
        pred_l = al * torch.exp(dl)
        pred_h = ah * torch.exp(dh)
        pred_yaw = ayaw + dyaw

        boxes = torch.stack([pred_x, pred_y, pred_z, pred_w, pred_l, pred_h, pred_yaw], dim=-1)
        boxes = boxes.cpu().numpy().reshape(-1, 7)
        scores = scores.cpu().numpy().reshape(-1)

        mask = scores > self.score_thr
        if mask.sum() == 0:
            return None
        boxes, scores = boxes[mask], scores[mask]
        best = np.argmax(scores)
        return boxes[best], scores[best]

    def detect(self, pc_xyzi: np.ndarray):
        if pc_xyzi.shape[0] < 100:
            return False, None

        pc_xyzi = pc_xyzi[:, :4].copy()

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
