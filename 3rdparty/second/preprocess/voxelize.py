"""体素化: 把点云转成体素特征。"""

import numpy as np


def points_to_voxels(points, voxel_size, point_cloud_range, max_points_per_voxel, max_num_voxels):
    """硬体素化。

    Args:
        points: (N, C) 点云, C 至少 4 维 (x, y, z, intensity)。
        voxel_size: (vx, vy, vz) 体素大小。
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max) 点云范围。
        max_points_per_voxel: 每个体素最多点数。
        max_num_voxels: 最多体素数。

    Returns:
        voxel_features: (M, T, C) 体素特征。
        voxel_coords: (M, 3) 体素坐标 (ix, iy, iz)。
        voxel_num_points: (M,) 每个体素的实际点数。
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size

    # 过滤范围外的点
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
            np.zeros((M, max_points_per_voxel, C), dtype=np.float32),
            np.zeros((M, 3), dtype=np.int32),
            np.zeros((M,), dtype=np.int32),
        )

    # 计算体素索引
    ix = np.floor((pts[:, 0] - x_min) / vx).astype(np.int32)
    iy = np.floor((pts[:, 1] - y_min) / vy).astype(np.int32)
    iz = np.floor((pts[:, 2] - z_min) / vz).astype(np.int32)

    grid_dx = int(round((x_max - x_min) / vx))
    grid_dy = int(round((y_max - y_min) / vy))
    voxel_keys = ix * grid_dy * grid_dx + iy * grid_dx + iz

    # 按体素 key 排序
    order = np.argsort(voxel_keys)
    sorted_keys = voxel_keys[order]
    sorted_pts = pts[order]

    # 找每个体素的起止位置
    diff = np.diff(sorted_keys)
    split_idx = np.where(diff != 0)[0] + 1
    starts = np.concatenate([[0], split_idx])
    ends = np.concatenate([split_idx, [len(sorted_keys)]])

    num_voxels = len(starts)
    C = pts.shape[1]

    voxel_features = np.zeros((num_voxels, max_points_per_voxel, C), dtype=np.float32)
    voxel_coords = np.zeros((num_voxels, 3), dtype=np.int32)
    voxel_num_points = np.zeros((num_voxels,), dtype=np.int32)

    for i in range(num_voxels):
        s, e = starts[i], ends[i]
        n = min(e - s, max_points_per_voxel)
        voxel_features[i, :n] = sorted_pts[s : s + n]
        voxel_num_points[i] = n
        p0 = sorted_pts[s]
        voxel_coords[i, 0] = int(np.floor((p0[0] - x_min) / vx))
        voxel_coords[i, 1] = int(np.floor((p0[1] - y_min) / vy))
        voxel_coords[i, 2] = int(np.floor((p0[2] - z_min) / vz))

    # 截断最多体素数
    if num_voxels > max_num_voxels:
        voxel_features = voxel_features[:max_num_voxels]
        voxel_coords = voxel_coords[:max_num_voxels]
        voxel_num_points = voxel_num_points[:max_num_voxels]

    return voxel_features, voxel_coords, voxel_num_points
