"""PCD 文件读取工具。"""

import numpy as np


def read_pcd(path: str) -> np.ndarray:
    """读取 ASCII PCD 文件, 返回 (N, 4) [x, y, z, intensity]。"""
    with open(path, "r") as f:
        lines = f.readlines()

    # 找 DATA 行
    data_idx = None
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            data_idx = i + 1
            break

    if data_idx is None:
        raise ValueError(f"No DATA section in {path}")

    # 读取点云数据
    points = []
    for line in lines[data_idx:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            x, y, z, intensity = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            points.append([x, y, z, intensity])

    return np.array(points, dtype=np.float32)
