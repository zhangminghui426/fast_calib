"""数据预处理: PCD 读取、标注解析。"""

import os
import numpy as np


def read_pcd_bin(path: str) -> np.ndarray:
    """读取二进制 PCD 文件, 返回 (N, 4) [x, y, z, intensity]。"""
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


def read_label_txt(path: str) -> np.ndarray:
    """读取标注文件, 返回 (cx, cy, cz, w, l, h, yaw)。"""
    with open(path, "r") as f:
        line = f.readline().strip()
    return np.array([float(x) for x in line.split()], dtype=np.float32)


def prepare_data(raw_dir: str, out_dir: str):
    """把原始 PCD 和标注转成训练用的 bin 和 txt。"""
    os.makedirs(os.path.join(out_dir, "pcd"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "label"), exist_ok=True)

    # 这里是示例, 实际根据数据格式修改
    print(f"Preparing data from {raw_dir} to {out_dir}")
