from typing import Optional, Tuple
import numpy as np


class CalibrationBoardDetector:
    """预留标定板检测器接口，输出AABB ROI"""
    def detect(self, board_point_cloud: np.ndarray) -> Tuple[bool, Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]]:
        raise NotImplementedError("Implement detect() for board detector")
