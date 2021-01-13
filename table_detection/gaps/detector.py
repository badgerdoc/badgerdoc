from dataclasses import dataclass
from typing import List

import numpy as np

from table_detection.common import Axis
from table_detection.gaps.gaps import GapMask


@dataclass
class GapDetector:
    """
    :param: direction: an axis which sliding window aligned with,
        when set to Y - vertical window slides along X axis and vice versa
    :param: background: color of background
    :param: max_gap_break: if break between two gaps is less than this value, they
        considered a one gap
    """

    direction: Axis
    background: int
    max_gap_break: int

    def find_gaps(self, img: np.ndarray):
        gaps = []
        for i in range(img.shape[self.direction]):
            # 1d sliding window
            y_slice, x_slice = (
                (slice(None), slice(i, i + 1))
                if self.direction is Axis.X
                else (slice(i, i + 1), slice(None))
            )
            window = img[y_slice, x_slice].ravel()
            gaps.append(self._is_gap(window))
        return GapMask(gaps, self.direction)

    def _is_gap(self, window: List[float]):
        length = 0
        for value in window:
            if value != self.background:
                length += 1
                if length > self.max_gap_break:
                    return False
            else:
                length = 0
        return True
