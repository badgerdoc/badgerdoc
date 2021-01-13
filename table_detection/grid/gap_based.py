from typing import List, Tuple

import numpy as np
from sortedcontainers import SortedSet

from table_detection.common import Axis, BBox
from table_detection.gaps.detector import GapDetector
from table_detection.gaps.gaps import GapMask
from table_detection.grid.utils import bboxes_from_coordinates
from table_detection.preprocessing import to_binary_img


class GapBasedDetectorConfig:
    def __init__(
        self,
        row_max_gap_break=3,
        col_max_gap_break=3,
        col_filter_non_gap=7,
        col_filter_gap=21,
        row_filter_non_gap=7,
        row_filter_gap=7,
    ):
        """
        :param row_max_gap_break: the number of non-background pixels, when this number
            is exceeded, search for the next gap is initiated.
        :param col_max_gap_break: the number of non-background pixels, when this number
            is exceeded, search for the next gap is initiated.
        :param col_filter_non_gap: threshold value for horizontal non-gap sequences, if
            length of the sequence is below threshold, this sequence is considered
            noise and merged into closest gaps
        :param col_filter_gap: threshold value for horizontal gap sequences, if
            length of the sequence is below threshold, this sequence is considered
            noise and merged into closest non-gaps
        :param row_filter_non_gap: threshold value for vertical non-gap sequences, if
            length of the sequence is below threshold, this sequence is considered
            noise and merged into closest gaps
        :param row_filter_gap: threshold value for vertical gap sequences, if
            length of the sequence is below threshold, this sequence is considered
            noise and merged into closest non-gaps
        """
        self.row_max_gap_break = row_max_gap_break
        self.col_max_gap_break = col_max_gap_break
        self.col_filter_non_gap = col_filter_non_gap
        self.col_filter_gap = col_filter_gap
        self.row_filter_non_gap = row_filter_non_gap
        self.row_filter_gap = row_filter_gap


class GapBasedDetector:
    """
    The simplest grid detection algorithm for borderless and semi-bordered tables.
    Achieves the best performance on tables, that do not contain hierarchical structures
    (e.g. column having sub-columns).
    In order to detect cells in complex tables with this algorithm, these tables should
    be split into simple homogeneous regions and then detection should be performed on
    each region independently.
    """

    def __init__(self, config: GapBasedDetectorConfig = GapBasedDetectorConfig()):
        self.config = config

    def _apply_gap_filters(
        self, col_mask_1d: GapMask, row_mask_1d: GapMask
    ) -> Tuple[GapMask, GapMask]:
        col_mask_1d = col_mask_1d.filter(self.config.col_filter_non_gap, False)
        col_mask_1d = col_mask_1d.filter(self.config.col_filter_gap, True)
        row_mask_1d = row_mask_1d.filter(self.config.row_filter_non_gap, False)
        row_mask_1d = row_mask_1d.filter(self.config.row_filter_gap, True)
        return col_mask_1d, row_mask_1d

    def detect(self, img: np.ndarray) -> List[BBox]:
        img_bin = to_binary_img(img)
        h_gap_det = GapDetector(Axis.X, 255, self.config.col_max_gap_break)
        v_gap_det = GapDetector(Axis.Y, 255, self.config.row_max_gap_break)
        col_mask_1d = h_gap_det.find_gaps(img_bin)
        row_mask_1d = v_gap_det.find_gaps(img_bin)
        col_mask_1d, row_mask_1d = self._apply_gap_filters(col_mask_1d, row_mask_1d)
        col_centers = SortedSet(col_mask_1d.centers)
        row_centers = SortedSet(row_mask_1d.centers)

        for value in {min(col_centers), max(col_centers)}:
            col_centers.remove(value)
        col_centers.add(0)
        col_centers.add(img.shape[1])

        for value in {min(row_centers), max(row_centers)}:
            row_centers.remove(value)
        row_centers.add(0)
        row_centers.add(img.shape[0])
        bboxes = bboxes_from_coordinates(col_centers, row_centers)
        return bboxes
