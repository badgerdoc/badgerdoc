from typing import List, Optional, Tuple

import numpy as np
from sortedcontainers import SortedSet

from table_detection.common import ROI, BBox
from table_detection.grid.utils import bboxes_from_coordinates
from table_detection.lines.detector import (BasisLinesDetector,
                                            BasisLinesDetectorConfig)
from table_detection.lines.lines import Line


class LineLayerConfig:
    def __init__(
        self,
        detector_config=BasisLinesDetectorConfig(),
        primary_line_threshold=0.85,
    ):
        self.detector_config = detector_config
        self.primary_line_threshold = primary_line_threshold


class LineLayer:
    def __init__(self, config: LineLayerConfig):
        """Collection of axis aligned lines on the image."""
        self.detector = BasisLinesDetector(config.detector_config)
        self.config = config
        self.horizontal: Optional[List[Line]] = None
        self.vertical: Optional[List[Line]] = None

    def _detect_lines(self, img: np.ndarray):
        self.horizontal, self.vertical = self.detector.detect(img)

    @staticmethod
    def _group_lines_by_size(lines: List[Line], threshold: float):
        primary_lines = []
        secondary_lines = []
        for line in lines:
            (
                primary_lines if line.vector.length > threshold else secondary_lines
            ).append(line)
        secondary_lines.sort(key=lambda line: line.vector.length, reverse=True)
        return primary_lines, secondary_lines

    def get_roi(self, img: np.ndarray) -> Tuple[List[ROI], List[ROI]]:
        self._detect_lines(img)
        height, width = img.shape[:2]
        if not self.horizontal and not self.vertical:
            # Single primary ROI equal to the given image
            return [ROI(img.shape[:2], (0, 0))], []
        row_primary_lines, row_secondary_lines = self._group_lines_by_size(
            self.horizontal, self.config.primary_line_threshold * width
        )
        col_primary_lines, col_secondary_lines = self._group_lines_by_size(
            self.vertical, self.config.primary_line_threshold * height
        )
        # FIXME: vertical lines are causing a lot of issues with overlapping ROI
        # so they are excluded for now, need to implement better algorithm for
        # box selection
        col_secondary_lines = []
        col_centers = SortedSet(
            [line.center.x for line in col_primary_lines] + [0, width]
        )
        row_centers = SortedSet(
            [line.center.y for line in row_primary_lines] + [0, height]
        )
        primary_bboxes = bboxes_from_coordinates(col_centers, row_centers)
        secondary_bboxes = []
        for line in row_secondary_lines:
            x1, y1, x2, y2 = line.coords
            yc = line.center.y
            row_centers.add(yc)
            yc_idx = row_centers.index(yc)
            if yc_idx == 0:
                secondary_bboxes.append(BBox(x1, yc, x2, row_centers[1]))
            elif yc_idx >= len(row_centers) - 1:
                secondary_bboxes.append(BBox(x1, row_centers[yc_idx - 1], x2, y2))
            else:
                secondary_bboxes.append(BBox(x1, row_centers[yc_idx - 1], x2, yc))
                secondary_bboxes.append(BBox(x1, yc, x2, row_centers[yc_idx + 1]))

        for line in col_secondary_lines:
            x1, y1, x2, y2 = line.coords
            xc = line.center.x
            col_centers.add(xc)
            xc_idx = col_centers.index(xc)
            if xc_idx == 0:
                secondary_bboxes.append(BBox(xc, y1, col_centers[1], y2))
            elif xc_idx >= len(col_centers) - 1:
                secondary_bboxes.append(BBox(col_centers[xc_idx - 1], y1, x2, y2))
            else:
                secondary_bboxes.append(BBox(col_centers[xc_idx - 1], y1, xc, y2))
                secondary_bboxes.append(BBox(xc, y1, col_centers[xc_idx + 1], y2))

        return (
            [ROI.from_bbox(bbox) for bbox in primary_bboxes],
            [ROI.from_bbox(bbox) for bbox in secondary_bboxes],
        )
