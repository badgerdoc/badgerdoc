from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from table_detection.common import Axis
from table_detection.lines.lines import Line
from table_detection.lines.preprocessing import \
    default_line_detection_preprocessing
from table_detection.lines.utils import merge_lines, select_basis_lines


@dataclass
class HoughLinesDefaultArgs:
    rho: int = 1
    theta: float = np.pi / 180
    threshold: int = 50
    lines: Optional[List[Tuple[int, int, int, int]]] = None
    minLineLength: int = 50
    maxLineGap: int = 9


class BasisLinesDetectorConfig:
    def __init__(
        self,
        preprocessing_func: Callable[
            [np.ndarray], np.ndarray
        ] = default_line_detection_preprocessing,
        axis_alignment_tolerance=0.5,
        line_group_distance=19,
        line_group_gap=0.2,  # Percent of the length of the longest (of two) line
        mergeable_line_min_length=70,
        hough_lines_args=asdict(HoughLinesDefaultArgs()),
    ):
        self.preprocessing_func = preprocessing_func
        self.axis_alignment_tolerance = axis_alignment_tolerance
        self.line_group_distance = line_group_distance
        self.line_group_gap = line_group_gap
        self.mergeable_line_min_length = mergeable_line_min_length
        self.hough_lines_args = hough_lines_args


class BasisLinesDetector:
    def __init__(self, config: BasisLinesDetectorConfig = BasisLinesDetectorConfig()):
        """Line detector used to find lines aligned with axis."""
        self.config = config

    def detect(self, img: np.ndarray):
        prep_img = self.config.preprocessing_func(img)
        lines_p = cv2.HoughLinesP(prep_img, **self.config.hough_lines_args)
        lines = [Line.from_coords(line[0]) for line in lines_p]
        lines = [
            line
            for line in lines
            if line.vector.length > self.config.mergeable_line_min_length
        ]
        h_lines, v_lines = select_basis_lines(
            lines, self.config.axis_alignment_tolerance
        )
        h_lines = merge_lines(
            h_lines,
            Axis.Y,
            self.config.line_group_distance,
            self.config.line_group_gap,
            self.config.axis_alignment_tolerance,
        )
        v_lines = merge_lines(
            v_lines,
            Axis.X,
            self.config.line_group_distance,
            self.config.line_group_gap,
            self.config.axis_alignment_tolerance,
        )
        # TODO: text provides a lot of noise for line detection
        # try masking text before detection

        # TODO: consider other methods of filtration or move constant to config
        noise_culling = 0.33
        h_lines = [
            line
            for line in h_lines
            if line.vector.length > img.shape[1] * noise_culling
        ]
        v_lines = [
            line
            for line in v_lines
            if line.vector.length > img.shape[0] * noise_culling
        ]
        return h_lines, v_lines
