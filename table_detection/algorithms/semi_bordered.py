import functools
from typing import List

import cv2
import numpy as np

from table_detection.common import ROI, BBox
from table_detection.grid.gap_based import (GapBasedDetector,
                                            GapBasedDetectorConfig)
from table_detection.grid.line_layer import LineLayer, LineLayerConfig
from table_detection.preprocessing import convert_to_grayscale, create_frame

gap_detection_config_for_roi = GapBasedDetectorConfig(
    row_max_gap_break=3,
    col_max_gap_break=3,
    col_filter_non_gap=2,
    col_filter_gap=15,
    row_filter_non_gap=3,
    row_filter_gap=6,
)


def line_based_preprocessing(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    edges = cv2.Canny(img, 50, 150, 200)
    return convert_to_grayscale(~edges)


class LineBasedSearchConfig:
    def __init__(
        self,
        preprocessing=line_based_preprocessing,
        grid_recognizer_config=gap_detection_config_for_roi,
        line_layer_config=LineLayerConfig(),
        frame_size=0,
        filter_empty_cells=True,
    ):
        self.preprocessing = preprocessing
        self.grid_recognizer_config = grid_recognizer_config
        self.line_layer_config = line_layer_config
        self.frame_size = frame_size
        self.filter_empty_cells = filter_empty_cells


class LineBasedSearch:
    def __init__(self, config: LineBasedSearchConfig = LineBasedSearchConfig()):
        """
        Cell detection algorithm for semi-bordered tables. It simplifies task of cell
        detection by splitting table into regions of interest (ROI), then gap based
        detection is performed on each ROI individually, allowing to create more
        precise gap masks. ROI splitting is performed along detected lines.
        """
        self.config = config
        self.grid_detector = GapBasedDetector(self.config.grid_recognizer_config)
        self.line_layer = LineLayer(self.config.line_layer_config)

    def _filter_small_roi(self, roi_lists: List[List[ROI]]) -> List[List[ROI]]:
        return [
            [
                roi
                for roi in roi_list
                if all([dim > self.config.frame_size * 2 for dim in roi.shape])
            ]
            for roi_list in roi_lists
        ]

    def _check_box(self, box: BBox, preprocessed_img: np.ndarray) -> bool:
        roi = ROI.from_bbox(box)
        cv2.waitKey(0)
        bg_color = 255
        if any([d < 15 for d in box.shape]):
            return False
        if self.config.filter_empty_cells:
            img_cropped = create_frame(roi(preprocessed_img), 3, bg_color)
            return any(img_cropped.ravel() != bg_color)
        return True

    def _detect_boxes_in_roi_list(self, img: np.ndarray, roi_list: List[ROI]):
        boxes = []
        for roi in roi_list:
            cropped_img = roi(img)
            # Create white frame (inner padding) around selected area to remove dark
            # pixels on the ROI borders. May improve robustness of column
            # and row detection
            cropped_img = create_frame(cropped_img, self.config.frame_size, 255)
            roi_boxes = self.grid_detector.detect(cropped_img)
            boxes += [
                b.with_offset(roi.offset) for b in roi_boxes
                if self._check_box(b, cropped_img)
            ]
        return boxes

    @staticmethod
    def _filter_overlapping_boxes(
        primary_boxes: List[BBox], secondary_boxes: List[BBox]
    ) -> List[BBox]:
        return [
            p_box
            for p_box in primary_boxes
            if not any(
                [s_box.is_inside(p_box, threshold=0.1) for s_box in secondary_boxes]
            )
        ]

    def detect(self, img: np.ndarray) -> List[BBox]:
        prep_img = self.config.preprocessing(img)
        primary_roi_list, secondary_roi_list = self.line_layer.get_roi(prep_img)
        primary_roi_list, secondary_roi_list = self._filter_small_roi(
            [primary_roi_list, secondary_roi_list]
        )
        primary_boxes = self._detect_boxes_in_roi_list(prep_img, primary_roi_list)
        secondary_boxes = self._detect_boxes_in_roi_list(prep_img, secondary_roi_list)
        primary_boxes = self._filter_overlapping_boxes(primary_boxes, secondary_boxes)
        boxes = primary_boxes + secondary_boxes
        return boxes
