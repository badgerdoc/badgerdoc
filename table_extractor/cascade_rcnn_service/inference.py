import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from mmdet.apis import inference_detector, init_detector

from table_extractor.bordered_service.models import (
    InferenceTable,
    match_cells_and_tables,
)
from table_extractor.cascade_rcnn_service.utils import (
    extract_boxes_from_result,
    has_image_extension,
)
from table_extractor.model.table import BorderBox, Cell

CLASS_NAMES = ("table", "Cell", "header")
DEFAULT_THRESHOLD = 0.3
TABLE_TAGS = "table"
CELL_TAG = "Cell"
logger = logging.getLogger(__name__)


def _filter_double_detection(inference_tables: List[InferenceTable]):
    filtered = []
    stack = inference_tables.copy()
    while len(stack) > 0:
        table = stack.pop()
        to_remove = []
        candidates = [table]
        for i in range(len(stack)):
            other_table = stack[i]
            if table.bbox.box_is_inside_another(other_table.bbox):
                candidates.append(other_table)
                to_remove.append(other_table)
        filtered.append(max(candidates, key=lambda c: c.confidence))
        for i in to_remove:
            stack.remove(i)
    return filtered


def _raw_to_table(raw_table: Dict[str, Any]) -> InferenceTable:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = raw_table["bbox"]
    return InferenceTable(
        bbox=BorderBox(
            top_left_y=top_left_y,
            top_left_x=top_left_x,
            bottom_right_y=bottom_right_y,
            bottom_right_x=bottom_right_x,
        ),
        confidence=raw_table["score"],
        label=raw_table["label"],
    )


def _raw_to_cell(raw_cell: Dict[str, Any]) -> Cell:
    return Cell(
        top_left_x=raw_cell["bbox"][0],
        top_left_y=raw_cell["bbox"][1],
        bottom_right_x=raw_cell["bbox"][2],
        bottom_right_y=raw_cell["bbox"][3],
        confidence=raw_cell["score"],
    )


def inference_result_to_boxes(
    inference_page_result: List[Dict[str, Any]]
) -> Tuple[List[InferenceTable], List[Cell], List[BorderBox]]:
    raw_tables = [
        tag for tag in inference_page_result if tag["label"] == TABLE_TAGS
    ]
    raw_headers = [
        _raw_to_cell(tag)
        for tag in inference_page_result
        if tag["label"] == "header"
    ]
    inference_tables: List[InferenceTable] = [
        _raw_to_table(raw_table) for raw_table in raw_tables
    ]

    filtered = _filter_double_detection(inference_tables)

    raw_cells = [
        _raw_to_cell(cell)
        for cell in inference_page_result
        if cell["label"] == CELL_TAG
    ]

    not_matched = match_cells_and_tables(raw_cells, filtered)

    return filtered, raw_headers, not_matched


class CascadeRCNNInferenceService:
    def __init__(
        self, config: Path, model: Path, should_visualize: bool = False
    ):
        self.model = init_detector(
            str(config.absolute()), str(model.absolute()), device="cpu"
        )
        self.should_visualize = should_visualize

    def inference_image(
        self, img: Path, threshold: float = DEFAULT_THRESHOLD
    ):
        if not has_image_extension(img):
            logger.warning(f"Not image {img}")
            return
        logger.info(f"Cascade inference image {img}")
        result = inference_detector(self.model, img)
        if self.should_visualize:
            inference_image = self.model.show_result(
                img, result, thickness=2
            )
            image_path = img.parent.parent / "raw_model" / img.name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path.absolute()), inference_image)
        inf_tables, headers, _ = inference_result_to_boxes(
            extract_boxes_from_result(result, CLASS_NAMES, score_thr=threshold)
        )
        return inf_tables, headers
