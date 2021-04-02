import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
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

CLASS_NAMES = ("Bordered", "Cell", "Borderless", "Header", "Table_annotation")
DEFAULT_THRESHOLD = 0.3
TABLE_TAGS = ("Bordered", "Borderless")
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
        tag for tag in inference_page_result if tag["label"] in TABLE_TAGS
    ]
    raw_headers = [
        _raw_to_cell(tag)
        for tag in inference_page_result
        if tag["label"] == "Header"
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

    if len(raw_cells) > 20 and not inference_tables:
        filtered.append(
            InferenceTable(
                bbox=BorderBox(
                    top_left_y=min([cell.top_left_y for cell in raw_cells])
                    - 50,
                    top_left_x=min([cell.top_left_x for cell in raw_cells])
                    - 50,
                    bottom_right_y=max(
                        [cell.bottom_right_y for cell in raw_cells]
                    )
                    + 50,
                    bottom_right_x=max(
                        [cell.bottom_right_x for cell in raw_cells]
                    )
                    + 50,
                ),
                confidence=0.5,
                label="Borderless",
                tags=raw_cells,
            )
        )
    if len(not_matched) > 20:
        filtered.append(
            InferenceTable(
                bbox=BorderBox(
                    top_left_y=min([cell.top_left_y for cell in not_matched])
                    - 50,
                    top_left_x=min([cell.top_left_x for cell in not_matched])
                    - 50,
                    bottom_right_y=max(
                        [cell.bottom_right_y for cell in not_matched]
                    )
                    + 50,
                    bottom_right_x=max(
                        [cell.bottom_right_x for cell in not_matched]
                    )
                    + 50,
                ),
                confidence=0.5,
                label="Borderless",
                tags=not_matched,
            )
        )

    return filtered, raw_headers, not_matched


def add_padding(img: Union[Path, np.ndarray], padding: int):
    if isinstance(img, Path):
        img_arr = cv2.imread(str(img.absolute()))
    else:
        img_arr = img
    new_shape = (
        img_arr.shape[0] + 2 * padding,
        img_arr.shape[1] + 2 * padding,
        img_arr.shape[2],
    )
    new_img = np.ones(shape=new_shape) * 255
    new_img[
        padding : img_arr.shape[0] + padding,
        padding : img_arr.shape[1] + padding,
        :,
    ] = img_arr
    return new_img


def shift(
    inf_tables: List[InferenceTable],
    headers: List[Cell],
    pic_shift: int,
    padding: int,
):
    for table in inf_tables:
        table.bbox.top_left_y = table.bbox.top_left_y + pic_shift
        table.bbox.top_left_x = table.bbox.top_left_x + padding
        table.bbox.bottom_right_y = table.bbox.bottom_right_y + pic_shift
        table.bbox.bottom_right_x = table.bbox.bottom_right_x + padding
        for cell in table.tags:
            cell.top_left_y = cell.top_left_y + pic_shift
            cell.top_left_x = cell.top_left_x + padding
            cell.bottom_right_y = cell.bottom_right_y + pic_shift
            cell.bottom_right_x = cell.bottom_right_x + padding
    for header in headers:
        header.top_left_y = header.top_left_y + pic_shift
        header.top_left_x = header.top_left_x + padding
        header.bottom_right_y = header.bottom_right_y + pic_shift
        header.bottom_right_x = header.bottom_right_x + padding


def crop_padding(
    inf_results: List[Tuple[List[InferenceTable], List[Cell]]], padding: int
):
    for inf_tables, headers in inf_results:
        for table in inf_tables:
            table.bbox.top_left_x = table.bbox.top_left_x - padding
            table.bbox.top_left_y = table.bbox.top_left_y - padding
            table.bbox.bottom_right_x = table.bbox.bottom_right_x - padding
            table.bbox.bottom_right_y = table.bbox.bottom_right_y - padding
            for cell in table.tags:
                cell.top_left_x = cell.top_left_x - padding
                cell.top_left_y = cell.top_left_y - padding
                cell.bottom_right_x = cell.bottom_right_x - padding
                cell.bottom_right_y = cell.bottom_right_y - padding
        for header in headers:
            header.top_left_x = header.top_left_x - padding
            header.top_left_y = header.top_left_y - padding
            header.bottom_right_x = header.bottom_right_x - padding
            header.bottom_right_y = header.bottom_right_y - padding


class CascadeRCNNInferenceService:
    def __init__(
        self, config: Path, model: Path, should_visualize: bool = False
    ):
        self.model = init_detector(
            str(config.absolute()), str(model.absolute()), device="cpu"
        )
        self.should_visualize = should_visualize

    def inference_split(
        self, img: Path, threshold: float = DEFAULT_THRESHOLD, padding: int = 0
    ):
        image = cv2.imread(str(img.absolute()))
        split_weight = [0.2, 0.43, 0.65, 0.84, 1]
        splits = []
        prev = 0
        for split_w in split_weight:
            border = int(split_w * image.shape[0])
            splits.append(image[prev:border])
            prev = border

        tables = []
        headers = []

        prev_shape = 0
        for num, split in enumerate(splits):
            split = add_padding(split, padding)
            result = inference_detector(self.model, split)
            if self.should_visualize:
                inference_image = self.model.show_result(
                    split, result, thickness=2
                )
                image_path = (
                    img.parent.parent / "raw_model" / img.name / f"{num}.png"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path.absolute()), inference_image)
            inf_tables, header, _ = inference_result_to_boxes(
                extract_boxes_from_result(
                    result, CLASS_NAMES, score_thr=threshold
                )
            )
            crop_padding([(inf_tables, headers)], padding)
            shift(inf_tables, headers, prev_shape, 0)
            tables.extend(inf_tables)
            headers.extend(header)
            prev_shape += split.shape[0]
        return tables, headers

    def inference_image(
        self, img: Path, threshold: float = DEFAULT_THRESHOLD, padding: int = 0
    ):
        if not has_image_extension(img):
            logger.warning(f"Not image {img}")
            return
        logger.info(f"Cascade inference image {img}")
        if padding:
            img_arr = add_padding(img, padding)
            if img_arr.shape[0] > 12000:
                return self.inference_split(img, threshold, padding)
            result = inference_detector(self.model, img_arr)
            if self.should_visualize:
                inference_image = self.model.show_result(
                    img_arr, result, thickness=2
                )
                image_path = img.parent.parent / "raw_model" / img.name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path.absolute()), inference_image)
            inf_tables, headers, _ = inference_result_to_boxes(
                extract_boxes_from_result(
                    result, CLASS_NAMES, score_thr=threshold
                )
            )
            crop_padding([(inf_tables, headers)], padding)
            return inf_tables, headers
        else:
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
