import logging
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy

from table_extractor.bordered_service.models import InferenceTable
from table_extractor.cascade_rcnn_service.utils import has_image_extension
from table_extractor.model.table import StructuredTable, BorderBox, TextField, Table, StructuredTableHeadered, \
    CellLinked

LOGGER = logging.getLogger(__name__)


TEXT_THICKNESS = 2

INFERENCE_THICKNESS = 3

INFERENCE_COLOR = (255, 0, 0)

HEADER_CELL_COLOR = (0, 128, 128)

CELL_WITHOUT_TEXT_COLOR = (0, 128, 128)

CELL_THICKNESS = 5

CELL_WITH_TEXT_COLOR = (0, 0, 255)


def _draw_rectangle(color: Tuple[int, int, int], thickness: int, img: numpy.ndarray, bbox: BorderBox):
    cv2.rectangle(img,
                  (int(bbox.top_left_x), int(bbox.top_left_y)),
                  (int(bbox.bottom_right_x), int(bbox.bottom_right_y)),
                  color,
                  thickness)


def draw_text_boxes(img: numpy.ndarray, text_fields: List[TextField]):
    for text_field in text_fields:
        text_box_color = (0, 255, 0)
        text_box_thickness = 3
        _draw_rectangle(text_box_color, text_box_thickness, img, text_field.bbox)


def draw_cell_scores(img: numpy.ndarray, cells_scores: List[Tuple[CellLinked, float, float]]):
    text_box_thickness = 3
    for cell, header_score, non_header_score in cells_scores:
        if header_score > non_header_score:
            _draw_rectangle(HEADER_CELL_COLOR, text_box_thickness, img, cell)
        else:
            _draw_rectangle(CELL_WITH_TEXT_COLOR, 1, img, cell)


def draw_inference(img: numpy.ndarray, inference_result: List[InferenceTable]):
    for inference_table in inference_result:
        _draw_rectangle(INFERENCE_COLOR, INFERENCE_THICKNESS, img, inference_table.bbox)
        cv2.putText(img,
                    f"{inference_table.label}: {inference_table.confidence}",
                    (inference_table.bbox[0], inference_table.bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    INFERENCE_COLOR,
                    TEXT_THICKNESS)
        for box in inference_table.tags:
            _draw_rectangle(INFERENCE_COLOR, INFERENCE_THICKNESS, img, box)


def draw_table(img: numpy.ndarray, tables: List[Table]):
    for table in tables:
        for row in table.rows:
            for obj in row.objs:
                if obj.text_boxes:
                    _draw_rectangle(CELL_WITH_TEXT_COLOR, CELL_THICKNESS, img, obj)
                else:
                    _draw_rectangle(CELL_WITHOUT_TEXT_COLOR, CELL_THICKNESS, img, obj)


def draw_structured_table(img: numpy.ndarray, tables: List[StructuredTable]):
    for table in tables:
        for cell in table.cells:
            if cell.text_boxes:
                _draw_rectangle(CELL_WITH_TEXT_COLOR, CELL_THICKNESS, img, cell)
            else:
                _draw_rectangle(CELL_WITHOUT_TEXT_COLOR, CELL_THICKNESS, img, cell)


def draw_structured_table_headered(img: numpy.ndarray, tables: List[StructuredTableHeadered]):
    for table in tables:
        for cell in table.cells:
            if cell.text_boxes:
                _draw_rectangle(CELL_WITH_TEXT_COLOR, CELL_THICKNESS, img, cell)
            else:
                _draw_rectangle(CELL_WITHOUT_TEXT_COLOR, CELL_THICKNESS, img, cell)
        for row in table.header:
            for cell in row:
                _draw_rectangle(HEADER_CELL_COLOR, CELL_THICKNESS, img, cell)


def _check_and_create_path(output_path: Path):
    if not output_path and not has_image_extension(output_path):
        raise ValueError("Incorrect path provided")
    output_path.parent.mkdir(parents=True, exist_ok=True)


def draw_object(img, obj: Union[List[Table], List[InferenceTable], List[TextField], List[StructuredTable]]):
    if not obj or not isinstance(obj, List) or img is None:
        return img
    img = img.copy()
    if isinstance(obj[0], Table):
        draw_table(img, obj)
    elif isinstance(obj[0], InferenceTable):
        draw_inference(img, obj)
    elif isinstance(obj[0], StructuredTableHeadered):
        draw_structured_table_headered(img, obj)
    elif isinstance(obj[0], StructuredTable):
        draw_structured_table(img, obj)
    elif isinstance(obj[0], TextField):
        draw_text_boxes(img, obj)
    elif isinstance(obj[0], Tuple):
        draw_cell_scores(img, obj)
    else:
        raise ValueError(f"Unsupported type for visualization: {type(obj[0])}")
    return img


class TableVisualizer:
    def __init__(self, should_visualize: bool):
        self.should_visualize = should_visualize

    def draw_object_and_save(self,
                             img: numpy.ndarray,
                             obj: Union[List[Table], List[InferenceTable], List[TextField],
                                        List[StructuredTable], List[StructuredTableHeadered]],
                             output_path: Path):
        if not self.should_visualize:
            return
        if img is None:
            raise ValueError("Image is None")
        if not obj:
            LOGGER.warning("Object to draw wasn't provided")
            return
        _check_and_create_path(output_path)
        img = draw_object(img, obj)
        cv2.imwrite(str(output_path.absolute()), img)
