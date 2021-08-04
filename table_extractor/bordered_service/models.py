from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from table_extractor.model.table import (
    BorderBox,
    Cell,
    Column,
    Row,
    StructuredTable,
    Table,
    TextField, StructuredTableHeadered,
)
from table_extractor.tesseract_service.tesseract_extractor import TextExtractor

TABLE_TAGS = ("Bordered", "Borderless")
CELL_TAG = "Cell"


@dataclass
class ImageDTO:
    name: str
    tables: List
    content_map: Dict

    @classmethod
    def from_image(cls, image: "Image"):
        return cls(
            name=image.path.name,
            tables=image.tables,
            content_map=image.content_map,
        )


@dataclass
class Image:
    path: Path
    shape: Optional[Tuple] = None
    pdf_page_shape: List = field(default_factory=list)
    objs: List = field(default_factory=list)
    bboxes: List[BorderBox] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    content_map: Optional[Dict] = field(default_factory=dict)

    def scale_bboxes(self):
        scale_x = self.pdf_page_shape[0] / self.shape[1]
        scale_y = self.pdf_page_shape[1] / self.shape[0]

        for box in self.objs:
            box.top_left_x = int(np.round(box[0] * scale_x))
            box.top_left_y = int(np.round(box[1] * scale_y))
            box.bottom_right_x = int(np.round(box[2] * scale_x))
            box.bottom_right_y = int(np.round(box[3] * scale_y))

    def sort_boxes_topographically(self):
        self.boxes = sorted(
            self.bboxes, key=lambda x: (x.top_left_x, x.top_left_y)
        )

    def find_tables_in_boxes(self, min_rows=2) -> Optional[List[Table]]:
        tables = []
        h_lines = {}
        v_lines = {}

        for box in sorted(
            self.objs, key=lambda x: (x.top_left_x, x.top_left_y)
        ):
            for table in tables:
                if table.is_box_from_table(box):
                    target_table = table
                    break
            else:
                tables.append(Table(bbox=box, table_id=len(tables)))
                continue

            h_line_key = box[1]
            v_line_key = box[0]

            if (
                h_line_key not in h_lines
                or h_lines[h_line_key].table_id != target_table.table_id
            ):
                row = Row(
                    bbox=BorderBox(
                        box[0], box[1], target_table.bbox[2], box[3]
                    ),
                    table_id=target_table.table_id,
                )
                row.add(box)
                target_table.rows.append(row)
                h_lines[h_line_key] = row
            else:
                h_lines[h_line_key].add(box)

            if (
                v_line_key not in v_lines
                or v_lines[v_line_key].table_id != target_table.table_id
            ):
                col = Column(
                    bbox=BorderBox(
                        box[0], box[1], box[2], target_table.bbox[3]
                    ),
                    table_id=target_table.table_id,
                )
                col.add(box)
                target_table.cols.append(col)
                v_lines[v_line_key] = col
            else:
                v_lines[v_line_key].add(box)

        res = [i for i in tables if len(i.rows) >= min_rows]

        return res if res else None

    def analyze(self):
        self.sort_boxes_topographically()
        self.tables = self.find_tables_in_boxes()

    def extract_text(self):
        if not self.tables:
            return

        with TextExtractor(str(self.path.absolute())) as te:
            for table in self.tables:
                for row in table.rows:
                    for box in row.objs:
                        text, conf = te.extract(
                            box.top_left_x,
                            box.top_left_y,
                            box.width,
                            box.height,
                        )
                        self.content_map[box.bbox_id] = TextContent(
                            bbox=box, text=text, confidence=conf
                        )


@dataclass
class Page:
    page_num: int
    bbox: BorderBox
    tables: List[Union[StructuredTable, StructuredTableHeadered]] = field(default_factory=list)
    text: List[TextField] = field(default_factory=list)

    @property
    def blocks(self) -> List[Union[StructuredTable, TextField]]:
        return sorted(self.tables + self.text, key=lambda x: x.bbox.top_left_y)

    @classmethod
    def from_dict(cls, d):
        blocks = d.get('blocks', [])
        tables = [block for block in blocks if block['type'] == 'table']
        # ToDo: include or replace with DoD
        text_blocks = []

        return cls(
            page_num=int(d['page_num']),
            bbox=BorderBox.from_dict(d['bbox']),
            tables=[StructuredTableHeadered.from_dict(t) for t in tables],
            text=text_blocks,
        )


@dataclass
class InferenceTable:
    bbox: BorderBox
    tags: List[Cell] = field(default_factory=list)
    confidence: float = field(default=0.0)
    label: str = field(default="")
    paddler: List[Cell] = field(default_factory=list)
    header_boxes: List[Cell] = field(default_factory=list)


def match_cells_and_tables(
    raw_cells: List[BorderBox], inference_tables: List[InferenceTable]
) -> List[BorderBox]:
    cells_stack = raw_cells.copy()

    def check_inside_and_put(inf_table: InferenceTable, inf_cell: BorderBox):
        if inf_cell.box_is_inside_another(inf_table.bbox, 0.05):
            inf_table.tags.append(inf_cell)
            return True
        return False

    not_matched_cells: List[BorderBox] = []
    while len(cells_stack) > 0:
        cell = cells_stack.pop()
        matched = False
        for table in inference_tables:
            matched = matched or check_inside_and_put(table, cell)
            if matched:
                break
        if not matched:
            not_matched_cells.append(cell)

    for table in inference_tables:
        filtered = []
        stack = table.tags.copy()
        while len(stack) > 0:
            cell = stack.pop()
            to_remove = []
            candidates = [cell]
            for i in range(len(stack)):
                other_cell = stack[i]
                if cell.box_is_inside_another(other_cell):
                    candidates.append(other_cell)
                    to_remove.append(other_cell)
            filtered.append(max(candidates, key=lambda c: c.confidence))
            for i in to_remove:
                not_matched_cells.append(i)
                stack.remove(i)
        table.tags = filtered
        if table.tags:
            table.bbox.top_left_y = min(table.bbox.top_left_y,
                                        min([cell.top_left_y for cell in table.tags]) - 10,)
            table.bbox.top_left_x = min(table.bbox.top_left_x,
                                        min([cell.top_left_x for cell in table.tags]) - 10,)
            table.bbox.bottom_right_y = max(table.bbox.bottom_right_y,
                                            max([cell.bottom_right_y for cell in table.tags]) + 10,)
            table.bbox.bottom_right_x = max(table.bbox.bottom_right_x,
                                            max([cell.bottom_right_x for cell in table.tags]) + 10,)

    return not_matched_cells


def match_headers_and_tables(
    headers: List[Cell], inference_tables: List[InferenceTable]
) -> List[BorderBox]:
    headers_stack = headers.copy()

    def check_inside_and_put(inf_table: InferenceTable, inf_header: BorderBox):
        if inf_header.box_is_inside_another(inf_table.bbox):
            inf_table.header_boxes.append(inf_header)
            return True
        return False

    not_matched_cells: List[BorderBox] = []
    while len(headers_stack) > 0:
        cell = headers_stack.pop()
        matched = False
        for table in inference_tables:
            matched = matched or check_inside_and_put(table, cell)
            if matched:
                break
        if not matched:
            not_matched_cells.append(cell)

    return not_matched_cells


@dataclass
class TextContent:
    bbox: BorderBox
    text: str
    confidence: int

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.text}"

    def __str__(self):
        return self.text
