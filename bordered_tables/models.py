from dataclasses import dataclass, field
from tesseract.extractor import TextExtractor
from pathlib import Path
from typing import List, ClassVar, Tuple, Optional, Any, Dict

import numpy as np

TABLE_TAGS = ("Bordered", "Borderless")
CELL_TAG = 'Cell'


@dataclass(unsafe_hash=True)
class BorderBox:
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    bbox_id: int = field(init=False)

    def __post_init__(self):
        self.bbox_id = id(self)

    @property
    def box(self):
        return (
            self.top_left_x,
            self.top_left_y,
            self.bottom_right_x,
            self.bottom_right_y,
        )

    @property
    def width(self):
        return self.bottom_right_x - self.top_left_x

    @property
    def height(self):
        return self.bottom_right_y - self.top_left_y

    def merge(self, bb: "BorderBox") -> "BorderBox":
        return BorderBox(
            top_left_x=min(self.top_left_x, bb.top_left_x),
            top_left_y=min(self.top_left_y, bb.top_left_y),
            bottom_right_x=max(self.bottom_right_x, bb.bottom_right_x),
            bottom_right_y=max(self.bottom_right_y, bb.bottom_right_y),
        )

    def box_is_inside_another(self, bb2, threshold=0.9) -> bool:
        intersection_area, bb1_area, bb2_area = self.get_boxes_intersection_area(
            other_box=bb2
        )
        if intersection_area == 0:
            return False
        return any((intersection_area / bb) > threshold for bb in (bb1_area, bb2_area))

    def get_boxes_intersection_area(self, other_box) -> Tuple:
        bb1 = self.box
        bb2 = other_box.box
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0.0
        else:
            intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
        bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
        bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)
        return intersection_area, bb1_area, bb2_area

    def __getitem__(self, item):
        return self.box[item]


@dataclass(unsafe_hash=True)
class TextField:
    bbox: BorderBox
    text: str


@dataclass
class Cell(BorderBox):
    text_boxes: List[TextField] = field(default_factory=list)
    confidence: float = field(default=0.0)


@dataclass
class TableItem:
    bbox: BorderBox
    table_id: int
    objs: List = field(default_factory=list)

    def add(self, obj: BorderBox):
        self.objs.append(obj)
        self.bbox = self.bbox.merge(obj)


@dataclass
class Table:
    bbox: BorderBox
    table_id: int
    cols: List = field(default_factory=list)
    rows: List = field(default_factory=list)

    def is_box_from_table(self, box: BorderBox, threshold=0.99) -> bool:
        return self.bbox.box_is_inside_another(box, threshold)


@dataclass
class TableHeadered(Table):
    header: List[Cell] = field(default_factory=list)


@dataclass
class Row(TableItem):
    VERTICAL_MARGIN: ClassVar = 10

    def is_box_from_same_line(self, box: BorderBox):
        return (
            abs(self.bbox.top_left_y - box.top_left_y) <= self.VERTICAL_MARGIN
            and abs(self.bbox.bottom_right_y - box.bottom_right_y)
            <= self.VERTICAL_MARGIN
        )


@dataclass
class Column(TableItem):
    HORIZONTAL_MARGIN: ClassVar = 10

    def is_box_from_same_line(self, box: BorderBox):
        return (
            abs(self.bbox.top_left_x - box.top_left_x) <= self.HORIZONTAL_MARGIN
            and abs(self.bbox.bottom_right_x - box.bottom_right_x)
            <= self.HORIZONTAL_MARGIN
        )


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
            content_map=image.content_map
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
            box.top_left_x = np.round(box[0] * scale_x)
            box.top_left_y = np.round(box[1] * scale_y)
            box.bottom_right_x = np.round(box[2] * scale_x)
            box.bottom_right_y = np.round(box[3] * scale_y)

    def sort_boxes_topographically(self):
        self.boxes = sorted(self.bboxes, key=lambda x: (x.top_left_x, x.top_left_y))

    def find_tables_in_boxes(self, min_rows=2) -> Optional[List[Table]]:
        tables = []
        h_lines = {}
        v_lines = {}

        for box in sorted(self.objs, key=lambda x: (x.top_left_x, x.top_left_y)):
            for table in tables:
                if table.is_box_from_table(box):
                    target_table = table
                    break
            else:
                tables.append(Table(bbox=box, table_id=len(tables)))
                continue

            h_line_key = box[1]
            v_line_key = box[0]

            if h_line_key not in h_lines or h_lines[h_line_key].table_id != target_table.table_id:
                row = Row(
                    bbox=BorderBox(box[0], box[1], target_table.bbox[2], box[3]),
                    table_id=target_table.table_id,
                )
                row.add(box)
                target_table.rows.append(row)
                h_lines[h_line_key] = row
            else:
                h_lines[h_line_key].add(box)

            if v_line_key not in v_lines or v_lines[v_line_key].table_id != target_table.table_id:
                col = Column(
                    bbox=BorderBox(box[0], box[1], box[2], target_table.bbox[3]),
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
                            box.top_left_x, box.top_left_y,
                            box.width, box.height
                        )
                        self.content_map[box.bbox_id] = TextContent(
                            bbox=box, text=text, confidence=conf
                        )

@dataclass
class Page:
    tables: List[Table] = field(default_factory=list)
    text: List[TextField] = field(default_factory=list)


@dataclass
class InferenceTable:
    bbox: BorderBox
    tags: List[BorderBox] = field(default_factory=list)
    confidence: float = field(default=0.)
    label: str = field(default='')
    paddler: List[Cell] = field(default_factory=list)


def match_cells_and_tables(raw_cells: List[BorderBox], inference_tables: List[InferenceTable]) -> List[BorderBox]:
    cells_stack = raw_cells.copy()

    def check_inside_and_put(inf_table: InferenceTable, inf_cell: BorderBox):
        if inf_cell.box_is_inside_another(inf_table.bbox):
            inf_table.tags.append(inf_cell)
            return True
        return False

    not_matched_cells: List[BorderBox] = []
    while len(cells_stack) > 0:
        cell = cells_stack.pop()
        matched = False
        for table in inference_tables:
            if table.bbox.width * 0.7 < cell.width:
                not_matched_cells.append(cell)
                break
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
            filtered.append(
                max(candidates, key=lambda c: c.confidence)
            )
            for i in to_remove:
                not_matched_cells.append(i)
                stack.remove(i)
        table.tags = filtered

    return not_matched_cells


def inference_result_to_boxes(inference_page_result: List[Dict[str, Any]]) \
        -> Tuple[List[InferenceTable], List[BorderBox]]:
    raw_tables = [tag for tag in inference_page_result if tag['label'] in TABLE_TAGS]
    inference_tables: List[InferenceTable] = []
    for raw_table in raw_tables:
        if raw_table['score'] < 0.0:
            continue
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = raw_table['bbox']
        inference_tables.append(
            InferenceTable(
                bbox=BorderBox(
                    top_left_y=top_left_y,
                    top_left_x=top_left_x,
                    bottom_right_y=bottom_right_y,
                    bottom_right_x=bottom_right_x
                ),
                confidence=raw_table['score'],
                label=raw_table['label']
            )
        )

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
        filtered.append(
            max(candidates, key=lambda c: c.confidence)
        )
        for i in to_remove:
            stack.remove(i)

    raw_cells = [Cell(
        top_left_x=tag['bbox'][0],
        top_left_y=tag['bbox'][1],
        bottom_right_x=tag['bbox'][2],
        bottom_right_y=tag['bbox'][3],
        confidence=tag['score'],
    ) for tag in inference_page_result if tag['label'] in CELL_TAG]

    not_matched = match_cells_and_tables(raw_cells, filtered)

    return filtered, not_matched


@dataclass
class TextContent:
    bbox: BorderBox
    text: str
    confidence: int

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.text}"

    def __str__(self):
        return self.text
