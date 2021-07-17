from dataclasses import dataclass, field
from functools import reduce
from typing import ClassVar, List, Tuple


@dataclass
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
        (
            intersection_area,
            bb1_area,
            bb2_area,
        ) = self.get_boxes_intersection_area(other_box=bb2)
        if intersection_area == 0:
            return False
        return any(
            (intersection_area / bb) > threshold for bb in (bb1_area, bb2_area)
        )

    def box_is_inside_box(self, bb2, threshold=0.95) -> bool:
        (
            intersection_area,
            bb1_area,
            bb2_area,
        ) = self.get_boxes_intersection_area(other_box=bb2)
        if intersection_area == 0:
            return False
        return (intersection_area / bb1_area) > threshold

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

    @classmethod
    def from_dict(cls, d):
        return cls(
            top_left_x=d.get('left', 0),
            top_left_y=d.get('top', 0),
            bottom_right_x=d.get('left', 0) + d.get('width', 0),
            bottom_right_y=d.get('top', 0) + d.get('height', 0),
        )


@dataclass(unsafe_hash=True)
class TextField:
    bbox: BorderBox
    text: str


@dataclass
class Cell(BorderBox):
    text_boxes: List[TextField] = field(default_factory=list)
    confidence: float = field(default=0.0)

    def merge(self, bb: "Cell") -> "Cell":
        bbox = super(Cell, self).merge(bb)
        text_boxes = []
        text_boxes.extend(self.text_boxes)
        text_boxes.extend(bb.text_boxes)
        return Cell(
            bbox.top_left_x,
            bbox.top_left_y,
            bbox.bottom_right_x,
            bbox.bottom_right_y,
            text_boxes=text_boxes,
        )

    def is_empty(self):
        return not any(text_field.text for text_field in self.text_boxes)


@dataclass
class GridCell(BorderBox):
    row: int = None
    col: int = None
    cells: List[Cell] = field(default_factory=list)


@dataclass
class GridRow(BorderBox):
    g_cells: List[GridCell] = field(default_factory=list)


@dataclass
class GridCol(BorderBox):
    g_cells: List[GridCell] = field(default_factory=list)


@dataclass
class GridTable:
    rows: List[GridRow] = field(default_factory=list)
    cols: List[GridCol] = field(default_factory=list)
    cells: List[GridCell] = field(default_factory=list)


@dataclass
class CellLinked(Cell):
    col: int = 0
    row: int = 0
    col_span: int = 0
    row_span: int = 0

    @classmethod
    def from_dict(cls, d):
        return CellLinked(
            *BorderBox.from_dict(d['bbox']).box,
            col=d['column'],
            row=d['row'],
            col_span=d['colspan'],
            row_span=d['rowspan'],
            text_boxes=[]
        )


@dataclass
class StructuredTable:
    bbox: BorderBox
    cells: List[CellLinked] = field(default_factory=list)

    @property
    def rows(self):
        rows = {}
        for cell in self.cells:
            for row in range(cell.row, cell.row + cell.row_span):
                if row not in rows:
                    rows[row] = [cell]
                else:
                    rows[row].append(cell)
        return [
            row
            for num, row in sorted(
                [(num, row) for num, row in rows.items()], key=lambda x: x[0]
            )
        ]

    @property
    def cols(self):
        cols = {}
        for cell in self.cells:
            for col in range(cell.col, cell.col + cell.col_span):
                if col not in cols:
                    cols[col] = [cell]
                else:
                    cols[col].append(cell)
        return [
            col
            for num, col in sorted(
                [(num, col) for num, col in cols.items()], key=lambda x: x[0]
            )
        ]


def flatten(list_fo_lists: List[List]) -> List:
    res = []
    for l in list_fo_lists:
        for element in l:
            res.append(element)
    return res


@dataclass
class StructuredTableHeadered(StructuredTable):
    header: List[List[CellLinked]] = field(default_factory=list)
    header_rows: List[List[CellLinked]] = field(default_factory=list)
    header_cols: List[List[CellLinked]] = field(default_factory=list)

    @property
    def all_cells(self):
        return self.cells + flatten(self.header_rows) + flatten(self.header_cols)

    @staticmethod
    def from_structured_and_rows(
        table: StructuredTable, header: List[List[CellLinked]]
    ):
        header_row_nums = set()
        for row in header:
            for cell in row:
                header_row_nums.add(cell.row)

        body_cells = [
            cell for cell in table.cells if cell.row not in header_row_nums
        ]

        return StructuredTableHeadered(
            bbox=table.bbox, cells=body_cells, header=header
        )

    def actualize_header_with_cols(self, header_cols: List[List[CellLinked]]):
        header_col_nums = set()
        for col in header_cols:
            for cell in col:
                header_col_nums.add(cell.col)

        body_cells = [
            cell for cell in self.cells if cell.col not in header_col_nums
        ]

        self.header.extend(header_cols)
        self.cells = body_cells

    @classmethod
    def from_dict(cls, d):
        header_cells = [CellLinked.from_dict(h_cell) for h_cell in d['header']]
        body_cells = [CellLinked.from_dict(cell) for cell in d['cells']]
        all_cells = header_cells + body_cells
        num_cols = max([cell.col + cell.col_span - 1 for cell in all_cells]) + 1
        header_by_rows = {}
        for cell in header_cells:
            if cell.row in header_by_rows:
                header_by_rows[cell.row].append(cell)
            else:
                header_by_rows[cell.row] = [cell]
        row_header = []
        col_header = []
        for row, cells in header_by_rows.items():
            max_col = max([cell.col + cell.col_span - 1 for cell in cells])
            if max_col >= num_cols - 1:
                row_header.append(cells)
            else:
                col_header.append(cells)

        return cls(
            bbox=BorderBox.from_dict(d['bbox']),
            cells=body_cells,
            header=row_header + col_header,
            header_rows=row_header,
            header_cols=col_header
        )


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

    def count_cells(self):
        if self.rows:
            i = reduce(
                lambda len1, len2: len1 + len2,
                [len(row.objs) for row in self.rows],
            )
        else:
            i = 0
        return i


@dataclass
class TableHeadered(Table):
    header: List[Cell] = field(default_factory=list)

    def count_cells(self):
        return super(TableHeadered, self).count_cells() + len(self.header)


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
            abs(self.bbox.top_left_x - box.top_left_x)
            <= self.HORIZONTAL_MARGIN
            and abs(self.bbox.bottom_right_x - box.bottom_right_x)
            <= self.HORIZONTAL_MARGIN
        )
