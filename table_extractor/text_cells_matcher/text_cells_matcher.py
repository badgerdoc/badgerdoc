from typing import List, Tuple

from table_extractor.model.table import Cell, Table, TextField


def match_table_text(table: Table, text_boxes: List[TextField]):
    in_table: List[TextField] = []
    out_of_table: List[TextField] = []
    for t_box in text_boxes:
        if t_box.bbox.box_is_inside_another(table.bbox):
            in_table.append(t_box)
        else:
            out_of_table.append(t_box)
    return in_table, out_of_table


def match_cells_text(cells: List[Cell], text_box: TextField):
    for cell in cells:
        if text_box.bbox.box_is_inside_another(cell, threshold=0.4):
            cell.text_boxes.append(text_box)
            return True
    return False


def match_cells_text_fields(
    cells: List[Cell], text_boxes: List[TextField]
) -> Tuple[int, List[TextField]]:
    count = 0
    for cell in cells:
        cell.text_boxes = []
    not_matched = []
    for t_box in text_boxes:
        if match_cells_text(cells, t_box):
            count += 1
        else:
            not_matched.append(t_box)

    return count, not_matched
