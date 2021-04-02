import logging
from typing import Dict, List, Optional, Tuple

from table_extractor.model.table import (
    BorderBox,
    Cell,
    CellLinked,
    GridCell,
    GridCol,
    GridRow,
    GridTable,
    StructuredTable,
)

LOGGER = logging.getLogger(__name__)


def _match_cells_and_table(table: GridTable, cells: List[Cell]):
    not_matched = []
    for cell in cells:
        row_num = None
        for r_idx, row in enumerate(table.rows):
            if row.box_is_inside_another(cell):
                row_num = r_idx
                break
        col_num = None
        for c_idx, col in enumerate(table.cols):
            if col.box_is_inside_another(cell):
                col_num = c_idx
                break
        if row_num is not None and col_num is not None:
            g_cell = table.cells[row_num * len(table.cols) + col_num]
            if not g_cell.box_is_inside_another(cell):
                not_matched.append(cell)
            g_cell.cells.append(cell)
        else:
            not_matched.append(cell)

    return not_matched


def _merge_closest_cells(g_cells: List[GridCell]):
    for g_cell in g_cells:
        cells = g_cell.cells.copy()
        new_cells = []
        while len(cells) > 0:
            curr_cell = cells.pop()
            for orig_cell in cells:
                if curr_cell.box_is_inside_another(orig_cell, 0.05):
                    curr_cell = curr_cell.merge(orig_cell)
                    cells.remove(orig_cell)
            new_cells.append(curr_cell)
        g_cell.cells = new_cells


def _find_gaps_in_zone(
    line: List[bool], zone: Tuple[int, int]
) -> List[List[int]]:
    gaps = []
    curr_gap = []
    for point_idx in range(zone[0], zone[1]):
        if line[point_idx] and curr_gap:
            curr_gap.append(point_idx - 1)
            gaps.append(curr_gap)
            curr_gap = []
        if not line[point_idx] and not curr_gap:
            curr_gap.append(point_idx)
    if curr_gap:
        curr_gap.append(zone[1])
        gaps.append(curr_gap)
    return gaps


def _get_line_coord_from_gaps(gaps: List[List[int]]) -> List[int]:
    return [(gap[0] + gap[1]) // 2 for gap in gaps]


def find_grid_table(h_lines, v_lines) -> GridTable:
    cells: List[GridCell] = []
    rows: List[GridRow] = []
    cols: List[GridCol] = []
    prev_h_line = h_lines[0]
    for h_idx, h_line in enumerate(h_lines[1:]):
        if h_idx >= len(rows):
            curr_row = GridRow(
                top_left_x=v_lines[0],
                top_left_y=prev_h_line,
                bottom_right_x=v_lines[-1],
                bottom_right_y=h_line,
            )
            rows.append(curr_row)
        else:
            curr_row = rows[h_idx]

        prev_v_line = v_lines[0]
        for v_idx, v_line in enumerate(v_lines[1:]):
            if v_idx >= len(cols):
                curr_col = GridCol(
                    top_left_x=prev_v_line,
                    top_left_y=h_lines[0],
                    bottom_right_x=v_line,
                    bottom_right_y=h_lines[-1],
                )
                cols.append(curr_col)
            else:
                curr_col = cols[v_idx]
            cell = GridCell(
                top_left_x=prev_v_line,
                bottom_right_x=v_line,
                top_left_y=prev_h_line,
                bottom_right_y=h_line,
                row=h_idx,
                col=v_idx,
            )
            curr_row.g_cells.append(cell)
            curr_col.g_cells.append(cell)
            cells.append(cell)
            prev_v_line = v_line
        prev_h_line = h_line
    grid_table = GridTable(rows=rows, cols=cols, cells=cells)
    return grid_table


def _find_lines(
    table_bbox: BorderBox, cells: List[Cell], image_shape: Tuple[int, int]
):
    if not cells:
        return [], []
    h_proj = [False for _ in range(0, image_shape[1])]
    v_proj = [False for _ in range(0, image_shape[0])]

    for cell in cells:
        h_proj[cell.top_left_x : cell.bottom_right_x + 1] = [
            True for _ in range(cell.top_left_x, cell.bottom_right_x + 1)
        ]
        v_proj[cell.top_left_y : cell.bottom_right_y + 1] = [
            True for _ in range(cell.top_left_y, cell.bottom_right_y + 1)
        ]

    v_gaps = _find_gaps_in_zone(
        h_proj, (table_bbox.top_left_x, table_bbox.bottom_right_x)
    )
    v_line_coords = _get_line_coord_from_gaps(v_gaps)

    h_gaps = _find_gaps_in_zone(
        v_proj, (table_bbox.top_left_y, table_bbox.bottom_right_y)
    )
    h_line_coords = _get_line_coord_from_gaps(h_gaps)
    return h_line_coords, v_line_coords


def _actualize_line_separators(
    table: GridTable, image_shape: Tuple[int, int]
) -> Tuple[List[int], List[int]]:
    span_candidates: Dict[int, GridCell] = {}
    for g_cell in table.cells:
        if len(g_cell.cells) > 1:
            span_candidates[len(table.cols) * g_cell.row + g_cell.col] = g_cell

    if not span_candidates:
        return [], []

    col_candidates = {}
    for g_cell in span_candidates.values():
        col_candidates[g_cell.col] = table.cols[g_cell.col]

    row_candidates = {}
    for g_cell in span_candidates.values():
        row_candidates[g_cell.row] = table.rows[g_cell.row]

    v_lines_to_add = []
    h_lines_to_add = []
    for cand_col in col_candidates.values():
        v_lines = []
        for g_cell in cand_col.g_cells:
            _, v_cell_lines = _find_lines(g_cell, g_cell.cells, image_shape)
            if v_cell_lines:
                min_v_cells = min([cell.top_left_x for cell in g_cell.cells])
                max_v_cells = max(
                    [cell.bottom_right_x for cell in g_cell.cells]
                )
                v_cell_lines = list(
                    filter(
                        lambda line: min_v_cells < line < max_v_cells,
                        v_cell_lines,
                    )
                )
            v_lines.append(v_cell_lines)
        g_cell_v_line = list(zip(cand_col.g_cells, v_lines))
        cand_v_sort = list(
            filter(
                lambda x: x[3],
                sorted(
                    [
                        (idx, len(v_cell_lines), g_cell, v_cell_lines)
                        for idx, (g_cell, v_cell_lines) in enumerate(
                            g_cell_v_line
                        )
                    ],
                    key=lambda x: (x[1], x[2].top_left_y),
                ),
            )
        )

        i = 0
        while i < len(cand_v_sort):
            idx, l, g_cell, v_cell_lines = cand_v_sort[i]
            if not l:
                i += 1
                continue
            cand_g_cells = g_cell.cells.copy()
            new_v_lines = v_cell_lines
            count_not_broken = 0
            for j in range(i + 1, len(cand_v_sort)):
                jdx, _, cand_j, v_lines_j = cand_v_sort[j]
                # Try compute common v_lines
                cells_to_check = cand_g_cells.copy()
                cells_to_check.extend(cand_j.cells)
                zone = BorderBox(
                    top_left_x=g_cell.top_left_x,
                    top_left_y=g_cell.top_left_y,
                    bottom_right_x=g_cell.bottom_right_x,
                    bottom_right_y=cand_j.bottom_right_y,
                )
                _, v = _find_lines(zone, cells_to_check, image_shape)
                if v:
                    min_v_cells = min(
                        [cell.top_left_x for cell in cells_to_check]
                    )
                    max_v_cells = max(
                        [cell.bottom_right_x for cell in cells_to_check]
                    )
                    v = list(
                        filter(
                            lambda line: min_v_cells < line < max_v_cells, v
                        )
                    )
                if len(v) >= len(v_cell_lines):
                    cand_g_cells = cells_to_check
                    new_v_lines = v
                    count_not_broken += 1
                else:
                    break
            i += count_not_broken + 1
            v_lines_to_add.extend(new_v_lines)

    for cand_row in row_candidates.values():
        h_lines = []
        for g_cell in cand_row.g_cells:
            h_cell_lines, _ = _find_lines(g_cell, g_cell.cells, image_shape)
            if h_cell_lines:
                min_h_cells = min([cell.top_left_y for cell in g_cell.cells])
                max_h_cells = max(
                    [cell.bottom_right_y for cell in g_cell.cells]
                )
                h_cell_lines = list(
                    filter(
                        lambda line: min_h_cells < line < max_h_cells,
                        h_cell_lines,
                    )
                )
            h_lines.append(h_cell_lines)
        g_cell_h_line = list(zip(cand_row.g_cells, h_lines))
        cand_h_sort = sorted(
            [
                (idx, len(h_cell_lines), g_cell, h_cell_lines)
                for idx, (g_cell, h_cell_lines) in enumerate(g_cell_h_line)
            ],
            key=lambda x: (x[1], x[2].top_left_y),
        )

        i = 0
        while i < len(cand_h_sort):
            idx, l, g_cell, h_cell_lines = cand_h_sort[i]
            if not l:
                i += 1
                continue
            cand_g_cells = g_cell.cells.copy()
            new_h_lines = h_cell_lines
            count_not_broken = 0
            for j in range(i + 1, len(cand_h_sort)):
                jdx, _, cand_j, h_lines_j = cand_h_sort[j]
                # Try compute common v_lines
                cells_to_check = cand_g_cells.copy()
                cells_to_check.extend(cand_j.cells)
                zone = BorderBox(
                    top_left_x=g_cell.top_left_x,
                    top_left_y=g_cell.top_left_y,
                    bottom_right_x=g_cell.bottom_right_x,
                    bottom_right_y=cand_j.bottom_right_y,
                )
                h, _ = _find_lines(zone, cells_to_check, image_shape)
                if h:
                    min_h_cells = min(
                        [cell.top_left_y for cell in cells_to_check]
                    )
                    max_h_cells = max(
                        [cell.bottom_right_y for cell in cells_to_check]
                    )
                    h = list(
                        filter(
                            lambda line: min_h_cells < line < max_h_cells, h
                        )
                    )
                if len(h) >= len(h_cell_lines):
                    cand_g_cells = cells_to_check
                    new_h_lines = h
                    count_not_broken += 1
                else:
                    break
            i += count_not_broken + 1
            h_lines_to_add.extend(new_h_lines)
    return list(set(v_lines_to_add)), list(set(h_lines_to_add))


def reconstruct_table_from_grid(
    grid_table: GridTable, cells: List[Cell]
) -> Tuple[Optional[StructuredTable], List[Cell]]:
    not_matched = []
    linked_cells = []
    grid_cells_dict = {}
    for g_cell in grid_table.cells:
        grid_cells_dict[
            g_cell.row * len(grid_table.cols) + g_cell.col
        ] = g_cell
    for cell in cells:
        rows = []
        for r_idx, row in enumerate(grid_table.rows):
            if row.box_is_inside_another(cell, 0.0):
                rows.append((r_idx, row))
        cols = []
        for c_idx, col in enumerate(grid_table.cols):
            if col.box_is_inside_another(cell, 0.0):
                cols.append((c_idx, col))
        if rows and cols:
            linked_cells.append(
                CellLinked(
                    top_left_y=rows[0][1].top_left_y,
                    top_left_x=cols[0][1].top_left_x,
                    bottom_right_y=rows[-1][1].bottom_right_y,
                    bottom_right_x=cols[-1][1].bottom_right_x,
                    row=rows[0][0],
                    col=cols[0][0],
                    row_span=len(rows),
                    col_span=len(cols),
                    text_boxes=cell.text_boxes,
                )
            )
            for row in rows:
                for col in cols:
                    if grid_cells_dict.get(
                        row[0] * len(grid_table.cols) + col[0]
                    ):
                        _ = grid_cells_dict.pop(
                            row[0] * len(grid_table.cols) + col[0]
                        )
        else:
            not_matched.append(cell)
    for _, g_cell in grid_cells_dict.items():
        linked_cells.append(
            CellLinked(
                top_left_y=g_cell.top_left_y,
                top_left_x=g_cell.top_left_x,
                bottom_right_y=g_cell.bottom_right_y,
                bottom_right_x=g_cell.bottom_right_x,
                row=g_cell.row,
                col=g_cell.col,
                row_span=1,
                col_span=1,
                text_boxes=[],
            )
        )
    if not grid_table.cols or not grid_table.rows or not grid_table.cells:
        return None, cells
    table = StructuredTable(
        bbox=BorderBox(
            top_left_y=grid_table.rows[0].top_left_y,
            top_left_x=grid_table.cols[0].top_left_x,
            bottom_right_y=grid_table.rows[-1].bottom_right_y,
            bottom_right_x=grid_table.cols[-1].bottom_right_x,
        ),
        cells=linked_cells,
    )
    return table, not_matched


def construct_table_from_cells(
    table_bbox: BorderBox, cells: List[Cell], image_shape: Tuple[int, int]
) -> Optional[StructuredTable]:
    if not table_bbox or not image_shape or not cells or len(cells) < 2:
        return None
    h_lines, v_lines = _find_lines(table_bbox, cells, image_shape)
    if not h_lines:
        h_lines = [table_bbox.top_left_y, table_bbox.bottom_right_y]
    elif len(h_lines) == 1:
        h_lines = (
            [table_bbox.top_left_y] + h_lines + [table_bbox.bottom_right_y]
        )
    if not v_lines:
        v_lines = [table_bbox.top_left_x, table_bbox.bottom_right_x]
    elif len(v_lines) == 1:
        v_lines = (
            [table_bbox.top_left_x] + v_lines + [table_bbox.bottom_right_x]
        )

    grid_table = find_grid_table(h_lines, v_lines)

    while True:
        _match_cells_and_table(grid_table, cells)
        vv_lines, hh_lines = _actualize_line_separators(
            grid_table, image_shape
        )
        if not vv_lines and not hh_lines:
            break
        h_lines.extend(hh_lines)
        v_lines.extend(vv_lines)
        grid_table = find_grid_table(sorted(h_lines), sorted(v_lines))

    table, _ = reconstruct_table_from_grid(grid_table, cells)
    if _:
        LOGGER.debug(f"Not matched: {len(_)}")
    return table
