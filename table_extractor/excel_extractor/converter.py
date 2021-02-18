from typing import List, Dict
from table_extractor.model.table import (
    StructuredTable,
    StructuredTableHeadered,
    BorderBox,
    CellLinked,
    TextField
)
from table_extractor.headers.header_utils import HeaderChecker
from openpyxl.cell.cell import Cell

header_checker = HeaderChecker()


def should_skip_attr(cells: Dict[tuple, tuple], cell: Cell, another_cell: Cell, key, style) -> bool:
    if key.startswith('__') or key == 'copy':
        return True

    if style == 'border':
        top = key == 'top' and (cell.row == 1 or another_cell.row == 1)
        bottom = key == 'bottom' and (cell.row == max(cells.keys())[1] or another_cell == max(cells.keys())[1])
        left = key == 'left' and (cell.column == 1 or another_cell.column == 1)
        right = key == 'right' and (cell.row == max(cells.keys())[0] or another_cell == max(cells.keys())[0])
        return any([top, bottom, left, right])


def convert_cells(cells: dict) -> list:
    converted_cells = []
    for coords, params in cells.items():
        coords_in_px = params[0]
        text_boxes = TextField(
            bbox=BorderBox(
                coords_in_px['top_left'][0],
                coords_in_px['top_left'][1],
                coords_in_px['bottom_right'][0],
                coords_in_px['bottom_right'][1]
            ),
            text=params[-2]
        )
        new_cell = CellLinked(
            coords_in_px['top_left'][0],
            coords_in_px['top_left'][1],
            coords_in_px['bottom_right'][0],
            coords_in_px['bottom_right'][1],
            text_boxes=[text_boxes],
            col=coords[0],
            row=coords[1],
            col_span=params[1],
            row_span=params[2]
        )

        converted_cells.append(new_cell)
    return converted_cells


def excel_to_structured(excel_tables_sheets: dict) -> dict:
    """
    Converts data from excel to structured table
    """
    tables = {}
    for sheet, sheet_tables in excel_tables_sheets.items():
        tables.setdefault(sheet, [])
        for excel_table in sheet_tables:
            table = StructuredTable(
                cells=convert_cells(excel_table['cells']),
                bbox=BorderBox(
                    excel_table['dimensions'][0]['top_left'][0],
                    excel_table['dimensions'][0]['top_left'][1],
                    excel_table['dimensions'][1]['bottom_right'][0],
                    excel_table['dimensions'][1]['bottom_right'][1],
                )
            )
            tables[sheet].append(table)

    return tables


def create_header(series: List[List[CellLinked]], header_limit: int):
    """
    Search for headers based on cells contents
    @param series: cols or rows of the table
    @param table:
    @param header_limit:
    """
    header_candidates = []
    last_header = None
    for idx, line in enumerate(series[:header_limit]):
        if analyse(line):
            header_candidates.append((idx, True, line))
            last_header = idx
        else:
            header_candidates.append((idx, False, line))
    if last_header is not None:
        header = [line for idx, is_header, line in header_candidates[:last_header + 1]]
    else:
        header = []
    if len(header) > 0.75 * len(series):
        header = []
    return header


def analyse(series: List[CellLinked]):
    # Check if series is header
    headers = []
    for cell in series:
        header_score, _ = header_checker.get_cell_score(cell)
        if header_score > 0:
            headers.append(cell)
    return len(headers) > (len(series) / 5) if len(series) > 5 else len(headers) > (len(series) / 2)


def get_headers_using_structured(tables: dict) -> dict:
    tables_with_header = {}
    for sheet, sheet_tables in excel_to_structured(tables).items():
        tables_with_header.setdefault(sheet, [])
        for table in sheet_tables:
            header_rows = create_header(table.rows, 6)
            table_with_header = StructuredTableHeadered.from_structured_and_rows(table, header_rows)
            header_cols = create_header(table.cols, 4)
            table_with_header.actualize_header_with_cols(header_cols)
            tables_with_header[sheet].append(table_with_header)

    return tables_with_header


def get_header_using_styles(tables: dict, styles_to_check: List = None, matched_counts: int = 1) -> dict:
    if not styles_to_check:
        styles_to_check = ('font', 'fill', 'border')

    tables_with_header = {}
    for sheet, sheet_tables in tables.items():
        tables_with_header.setdefault(sheet, [])
        for table in sheet_tables:
            groups = ([], [], [])
            for _, _, _, _, cell in table['cells'].values():

                if not cell.has_style:
                    groups[2].append(cell)
                    continue
                if not groups[0]:
                    groups[0].append(cell)
                    continue

                matched_styles_count = 0
                for style in styles_to_check:

                    cell_attr = getattr(cell, style)
                    group_attr = getattr(groups[0][0], style)
                    if all([getattr(cell_attr, key) == getattr(group_attr, key) for key in dir(cell_attr) if
                            not should_skip_attr(table['cells'], cell, groups[0][0], key, style)]):
                        matched_styles_count += 1
                    if sheet == 'Sheet': print(cell, matched_styles_count)
                if len(styles_to_check) - matched_styles_count <= matched_counts:
                    groups[0].append(cell)
                else:
                    groups[1].append(cell)

            max_first_row_col = 0
            headers = None
            import pprint
            if sheet == 'Sheet 2': pprint.pprint(groups)
            for group in groups:
                max_rows_cols = len([cell for cell in group if
                                     cell.row == min(table['cells'].keys())[1] or cell.column ==
                                     min(table['cells'].keys())[0]])
                if max_rows_cols > max_first_row_col:
                    headers = group
                    max_first_row_col = max_rows_cols
            if headers:
                table['headers'] = headers
                tables_with_header[sheet].append(table)

    return tables_with_header
