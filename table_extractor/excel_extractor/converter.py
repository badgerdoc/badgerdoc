from typing import List
from table_extractor.model.table import (
    StructuredTable,
    StructuredTableHeadered,
    BorderBox,
    CellLinked,
    TextField
)
from table_extractor.headers.header_utils import HeaderChecker

header_checker = HeaderChecker()


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
            text=params[-1]
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
            print(excel_table['dimensions'])
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


def get_headers(tables: dict) -> dict:
    tables_with_header = {}
    for sheet, sheet_tables in tables.items():
        tables_with_header.setdefault(sheet, [])
        for table in sheet_tables:
            header_rows = create_header(table.rows, 6)
            table_with_header = StructuredTableHeadered.from_structured_and_rows(table, header_rows)
            header_cols = create_header(table.cols, 4)
            table_with_header.actualize_header_with_cols(header_cols)
            tables_with_header[sheet].append(table_with_header)

    return tables_with_header
