from typing import List
from table_extractor.model.table import (
    StructuredTable,
    StructuredTableHeadered,
    BorderBox,
    CellLinked,
    TextField
)


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


def excel_to_structured(excel_tables_sheets: dict) -> List[StructuredTable]:
    """
    Converts data from excel to structured table
    """
    tables = []
    for sheet_tables in excel_tables_sheets.values():
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
            tables.append(table)

    return tables