from typing import Dict
from copy import copy

from table_extractor.excel_extractor.constants import (
    HEADER_FONT,
    HEADER_BORDER,
    HEADER_FILL
)
from table_extractor.excel_extractor.converter import (
    get_headers_using_structured,
    get_header_using_styles,
)
from openpyxl import Workbook


class BaseWriter:
    """
    Base class for writers
    """

    def __init__(self, data: Dict[str, list], outpath: str):
        self.data = data
        self.outpath = outpath
        self._tables_with_headers = None

    def write(self):
        raise NotImplemented

    @property
    def converted_data(self) -> dict:
        raise NotImplemented


class ExcelWriter(BaseWriter):
    """
    Generate excel from structured table
    """
    wb = Workbook()

    machine_learning_used = False

    def write(self, data: dict = None):
        if not data:
            data = self.tables_with_headers.items()

        for i, (sheet, tables) in enumerate(data.items()):
            if not i:
                ws = self.wb.active
            else:
                ws = self.wb.create_sheet(sheet)

            for table in tables:
                if not table['headers']:
                    try:
                        processed_table = get_headers_using_structured(table)
                        for header_cells in processed_table.header:
                            for cell in header_cells:
                                added_cell = ws.cell(row=cell.row, column=cell.col, value=cell.text_boxes[0].text)
                                added_cell.fill = HEADER_FILL
                                added_cell.font = HEADER_FONT

                        for cell in processed_table.cells:
                            ws.cell(row=cell.row, column=cell.col, value=cell.text_boxes[0].text)
                    except Exception:
                        # TODO: skip this step if json
                        continue
                else:
                    # TODO: Merged cells
                    for _cell in table['headers']:
                        cell = _cell[-1]
                        added_cell = ws.cell(row=cell.row, column=cell.column, value=cell.value)
                        added_cell.fill = HEADER_FILL
                        added_cell.font = HEADER_FONT

                    for _cell in table['cells'].values():
                        cell = _cell[-1]
                        ws.cell(row=cell.row, column=cell.column, value=cell.value)

        self.wb.save(self.outpath)

    @property
    def tables_with_headers(self) -> dict:
        if not self._tables_with_headers:
            self._tables_with_headers = get_header_using_styles(self.data)
        return self._tables_with_headers
