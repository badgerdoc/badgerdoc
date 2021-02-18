from typing import Dict

from table_extractor.excel_extractor.constants import (
    HEADER_FONT,
    HEADER_BORDER,
    HEADER_FILL
)
from table_extractor.excel_extractor.converter import (
    get_headers_using_structured,
    get_header_using_styles
)
from openpyxl import Workbook


class BaseWriter:
    """
    Base class for writers
    """

    def __init__(self, data: Dict[str, list], outpath: str):
        self.data = data
        self.outpath = outpath
        self._converted_data = None

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

    def write(self):

        for i, (sheet, tables) in enumerate(self.converted_data.items()):
            if not i:
                ws = self.wb.active
            else:
                ws = self.wb.create_sheet(sheet)

            for table in tables:
                for header_cells in table.header:
                    for cell in header_cells:
                        added_cell = ws.cell(row=cell.row, column=cell.col, value=cell.text_boxes[0].text)
                        added_cell.fill = HEADER_FILL
                        added_cell.font = HEADER_FONT

                for cell in table.cells:
                    ws.cell(row=cell.row, column=cell.col, value=cell.text_boxes[0].text)

        self.wb.save(self.outpath)

    @property
    def converted_data(self) -> dict:
        if not self._converted_data:
            self._converted_data = get_header_using_styles(self.data)
            self._converted_data = get_headers_using_structured(self.data)
        return self._converted_data
