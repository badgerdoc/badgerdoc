from typing import List, Dict
from dataclasses import dataclass

from table_extractor.model.table import StructuredTableHeadered

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Border, Side, Alignment, Protection, Font


class BaseWriter:
    """
    Base class for writers
    """

    def __init__(self, data: Dict[str, list], outpath: str):
        self.data = data
        self.outpath = outpath

    def write(self):
        raise NotImplemented


class ExcelWriter(BaseWriter):
    """
    Generate excel from structured table
    """
    wb = Workbook()

    def write(self):
        header_font = Font(bold=True)
        header_fill = PatternFill(start_color="00C0C0C0", end_color="00C0C0C0", fill_type = "solid")
        header_border = thin_border = Border(left=Side(style='thick'),
                     right=Side(style='thick'),
                     top=Side(style='thick'),
                     bottom=Side(style='thick'))

        for i, (sheet, tables) in enumerate(self.data.items()):
            print(i, sheet)
            if not i:
                ws = self.wb.active
            else:
                ws = self.wb.create_sheet(sheet)

            for table in tables:
                for header_cells in table.header:
                    for cell in header_cells:
                        added_cell = ws.cell(row=cell.row, column=cell.col, value=cell.text_boxes[0].text)
                        added_cell.fill = header_fill
                        added_cell.font = header_font

                for cell in table.cells:
                    ws.cell(row=cell.row, column=cell.col, value=cell.text_boxes[0].text)

        self.wb.save(self.outpath)
