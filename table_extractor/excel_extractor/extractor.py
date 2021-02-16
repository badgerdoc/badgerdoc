import sys

from typing import Union, Generator
from enum import Enum
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.cell.cell import Cell, MergedCell


class Output(Enum):
    PDF = 'pdf'
    EXCEL = 'excel'
    HTML = 'html'
    JSON = 'json'
    PNG = 'png'


class ExcelExtractor:
    """
    Render excel table to png file
    """

    tables = {}
    ws_tables = []

    def extract(self, file: Union[str, Path]) -> dict:
        """
        Exctract information and save it in given ouput format
        """
        wb = load_workbook(file)

        for sheet in wb:
            self.parse_sheet(sheet)
            self.tables[sheet.title] = self.ws_tables
            self.ws_tables = []

        return self.tables

    def _check_interception(self, attr, table, cell) -> bool:
        return getattr(table['end'], attr) >= getattr(cell, attr) >= getattr(table['start'], attr)

    def iter_subtable_rows(self,sheet, first_cell: Cell) -> Generator:
        for row in sheet.iter_rows(min_row=first_cell.row, min_col=first_cell.column):
            if not row[0].value and not isinstance(row[0], MergedCell):
                break
            yield row

    def in_table(self, cell: Cell) -> bool:
        for table in self.ws_tables:
            if self._check_interception('row', table, cell) and self._check_interception('column', table, cell):
                return True
        return False

    def get_end_cell(self, sheet: Worksheet, first_cell: Cell, current_cell: Cell) -> Cell:
        """
        Currently not used
        """
        row_number = first_cell.row
        for row in self.iter_subtable_rows(sheet, first_cell):
            row_number += 1

        column = current_cell.column if current_cell.column == sheet.max_column else current_cell.column -1
        row_number -= 1
        return sheet.cell(row=row_number, column=column)

    def fill_cells(self, sheet: Worksheet, first_cell: Cell, current_cell: Cell) -> dict:
        cells = {}
        for row in self.iter_subtable_rows(sheet, first_cell):
            for cell in row:
                cells[(cell.column, cell.row)] = (
                    sheet.column_dimensions[cell.column].width,
                    sheet.row_dimensions[cell.row].height
                )
        return cells

    def get_table(self, sheet: Worksheet, cell: Cell, table_started: Cell) -> dict:
        """
        Returns dict with data table
        """
        table = {}
        table['start'] = (table_started.column, table_started.row)
        table['cells'] = self.fill_cells(sheet, table_started, cell)
        table['end'] = max(table['cells'].keys())
        return table

    def finish_table(self, sheet, cell, table_started):
        column = cell.column if cell.column == sheet.max_column else cell.column -1
        self.ws_tables.append({
            'start': table_started,
            'end': sheet.cell(
                column=column,
                row=self.get_end_cell(
                    sheet,
                    table_started,
                    cell
                ).row
            )
        })

    def parse_sheet(self, sheet: Worksheet) -> list:
        tables = []
        for row in sheet.iter_rows():

            table_started = None

            for cell in row:
                # print(cell)
                if cell.value:
                    if self.in_table(cell):
                        # print('in table')
                        continue

                    if table_started:
                        if cell.column == sheet.max_column:
                            self.finish_table(sheet, cell, table_started)
                            table_started = None
                            continue
                    else:
                        table_started = cell
                else:
                    # print('ts',table_started)
                    if isinstance(cell, MergedCell):
                        print(cell)
                        continue

                    if table_started:
                        self.finish_table(sheet, cell, table_started)
                        table_started = None

        return tables


if __name__ == "__main__":
    ext = ExcelExtractor()
    ext.extract(sys.argv[1])