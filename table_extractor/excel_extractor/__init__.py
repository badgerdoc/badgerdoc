from typing import Union

from table_extractor.excel_extractor.extractor import ExcelExtractor
from table_extractor.excel_extractor.converter import excel_to_structured, get_headers


def run_excel_job(files: Union[list, tuple]):
    extractor = ExcelExtractor()

    for file in files:
        results = extractor.extract(file)
        tables = excel_to_structured(results)
        headerd_tables = get_headers(tables)


if __name__ == "__main__":
    run_excel_job(["/Users/matvei_vargasov/code/novatris/excel_extractor/base/samples/Examples.xlsx"])