from table_extractor.excel_extractor.extractor import ExcelExtractor
from table_extractor.excel_extractor.converter import excel_to_structured, get_headers
from table_extractor.excel_extractor.writer import ExcelWriter


def run_excel_job(file: str, outpath: str):
    excel_extractor = ExcelExtractor()

    results = excel_extractor.extract(file)
    tables = excel_to_structured(results)
    tables_with_headers = get_headers(tables)

    excel_writer = ExcelWriter(data=tables_with_headers, outpath=outpath)
    excel_writer.write()


if __name__ == "__main__":
    run_excel_job(["/Users/matvei_vargasov/code/novatris/excel_extractor/base/samples/Examples.xlsx"])