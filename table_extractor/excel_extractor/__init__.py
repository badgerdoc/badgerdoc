from table_extractor.excel_extractor.extractor import ExcelExtractor
from table_extractor.excel_extractor.converter import excel_to_structured, get_header_using_styles, get_headers_using_structured
from table_extractor.excel_extractor.writer import ExcelWriter


def run_excel_job(file: str, outpath: str):
    excel_extractor = ExcelExtractor()

    results = excel_extractor.extract(file)

    excel_writer = ExcelWriter(data=results, outpath=outpath)
    excel_writer.write()


if __name__ == "__main__":
    run_excel_job(["/Users/matvei_vargasov/code/novatris/excel_extractor/base/samples/Examples.xlsx"])