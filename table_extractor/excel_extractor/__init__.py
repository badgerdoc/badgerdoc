import json
from pathlib import Path
from typing import Dict

from table_extractor.excel_extractor.extractor import ExcelExtractor
from table_extractor.excel_extractor.converter import excel_to_structured, get_header_using_styles, get_headers_using_structured
from table_extractor.excel_extractor.writer import ExcelWriter


def save_document(document: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path.absolute()), 'w') as f:
        f.write(json.dumps(document, indent=4))


def run_excel_job(file: str, outpath: str):
    excel_extractor = ExcelExtractor()

    results = excel_extractor.extract(file)

    excel_writer = ExcelWriter(data=results, outpath=str((Path(outpath).parent / 'excel_result.xlsx').absolute()))
    pages = excel_writer.write()
    document = {
        'doc_name': str(Path(file).name),
        'pages': pages
    }
    save_document(document, Path(outpath))


if __name__ == "__main__":
    run_excel_job(["/Users/matvei_vargasov/code/novatris/excel_extractor/base/samples/Examples.xlsx"])