import logging
import os
from pathlib import Path

from poppler import load_from_file, PageRenderer
from image_extractor import extract_image
from text_extractor import extract_text


def _check_paths(
    output_img_path: Path,
    output_text_path: Path,
) -> None:
    try:
        os.mkdir(output_img_path)
    except FileExistsError:
        pass
    try:
        os.mkdir(output_text_path)
    except FileExistsError:
        pass


def preprocess(
    path_to_file: Path,
    output_img_path: Path,
    output_text_path: Path,
    owner_password: str = None,
    user_password: str = None,
    start: int = 0,
    stop: int = -1,
) -> None:
    """All paths should be an absolute paths"""
    logging.info(
        f"Start doing {path_to_file}, pages {start}-{stop}",
        " (-1 means to the end of the document)",
    )
    pdf_document = load_from_file(path_to_file, owner_password, user_password)

    if pdf_document.is_locked():
        logging.error(f"PDF {pdf_document.title} is locked")
        return
    _check_paths(output_img_path, output_text_path)
    if stop == -1:
        stop = pdf_document.pages

    renderer = PageRenderer()
    for page_number in range(start, stop):
        logging.info(f"Processing page {page_number}")
        page = pdf_document.create_page(page_number)
        extract_image(page, output_img_path, page_number, renderer)
        extract_text(page, output_text_path, page_number)


if __name__ == "__main__":
    path = Path("test/test_pdf/img_and_vector.pdf")
    preprocess(path, Path("folder_img"), Path("folder_text"))
