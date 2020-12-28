import logging
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict

from dataclasses import dataclass
from pdf2image import convert_from_path
from poppler import load_from_file

from badgerdoc.bordered_tables.models import BorderBox, TextField

logger = logging.getLogger(__name__)
DPI = 400


@dataclass
class PopplerBoundingBox:
    x: float
    y: float
    height: float
    width: float


@dataclass
class PopplerTextField:
    bbox: PopplerBoundingBox
    text: str


@dataclass
class PopplerPage:
    bbox: PopplerBoundingBox
    page_num: int
    orientation: str
    text_fields: Optional[List[PopplerTextField]]


def bounding_box_to_bbox(bounding_box: PopplerBoundingBox, scale: float):
    return BorderBox(
        top_left_x=int(bounding_box.x * scale),
        top_left_y=int(bounding_box.y * scale),
        bottom_right_x=int((bounding_box.x + bounding_box.width) * scale),
        bottom_right_y=int((bounding_box.y + bounding_box.height) * scale)
    )


def poppler_text_field_to_text_field(pt_field: PopplerTextField, scale: float):
    return TextField(
        bbox=bounding_box_to_bbox(pt_field.bbox, scale),
        text=pt_field.text
    )


def convert_pdf_to_images(pdf_file: Path, out_dir: Path) -> Path:
    logger.info("Start pdf to png conversion")
    out_dir = out_dir.joinpath(Path(f"{pdf_file.name}/images/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(
        pdf_file,
        dpi=DPI,
        output_folder=str(out_dir.absolute()),
        paths_only=True,
        fmt="png"
    )
    for i, page in enumerate(pages):
        shutil.move(page, out_dir.absolute() / f"{i}.png")
    logger.info("Done conversion")
    return out_dir


def extract_text(pdf_file: Path) -> Dict[str, PopplerPage]:
    pdf_document = load_from_file(pdf_file.absolute())
    logger.info("Text extraction for: %s, pages %s", pdf_file.name, pdf_document.pages)
    pages = {}
    for page_num in range(pdf_document.pages):
        logger.debug("Processing page %s", page_num)
        page = pdf_document.create_page(page_num)
        page_rect = page.page_rect()
        text_fields: List[PopplerTextField] = []
        for text_field in page.text_list():
            text_fields.append(
                PopplerTextField(
                    bbox=PopplerBoundingBox(
                        x=text_field.bbox.x,
                        y=text_field.bbox.y,
                        height=text_field.bbox.height,
                        width=text_field.bbox.width
                    ),
                    text=text_field.text))
        pages[str(page_num)] = PopplerPage(
            bbox=PopplerBoundingBox(
                x=page_rect.x,
                y=page_rect.y,
                height=page_rect.height,
                width=page_rect.width
            ),
            page_num=page_num,
            orientation=str(page.orientation),
            text_fields=text_fields
        )
    logger.info("Text extraction for: %s done", pdf_file.name)
    return pages


def extract_text_to_json(pdf_file: Path, out_dir: Path):
    out_path = out_dir.joinpath(Path(f"{pdf_file.name}/text_data.json"))
    data = extract_text(pdf_file)
    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(list(data)))

    return out_path