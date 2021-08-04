"""Module implement text page extractor"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from poppler import load_from_file
from poppler.page import Page


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

    def convert_to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "bbox": {
                "left": self.bbox.x,
                "top": self.bbox.y,
                "height": self.bbox.height,
                "width": self.bbox.width,
            },
            "blocks": self._convert_text_fields(self.text_fields),
        }

    def save_as_json(self, fp) -> None:
        logging.info("Save JSON")
        json.dump(self.convert_to_dict(), fp, ensure_ascii=False)

    @staticmethod
    def _convert_text_fields(text_fields) -> List[dict]:
        res = []
        for text_field in text_fields:
            res.append(
                {
                    "bbox": {
                        "left": text_field.bbox.x,
                        "top": text_field.bbox.y,
                        "height": text_field.bbox.height,
                        "width": text_field.bbox.width,
                    },
                    "text": text_field.text,
                }
            )
        return res


def get_poppler_text_field(text_list: Page.text_list) -> List[PopplerTextField]:
    res = []
    for text in text_list:
        bound = PopplerBoundingBox(
            text.bbox.x, text.bbox.y, text.bbox.height, text.bbox.width
        )
        res.append(PopplerTextField(bound, text.text))
    return res


def extract_text(
    page: Page,
    dir_to_save: Path,
    page_number: int,
) -> None:
    logging.info("Start to extract text")
    page_data = PopplerPage(
        PopplerBoundingBox(
            page.page_rect().x,
            page.page_rect().y,
            page.page_rect().height,
            page.page_rect().width,
        ),
        page_number,
        page.orientation,
        [i for i in get_poppler_text_field(page.text_list())],
    )
    with open(dir_to_save / f"{page_number}", "w") as file_to_save:
        page_data.save_as_json(file_to_save)


if __name__ == "__main__":
    path = Path("test/test_pdf/img_and_vector.pdf")

    pdf_document = load_from_file(path)
    page = pdf_document.create_page(0)
    extract_text(page, Path("new"), 0)
