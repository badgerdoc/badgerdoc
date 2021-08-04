"""Module implement image page extractor"""
import logging
from pathlib import Path

from poppler import PageRenderer
from poppler.page import Page

DPI = 400  # DPI for extracting image


def extract_image(
    page: Page, output_img_path: Path, page_number: int, renderer: PageRenderer
) -> bool:
    """Save single page as output_img_path/page_number.png"""
    logging.info(f"Start extracting image {page_number}")
    image = renderer.render_page(page, xres=400, yres=400)
    if not image.is_valid:
        logging.error("Image is invalid")
        return False
    image.save(str(output_img_path) + f"/{page_number}.png", "png", DPI)
    return True
