import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

import cv2
import numpy
import numpy as np
from tqdm import tqdm

from ..model.table import Cell
from .models import Image, ImageDTO
from .utils import draw_cols_and_rows

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = ("png", "jpg", "jpeg", "bmp")


def has_image_extension(path: Path, allowed_extensions=IMG_EXTENSIONS):
    return any(
        path.name.lower().endswith(e.lower()) for e in allowed_extensions
    )


def detect_bordered_tables_on_image(
    image: Image, draw=True, mask: numpy.ndarray = None
):
    if mask is None:
        mask = cv2.imread(str(image.path.absolute()))
    image.shape = mask.shape[:2]
    img_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    (thresh, img_bin) = cv2.threshold(
        img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    img_bin = cv2.bitwise_not(img_bin)

    kernel_length_v = np.array(img_gray).shape[1] // 120
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length_v)
    )
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

    kernel_length_h = np.array(img_gray).shape[1] // 40
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_length_h, 1)
    )
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(
        im_temp2, horizontal_kernel, iterations=3
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(
        vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0
    )
    table_segment = cv2.erode(
        cv2.bitwise_not(table_segment), kernel, iterations=2
    )
    thresh, table_segment = cv2.threshold(
        table_segment, 0, 255, cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(
        table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    image_width = mask.shape[1]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # filter FP detection
        if h <= 25 or w <= 25:
            continue

        # excluding page shape boxes
        if w < 0.95 * image_width:
            boxes.append(Cell(x, y, x + w, y + h))

        if draw:
            cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if draw:
        res_path = image.path.parent.parent / "detected_boxes"
        res_path.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str((res_path / image.path.name).absolute()), mask)
    image.objs = boxes


def detect_images(
    images: Path, pdf_pages_size: List, draw: bool = True
) -> dict:
    result = []
    logger.info(f"Start detection for {images}")
    for image_path, original_size in tqdm(
        zip(
            sorted(
                filter(lambda x: has_image_extension(x), images.iterdir()),
                key=lambda x: int(x.name.split(".")[0]),
            ),
            pdf_pages_size,
        )
    ):
        image = Image(path=image_path, pdf_page_shape=original_size)
        detect_bordered_tables_on_image(image, draw=draw)
        image.analyze()
        image.extract_text()

        if draw:
            draw_cols_and_rows(image)
        image.scale_bboxes()
        image_dto = ImageDTO.from_image(image)
        result.append(asdict(image_dto))

    return {"detections": result}


def detect_tables_on_page(image_path: Path, draw=False):
    mask = cv2.imread(str(image_path.absolute()))
    image = Image(
        path=image_path, pdf_page_shape=[mask.shape[1], mask.shape[0]]
    )
    image.shape = mask.shape[:2]

    detect_bordered_tables_on_image(image, draw=True, mask=mask)

    image.analyze()

    if draw:
        draw_cols_and_rows(image)

    image.scale_bboxes()
    return image
