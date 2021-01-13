from typing import List

import cv2
import numpy as np

from table_detection.common import BBox


def recognize_bordered_table(img: np.ndarray) -> List[BBox]:
    if len(img.shape) == 3 and img.shape[2] != 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Normalize contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)

    (thresh, img_bin) = cv2.threshold(
        img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    img_bin = cv2.bitwise_not(img_bin)

    kernel_length_v = np.array(img_gray).shape[1] // 120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

    kernel_length_h = np.array(img_gray).shape[1] // 40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(
        vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0
    )
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(
        table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    image_width = img.shape[1]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # filter FP detection
        if h <= 25 or w <= 25:
            continue

        # excluding page shape boxes
        if w < 0.95 * image_width:
            boxes.append(BBox(x, y, x + w, y + h))
    return boxes
