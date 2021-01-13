from typing import List, Tuple, Union

import cv2
import numpy as np

from table_detection.common import Axis, BBox
from table_detection.gaps.gaps import GapMask
from table_detection.lines.lines import Line


def draw_line(
    img: np.ndarray, line: Line, color: Union[int, Tuple[int, int, int]], thickness=3
) -> np.ndarray:
    img = img.copy()
    return cv2.line(img, line.vector.pt1, line.vector.pt2, color, thickness)


def draw_boxes(
    img: np.ndarray,
    boxes: List[BBox],
    color: Union[int, Tuple[int, int, int]] = 120,
    thickness=1,
    numbered=True,
):
    img = img.copy()
    for i, box in enumerate(boxes):
        cv2.rectangle(
            img,
            pt1=(int(box.x1), int(box.y1)),
            pt2=(int(box.x2), int(box.y2)),
            color=color,
            thickness=thickness,
        )
        if numbered:
            cv2.putText(
                img,
                str(i),
                org=(int(box.x1) + 10, int(box.y1) + 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color,
                thickness=thickness
            )
    return img


def draw_mask_centers(
    img: np.ndarray,
    gap_mask: GapMask,
    color: Union[int, Tuple[int, int, int]],
    thickness=3,
    copy_img=True,
) -> np.ndarray:
    if copy_img:
        img = img.copy()
    if gap_mask.centers is None:
        return img
    for pos in gap_mask.centers:
        slc = slice(int(pos - thickness / 2), int(pos + thickness / 2))
        if gap_mask.direction is Axis.X:
            img[:, slc] = color
        elif gap_mask.direction is Axis.Y:
            img[slc, :] = color
        else:
            raise ValueError(f'Incorrect axis value {gap_mask.direction}')
    return img
