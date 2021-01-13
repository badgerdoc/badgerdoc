import math
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

Point = namedtuple('Point', ['x', 'y'])


class Axis(int, Enum):
    Y = 0
    X = 1


@dataclass
class Vector:
    pt1: Point
    pt2: Point

    @property
    def length(self):
        x_delta, y_delta = self.pt2.x - self.pt1.x, self.pt2.y - self.pt1.y
        return math.sqrt(x_delta ** 2 + y_delta ** 2)

    @property
    def unit_vector(self):
        x_delta, y_delta = self.pt2.x - self.pt1.x, self.pt2.y - self.pt1.y
        return Vector(Point(0, 0), Point(x_delta / self.length, y_delta / self.length))

    @classmethod
    def from_coords(cls, coords: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = coords
        return cls(Point(x1, y1), Point(x2, y2))


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        self.x1 = int(self.x1)
        self.y1 = int(self.y1)
        self.x2 = int(self.x2)
        self.y2 = int(self.y2)

    def with_offset(self, offset: Tuple[int, int]):
        x0, y0 = offset
        return BBox(
            self.x1 + x0,
            self.y1 + y0,
            self.x2 + x0,
            self.y2 + y0,
        )

    @property
    def shape(self):
        return self.y2 - self.y1, self.x2 - self.x1

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def is_inside(self, bb2: 'BBox', threshold=0.9) -> bool:
        intersection_area = self.get_boxes_intersection_area(other_box=bb2)
        if intersection_area == 0:
            return False
        return any((intersection_area / bb) > threshold for bb in (self.area, bb2.area))

    def get_boxes_intersection_area(self, other_box) -> float:
        bb1 = self
        bb2 = other_box
        x_left = max(bb1.x1, bb2.x1)
        y_top = max(bb1.y1, bb2.y1)
        x_right = min(bb1.x2, bb2.x2)
        y_bottom = min(bb1.y2, bb2.y2)
        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0.0
        else:
            intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
        return intersection_area


@dataclass
class ROI:
    """Region of interest"""

    shape: Tuple[int, int]
    offset: Tuple[int, int]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        w, h = self.shape
        x0, y0 = self.offset
        return img[y0 : y0 + h, x0 : x0 + w]

    @classmethod
    def from_bbox(cls, bbox: BBox):
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1
        return cls((w, h), (bbox.x1, bbox.y1))


def get_orthogonal(axis: Axis):
    return Axis.X if axis is axis.Y else Axis.Y
