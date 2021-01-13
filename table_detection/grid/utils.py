from typing import List

from sortedcontainers import SortedSet

from table_detection.common import BBox


def bboxes_from_coordinates(x_coords: SortedSet, y_coords: SortedSet) -> List[BBox]:
    """
    Create boundary boxes from intersections between horizontal and vertical lines
    :param x_coords: list of x coordinates to represent vertical lines
        (x = C for any y)
    :param y_coords: list of y coordinates to represent horizontal lines
        (y = C for any x)
    :return: List of boundary boxes
    """
    bboxes = []
    for x1 in x_coords:
        for y1 in y_coords:
            xr_idx = x_coords.index(x1) + 1
            if xr_idx >= len(x_coords):
                continue
            xr = x_coords[xr_idx]
            yb_idx = y_coords.index(y1) + 1
            if yb_idx >= len(y_coords):
                continue
            yb = y_coords[yb_idx]
            bboxes.append(BBox(x1, y1, xr, yb))
    return bboxes
