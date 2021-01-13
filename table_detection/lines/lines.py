import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from table_detection.common import \
    BBox  # TODO: replace with commonly used bbox
from table_detection.common import Axis, Point, Vector

AXIS_ALIGNMENT_TOLERANCE = 0.5


@dataclass
class Line:
    vector: Vector
    bbox: Optional[BBox] = None

    @classmethod
    def from_coords(
        cls, line: Tuple[int, int, int, int], bbox: Optional[BBox] = None
    ) -> 'Line':
        x1, y1, x2, y2 = line
        return cls(Vector(Point(x1, y1), Point(x2, y2)), bbox=bbox)

    @property
    def coords(self):
        return (
            self.vector.pt1.x,
            self.vector.pt1.y,
            self.vector.pt2.x,
            self.vector.pt2.y,
        )

    @property
    def center(self):
        x1, y1, x2, y2 = self.coords
        return Point((x1 + x2) // 2, (y1 + y2) // 2)

    @classmethod
    def from_line_group(cls, group: 'LineGroup') -> 'Line':
        max_y, max_x = 0, 0
        min_y, min_x = math.inf, math.inf
        for line in group.lines:
            x1, y1, x2, y2 = line.coords
            max_x = max(x1, x2, max_x)
            min_x = min(x1, x2, min_x)
            max_y = max(y1, y2, max_y)
            min_y = min(y1, y2, min_y)
        bbox = BBox(min_x, min_y, max_x, max_y)
        if group.orientation is Axis.X:
            avg_y = (min_y + max_y) // 2
            vector = Vector.from_coords((min_x, avg_y, max_x, avg_y))
        elif group.orientation is Axis.Y:
            avg_x = (min_x + max_x) // 2
            vector = Vector.from_coords((avg_x, min_y, avg_x, max_y))
        return cls(vector, bbox)

    @property
    def y_angle(self) -> float:
        """Return the the smallest angle between line and Y axis."""
        x, y = self.vector.unit_vector.pt2
        # Flip vector if its angle is not between 0 and 180 deg.
        if x < 0:
            x, y = -x, -y
        return math.atan2(x, y) * 180 / math.pi

    def get_axis_alignment(self, tolerance=AXIS_ALIGNMENT_TOLERANCE) -> Optional[Axis]:
        if (
            (180 - self.y_angle) <= (180 - tolerance) and self.y_angle > 90
        ) or self.y_angle <= 0 + tolerance:
            return Axis.Y
        elif 90 - tolerance <= self.y_angle <= 90 + tolerance:
            return Axis.X
        return None


class LineGroup:
    def __init__(self, lines: List[Line], tolerance: float):
        self.lines = lines
        self.tolerance = tolerance
        self.orientation = self._get_group_orientation(lines, tolerance)

    @staticmethod
    def _get_group_orientation(group, tolerance):
        axis_set = {line.get_axis_alignment(tolerance) for line in group}
        if None in axis_set:
            raise ValueError(
                'Unable to create a new line from group which contains lines that are'
                'not aligned with any axis'
            )
        elif len(axis_set) != 1:
            raise ValueError(f'Group has lines aligned with different axis: {axis_set}')
        return axis_set.pop()
