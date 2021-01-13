from typing import List, Tuple

from table_detection.common import Axis
from table_detection.lines.lines import Line, LineGroup


def select_basis_lines(lines: List[Line], tolerance) -> Tuple[List[Line], List[Line]]:
    """Select lines aligned with axis"""
    horizontal_lines, vertical_lines = [], []
    for line in lines:
        if line.get_axis_alignment(tolerance) is Axis.Y:
            vertical_lines.append(line)
        elif line.get_axis_alignment(tolerance) is Axis.X:
            horizontal_lines.append(line)
    return horizontal_lines, vertical_lines


def lines_can_be_merged(
    l1: Line,
    l2: Line,
    max_dist_ort: int,
    max_gap: int,
    axis: Axis,
    tolerance: float,
) -> bool:
    """Return True if both lines are within max_dist along given axis."""
    l1_alignment = l1.get_axis_alignment(tolerance)
    l2_alignment = l2.get_axis_alignment(tolerance)
    max_gap_dist = max(l2.vector.length, l1.vector.length) * max_gap
    if l1_alignment != l2_alignment:
        raise ValueError('Trying to merge lines with different orientation')
    elif l1_alignment is Axis.X:
        shortest_x_dist = min(
            [abs(l1x - l2x) for l1x in l1.coords[0::2] for l2x in l2.coords[0::2]]
        )
        ort_dist = abs(l1.center.y - l2.center.y)
        return ort_dist <= max_dist_ort and shortest_x_dist < max_gap_dist
    elif l1_alignment is Axis.Y:
        shortest_y_dist = min(
            [abs(l1y - l2y) for l1y in l1.coords[1::2] for l2y in l2.coords[1::2]]
        )
        ort_dist = abs(l1.center.x - l2.center.x)
        return ort_dist <= max_dist_ort and shortest_y_dist < max_gap_dist
    raise ValueError(f'Incorrect axis value {axis}')


def group_lines_by_distance(
    lines: List[Line], max_dist_ort: int, max_gap: int, axis: Axis, tolerance: float
) -> List[LineGroup]:
    """Create groups of lines that are close to each other"""
    if len(lines) == 0:
        return []
    elif len(lines) == 1:
        return lines
    sorted_lines = sorted(
        lines, key=lambda line: line.center[0 if axis is Axis.X else 1]
    )
    l1 = sorted_lines.pop(0)
    l2 = sorted_lines.pop(0)
    group = [l1]
    line_groups = []
    for i in range(len(sorted_lines)):
        if lines_can_be_merged(l1, l2, max_dist_ort, max_gap, axis, tolerance):
            group.append(l2)
        else:
            line_groups.append(group)
            l1 = l2
            group = [l1]
        l2 = sorted_lines.pop(0)
    else:
        if lines_can_be_merged(l1, l2, max_dist_ort, max_gap, axis, tolerance):
            group.append(l2)
        else:
            line_groups.append([l2])
        line_groups.append(group)
    return [LineGroup(group, tolerance) for group in line_groups]


def merge_lines(
    lines: List[Line],
    axis: Axis,
    max_dist_ort: int,
    max_gap: int,
    tolerance: float,
) -> List[Line]:
    line_groups = group_lines_by_distance(lines, max_dist_ort, max_gap, axis, tolerance)
    return [Line.from_line_group(group) for group in line_groups]
