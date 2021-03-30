import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from table_extractor.bordered_service.models import Image, InferenceTable
from table_extractor.model.table import BorderBox, Cell, Row, Table

logger = logging.getLogger(__name__)

GAPS_ROW_THRESHOLD = 12
GAPS_COLUMN_THRESHOLD = 30
GAP_BREAK_THRESHOLD = 3
AXIS_ALIGNMENT_TOLERANCE = 0.5
LINE_GROUP_DISTANCE = 9
ROW_LINE_THRESHOLD = 0.8

# ROIs with height or width below this threshold are filtered
ROI_DIMENSION_THRESHOLD = 10

ROI_PADDING = 2

CONTOURS_DIM_THRESHOLD = 10


class Axis(int, Enum):
    x = 0
    y = 1


@dataclass
class TableDetectionBBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def origin_and_shape(self):
        """Alternative representation: x, y, width, height"""
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    @property
    def coords(self):
        return self.x1, self.y1, self.x2, self.y2

    def draw(self, img: np.ndarray, color=(0, 255, 0), stroke=2) -> np.ndarray:
        # TODO: add optional parameter to write label of bbox
        new_img = img.copy()
        cv2.rectangle(new_img, self.coords[:2], self.coords[2:], color, stroke)
        return new_img


@dataclass
class TableROI:
    """Region of interest"""

    origin: Tuple[int]  # In np.array coords (y1, x1)
    shape: Tuple[int]
    img: np.ndarray
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_img_limit(cls, img, limit, axis):
        ort_limit = limit[0], img.shape[:2][axis]
        if axis is Axis.y:
            x1, x2 = limit
            y1, y2 = ort_limit
        else:
            x1, x2 = ort_limit
            y1, y2 = limit
        origin = (y1, x1)
        shape = (x2 - x1, y2 - y1)
        crop_img = (
            img[y1 : y1 + shape[0], 0 : x1 + shape[1]]
            if axis is Axis.y
            else img[0 : y1 + shape[0], x1 : x1 + shape[1]]
        )
        return cls(origin, shape, crop_img)

    def crop_with_padding(self, x_pad, y_pad):
        h, w = self.img.shape[:2]
        return TableROI(
            self.origin,
            self.shape,
            self.img[y_pad : h - y_pad, x_pad : w - x_pad],
        )

    @property
    def area(self):
        return self.shape[0] * self.shape[1]

    def set_mask(
        self, v_thres=GAP_BREAK_THRESHOLD, h_thres=GAP_BREAK_THRESHOLD
    ):
        mask_h = get_column_mask(self.img, gap_thres=h_thres)
        mask_v = get_row_mask(self.img, gap_thres=v_thres)
        self.mask = np.logical_or(mask_v, mask_h)


@dataclass
class Line:
    coords: tuple
    bbox: Optional[TableDetectionBBox] = None

    @classmethod
    def from_line_group(
        cls,
        img: np.ndarray,
        group: Iterable["Line"],
        tolerance=AXIS_ALIGNMENT_TOLERANCE,
    ) -> "Line":
        max_y, max_x = (0, 0)
        min_y, min_x = img.shape[:2]
        for line in group:
            x1, y1, x2, y2 = line.coords
            max_x = max(x1, x2, max_x)
            min_x = min(x1, x2, min_x)
            max_y = max(y1, y2, max_y)
            min_y = min(y1, y2, min_y)
        bbox = TableDetectionBBox(min_x, min_y, max_x, max_y)
        axis = group[0].get_axis_alignment(tolerance)
        if axis is Axis.x:
            avg_y = int(mean((min_y, max_y)))
            coords = (min_x, avg_y, max_x, avg_y)
        elif axis is Axis.y:
            avg_x = int(mean((min_x, max_x)))
            coords = (avg_x, min_y, avg_x, max_y)
        else:
            # FIXME: decide whether it should raise an exception, or how to treat this unexpected scenario
            logging.warning(
                "Line group is not aligned with any axis, merge will produce inaccurate value"
            )
            coords = group[0].coords
        return cls(coords, bbox)

    @property
    def length(self):
        return get_vector_length(*self.coords)

    @property
    def angle(self):
        """Return the angle between line and Y axis. """
        x, y = get_unit_vector(*self.coords)
        # Flip vector horizontally if its angle is not between 0 and 180 deg.
        if x < 0:
            x, y = -x, -y
        return math.atan2(x, y) * 180 / math.pi

    def get_axis_alignment(
        self, tolerance=AXIS_ALIGNMENT_TOLERANCE
    ) -> Optional[Axis]:
        if (
            (180 - tolerance) <= self.angle and self.angle > 90
        ) or self.angle <= 0 + tolerance:
            return Axis.y
        elif 90 - tolerance <= self.angle <= 90 + tolerance:
            return Axis.x
        return None


def draw_boxes(img, boxes, origin=(0, 0), color=(0, 255, 0), stroke=2):
    new_img = img.copy()
    for box in boxes:
        x, y, w, h = box
        x += origin[0]
        y += origin[1]
        cv2.rectangle(new_img, (x, y), (x + w, y + h), color, stroke)
    return new_img


def can_be_a_gap(window: List[float], bg_value, threshold):
    length = 0
    for value in window:
        if value != bg_value:
            length += 1
            if length > threshold:
                return False
        else:
            length = 0
    return True


def filter_gaps(gaps_1d: List[bool], threshold):
    # TODO: make adaptive threshold
    filtered_gaps = []
    ctr = 0
    for val in gaps_1d:
        if val:
            ctr += 1
        else:
            filtered_gaps += [True if ctr >= threshold else False] * ctr
            ctr = 1
    else:
        filtered_gaps += [True] * ctr
    return filtered_gaps


def find_gaps(img, axis, bg_value, threshold=3):
    if axis not in (0, 1):
        raise ValueError("Axis value should be either 0 or 1")
    gaps = []
    for i in range(img.shape[axis]):
        # 1d sliding window
        y_slice, x_slice = (
            (slice(None), slice(i, i + 1))
            if axis
            else (slice(i, i + 1), slice(None))
        )
        window = img[y_slice, x_slice].ravel()
        gaps.append(
            True if can_be_a_gap(window, bg_value, threshold) else False
        )
    return gaps


def gap_to_2d_mask(gaps_1d: List[bool], axis, shape: Tuple[int]):
    shape = shape[:2]
    gaps_1d = np.atleast_2d(gaps_1d)
    if axis:
        gaps_1d = gaps_1d.T
    if axis not in (0, 1):
        raise ValueError("Axis value should be either 0 or 1")
    # length = shape[0 if axis else 1]
    return np.repeat(gaps_1d, shape[axis], axis % 2)


def get_lines_on_orthogonal_axis(img, line: Line):
    if line.get_axis_alignment() is Axis.x:
        x1, _, x2, _ = line.coords
        y1, y2 = 0, img.shape[Axis.y]
        return [Line((x1, y1, x1, y2)), Line((x2, y1, x2, y2))]


def get_column_mask(img, custom_shape=None, gap_thres=3):
    shape = custom_shape if custom_shape is not None else img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    (thresh, img_bin) = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    gaps = find_gaps(img_bin, 1, 255, gap_thres)
    gaps = filter_gaps(gaps, GAPS_COLUMN_THRESHOLD)
    gaps = gap_to_2d_mask(gaps, 0, shape)
    assert gaps.shape[:2] == shape[:2]
    return gaps


def get_row_mask(img, custom_shape=None, gap_thres=3):
    shape = custom_shape if custom_shape is not None else img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    (thresh, img_bin) = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    gaps = find_gaps(img_bin, 0, 255, gap_thres)
    gaps = filter_gaps(gaps, GAPS_ROW_THRESHOLD)
    gaps = gap_to_2d_mask(gaps, 1, shape)
    assert gaps.shape[:2] == shape[:2]
    return gaps


def contours_to_boxes(
    img, contours, threshold=CONTOURS_DIM_THRESHOLD, v_padding=0, h_padding=5
):
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > img.shape[1] - 20:
            continue
        if w < threshold or h < threshold:
            continue
        boxes.append(
            [
                x - h_padding,
                y - v_padding,
                w + h_padding * 2,
                h + v_padding * 2,
            ]
        )
    return boxes


def get_vector_length(x1, y1, x2, y2) -> Tuple[float]:
    x, y = x2 - x1, y2 - y1
    return math.sqrt(x ** 2 + y ** 2)


def get_unit_vector(x1, y1, x2, y2) -> Tuple[float]:
    length = get_vector_length(x1, y1, x2, y2)
    return (x2 - x1) / length, (y2 - y1) / length


def group_lines_by_orientation(
    lines: Iterable[Line], tolerance=AXIS_ALIGNMENT_TOLERANCE
) -> Tuple[List[Line]]:
    h_lines, v_lines = [], []
    for line in lines:
        if line.get_axis_alignment(tolerance) is Axis.y:
            v_lines.append(line)
        elif line.get_axis_alignment(tolerance) is Axis.x:
            h_lines.append(line)
    return h_lines, v_lines


def lines_can_be_merged(l1: Line, l2: Line, max_dist: int, axis: int) -> bool:
    """
    Return True if both lines are within max_dist across given axis and both ends of the lines are aligned
    in axis orthogonal to given axis. If distance in X axis is given then lines expected to be aligned in
    Y axis and vice versa.
    """
    (axis + 1) % 2
    l1 = l1.coords
    l2 = l2.coords
    # FIXME: What about crosslines?
    # ort_intersection = l2[ort_axis] <= l1[ort_axis + 2] <= l2[ort_axis + 2] or l1[ort_axis] <= l2[ort_axis + 2] <= l1[ort_axis + 2]
    return abs(l1[axis] - l2[axis]) <= max_dist  # and ort_intersection


def group_lines_by_distance(
    lines: Iterable[Line], max_dist: int, axis: int
) -> List[List[Line]]:
    if not lines:
        return []
    if len(lines) == 1:
        return [lines]
    sorted_lines = sorted(lines, key=lambda line: line.coords[axis])
    l1 = sorted_lines.pop(0)
    l2 = sorted_lines.pop(0)
    group = [l1]
    line_groups = []
    for i in range(len(sorted_lines)):
        if lines_can_be_merged(l1, l2, max_dist, axis):
            group.append(l2)
        else:
            line_groups.append(group)
            l1 = l2
            group = [l1]
        l2 = sorted_lines.pop(0)
    else:
        if lines_can_be_merged(l1, l2, max_dist, axis):
            group.append(l2)
        else:
            line_groups.append([l2])
        line_groups.append(group)
    return line_groups


def merge_lines(
    img: np.ndarray,
    lines: Iterable[Line],
    axis: int,
    group_distance=LINE_GROUP_DISTANCE,
    tolerance=AXIS_ALIGNMENT_TOLERANCE,
) -> List[Line]:
    line_groups = group_lines_by_distance(lines, group_distance, axis)
    return [
        Line.from_line_group(img, group, tolerance=tolerance)
        for group in line_groups
    ]


def lines_to_limits(
    img, lines: Iterable[Line], axis: int, tolerance=AXIS_ALIGNMENT_TOLERANCE
):
    """Axis should be orthogonal to lines"""
    ort_axis = Axis.x if axis is Axis.y else Axis.y
    filtered_lines = [
        line
        for line in lines
        if line.get_axis_alignment(tolerance) is ort_axis
    ]
    sorted_lines = sorted(filtered_lines, key=lambda l: l.coords[axis])
    img_edge1, img_edge2 = 0, img.shape[axis]
    slices_lst = []
    lim1, lim2 = img_edge1, sorted_lines.pop(0).coords[axis]
    slices_lst.append((lim1, lim2))
    for i in range(len(sorted_lines)):
        lim1 = lim2
        lim2 = sorted_lines.pop(0).coords[axis]
        slices_lst.append((lim1, lim2))
    slices_lst.append((lim2, img_edge2))
    return slices_lst


def is_empty_img(img, threshold=None):
    # TODO: implement thresholding instead of checking all values
    if (
        img.shape[0] < ROI_DIMENSION_THRESHOLD
        or img.shape[1] < ROI_DIMENSION_THRESHOLD
    ):
        return True
    edges = cv2.Canny(img, 50, 150, 200)
    if all(edges.ravel() == 0):
        return True
    return False


def get_header(roi_lst: List[TableROI]):
    # TODO: implement logic for header selection and validation, check gaps distribution
    if len(roi_lst) > 2 and roi_lst[0].origin[1] < 40:
        return roi_lst[1], roi_lst[2:]
    return roi_lst[0], roi_lst[1:]


def parse_header(roi: TableROI, img):
    # TODO: optimize work with lines, do not repeat operations which were done previously
    blur = cv2.GaussianBlur(roi.img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 1)
    if lines_p is None:
        return None, None, None
    lines = [Line(line[0]) for line in lines_p]
    h_lines, v_lines = group_lines_by_orientation(lines)
    merged_h_lines = merge_lines(roi.img, h_lines, Axis.y)

    # FIXME: most likely won't need filtering out row-like lines, also to check that it is a truly concept line
    # should check that it is aligned to the right
    concept_separator = [
        line
        for line in merged_h_lines
        if line.length < img.shape[1] * ROW_LINE_THRESHOLD
    ]
    if concept_separator:
        concept_separator: Line = concept_separator[0]
        s_x1 = min(concept_separator.coords[0::2])
        s_y = concept_separator.coords[1] + roi.origin[0]
        y1, x1 = roi.origin
        y2, x2 = x1 + roi.shape[0], y1 + roi.shape[1]
        regular_header_img = img[y1:y2, x1:s_x1]

        # FIXME: should shapes be inverted?
        regular_header_roi = TableROI(
            roi.origin, (y2 - y1, s_x1 - x1), regular_header_img
        )

        composite_header_top_img = img[y1:s_y, s_x1:x2]
        composite_header_top_roi = TableROI(
            roi.origin, (s_y - y1, x2 - s_x1), composite_header_top_img
        )
        assert (
            composite_header_top_roi.shape
            == composite_header_top_img.shape[:2]
        )
        composite_header_bot_img = img[s_y:y2, s_x1:x2]
        composite_header_bot_roi = TableROI(
            roi.origin, (y2 - s_y, x2 - s_x1), composite_header_bot_img
        )
        assert (
            composite_header_bot_roi.shape
            == composite_header_bot_img.shape[:2]
        )
        return (
            regular_header_roi,
            composite_header_top_roi,
            composite_header_bot_roi,
        )
    else:
        logger.warning("Concept separator is not found")
        return None, None, None


def get_pos_of_max_gap(gap_mask):
    max_gap_center = -1
    max_ctr = -1
    ctr = 0
    start = 0
    for i, val in enumerate(gap_mask):
        if val is True:
            ctr += 1
        else:
            if ctr > max_ctr:
                max_ctr = ctr
                max_gap_center = (start + i) // 2
            ctr = 0
            start = i
    else:
        if ctr > max_ctr:
            max_gap_center = (start + i) // 2
    return max_gap_center


def parse_borderless(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    (thresh, img_bin) = cv2.threshold(
        blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Split header and body
    row_mask_1d = find_gaps(img_bin, 0, 255, 3)
    v_split_pos = get_pos_of_max_gap(row_mask_1d)

    table = TableROI((0, 0), img.shape, img)
    table.set_mask()
    padding = 9
    table.mask[:, :padding] = True
    table.mask[:, table.mask.shape[1] - padding :] = True
    table.mask[:padding, :] = True
    table.mask[table.mask.shape[0] - padding :, :] = True

    mask_array = np.full(table.shape, 0, dtype="int32")
    mask_array[table.mask] = 255

    # FIXME: yes, this is utterly horrible, need more time to deal with type conversions in opencv
    cv2.imwrite("temp.png", mask_array)
    mask_array = cv2.imread("temp.png", 0)
    (thresh, im_bw) = cv2.threshold(mask_array, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = contours_to_boxes(img, contours)
    hy1, hx1 = 0, 0
    hy2, hx2 = hy1 + v_split_pos, hx1 + table.shape[1]
    header_box = [hx1, hy1, hx2, hy2]
    boxes = [
        b
        for b in boxes
        if not is_empty_img(img[b[1] : b[1] + b[3], b[0] : b[0] + b[2]])
    ]
    return boxes, header_box


def parse_semi_bordered(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 9)
    if lines_p is None:
        return parse_borderless(img)

    lines = [Line(line[0]) for line in lines_p]

    h_lines, v_lines = group_lines_by_orientation(lines)
    h_lines = [line for line in h_lines if line.length > 0.5 * img.shape[1]]
    merged_h_lines = merge_lines(img, h_lines, Axis.y)
    merged_v_lines = merge_lines(img, v_lines, Axis.x)

    # FIXME: np axis are (y, x) while lines have axis (x, y)
    row_separator_lines = [
        line
        for line in merged_h_lines
        if line.length > img.shape[1] * ROW_LINE_THRESHOLD
    ]
    if not row_separator_lines:
        img_borderless = img.copy()
        for l in merged_v_lines + merged_h_lines:
            x1, y1, x2, y2 = l.bbox.coords
            img_borderless[y1:y2, x1:x2] = (255, 255, 255)
        return parse_borderless(img_borderless)
    row_limits = lines_to_limits(img, row_separator_lines, Axis.y)
    row_roi_lst = [
        TableROI.from_img_limit(img, lim, Axis.y) for lim in row_limits
    ]

    # Post processing for ROIs
    row_roi_lst = [
        roi.crop_with_padding(ROI_PADDING, ROI_PADDING) for roi in row_roi_lst
    ]
    row_roi_lst = [roi for roi in row_roi_lst if not is_empty_img(roi.img)]

    header_row_roi, body_row_roi_lst = get_header(row_roi_lst)
    head_v_offset, head_h_offset = header_row_roi.origin
    (
        regular_header_roi,
        composite_header_top_roi,
        composite_header_bot_roi,
    ) = parse_header(header_row_roi, img)
    if regular_header_roi:
        assert (
            regular_header_roi.shape[1]
            + composite_header_top_roi.shape[1]
            + head_h_offset
            == img.shape[1]
        )
        assert (
            regular_header_roi.shape[0]
            == composite_header_top_roi.shape[0]
            + composite_header_bot_roi.shape[0]
        )
        assert regular_header_roi.shape[0] == header_row_roi.shape[0]
        regular_header_roi.set_mask()
        composite_header_top_roi.set_mask()
        composite_header_bot_roi.set_mask()

        body_v_offset = head_v_offset + header_row_roi.shape[0]
        potential_body_end = row_separator_lines[-1].coords[1]
        # FIXME: replace magic number which denotes that if there is a line in 10% distance from
        # the bottom of image this line is considered end line of table
        body_end = (
            potential_body_end
            if potential_body_end > img.shape[0] * 0.9
            else img.shape[0]
        )
        table_body_img = img[body_v_offset:body_end, :]
        body_row_roi = TableROI(
            (body_v_offset, 0), table_body_img.shape[:2], table_body_img
        )
        body_row_roi.set_mask()

        header_shape = (body_v_offset, img.shape[1])
        table_mask = np.full(header_shape, True)
        # FIXME: array stacking is slow
        concept_mask = np.concatenate(
            [composite_header_top_roi.mask, composite_header_bot_roi.mask],
            axis=0,
        )
        header_mask = np.concatenate(
            [regular_header_roi.mask, concept_mask], axis=1
        )

        # offset correction
        table_mask[
            head_v_offset : header_mask.shape[0] + head_v_offset,
            head_h_offset : header_mask.shape[1] + head_h_offset,
        ] = header_mask
        table_mask = np.concatenate([table_mask, body_row_roi.mask], axis=0)
        mask_array = np.full(table_mask.shape, 0, dtype="int32")
    else:
        potential_body_end = row_separator_lines[-1].coords[1]
        # FIXME: replace magic number which denotes that if there is a line in 10% distance from
        # the bottom of image this line is considered end line of table
        body_end = (
            potential_body_end
            if potential_body_end > img.shape[0] * 0.9
            else img.shape[0]
        )
        table_body_img = img[0:body_end, :]
        body_row_roi = TableROI(
            (0, body_end), table_body_img.shape[:2], table_body_img
        )
        body_row_roi.set_mask()
        table_mask = body_row_roi.mask
        mask_array = np.full(body_row_roi.shape, 0, dtype="int32")

    for l in merged_v_lines:
        x1, y1, x2, y2 = l.bbox.coords
        table_mask[y1:y2, x1:x2] = True
    mask_array[table_mask] = 255
    mask_array[:, :10] = 255

    # FIXME: yes, this is utterly horrible, need more time to deal with type conversions in opencv
    cv2.imwrite("temp.png", mask_array)
    mask_array = cv2.imread("temp.png", 0)
    (thresh, im_bw) = cv2.threshold(mask_array, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = contours_to_boxes(img, contours)
    boxes = [
        b
        for b in boxes
        if not is_empty_img(img[b[1] : b[1] + b[3], b[0] : b[0] + b[2]])
    ]
    if regular_header_roi:
        hy1, hx1 = (0, 0)
        hy2, hx2 = hy1 + header_shape[0], hx1 + header_shape[1]
        header_box = [hx1, hy1, hx2, hy2]
    else:
        hy1, hx1 = (0, 0)
        header_candidate = row_separator_lines[0]
        if (
            len(row_separator_lines) > 1
            and row_separator_lines[0].coords[1] < 40
        ):
            header_candidate = row_separator_lines[1]
        hy2, hx2 = hy1 + header_candidate.coords[1], hx1 + img.shape[1]
        header_box = [hx1, hy1, hx2, hy2]
    return boxes, header_box


def boxes_to_image_obj(path: Path, boxes, origin):
    y0, x0 = origin
    # Convert (x,y,w,h -> x1,x2,y1,y2 and shift according to origin)
    new_boxes = []
    for box in boxes:
        x, y, w, h = box
        new_boxes.append(Cell(x + x0, y + y0, x + x0 + w, y + y0 + h))
    img_data = Image(path, None, None, objs=new_boxes, bboxes=[])
    # boxes = [(b[0] + x0, b[1] + y0, b[2], b[3]) for b in boxes]
    return img_data


def box_to_cell(box, table_origin_shift) -> Cell:
    y0, x0 = table_origin_shift
    x, y, w, h = box
    return Cell(x + x0, y + y0, x + x0 + w, y + y0 + h)


def construct_rows_from_boxes(cells: List[Cell], x_max) -> List[Row]:
    h_lines = {}

    for box in sorted(cells, key=lambda x: (x.top_left_x, x.top_left_y)):
        h_line_key = box[1]
        if h_line_key not in h_lines:
            row = Row(
                bbox=BorderBox(box[0], box[1], x_max, box[3]),
                table_id=1,
            )
            row.add(box)
            h_lines[h_line_key] = row
        else:
            h_lines[h_line_key].add(box)

    return list(h_lines.values())


def semi_bordered(
    page_img: np.ndarray, inference_table: InferenceTable
) -> Optional[Table]:
    top_left_x = inference_table.bbox.top_left_x
    top_left_y = inference_table.bbox.top_left_y
    bottom_right_x = inference_table.bbox.bottom_right_x
    bottom_right_y = inference_table.bbox.bottom_right_y
    table_image = page_img[
        top_left_y:bottom_right_y, top_left_x:bottom_right_x
    ]
    table_origin_shift = (top_left_y, top_left_x)  # (y1, x1)
    # TODO: rewrite try ... catch
    try:
        boxes, _ = parse_semi_bordered(table_image)
    except Exception as e:
        logger.warning(str(e))
        return None
    if not boxes:
        return None
    cells = [box_to_cell(box, table_origin_shift) for box in boxes]

    rows = construct_rows_from_boxes(cells, bottom_right_x)

    # TODO: find also cols
    table = Table(bbox=inference_table.bbox, table_id=0, cols=[], rows=rows)

    return table
