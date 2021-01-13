from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from table_detection.common import Axis, get_orthogonal


@dataclass
class GapMask:
    """
    Represents visual gaps between rows or columns.
    :param: array: 2d array of boolean, dimensions (n, 1) or (1, n),
        True values correspond to gaps
    :param: direction: orientation of the mask array along the axis
    """

    array: np.ndarray
    direction: Axis
    centers: Optional[List[int]] = None

    def __post_init__(self):
        self.array = np.atleast_2d(self.array)
        if self.direction is Axis.Y:
            self.array = self.array.T

    def filter(self, threshold, filter_value):
        """
        Count values of the same type in a row, if their count is below threshold, the
        type of all values is swapped to the opposite.

        :param threshold:
        :param filter_value: when True gap values are filtered, otherwise
            non-gap values are filtered
        :return:
        """
        filtered = []
        centers = []
        ctr = 0
        for val in self.array.ravel():
            if bool(val) is filter_value:
                ctr += 1
            else:
                sequence = [
                    filter_value if ctr >= threshold else not filter_value
                ] * ctr
                if ctr >= threshold:
                    centers.append(len(filtered) + len(sequence) // 2)
                filtered += sequence
                ctr = 1
        else:
            sequence = [filter_value if ctr >= threshold else not filter_value] * ctr
            centers.append(len(filtered) + len(sequence) // 2)
            filtered += sequence
        return GapMask(filtered, self.direction, centers)  # TODO: find gap centers

    def to_2d_mask(self, mask_shape: Tuple[int]) -> np.ndarray:
        """
        Extend mask along the axis orthogonal to `self.direction`.
        This method is required to combine masks.

        Example: given the mask of size (n, 1) (i.e window with height = n [px]
        and width = 1 [px]), it is aligned with Y axis so the direction is set
        accordingly. This mask may be extended horizontally to the shape of original
        image. Extended vertical mask can be combined with extended horizontal mask
        to find boxes with some (non-background) content on the image.

        :param mask_shape: desired shape of 2d mask
        :return: 2d mask of the given shape
        """
        mask_shape = mask_shape[:2]
        if mask_shape[self.direction] != self.array.shape[self.direction]:
            raise ValueError(
                (
                    f'Can not create mask of shape {mask_shape} '
                    f'from array of shape {self.array.shape}'
                )
            )
        expansion_direction = get_orthogonal(self.direction)
        return np.repeat(
            self.array, mask_shape[expansion_direction], expansion_direction
        )
