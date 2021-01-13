import cv2
import numpy as np


def default_line_detection_preprocessing(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.Canny(blur, 50, 150, apertureSize=3)
