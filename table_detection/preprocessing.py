import cv2
import numpy as np


def to_binary_img(grayscale_img, kernel=(3, 3), lower_thres=128, upper_thres=255):
    blurred = cv2.GaussianBlur(grayscale_img, kernel, 0)
    thresh, img_bin = cv2.threshold(
        blurred, lower_thres, upper_thres, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return img_bin


def create_frame(grayscale_img, size, color):
    img = grayscale_img.copy()
    h, w = img.shape[:2]
    if size > h // 2 or size > w // 2:
        raise ValueError(
            f'Unable to create frame of size {size} for image of shape {(h, w)}'
        )
    img[:, 0:size] = color
    img[0:size, :] = color
    img[:, w - size : w] = color
    img[h - size : h, :] = color
    return img


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] != 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    return img_gray
