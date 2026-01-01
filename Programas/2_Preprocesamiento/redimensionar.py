import cv2
import numpy as np

def make_square(image, desired_size=224):
    """Redimensiona con proporción y padding para formar imagen cuadrada."""
    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img_resized = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [128, 128, 128]
    new_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return new_img
