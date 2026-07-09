"""
Preprocessing functions for dual-branch inference.

- preprocess_yolo: 640x640, /255 normalization, CHW format
- preprocess_patchcore: 256x256, ImageNet normalization, CHW format

Both take BGR image and return (1, C, H, W) float32 numpy arrays ready for ONNX.
"""
import cv2
import numpy as np
from patchcore.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def preprocess_yolo(image_bgr: np.ndarray) -> np.ndarray:
    """
    YOLOv8n preprocessing: resize to 640x640, normalize to [0, 1], CHW.

    Args:
        image_bgr: (H, W, 3) BGR image

    Returns:
        (1, 3, 640, 640) float32
    """
    img = cv2.resize(image_bgr, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)    # (1, 3, 640, 640)
    return img.astype(np.float32)


def preprocess_patchcore(image_bgr: np.ndarray) -> np.ndarray:
    """
    PatchCore preprocessing: resize to 256x256, ImageNet normalization, CHW.

    Args:
        image_bgr: (H, W, 3) BGR image

    Returns:
        (1, 3, 256, 256) float32
    """
    img = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)    # (1, 3, 256, 256)
    return img.astype(np.float32)
