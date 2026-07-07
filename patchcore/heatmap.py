"""
Anomaly heatmap generation for PatchCore.

Generates a spatial heatmap from patch-level anomaly distances,
upsampled to the original image size for overlay visualization.
"""
import cv2
import numpy as np
from typing import Tuple


def generate_heatmap(
    patch_distances: np.ndarray,
    original_shape: Tuple[int, int],
    pool_size: int = 16,
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """
    Generate anomaly heatmap from patch-level distances.

    Args:
        patch_distances: (N_patches,) = (256,) for 16x16 grid
        original_shape: (height, width) of the original image
        pool_size: spatial grid size (16 for 16x16)
        gaussian_sigma: sigma for Gaussian blur smoothing

    Returns:
        heatmap: (H_orig, W_orig) float32, normalized to [0, 1]
    """
    # Reshape to spatial grid
    anomaly_map = patch_distances.reshape(pool_size, pool_size)

    # Gaussian smoothing to fill gaps
    anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=gaussian_sigma)

    # Upsample to original image size
    h_orig, w_orig = original_shape[:2]
    heatmap = cv2.resize(
        anomaly_map,
        (w_orig, h_orig),
        interpolation=cv2.INTER_LINEAR,
    )

    # Normalize to [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


def overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay anomaly heatmap on original BGR image.

    Args:
        image_bgr: original image (H, W, 3) BGR
        heatmap: (H, W) float32 in [0, 1]
        alpha: blending weight for heatmap (0=only image, 1=only heatmap)
        colormap: OpenCV colormap code

    Returns:
        overlay: (H, W, 3) BGR image with heatmap overlay
    """
    # Apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Alpha blend
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay
