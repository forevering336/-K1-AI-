"""
Postprocessing for YOLOv8n detections.

Parses raw ONNX output into bounding boxes with NMS.
Only classes 0/1/2 (Crack, Porosity, Spatters) are considered defects.
Class 3 (Welding line) is the weld seam itself — not a defect.
"""
import numpy as np
from typing import List, Tuple


CLASS_NAMES = ['Crack', 'Porosity', 'Spatters', 'Welding line']
DEFECT_CLASSES = {0, 1, 2}  # Only these trigger NG


def parse_yolo_output(
    output: np.ndarray,
    conf_threshold: float = 0.5,
    input_size: int = 640,
) -> List[List]:
    """
    Parse YOLOv8 ONNX output to bounding boxes.

    YOLOv8 output format: (1, 84, 8400) where 84 = 4 (bbox) + 4 (classes)
    or (1, 4 + num_classes, num_anchors)

    Args:
        output: raw ONNX output
        conf_threshold: minimum confidence
        input_size: model input size (for bbox coordinate scaling)

    Returns:
        detections: list of [x1, y1, x2, y2, confidence, class_id]
    """
    output = np.squeeze(output)  # (84, 8400) or similar

    num_classes = 4
    detections = []

    # YOLOv8n output: first 4 rows are bbox center (cx, cy, w, h)
    # Remaining rows are class scores
    for i in range(output.shape[1]):
        bbox = output[:4, i]       # cx, cy, w, h
        scores = output[4:4 + num_classes, i]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])

        if score < conf_threshold:
            continue

        cx, cy, w, h = bbox
        # Convert center to corner (in normalized 0-1 coords)
        x1 = float((cx - w / 2))
        y1 = float((cy - h / 2))
        x2 = float((cx + w / 2))
        y2 = float((cy + h / 2))

        # Clamp to [0, 1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1, x2), min(1, y2)

        detections.append([x1, y1, x2, y2, score, class_id])

    return detections


def nms(
    detections: List[List],
    iou_threshold: float = 0.45,
) -> List[List]:
    """Non-maximum suppression."""
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []

    while len(detections) > 0:
        best = detections[0]
        keep.append(best)
        detections = detections[1:]

        filtered = []
        for d in detections:
            if best[5] != d[5]:  # different class
                filtered.append(d)
                continue

            # IoU
            xi1 = max(best[0], d[0])
            yi1 = max(best[1], d[1])
            xi2 = min(best[2], d[2])
            yi2 = min(best[3], d[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            area_b = (best[2] - best[0]) * (best[3] - best[1])
            area_d = (d[2] - d[0]) * (d[3] - d[1])
            iou = inter / (area_b + area_d - inter + 1e-6)

            if iou < iou_threshold:
                filtered.append(d)
        detections = filtered

    return keep


def scale_detections(
    detections: List[List],
    original_shape: Tuple[int, int],
) -> List[List]:
    """
    Scale normalized detections to original image coordinates.

    Args:
        detections: list of [x1, y1, x2, y2, conf, cls] in [0,1]
        original_shape: (height, width) of original image

    Returns:
        detections with pixel coordinates
    """
    h_orig, w_orig = original_shape[:2]
    scaled = []
    for d in detections:
        x1, y1, x2, y2, conf, cls_id = d
        scaled.append([
            int(x1 * w_orig), int(y1 * h_orig),
            int(x2 * w_orig), int(y2 * h_orig),
            conf, cls_id,
        ])
    return scaled


def postprocess_yolo(
    output: np.ndarray,
    original_shape: Tuple[int, int],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> List[List]:
    """
    Full YOLO postprocessing pipeline.

    Returns:
        detections: list of [x1, y1, x2, y2, confidence, class_id] in pixel coords
    """
    detections = parse_yolo_output(output, conf_threshold)
    detections = nms(detections, iou_threshold)
    detections = scale_detections(detections, original_shape)
    return detections


def filter_defects(detections: List[List]) -> List[List]:
    """
    Filter detections to only actual defects (classes 0, 1, 2).
    Welding line (class 3) is NOT a defect.
    """
    return [d for d in detections if d[5] in DEFECT_CLASSES]
