"""
Fusion logic for dual-branch defect detection.

OR logic: either YOLO or PatchCore triggers → NG (defect detected).
"""
from typing import List, Dict, Tuple, Optional


def fusion(
    yolo_detections: List[List],
    anomaly_score: Optional[float],
    anomaly_threshold: float,
    anomaly_active: bool = True,
) -> Tuple[bool, List[str], Dict]:
    """
    OR fusion logic for dual-branch inspection.

    Args:
        yolo_detections: list of [x1, y1, x2, y2, conf, cls_id] from YOLO
        anomaly_score: PatchCore anomaly score (or None if skipped/not computed)
        anomaly_threshold: threshold for anomaly classification
        anomaly_active: whether PatchCore ran this frame

    Returns:
        is_defect: bool — overall NG/OK decision
        sources: list of str — which branches triggered
        details: dict — per-branch diagnostic info
    """
    sources = []

    # Branch A: YOLO — known defect classes (0=Crack, 1=Porosity, 2=Spatters)
    yolo_ng = False
    yolo_defect_count = 0
    if yolo_detections:
        defect_detections = [d for d in yolo_detections if d[5] in {0, 1, 2}]
        yolo_defect_count = len(defect_detections)
        if yolo_defect_count > 0:
            yolo_ng = True
            sources.append("yolo")

    # Branch B: PatchCore — unknown anomalies
    patchcore_ng = False
    if anomaly_score is not None and anomaly_active:
        if anomaly_score > anomaly_threshold:
            patchcore_ng = True
            sources.append("patchcore")

    # OR fusion
    is_defect = yolo_ng or patchcore_ng

    details = {
        "yolo_ng": yolo_ng,
        "yolo_defect_count": yolo_defect_count,
        "yolo_total_boxes": len(yolo_detections),
        "patchcore_ng": patchcore_ng,
        "anomaly_score": float(anomaly_score) if anomaly_score is not None else None,
        "anomaly_threshold": float(anomaly_threshold),
        "anomaly_active": anomaly_active,
    }

    return is_defect, sources, details
