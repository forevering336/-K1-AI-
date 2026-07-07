"""
Dual-branch welding defect inspection system.

Branch A: YOLOv8n ONNX — detects known defects (Crack, Porosity, Spatters)
Branch B: PatchCore ONNX — detects unknown anomalies
Fusion: OR logic — either branch triggers → NG

Runs on K1 (RISC-V) with onnxruntime.
Also runs on PC for testing/debugging.
"""
import os
import sys
import time
import argparse
import json
import cv2
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchcore.config import (
    INPUT_SIZE as PC_INPUT_SIZE,
    NUM_PATCHES, FEATURE_DIM, TOP_K,
    PATCHCORE_MODEL_DIR,
)
from patchcore.anomaly_scorer import compute_anomaly_scores
from patchcore.heatmap import generate_heatmap, overlay_heatmap
from deployment.preprocess import preprocess_yolo, preprocess_patchcore
from deployment.postprocess import postprocess_yolo, filter_defects, CLASS_NAMES, DEFECT_CLASSES
from deployment.fusion import fusion

# Colors for bounding boxes (BGR)
COLORS = [
    (0, 0, 255),    # Crack - red
    (0, 255, 0),    # Porosity - green
    (255, 0, 0),    # Spatters - blue
    (255, 255, 0),  # Welding line - cyan
]

# ROI expansion ratios (from training data analysis: 60454 defect instances)
ROI_PERP_RATIO = 1.0   # perpendicular to weld: 1.0 × short_side (covers ~95%)
ROI_ALONG_RATIO = 0.3  # along the weld: 0.3 × long_side (covers ~95%)
ROI_MIN_MARGIN = 16     # minimum margin in pixels for very small welds


def extract_weld_roi(yolo_detections, frame_w, frame_h):
    """
    Extract the weld region from YOLO detections with directional expansion.

    Uses class 3 (Welding line) bboxes to locate the weld, then expands:
    - Perpendicular to weld: 1.0 × short_side (covers 95% of defects)
    - Along the weld: 0.3 × long_side

    Returns (x1, y1, x2, y2) or None if no weld detected.
    """
    weld_boxes = [d for d in yolo_detections if d[5] == 3]  # class 3 = Welding line
    if not weld_boxes:
        return None

    # Merge all weld boxes into one encompassing bbox
    x1 = min(b[0] for b in weld_boxes)
    y1 = min(b[1] for b in weld_boxes)
    x2 = max(b[2] for b in weld_boxes)
    y2 = max(b[3] for b in weld_boxes)

    weld_w = x2 - x1
    weld_h = y2 - y1
    if weld_w < 2 or weld_h < 2:
        return None

    # Directional expansion
    short_side = min(weld_w, weld_h)
    long_side = max(weld_w, weld_h)

    if weld_w >= weld_h:
        # Horizontal weld: perpendicular = vertical (Y), along = horizontal (X)
        perp_expand = max(ROI_PERP_RATIO * short_side, ROI_MIN_MARGIN)
        along_expand = max(ROI_ALONG_RATIO * long_side, ROI_MIN_MARGIN)
        roi_x1 = max(0, int(x1 - along_expand))
        roi_x2 = min(frame_w, int(x2 + along_expand))
        roi_y1 = max(0, int(y1 - perp_expand))
        roi_y2 = min(frame_h, int(y2 + perp_expand))
    else:
        # Vertical weld: perpendicular = horizontal (X), along = vertical (Y)
        perp_expand = max(ROI_PERP_RATIO * short_side, ROI_MIN_MARGIN)
        along_expand = max(ROI_ALONG_RATIO * long_side, ROI_MIN_MARGIN)
        roi_x1 = max(0, int(x1 - perp_expand))
        roi_x2 = min(frame_w, int(x2 + perp_expand))
        roi_y1 = max(0, int(y1 - along_expand))
        roi_y2 = min(frame_h, int(y2 + along_expand))

    return (roi_x1, roi_y1, roi_x2, roi_y2)


def load_yolo(onnx_path: str) -> "ort.InferenceSession":
    """Load YOLOv8 ONNX model."""
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[YOLO] Loaded: {onnx_path}")
    print(f"  Input:  {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"  Output: {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    return session


def load_patchcore(onnx_path: str, bank_path: str, pca_dir: str = None):
    """Load PatchCore ONNX backbone, memory bank, and optional PCA transform."""
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    memory_bank = np.load(bank_path).astype(np.float32)
    memory_bank = memory_bank / (np.linalg.norm(memory_bank, axis=1, keepdims=True) + 1e-8)

    pca_matrix = None
    pca_mean = None
    if pca_dir:
        pca_path = os.path.join(pca_dir, "pca_matrix.npy")
        pca_mean_path = os.path.join(pca_dir, "pca_mean.npy")
        if os.path.exists(pca_path) and os.path.exists(pca_mean_path):
            pca_matrix = np.load(pca_path).astype(np.float32)
            pca_mean = np.load(pca_mean_path).astype(np.float32)
            print(f"  PCA:    448 -> {pca_matrix.shape[1]} dims")

    print(f"[PatchCore] Loaded: {onnx_path}")
    print(f"  Input:  {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"  Output: {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    print(f"  Bank:   {memory_bank.shape}, {memory_bank.nbytes/1024/1024:.1f} MB")
    return session, memory_bank, pca_matrix, pca_mean


def draw_results(
    frame: np.ndarray,
    yolo_detections: list,
    anomaly_score: Optional[float],
    anomaly_threshold: float,
    heatmap: Optional[np.ndarray],
    sources: list,
    details: dict,
    avg_fps: float,
) -> np.ndarray:
    """Draw detection results on frame."""
    result = frame.copy()

    # Draw ROI (weld region that PatchCore inspects)
    if details.get("roi_coords"):
        rx1, ry1, rx2, ry2 = details["roi_coords"]
        cv2.rectangle(result, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
        cv2.putText(result, "PatchCore ROI", (rx1, ry1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # Draw YOLO bounding boxes
    for det in yolo_detections:
        x1, y1, x2, y2, conf, cls_id = det
        color = COLORS[cls_id % len(COLORS)]
        name = CLASS_NAMES[cls_id]
        label = f"{name}: {conf:.2f}"
        is_defect = cls_id in DEFECT_CLASSES

        thickness = 3 if is_defect else 1
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(result, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Overlay anomaly heatmap if available
    if heatmap is not None and np.max(heatmap) > 0.01:
        result = overlay_heatmap(result, heatmap, alpha=0.35)

    # Status bar
    is_defect = bool(details.get("yolo_ng") or details.get("patchcore_ng"))
    status = "NG" if is_defect else "OK"
    status_color = (0, 0, 255) if is_defect else (0, 255, 0)
    cv2.rectangle(result, (0, 0), (result.shape[1], 90), (40, 40, 40), -1)
    cv2.putText(result, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # Branch info
    y_str = f"YOLO: {'NG' if details.get('yolo_ng') else 'OK'} "
    y_str += f"({details.get('yolo_defect_count', 0)} defects / "
    y_str += f"{details.get('yolo_total_boxes', 0)} boxes)"
    cv2.putText(result, y_str, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    pc_str = f"PatchCore: {'NG' if details.get('patchcore_ng') else 'OK'} "
    pc_str += f"(score={anomaly_score:.4f}" if anomaly_score is not None else "(skipped"
    pc_str += f", thresh={anomaly_threshold:.4f})" if anomaly_score is not None else ")"
    pc_color = (200, 200, 200)
    if not details.get("anomaly_active", True):
        pc_str += " [cached]"
    cv2.putText(result, pc_str, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, pc_color, 1)

    # Sources
    if sources:
        cv2.putText(result, f"Triggered by: {', '.join(sources)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # FPS
    cv2.putText(result, f"FPS: {avg_fps:.1f}", (result.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return result


def run_inference(
    yolo_onnx: str,
    patchcore_onnx: str,
    memory_bank_path: str,
    anomaly_threshold: float = 0.3,
    anomaly_skip: int = 4,
    yolo_conf: float = 0.5,
    yolo_iou: float = 0.45,
    camera_id: int = 0,
    image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    no_display: bool = False,
):
    """Main dual-branch inference loop."""
    import onnxruntime as ort

    print("=" * 60)
    print("Dual-Branch Weld Defect Inspection")
    print("  A: YOLOv8n (known defects)")
    print("  B: PatchCore (unknown anomalies)")
    print("=" * 60)

    # Load models
    session_yolo = load_yolo(yolo_onnx)
    session_pc, memory_bank, pca_matrix, pca_mean = load_patchcore(
        patchcore_onnx, memory_bank_path,
        pca_dir=os.path.dirname(patchcore_onnx))

    yolo_input_name = session_yolo.get_inputs()[0].name
    yolo_output_name = session_yolo.get_outputs()[0].name
    pc_input_name = session_pc.get_inputs()[0].name
    pc_output_name = session_pc.get_outputs()[0].name

    # State
    frame_counter = 0
    cached_anomaly_score = 0.0
    cached_heatmap = None
    cached_anomaly_active = False
    fps_history = []
    detection_log = []  # for CSV logging

    # Input source
    if image_path:
        single_image_mode = True
        print(f"\nSingle image mode: {image_path}")
    else:
        single_image_mode = False
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"\nCamera mode: device {camera_id}")
        print("Press 'q' to quit, 's' to save current frame")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfig: conf={yolo_conf}, iou={yolo_iou}, "
          f"anomaly_thresh={anomaly_threshold}, skip={anomaly_skip}")
    print("-" * 60)

    try:
        while True:
            if single_image_mode:
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"ERROR: Cannot read image: {image_path}")
                    break
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Camera read failed")
                    break

            t_start = time.time()
            h_orig, w_orig = frame.shape[:2]

            # --- Branch A: YOLO (every frame) ---
            yolo_input = preprocess_yolo(frame)
            yolo_output = session_yolo.run([yolo_output_name], {yolo_input_name: yolo_input})
            yolo_detections = postprocess_yolo(yolo_output[0], (h_orig, w_orig),
                                               conf_threshold=yolo_conf,
                                               iou_threshold=yolo_iou)

            # --- Branch B: PatchCore (every N frames, only on weld ROI) ---
            anomaly_score = cached_anomaly_score
            heatmap = cached_heatmap
            anomaly_active = False
            roi_coords = None

            if frame_counter % anomaly_skip == 0:
                roi = extract_weld_roi(yolo_detections, w_orig, h_orig)
                if roi is not None:
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi
                    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    roi_h, roi_w = roi_frame.shape[:2]

                    if roi_w >= 32 and roi_h >= 32:
                        pc_input = preprocess_patchcore(roi_frame)
                        pc_features = session_pc.run(
                            [pc_output_name], {pc_input_name: pc_input})[0]
                        pc_features = pc_features.reshape(NUM_PATCHES, FEATURE_DIM)

                        patch_distances, anomaly_score = compute_anomaly_scores(
                            pc_features, memory_bank, top_k=TOP_K,
                            pca_matrix=pca_matrix, pca_mean=pca_mean)

                        # Generate heatmap at ROI size, then place on full frame
                        heatmap_roi = generate_heatmap(patch_distances, (roi_h, roi_w),
                                                       pool_size=16, gaussian_sigma=4.0)
                        # Embed ROI heatmap into full-frame heatmap
                        heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)
                        heatmap[roi_y1:roi_y2, roi_x1:roi_x2] = heatmap_roi

                        roi_coords = roi
                        anomaly_active = True

                if not anomaly_active:
                    # No weld in frame → reset anomaly score
                    anomaly_score = 0.0
                    heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)

                cached_anomaly_score = anomaly_score
                cached_heatmap = heatmap
                cached_anomaly_active = anomaly_active
            else:
                anomaly_active = cached_anomaly_active

            # --- Fusion ---
            is_defect, sources, details = fusion(
                yolo_detections, anomaly_score, anomaly_threshold,
                anomaly_active=anomaly_active or cached_anomaly_active)
            if roi_coords:
                details["roi_coords"] = roi_coords

            # --- Timing ---
            t_elapsed = (time.time() - t_start) * 1000
            fps_history.append(1.0 / (time.time() - t_start + 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # --- Log ---
            defect_defects = filter_defects(yolo_detections)
            if is_defect:
                defect_names = [CLASS_NAMES[d[5]] for d in defect_defects]
                extra_info = f" + anomaly({anomaly_score:.4f})" if details.get("patchcore_ng") else ""
                print(f"[NG] YOLO: {defect_names}{extra_info} | sources: {sources} | "
                      f"{t_elapsed:.0f}ms")

            # --- Draw ---
            result_frame = draw_results(
                frame, yolo_detections, anomaly_score, anomaly_threshold,
                heatmap, sources, details, avg_fps)

            if not no_display:
                cv2.imshow("Dual-Branch Weld Inspection", result_frame)

            # --- Save ---
            if output_dir and (is_defect or frame_counter % 30 == 0):
                tag = "ng" if is_defect else "ok"
                save_path = os.path.join(output_dir, f"f{frame_counter:06d}_{tag}.jpg")
                cv2.imwrite(save_path, result_frame)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_path = f"screenshot_{frame_counter:06d}.jpg"
                cv2.imwrite(save_path, result_frame)
                print(f"[SAVED] {save_path}")

            frame_counter += 1
            if single_image_mode:
                print(f"\nSingle image processed. Press any key to exit.")
                cv2.waitKey(0)
                break

    finally:
        if not single_image_mode:
            cap.release()
        cv2.destroyAllWindows()

        # Summary
        if fps_history:
            total_time = sum(1.0 / fps for fps in fps_history)
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print(f"  Frames processed: {frame_counter}")
            print(f"  Average FPS:      {avg_fps:.1f}")
            print(f"  Total time:       {total_time:.1f}s")
            print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Dual-branch weld defect inspection")
    parser.add_argument("--yolo-onnx", default="models/best_award_int8.onnx",
                        help="YOLOv8 ONNX model path")
    parser.add_argument("--patchcore-onnx", default="models/backbone3.onnx",
                        help="PatchCore ONNX backbone path")
    parser.add_argument("--memory-bank", default="models/memory_bank2.npy",
                        help="PatchCore memory bank path")
    parser.add_argument("--anomaly-threshold", type=float, default=0.3,
                        help="Anomaly score threshold")
    parser.add_argument("--anomaly-skip", type=int, default=4,
                        help="Run PatchCore every N frames")
    parser.add_argument("--yolo-conf", type=float, default=0.5,
                        help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45,
                        help="YOLO IoU threshold for NMS")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID")
    parser.add_argument("--image", type=str, default=None,
                        help="Process single image instead of camera")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Save output images to directory")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no GUI windows)")
    args = parser.parse_args()

    run_inference(
        yolo_onnx=args.yolo_onnx,
        patchcore_onnx=args.patchcore_onnx,
        memory_bank_path=args.memory_bank,
        anomaly_threshold=args.anomaly_threshold,
        anomaly_skip=args.anomaly_skip,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        camera_id=args.camera,
        image_path=args.image,
        output_dir=args.output_dir,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()
