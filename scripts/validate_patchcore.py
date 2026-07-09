"""
Validate PatchCore anomaly detection performance.

Runs PatchCore inference on the validation set (800 images),
computes AUROC, F1 score, and identifies optimal threshold.
Uses onnxruntime for inference (simulates K1 behavior).
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import onnxruntime as ort
import cv2
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchcore.config import (
    INPUT_SIZE, POOL_SIZE, FEATURE_DIM, NUM_PATCHES, TOP_K,
    IMAGENET_MEAN, IMAGENET_STD, PATCHCORE_MODEL_DIR, DATA_YAML,
)
from patchcore.anomaly_scorer import compute_anomaly_scores
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve


def load_models(onnx_path: str, bank_path: str):
    """Load PatchCore ONNX backbone, memory bank, and optional PCA transform."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    memory_bank = np.load(bank_path).astype(np.float32)
    memory_bank = memory_bank / (np.linalg.norm(memory_bank, axis=1, keepdims=True) + 1e-8)

    pca_dir = os.path.dirname(bank_path)
    pca_matrix = None
    pca_mean = None
    pca_path = os.path.join(pca_dir, "pca_matrix.npy")
    pca_mean_path = os.path.join(pca_dir, "pca_mean.npy")
    if os.path.exists(pca_path) and os.path.exists(pca_mean_path):
        pca_matrix = np.load(pca_path).astype(np.float32)
        pca_mean = np.load(pca_mean_path).astype(np.float32)
        print(f"  PCA: {pca_matrix.shape[0]} -> {pca_matrix.shape[1]} dims")

    return session, memory_bank, pca_matrix, pca_mean


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Preprocess image for PatchCore: resize, normalize to ImageNet stats."""
    img = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)    # (1, 3, 256, 256)
    return img


def get_ground_truth(label_path: str) -> bool:
    """
    Returns True if image has defects (class 0,1,2), False if good.
    """
    if not os.path.exists(label_path):
        return False  # no label = assume good
    with open(label_path, "r") as f:
        content = f.read().strip()
    if not content:
        return False
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        cls_id = int(line.split()[0])
        if cls_id in {0, 1, 2}:
            return True  # has defect
    return False


def validate(
    onnx_path: str,
    bank_path: str,
    valid_images_dir: str,
    valid_labels_dir: str,
) -> Dict:
    """Run full validation and return metrics."""
    print(f"Loading models...")
    session, memory_bank, pca_matrix, pca_mean = load_models(onnx_path, bank_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Memory bank: {memory_bank.shape}, ONNX input: {session.get_inputs()[0].shape}")

    image_files = sorted([
        f for f in os.listdir(valid_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    scores = []
    ground_truths = []
    inference_times = []

    print(f"\nProcessing {len(image_files)} validation images...")
    for img_file in tqdm(image_files):
        img_path = os.path.join(valid_images_dir, img_file)

        # Ground truth
        stem = os.path.splitext(img_file)[0]
        label_path = os.path.join(valid_labels_dir, f"{stem}.txt")
        is_defect = get_ground_truth(label_path)
        ground_truths.append(1 if is_defect else 0)

        # Inference
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            scores.append(0.0)
            inference_times.append(0)
            continue

        t0 = time.time()
        input_tensor = preprocess_image(img_bgr)
        features = session.run([output_name], {input_name: input_tensor})[0]
        features = features.reshape(NUM_PATCHES, FEATURE_DIM)
        _, score = compute_anomaly_scores(features, memory_bank, top_k=TOP_K,
                                            pca_matrix=pca_matrix, pca_mean=pca_mean)
        t = (time.time() - t0) * 1000

        scores.append(score)
        inference_times.append(t)

    scores = np.array(scores)
    ground_truths = np.array(ground_truths)

    # Compute metrics
    auroc = roc_auc_score(ground_truths, scores)

    # Find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(ground_truths, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Metrics at optimal threshold
    preds = (scores >= optimal_threshold).astype(int)
    f1 = f1_score(ground_truths, preds)
    precision = precision_score(ground_truths, preds)
    recall = recall_score(ground_truths, preds)

    # Threshold at FPR=5%
    fpr05_idx = np.argmin(np.abs(fpr - 0.05))
    threshold_fpr05 = thresholds[fpr05_idx]

    # FPR=1%
    fpr01_idx = np.argmin(np.abs(fpr - 0.01))
    threshold_fpr01 = thresholds[fpr01_idx]

    # FPR=10%
    fpr10_idx = np.argmin(np.abs(fpr - 0.10))
    threshold_fpr10 = thresholds[fpr10_idx]

    results = {
        "num_images": int(len(image_files)),
        "num_defects": int(ground_truths.sum()),
        "num_good": int(len(image_files) - ground_truths.sum()),
        "auroc": float(auroc),
        "optimal_threshold": float(optimal_threshold),
        "f1_optimal": float(f1),
        "precision_optimal": float(precision),
        "recall_optimal": float(recall),
        "threshold_fpr_0.01": float(threshold_fpr01),
        "threshold_fpr_0.05": float(threshold_fpr05),
        "threshold_fpr_0.10": float(threshold_fpr10),
        "avg_inference_time_ms": float(np.mean(inference_times)),
        "median_inference_time_ms": float(np.median(inference_times)),
    }

    return results, scores, ground_truths


def main():
    parser = argparse.ArgumentParser(description="Validate PatchCore performance")
    parser.add_argument("--onnx", default=os.path.join(PATCHCORE_MODEL_DIR, "backbone.onnx"),
                        help="Path to backbone.onnx")
    parser.add_argument("--bank", default=os.path.join(PATCHCORE_MODEL_DIR, "memory_bank.npy"),
                        help="Path to memory_bank.npy")
    parser.add_argument("--valid-images", default="valid/images",
                        help="Validation images directory")
    parser.add_argument("--valid-labels", default="valid/labels",
                        help="Validation labels directory")
    parser.add_argument("--output-dir", default="runs/patchcore_validation",
                        help="Output directory for results")
    args = parser.parse_args()

    results, scores, ground_truths = validate(
        args.onnx, args.bank, args.valid_images, args.valid_labels
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("PATCHCORE VALIDATION RESULTS")
    print("=" * 60)
    print(f"Images:       {results['num_images']}")
    print(f"Defects:      {results['num_defects']}")
    print(f"Good:         {results['num_good']}")
    print(f"AUROC:        {results['auroc']:.4f}")
    print(f"F1 (optimal): {results['f1_optimal']:.4f}")
    print(f"Precision:    {results['precision_optimal']:.4f}")
    print(f"Recall:       {results['recall_optimal']:.4f}")
    print(f"Avg infer:    {results['avg_inference_time_ms']:.1f} ms")
    print(f"Med infer:    {results['median_inference_time_ms']:.1f} ms")
    print("-" * 60)
    print("Thresholds:")
    print(f"  Optimal (Youden): {results['optimal_threshold']:.4f}")
    print(f"  FPR=1%:           {results['threshold_fpr_0.01']:.4f}")
    print(f"  FPR=5%:           {results['threshold_fpr_0.05']:.4f}")
    print(f"  FPR=10%:          {results['threshold_fpr_0.10']:.4f}")
    print("=" * 60)

    # Save results
    result_path = os.path.join(args.output_dir, "validation_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    # Save scores for threshold selection
    scores_path = os.path.join(args.output_dir, "scores.npz")
    np.savez(scores_path, scores=scores, ground_truths=ground_truths)
    print(f"Scores saved to: {scores_path}")


if __name__ == "__main__":
    main()
