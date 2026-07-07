"""
Select optimal anomaly threshold for PatchCore deployment.

Based on validation scores, picks threshold at target FPR (false positive rate).
"""
import os
import sys
import argparse
import json
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchcore.config import PATCHCORE_MODEL_DIR


def select_threshold(
    scores_path: str,
    target_fpr: float = 0.05,
    config_path: str = "configs/dual_branch_config.yaml",
):
    """
    Select anomaly threshold and update config file.

    Args:
        scores_path: path to scores.npz from validate_patchcore.py
        target_fpr: desired false positive rate (0.01, 0.05, 0.10)
        config_path: path to YAML config to update
    """
    data = np.load(scores_path)
    scores = data["scores"]
    ground_truths = data["ground_truths"]

    # Get good-weld scores only
    good_scores = scores[ground_truths == 0]

    # Threshold at given percentile of good scores
    threshold = float(np.percentile(good_scores, (1 - target_fpr) * 100))

    # Compute TPR at this threshold
    defect_scores = scores[ground_truths == 1]
    tpr = float(np.mean(defect_scores >= threshold))
    fpr_actual = float(np.mean(good_scores >= threshold))

    print(f"Threshold selection (target FPR={target_fpr:.0%}):")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Actual FPR: {fpr_actual:.4f} ({fpr_actual*100:.2f}%)")
    print(f"  Actual TPR: {tpr:.4f} ({tpr*100:.2f}%)")
    print(f"  Good samples: {len(good_scores)}")
    print(f"  Defect samples: {len(defect_scores)}")

    # Update config
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config.setdefault("model", {}).setdefault("patchcore", {})
    config["model"]["patchcore"]["anomaly_threshold"] = float(threshold)
    config["model"]["patchcore"]["selected_fpr"] = float(target_fpr)

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nThreshold saved to: {config_path}")
    print(f"  model.patchcore.anomaly_threshold = {threshold:.6f}")

    return threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select PatchCore anomaly threshold")
    parser.add_argument("--scores", default="runs/patchcore_validation/scores.npz",
                        help="Path to scores.npz")
    parser.add_argument("--fpr", type=float, default=0.05,
                        help="Target false positive rate")
    parser.add_argument("--config", default="configs/dual_branch_config.yaml",
                        help="Config file to update")
    args = parser.parse_args()

    select_threshold(args.scores, args.fpr, args.config)
