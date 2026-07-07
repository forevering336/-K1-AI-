"""
PC-side PatchCore training pipeline.

Steps:
1. Load ResNet-18 backbone (pretrained)
2. Extract features from all good weld images
3. Build memory bank with coreset subsampling
4. Save memory bank + metadata
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchcore.config import (
    INPUT_SIZE, POOL_SIZE, LAYERS, FEATURE_DIM, NUM_PATCHES,
    CORESET_SIZE, IMAGENET_MEAN, IMAGENET_STD, PATCHCORE_MODEL_DIR, GOOD_WELDS_DIR,
)
from patchcore.backbone import create_feature_extractor
from patchcore.memory_bank import build_memory_bank, reduce_dimension_pca


class GoodWeldDataset(Dataset):
    """Dataset of defect-free weld images for PatchCore feature extraction."""

    def __init__(self, image_dir: str, input_size: int = 256):
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}. "
                                    f"Run filter_good_samples.py first.")

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, os.path.basename(path)


def extract_features(
    image_dir: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract PatchCore features from all images in a directory.

    Returns:
        features: (N_total_patches, FEATURE_DIM) — all patch features stacked
    """
    model = create_feature_extractor(pretrained=True)
    model.to(device)
    model.eval()

    dataset = GoodWeldDataset(image_dir, input_size=INPUT_SIZE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    total_images = len(dataset)

    print(f"Extracting features from {total_images} good weld images...")
    print(f"Device: {device}, Batch size: {batch_size}")
    print(f"Feature dim: {FEATURE_DIM}, Patches per image: {NUM_PATCHES}")
    print(f"Expected total patches: {total_images * NUM_PATCHES}")

    start_time = time.time()
    with torch.no_grad():
        for batch, names in tqdm(loader, desc="Extracting features"):
            batch = batch.to(device)
            # features: (B, 256, 448)
            features = model(batch)
            # Reshape: (B * 256, 448)
            features = features.reshape(-1, FEATURE_DIM)
            all_features.append(features.cpu().numpy())

    elapsed = time.time() - start_time
    features = np.concatenate(all_features, axis=0).astype(np.float32)

    print(f"\nDone in {elapsed:.1f}s ({total_images / elapsed:.1f} images/s)")
    print(f"Feature matrix: {features.shape}, "
          f"memory={features.nbytes / 1024 / 1024:.1f} MB")

    return features


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore memory bank")
    parser.add_argument("--good-images", default=GOOD_WELDS_DIR,
                        help="Directory of defect-free images")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for feature extraction")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--coreset-size", type=int, default=CORESET_SIZE,
                        help="Memory bank size after subsampling")
    parser.add_argument("--coreset-method", default="kmeans",
                        choices=["kmeans", "greedy"],
                        help="Coreset method: kmeans (fast) or greedy (exact)")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="PCA target dimension (0 = no PCA, keep original 448)")
    parser.add_argument("--output-dir", default=PATCHCORE_MODEL_DIR,
                        help="Output directory for model artifacts")
    args = parser.parse_args()

    # Verify good images exist
    if not os.path.isdir(args.good_images):
        print(f"ERROR: Good images directory not found: {args.good_images}")
        print("Run 'python scripts/filter_good_samples.py --copy' first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Extract features
    features = extract_features(
        image_dir=args.good_images,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Step 2: Optional PCA dimension reduction
    pca_matrix = None
    pca_mean = None
    active_dim = FEATURE_DIM

    if args.pca_dim > 0 and args.pca_dim < FEATURE_DIM:
        features, pca_matrix, pca_mean = reduce_dimension_pca(
            features, target_dim=args.pca_dim)
        active_dim = args.pca_dim

    # Step 3: Build memory bank
    memory_bank = build_memory_bank(
        features=features,
        coreset_size=args.coreset_size,
        method=args.coreset_method,
    )

    # Step 4: Save
    bank_path = os.path.join(args.output_dir, "memory_bank.npy")
    np.save(bank_path, memory_bank)
    print(f"\nMemory bank saved to: {bank_path}")

    if pca_matrix is not None:
        pca_path = os.path.join(args.output_dir, "pca_matrix.npy")
        pca_mean_path = os.path.join(args.output_dir, "pca_mean.npy")
        np.save(pca_path, pca_matrix)
        np.save(pca_mean_path, pca_mean)
        print(f"PCA matrix saved to: {pca_path} ({pca_matrix.shape})")
        print(f"PCA mean saved to:   {pca_mean_path}")

    # Save metadata
    metadata = {
        "num_patches": int(memory_bank.shape[0]),
        "feature_dim": int(memory_bank.shape[1]),
        "original_dim": FEATURE_DIM,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "backbone": "resnet18",
        "layers": LAYERS,
        "pool_size": POOL_SIZE,
        "normalization": "l2",
        "top_k": 5,
        "num_good_images": len(os.listdir(args.good_images)),
        "coreset_method": args.coreset_method,
        "pca_dim": args.pca_dim if pca_matrix is not None else 0,
    }
    meta_path = os.path.join(args.output_dir, "bank_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    print("\n=== PatchCore training complete ===")
    print(f"  Memory bank: {memory_bank.shape}")
    print(f"  Next: run 'python scripts/export_patchcore_onnx.py' to export ONNX backbone")
    print(f"  Then: run 'python scripts/validate_patchcore.py' to evaluate")


if __name__ == "__main__":
    main()
