"""
Filter defect-free images from the welding dataset for PatchCore training.
A "good" image has NO annotations of class 0 (Crack), 1 (Porosity), or 2 (Spatters).
Class 3 (Welding line) alone is considered normal/acceptable.
"""
import os
import shutil
import argparse
import yaml
from pathlib import Path


def is_defect_free(label_path: str) -> bool:
    """
    Check if a label file contains NO defect annotations.
    Defect classes: 0=Crack, 1=Porosity, 2=Spatters
    Class 3 (Welding line) is NOT a defect.
    """
    with open(label_path, "r") as f:
        content = f.read().strip()
    if not content:
        return True  # empty = no annotations = good
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        cls_id = int(line.split()[0])
        if cls_id in {0, 1, 2}:
            return False
    return True


def filter_good_samples(data_yaml_path: str, output_dir: str, copy_files: bool = False):
    """
    Walk the training set, find defect-free images, and optionally copy them.
    """
    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    train_images_dir = config["train"]
    # data.yaml uses train: path/to/images, derive labels dir
    labels_dir = train_images_dir.replace("images", "labels")

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    os.makedirs(output_dir, exist_ok=True)

    all_images = sorted(os.listdir(train_images_dir))
    all_labels = sorted(os.listdir(labels_dir))
    good_samples = []
    defect_samples = 0

    # Match labels to images by stem
    label_stems = {os.path.splitext(f)[0]: f for f in all_labels}
    image_stems = {os.path.splitext(f)[0]: f for f in all_images}

    matched = 0
    for stem, img_file in image_stems.items():
        if stem not in label_stems:
            continue  # image without label (unlikely but skip)
        matched += 1
        label_file = os.path.join(labels_dir, label_stems[stem])
        if is_defect_free(label_file):
            good_samples.append((os.path.join(train_images_dir, img_file), stem))
        else:
            defect_samples += 1

    print(f"Total train images with labels: {matched}")
    print(f"Defect-free (good):  {len(good_samples)}")
    print(f"Has defects:         {defect_samples}")

    # Also check for images with empty labels
    empty_count = 0
    for _, stem in good_samples:
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.getsize(label_path) == 0:
            empty_count += 1
    print(f"  (of which empty annotations: {empty_count})")
    print(f"  (class-3-only annotations:   {len(good_samples) - empty_count})")

    if copy_files:
        print(f"\nCopying {len(good_samples)} images to {output_dir} ...")
        for src_path, stem in good_samples:
            dst = os.path.join(output_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst)
        print("Done.")

    # Write manifest
    manifest_path = os.path.join(output_dir, "good_welds_manifest.txt")
    with open(manifest_path, "w") as f:
        for _, stem in good_samples:
            f.write(f"{stem}\n")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter defect-free weld images")
    parser.add_argument("--data-yaml", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--output", default="data/good_welds", help="Output directory")
    parser.add_argument("--copy", action="store_true", help="Copy images to output dir")
    args = parser.parse_args()

    filter_good_samples(args.data_yaml, args.output, args.copy)
