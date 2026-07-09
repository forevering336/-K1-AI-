"""
PatchCore configuration constants.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_YAML = os.path.join(PROJECT_ROOT, "data.yaml")
GOOD_WELDS_DIR = os.path.join(PROJECT_ROOT, "archive", "training_cache", "good_welds")

# Model subdirectories
PATCHCORE_MODEL_DIR = MODEL_DIR
YOLO_MODEL_DIR = MODEL_DIR
PRETRAINED_DIR = MODEL_DIR

# Key model files
YOLO_ONNX_PATH = os.path.join(YOLO_MODEL_DIR, "best_award_int8.onnx")
PATCHCORE_ONNX_PATH = os.path.join(PATCHCORE_MODEL_DIR, "backbone3.onnx")
MEMORY_BANK_PATH = os.path.join(PATCHCORE_MODEL_DIR, "memory_bank2.npy")

# --- Model ---
BACKBONE_NAME = "resnet18"              # torchvision model name
INPUT_SIZE = 256                         # PatchCore input resolution
POOL_SIZE = 16                           # spatial grid for AdaptiveAvgPool2d (16x16)
LAYERS = ["layer1", "layer2", "layer3"]  # ResNet-18 intermediate layers to extract

# --- Feature dimensions (ResNet-18) ---
# layer1: 64 channels, layer2: 128 channels, layer3: 256 channels
LAYER_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256}
FEATURE_DIM = sum(LAYER_CHANNELS[l] for l in LAYERS)  # 448
NUM_PATCHES = POOL_SIZE * POOL_SIZE                     # 256

# --- Memory bank ---
CORESET_SIZE = 10000   # number of feature vectors in final memory bank
TOP_K = 5              # top-k max distances for anomaly score (robustness)

# --- ImageNet normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- ONNX ---
ONNX_OPSET = 12
ONNX_INPUT_NAME = "image"
ONNX_OUTPUT_NAME = "features"
