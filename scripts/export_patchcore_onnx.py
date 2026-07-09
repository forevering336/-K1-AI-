"""
Export PatchCore ResNet-18 backbone to ONNX format.

The exported model includes:
- ResNet-18 forward pass (conv1 through layer3)
- AdaptiveAvgPool2d to fixed spatial grid
- Channel concatenation
- Output reshape to (1, 256, 448)

Input:  (1, 3, 256, 256) float32, ImageNet normalized
Output: (1, 256, 448) float32
"""
import os
import sys
import argparse
import numpy as np
import torch
import onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchcore.config import (
    INPUT_SIZE, ONNX_OPSET, ONNX_INPUT_NAME, ONNX_OUTPUT_NAME,
    PATCHCORE_MODEL_DIR, IMAGENET_MEAN, IMAGENET_STD,
)
from patchcore.backbone import create_feature_extractor


def export_onnx(output_path: str, verify: bool = True):
    """
    Export the PatchCore feature extractor to ONNX.

    Args:
        output_path: where to save the .onnx file
        verify: check ONNX model validity and compare outputs
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Creating feature extractor...")
    model = create_feature_extractor(pretrained=True)
    model.eval()

    # Create dummy input (ImageNet normalized)
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Verify PyTorch output shape
    with torch.no_grad():
        torch_output = model(dummy_input)
    print(f"PyTorch output shape: {torch_output.shape}")

    # Export to ONNX
    # Use dynamo=False for compatibility with opset 12 on older ONNX runtimes
    print(f"Exporting to ONNX (opset={ONNX_OPSET})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        dynamo=False,  # use JIT-based exporter for opset 12 compat
    )
    print(f"ONNX model saved to: {output_path}")

    # Simplify if onnx-simplifier is available
    try:
        import onnxsim
        print("Simplifying ONNX model...")
        model_onnx = onnx.load(output_path)
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output_path)
            print("ONNX model simplified successfully.")
        else:
            print("Warning: ONNX simplification check failed, using original model.")
    except ImportError:
        print("Note: onnx-simplifier not installed, skipping simplification.")
        print("  Install with: pip install onnx-simplifier")

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check: PASSED")

    # Compare ONNX vs PyTorch outputs
    if verify:
        print("\nComparing ONNX vs PyTorch outputs...")
        import onnxruntime as ort

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        onnx_output = session.run(
            [ONNX_OUTPUT_NAME],
            {ONNX_INPUT_NAME: dummy_input.numpy()},
        )[0]

        max_diff = np.max(np.abs(torch_output.numpy() - onnx_output))
        print(f"Max absolute difference: {max_diff:.6e}")
        if max_diff > 1e-3:
            print("WARNING: Large difference between PyTorch and ONNX outputs!")
            print("This may indicate an unsupported op in ONNX runtime.")
        else:
            print("Output comparison: PASSED")

    # Print model info
    model_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nModel size: {model_size_mb:.1f} MB")
    print(f"Input:  {ONNX_INPUT_NAME} (1, 3, {INPUT_SIZE}, {INPUT_SIZE}) float32")
    print(f"Output: {ONNX_OUTPUT_NAME} (1, 256, 448) float32")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PatchCore backbone to ONNX")
    parser.add_argument("--output", default=os.path.join(PATCHCORE_MODEL_DIR, "backbone.onnx"),
                        help="Output ONNX path")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip PyTorch vs ONNX comparison")
    args = parser.parse_args()

    export_onnx(args.output, verify=not args.no_verify)
