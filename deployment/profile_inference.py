"""
K1 推理性能诊断脚本。
在 K1 上运行，打印每个环节的精确耗时，定位瓶颈。
"""
import time
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchcore.config import NUM_PATCHES, FEATURE_DIM, TOP_K, IMAGENET_MEAN, IMAGENET_STD
from patchcore.anomaly_scorer import compute_anomaly_scores
from deployment.preprocess import preprocess_yolo, preprocess_patchcore
from deployment.postprocess import postprocess_yolo


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.time() - self.start) * 1000

    def ms(self):
        return (time.time() - self.start) * 1000


def profile(yolo_onnx, pc_onnx, bank_path, image_path, warmup=3, runs=10):
    """精确测量每个阶段的耗时"""
    import onnxruntime as ort

    print("=" * 65)
    print("K1 推理性能诊断")
    print("=" * 65)

    # --- 检查可用的 ONNX Runtime providers ---
    print(f"\n[1] ONNX Runtime 可用 providers: {ort.get_available_providers()}")
    print(f"    当前使用: CPUExecutionProvider")

    # --- 加载模型 ---
    print("\n[2] 加载模型...")
    t = time.time()
    sess_yolo = ort.InferenceSession(yolo_onnx, providers=["CPUExecutionProvider"])
    print(f"    YOLO 加载: {(time.time()-t)*1000:.0f}ms")
    print(f"    输入: {sess_yolo.get_inputs()[0].name} {sess_yolo.get_inputs()[0].shape}")

    t = time.time()
    sess_pc = ort.InferenceSession(pc_onnx, providers=["CPUExecutionProvider"])
    print(f"    PatchCore 加载: {(time.time()-t)*1000:.0f}ms")
    print(f"    输入: {sess_pc.get_inputs()[0].name} {sess_pc.get_inputs()[0].shape}")

    t = time.time()
    memory_bank = np.load(bank_path).astype(np.float32)
    memory_bank = memory_bank / (np.linalg.norm(memory_bank, axis=1, keepdims=True) + 1e-8)
    print(f"    记忆库加载: {(time.time()-t)*1000:.0f}ms, shape={memory_bank.shape}")

    yolo_in = sess_yolo.get_inputs()[0].name
    yolo_out = sess_yolo.get_outputs()[0].name
    pc_in = sess_pc.get_inputs()[0].name
    pc_out = sess_pc.get_outputs()[0].name

    # --- 读取图片 ---
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: 无法读取 {image_path}")
        return
    h, w = frame.shape[:2]
    print(f"\n[3] 测试图片: {w}×{h}")

    # --- Warmup ---
    print(f"\n[4] 预热 ({warmup} 次)...")
    for i in range(warmup):
        yi = preprocess_yolo(frame)
        _ = sess_yolo.run([yolo_out], {yolo_in: yi})
        pi = preprocess_patchcore(frame)
        feats = sess_pc.run([pc_out], {pc_in: pi})[0]
        feats = feats.reshape(NUM_PATCHES, FEATURE_DIM)
        _, _ = compute_anomaly_scores(feats, memory_bank, top_k=TOP_K)
        print(f"    预热 {i+1}/{warmup} 完成")

    # --- 精确计时 ---
    print(f"\n[5] 精确计时 ({runs} 次，取平均)\n")
    print(f"{'阶段':<35} {'平均耗时':>10} {'占比':>8}")
    print("-" * 55)

    timings = {
        "YOLO 预处理": [],
        "YOLO ONNX 推理": [],
        "YOLO 后处理 (NMS)": [],
        "PatchCore 预处理": [],
        "PatchCore ONNX 推理": [],
        "PatchCore kNN 搜索": [],
        "PatchCore 热力图": [],
        "融合 + 绘制": [],
        "帧总计": [],
    }

    def once(img):
        stages = {}
        t0 = time.time()

        # YOLO pipeline
        t = time.time()
        yi = preprocess_yolo(img)
        stages["YOLO 预处理"] = (time.time() - t) * 1000

        t = time.time()
        yo = sess_yolo.run([yolo_out], {yolo_in: yi})
        stages["YOLO ONNX 推理"] = (time.time() - t) * 1000

        t = time.time()
        dets = postprocess_yolo(yo[0], (h, w), conf_threshold=0.5, iou_threshold=0.45)
        stages["YOLO 后处理 (NMS)"] = (time.time() - t) * 1000

        # PatchCore pipeline
        t = time.time()
        pi = preprocess_patchcore(img)
        stages["PatchCore 预处理"] = (time.time() - t) * 1000

        t = time.time()
        feats = sess_pc.run([pc_out], {pc_in: pi})[0]
        stages["PatchCore ONNX 推理"] = (time.time() - t) * 1000

        t = time.time()
        feats = feats.reshape(NUM_PATCHES, FEATURE_DIM)
        patch_dists, score = compute_anomaly_scores(feats, memory_bank, top_k=TOP_K)
        stages["PatchCore kNN 搜索"] = (time.time() - t) * 1000

        t = time.time()
        from patchcore.heatmap import generate_heatmap
        hm = generate_heatmap(patch_dists, (h, w), pool_size=16, gaussian_sigma=4.0)
        stages["PatchCore 热力图"] = (time.time() - t) * 1000

        t = time.time()
        from deployment.fusion import fusion
        _, _, _ = fusion(dets, score, 0.471, True)
        stages["融合 + 绘制"] = (time.time() - t) * 1000

        stages["帧总计"] = (time.time() - t0) * 1000
        return stages

    for r in range(runs):
        s = once(frame)
        for k in timings:
            timings[k].append(s[k])

    total = 0
    for name in timings:
        avg = np.mean(timings[name])
        total += avg if name != "帧总计" else 0
        print(f"{name:<35} {avg:>8.1f} ms")

    avg_total = np.mean(timings["帧总计"])
    # Recalculate pct based on individual stages
    stage_total = sum(np.mean(timings[k]) for k in timings if k != "帧总计")

    print("-" * 55)
    print(f"{'(各阶段之和)':<35} {stage_total:>8.1f} ms")
    print(f"{'帧总计':<35} {avg_total:>8.1f} ms")
    print(f"{'≈ FPS':<35} {1000/avg_total:>8.1f}")

    # 占比分析
    print(f"\n[6] 各阶段占比:")
    yolo_total = np.mean(timings["YOLO 预处理"]) + np.mean(timings["YOLO ONNX 推理"]) + np.mean(timings["YOLO 后处理 (NMS)"])
    pc_total = np.mean(timings["PatchCore 预处理"]) + np.mean(timings["PatchCore ONNX 推理"]) + np.mean(timings["PatchCore kNN 搜索"]) + np.mean(timings["PatchCore 热力图"])
    other = np.mean(timings["融合 + 绘制"])

    print(f"  YOLO 分支合计:       {yolo_total:.0f} ms ({yolo_total/avg_total*100:.0f}%)")
    for k in ["YOLO 预处理", "YOLO ONNX 推理", "YOLO 后处理 (NMS)"]:
        v = np.mean(timings[k])
        print(f"    └─ {k}: {v:.0f} ms ({v/avg_total*100:.0f}%)")

    print(f"  PatchCore 分支合计:  {pc_total:.0f} ms ({pc_total/avg_total*100:.0f}%)")
    for k in ["PatchCore 预处理", "PatchCore ONNX 推理", "PatchCore kNN 搜索", "PatchCore 热力图"]:
        v = np.mean(timings[k])
        print(f"    └─ {k}: {v:.0f} ms ({v/avg_total*100:.0f}%)")

    print(f"  其他:                {other:.0f} ms ({other/avg_total*100:.0f}%)")

    # 跳帧模式下等效 FPS
    print(f"\n[7] 跳帧预估（PatchCore 每 N 帧跑一次）:")
    yolo_per_frame = yolo_total + other
    pc_per_run = pc_total
    for skip in [1, 4, 8, 16, 30]:
        avg_frame_time = yolo_per_frame + pc_per_run / skip
        fps = 1000 / avg_frame_time
        print(f"  N={skip:<3}: {avg_frame_time:.0f}ms/frame → {fps:.1f} FPS")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-onnx", default="models/best_award_int8.onnx")
    parser.add_argument("--pc-onnx", default="models/backbone3.onnx")
    parser.add_argument("--bank", default="models/memory_bank2.npy")
    parser.add_argument("--image", default="test/images/bad_weld_vid177_jpeg_jpg.rf.13f2c95545fae59d1aa0e91f373514af.jpg")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    profile(args.yolo_onnx, args.pc_onnx, args.bank, args.image, args.warmup, args.runs)
