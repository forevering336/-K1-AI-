#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无监督异常检测分支（PatchCore 推理）—— 兜住 YOLO 没见过的「未知缺陷」

输入产物（队友导出）:
    backbone.onnx   特征提取器, 输入 [1,3,256,256], 输出 [1,256,448]
                    (256 = 16x16 个 patch, 448 = 每 patch 特征维度)
    memory_bank.npy 正常焊件特征库 [N,448], 已 L2 归一化

原理:
    新图过 backbone → 256 个 patch 特征 → 各自 L2 归一化 →
    在 memory_bank 里找最近邻(用余弦, 因两边都已归一化) →
    离正常特征越远 = 越异常 → 拼成 16x16 异常热力图 →
    图像级异常分 = 所有 patch 的最大异常距离。

⚠️ 预处理假设(队友没给建库代码, 按 PatchCore/ImageNet 标准默认):
    resize 到 256x256 → BGR转RGB → /255 → ImageNet 均值方差归一化。
    若板上异常分明显偏高/偏低, 先怀疑这里的均值方差或 resize 方式和建库时不一致。
"""
import numpy as np


# ImageNet 归一化参数（ResNet 系 backbone 的标准；如与建库不一致在此改）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class AnomalyScorer:
    def __init__(self, backbone_path, memory_bank_path,
                 input_size=256, grid=16, threshold=0.5):
        import onnxruntime as ort
        try:
            import spacemit_ort  # noqa: 板端 NPU 加速(有就用)
        except Exception:
            pass
        avail = ort.get_available_providers()
        providers = (['SpaceMITExecutionProvider', 'CPUExecutionProvider']
                     if 'SpaceMITExecutionProvider' in avail else ['CPUExecutionProvider'])
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(backbone_path, sess_options=so, providers=providers)
        self.iname = self.sess.get_inputs()[0].name

        mb = np.load(memory_bank_path).astype(np.float32)
        # 保险：确保记忆库是 L2 归一化的
        n = np.linalg.norm(mb, axis=1, keepdims=True)
        n[n == 0] = 1.0
        self.bank = (mb / n)                 # [N,448]
        self.bank_T = np.ascontiguousarray(self.bank.T)  # [448,N] 加速矩阵乘

        self.input_size = input_size
        self.grid = grid                     # 16 → 16x16=256 patch
        self.threshold = float(threshold)
        print(f"[ANOMALY] backbone={backbone_path} 记忆库={mb.shape} 阈值={threshold}")

    # ---- 预处理 ----
    def _preprocess(self, frame_bgr):
        import cv2
        img = cv2.resize(frame_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))    # CHW
        img = (img - _MEAN) / _STD
        return img[None].astype(np.float32)   # [1,3,H,W]

    # ---- 打分 ----
    def score(self, frame_bgr):
        """返回 (image_score, heatmap[HxW float], patch_scores[grid,grid])。"""
        feat = self.sess.run(None, {self.iname: self._preprocess(frame_bgr)})[0]  # [1,256,448]
        f = feat[0].astype(np.float32)                                            # [256,448]
        n = np.linalg.norm(f, axis=1, keepdims=True); n[n == 0] = 1.0
        f = f / n                                                                 # L2 归一化(必须)
        cos = f @ self.bank_T                       # [256,N] 与所有正常特征的余弦相似度
        best = cos.max(axis=1)                       # 每个 patch 的最近邻相似度
        patch_dist = 1.0 - best                      # 余弦距离, 越大越异常, ∈[0,2]
        g = self.grid
        patch_scores = patch_dist.reshape(g, g)
        image_score = float(patch_dist.max())
        H, W = frame_bgr.shape[:2]
        import cv2
        heatmap = cv2.resize(patch_scores.astype(np.float32), (W, H))
        return image_score, heatmap, patch_scores

    def is_anomaly(self, image_score):
        return image_score > self.threshold

    # ---- 把异常区域画到画面上(未知缺陷用洋红框区分已知缺陷) ----
    def overlay(self, frame_bgr, heatmap, thresh=None):
        import cv2
        thresh = self.threshold if thresh is None else thresh
        out = frame_bgr.copy()
        mask = (heatmap > thresh).astype(np.uint8) * 255
        if mask.any():
            # 半透明热区
            color = np.zeros_like(out); color[..., 2] = 255; color[..., 0] = 255  # 洋红(BGR)
            m3 = mask.astype(bool)
            out[m3] = (0.6 * out[m3] + 0.4 * color[m3]).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 30:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(out, "Unknown", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        return out

    # ---- 阈值标定: 拿一批正常图, 给出建议阈值 ----
    def calibrate(self, frames):
        scores = [self.score(f)[0] for f in frames]
        scores = np.array(scores, dtype=np.float32)
        suggest = float(scores.max() * 1.1)          # 正常图最高分 +10% 余量
        print(f"[CALIB] 正常图 {len(scores)} 张  分数 min/mean/max = "
              f"{scores.min():.4f}/{scores.mean():.4f}/{scores.max():.4f}")
        print(f"[CALIB] 建议阈值 ≈ {suggest:.4f}  (可用 --anomaly-thresh 设置)")
        return suggest


# ---- 命令行: 单图打分 / 文件夹标定 ----
if __name__ == "__main__":
    import argparse, os, cv2
    ap = argparse.ArgumentParser(description="PatchCore 异常检测 —— 单图打分 / 标定阈值")
    ap.add_argument("--backbone", default="backbone.onnx")
    ap.add_argument("--bank", default="memory_bank.npy")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--image", help="单图打分并输出叠加图")
    ap.add_argument("--calib", help="正常图文件夹, 计算建议阈值")
    args = ap.parse_args()

    scorer = AnomalyScorer(args.backbone, args.bank, threshold=args.thresh)

    if args.calib:
        frames = []
        for fn in sorted(os.listdir(args.calib)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                im = cv2.imread(os.path.join(args.calib, fn))
                if im is not None:
                    frames.append(im)
        if frames:
            scorer.calibrate(frames)
        else:
            print("[ERROR] 文件夹里没有可读图片")
    elif args.image:
        im = cv2.imread(args.image)
        if im is None:
            print("[ERROR] 读不了图片"); raise SystemExit(1)
        s, hm, _ = scorer.score(im)
        print(f"异常分 = {s:.4f}  ->  {'⚠️ 异常(NG)' if s > args.thresh else 'OK'}")
        out = scorer.overlay(im, hm)
        cv2.imwrite("anomaly_result.jpg", out)
        print("已存叠加图: anomaly_result.jpg")
    else:
        ap.print_help()
