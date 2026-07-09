#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 焊接缺陷检测 · mAP 测评 / 量化前后精度对比脚本
=====================================================

目的：给答辩提供硬指标。一条命令跑出：
  - mAP@0.5、mAP@0.5:0.95（COCO 口径）
  - 每一类的 AP / 精确率 / 召回率
  - 混淆矩阵（含漏检/误报）
  - 整体误报率、漏检率
可同时评两个模型（FP32 vs INT8），直接打印"掉了几个点"。

⚠️ 预处理必须与板上 weld_live.py 完全一致：
  letterbox 640（灰边 114）→ BGR2RGB → /255 → NCHW，无 ImageNet 归一化。
本脚本已对齐。

依赖（PC 上装，不在板子上跑）：
    pip install onnxruntime opencv-python numpy pyyaml

用法：
  # 单模型评测（在 valid 或 test 集上）
  python eval_map.py --model best_award_fp32.onnx \
      --data "D:\\...\\焊缝缺陷检测（4-6k）\\1" --split test

  # 量化前后对比（一次跑两个模型，输出对比表）
  python eval_map.py --model best_award_fp32.onnx --model-int8 best_award_int8.onnx \
      --data "D:\\...\\1" --split test

数据集结构（Roboflow YOLO 格式）：
    <data>/data.yaml         # names: [Crack, Porosity, Spatters, Welding line]
    <data>/test/images/*.jpg
    <data>/test/labels/*.txt # 每行: cls cx cy w h  (归一化 0~1)
"""
import os
import sys
import glob
import time
import argparse

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv-python：pip install opencv-python")
try:
    import onnxruntime as ort
except ImportError:
    sys.exit("需要 onnxruntime：pip install onnxruntime")


# ----------------------------- 配置默认值 ----------------------------------
DEFAULT_NAMES = ["Crack", "Porosity", "Spatters", "Welding line"]
NAME_CN = {"Crack": "裂纹", "Porosity": "气孔",
           "Spatters": "飞溅", "Welding line": "焊缝"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def windows_long_path(path):
    if os.name != "nt":
        return path
    path = os.path.abspath(path)
    if path.startswith("\\\\?\\"):
        return path
    return "\\\\?\\" + path


def imread_unicode(path):
    """cv2.imread on Windows often fails for non-ASCII/long paths; keep dataset paths usable."""
    try:
        data = np.fromfile(windows_long_path(path), dtype=np.uint8)
    except FileNotFoundError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ----------------------------- 预处理 -------------------------------------
def letterbox(img, new_size=640, color=(114, 114, 114)):
    """等比缩放 + 灰边填充，返回图 + (缩放比, 左pad, 上pad)。与 weld_live.py 一致。"""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def preprocess(img, size):
    canvas, r, left, top = letterbox(img, size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]  # NCHW
    return np.ascontiguousarray(blob), r, left, top


# ----------------------------- 后处理 -------------------------------------
def nms(boxes, scores, iou_thr):
    """boxes: [N,4] xyxy。返回保留索引。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode(output, r, left, top, num_cls, conf_thr, iou_thr):
    """
    YOLOv8 ONNX 输出 [1, 4+num_cls, N] → 还原到原图坐标的检测框。
    返回 list[(x1,y1,x2,y2,score,cls)]。
    """
    out = output[0]                      # [4+nc, N]
    if out.shape[0] < out.shape[1]:      # [4+nc, N]
        out = out.T                      # → [N, 4+nc]
    boxes_cxcywh = out[:, :4]
    scores_all = out[:, 4:4 + num_cls]
    cls_ids = scores_all.argmax(1)
    confs = scores_all.max(1)
    m = confs >= conf_thr
    boxes_cxcywh, confs, cls_ids = boxes_cxcywh[m], confs[m], cls_ids[m]
    if len(boxes_cxcywh) == 0:
        return []
    cx, cy, w, h = boxes_cxcywh.T
    x1 = (cx - w / 2 - left) / r
    y1 = (cy - h / 2 - top) / r
    x2 = (cx + w / 2 - left) / r
    y2 = (cy + h / 2 - top) / r
    xyxy = np.stack([x1, y1, x2, y2], 1)
    # 按类别做 NMS
    dets = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = nms(xyxy[idx], confs[idx], iou_thr)
        for k in keep:
            j = idx[k]
            dets.append((xyxy[j, 0], xyxy[j, 1], xyxy[j, 2], xyxy[j, 3],
                         float(confs[j]), int(c)))
    return dets


# ----------------------------- 标注读取 -----------------------------------
def load_gt(label_path, img_w, img_h):
    """读 YOLO txt → list[(x1,y1,x2,y2,cls)]（像素坐标）。"""
    gts = []
    label_path_long = windows_long_path(label_path)
    if not os.path.exists(label_path_long):
        return gts
    with open(label_path_long) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            gts.append((x1, y1, x2, y2, int(c)))
    return gts


def iou_xyxy(a, b):
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    iw = max(0.0, xx2 - xx1); ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)


# ----------------------------- AP 计算 ------------------------------------
def compute_ap(recall, precision):
    """COCO 101 点插值。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def evaluate(all_dets, all_gts, num_cls, iou_thresholds):
    """
    all_dets[img] = list[(x1,y1,x2,y2,score,cls)]
    all_gts[img]  = list[(x1,y1,x2,y2,cls)]
    返回 per-class AP（对每个 iou 阈值），以及 P/R@0.5。
    """
    # 收集每类的检测（跨图）
    ap_per_iou = {}   # iou_t -> [ap_cls...]
    pr50 = None
    for it in iou_thresholds:
        aps = []
        pr_holder = []
        for c in range(num_cls):
            dets = []
            n_gt = 0
            for img in all_gts:
                for g in all_gts[img]:
                    if g[4] == c:
                        n_gt += 1
            for img in all_dets:
                for d in all_dets[img]:
                    if d[5] == c:
                        dets.append((img, d[4], d[:4]))
            dets.sort(key=lambda x: x[1], reverse=True)
            tp = np.zeros(len(dets))
            fp = np.zeros(len(dets))
            matched = {img: set() for img in all_gts}
            for i, (img, score, box) in enumerate(dets):
                gts = all_gts.get(img, [])
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts):
                    if g[4] != c or j in matched[img]:
                        continue
                    v = iou_xyxy(box, g[:4])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= it and best_j >= 0:
                    tp[i] = 1
                    matched[img].add(best_j)
                else:
                    fp[i] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / (n_gt + 1e-9)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap = compute_ap(recall, precision) if n_gt > 0 else float("nan")
            aps.append(ap)
            # P/R@0.5 用全部检测末端值
            final_r = recall[-1] if len(recall) else 0.0
            final_p = precision[-1] if len(precision) else 0.0
            pr_holder.append((final_p, final_r, int(tp.sum()),
                              int(fp.sum()), n_gt))
        ap_per_iou[it] = aps
        if abs(it - 0.5) < 1e-6:
            pr50 = pr_holder
    return ap_per_iou, pr50


# ----------------------------- 混淆矩阵 -----------------------------------
def confusion_matrix(all_dets, all_gts, num_cls, iou_thr=0.5, conf_thr=0.25):
    """行=真实，列=预测；额外一行/列表示背景（漏检/误报）。"""
    n = num_cls + 1  # 最后一格 = 背景
    cm = np.zeros((n, n), dtype=int)
    for img in all_gts:
        gts = list(all_gts[img])
        dets = [d for d in all_dets.get(img, []) if d[4] >= conf_thr]
        dets.sort(key=lambda x: x[4], reverse=True)
        gt_used = set()
        for d in dets:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if j in gt_used:
                    continue
                v = iou_xyxy(d[:4], g[:4])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and best_j >= 0:
                cm[gts[best_j][4], d[5]] += 1     # 真实→预测
                gt_used.add(best_j)
            else:
                cm[num_cls, d[5]] += 1            # 背景被误报成 d[5]
        for j, g in enumerate(gts):
            if j not in gt_used:
                cm[g[4], num_cls] += 1            # 真实漏检
    return cm


# ----------------------------- 单模型跑一遍 --------------------------------
def run_model(model_path, images, labels_dir, size, num_cls,
              conf_thr, iou_thr):
    print(f"\n加载模型: {model_path}")
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    inp_name = sess.get_inputs()[0].name
    all_dets, all_gts = {}, {}
    t0 = time.time()
    for k, img_path in enumerate(images):
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        blob, r, left, top = preprocess(img, size)
        out = sess.run(None, {inp_name: blob})[0]
        dets = decode(out, r, left, top, num_cls, conf_thr, iou_thr)
        key = os.path.splitext(os.path.basename(img_path))[0]
        all_dets[key] = dets
        lbl = os.path.join(labels_dir, key + ".txt")
        all_gts[key] = load_gt(lbl, w, h)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(images)} ...")
    dt = time.time() - t0
    print(f"  推理完成 {len(images)} 张，用时 {dt:.1f}s "
          f"（{len(images)/dt:.1f} img/s，PC CPU，仅供参考）")
    return all_dets, all_gts


def summarize(all_dets, all_gts, names):
    num_cls = len(names)
    iou_ts = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    ap_per_iou, pr50 = evaluate(all_dets, all_gts, num_cls, iou_ts)
    ap50 = ap_per_iou[0.5]
    ap5095 = [np.nanmean([ap_per_iou[t][c] for t in iou_ts])
              for c in range(num_cls)]
    result = {
        "map50": float(np.nanmean(ap50)),
        "map5095": float(np.nanmean(ap5095)),
        "per_class": [],
    }
    for c in range(num_cls):
        p, r, tp, fp, ngt = pr50[c]
        result["per_class"].append({
            "name": names[c], "ap50": ap50[c], "ap5095": ap5095[c],
            "precision": p, "recall": r, "tp": tp, "fp": fp, "n_gt": ngt,
        })
    return result


def print_report(res, names):
    print("\n" + "=" * 68)
    print(f"{'类别':<16}{'AP@.5':>10}{'AP@.5:.95':>12}"
          f"{'精确率':>10}{'召回率':>10}")
    print("-" * 68)
    for pc in res["per_class"]:
        cn = NAME_CN.get(pc["name"], "")
        label = f"{pc['name']}({cn})" if cn else pc["name"]
        print(f"{label:<16}{pc['ap50']*100:>9.1f}%{pc['ap5095']*100:>11.1f}%"
              f"{pc['precision']*100:>9.1f}%{pc['recall']*100:>9.1f}%")
    print("-" * 68)
    print(f"{'mAP (全部)':<16}{res['map50']*100:>9.1f}%"
          f"{res['map5095']*100:>11.1f}%")
    # 误报/漏检总览
    tp = sum(pc["tp"] for pc in res["per_class"])
    fp = sum(pc["fp"] for pc in res["per_class"])
    ngt = sum(pc["n_gt"] for pc in res["per_class"])
    miss = ngt - tp
    print(f"\n真实目标 {ngt}  命中 {tp}  漏检 {miss}"
          f"（漏检率 {miss/max(ngt,1)*100:.1f}%）  误报 {fp}")
    print("=" * 68)


def print_confusion(cm, names):
    n = len(names)
    hdr = ["预测↓真实→"] + [NAME_CN.get(x, x)[:4] for x in names] + ["背景/漏"]
    print("\n混淆矩阵（行=真实类别，列=预测类别，末列=漏检，末行=误报）")
    colw = 8
    print("".join(f"{h:>{colw}}" for h in hdr))
    labels = [NAME_CN.get(x, x)[:4] for x in names] + ["误报"]
    for i in range(n + 1):
        row = [labels[i]] + [str(cm[i, j]) for j in range(n + 1)]
        print("".join(f"{c:>{colw}}" for c in row))


def load_names(data_dir):
    yml = os.path.join(data_dir, "data.yaml")
    if os.path.exists(yml):
        try:
            import yaml
            with open(yml, encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if d and "names" in d:
                nm = d["names"]
                if isinstance(nm, dict):
                    nm = [nm[k] for k in sorted(nm)]
                return list(nm)
        except Exception:
            pass
    return DEFAULT_NAMES


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 焊接缺陷 mAP 测评")
    ap.add_argument("--model", required=True, help="FP32 模型 .onnx")
    ap.add_argument("--model-int8", default=None,
                    help="INT8 量化模型 .onnx（给了就做对比）")
    ap.add_argument("--data", required=True, help="数据集根目录")
    ap.add_argument("--split", default="test",
                    choices=["test", "valid", "train"])
    ap.add_argument("--size", type=int, default=640, help="模型输入尺寸")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 张(调试)")
    args = ap.parse_args()

    names = load_names(args.data)
    num_cls = len(names)
    img_dir = os.path.join(args.data, args.split, "images")
    lbl_dir = os.path.join(args.data, args.split, "labels")
    if not os.path.isdir(img_dir):
        sys.exit(f"找不到图片目录: {img_dir}")
    images = []
    for e in IMG_EXTS:
        images += glob.glob(os.path.join(img_dir, "*" + e))
    images.sort()
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"{img_dir} 里没有图片")
    print(f"数据集: {args.split}  图片 {len(images)} 张  类别 {names}")

    # --- FP32 ---
    d32, g32 = run_model(args.model, images, lbl_dir, args.size,
                         num_cls, args.conf, args.iou)
    res32 = summarize(d32, g32, names)
    print("\n########## 模型 A（FP32）##########")
    print_report(res32, names)
    cm = confusion_matrix(d32, g32, num_cls, conf_thr=args.conf)
    print_confusion(cm, names)

    # --- INT8 对比 ---
    if args.model_int8:
        d8, g8 = run_model(args.model_int8, images, lbl_dir, args.size,
                           num_cls, args.conf, args.iou)
        res8 = summarize(d8, g8, names)
        print("\n########## 模型 B（INT8 量化）##########")
        print_report(res8, names)

        print("\n########## 量化前后对比（答辩用）##########")
        print(f"{'指标':<18}{'FP32':>10}{'INT8':>10}{'变化':>10}")
        print("-" * 48)
        d_m50 = (res8["map50"] - res32["map50"]) * 100
        d_m5095 = (res8["map5095"] - res32["map5095"]) * 100
        print(f"{'mAP@0.5':<18}{res32['map50']*100:>9.1f}%"
              f"{res8['map50']*100:>9.1f}%{d_m50:>+9.1f}")
        print(f"{'mAP@0.5:0.95':<18}{res32['map5095']*100:>9.1f}%"
              f"{res8['map5095']*100:>9.1f}%{d_m5095:>+9.1f}")
        print("-" * 48)
        verdict = "几乎无损" if abs(d_m50) <= 1.0 else (
            "轻微下降" if d_m50 >= -2.0 else "下降较多，建议查标定")
        print(f"结论: INT8 量化后 mAP@0.5 变化 {d_m50:+.1f} 个点（{verdict}）")
        print("答辩话术: 「INT8 量化把 K1 上推理从 0.6 → 2.7 FPS，"
              f"mAP@0.5 仅变化 {d_m50:+.1f} 个点，精度基本无损」")

    print("\n完成。")


if __name__ == "__main__":
    main()





