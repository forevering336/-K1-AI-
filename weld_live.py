#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
焊接缺陷实时检测系统 —— 进迭时空 K1 (MUSE Pi Pro) · USB摄像头版
模型: best_award_int8.onnx (YOLOv8, 4类输出, 3类缺陷报警, 输入640x640, 输出[1,8,8400])
摄像头: USB UVC, 设备 /dev/video20 (用 ffmpeg 抓帧, 绕开 opencv 的坏抓帧)
推理: ONNX Runtime + SpaceMITExecutionProvider (RVV/AI指令硬件加速)

本版新增（不动模型、不动推理，全部按需开启，向后兼容）:
    · 声光报警 (alarm.py): 检测到 NG 时 GPIO 触发 LED+蜂鸣器, 无硬件自动降级软件报警
    · Web 仪表盘 (web_server.py): 局域网浏览器实时监控 + 实时带框画面 + 历史/截图回看
    · headless 无头模式: 不接显示器也能跑 (远程 Web 监控)

用法(用系统 python3 跑, 不要进 venv):
    python3 weld_live.py --test          # 抓30帧跑推理, 存一张标注图, 不开窗口
    python3 weld_live.py                 # 实时检测(开窗口, 画面上按 q 退出)  ← 原有行为不变
    python3 weld_live.py --image x.jpg   # 单图测试(保底)
    python3 weld_live.py --folder 目录    # 图片文件夹循环检测(保底演示)

    python3 weld_live.py --web                       # 实时检测 + Web 仪表盘(:5000)
    python3 weld_live.py --web --web-video           # Web 里显示实时带框画面
    python3 weld_live.py --web --no-display          # 无头模式(不接显示器, 纯远程监控)
    python3 weld_live.py --web-video --gpio-buzzer 22 --gpio-led 23   # 全功能
"""
import os
import csv
import sys
import time
import argparse
import datetime
import subprocess
import threading
import cv2
import numpy as np

# ============================================================
# 配置区
# ============================================================
CAMERA_DEV   = "/dev/video20"       # USB摄像头设备节点
CAM_W, CAM_H = 640, 480             # ffmpeg列出的支持分辨率(MJPG 640x480)
CAM_FMT      = "mjpeg"              # 摄像头支持 mjpeg / yuyv422
INPUT_SIZE   = 640
CONF_THRES   = 0.45                 # 误报多调高, 漏检多调低
IOU_THRES    = 0.45
CLASS_NAMES  = ['Crack', 'Porosity', 'Spatters', 'Welding line']
CLASS_CN     = ['裂纹', '气孔', '飞溅', '焊缝']
DEFECT_IDS   = {0, 1, 2}            # 焊缝(3)是正常特征,不算缺陷
COLORS       = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)]
SAVE_DIR     = "output"
LOG_CSV      = "detection_log.csv"
SAVE_DEFECT  = True

# 模型自动查找位置(按顺序找第一个存在的)
MODEL_CANDIDATES = [
    "models/best_award_int8.onnx",  # final INT8 deployment model
    "models/best_award_fp32.onnx",  # FP32 reference model
    "best_award_int8.onnx",
    "best_award_fp32.onnx",
    os.path.expanduser("~/weld/best_award_int8.onnx"),
    os.path.expanduser("~/weld/best_award_fp32.onnx"),
    "/media/bb/C68E-AC53/weld/best_award_int8.onnx",
    "/media/bb/C68E-AC53/weld/best_award_fp32.onnx",
]


def find_model(user_path=None):
    if user_path:
        if os.path.exists(user_path):
            return user_path
        print(f"[ERROR] 指定的模型不存在: {user_path}"); sys.exit(1)
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    print("[ERROR] 没找到模型。请把 best_award_int8.onnx 放到当前目录,或用 --model 指定路径。")
    sys.exit(1)


# ============================================================
# 摄像头自动识别(免得手动查 /dev/videoN)
# ============================================================
def _probe_capture(dev, fmt, w=640, h=480, timeout=4):
    """用 ffmpeg 抓 1 帧, 拿到足够数据就算这个节点+格式能用。"""
    cmd = ["ffmpeg", "-loglevel", "error", "-nostdin",
           "-f", "v4l2", "-input_format", fmt,
           "-video_size", f"{w}x{h}", "-i", dev,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return len(p.stdout) >= w * h  # 拿到一帧的量级即认为可用
    except Exception:
        return False


def autodetect_camera():
    """扫描所有 /dev/video*, 找出真正能出图的 USB 摄像头节点和像素格式。
    返回 (dev, fmt); 找不到返回 (None, None)。"""
    cands = []
    base = "/sys/class/video4linux"
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if not d.startswith("video"):
                continue
            try:
                nm = open(os.path.join(base, d, "name")).read().strip()
            except Exception:
                nm = ""
            cands.append(("/dev/" + d, nm))
    else:
        cands = [("/dev/" + f, "") for f in sorted(os.listdir("/dev")) if f.startswith("video")]

    # 名字像 USB 摄像头的优先探测(内部 ISP 节点排后面)
    kw = ("usb", "camera", "composite", "uvc", "cam", "web")
    cands.sort(key=lambda x: 0 if any(k in x[1].lower() for k in kw) else 1)

    print("[CAM] 自动识别摄像头中(试抓帧, 稍等)...")
    for node, nm in cands:
        for fmt in ("mjpeg", "yuyv422"):
            if _probe_capture(node, fmt):
                print(f"[CAM] ✔ 找到摄像头: {node}  名称='{nm}'  格式={fmt}")
                return node, fmt
    return None, None


# ============================================================
# 推理引擎(与已验证版本一致)
# ============================================================
class Detector:
    def __init__(self, model_path):
        import onnxruntime as ort
        try:
            import spacemit_ort  # noqa: 注册SpaceMIT加速器
        except Exception:
            pass
        avail = ort.get_available_providers()
        if 'SpaceMITExecutionProvider' in avail:
            providers = ['SpaceMITExecutionProvider', 'CPUExecutionProvider']
            print("[INFO] 使用 SpaceMIT 硬件加速推理")
        else:
            providers = ['CPUExecutionProvider']
            print("[INFO] 未检测到SpaceMIT加速器, 使用CPU推理")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        print(f"[INFO] 模型已加载: {model_path}")

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        dw, dh = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized
        return canvas, r, dw, dh

    def infer(self, frame):
        img, r, dw, dh = self._letterbox(frame)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        out = self.sess.run(None, {self.input_name: blob})[0]
        return self._postprocess(out, r, dw, dh, frame.shape)

    def _postprocess(self, out, r, dw, dh, orig_shape):
        out = np.squeeze(out, 0).T
        boxes_xywh = out[:, :4]
        scores_all = out[:, 4:]
        cls_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(len(scores_all)), cls_ids]
        mask = confs > CONF_THRES
        if not np.any(mask):
            return []
        boxes_xywh = boxes_xywh[mask]; confs = confs[mask]; cls_ids = cls_ids[mask]
        cx, cy, ww, hh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - ww / 2; y1 = cy - hh / 2; x2 = cx + ww / 2; y2 = cy + hh / 2
        x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
        H, W = orig_shape[:2]
        x1 = np.clip(x1, 0, W); y1 = np.clip(y1, 0, H)
        x2 = np.clip(x2, 0, W); y2 = np.clip(y2, 0, H)
        rects = [[int(a), int(b), int(c - a), int(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        idxs = cv2.dnn.NMSBoxes(rects, confs.tolist(), CONF_THRES, IOU_THRES)
        dets = []
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                dets.append([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
                             float(confs[i]), int(cls_ids[i])])
        return dets


# ============================================================
# ffmpeg 抓帧(绕开 opencv 的坏 V4L2 抓帧)
# ============================================================
class FfmpegCamera:
    """后台线程持续抓帧, 只保留最新一帧, 消除管道积压导致的延迟/卡顿。"""
    def __init__(self, device=CAMERA_DEV, width=CAM_W, height=CAM_H, fmt=CAM_FMT, fps=15):
        self.width = width; self.height = height
        self.frame_size = width * height * 3
        cmd = [
            "ffmpeg", "-loglevel", "error", "-nostdin",
            "-f", "v4l2", "-input_format", fmt,
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", device, "-an",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=self.frame_size)
        self.latest = None
        self.lock = threading.Lock()
        self.running = True
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        fs = self.frame_size
        while self.running:
            buf = b""
            while len(buf) < fs:
                chunk = self.proc.stdout.read(fs - len(buf))
                if not chunk:
                    self.running = False
                    return
                buf += chunk
            frame = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
            with self.lock:
                self.latest = frame

    def read(self):
        # 等第一帧最多2秒
        for _ in range(100):
            with self.lock:
                if self.latest is not None:
                    return True, self.latest.copy()
            if not self.running:
                return False, None
            time.sleep(0.02)
        with self.lock:
            if self.latest is not None:
                return True, self.latest.copy()
        return False, None

    def release(self):
        self.running = False
        try:
            self.proc.terminate(); self.proc.wait(timeout=2)
        except Exception:
            try: self.proc.kill()
            except Exception: pass


# ============================================================
# 界面绘制
# ============================================================
def draw_ui(frame, dets, fps, total_count, defect_count):
    out = frame.copy()
    H, W = out.shape[:2]
    has_defect = any(d[5] in DEFECT_IDS for d in dets)
    for x1, y1, x2, y2, conf, cid in dets:
        color = COLORS[cid % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cid]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(out, (0, 0), (W, 40), (40, 40, 40), -1)
    cv2.putText(out, "Weld Defect Inspection - K1 RISC-V", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, f"FPS:{fps:4.1f}", (W - 110, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    status = "NG" if has_defect else "OK"
    scolor = (0, 0, 255) if has_defect else (0, 200, 0)
    cv2.rectangle(out, (W - 120, 45), (W - 10, 95), scolor, -1)
    cv2.putText(out, status, (W - 105, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    pass_count = total_count - defect_count
    rate = (pass_count / total_count * 100) if total_count else 100.0
    cv2.rectangle(out, (0, H - 30), (W, H), (40, 40, 40), -1)
    cv2.putText(out, f"Total:{total_count}  Defect:{defect_count}  Pass:{rate:.1f}%",
                (10, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out, has_defect


def open_log():
    new_log = not os.path.exists(LOG_CSV)
    logf = open(LOG_CSV, "a", newline="")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["time", "class", "confidence", "bbox"])
    return logf, writer


def log_detection(writer, dets):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for x1, y1, x2, y2, conf, cid in dets:
        writer.writerow([ts, CLASS_NAMES[cid], f"{conf:.3f}", f"{x1},{y1},{x2},{y2}"])


# ============================================================
# 模式: --test  (不开窗口, 抓N帧跑推理, 存一张标注图)
# ============================================================
def run_test(detector, n=30):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[INFO] 测试模式: 通过 ffmpeg 抓 {n} 帧并推理 ...")
    cam = FfmpegCamera(device=CAMERA_DEV, fmt=CAM_FMT)
    ok = 0; last_ui = None; t_start = time.time()
    for i in range(n):
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] 抓帧失败,摄像头可能断了"); break
        dets = detector.infer(frame)
        ui, has_defect = draw_ui(frame, dets, 0.0, i + 1, 0)
        last_ui = ui; ok += 1
        names = [CLASS_NAMES[d[5]] for d in dets]
        print(f"  帧{i+1:02d}: 检测到 {len(dets)} 个目标 {names}")
    cam.release()
    if last_ui is not None:
        outp = os.path.join(SAVE_DIR, "test_annotated.jpg")
        cv2.imwrite(outp, last_ui)
        dt = time.time() - t_start
        print(f"[OK] 成功推理 {ok} 帧, 平均 {ok/dt:.1f} FPS")
        print(f"[OK] 已存标注图: {outp}  —— 用 xdg-open {outp} 打开看看")
    else:
        print("[ERROR] 一帧都没抓到")


# ============================================================
# 模式: 实时 (窗口 / 无头, 可选 报警 + Web)
# ============================================================
def run_live(detector, alarm=None, web=False, port=5000, display=True, web_video=False,
             scorer=None, anomaly_every=3):
    os.makedirs(SAVE_DIR, exist_ok=True)
    logf, writer = open_log()
    cam = FfmpegCamera(device=CAMERA_DEV, fmt=CAM_FMT)

    state = {
        # ---- 原有字段 ----
        "dets": [], "total": 0, "defect": 0, "ifps": 0.0,
        "running": True, "last_save": 0.0,
        # ---- 声光/Web 字段 ----
        "status": "OK",              # "OK" | "NG"
        "recent_alerts": [],         # 最近告警(最新在前, 最多20条)
        "start_time": time.time(),   # 系统启动时间
        "ui_frame": None,            # 供 Web 视频流的最新带框画面
        # ---- 异常检测分支字段 ----
        "anomaly_on": scorer is not None,
        "anomaly_score": 0.0,
        "anomaly_thresh": (scorer.threshold if scorer is not None else 0.0),
        "last_heatmap": None,
        "anomaly_hit": False,
    }
    lock = threading.Lock()
    frame_count = {"n": 0}

    def get_frame():
        with lock:
            f = state.get("ui_frame")
            return None if f is None else f.copy()

    # ---- 启动 Web 服务(可选) ----
    if web:
        try:
            import web_server
            threading.Thread(target=web_server.serve, kwargs=dict(
                state=state, lock=lock, get_frame=get_frame,
                log_csv=LOG_CSV, save_dir=SAVE_DIR,
                host="0.0.0.0", port=port, enable_video=web_video,
            ), daemon=True).start()
        except Exception as e:
            print(f"[WEB] 启动失败({e})，继续跑检测，仅无 Web")

    # ---- 推理线程(唯一的 state 写入方) ----
    def infer_worker():
        while state["running"]:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.05); continue
            t0 = time.time()
            dets = detector.infer(frame)          # 分支A: YOLO 已知缺陷
            dt = time.time() - t0
            defect_yolo = any(d[5] in DEFECT_IDS for d in dets)
            hms = datetime.datetime.now().strftime("%H:%M:%S")

            # 分支B: 无监督异常检测(隔 anomaly_every 帧跑一次, 省 CPU)
            anomaly_hit = False; anomaly_score = 0.0; heatmap = None
            if scorer is not None:
                frame_count["n"] += 1
                if frame_count["n"] % max(1, anomaly_every) == 0:
                    try:
                        anomaly_score, heatmap, _ = scorer.score(frame)
                        anomaly_hit = scorer.is_anomaly(anomaly_score)
                    except Exception as e:
                        print(f"[ANOMALY] 打分失败: {e}")

            has_defect = defect_yolo or anomaly_hit   # 融合: 任一分支报缺陷即 NG
            with lock:
                state["dets"] = dets
                state["total"] += 1
                state["status"] = "NG" if has_defect else "OK"
                if heatmap is not None:
                    state["anomaly_score"] = anomaly_score
                    state["last_heatmap"] = heatmap
                    state["anomaly_hit"] = anomaly_hit
                if has_defect:
                    state["defect"] += 1
                    for x1, y1, x2, y2, conf, cid in dets:
                        if cid in DEFECT_IDS:
                            state["recent_alerts"].insert(0, {
                                "time": hms, "class_en": CLASS_NAMES[cid],
                                "class_cn": CLASS_CN[cid], "conf": round(float(conf), 3),
                                "status": "NG",
                            })
                    if anomaly_hit:
                        state["recent_alerts"].insert(0, {
                            "time": hms, "class_en": "Unknown", "class_cn": "未知异常",
                            "conf": round(float(anomaly_score), 3), "status": "NG",
                        })
                    del state["recent_alerts"][20:]
                state["ifps"] = 0.9 * state["ifps"] + 0.1 * (1.0 / (dt + 1e-6))

            if has_defect:
                # 声光报警(事件驱动, 内部冷却, 非阻塞)
                if alarm is not None:
                    names = ",".join(CLASS_CN[d[5]] for d in dets if d[5] in DEFECT_IDS)
                    if anomaly_hit:
                        names = (names + ",未知异常") if names else "未知异常"
                    alarm.trigger(names)
                log_detection(writer, dets)
                if anomaly_hit:                  # 未知缺陷也记进日志
                    writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     "Unknown", f"{anomaly_score:.3f}", ""])
                logf.flush()
                if SAVE_DEFECT and time.time() - state["last_save"] > 1.0:
                    ui_s, _ = draw_ui(frame, dets, state["ifps"], state["total"], state["defect"])
                    if scorer is not None and anomaly_hit and heatmap is not None:
                        ui_s = scorer.overlay(ui_s, heatmap)
                    fn = os.path.join(SAVE_DIR,
                        datetime.datetime.now().strftime("defect_%Y%m%d_%H%M%S.jpg"))
                    cv2.imwrite(fn, ui_s); state["last_save"] = time.time()

    wt = threading.Thread(target=infer_worker, daemon=True)
    wt.start()

    # ---- 视频线程: 给 Web 的画面用摄像头原始帧跑满帧(~15fps), 叠"最近一次"的检测框,
    #      与慢速推理解耦 —— 推理 0.x fps 也不影响手机端画面流畅 ----
    def video_worker():
        while state["running"]:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.03); continue
            with lock:
                dets = list(state["dets"]); total = state["total"]
                defect = state["defect"]; ifps = state["ifps"]
                hm = state.get("last_heatmap"); hit = state.get("anomaly_hit", False)
            ui, _ = draw_ui(frame, dets, ifps, total, defect)
            if scorer is not None and hit and hm is not None and hm.shape[:2] == ui.shape[:2]:
                ui = scorer.overlay(ui, hm)     # 叠加未知缺陷洋红热区
            with lock:
                state["ui_frame"] = ui
            time.sleep(1.0 / 15)                # ~15fps, 匹配摄像头帧率

    if web_video:
        threading.Thread(target=video_worker, daemon=True).start()

    try:
        if display:
            print("[INFO] 实时检测中 —— 在画面窗口上按 q 退出")
            while True:
                ret, frame = cam.read()
                if not ret:
                    time.sleep(0.03); continue
                with lock:
                    dets = list(state["dets"]); total = state["total"]
                    defect = state["defect"]; ifps = state["ifps"]
                    hm = state.get("last_heatmap"); hit = state.get("anomaly_hit", False)
                ui, _ = draw_ui(frame, dets, ifps, total, defect)
                if scorer is not None and hit and hm is not None and hm.shape[:2] == ui.shape[:2]:
                    ui = scorer.overlay(ui, hm)     # 本地窗口也叠加未知缺陷热区
                cv2.imshow("Weld Defect Inspection", ui)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
        else:
            tip = f"http://<K1_IP>:{port}" if web else ""
            print(f"[INFO] 无头模式运行中(后台推理/报警/Web {tip}) —— 按 Ctrl+C 退出")
            while state["running"]:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        state["running"] = False
        cam.release(); logf.close()
        if alarm is not None:
            alarm.cleanup()
        try: cv2.destroyAllWindows()
        except Exception: pass
    print("[INFO] 已退出")


# ============================================================
# 保底模式: 单图 / 文件夹
# ============================================================
def run_image(detector, path):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] 读不了图片: {path}"); return
    dets = detector.infer(img)
    ui, has_defect = draw_ui(img, dets, 0.0, 1, 1 if has_defect_of(dets) else 0)
    outp = os.path.join(SAVE_DIR, "result.jpg")
    os.makedirs(SAVE_DIR, exist_ok=True)
    cv2.imwrite(outp, ui)
    print(f"[OK] 检测到 {len(dets)} 个目标, 结果存于 {outp}")
    try:
        cv2.imshow("result", ui); cv2.waitKey(0); cv2.destroyAllWindows()
    except Exception:
        pass


def has_defect_of(dets):
    return any(d[5] in DEFECT_IDS for d in dets)


def run_folder(detector, folder):
    os.makedirs(SAVE_DIR, exist_ok=True)
    files = [f for f in sorted(os.listdir(folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not files:
        print(f"[ERROR] 文件夹里没有图片: {folder}"); return
    total = 0; defects = 0
    print(f"[INFO] 文件夹循环检测 {len(files)} 张, 按 q 退出")
    try:
        while True:
            for f in files:
                img = cv2.imread(os.path.join(folder, f))
                if img is None:
                    continue
                t0 = time.time()
                dets = detector.infer(img)
                total += 1
                if has_defect_of(dets):
                    defects += 1
                fps = 1.0 / (time.time() - t0 + 1e-6)
                ui, _ = draw_ui(img, dets, fps, total, defects)
                try:
                    cv2.imshow("Weld Defect Inspection", ui)
                    if (cv2.waitKey(800) & 0xFF) == ord('q'):
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise
                except Exception:
                    print(f"  {f}: {len(dets)} 个目标")
                    time.sleep(0.8)
    except KeyboardInterrupt:
        pass
    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="ONNX 模型路径")
    ap.add_argument("--test", action="store_true", help="抓30帧跑推理,不开窗口")
    ap.add_argument("--image", default=None, help="单图测试")
    ap.add_argument("--folder", default=None, help="图片文件夹循环检测")
    ap.add_argument("--dev", default="auto", help="摄像头设备(默认 auto 自动识别; 也可写 /dev/videoN)")
    ap.add_argument("--fmt", default=None, help="像素格式 mjpeg/yuyv422(默认自动)")
    # ---- Web 仪表盘 ----
    ap.add_argument("--web", action="store_true", help="启用 Web 仪表盘(默认不启用)")
    ap.add_argument("--port", type=int, default=5000, help="Web 服务端口(默认5000)")
    ap.add_argument("--web-video", action="store_true",
                    help="Web 仪表盘显示实时带框画面(会自动启用 --web)")
    ap.add_argument("--no-display", action="store_true",
                    help="关闭 OpenCV 窗口(无头模式, 不接显示器)")
    # ---- 声光报警 ----
    ap.add_argument("--gpio-buzzer", type=int, default=None, help="蜂鸣器 GPIO 引脚号")
    ap.add_argument("--gpio-led", type=int, default=None, help="LED GPIO 引脚号")
    ap.add_argument("--alarm-cooldown", type=float, default=2.0, help="报警冷却时间,秒(默认2.0)")
    ap.add_argument("--no-alarm", action="store_true", help="完全禁用报警")
    # ---- 异常检测分支(PatchCore, 兜未知缺陷) ----
    ap.add_argument("--anomaly", action="store_true", help="启用无监督异常检测分支(检测未知缺陷)")
    ap.add_argument("--anomaly-backbone", default="models/backbone3.onnx", help="特征提取 ONNX")
    ap.add_argument("--anomaly-bank", default="models/memory_bank2.npy", help="正常样本记忆库 .npy")
    ap.add_argument("--anomaly-thresh", type=float, default=0.5, help="异常判定阈值(先用 anomaly_scorer.py --calib 标定)")
    ap.add_argument("--anomaly-every", type=int, default=3, help="每N帧跑一次异常检测,省CPU(默认3)")
    args = ap.parse_args()

    global CAMERA_DEV, CAM_FMT
    # 摄像头设备: auto=自动识别(单图/文件夹模式不需要摄像头, 跳过识别)
    if args.dev == "auto" and not (args.image or args.folder):
        dev, fmt = autodetect_camera()
        if dev is None:
            print("[ERROR] 没自动找到能出图的摄像头。检查是否插好, 或手动指定 --dev /dev/videoN")
            CAMERA_DEV = "/dev/video20"      # 兜底, 实时模式若失败请手动指定
        else:
            CAMERA_DEV = dev
            if args.fmt is None:
                CAM_FMT = fmt
    elif args.dev != "auto":
        CAMERA_DEV = args.dev
    if args.fmt:
        CAM_FMT = args.fmt

    model = find_model(args.model)
    detector = Detector(model)

    if args.image:
        run_image(detector, args.image)
    elif args.folder:
        run_folder(detector, args.folder)
    elif args.test:
        run_test(detector)
    else:
        # 实时模式: 组装报警 + Web
        alarm = None
        if not args.no_alarm:
            try:
                from alarm import AlarmController
                alarm = AlarmController(gpio_buzzer_pin=args.gpio_buzzer,
                                        gpio_led_pin=args.gpio_led,
                                        cooldown=args.alarm_cooldown)
            except Exception as e:
                print(f"[ALARM] 报警模块加载失败({e})，无报警继续运行")
        # 异常检测分支(可选)
        scorer = None
        if args.anomaly:
            try:
                from anomaly_scorer import AnomalyScorer
                scorer = AnomalyScorer(args.anomaly_backbone, args.anomaly_bank,
                                       threshold=args.anomaly_thresh)
            except Exception as e:
                print(f"[ANOMALY] 异常分支加载失败({e})，仅用 YOLO 继续")
        web = args.web or args.web_video
        run_live(detector, alarm=alarm, web=web, port=args.port,
                 display=not args.no_display, web_video=args.web_video,
                 scorer=scorer, anomaly_every=args.anomaly_every)


if __name__ == "__main__":
    main()





