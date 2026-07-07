#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web 仪表盘后端 —— 纯 Flask（不用 SocketIO）

为什么不用 Flask-SocketIO：
- 数据每秒更新一次，用不上 WebSocket 的双向低延迟；
- 浏览器每秒轮询 /api/stats 就够，少装两个包，在 RISC-V 上更稳。

提供：
- /                实时仪表盘页面（离线可用，无 CDN 依赖）
- /api/stats       实时统计 JSON（前端每秒轮询）
- /api/history     历史记录分页 + 按日期/类别筛选（读 detection_log.csv）
- /api/images      NG 截图文件列表
- /api/image/<fn>  单张截图
- /video_feed      带检测框的实时画面（MJPEG，multipart/x-mixed-replace）

用法（在 weld_live.py 里）：
    import web_server
    threading.Thread(target=web_server.serve, kwargs=dict(
        state=state, lock=lock, get_frame=get_frame,
        log_csv=LOG_CSV, save_dir=SAVE_DIR,
        host="0.0.0.0", port=5000, enable_video=True,
    ), daemon=True).start()
"""
import os
import csv
import time

import cv2
from flask import (Flask, jsonify, request, render_template,
                   send_from_directory, Response, abort)


def _fmt_uptime(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def serve(state, lock, get_frame=None, log_csv="detection_log.csv",
          save_dir="output", host="0.0.0.0", port=5000, enable_video=True):
    """启动 Flask 服务（阻塞）。通常放到 daemon 线程里跑。"""
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(__name__,
                template_folder=os.path.join(here, "web", "templates"),
                static_folder=os.path.join(here, "web", "static"))
    # 关掉访问日志噪音
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    save_dir_abs = os.path.abspath(save_dir)

    @app.route("/")
    def index():
        return render_template("index.html", enable_video=enable_video)

    @app.route("/api/stats")
    def api_stats():
        with lock:
            total = state.get("total", 0)
            defect = state.get("defect", 0)
            fps = state.get("ifps", 0.0)
            status = state.get("status", "OK")
            start = state.get("start_time", time.time())
            alerts = list(state.get("recent_alerts", []))[:5]
            anomaly_on = state.get("anomaly_on", False)
            anomaly_score = state.get("anomaly_score", 0.0)
            anomaly_thresh = state.get("anomaly_thresh", 0.0)
        pass_rate = ((total - defect) / total * 100.0) if total else 100.0
        return jsonify({
            "fps": round(fps, 1),
            "total": total,
            "defect": defect,
            "pass_rate": round(pass_rate, 1),
            "status": status,
            "uptime": _fmt_uptime(time.time() - start),
            "alerts": alerts,
            "video": enable_video,
            "anomaly_on": anomaly_on,               # 是否启用了异常检测分支
            "anomaly_score": round(anomaly_score, 3),
            "anomaly_thresh": round(anomaly_thresh, 3),
        })

    @app.route("/api/recent_alerts")
    def api_recent_alerts():
        with lock:
            return jsonify(list(state.get("recent_alerts", [])))

    @app.route("/api/history")
    def api_history():
        page = max(1, request.args.get("page", 1, type=int))
        limit = min(500, max(1, request.args.get("limit", 50, type=int)))
        date = (request.args.get("date") or "").strip()
        cls = (request.args.get("class") or "").strip()

        rows = []
        if os.path.exists(log_csv):
            try:
                with open(log_csv, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for r in reader:
                        if len(r) < 3:
                            continue
                        t, c, conf = r[0], r[1], r[2]
                        bbox = r[3] if len(r) > 3 else ""
                        if date and not t.startswith(date):
                            continue
                        if cls and cls.lower() not in c.lower():
                            continue
                        rows.append({"time": t, "class": c,
                                     "confidence": conf, "bbox": bbox})
            except Exception as e:
                return jsonify({"error": str(e), "rows": [], "total": 0})

        rows.reverse()  # 最新在前
        total = len(rows)
        start = (page - 1) * limit
        return jsonify({
            "rows": rows[start:start + limit],
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit,
        })

    @app.route("/api/images")
    def api_images():
        files = []
        if os.path.isdir(save_dir_abs):
            for fn in os.listdir(save_dir_abs):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn.startswith("defect_"):
                    files.append(fn)
        files.sort(reverse=True)  # 文件名带时间戳，倒序=最新在前
        return jsonify(files[:200])

    @app.route("/api/image/<path:filename>")
    def api_image(filename):
        # 只允许目录内的文件，防目录穿越
        safe = os.path.basename(filename)
        if not safe or safe in (".", ".."):
            abort(404)
        if not os.path.exists(os.path.join(save_dir_abs, safe)):
            abort(404)
        return send_from_directory(save_dir_abs, safe)

    @app.route("/video_feed")
    def video_feed():
        if not enable_video or get_frame is None:
            abort(404)

        def gen():
            while True:
                frame = get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg.tobytes() + b"\r\n")
                time.sleep(0.1)  # ~10 fps，控 CPU

        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"[WEB] 仪表盘已启动: http://{host}:{port}  (局域网内浏览器可访问 http://<K1_IP>:{port})")
    app.run(host=host, port=port, threaded=True, use_reloader=False, debug=False)
