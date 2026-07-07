#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USB 红绿报警灯驱动 (LCUS 继电器, CH341 串口)。独立进程, 轮询 /api/stats 驱动灯。"""
import serial, time, json, argparse, urllib.request

def cmd(ch, on):
    return bytes([0xA0, ch, 1 if on else 0, (0xA0 + ch + (1 if on else 0)) & 0xFF])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--red", type=int, default=1, help="红灯通道(默认1)")
    ap.add_argument("--green", type=int, default=2, help="绿灯通道(默认2)")
    ap.add_argument("--url", default="http://127.0.0.1:5000/api/stats")
    ap.add_argument("--hold", type=float, default=2.0, help="NG后红灯至少保持秒数")
    ap.add_argument("--test", action="store_true", help="测试红绿映射")
    a = ap.parse_args()

    s = serial.Serial(a.port, a.baud, timeout=1)
    def red(on):   s.write(cmd(a.red, on));   s.flush()
    def green(on): s.write(cmd(a.green, on)); s.flush()
    def all_off(): red(False); green(False)

    if a.test:
        print("红灯(通道%d) 亮3秒..." % a.red); all_off(); red(True); time.sleep(3)
        print("绿灯(通道%d) 亮3秒..." % a.green); red(False); green(True); time.sleep(3)
        all_off(); print("灭。接反就加 --red 2 --green 1 调换")
        return

    print(f"[灯] 已连 {a.port}  红=通道{a.red} 绿=通道{a.green}  轮询 {a.url}")
    all_off(); green(True)
    state = "OK"; last_ng = 0.0
    try:
        while True:
            try:
                with urllib.request.urlopen(a.url, timeout=1) as r:
                    st = json.load(r).get("status", "OK")
            except Exception:
                time.sleep(0.5); continue
            if st == "NG":
                last_ng = time.time()
                if state != "NG":
                    green(False); red(True); state = "NG"; print("[灯] NG -> 红")
            else:
                if state == "NG" and time.time() - last_ng >= a.hold:
                    red(False); green(True); state = "OK"; print("[灯] OK -> 绿")
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        all_off(); s.close(); print("\n[灯] 退出, 已灭灯")

if __name__ == "__main__":
    main()
