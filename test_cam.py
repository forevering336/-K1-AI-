#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 摄像头取流测试：自动尝试多条 spacemitsrc GStreamer 管线，
# 找出哪一条能把实时画面送进 OpenCV。
# 前提：/tmp/sdktest.json 已存在（spacemitsrc 需要它）。
#   如果重启过板子，先执行：
#   cp /usr/share/camera_json/csi3_camera_auto.json /tmp/sdktest.json

import cv2
import time
import os

# 依次尝试的候选管线（从最可能到最宽松）
PIPELINES = [
    "spacemitsrc ! video/x-raw,format=NV12,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1",
    "spacemitsrc ! video/x-raw,format=NV12,width=1280,height=720 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1",
    "spacemitsrc ! video/x-raw,format=NV12 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1",
    "spacemitsrc ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1",
    "spacemitsrc ! videoconvert ! appsink drop=true max-buffers=1",
]


def show_build_info():
    info = cv2.getBuildInformation()
    print("OpenCV 版本:", cv2.__version__)
    for line in info.splitlines():
        if "GStreamer" in line:
            print("构建信息:", line.strip())
            break


def try_pipeline(idx, p):
    print("=" * 55)
    print("[%d] 尝试管线:" % idx)
    print("    " + p)
    cap = cv2.VideoCapture(p, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(">> 打开失败 (isOpened=False)")
        cap.release()
        return False

    print(">> 打开成功, 正在读取画面...")
    ok = False
    for i in range(40):            # 最多等约 4 秒拿到第一帧
        ret, frame = cap.read()
        if ret and frame is not None:
            print(">> 读到画面! 尺寸 =", frame.shape)
            cv2.imwrite("/tmp/gstgrab.jpg", frame)
            print(">> 已存图: /tmp/gstgrab.jpg")
            ok = True
            break
        time.sleep(0.1)
    if not ok:
        print(">> 打开了但 4 秒内读不到帧")
    cap.release()
    time.sleep(0.5)               # 给摄像头一点时间释放
    return ok


def main():
    show_build_info()
    if not os.path.exists("/tmp/sdktest.json"):
        print("\n!!! 缺少 /tmp/sdktest.json")
        print("先运行: cp /usr/share/camera_json/csi3_camera_auto.json /tmp/sdktest.json")
        return

    for idx, p in enumerate(PIPELINES, 1):
        if try_pipeline(idx, p):
            print("\n*** 成功! 第 [%d] 条管线可用 ***" % idx)
            print("请打开 /tmp/gstgrab.jpg 看是不是摄像头实时画面。")
            print("\n能用的管线字符串(发给Claude):")
            print(p)
            return

    print("\n!!! 所有管线都没成功。请把这一整屏拍给Claude。")


if __name__ == "__main__":
    main()
