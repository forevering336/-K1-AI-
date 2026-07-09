#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USB串口灯 协议扫描器：一组组发常见命令，盯着灯看哪组亮。"""
import serial, time, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"

# 常见 USB 继电器/串口报警灯的"开"命令(多种波特率+多套协议)
TESTS = [
    # (编号, 波特率, 说明, 开命令bytes, 关命令bytes)
    (1,  9600, "LCUS/常见继电器 通道1开",  bytes([0xA0,0x01,0x01,0xA2]), bytes([0xA0,0x01,0x00,0xA1])),
    (2,  9600, "LCUS 通道2开",            bytes([0xA0,0x02,0x01,0xA3]), bytes([0xA0,0x02,0x00,0xA2])),
    (3,  9600, "单字节 0x01/0x00",         bytes([0x01]),                bytes([0x00])),
    (4,  9600, "单字节 0xFF/0x00",         bytes([0xFF]),                bytes([0x00])),
    (5,  9600, "文本 ON/OFF",             b"ON\r\n",                     b"OFF\r\n"),
    (6,  9600, "文本 open/close",         b"open\r\n",                   b"close\r\n"),
    (7,  9600, "5字节继电器 通道1",         bytes([0x55,0x01,0x01,0x00,0x57]), bytes([0x55,0x01,0x00,0x00,0x56])),
    (8, 115200,"115200 LCUS 通道1开",     bytes([0xA0,0x01,0x01,0xA2]), bytes([0xA0,0x01,0x00,0xA1])),
    (9, 115200,"115200 单字节0x01",        bytes([0x01]),                bytes([0x00])),
    (10, 9600, "DTR/RTS 拉高(有些灯靠这个)", None, None),
]

for num, baud, desc, on, off in TESTS:
    print(f"\n=== [{num}] 波特率{baud} : {desc} ===  盯着灯!")
    try:
        s = serial.Serial(PORT, baud, timeout=1)
        if on is None:  # DTR/RTS 测试
            s.dtr = True; s.rts = True; time.sleep(2)
            print("    (DTR/RTS 拉高中...)"); s.dtr = False; s.rts = False
        else:
            s.write(on); s.flush(); print(f"    发送开: {on.hex(' ') if isinstance(on,bytes) else on}")
            time.sleep(2)
            s.write(off); s.flush(); print("    发送关")
        s.close()
    except Exception as e:
        print("    出错:", e)
    time.sleep(0.5)

print("\n完成。哪个编号让灯亮了/响了，告诉我编号即可。")
