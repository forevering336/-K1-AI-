#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声光报警模块 —— 焊接缺陷检测系统

特点：
- 用 Linux sysfs 标准接口操作 GPIO（/sys/class/gpio），不依赖任何硬件库，
  K1 的 Linux 原生支持；接线引脚通过参数指定，随时可改。
- 无硬件 / 无权限 / 初始化失败 → 自动降级为软件报警（屏幕响铃 \a + [ALARM] 打印），
  绝不阻塞主检测流程。
- 事件驱动：检测到 NG 时调用 trigger()，脉冲在独立短时线程里跑（非阻塞），
  不拖慢推理帧率；带冷却时间，避免连续 NG 频繁触发。
"""
import os
import time
import threading


class _SysfsGPIO:
    """极简 sysfs GPIO 输出封装。任何一步失败都抛异常，交给上层降级。"""

    BASE = "/sys/class/gpio"

    def __init__(self, pin: int):
        self.pin = int(pin)
        self._path = f"{self.BASE}/gpio{self.pin}"
        if not os.path.exists(self._path):
            self._write(f"{self.BASE}/export", str(self.pin))
            # 导出后内核创建节点需要一点时间
            for _ in range(20):
                if os.path.exists(self._path):
                    break
                time.sleep(0.02)
        self._write(f"{self._path}/direction", "out")
        self.low()

    @staticmethod
    def _write(path: str, value: str):
        with open(path, "w") as f:
            f.write(value)

    def high(self):
        self._write(f"{self._path}/value", "1")

    def low(self):
        self._write(f"{self._path}/value", "0")

    def cleanup(self):
        try:
            self.low()
            self._write(f"{self.BASE}/unexport", str(self.pin))
        except Exception:
            pass


class AlarmController:
    """
    缺陷报警控制器。

    参数:
        gpio_buzzer_pin: 蜂鸣器 GPIO 引脚号（None=不启用蜂鸣器）
        gpio_led_pin:    LED GPIO 引脚号（None=不启用 LED）
        cooldown:        冷却时间(秒)，默认 2.0，避免连续 NG 频繁触发
        pulse:           每次报警脉冲时长(秒)，默认 0.3
        enable:          总开关，False 则完全静默
    """

    def __init__(self, gpio_buzzer_pin=None, gpio_led_pin=None,
                 cooldown=2.0, pulse=0.3, enable=True):
        self.enable = enable
        self.cooldown = float(cooldown)
        self.pulse = float(pulse)
        self._last = 0.0
        self._lock = threading.Lock()
        self._busy = False
        self.software_fallback = True   # 默认软件模式，成功初始化硬件后置 False
        self._buzzer = None
        self._led = None

        if not enable:
            print("[ALARM] 报警已禁用")
            return

        # 尝试初始化 GPIO；任一失败即整体降级为软件报警
        try:
            if gpio_buzzer_pin is not None:
                self._buzzer = _SysfsGPIO(gpio_buzzer_pin)
            if gpio_led_pin is not None:
                self._led = _SysfsGPIO(gpio_led_pin)
            if self._buzzer or self._led:
                self.software_fallback = False
                pins = []
                if self._buzzer: pins.append(f"蜂鸣器=GPIO{gpio_buzzer_pin}")
                if self._led:    pins.append(f"LED=GPIO{gpio_led_pin}")
                print(f"[ALARM] 硬件报警就绪 ({', '.join(pins)})")
        except Exception as e:
            self._buzzer = self._led = None
            self.software_fallback = True
            print(f"[ALARM] GPIO 初始化失败({e})，降级为软件报警")

        if self.software_fallback:
            print("[ALARM] 软件报警模式（屏幕响铃 + 控制台提示；Web 端报警不受影响）")

    def trigger(self, defect_info=None):
        """检测到 NG 时调用。冷却期内忽略；脉冲在后台线程执行，不阻塞调用方。"""
        if not self.enable:
            return
        now = time.time()
        with self._lock:
            if now - self._last < self.cooldown or self._busy:
                return
            self._last = now
            self._busy = True
        threading.Thread(target=self._pulse, args=(defect_info,), daemon=True).start()

    def _pulse(self, defect_info):
        try:
            info = f" -> {defect_info}" if defect_info else ""
            if self.software_fallback:
                # 软件降级：系统响铃 + 控制台醒目输出
                try:
                    print("\a", end="", flush=True)
                except Exception:
                    pass
                print(f"[ALARM] 检测到缺陷{info}")
            else:
                try:
                    if self._buzzer: self._buzzer.high()
                    if self._led:    self._led.high()
                    time.sleep(self.pulse)
                finally:
                    if self._buzzer: self._buzzer.low()
                    if self._led:    self._led.low()
                print(f"[ALARM] 声光报警触发{info}")
        finally:
            with self._lock:
                self._busy = False

    def cleanup(self):
        for g in (self._buzzer, self._led):
            if g:
                g.cleanup()
