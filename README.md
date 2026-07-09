# 基于进迭时空 RISC-V K1 的工业焊接缺陷 AI 视觉检测系统

> 参赛队伍：鹿茸粥小队  
> 开发平台：进迭时空 K1 MUSE Pi Pro（RISC-V）  
> 赛题方向：边缘 AI 应用

## 项目简介

本项目面向小型工业焊接质检场景，基于进迭时空 K1 RISC-V 边缘计算平台实现一套全本地部署的工业缺陷检测终端。系统通过 USB 摄像头采集工件图像，在 K1 本地运行 YOLOv8 ONNX 模型，对裂纹、气孔、飞溅和焊缝区域进行实时识别；检测到不合格品时触发声光报警，并通过 Web 仪表盘展示实时画面、FPS、缺陷类型、合格率、历史记录和缺陷截图。

系统构建了完整的本地闭环：

```text
图像采集 -> AI 推理 -> OK/NG 判定 -> 声光报警 -> Web 监控 -> 日志与截图留痕
```

## 核心功能

- USB UVC 摄像头实时采集，使用 ffmpeg 抓帧以适配 K1 Linux 环境。
- YOLOv8 已知缺陷检测：Crack、Porosity、Spatters、Welding line。
- PatchCore 未知异常兜底分支，用于发现训练集中未覆盖的异常外观。
- INT8 ONNX 本地模型部署，推荐使用 `models/best_award_int8.onnx` 作为 K1 端部署模型。
- 声光报警：支持 USB 继电器红绿灯，也保留 GPIO/软件报警封装。
- Web 远程看板：局域网浏览器查看实时状态、历史记录和缺陷截图。
- 本地数据留痕：CSV 检测日志和 NG 截图，便于质量追溯。

## 检测类别

| ID | 英文名 | 中文名 | 判定 |
|---:|---|---|---|
| 0 | Crack | 裂纹 | NG |
| 1 | Porosity | 气孔 | NG |
| 2 | Spatters | 飞溅 | NG |
| 3 | Welding line | 焊缝 | OK，仅显示焊缝区域 |

## 推荐模型

| 文件 | 说明 |
|---|---|
| `models/best_award_int8.onnx` | 推荐 K1 端 INT8 部署模型 |
| `models/best_award_fp32.onnx` | FP32 对照模型 |
| `models/best_award.pt` | 对应 PyTorch 权重 |
| `models/backbone3.onnx` | PatchCore 特征提取 backbone |
| `models/memory_bank2.npy` | PatchCore 正常样本记忆库 |

当前主展示指标采用验证集 valid 结果：

| 模型/划分 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 说明 |
|---|---:|---:|---:|---:|---|
| `best_award.pt` / valid | 86.96% | 79.83% | 86.13% | 62.11% | 验证集 800 张、无背景空图 |
| `best_award_fp32.onnx` / valid | 85.75% | 80.03% | 86.07% | 62.39% | FP32 ONNX 对照模型 |

`best_award_int8.onnx` 作为 K1 端部署模型，用于降低模型体积并提升边缘端推理效率。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── start.sh
├── weld_live.py                 # 实时检测主程序
├── web_server.py                # Flask Web 仪表盘后端
├── usb_alarm.py                 # USB 继电器红绿灯控制
├── alarm.py                     # GPIO/软件报警封装备用
├── anomaly_scorer.py            # PatchCore 异常检测封装
├── test_cam.py                  # 摄像头测试工具
├── light_test.py                # 报警灯测试工具
├── eval_map.py                  # ONNX mAP 评估脚本
├── data.yaml                    # 数据集配置模板
├── data_weld_local.yaml         # 本地评估数据配置模板
├── models/                      # 模型文件
├── web/templates/index.html     # Web 前端页面
├── docs/                        # 报告与说明文档
├── eval/                        # 评估日志与图表
├── samples/                     # 少量演示样图
├── configs/                     # 双分支推理配置
├── patchcore/                   # PatchCore 训练/推理模块
├── scripts/                     # 训练与导出工具脚本
└── deployment/                  # 双分支部署实验代码
```

## K1 板端运行

安装依赖：

```bash
pip3 install -r requirements.txt
sudo apt install ffmpeg
```

一键启动：

```bash
chmod +x start.sh
./start.sh
```

手动启动：

```bash
python3 weld_live.py --web-video --no-display --model models/best_award_int8.onnx
```

浏览器访问：

```text
http://<K1的IP>:5000
```

## 常用测试命令

摄像头测试：

```bash
python3 test_cam.py
```

报警灯测试：

```bash
python3 light_test.py
python3 usb_alarm.py --port /dev/ttyUSB0 --red 1 --green 2
```

PC 端模型评估：

```bash
yolo detect val model=models/best_award_fp32.onnx data=data_weld_local.yaml split=val imgsz=640 batch=16 device=cpu plots=True
yolo detect val model=models/best_award_int8.onnx data=data_weld_local.yaml split=val imgsz=640 batch=16 device=cpu plots=True
```

## 数据集说明

本项目使用 Roboflow Universe 公开焊接缺陷数据集，类别为 Crack、Porosity、Spatters、Welding line。完整训练/验证/测试图片暂未提供，可自行上网查询。

```text
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
└── test/labels
```

如后续改为单独的 `dataset/` 子目录存放数据，需要同步把两个 YAML 文件中的 `path` 改为 `./dataset`。

## 初审重点材料

- `docs/设计报告.md`
- `docs/项目总结.md`
- `docs/源码说明.md`
- `docs/训练实验记录.md`
- `eval/`：模型评估日志与曲线图
- `eval/best_award_fp32_valid_assets/`：valid 验证集可视化图表

旧版训练图表、完整数据集、训练缓存和内部 test 分析已移入 `非提交材料_数据集训练缓存/`，最终模型评估请以 `eval/` 目录为准。

