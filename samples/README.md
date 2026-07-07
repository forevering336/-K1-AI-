# 测试样图

用于无摄像头环境下快速验证模型效果。

## 样图说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `sample_crack.jpg` | 含裂纹缺陷 | 测试缺陷检测 (Crack) |
| `sample_porosity.jpg` | 含气孔缺陷 | 测试缺陷检测 (Porosity) |
| `sample_normal.jpg` | 正常焊缝 | 测试 OK 判定 |
| `sample_normal2.jpg` | 正常焊缝2 | 测试 OK 判定 |
| `sample_normal3.jpg` | 正常焊缝3 | 测试 OK 判定 |

## 使用方式

```bash
# 单张图片测试
python3 weld_live.py --image samples/sample_crack.jpg

# 文件夹循环检测
python3 weld_live.py --folder samples/
```

> 更多样图请从数据集下载：https://universe.roboflow.com/trial-z2qzo/weld-ddeij-fkpig/dataset/1
