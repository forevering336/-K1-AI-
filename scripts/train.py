from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        device=0,
        workers=4,
        project='runs/train',
        name='weld_defect',
        patience=20,
        save=True,
        save_period=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    model.export(format='onnx', imgsz=640, opset=12, simplify=True)
    print("ONNX exported to runs/train/weld_defect/weights/best.onnx")


if __name__ == '__main__':
    main()
