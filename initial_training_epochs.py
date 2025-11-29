from ultralytics import YOLO

DATA_YAML = "/Users/juanpablogarza/Desktop/traffic_sign_dataset/data.yaml"

model = YOLO("yolov8n.pt")

model.train(data=DATA_YAML, 
            imgsz=640, 
            epochs=10, 
            batch=16,
            workers=0,              # safer on macOS
            device="mps",           # or "mps" if available
            project="/Users/juanpablogarza/runs",
            name="traffic_signs_v1",
            val=True
            )


model = YOLO("/Users/juanpablogarza/runs/traffic_signs_v1/weights/best.pt")

model.predict(source="/Users/juanpablogarza/Desktop/traffic_sign_dataset/images/valid",save=True)