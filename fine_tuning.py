from ultralytics import YOLO

DATA_YAML = "/Users/juanpablogarza/Desktop/traffic_sign_dataset/data.yaml"

model = YOLO("/Users/juanpablogarza/runs/traffic_signs_v12/weights/best.pt")

model.train(
    data=DATA_YAML,
    imgsz=640,
    epochs=5,   # 5–10 is good for this fine-tune
    batch=16,
    workers=0,
    device="mps",
    project="/Users/juanpablogarza/runs",
    name="traffic_signs_v12_ft_f1_env",
    val=True,
)