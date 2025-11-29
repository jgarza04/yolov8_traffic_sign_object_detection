from ultralytics import YOLO

DATA_YAML = "/Users/juanpablogarza/Desktop/traffic_sign_dataset/data.yaml"

head_best = "/Users/juanpablogarza/runs/traffic_signs_nc6_v1_head/weights/best.pt"

model = YOLO(head_best)

model.train(
    data=DATA_YAML,
    imgsz=640,
    epochs=15,
    batch=16,
    workers=0,
    device="mps",
    project="/Users/juanpablogarza/runs",
    name="traffic_signs_nc6_v1_full",
    freeze=0,
    val=True,
)
