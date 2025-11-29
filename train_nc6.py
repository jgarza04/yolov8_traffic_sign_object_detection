from ultralytics import YOLO

DATA_YAML = "/Users/juanpablogarza/Desktop/traffic_sign_dataset/data.yaml"

# COCO-pretrained YOLOv8n, will adapt head to nc=6
model = YOLO("yolov8n.pt")

# Phase 1: warmup, freeze backbone
model.train(
    data=DATA_YAML,
    imgsz=640,
    epochs=5,
    batch=16,
    workers=0,
    device="mps",
    project="/Users/juanpablogarza/runs",
    name="traffic_signs_nc6_v1_head",
    freeze=10,
    val=True,
)

# Phase 2: unfreeze, full fine-tune
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
