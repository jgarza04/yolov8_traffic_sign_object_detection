from ultralytics import YOLO

model = YOLO("/Users/juanpablogarza/runs/traffic_signs_nc6_v1_full3/weights/best.pt")


model.predict(
    source="/Users/juanpablogarza/Desktop/new_car_images",
    save=True,
    save_txt=True,  
    save_conf=True,   
    conf=0.4,
    device="mps",
)