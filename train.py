"""Optional: fine-tune YOLO11n on RDD2022 (needs the dataset in YOLO format + a data.yaml).
Usage: python train.py path/to/rdd2022.yaml"""
import sys
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data=sys.argv[1], epochs=50, imgsz=640, batch=16, project="outputs", name="rdd2022_yolo11n")
print("Best weights: outputs/rdd2022_yolo11n/weights/best.pt -> copy to models/road_damage.pt")
