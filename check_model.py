"""Run this after adding weights: prints class names so you can verify the model."""
from ultralytics import YOLO
m = YOLO("models/road_damage.pt")
print("Classes:", m.names)
