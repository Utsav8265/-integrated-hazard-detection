"""Download road-damage weights into models/road_damage.pt.
The URL comes from a public Hugging Face Space (YOLOv8s trained on RDD2022).
Verify the source and licence yourself; if it fails, download manually (README Step 3)."""
import sys
from pathlib import Path
import requests

URL = ("https://huggingface.co/spaces/ProjectRoadDamageDetection/"
       "Automated-Road-Damage-Detection/resolve/main/yolov8s_best.pt?download=true")
out = Path(__file__).parent / "models" / "road_damage.pt"
out.parent.mkdir(exist_ok=True)
try:
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print("Saved", out)
except Exception as e:
    print("Download failed:", e)
    print("Open the Space's Files tab, download yolov8s_best.pt manually, save as models/road_damage.pt")
    sys.exit(1)
