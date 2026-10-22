# Integrated Hazard Detection for Long-Distance Drivers

Video road-damage AI (YOLO) + route weather (Open-Meteo) + area context (OpenStreetMap)
+ explainable risk score + Streamlit dashboard.

## Setup (Windows PowerShell, Python 3.10-3.12 recommended)
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Mac/Linux: `python3 -m venv .venv && source .venv/bin/activate`

## Step 3 - Model weights (REQUIRED for pothole detection)
Standard YOLO (COCO) weights do NOT detect potholes. You need road-damage weights.
- Option A: `python download_model.py` (YOLOv8s trained on RDD2022 from a public Hugging Face Space - verify source/licence)
- Option B: download the .pt manually and save as `models/road_damage.pt`
- Option C: fine-tune yourself: get RDD2022 in YOLO format, then `python train.py rdd2022.yaml`
Then run `python check_model.py` to confirm class names (D40 = pothole).

## Run
```
streamlit run app.py
```
Open http://localhost:8501. Put test videos in `data/sample_videos/`.

## No internet / API failure?
Tab 2 -> "Use offline demo route" (clearly labelled synthetic data).

## Limits (say these to judges)
- Risk score = rule-based heuristic, not a validated accident probability.
- Video has no GPS: you assign it to a route segment manually.
- Traffic is a manual input; severity/depth is not measured.
- Counts use tracking when available; otherwise raw detections may double count.
