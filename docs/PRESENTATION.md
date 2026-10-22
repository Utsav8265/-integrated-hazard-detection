# 3-minute demo script
0:00 Problem: long-distance drivers face potholes, cracks, rain/fog, traffic; existing tools detect only one.
0:30 Tab 1: upload video -> boxes, confidence, counts, FPS, CSV.
1:15 Tab 2: Delhi -> Jaipur, weather forecast at each segment's ETA, forest/populated context.
2:00 Tab 3: set traffic, assign video to a segment, show High-risk segment + alerts + map.
2:30 Research gap: fuses road damage + weather + traffic + geography (not isolated pothole detection).
2:45 Future: GPS-tagged hazards, severity/depth, night/rain robustness, XGBoost vs rule baseline.

# Honest evaluation to quote
Report precision, recall, mAP50, ms/frame on a held-out RDD2022 split (`yolo val`). Compare clear vs rain/night clips.

# Likely judge questions
- "Is the risk score accurate?" -> Transparent heuristic baseline; next step is calibrating with labelled crash data.
- "Why not train from scratch?" -> One-week scope; pretrained RDD2022 model. Fine-tuning is Option C in README.
