"""Road-damage detection on video using an Ultralytics YOLO checkpoint."""
import subprocess
import time
from pathlib import Path

import cv2
import pandas as pd


def load_model(path):
    from ultralytics import YOLO
    return YOLO(str(path))


def is_pothole(name: str) -> bool:
    # RDD2022 convention: D40 = pothole. Some models use the plain word.
    return str(name).strip().lower() in ("d40", "pothole", "potholes")


def _to_browser_mp4(raw: Path, out: Path):
    """mp4v is not playable in browsers; re-encode to H.264 if ffmpeg exists."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-i", str(raw), "-vcodec", "libx264",
               "-pix_fmt", "yuv420p", "-an", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        raw.unlink(missing_ok=True)
    except Exception:
        raw.replace(out)


def analyze_video(model, video_path, out_path, conf=0.25, frame_stride=2,
                  imgsz=640, progress_cb=None, max_frames=None):
    video_path, out_path = Path(video_path), Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    raw = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(fps / frame_stride, 1), (w, h))

    rows, snapshots, infer_ms = [], [], []
    tracking = True
    idx = processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride:
            idx += 1
            continue
        if max_frames and processed >= max_frames:
            break

        t0 = time.perf_counter()
        res = None
        if tracking:
            try:
                res = model.track(frame, persist=True, conf=conf, imgsz=imgsz,
                                  verbose=False)[0]
            except Exception:
                tracking = False  # tracker deps missing -> plain detection
        if res is None:
            res = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        infer_ms.append((time.perf_counter() - t0) * 1000)

        boxes = res.boxes
        best = 0.0
        if boxes is not None and len(boxes):
            cls = boxes.cls.cpu().numpy().astype(int)
            cf = boxes.conf.cpu().numpy()
            xy = boxes.xyxy.cpu().numpy()
            ids = (boxes.id.cpu().numpy().astype(int)
                   if getattr(boxes, "id", None) is not None else [None] * len(cls))
            for c, p, b, tid in zip(cls, cf, xy, ids):
                rows.append({
                    "frame": idx, "time_s": round(idx / fps, 2),
                    "class": model.names[int(c)], "confidence": round(float(p), 3),
                    "track_id": tid, "x1": int(b[0]), "y1": int(b[1]),
                    "x2": int(b[2]), "y2": int(b[3]),
                })
                best = max(best, float(p))

        annotated = res.plot()
        writer.write(annotated)
        if best > 0:
            snapshots.append((best, annotated[:, :, ::-1].copy()))  # RGB
            snapshots = sorted(snapshots, key=lambda s: -s[0])[:6]

        processed += 1
        idx += 1
        if progress_cb:
            progress_cb(idx / total)

    cap.release()
    writer.release()
    _to_browser_mp4(raw, out_path)

    df = pd.DataFrame(rows, columns=["frame", "time_s", "class", "confidence",
                                     "track_id", "x1", "y1", "x2", "y2"])
    pothole_mask = df["class"].apply(is_pothole) if len(df) else pd.Series([], dtype=bool)

    def unique_or_total(sub):
        if tracking and len(sub) and sub["track_id"].notna().any():
            return int(sub["track_id"].nunique())
        return int(len(sub))

    summary = {
        "frames_processed": processed,
        "total_detections": int(len(df)),
        "potholes": unique_or_total(df[pothole_mask]) if len(df) else 0,
        "cracks": unique_or_total(df[~pothole_mask]) if len(df) else 0,
        "count_method": "unique tracked objects" if tracking else
                        "raw detections (may double count)",
        "avg_inference_ms": round(sum(infer_ms) / max(len(infer_ms), 1), 1),
        "pipeline_fps": round(1000 / (sum(infer_ms) / max(len(infer_ms), 1)), 1)
                        if infer_ms else 0,
    }
    return {"detections": df, "summary": summary, "video": out_path,
            "snapshots": [s[1] for s in snapshots]}
