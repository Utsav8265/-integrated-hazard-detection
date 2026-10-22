"""Transparent rule-based risk score (0-100). Demo heuristic, NOT a validated
accident-probability model. Weights should be calibrated with real data."""

TRAFFIC = {"Low": 0, "Medium": 8, "High": 15}
AREA = {"Highway / open road": 0, "Forest / low-service area": 10,
        "Dense populated area": 8, "Unknown": 3}


def band(score):
    return "Low" if score < 35 else "Medium" if score < 65 else "High"


def score_segment(potholes=0, cracks=0, rain_prob=0, precip_mm=0.0,
                  visibility_m=10000, traffic="Low", area="Unknown"):
    parts, warnings = {}, []

    parts["Road damage"] = min(35, 8 * potholes + 3 * cracks)
    if potholes:
        warnings.append(f"{potholes} pothole(s) detected on this segment")
    if cracks:
        warnings.append(f"{cracks} crack(s) detected on this segment")

    parts["Rain"] = round(min(25, rain_prob / 100 * 15 + min(10, precip_mm * 2)), 1)
    if rain_prob >= 60 or precip_mm >= 2:
        warnings.append(f"Heavy rain likely near arrival ({rain_prob:.0f}% chance, {precip_mm} mm)")

    parts["Visibility"] = 15 if visibility_m < 1000 else 8 if visibility_m < 3000 else 3 if visibility_m < 5000 else 0
    if visibility_m < 3000:
        warnings.append(f"Low visibility expected ({visibility_m/1000:.1f} km)")

    parts["Traffic"] = TRAFFIC.get(traffic, 0)
    if traffic == "High":
        warnings.append("High traffic expected")

    parts["Area context"] = AREA.get(area, 3)
    if area == "Forest / low-service area":
        warnings.append("Forest stretch: limited services, possible animal crossing")
    elif area == "Dense populated area":
        warnings.append("Populated area: pedestrians and slow traffic likely")

    score = round(min(100, sum(parts.values())), 1)
    return {"score": score, "band": band(score), "parts": parts, "warnings": warnings}
