"""Geocoding, routing and surrounding-area context (free public APIs)."""
import math
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def geocode(name):
    r = requests.get(GEOCODE_URL, params={"name": name, "count": 1}, timeout=10)
    r.raise_for_status()
    res = r.json().get("results")
    if not res:
        raise ValueError(f"Place not found: {name}")
    return {"name": res[0]["name"], "lat": res[0]["latitude"], "lon": res[0]["longitude"]}


def get_route(a, b):
    url = OSRM_URL.format(lon1=a["lon"], lat1=a["lat"], lon2=b["lon"], lat2=b["lat"])
    r = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=20)
    r.raise_for_status()
    route = r.json()["routes"][0]
    coords = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
    return {"coords": coords, "distance_km": route["distance"] / 1000,
            "duration_h": route["duration"] / 3600}


def haversine_km(p, q):
    r = 6371.0
    dlat, dlon = math.radians(q[0] - p[0]), math.radians(q[1] - p[1])
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(p[0])) * math.cos(math.radians(q[0])) * math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def sample_points(coords, n, duration_h):
    """n evenly spaced points along the route, each with distance and ETA."""
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + haversine_km(coords[i - 1], coords[i]))
    total = cum[-1] or 1.0
    pts = []
    for k in range(n):
        target = total * (k + 0.5) / n  # centre of each segment
        j = min(range(len(cum)), key=lambda i: abs(cum[i] - target))
        pts.append({"segment": k + 1, "lat": coords[j][0], "lon": coords[j][1],
                    "km": round(cum[j], 1), "eta_h": round(duration_h * cum[j] / total, 2)})
    return pts


def demo_route(n=5):
    """Offline fallback: approximate Delhi -> Jaipur straight line. DEMO ONLY."""
    a, b = (28.6139, 77.2090), (26.9124, 75.7873)
    coords = [(a[0] + (b[0] - a[0]) * t / 50, a[1] + (b[1] - a[1]) * t / 50) for t in range(51)]
    return {"coords": coords, "distance_km": 270.0, "duration_h": 5.0}


def area_context(lat, lon):
    """Forest / populated / open road from OpenStreetMap tags near the point."""
    q = f"""[out:json][timeout:12];
    (way(around:2000,{lat},{lon})["landuse"~"forest|residential|commercial"];
     way(around:2000,{lat},{lon})["natural"="wood"];
     node(around:3000,{lat},{lon})["place"~"city|town|village"];);
    out tags 60;"""
    try:
        r = requests.post(OVERPASS_URL, data={"data": q}, timeout=15)
        r.raise_for_status()
        forest = built = 0
        for el in r.json().get("elements", []):
            t = el.get("tags", {})
            if t.get("landuse") == "forest" or t.get("natural") == "wood":
                forest += 1
            elif t.get("landuse") in ("residential", "commercial") or "place" in t:
                built += 1
        if forest == 0 and built == 0:
            return "Highway / open road"
        return "Forest / low-service area" if forest > built else "Dense populated area"
    except Exception:
        return "Unknown"
