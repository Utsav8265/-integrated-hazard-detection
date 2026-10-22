"""Open-Meteo forecast evaluated around the vehicle's estimated arrival time."""
import requests

URL = "https://api.open-meteo.com/v1/forecast"


def _clean(vals):
    return [v for v in vals if v is not None]


def get_forecast(lat, lon, eta_h=0.0):
    params = {
        "latitude": lat, "longitude": lon, "forecast_days": 3, "timezone": "auto",
        "current": "temperature_2m",
        "hourly": "precipitation_probability,precipitation,visibility,wind_speed_10m",
    }
    r = requests.get(URL, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    hours = j["hourly"]["time"]
    now_key = j["current"]["time"][:13] + ":00"
    i0 = hours.index(now_key) if now_key in hours else 0
    i = min(i0 + int(round(eta_h)), len(hours) - 1)
    lo, hi = max(0, i - 1), min(len(hours), i + 2)  # +/- 1 hour window

    def window(key):
        return _clean(j["hourly"][key][lo:hi])

    prob, prec = window("precipitation_probability"), window("precipitation")
    vis, wind = window("visibility"), window("wind_speed_10m")
    return {
        "forecast_time": hours[i],
        "rain_prob": max(prob) if prob else 0,
        "precip_mm": round(sum(prec), 1) if prec else 0.0,
        "visibility_m": min(vis) if vis else 10000,
        "wind_kmh": max(wind) if wind else 0,
        "temp_c": j["current"].get("temperature_2m"),
    }


DEMO_WEATHER = [  # used only by the offline demo route
    {"forecast_time": "DEMO", "rain_prob": 10, "precip_mm": 0.0, "visibility_m": 10000, "wind_kmh": 8, "temp_c": 28},
    {"forecast_time": "DEMO", "rain_prob": 35, "precip_mm": 0.4, "visibility_m": 7000, "wind_kmh": 12, "temp_c": 27},
    {"forecast_time": "DEMO", "rain_prob": 80, "precip_mm": 4.5, "visibility_m": 1800, "wind_kmh": 25, "temp_c": 24},
    {"forecast_time": "DEMO", "rain_prob": 90, "precip_mm": 7.0, "visibility_m": 900, "wind_kmh": 30, "temp_c": 23},
    {"forecast_time": "DEMO", "rain_prob": 20, "precip_mm": 0.0, "visibility_m": 9000, "wind_kmh": 10, "temp_c": 26},
]
