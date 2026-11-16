"""
Flight Delay Predictor API.

The model in model/model.joblib is trained by train.py on real US BTS flight data.
Nothing here is random or hard-coded: every number comes from the model or from
statistics saved in model/metadata.json at training time.

Routes:
    GET  /                 React UI (built into static/)
    GET  /health           health check
    GET  /options          allowed airlines and airports (fills the UI dropdowns)
    GET  /stats            model metrics + live prediction count (dashboard)
    GET  /historical-data  delay patterns from the training data (charts)
    GET  /airport-network  airports and busiest routes from the training data (network map)
    POST /predict          delay probability + top reasons + rule-based tips
    POST /explain          plain-English explanation written by a local LLM (Ollama),
                           falling back to a template when no LLM is available
"""

import json
import os
from datetime import date, datetime, timezone

import joblib
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from features import HOLIDAY_WINDOW, MAX_DAYS, date_features, nearest_holiday

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_PATH = os.environ.get("PREDICTION_LOG", os.path.join(BASE_DIR, "logs", "predictions.jsonl"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
ENABLE_LLM = os.environ.get("ENABLE_LLM", "1") == "1"

AIRLINE_NAMES = {
    "AA": "American Airlines", "AS": "Alaska Airlines", "B6": "JetBlue", "DL": "Delta Air Lines",
    "F9": "Frontier Airlines", "G4": "Allegiant Air", "HA": "Hawaiian Airlines", "MQ": "Envoy Air",
    "NK": "Spirit Airlines", "OH": "PSA Airlines", "OO": "SkyWest Airlines", "UA": "United Airlines",
    "WN": "Southwest Airlines", "YX": "Republic Airways", "9E": "Endeavor Air",
}
DAY_NAMES = ["", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
FEATURE_LABELS = {
    "month": "month of travel", "day_of_week": "day of the week", "dep_hour": "departure time",
    "day_of_month": "day of the month", "days_to_holiday": "closeness to a public holiday",
    "distance": "flight distance", "airline": "airline", "origin": "departure airport",
    "dest": "arrival airport",
}

# Loaded once at startup, not per request.
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    META = json.load(f)
FEATURES = META["features"]
INSIGHTS = META["insights"]


class FlightInput(BaseModel):
    """Data contract: every rule here is listed in the README table. Violations return HTTP 422."""
    flight_date: date = Field(..., description="Date of the flight, YYYY-MM-DD (2024-01-01 to 2030-12-31)")
    dep_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour, 24h clock")
    distance: int = Field(..., ge=50, le=5000, description="Route distance in miles")
    airline: str = Field(..., min_length=2, max_length=2, description="IATA carrier code, e.g. DL")
    origin: str = Field(..., min_length=3, max_length=3, description="IATA airport code, e.g. ATL")
    dest: str = Field(..., min_length=3, max_length=3, description="IATA airport code, e.g. LAX")

    @field_validator("flight_date")
    @classmethod
    def date_in_range(cls, v):
        if not date(2024, 1, 1) <= v <= date(2030, 12, 31):
            raise ValueError("flight_date must be between 2024-01-01 and 2030-12-31")
        return v

    @field_validator("airline", "origin", "dest", mode="before")
    @classmethod
    def upper(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("airline")
    @classmethod
    def known_airline(cls, v):
        if v not in META["allowed"]["airline"]:
            raise ValueError(f"unknown airline code; allowed: {META['allowed']['airline']}")
        return v

    @field_validator("origin", "dest")
    @classmethod
    def known_airport(cls, v):
        if v not in META["allowed"]["origin"]:
            raise ValueError(f"unknown airport code; allowed: {META['allowed']['origin']}")
        return v

    @model_validator(mode="after")
    def different_airports(self):
        if self.origin == self.dest:
            raise ValueError("origin and dest must be different airports")
        return self


app = FastAPI(title="Flight Delay Predictor", version="2.1.0")


# ---------- model helpers ----------

def build_row(flight: FlightInput) -> dict:
    """Turns the request into the model's features (same date logic as training, see features.py)."""
    data = flight.model_dump()
    day = data.pop("flight_date")
    return {"flight_date": day.isoformat(), **data, **date_features(day)}


def date_info(row):
    day = date.fromisoformat(row["flight_date"])
    holiday, days = nearest_holiday(day)
    return {"weekday": DAY_NAMES[row["day_of_week"]], "month": MONTH_NAMES[row["month"]],
            "holiday": holiday if days < MAX_DAYS else None, "days_to_holiday": days}

def predict_proba(rows):
    return model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]


def top_reasons(flight, prob):
    """
    Per-prediction explanation: replace one feature at a time with its most typical
    training value and measure how much the delay probability changes.
    A positive impact means this feature pushes the flight towards a delay.
    """
    variants = []
    for feat in FEATURES:
        row = dict(flight)
        row[feat] = META["typical"][feat]
        variants.append(row)
    baseline_probs = predict_proba(variants)
    reasons = []
    for feat, base in zip(FEATURES, baseline_probs):
        impact = float(prob - base)
        if abs(impact) >= 0.005:
            reasons.append({"feature": feat, "label": FEATURE_LABELS[feat],
                            "value": flight[feat], "impact": round(impact, 3)})
    reasons.sort(key=lambda r: abs(r["impact"]), reverse=True)
    return reasons[:4]


def tips_for(flight, prob):
    """Rule-based advice built from the delay rates seen in the training data."""
    rates = META["delay_rate_by"]
    overall = META["delay_rate"]
    tips = []
    holiday, days = nearest_holiday(date.fromisoformat(flight["flight_date"]))
    holiday_rate = rates.get("holiday", {}).get(holiday)
    if days <= HOLIDAY_WINDOW and holiday_rate is not None:
        when = "on" if days == 0 else f"{days} day{'s' if days > 1 else ''} from"
        normal = rates["not_near_holiday"]
        advice = ("Allow extra time at the airport." if holiday_rate > normal + 0.02 else
                  "So travel around this holiday was actually smoother than usual."
                  if holiday_rate < normal - 0.02 else "That is close to a normal day.")
        tips.append(f"Your flight is {when} {holiday}. In 2024, flights within {HOLIDAY_WINDOW} days of "
                    f"{holiday} were delayed {holiday_rate:.0%} of the time (normal days: {normal:.0%}). "
                    + advice)
    if flight["dep_hour"] >= 17:
        morning = min(rates["dep_hour"].get(str(h), 1) for h in range(5, 9))
        tips.append(f"Evening flights pick up delays from earlier in the day. Morning departures "
                    f"(before 9 am) were delayed only {morning:.0%} of the time.")
    airline_rate = rates["airline"].get(flight["airline"], overall)
    if airline_rate > overall + 0.03:
        best = min(rates["airline"], key=rates["airline"].get)
        tips.append(f"{AIRLINE_NAMES.get(flight['airline'], flight['airline'])} was delayed "
                    f"{airline_rate:.0%} of the time; {AIRLINE_NAMES.get(best, best)} had the best "
                    f"record ({rates['airline'][best]:.0%}).")
    origin_rate = rates["origin"].get(flight["origin"], overall)
    if origin_rate > overall + 0.03:
        tips.append(f"{flight['origin']} is a delay-prone airport ({origin_rate:.0%} of departures "
                    f"delayed). Reach the airport early and track the inbound aircraft.")
    level = risk_level(prob)
    if level == "High":
        tips.append("High delay risk: leave at least 3 hours for any connecting flight.")
    elif level == "Medium":
        tips.append("Medium delay risk: leave at least 2 hours for any connecting flight.")
    if not tips:
        tips.append("Low delay risk. Standard check-in timing should be fine.")
    return tips[:3]


def risk_level(prob):
    return "High" if prob >= 0.6 else "Medium" if prob >= 0.4 else "Low"


def log_prediction(flight, prob):
    """Every input is logged so drift_check.py can compare live traffic with training data."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat(), **flight, "probability": round(prob, 4)}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def count_predictions():
    if not os.path.exists(LOG_PATH):
        return 0
    with open(LOG_PATH) as f:
        return sum(1 for _ in f)


# ---------- generative AI explanation ----------

def template_explanation(flight, prob, reasons):
    parts = [f"This flight has a {prob:.0%} chance of arriving 15+ minutes late ({risk_level(prob).lower()} risk)."]
    up = [r["label"] for r in reasons if r["impact"] > 0]
    down = [r["label"] for r in reasons if r["impact"] < 0]
    if up:
        parts.append("The main factors raising the risk are the " + ", ".join(up) + ".")
    if down:
        parts.append("The " + ", ".join(down) + " lower the risk.")
    return " ".join(parts)


def llm_explanation(flight, prob, reasons, tips):
    """Asks the local Llama model (via Ollama, runs offline, no API key) to explain the prediction."""
    facts = {
        "airline": AIRLINE_NAMES.get(flight["airline"], flight["airline"]),
        "route": f"{flight['origin']} to {flight['dest']}",
        "date": f"{DAY_NAMES[flight['day_of_week']]} {flight['flight_date']}",
        "nearest_public_holiday": date_info(flight)["holiday"] or "none within two weeks",
        "departure_hour": flight["dep_hour"],
        "delay_probability": f"{prob:.0%}",
        "factors_that_RAISE_delay_risk": [f"{r['label']} (+{r['impact']:.0%})" for r in reasons if r["impact"] > 0],
        "factors_that_LOWER_delay_risk": [f"{r['label']} (-{abs(r['impact']):.0%})" for r in reasons if r["impact"] < 0],
        "tips": tips,
    }
    prompt = ("You are a helpful travel assistant. A machine learning model produced the facts below. "
              "Speak directly to the passenger ('your flight'). In exactly 3 short sentences: state the "
              "delay chance, name the factors that raise it (and any that lower it, only if that list is not "
              "empty), and repeat one tip from the tips list. "
              "Use ONLY these facts and do not invent numbers. Reply with the 3 sentences only, "
              "no introduction and no quotes.\n" + json.dumps(facts))
    resp = requests.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                               "options": {"temperature": 0.3, "num_predict": 150}},
                         timeout=(1, 60))
    resp.raise_for_status()
    return resp.json()["response"].strip().strip('"').strip()


def llm_available():
    if not ENABLE_LLM:
        return False
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=0.5).ok
    except requests.RequestException:
        return False


# ---------- routes ----------

@app.get("/health")
def health():
    return {"status": "ok", "model": META["algorithm"]}


@app.get("/options")
def options():
    airline_stats = INSIGHTS["by_airline"]
    return {
        "airlines": [{"code": c, "name": AIRLINE_NAMES.get(c, c), "delay_rate": airline_stats[c]["rate"]}
                     for c in META["allowed"]["airline"]],
        "airports": META["allowed"]["origin"],
        "route_distance": INSIGHTS["route_distance"],
    }


@app.get("/stats")
def stats():
    return {
        "algorithm": META["algorithm"],
        "params": META.get("params", {}),
        "metrics": META["metrics"],
        "training_rows": INSIGHTS["rows"],
        "delay_rate": META["delay_rate"],
        "avg_delay": INSIGHTS["avg_delay"],
        "delay_buckets": INSIGHTS["delay_buckets"],
        "by_hour": INSIGHTS["by_hour"],
        "predictions_served": count_predictions(),
        "llm": {"available": llm_available(), "model": OLLAMA_MODEL},
    }


@app.get("/historical-data")
def historical_data():
    return {key: INSIGHTS[key] for key in ("by_hour", "by_day", "by_month", "by_airline", "by_origin")} | {
        "airline_names": {c: AIRLINE_NAMES.get(c, c) for c in INSIGHTS["by_airline"]},
        "delay_rate": META["delay_rate"],
        "rows": INSIGHTS["rows"],
        "by_days_to_holiday": INSIGHTS.get("by_days_to_holiday", {}),
    }


@app.get("/airport-network")
def airport_network():
    return {"airports": INSIGHTS["by_origin"], "connections": INSIGHTS["top_routes"]}


@app.post("/predict")
def predict(flight: FlightInput):
    data = build_row(flight)
    prob = float(predict_proba([data])[0])
    reasons = top_reasons(data, prob)
    log_prediction(data, prob)
    return {
        "delay_probability": round(prob, 4),
        "prediction": "Delayed" if prob >= 0.5 else "On time",
        "risk_level": risk_level(prob),
        "reasons": reasons,
        "tips": tips_for(data, prob),
        "date_info": date_info(data),
    }


@app.post("/explain")
def explain(flight: FlightInput):
    data = build_row(flight)
    prob = float(predict_proba([data])[0])
    reasons = top_reasons(data, prob)
    tips = tips_for(data, prob)
    if ENABLE_LLM:
        try:
            return {"explanation": llm_explanation(data, prob, reasons, tips), "source": "llm"}
        except (requests.RequestException, KeyError, ValueError):
            pass
    return {"explanation": template_explanation(data, prob, reasons), "source": "template"}


@app.get("/")
def index():
    page = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(page):
        return JSONResponse({"detail": "UI not built yet: run `npm run build` in frontend/"}, status_code=404)
    # no-cache: the browser always checks for a new index.html, so a rebuilt UI shows up at once
    # (the JS/CSS files it points to have content hashes in their names and can be cached)
    return FileResponse(page, headers={"Cache-Control": "no-cache"})


# Serves the built React files (JS, CSS, images). Mounted last so the API routes above win.
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
