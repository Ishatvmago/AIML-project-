"""
API tests for the Flight Delay Predictor.

Run:  pytest -v
The LLM is switched off here so the tests are fast and work in CI without Ollama.
"""

import os
import tempfile

os.environ["ENABLE_LLM"] = "0"
os.environ["PREDICTION_LOG"] = os.path.join(tempfile.mkdtemp(), "predictions.jsonl")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

client = TestClient(app)

VALID_FLIGHT = {
    "flight_date": "2026-07-17",   # a Friday in July
    "dep_hour": 18,
    "distance": 802,
    "airline": "AA",
    "origin": "DFW",
    "dest": "ORD",
}


def predict(**changes):
    return client.post("/predict", json={**VALID_FLIGHT, **changes})


# 1. Health check
def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# 2. UI loads
def test_ui_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# 3. Prediction works and returns a valid probability
def test_predict_returns_probability_reasons_and_tips():
    response = predict()
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["prediction"] in {"Delayed", "On time"}
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert 1 <= len(body["tips"]) <= 3
    assert isinstance(body["reasons"], list)


# 4. Domain logic: delays build up during the day, so an evening flight must be riskier
#    than the same flight early in the morning (seen clearly in the BTS data).
def test_evening_flight_riskier_than_early_morning():
    morning = predict(dep_hour=6).json()["delay_probability"]
    evening = predict(dep_hour=19).json()["delay_probability"]
    assert evening > morning


# 5. Domain logic: July (summer storms, holiday traffic) is the worst month in the 2024 data,
#    October the best, so the same Friday flight must be riskier in July
def test_summer_riskier_than_autumn():
    october = predict(flight_date="2026-10-16").json()["delay_probability"]
    july = predict(flight_date="2026-07-17").json()["delay_probability"]
    assert july > october


# 5b. The API derives the calendar features from the date itself
def test_date_is_turned_into_weekday_and_holiday():
    info = predict(flight_date="2026-11-25").json()["date_info"]
    assert info["weekday"] == "Wednesday"
    assert info["holiday"] == "Thanksgiving Day"
    assert info["days_to_holiday"] == 1


# 6. Boundary checks: the data contract rejects impossible values with HTTP 422
@pytest.mark.parametrize(
    "field,value",
    [
        ("dep_hour", 24),
        ("dep_hour", -1),
        ("flight_date", "2026-13-01"),
        ("flight_date", "2023-12-31"),
        ("flight_date", "not-a-date"),
        ("distance", 10),
        ("distance", 6000),
        ("airline", "ZZ"),
        ("origin", "XXX"),
    ],
)
def test_out_of_range_input_returns_422(field, value):
    assert predict(**{field: value}).status_code == 422


def test_same_origin_and_destination_returns_422():
    assert predict(dest="DFW").status_code == 422


def test_missing_field_returns_422():
    flight = dict(VALID_FLIGHT)
    del flight["airline"]
    assert client.post("/predict", json=flight).status_code == 422


# 7. Lower-case codes are accepted and normalised
def test_lowercase_codes_are_accepted():
    assert predict(airline="aa", origin="dfw", dest="ord").status_code == 200


# 8. GenAI explanation falls back to a template when no LLM is available
def test_explain_falls_back_to_template():
    response = client.post("/explain", json=VALID_FLIGHT)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "template"
    assert "%" in body["explanation"]


# 9. Every prediction is logged for drift monitoring
def test_prediction_is_logged():
    log_path = os.environ["PREDICTION_LOG"]
    before = sum(1 for _ in open(log_path)) if os.path.exists(log_path) else 0
    predict()
    after = sum(1 for _ in open(log_path))
    assert after == before + 1


# 10. Safer alternatives: every suggestion is safer than the original flight, airlines are only
#     those that flew the route in the training data, and invalid input is still rejected
def test_alternatives_are_safer_and_real():
    from app import INSIGHTS

    flight = {**VALID_FLIGHT, "airline": "F9", "origin": "ATL", "dest": "LAX", "distance": 1947, "dep_hour": 19}
    body = client.post("/alternatives", json=flight).json()
    assert body["options"], "a risky evening Frontier flight should have safer options"
    route_airlines = INSIGHTS["route_airlines"]["ATL-LAX"]
    for option in body["options"]:
        assert option["delay_probability"] < body["current_probability"]
        assert option["reduction"] >= 0.03
        if option["kind"] == "airline":
            assert option["flight"]["airline"] in route_airlines


def test_alternatives_rejects_invalid_input():
    assert client.post("/alternatives", json={**VALID_FLIGHT, "dep_hour": 30}).status_code == 422


# 11. Model Lab: every tracked run is exported and exactly one is marked as the served model
def test_experiments_lists_runs_and_the_selected_model():
    body = client.get("/experiments").json()
    assert len(body["runs"]) >= 4
    selected = [r for r in body["runs"] if r["selected"]]
    assert len(selected) == 1
    # the selected (served) model is the best run by ROC-AUC
    assert selected[0]["roc_auc"] >= max(r["roc_auc"] for r in body["runs"]) - 1e-9
    assert body["feature_importance"]


# 12. Flight number lookup: different spellings work, unknown numbers give 404, nonsense gives 422
def test_flight_number_lookup():
    from app import FLIGHT_LOOKUP

    code = next(iter(FLIGHT_LOOKUP))  # any flight number that exists in the 2024 data
    spaced = f"{code[:2].lower()} {code[2:]}"
    body = client.get(f"/flight/{spaced}").json()
    assert body["code"] == code
    leg = body["legs"][0]
    assert leg["origin"] != leg["dest"] and 0 <= leg["dep_hour"] <= 23
    # the leg can be sent straight to /predict
    assert predict(airline=leg["airline"], origin=leg["origin"], dest=leg["dest"],
                   dep_hour=leg["dep_hour"], distance=leg["distance"]).status_code == 200
    assert client.get("/flight/ZZ9999").status_code == 404
    assert client.get("/flight/hello").status_code == 422


# 13. Real delay causes from the BTS data are returned with each prediction
def test_prediction_includes_real_delay_causes():
    causes = predict().json()["delay_causes"]
    assert causes["delayed_flights"] > 0
    assert abs(sum(causes["shares"].values()) - 1) < 0.01
    assert set(causes["shares"]) == {"LateAircraftDelay", "CarrierDelay", "NASDelay", "WeatherDelay", "SecurityDelay"}
