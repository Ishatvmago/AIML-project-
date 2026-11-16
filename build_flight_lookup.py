"""
Builds two files from the raw BTS monthly files that download_data.py caches in data/raw/:

model/flight_lookup.json: flight number -> the route(s) and usual departure hour it flew in 2024.
model/delay_causes.json:  what caused the delays (minutes by official BTS cause) per route,
                          per departure airport and overall.

The web app uses it so a customer can type a flight number (e.g. "DL 1234") instead of
choosing the airline, airports and time by hand. It is a 2024 schedule: airlines can change
a flight number's route or time, so the app asks the customer to check their ticket.

Usage:
    python download_data.py          # once, fills data/raw/ with the 12 monthly zips
    python build_flight_lookup.py
"""

import glob
import io
import json
import os
import zipfile

import pandas as pd

RAW_GLOB = os.path.join("data", "raw", "2024_*.zip")
META_PATH = os.path.join("model", "metadata.json")
OUT_PATH = os.path.join("model", "flight_lookup.json")
CAUSES_PATH = os.path.join("model", "delay_causes.json")
CAUSES = ["LateAircraftDelay", "CarrierDelay", "NASDelay", "WeatherDelay", "SecurityDelay"]
MIN_DELAYED = 30  # a route needs this many delayed flights for its own cause breakdown
MIN_FLIGHTS = 4  # a leg must have flown at least this often in 2024 to count as scheduled

COLUMNS = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime", "Distance",
           "ArrDel15", "Cancelled", "Diverted"] + CAUSES


def read_month(path):
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), usecols=COLUMNS, low_memory=False)


def main():
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise SystemExit("No raw BTS files in data/raw/. Run python download_data.py first.")
    with open(META_PATH) as f:
        meta = json.load(f)
    airports = set(meta["allowed"]["origin"])
    airlines = set(meta["allowed"]["airline"])

    parts = []
    for path in files:
        df = read_month(path)
        df = df[df["Origin"].isin(airports) & df["Dest"].isin(airports) & df["Reporting_Airline"].isin(airlines)]
        parts.append(df)
        print(f"{os.path.basename(path)}: {len(df):,} flights between the 20 airports")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["Flight_Number_Reporting_Airline", "CRSDepTime"])
    df["code"] = df["Reporting_Airline"] + df["Flight_Number_Reporting_Airline"].astype(int).astype(str)
    df["dep_hour"] = (df["CRSDepTime"] // 100).clip(0, 23).astype(int)

    lookup = {}
    for (code, origin, dest), g in df.groupby(["code", "Origin", "Dest"]):
        if len(g) < MIN_FLIGHTS:
            continue
        lookup.setdefault(code, []).append({
            "airline": g["Reporting_Airline"].iat[0],
            "origin": origin,
            "dest": dest,
            "dep_hour": int(g["dep_hour"].mode().iat[0]),  # the usual scheduled hour
            "distance": int(g["Distance"].median()),
            "flights_2024": int(len(g)),
        })
    for legs in lookup.values():
        legs.sort(key=lambda leg: leg["flights_2024"], reverse=True)

    with open(OUT_PATH, "w") as f:
        json.dump(lookup, f, separators=(",", ":"))
    print(f"Saved {OUT_PATH}: {len(lookup):,} flight numbers, "
          f"{sum(len(v) for v in lookup.values()):,} routes, {os.path.getsize(OUT_PATH) / 1e6:.1f} MB")
    write_delay_causes(df)


def shares(group):
    """Share of delay minutes per official BTS cause, plus how many delayed flights it rests on."""
    minutes = group[CAUSES].sum()
    total = minutes.sum()
    return {"delayed_flights": int(len(group)),
            "shares": {c: round(float(minutes[c] / total), 3) if total else 0.0 for c in CAUSES}}


def write_delay_causes(df):
    delayed = df[(df["ArrDel15"] == 1) & (df["Cancelled"] == 0) & (df["Diverted"] == 0)].fillna({c: 0 for c in CAUSES})
    routes = {f"{o}-{d}": shares(g) for (o, d), g in delayed.groupby(["Origin", "Dest"]) if len(g) >= MIN_DELAYED}
    origins = {o: shares(g) for o, g in delayed.groupby("Origin")}
    with open(CAUSES_PATH, "w") as f:
        json.dump({"overall": shares(delayed), "by_route": routes, "by_origin": origins}, f, separators=(",", ":"))
    print(f"Saved {CAUSES_PATH}: {len(delayed):,} delayed flights, {len(routes)} routes with their own breakdown")
    print("overall:", shares(delayed)["shares"])


if __name__ == "__main__":
    main()
