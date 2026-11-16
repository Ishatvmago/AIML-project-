"""
Builds data/flights.csv from the public US Bureau of Transportation Statistics
(BTS) "Reporting Carrier On-Time Performance" data.

Source: https://www.transtats.bts.gov/  (public US government data, no account needed)

Steps:
1. Download the monthly zip files listed in MONTHS.
2. Keep only scheduled flights (filtered month by month) that were not cancelled or diverted.
3. Keep flights between the 20 busiest airports, so the API can validate airport codes.
4. Take a random sample of SAMPLE_ROWS flights (seed 42), so the CSV stays small enough to commit.

Usage:
    python download_data.py
"""

import io
import os
import zipfile

import pandas as pd
import requests

MONTHS = [(2024, m) for m in range(1, 13)]   # all 12 months of 2024
SAMPLE_ROWS = 120_000                         # about 10,000 flights per month
TOP_AIRPORTS = 20
URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
OUT_PATH = os.path.join("data", "flights.csv")
RAW_DIR = os.path.join("data", "raw")        # downloaded zips are cached here (not committed)

COLUMNS = {
    "Year": "year",
    "Month": "month",
    "DayofMonth": "day_of_month",
    "DayOfWeek": "day_of_week",            # 1 = Monday ... 7 = Sunday
    "Reporting_Airline": "airline",
    "Origin": "origin",
    "Dest": "dest",
    "CRSDepTime": "crs_dep_time",          # scheduled departure, hhmm
    "Distance": "distance",
    "ArrDelay": "arr_delay_min",
    "ArrDel15": "delayed",                 # 1 if arrival delay >= 15 minutes
    "Cancelled": "cancelled",
    "Diverted": "diverted",
}


def download_month(year, month):
    url = URL.format(year=year, month=month)
    cached = os.path.join(RAW_DIR, f"{year}_{month:02d}.zip")
    if os.path.exists(cached):
        print(f"Using cached {cached}")
        with open(cached, "rb") as f:
            content = f.read()
    else:
        print(f"Downloading {url}")
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        content = resp.content
        os.makedirs(RAW_DIR, exist_ok=True)
        with open(cached, "wb") as f:
            f.write(content)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, usecols=list(COLUMNS), low_memory=False)
    # filter each month straight away so 12 months fit comfortably in memory
    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    df = df.dropna(subset=["ArrDelay", "ArrDel15"])
    print(f"  {year}-{month:02d}: {len(df):,} completed flights")
    return df


def main():
    raw = pd.concat([download_month(y, m) for y, m in MONTHS], ignore_index=True)
    df = raw.rename(columns=COLUMNS)
    print(f"Completed flights (not cancelled or diverted): {len(df):,}")

    busiest = df["origin"].value_counts().head(TOP_AIRPORTS).index
    df = df[df["origin"].isin(busiest) & df["dest"].isin(busiest)]

    df["dep_hour"] = (df["crs_dep_time"] // 100).clip(0, 23).astype(int)
    df["delayed"] = df["delayed"].astype(int)
    df["arr_delay_min"] = df["arr_delay_min"].astype(int)
    df["distance"] = df["distance"].astype(int)

    df = df.sample(n=min(SAMPLE_ROWS, len(df)), random_state=42)
    df = df[["year", "month", "day_of_month", "day_of_week", "dep_hour", "airline", "origin", "dest",
             "distance", "arr_delay_min", "delayed"]]

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print("Delay rate:", round(df["delayed"].mean(), 3))


if __name__ == "__main__":
    main()
