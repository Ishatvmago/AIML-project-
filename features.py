"""
Date features shared by train.py (training) and app.py (serving).

Keeping them in one module guarantees the model sees exactly the same features at
prediction time as it saw in training (no training/serving skew).

From a flight date we derive:
    month            1-12
    day_of_month     1-31
    day_of_week      1 = Monday ... 7 = Sunday (same convention as the BTS data)
    days_to_holiday  days to the nearest US federal holiday (0 = on the holiday),
                     capped at MAX_DAYS because a holiday far away has no effect
"""

from datetime import date, timedelta

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

MAX_DAYS = 14
_CALENDAR = USFederalHolidayCalendar()


def nearest_holiday(day: date):
    """Returns (holiday name or None, days to it, capped at MAX_DAYS)."""
    window = timedelta(days=MAX_DAYS)
    hols = _CALENDAR.holidays(day - window, day + window, return_name=True)
    if hols.empty:
        return None, MAX_DAYS
    gaps = [abs((ts.date() - day).days) for ts in hols.index]
    i = min(range(len(gaps)), key=gaps.__getitem__)
    return str(hols.iloc[i]), gaps[i]


def date_features(day: date) -> dict:
    """Model features for one flight date (used by the API)."""
    _, days = nearest_holiday(day)
    return {
        "month": day.month,
        "day_of_month": day.day,
        "day_of_week": day.isoweekday(),
        "days_to_holiday": days,
    }


HOLIDAY_WINDOW = 3   # "near a holiday" = within 3 days of it (used for statistics and tips)


def holiday_window_names(df: pd.DataFrame) -> pd.Series:
    """Name of the holiday each flight is within HOLIDAY_WINDOW days of, or None."""
    dates = pd.to_datetime(pd.DataFrame({"year": df["year"], "month": df["month"], "day": df["day_of_month"]}))
    lookup = {}
    for d in dates.unique():
        name, days = nearest_holiday(d.date())
        lookup[d] = name if days <= HOLIDAY_WINDOW else None
    return dates.map(lookup)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds days_to_holiday to the training data (needs year, month, day_of_month columns)."""
    dates = pd.to_datetime(pd.DataFrame({"year": df["year"], "month": df["month"], "day": df["day_of_month"]}))
    lookup = {d: nearest_holiday(d.date())[1] for d in dates.unique()}
    return df.assign(days_to_holiday=dates.map(lookup).astype(int))
