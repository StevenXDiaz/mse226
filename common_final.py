"""Shared data and model helpers.

Data: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
Antonio, Almeida and Nunes, Hotel booking demand datasets (2019).
"""

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = "is_canceled"
SEED = 42
HOTELS = {"city": "City Hotel", "resort": "Resort Hotel"}

# Keep the existing exclusions so the model inputs stay the same.
LEAKY = [
    "reservation_status", "reservation_status_date", "assigned_room_type",
    "required_car_parking_spaces", "country", "booking_changes",
    "total_of_special_requests", "adr", "days_in_waiting_list", "deposit_type",
]

# Dates group bookings for evaluation; they are not model inputs.
TIME = [
    "arrival_date_year", "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month",
]

NO_AGENT = "no agent"
OTHER_AGENTS = "other agents"
MIN_AGENT_BOOKINGS = 10
RANGE = 0.80

BASIC = [
    "lead_time", "market_segment", "distribution_channel", "customer_type",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "meal", "reserved_room_type",
]

FULL = BASIC + [
    "agent", "has_company", "total_nights", "total_guests", "has_previous_cancel",
]

FEATURE_SETS = {"A (basic)": BASIC, "B (full)": FULL}

MONTHS = {
    month: number
    for number, month in enumerate([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ], start=1)
}


def load_raw():
    path = ROOT / "data" / "hotel_bookings.csv"
    if not path.exists():
        path = ROOT / "hotel_bookings.csv"

    df = pd.read_csv(path)

    # Check duplicates before adding columns, and keep the original row numbers.
    df = df.drop_duplicates().rename_axis("row_id").reset_index()

    df["arrival_date"] = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": df["arrival_date_month"].map(MONTHS),
        "day": df["arrival_date_day_of_month"],
    })

    return df


def split(df, test_frac=0.2):
    # Keep all bookings for an arrival date on the same side.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_frac,
        random_state=SEED,
    )
    dev_idx, hold_idx = next(
        splitter.split(df, groups=df["arrival_date"])
    )

    assignments = pd.DataFrame({
        "row_id": df["row_id"].to_numpy(),
        "set": "development",
    })
    assignments.loc[hold_idx, "set"] = "holdout"
    assignments.to_csv(OUT / "split.csv", index=False)

    return df.iloc[dev_idx], df.iloc[hold_idx]


def clean(df):
    df = df.drop(columns=LEAKY + TIME)

    df["children"] = df["children"].fillna(0)
    df["agent"] = df["agent"].map(
        lambda agent: NO_AGENT if pd.isna(agent) else str(int(agent))
    )

    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns="company")

    guests = df["adults"] + df["children"] + df["babies"]
    df = df.loc[guests > 0].copy()

    df["total_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_previous_cancel"] = (
        df["previous_cancellations"] > 0
    ).astype(int)

    return df.reset_index(drop=True)


def load_development():
    dev, _ = split(load_raw())
    return clean(dev)


# Each training fold learns its own agent groups through this transformer.
class RareAgentGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=MIN_AGENT_BOOKINGS):
        self.min_count = min_count

    def fit(self, X, y=None):
        if "agent" in X:
            counts = X["agent"].value_counts()
            self.kept_ = set(counts[counts >= self.min_count].index)
            self.kept_.add(NO_AGENT)

        return self

    def transform(self, X):
        if "agent" not in X:
            return X

        result = X.copy()
        result["agent"] = result["agent"].where(
            result["agent"].isin(self.kept_),
            OTHER_AGENTS,
        )
        return result


def predict_cancel_proba(saved, X):
    features = X[saved["features"]]
    probabilities = saved["pipeline"].predict_proba(features)[:, 1]
    return saved["calibrator"].predict(probabilities)


def expected_range(p, dispersion, width=RANGE):
    expected = float(np.sum(p))

    # Allow extra variation for bookings that cancel together.
    sd = np.sqrt(dispersion * np.sum(p * (1 - p)))
    z = NormalDist().inv_cdf(0.5 + width / 2)

    low = max(0.0, expected - z * sd)
    high = expected + z * sd

    return expected, low, high