"""Shared data loading, the holdout split, preprocessing and feature sets.

Data: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
(Antonio, Almeida & Nunes, "Hotel booking demand datasets", Data in Brief 22, 2019).
Place hotel_bookings.csv in data/.
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

# Columns that reveal the outcome or are only known after arrival.
LEAKY = ["reservation_status", "reservation_status_date", "assigned_room_type",
         # Likely recorded at check-in or updated after booking (final snapshot values).
         "required_car_parking_spaces", "country", "booking_changes",
         "total_of_special_requests", "adr", "days_in_waiting_list",
         # Non Refund bookings cancel 95% of the time: a data quirk, not a usable signal.
         "deposit_type"]
# Calendar time is not used as a feature (project rule); arrival_date is kept only to group
# each day's bookings together in the split and the cross-validation folds.
TIME = ["arrival_date_year", "arrival_date_month", "arrival_date_week_number",
        "arrival_date_day_of_month"]

NO_AGENT, OTHER_AGENTS = "no agent", "other agents"
MIN_AGENT_BOOKINGS = 10  # agents with fewer training bookings are grouped together
RANGE = 0.80  # width of the range given around each day's expected cancellations

# Feature set A: the booking as entered, without the travel-agent ID or engineered features.
BASIC = ["lead_time", "market_segment", "distribution_channel", "customer_type",
         "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
         "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
         "meal", "reserved_room_type"]
# Feature set B: A plus who made the booking (agent, company) and engineered features.
FULL = BASIC + ["agent", "has_company", "total_nights", "total_guests", "has_previous_cancel"]
FEATURE_SETS = {"A (basic)": BASIC, "B (full)": FULL}

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}


def load_raw() -> pd.DataFrame:
    """All original columns, exact duplicates removed, with the CSV row number as row_id."""
    df = pd.read_csv(ROOT / "data" / "hotel_bookings.csv")
    # Remove exact duplicate rows (identical on all original columns) before dropping
    # any columns, so bookings that only look alike after dropping are kept.
    df = df.drop_duplicates().rename_axis("row_id").reset_index()
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"], month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"]))
    return df


def split(df: pd.DataFrame, test_frac: float = 0.2):
    """Random 80/20 split of arrival dates into development and holdout data.

    Every booking for a given day lands on the same side, so holdout days are complete and
    no model sees part of a holdout day. The assignment is saved to outputs/split.csv.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=SEED)
    dev_idx, hold_idx = next(splitter.split(df, groups=df["arrival_date"]))
    pd.DataFrame({"row_id": df["row_id"], "set": "development"}).assign(
        set=lambda d: d["set"].where(~d.index.isin(hold_idx), "holdout")
    ).to_csv(OUT / "split.csv", index=False)
    return df.iloc[dev_idx], df.iloc[hold_idx]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row cleaning and feature construction; nothing here is learned from the data."""
    df = df.drop(columns=LEAKY + TIME)
    df["children"] = df["children"].fillna(0)
    # A missing agent means the booking was made without one (agent IDs start at 1).
    df["agent"] = df["agent"].map(lambda a: NO_AGENT if pd.isna(a) else str(int(a)))
    # Company IDs are too sparse to use individually; whether a company booked carries the signal.
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns="company")
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]  # bookings with no guests
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_previous_cancel"] = (df["previous_cancellations"] > 0).astype(int)
    return df.reset_index(drop=True)


def load_development() -> pd.DataFrame:
    """Cleaned development data (80%). The holdout is never returned."""
    dev, _ = split(load_raw())
    return clean(dev)


class RareAgentGrouper(BaseEstimator, TransformerMixin):
    """Group agents with fewer than min_count training bookings into "other agents".

    Fitted inside the model pipeline, so the counts come from the training portion of each
    split only. Agents never seen in training also become "other agents".
    """

    def __init__(self, min_count: int = MIN_AGENT_BOOKINGS):
        self.min_count = min_count

    def fit(self, X, y=None):
        if "agent" in X:
            counts = X["agent"].value_counts()
            self.kept_ = set(counts[counts >= self.min_count].index) | {NO_AGENT}
        return self

    def transform(self, X):
        if "agent" not in X:
            return X
        return X.assign(agent=X["agent"].where(X["agent"].isin(self.kept_), OTHER_AGENTS))


def predict_cancel_proba(saved: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated cancellation probabilities from a saved pipeline (see train.py)."""
    return saved["calibrator"].predict(saved["pipeline"].predict_proba(X[saved["features"]])[:, 1])


def expected_range(p: np.ndarray, dispersion: float, width: float = RANGE):
    """Expected number of cancellations among bookings with probabilities p, with a range.

    Expected = sum of probabilities. The range uses a normal approximation whose variance,
    sum p(1-p), is widened by the measured overdispersion (bookings that cancel together).
    """
    expected = float(np.sum(p))
    sd = np.sqrt(dispersion * np.sum(p * (1 - p)))
    z = NormalDist().inv_cdf(0.5 + width / 2)
    return expected, max(0.0, expected - z * sd), expected + z * sd
