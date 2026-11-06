"""Train a model to predict hotel booking cancellations.

Data: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
Usage: .venv/bin/python train.py
"""
from pathlib import Path
from statistics import NormalDist

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (RocCurveDisplay, accuracy_score, brier_score_loss,
                             classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TARGET = "is_canceled"

# Columns that reveal the outcome or are only known after arrival.
LEAKY = ["reservation_status", "reservation_status_date", "assigned_room_type",
         # Likely recorded at check-in or updated after booking (final snapshot values).
         "required_car_parking_spaces", "country", "booking_changes",
         "total_of_special_requests", "adr", "days_in_waiting_list",
         # Non Refund bookings cancel 99% of the time: a data quirk, not a usable signal.
         "deposit_type"]

NO_AGENT, OTHER_AGENTS = "no agent", "other agents"
MIN_AGENT_BOOKINGS = 10  # agents with fewer training bookings are grouped together

RANGE = 0.80  # width of the range given around each day's expected cancellations

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "hotel_bookings.csv")
    # Remove exact duplicate rows (identical on all original columns) before dropping
    # any columns, so bookings that only look alike after dropping are kept.
    df = df.drop_duplicates()
    df = df.drop(columns=LEAKY)
    df["children"] = df["children"].fillna(0)
    # agent is an ID; treat missing as "no agent" and keep as a category.
    # A missing agent means the booking was made without one (agent IDs start at 1).
    df["agent"] = df["agent"].map(lambda a: NO_AGENT if pd.isna(a) else str(int(a)))
    # company IDs are too sparse to use individually (~200 per hotel, most with < 10
    # bookings); whether a company made the booking carries nearly all the signal.
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns="company")
    # Drop bookings with no guests.
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]

    df["arrival_month"] = df["arrival_date_month"].map(MONTHS)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"], month=df["arrival_month"], day=df["arrival_date_day_of_month"]))
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_previous_cancel"] = (df["previous_cancellations"] > 0).astype(int)
    df = df.drop(columns=["arrival_date_month"])
    return df.sort_values("arrival_date").reset_index(drop=True)


def night_key(df: pd.DataFrame) -> pd.Series:
    """Arrival date as a single key, used to keep each night's bookings together."""
    return pd.to_datetime(dict(year=df["arrival_date_year"], month=df["arrival_month"],
                               day=df["arrival_date_day_of_month"]))


def split(df: pd.DataFrame, test_frac: float = 0.2):
    # Random 80/20 split of arrival dates: every booking for a given night lands in either
    # train or test, so test nights are complete and the model never sees part of a test night.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["arrival_date"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    print(f"Train: {len(train):,} rows, {train['arrival_date'].nunique()} nights  "
          f"Test: {len(test):,} rows, {test['arrival_date'].nunique()} nights")
    return train.drop(columns="arrival_date"), test.drop(columns="arrival_date")


def group_rare_agents(train: pd.DataFrame, test: pd.DataFrame):
    """Group agents with < MIN_AGENT_BOOKINGS training bookings into "other agents".

    Counts come from the training set only, so the test set doesn't shape the categories.
    Test agents never seen in training also become "other agents"; "no agent" is kept as is.
    """
    counts = train["agent"].value_counts()
    keep = set(counts[counts >= MIN_AGENT_BOOKINGS].index) | {NO_AGENT}
    train, test = train.copy(), test.copy()
    for d in (train, test):
        d["agent"] = d["agent"].where(d["agent"].isin(keep), OTHER_AGENTS)
    n_kept = len(keep - {NO_AGENT})
    print(f"Agents: kept {n_kept} with >= {MIN_AGENT_BOOKINGS} train bookings, grouped "
          f"{(counts.index != NO_AGENT).sum() - n_kept} into '{OTHER_AGENTS}' "
          f"({(train['agent'] == OTHER_AGENTS).mean():.1%} of train bookings)")
    return train, test


def apply_agent_groups(df: pd.DataFrame, kept_agents: set) -> pd.DataFrame:
    """Map agents to the categories the model was trained on (see group_rare_agents)."""
    return df.assign(agent=df["agent"].where(df["agent"].isin(kept_agents), OTHER_AGENTS))


def expected_range(p: np.ndarray, dispersion: float, width: float = RANGE):
    """Expected number of cancellations among bookings with probabilities p, with a range.

    Expected = sum of probabilities. The range uses a normal approximation whose variance,
    sum p(1-p), is widened by the measured overdispersion (group bookings cancel together).
    """
    expected = float(np.sum(p))
    sd = np.sqrt(dispersion * np.sum(p * (1 - p)))
    z = NormalDist().inv_cdf(0.5 + width / 2)
    return expected, max(0.0, expected - z * sd), expected + z * sd


def predict_cancel_proba(saved: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated cancellation probabilities from a saved model (see fit_and_evaluate)."""
    return saved["calibrator"].predict(saved["model"].predict_proba(X)[:, 1])


def calibration_table(y, proba, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs actual cancel rate within deciles of predicted risk."""
    d = pd.DataFrame({"pred": proba, "actual": np.asarray(y)})
    d["bin"] = pd.qcut(d["pred"], bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(n=("actual", "size"), predicted=("pred", "mean"),
                                               actual=("actual", "mean"))


def fit_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, label: str, slug: str) -> dict:
    """Train both models on one slice of the data, print metrics, and save plots/model."""
    print(f"\n{'#' * 70}\n# {label}: {len(train):,} train / {len(test):,} test, "
          f"cancel rate {train[TARGET].mean():.1%} / {test[TARGET].mean():.1%}\n{'#' * 70}")
    train, test = group_rare_agents(train, test)
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]

    cat_cols = X_train.select_dtypes(["object", "string"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Baseline: logistic regression.
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
    ])
    logreg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    logreg.fit(X_train, y_train)

    # Main model: gradient boosted trees with native categorical handling.
    gbm_pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1, max_categories=250), cat_cols),
        ("num", "passthrough", num_cols),
    ], verbose_feature_names_out=False)
    gbm = Pipeline([("pre", gbm_pre), ("clf", HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=30,
        l2_regularization=1.0, categorical_features=list(range(len(cat_cols))), random_state=42))])
    # Out-of-fold predictions on the training set, with whole nights held out per fold
    # (like the test set). These are honest scores for fitting the calibration below.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(clone(gbm), X_train, y_train, cv=cv, groups=night_key(X_train),
                            method="predict_proba")[:, 1]
    gbm.fit(X_train, y_train)

    # Calibration: the model's raw scores aren't guaranteed to be probabilities. Isotonic
    # regression maps raw score -> observed cancel rate, fitted on the out-of-fold scores.
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, y_train)

    # Overdispersion: how much more nightly cancellation totals swing than the
    # independent-bookings formula expects (group bookings cancel together). Measured on
    # training nights with out-of-fold probabilities; used to widen each day's range.
    p_oof = calibrator.predict(oof)
    nights = pd.DataFrame({"night": night_key(X_train).values, "y": y_train.values, "p": p_oof,
                           "v": p_oof * (1 - p_oof)}).groupby("night").sum()
    dispersion = float(((nights["y"] - nights["p"]) ** 2).sum() / nights["v"].sum())
    print(f"Overdispersion on training nights: real swings are {dispersion:.2f}x the "
          f"independent-bookings variance")

    results, summary = {}, {}
    for name, model in [("Logistic regression", logreg), ("Gradient boosting", gbm)]:
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = proba
        summary[name] = (roc_auc_score(y_test, proba), accuracy_score(y_test, pred), f1_score(y_test, pred))
        print(f"\n=== {name} ===")
        print(f"ROC AUC  {summary[name][0]:.3f}")
        print(f"Accuracy {summary[name][1]:.3f}")
        print(f"F1       {summary[name][2]:.3f}")
        print(classification_report(y_test, pred, target_names=["kept", "canceled"], digits=3))

    # Calibration check on the test set: raw scores vs calibrated probabilities.
    raw = results["Gradient boosting"]
    proba = calibrator.predict(raw)
    for name, p in [("raw scores", raw), ("calibrated", proba)]:
        print(f"Calibration, {name:10s} — test Brier {brier_score_loss(y_test, p):.4f}; "
              f"mean predicted {p.mean():.3f} vs actual {y_test.mean():.3f}")
    print(calibration_table(y_test, proba).round(3).to_string())
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_test, raw, n_bins=10, strategy="quantile", name="Raw scores", ax=ax)
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, strategy="quantile", name="Calibrated", ax=ax)
    ax.set_title(f"{label} — test set calibration (gradient boosting)")
    fig.tight_layout()
    fig.savefig(OUT / f"calibration_{slug}.png", dpi=150)

    # Daily totals: for each test arrival date, expected cancellations (sum of probabilities)
    # and a range, compared with what happened and with a no-model baseline
    # (training cancel rate x number of bookings that day).
    by_day = pd.DataFrame({"day": night_key(X_test).values, "p": proba,
                           "actual": y_test.values}).groupby("day")
    days = by_day.agg(bookings=("p", "size"), actual=("actual", "sum"))
    days[["expected", "low", "high"]] = pd.DataFrame(
        [expected_range(p.to_numpy(), dispersion) for _, p in by_day["p"]], index=days.index)
    days["baseline"] = y_train.mean() * days["bookings"]
    inside = days["actual"].between(days["low"], days["high"])
    print(f"\nDaily cancellations on {len(days)} test days ({days['bookings'].mean():.1f} bookings, "
          f"{days['actual'].mean():.1f} cancellations per day on average):")
    print(f"  Average miss, model:    {(days['expected'] - days['actual']).abs().mean():.2f} cancellations")
    print(f"  Average miss, baseline: {(days['baseline'] - days['actual']).abs().mean():.2f} cancellations")
    print(f"  Actual count inside the {RANGE:.0%} range on {inside.mean():.1%} of days")
    days.round(2).to_csv(OUT / f"daily_cancellations_test_{slug}.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in results.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{label} — test set ROC")
    fig.tight_layout()
    fig.savefig(OUT / f"roc_{slug}.png", dpi=150)

    # Permutation importance: drop in AUC when each feature is shuffled.
    perm = permutation_importance(gbm, X_test, y_test, scoring="roc_auc",
                                  n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("Top features (AUC drop when shuffled):")
    print(imp.head(10).round(4).to_string())
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.head(15)[::-1].plot.barh(ax=ax)
    ax.set_title(f"{label} — top 15 features (permutation importance, AUC drop)")
    fig.tight_layout()
    fig.savefig(OUT / f"feature_importance_{slug}.png", dpi=150)

    joblib.dump({"model": gbm, "calibrator": calibrator, "dispersion": dispersion,
                 "kept_agents": set(train["agent"]) - {OTHER_AGENTS}},
                OUT / f"cancellation_model_{slug}.joblib")
    return {"summary": summary, "importance": imp, "model": gbm}


def main():
    df = load()
    print(f"{len(df):,} bookings, cancel rate {df[TARGET].mean():.1%}")
    train, test = split(df)

    # One model per hotel.
    for hotel in ["City Hotel", "Resort Hotel"]:
        slug = hotel.split()[0].lower()
        fit_and_evaluate(train[train["hotel"] == hotel].drop(columns="hotel"),
                         test[test["hotel"] == hotel].drop(columns="hotel"), hotel, slug)
    print(f"\nSaved models and plots to {OUT}")


if __name__ == "__main__":
    main()
