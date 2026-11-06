"""Section 6: train, tune and compare models with cross-validation on the development data,
then save the final pipeline. The holdout is not used here (it is kept for Part 2).

Per hotel: 3 algorithms x 2 feature sets, each tuned over a small grid, all scored on the
same 5 folds. Folds hold out whole arrival days, like the holdout.

Usage: .venv/bin/python train.py      (~10 minutes)
"""
import itertools
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import (FEATURE_SETS, HOTELS, OUT, SEED, TARGET, RareAgentGrouper, load_development)

N_FOLDS = 5
# Tuning grids, searched exhaustively for every feature set.
GRIDS = {
    "Logistic regression": {"C": [0.01, 0.1, 1, 10]},
    "Random forest": {"max_depth": [10, 20, None], "min_samples_leaf": [5, 20]},
    "Gradient boosting": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 63]},
}


def make_pipeline(algorithm: str, features: list, cat_cols: list, params: dict) -> Pipeline:
    """Preprocessing + model. Every data-dependent step is fitted on the training fold only."""
    num_cols = [c for c in features if c not in cat_cols]
    if algorithm == "Logistic regression":
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat_cols)])
        model = LogisticRegression(max_iter=5000, **params)
    else:
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols)])
        if algorithm == "Random forest":
            model = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1,
                                           random_state=SEED, **params)
        else:
            model = HistGradientBoostingClassifier(
                max_iter=400, min_samples_leaf=30, l2_regularization=1.0, random_state=SEED,
                categorical_features=list(range(len(cat_cols))), **params)
    return Pipeline([("agents", RareAgentGrouper()), ("pre", pre), ("model", model)])


def out_of_fold(pipe: Pipeline, X, y, folds) -> np.ndarray:
    """Predicted probability for every development booking, from the model that left it out."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = pipe.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
    return oof


def scores(y, p, days, folds) -> dict:
    """Metrics averaged over the validation folds (mean and standard error for the Brier score)."""
    per_fold = []
    for _, va in folds:
        yv, pv = y.iloc[va], p[va]
        daily = pd.DataFrame({"d": days.iloc[va].values, "y": yv.values, "p": pv}).groupby("d").sum()
        per_fold.append({"Brier": brier_score_loss(yv, pv), "log loss": log_loss(yv, pv, labels=[0, 1]),
                         "AUC": roc_auc_score(yv, pv) if pv.std() > 0 else 0.5,
                         "daily MAE": (daily["y"] - daily["p"]).abs().mean()})
    f = pd.DataFrame(per_fold)
    return {**f.mean().to_dict(), "Brier SE": f["Brier"].std() / np.sqrt(len(f))}


def cross_fit_calibration(oof, y, folds) -> np.ndarray:
    """Isotonic calibration scored honestly: each fold's map is fitted on the other folds."""
    cal = np.zeros(len(y))
    for tr, va in folds:
        cal[va] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof[tr], y.iloc[tr]).predict(oof[va])
    return cal


def run_hotel(dev: pd.DataFrame, slug: str, label: str) -> pd.DataFrame:
    X, y, days = dev.drop(columns=TARGET), dev[TARGET], dev["arrival_date"]
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y, groups=days))
    print(f"\n{'#' * 70}\n# {label}: {len(dev):,} development bookings, {days.nunique()} days, "
          f"cancel rate {y.mean():.1%}\n{'#' * 70}")

    # Baseline: every booking gets the training fold's cancel rate.
    base = np.zeros(len(y))
    for tr, va in folds:
        base[va] = y.iloc[tr].mean()
    rows = [{"algorithm": "Baseline (training cancel rate)", "features": "none", "grid size": 0,
             "selected": "-", **scores(y, base, days, folds)}]
    best = {}
    for algorithm, grid in GRIDS.items():
        for fs_name, features in FEATURE_SETS.items():
            cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
            trials = []
            t0 = time.time()
            for values in itertools.product(*grid.values()):
                params = dict(zip(grid, values))
                oof = out_of_fold(make_pipeline(algorithm, features, cat_cols, params), X[features], y, folds)
                trials.append((scores(y, oof, days, folds), params, oof))
            s, params, oof = min(trials, key=lambda t: t[0]["Brier"])  # tuned on Brier score
            rows.append({"algorithm": algorithm, "features": fs_name, "grid size": len(trials),
                         "selected": ", ".join(f"{k}={v}" for k, v in params.items()), **s})
            best[(algorithm, fs_name)] = (s["Brier"], params, oof, features, cat_cols)
            print(f"{algorithm:20s} {fs_name:10s} Brier {s['Brier']:.4f}  AUC {s['AUC']:.3f}  "
                  f"daily MAE {s['daily MAE']:.2f}  [{rows[-1]['selected']}]  {time.time() - t0:.0f}s")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"model_comparison_{slug}.csv", index=False)

    # Final choice: lowest cross-validated Brier score.
    (algorithm, fs_name), (_, params, oof, features, cat_cols) = min(best.items(), key=lambda kv: kv[1][0])
    cal = cross_fit_calibration(oof, y, folds)
    cal_scores = scores(y, cal, days, folds)
    print(f"\nSelected: {algorithm}, feature set {fs_name}, {params}")
    print(f"Calibrated (cross-fitted): Brier {cal_scores['Brier']:.4f}, daily MAE {cal_scores['daily MAE']:.2f}; "
          f"mean predicted {cal.mean():.3f} vs actual {y.mean():.3f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y, oof, n_bins=10, strategy="quantile", name="Raw", ax=ax)
    CalibrationDisplay.from_predictions(y, cal, n_bins=10, strategy="quantile", name="Calibrated", ax=ax)
    ax.set_title(f"{label}: out-of-fold calibration, development data")
    fig.tight_layout()
    fig.savefig(OUT / f"calibration_{slug}.png", dpi=150)

    # Overdispersion: how much more daily cancellation totals swing than independent bookings
    # would, measured on the out-of-fold calibrated probabilities; used to widen each day's range.
    d = pd.DataFrame({"d": days.values, "y": y.values, "p": cal, "v": cal * (1 - cal)}).groupby("d").sum()
    dispersion = float(((d["y"] - d["p"]) ** 2).sum() / d["v"].sum())
    print(f"Overdispersion (phi): {dispersion:.2f}")

    # Refit on all development data and save everything needed to predict.
    pipeline = make_pipeline(algorithm, features, cat_cols, params).fit(X[features], y)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, y)
    joblib.dump({"pipeline": pipeline, "calibrator": calibrator, "dispersion": dispersion,
                 "features": features, "algorithm": algorithm, "feature_set": fs_name, "params": params},
                OUT / f"cancellation_model_{slug}.joblib")
    return table


def main():
    dev = load_development()
    for slug, hotel in HOTELS.items():
        table = run_hotel(dev[dev["hotel"] == hotel].drop(columns="hotel").reset_index(drop=True), slug, hotel)
        print("\n" + table.round(4).to_string(index=False))
    print(f"\nSaved comparison tables, calibration plots and final pipelines to {OUT}")


if __name__ == "__main__":
    main()
