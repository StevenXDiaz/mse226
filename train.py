"""Train and compare models using development data only."""

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

from common import (
    FEATURE_SETS, HOTELS, OUT, SEED, TARGET,
    RareAgentGrouper, load_development,
)


N_FOLDS = 5
GRIDS = {
    "Logistic regression": {"C": [0.01, 0.1, 1, 10]},
    "Random forest": {
        "max_depth": [10, 20, None],
        "min_samples_leaf": [5, 20],
    },
    "Gradient boosting": {
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [15, 63],
    },
}


def make_pipeline(algorithm, features, cat_cols, params):
    num_cols = [column for column in features if column not in cat_cols]

    if algorithm == "Logistic regression":
        preprocessing = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=20
            ), cat_cols),
        ])
        model = LogisticRegression(max_iter=5000, **params)
    else:
        preprocessing = ColumnTransformer([
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ), cat_cols),
            ("num", "passthrough", num_cols),
        ])

        if algorithm == "Random forest":
            model = RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                n_jobs=-1,
                random_state=SEED,
                **params,
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=400,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=SEED,
                categorical_features=list(range(len(cat_cols))),
                **params,
            )

    # These steps learn from the training fold each time fit() is called.
    return Pipeline([
        ("agents", RareAgentGrouper()),
        ("pre", preprocessing),
        ("model", model),
    ])


def out_of_fold(pipe, X, y, folds):
    probabilities = np.zeros(len(y))

    for train_idx, valid_idx in folds:
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]

    return probabilities


def scores(y, p, days, folds):
    fold_scores = []

    for _, valid_idx in folds:
        actual = y.iloc[valid_idx]
        predicted = p[valid_idx]

        daily = pd.DataFrame({
            "d": days.iloc[valid_idx].values,
            "y": actual.values,
            "p": predicted,
        }).groupby("d").sum()

        auc = 0.5
        if predicted.std() > 0:
            auc = roc_auc_score(actual, predicted)

        fold_scores.append({
            "Brier": brier_score_loss(actual, predicted),
            "log loss": log_loss(actual, predicted, labels=[0, 1]),
            "AUC": auc,
            "daily MAE": (daily["y"] - daily["p"]).abs().mean(),
        })

    results = pd.DataFrame(fold_scores)
    averages = results.mean().to_dict()
    averages["Brier SE"] = results["Brier"].std() / np.sqrt(len(results))
    return averages


def cross_fit_calibration(oof, y, folds):
    calibrated = np.zeros(len(y))

    for train_idx, valid_idx in folds:
        calibrator = IsotonicRegression(
            y_min=0, y_max=1, out_of_bounds="clip"
        )
        calibrator.fit(oof[train_idx], y.iloc[train_idx])
        calibrated[valid_idx] = calibrator.predict(oof[valid_idx])

    return calibrated


def run_hotel(dev, slug, label):
    X = dev.drop(columns=TARGET)
    y = dev[TARGET]
    days = dev["arrival_date"]

    # Reuse the same date-grouped folds for every comparison.
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    folds = list(splitter.split(X, y, groups=days))

    print(
        f"\n{'#' * 70}\n# {label}: {len(dev):,} development bookings, "
        f"{days.nunique()} days, cancel rate {y.mean():.1%}\n{'#' * 70}"
    )

    baseline = np.zeros(len(y))
    for train_idx, valid_idx in folds:
        baseline[valid_idx] = y.iloc[train_idx].mean()

    baseline_row = {
        "algorithm": "Baseline (training cancel rate)",
        "features": "none",
        "grid size": 0,
        "selected": "-",
    }
    baseline_row.update(scores(y, baseline, days, folds))
    rows = [baseline_row]
    candidates = []

    for algorithm, grid in GRIDS.items():
        for feature_set, features in FEATURE_SETS.items():
            cat_cols = [
                column for column in features
                if not pd.api.types.is_numeric_dtype(X[column])
            ]
            trials = []
            start = time.time()

            for values in itertools.product(*grid.values()):
                params = dict(zip(grid, values))
                pipe = make_pipeline(algorithm, features, cat_cols, params)
                oof = out_of_fold(pipe, X[features], y, folds)

                trials.append({
                    "scores": scores(y, oof, days, folds),
                    "params": params,
                    "oof": oof,
                })

            best = min(trials, key=lambda trial: trial["scores"]["Brier"])
            settings = ", ".join(
                f"{name}={value}" for name, value in best["params"].items()
            )

            row = {
                "algorithm": algorithm,
                "features": feature_set,
                "grid size": len(trials),
                "selected": settings,
            }
            row.update(best["scores"])
            rows.append(row)

            candidates.append({
                "algorithm": algorithm,
                "feature_set": feature_set,
                "features": features,
                "cat_cols": cat_cols,
                "params": best["params"],
                "oof": best["oof"],
                "Brier": best["scores"]["Brier"],
            })

            print(
                f"{algorithm:20s} {feature_set:10s} "
                f"Brier {row['Brier']:.4f}  AUC {row['AUC']:.3f}  "
                f"daily MAE {row['daily MAE']:.2f}  "
                f"[{settings}]  {time.time() - start:.0f}s"
            )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"model_comparison_{slug}.csv", index=False)

    best = min(candidates, key=lambda candidate: candidate["Brier"])
    algorithm = best["algorithm"]
    feature_set = best["feature_set"]
    features = best["features"]
    cat_cols = best["cat_cols"]
    params = best["params"]
    oof = best["oof"]

    calibrated = cross_fit_calibration(oof, y, folds)
    cal_scores = scores(y, calibrated, days, folds)

    print(f"\nSelected: {algorithm}, feature set {feature_set}, {params}")
    print(
        f"Calibrated (cross-fitted): Brier {cal_scores['Brier']:.4f}, "
        f"daily MAE {cal_scores['daily MAE']:.2f}; "
        f"mean predicted {calibrated.mean():.3f} vs actual {y.mean():.3f}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(
        y, oof, n_bins=10, strategy="quantile", name="Raw", ax=ax
    )
    CalibrationDisplay.from_predictions(
        y, calibrated, n_bins=10, strategy="quantile", name="Calibrated", ax=ax
    )
    ax.set_title(f"{label}: out-of-fold calibration, development data")
    fig.tight_layout()
    fig.savefig(OUT / f"calibration_{slug}.png", dpi=150)
    plt.close(fig)

    # Estimate the extra spread in daily cancellation counts.
    daily = pd.DataFrame({
        "d": days.values,
        "y": y.values,
        "p": calibrated,
        "v": calibrated * (1 - calibrated),
    }).groupby("d").sum()

    dispersion = float(
        ((daily["y"] - daily["p"]) ** 2).sum() / daily["v"].sum()
    )
    print(f"Overdispersion (phi): {dispersion:.2f}")

    pipeline = make_pipeline(algorithm, features, cat_cols, params)
    pipeline.fit(X[features], y)

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(oof, y)

    saved = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "dispersion": dispersion,
        "features": features,
        "algorithm": algorithm,
        "feature_set": feature_set,
        "params": params,
    }
    joblib.dump(saved, OUT / f"cancellation_model_{slug}.joblib")
    return table


def main():
    dev = load_development()

    for slug, hotel in HOTELS.items():
        hotel_data = dev.loc[dev["hotel"] == hotel]
        hotel_data = hotel_data.drop(columns="hotel").reset_index(drop=True)

        table = run_hotel(hotel_data, slug, hotel)
        print("\n" + table.round(4).to_string(index=False))

    print(f"\nSaved comparison tables, calibration plots and final pipelines to {OUT}")


if __name__ == "__main__":
    main()