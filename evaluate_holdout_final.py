"""Part 2 only: evaluate the saved models on the holdout. Do not run in Part 1."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import (
    HOTELS, OUT, RANGE, TARGET, clean, expected_range,
    load_raw, predict_cancel_proba, split,
)


def main():
    dev, holdout = split(load_raw())
    dev = clean(dev)
    holdout = clean(holdout)

    for slug, hotel in HOTELS.items():
        saved = joblib.load(OUT / f"cancellation_model_{slug}.joblib")
        bookings = holdout.loc[holdout["hotel"] == hotel]
        probabilities = predict_cancel_proba(saved, bookings)
        actual = bookings[TARGET]

        # Use the development cancellation rate for the constant baseline.
        baseline_rate = dev.loc[dev["hotel"] == hotel, TARGET].mean()
        baseline = np.full(len(actual), baseline_rate)

        brier = brier_score_loss(actual, probabilities)
        baseline_brier = brier_score_loss(actual, baseline)
        loss = log_loss(actual, probabilities)
        auc = roc_auc_score(actual, probabilities)

        print(
            f"\n# {hotel}: {len(bookings):,} holdout bookings, "
            f"{bookings['arrival_date'].nunique()} days"
        )
        print(
            f"Brier {brier:.4f} (baseline {baseline_brier:.4f})"
            f"  log loss {loss:.4f}  AUC {auc:.3f}"
        )

        rows = []
        for date, group in bookings.groupby("arrival_date"):
            mask = (bookings["arrival_date"] == date).to_numpy()
            daily_probabilities = probabilities[mask]

            expected, low, high = expected_range(
                daily_probabilities, saved["dispersion"]
            )
            rows.append({
                "day": date,
                "actual": group[TARGET].sum(),
                "expected": expected,
                "low": low,
                "high": high,
            })

        days = pd.DataFrame(rows)
        daily_mae = (days["expected"] - days["actual"]).abs().mean()
        coverage = days["actual"].between(days["low"], days["high"]).mean()

        print(
            f"Daily MAE {daily_mae:.2f}; actual inside the "
            f"{RANGE:.0%} range on {coverage:.1%} of days"
        )
        days.round(2).to_csv(OUT / f"holdout_daily_{slug}.csv", index=False)


if __name__ == "__main__":
    main()