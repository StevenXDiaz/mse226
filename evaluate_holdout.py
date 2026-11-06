"""Part 2 only: evaluate the saved pipelines on the untouched holdout. Do not run in Part 1.

Usage: .venv/bin/python evaluate_holdout.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import HOTELS, OUT, RANGE, TARGET, clean, expected_range, load_raw, predict_cancel_proba, split


def main():
    dev, hold = split(load_raw())
    dev, hold = clean(dev), clean(hold)
    for slug, hotel in HOTELS.items():
        saved = joblib.load(OUT / f"cancellation_model_{slug}.joblib")
        h = hold[hold["hotel"] == hotel]
        p = predict_cancel_proba(saved, h)
        y = h[TARGET]
        print(f"\n# {hotel}: {len(h):,} holdout bookings, {h['arrival_date'].nunique()} days")
        print(f"Brier {brier_score_loss(y, p):.4f} (baseline {brier_score_loss(y, np.full(len(y), dev[dev['hotel'] == hotel][TARGET].mean())):.4f})"
              f"  log loss {log_loss(y, p):.4f}  AUC {roc_auc_score(y, p):.3f}")
        rows = [(d, g[TARGET].sum(), *expected_range(p[(h["arrival_date"] == d).to_numpy()], saved["dispersion"]))
                for d, g in h.groupby("arrival_date")]
        days = pd.DataFrame(rows, columns=["day", "actual", "expected", "low", "high"])
        print(f"Daily MAE {(days['expected'] - days['actual']).abs().mean():.2f}; actual inside the "
              f"{RANGE:.0%} range on {days['actual'].between(days['low'], days['high']).mean():.1%} of days")
        days.round(2).to_csv(OUT / f"holdout_daily_{slug}.csv", index=False)


if __name__ == "__main__":
    main()
