"""For one hotel and arrival date: rank that day's bookings by how likely each is to cancel,
and give the expected total number of cancellations (with a range).

Usage: .venv/bin/python predict.py city 2016-08-12      (run train.py first)
       .venv/bin/python predict.py resort 2017-03-03 --top 20
"""
import argparse

import joblib
import pandas as pd

from train import (OUT, RANGE, TARGET, apply_agent_groups, expected_range, load,
                   predict_cancel_proba, split)

HOTELS = {"city": "City Hotel", "resort": "Resort Hotel"}
SHOW = ["lead_time", "agent", "market_segment", "customer_type", "total_guests", "total_nights",
        "previous_cancellations"]


def predict_day(df: pd.DataFrame, saved: dict) -> pd.DataFrame:
    """Bookings for one day, with their cancellation probability, most likely first."""
    X = apply_agent_groups(df.drop(columns=[TARGET, "hotel", "arrival_date"]), saved["kept_agents"])
    return df.assign(cancel_prob=predict_cancel_proba(saved, X)).sort_values("cancel_prob", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hotel", choices=HOTELS)
    ap.add_argument("date", type=pd.Timestamp, help="arrival date, YYYY-MM-DD")
    ap.add_argument("--top", type=int, default=10, help="how many bookings to list (default 10)")
    args = ap.parse_args()

    df = load()
    day = df[(df["hotel"] == HOTELS[args.hotel]) & (df["arrival_date"] == args.date)]
    if day.empty:
        first, last = df["arrival_date"].min().date(), df["arrival_date"].max().date()
        raise SystemExit(f"No {HOTELS[args.hotel]} bookings arriving {args.date.date()} "
                         f"(data covers {first} to {last}).")
    _, test = split(df)
    in_test = args.date in set(df.loc[test.index, "arrival_date"])

    saved = joblib.load(OUT / f"cancellation_model_{args.hotel}.joblib")
    ranked = predict_day(day, saved)
    expected, low, high = expected_range(ranked["cancel_prob"].to_numpy(), saved["dispersion"])

    print(f"\n{HOTELS[args.hotel]}, arrivals on {args.date:%A %d %B %Y}: {len(ranked)} bookings")
    print(f"Expected cancellations: {expected:.1f}  ({RANGE:.0%} range: {low:.0f} to {high:.0f})")
    print(f"\nTop {min(args.top, len(ranked))} most likely to cancel:")
    top = ranked.head(args.top)[SHOW + ["cancel_prob"]].rename_axis("booking")
    print(top.to_string(formatters={"cancel_prob": "{:.0%}".format}))

    # The dataset is historical, so we can show what actually happened.
    print(f"\nWhat actually happened: {ranked[TARGET].sum()} of {len(ranked)} cancelled.")
    if not in_test:
        print("Note: this date was in the training set, so the model has seen these bookings. "
              "Use a test-set date for an honest check.")

    path = OUT / f"prediction_{args.hotel}_{args.date.date()}.csv"
    ranked[SHOW + ["cancel_prob", TARGET]].rename_axis("booking").to_csv(path)
    print(f"All {len(ranked)} bookings, ranked, saved to {path}")


if __name__ == "__main__":
    main()
