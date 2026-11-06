"""For one hotel and arrival date: rank that day's bookings by how likely each is to cancel,
and give the expected total number of cancellations (with a range).

Holdout dates are refused unless --holdout is passed, so the holdout stays untouched
until Part 2.

Usage: .venv/bin/python predict.py city 2016-02-10      (run train.py first)
       .venv/bin/python predict.py resort 2017-03-03 --top 20
"""
import argparse

import joblib
import pandas as pd

from common import HOTELS, OUT, RANGE, TARGET, clean, expected_range, load_raw, predict_cancel_proba, split

SHOW = ["lead_time", "agent", "market_segment", "customer_type", "total_guests", "total_nights",
        "previous_cancellations"]


def predict_day(day: pd.DataFrame, saved: dict) -> pd.DataFrame:
    """Bookings for one day, with their cancellation probability, most likely first."""
    return day.assign(cancel_prob=predict_cancel_proba(saved, day)).sort_values("cancel_prob", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hotel", choices=HOTELS)
    ap.add_argument("date", type=pd.Timestamp, help="arrival date, YYYY-MM-DD")
    ap.add_argument("--top", type=int, default=10, help="how many bookings to list (default 10)")
    ap.add_argument("--holdout", action="store_true", help="allow a holdout date (Part 2 only)")
    args = ap.parse_args()

    dev, hold = split(load_raw())
    if args.date in set(hold["arrival_date"]) and not args.holdout:
        raise SystemExit(f"{args.date.date()} is a holdout date; it stays untouched until Part 2 "
                         f"(pass --holdout to use it).")
    df = clean(pd.concat([dev, hold]))
    day = df[(df["hotel"] == HOTELS[args.hotel]) & (df["arrival_date"] == args.date)]
    if day.empty:
        raise SystemExit(f"No {HOTELS[args.hotel]} bookings arriving {args.date.date()} "
                         f"(data covers {df['arrival_date'].min().date()} to {df['arrival_date'].max().date()}).")

    saved = joblib.load(OUT / f"cancellation_model_{args.hotel}.joblib")
    ranked = predict_day(day, saved)
    expected, low, high = expected_range(ranked["cancel_prob"].to_numpy(), saved["dispersion"])

    print(f"\n{HOTELS[args.hotel]}, arrivals on {args.date:%A %d %B %Y}: {len(ranked)} bookings")
    print(f"Expected cancellations: {expected:.1f}  ({RANGE:.0%} range: {low:.0f} to {high:.0f})")
    print(f"\nTop {min(args.top, len(ranked))} most likely to cancel:")
    top = ranked.head(args.top)[SHOW + ["cancel_prob"]].set_axis(range(1, min(args.top, len(ranked)) + 1)).rename_axis("rank")
    print(top.to_string(formatters={"cancel_prob": "{:.0%}".format}))

    print(f"\nWhat actually happened: {ranked[TARGET].sum()} of {len(ranked)} cancelled.")
    if args.date in set(dev["arrival_date"]):
        print("Note: this is a development date, so the model was trained on these bookings.")

    path = OUT / f"prediction_{args.hotel}_{args.date.date()}.csv"
    ranked[["row_id"] + SHOW + ["cancel_prob", TARGET]].to_csv(path, index=False)
    print(f"All {len(ranked)} bookings, ranked, saved to {path}")


if __name__ == "__main__":
    main()
