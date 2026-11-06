"""Show cancellation predictions for one hotel and arrival date."""

import argparse

import joblib
import pandas as pd

from common import (
    HOTELS, OUT, RANGE, TARGET, clean, expected_range,
    load_raw, predict_cancel_proba, split,
)


SHOW = [
    "lead_time", "agent", "market_segment", "customer_type",
    "total_guests", "total_nights", "previous_cancellations",
]


def predict_day(day, saved):
    probabilities = predict_cancel_proba(saved, day)
    ranked = day.assign(cancel_prob=probabilities)
    return ranked.sort_values("cancel_prob", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hotel", choices=HOTELS)
    parser.add_argument("date", type=pd.Timestamp, help="arrival date, YYYY-MM-DD")
    parser.add_argument(
        "--top", type=int, default=10,
        help="how many bookings to list (default 10)",
    )
    parser.add_argument(
        "--holdout", action="store_true",
        help="allow a holdout date (Part 2 only)",
    )
    args = parser.parse_args()

    dev, hold = split(load_raw())
    holdout_dates = set(hold["arrival_date"])
    development_dates = set(dev["arrival_date"])

    # Holdout predictions need an explicit flag.
    if args.date in holdout_dates and not args.holdout:
        raise SystemExit(
            f"{args.date.date()} is a holdout date; it stays untouched until Part 2 "
            "(pass --holdout to use it)."
        )

    data = clean(pd.concat([dev, hold]))
    hotel = HOTELS[args.hotel]
    day = data.loc[
        (data["hotel"] == hotel) & (data["arrival_date"] == args.date)
    ]

    if day.empty:
        first_date = data["arrival_date"].min().date()
        last_date = data["arrival_date"].max().date()
        raise SystemExit(
            f"No {hotel} bookings arriving {args.date.date()} "
            f"(data covers {first_date} to {last_date})."
        )

    saved = joblib.load(OUT / f"cancellation_model_{args.hotel}.joblib")
    ranked = predict_day(day, saved)
    probabilities = ranked["cancel_prob"].to_numpy()
    expected, low, high = expected_range(probabilities, saved["dispersion"])

    print(
        f"\n{hotel}, arrivals on {args.date:%A %d %B %Y}: "
        f"{len(ranked)} bookings"
    )
    print(
        f"Expected cancellations: {expected:.1f}  "
        f"({RANGE:.0%} range: {low:.0f} to {high:.0f})"
    )

    count = min(args.top, len(ranked))
    top = ranked.head(args.top)[SHOW + ["cancel_prob"]].copy()
    top.index = range(1, count + 1)
    top.index.name = "rank"

    print(f"\nTop {count} most likely to cancel:")
    print(top.to_string(formatters={"cancel_prob": "{:.0%}".format}))

    print(
        f"\nWhat actually happened: {ranked[TARGET].sum()} "
        f"of {len(ranked)} cancelled."
    )
    if args.date in development_dates:
        print("Note: this is a development date, so the model was trained on these bookings.")

    path = OUT / f"prediction_{args.hotel}_{args.date.date()}.csv"
    columns = ["row_id"] + SHOW + ["cancel_prob", TARGET]
    ranked[columns].to_csv(path, index=False)
    print(f"All {len(ranked)} bookings, ranked, saved to {path}")


if __name__ == "__main__":
    main()