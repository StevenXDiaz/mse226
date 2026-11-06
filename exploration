from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
TARGET = "is_canceled"

 # Read the bookings, remove duplicate and zero-guest rows, and keep the 80% development set.
def load_development():
    df = pd.read_csv(ROOT / "hotel_bookings.csv").drop_duplicates()

    # Match the guest filter in the existing code.
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df.loc[guests > 0].copy()

    months = {
        month: number
        for number, month in enumerate(
            [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December",
            ],
            start=1,
        )
    }

    # Keep the same row order before splitting.
    df["arrival_date"] = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": df["arrival_date_month"].map(months),
        "day": df["arrival_date_day_of_month"],
    })
    df = df.sort_values("arrival_date").reset_index(drop=True)

    development, _ = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["hotel"] + "_" + df[TARGET].astype(str),
    )

    return development.copy()

# Create the extra booking features and choose numeric predictors for the correlations.
def prepare_numeric_features(data):
    data["has_company"] = data["company"].notna().astype(int)

    data["total_nights"] = (
        data["stays_in_weekend_nights"]
        + data["stays_in_week_nights"]
    )

    # Leave the total missing if any guest count is unknown.
    data["total_guests"] = data[
        ["adults", "children", "babies"]
    ].sum(axis=1, min_count=3)

    data["has_previous_cancel"] = (
        data["previous_cancellations"] > 0
    ).astype(int)

    predictors = [
        "lead_time",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "has_company",
        "total_nights",
        "total_guests",
        "has_previous_cancel",
    ]

    return data, predictors

# Print how each numeric predictor relates to cancellation, ranked by correlation strength.
def outcome_correlations(data, predictors):
    rows = []

    for predictor in predictors:
        complete = data[[predictor, TARGET]].dropna()

        rows.append({
            "predictor": predictor,
            "correlation": complete[predictor].corr(complete[TARGET]),
        })

    results = pd.DataFrame(rows)
    results["absolute_correlation"] = results["correlation"].abs()
    results = results.sort_values(
        "absolute_correlation", ascending=False
    )

    print("\n1. Correlations with cancellation", flush=True)
    print(results.round(4).to_string(index=False), flush=True)

# Print how each numeric predictor relates to cancellation, ranked by correlation strength.
def outcome_correlations(data, predictors):
    rows = []

    for predictor in predictors:
        complete = data[[predictor, TARGET]].dropna()

        rows.append({
            "predictor": predictor,
            "correlation": complete[predictor].corr(complete[TARGET]),
        })

    results = pd.DataFrame(rows)
    results["absolute_correlation"] = results["correlation"].abs()
    results = results.sort_values(
        "absolute_correlation", ascending=False
    )

    print("\n1. Correlations with cancellation", flush=True)
    print(results.round(4).to_string(index=False), flush=True)

# Print the correlation for each pair of numeric predictors, strongest first.
def predictor_correlations(data, predictors):
    rows = []

    for i, first in enumerate(predictors):
        for second in predictors[i + 1:]:
            complete = data[[first, second]].dropna()

            rows.append({
                "predictor_1": first,
                "predictor_2": second,
                "correlation": complete[first].corr(complete[second]),
            })

    results = pd.DataFrame(rows)
    results["absolute_correlation"] = results["correlation"].abs()
    results = results.sort_values(
        "absolute_correlation", ascending=False
    )

    print("\n2. Correlations between covariates", flush=True)
    print(results.round(4).to_string(index=False), flush=True)

# Print the correlation for each pair of numeric predictors, strongest first.
def compare_categories(data):
    groups = [
        "hotel",
        "customer_type",
        "has_company",
        "market_segment",
        "distribution_channel",
        "agent",
        "meal",
        "reserved_room_type",
    ]

    for group in groups:
        summary = data.groupby(group, dropna=False).agg(
            bookings=(TARGET, "size"),
            cancellation_percent=(TARGET, "mean"),
        )

        summary["cancellation_percent"] *= 100

        print(f"\nResults by {group}", flush=True)
        print(summary.round(2).to_string(), flush=True)

# Print the number and percentage of missing values in each column that has them.
def missing_values(data):
    summary = pd.DataFrame({
        "missing_count": data.isna().sum(),
        "missing_percent": data.isna().mean() * 100,
    })

    summary = summary[summary["missing_count"] > 0]
    summary = summary.sort_values("missing_count", ascending=False)

    print("\nMissing values", flush=True)
    print(summary.round(4).to_string(), flush=True)

# Load the development data, check missing values, and run the comparisons.
def main():
    print("Loading development data...", flush=True)

    data = load_development()
    missing_values(data)
    data, predictors = prepare_numeric_features(data)

    print(f"Development bookings: {len(data):,}", flush=True)

    outcome_correlations(data, predictors)
    predictor_correlations(data, predictors)
    compare_categories(data)


if __name__ == "__main__":
    main()