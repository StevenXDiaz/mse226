import pandas as pd

from common import TARGET, clean, load_raw, split


# Use the same development split and cleaning as training.
def load_development():
    raw, _ = split(load_raw())
    return raw, clean(raw)


# Choose numeric predictors from the cleaned development data.
def prepare_numeric_features(data):
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


# Compare each numeric predictor with cancellation.
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


# Compare each pair once and rank by correlation strength.
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


# Compare booking counts and cancellation rates across groups.
def compare_categories(data):
    groups = [
        "hotel",
        "customer_type",
        "has_company",
        "market_segment",
    ]

    for group in groups:
        summary = data.groupby(group, dropna=False).agg(
            bookings=(TARGET, "size"),
            cancellation_percent=(TARGET, "mean"),
        )

        summary["cancellation_percent"] *= 100

        print(f"\nResults by {group}", flush=True)
        print(summary.round(2).to_string(), flush=True)


# Check missing values before cleaning fills or removes them.
def missing_values(data):
    summary = pd.DataFrame({
        "missing_count": data.isna().sum(),
        "missing_percent": data.isna().mean() * 100,
    })

    summary = summary[summary["missing_count"] > 0]
    summary = summary.sort_values(
        "missing_count", ascending=False
    )

    print("\nMissing values before cleaning", flush=True)
    print(summary.round(4).to_string(), flush=True)


def main():
    print("Loading development data...", flush=True)

    raw, data = load_development()
    missing_values(raw)
    data, predictors = prepare_numeric_features(data)

    print(f"\nDevelopment bookings: {len(data):,}", flush=True)

    outcome_correlations(data, predictors)
    predictor_correlations(data, predictors)
    compare_categories(data)


if __name__ == "__main__":
    main()