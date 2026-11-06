"""Simulate per-night overbooking policies on the test set.

For each hotel and arrival date, decide how many rooms to oversell, then score the
decision against what actually happened:
  - oversold N, X bookings cancelled, X >= N: all N resold, X - N rooms left empty
  - oversold N, X bookings cancelled, X <  N: N - X guests walked
An empty room costs 1; a walked guest costs `ratio` (run for several cost ratios).

Usage: .venv/bin/python simulate.py   (run train.py first)
"""
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd

from train import OUT, TARGET, group_rare_agents, load, predict_cancel_proba, split

COST_RATIOS = [3, 10]  # cost of walking a guest, relative to an empty room


def cancel_count_distribution(p: np.ndarray) -> np.ndarray:
    """Exact distribution of the number of cancellations, assuming independent bookings.

    dist[k] = P(exactly k of the bookings cancel) (Poisson-binomial, via dynamic programming).
    """
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pi in p:
        dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
        dist[0] *= 1 - pi
    return dist


def safe_oversell(p: np.ndarray, ratio: float) -> int:
    """Largest N with P(at least N cancellations) >= ratio / (ratio + 1), independent bookings.

    Overselling one more room pays off only when enough bookings are likely to cancel to
    cover it; with walked guests costing 3x an empty room that means at least 75% likely.
    """
    at_least = np.cumsum(cancel_count_distribution(p)[::-1])[::-1]  # at_least[n] = P(X >= n)
    return int(np.nonzero(at_least >= ratio / (ratio + 1))[0].max())


def safe_oversell_adjusted(p: np.ndarray, ratio: float, dispersion: float) -> int:
    """Same rule, with the spread of the night's total widened by the measured overdispersion.

    Uses a normal approximation: mean sum(p), variance dispersion * sum(p * (1 - p)), with a
    continuity correction. P(X >= n) >= f  <=>  n <= mean + 0.5 + sd * z_(1 - f).
    """
    mean = p.sum()
    sd = np.sqrt(dispersion * (p * (1 - p)).sum())
    z = NormalDist().inv_cdf(1 - ratio / (ratio + 1))
    return max(0, int(np.floor(mean + 0.5 + sd * z)))


def score(nights: pd.DataFrame, oversell: str, ratio: float) -> dict:
    n, x = nights[oversell], nights["cancelled"]
    walked = (n - x).clip(lower=0)
    empty = (x - n).clip(lower=0)
    return {"rooms oversold": int(n.sum()), "rooms filled": int(np.minimum(n, x).sum()),
            "guests walked": int(walked.sum()), "nights with a walk": f"{(walked > 0).mean():.1%}",
            "cost": ratio * walked.sum() + empty.sum()}


def main():
    train, test = split(load())
    for hotel in ["City Hotel", "Resort Hotel"]:
        slug = hotel.split()[0].lower()
        _, te = group_rare_agents(train[train["hotel"] == hotel].drop(columns="hotel"),
                                  test[test["hotel"] == hotel].drop(columns="hotel"))
        saved = joblib.load(OUT / f"cancellation_model_{slug}.joblib")
        te = te.assign(p=predict_cancel_proba(saved, te.drop(columns=TARGET)))
        te["arrival_date"] = pd.to_datetime(dict(year=te["arrival_date_year"], month=te["arrival_month"],
                                                 day=te["arrival_date_day_of_month"]))
        nights = te.groupby("arrival_date").agg(bookings=("p", "size"), cancelled=(TARGET, "sum"),
                                                expected=("p", "sum"))
        nights["variance"] = te.groupby("arrival_date")["p"].apply(lambda p: (p * (1 - p)).sum())

        print(f"\n{'#' * 70}\n# {hotel}: {len(nights)} test nights, {nights['bookings'].mean():.1f} "
              f"bookings per night on average\n{'#' * 70}")
        resid = nights["cancelled"] - nights["expected"]
        print(f"Predicted vs actual cancellations per night: mean {nights['expected'].mean():.2f} vs "
              f"{nights['cancelled'].mean():.2f}")
        print(f"Spread vs independent-bookings formula: {(resid ** 2).sum() / nights['variance'].sum():.2f}x "
              f"on test nights; {saved['dispersion']:.2f}x measured on training nights (used below)")

        for ratio in COST_RATIOS:
            f = ratio / (ratio + 1)
            by_night = te.groupby("arrival_date")["p"]
            nights["none"] = 0
            nights["cutoff"] = by_night.apply(lambda p: int((p >= f).sum()))
            nights["expected (rounded)"] = nights["expected"].round().astype(int)
            nights["rule"] = by_night.apply(lambda p: safe_oversell(p.to_numpy(), ratio))
            nights["rule (adjusted)"] = by_night.apply(
                lambda p: safe_oversell_adjusted(p.to_numpy(), ratio, saved["dispersion"]))

            policies = {"No overbooking": "none",
                        f"Per-booking cutoff ({f:.2f})": "cutoff",
                        "Oversell expected (sum, rounded)": "expected (rounded)",
                        f"Per-night {f:.0%} rule": "rule",
                        f"Per-night {f:.0%} rule, adjusted for swings": "rule (adjusted)"}
            table = pd.DataFrame({name: score(nights, col, ratio) for name, col in policies.items()}).T
            table["saving vs no overbooking"] = (1 - table["cost"] / table.loc["No overbooking", "cost"]
                                                 ).map("{:+.1%}".format)
            print(f"\n-- Cost ratio {ratio}:1 (oversell a room only if >= {f:.0%} likely to be covered; "
                  f"target: a walk on <= {1 - f:.0%} of nights)")
            print(table.to_string())
            nights.to_csv(OUT / f"overbooking_nights_{slug}_{ratio}to1.csv")
    print(f"\nPer-night details saved to {OUT}/overbooking_nights_<hotel>_<ratio>to1.csv")


if __name__ == "__main__":
    main()
