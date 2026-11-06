"""Section 5: data exploration on the development data only (the holdout is not loaded).

Usage: .venv/bin/python explore.py   -> tables printed, figure saved to outputs/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HOTELS, LEAKY, OUT, TARGET, clean, load_raw, split


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categorical columns, 0 (none) to 1 (one determines the other)."""
    table = pd.crosstab(a, b).to_numpy()
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / n / (min(table.shape) - 1)))


def main():
    raw_dev, _ = split(load_raw())  # development rows only, all original columns
    dev = clean(raw_dev)
    print(f"Development data: {len(raw_dev):,} unique rows, {len(dev):,} after cleaning\n")

    # (e) Missing values, on the raw development rows. Agent and company are coded "NULL".
    missing = pd.Series({c: raw_dev[c].isna().mean() for c in raw_dev.columns})
    missing["agent"] = raw_dev["agent"].isna().mean()
    print("Missing values:\n", missing[missing > 0].round(4).to_string(), "\n")

    num = ["lead_time", "previous_cancellations", "previous_bookings_not_canceled", "is_repeated_guest",
           "stays_in_weekend_nights", "stays_in_week_nights", "total_nights", "adults", "children",
           "babies", "total_guests", "has_company", "has_previous_cancel"]
    cat = ["market_segment", "distribution_channel", "customer_type", "meal", "reserved_room_type", "agent"]

    # (b) Numeric covariates vs outcome: correlation with the 0/1 outcome (point-biserial).
    corr = pd.DataFrame({h: dev[dev["hotel"] == name][num].corrwith(dev[dev["hotel"] == name][TARGET])
                         for h, name in HOTELS.items()})
    corr["both"] = dev[num].corrwith(dev[TARGET])
    print("Correlation with is_canceled:\n", corr.sort_values("both", key=abs, ascending=False).round(3).to_string(), "\n")
    # Leaky columns removed from the models, for context.
    leak = raw_dev.assign(nonrefund=(raw_dev["deposit_type"] == "Non Refund").astype(int))
    print("Removed columns, correlation with is_canceled:",
          {c: round(leak[c].corr(leak[TARGET]), 3) for c in ["required_car_parking_spaces",
           "total_of_special_requests", "booking_changes", "adr", "days_in_waiting_list", "nonrefund"]}, "\n")

    # (b) Categorical covariates: cancel rate by category, and spread across categories.
    for c in cat:
        g = dev.groupby(["hotel", c])[TARGET].agg(["size", "mean"]).reset_index()
        g = g[g["size"] >= 100]
        print(f"Cancel rate by {c} (categories with >= 100 bookings):")
        print(g.pivot(index=c, columns="hotel", values="mean").round(3).to_string(), "\n")
    print("Association with is_canceled (Cramér's V):",
          {c: round(cramers_v(dev[c], dev[TARGET]), 3) for c in cat}, "\n")

    bins = [-1, 7, 30, 90, 180, 365, 1000]
    labels = ["0-7", "8-30", "31-90", "91-180", "181-365", "366+"]
    dev["lead_bin"] = pd.cut(dev["lead_time"], bins, labels=labels)
    lead = dev.groupby(["lead_bin", "hotel"], observed=True)[TARGET].mean().unstack()
    print("Cancel rate by lead time (days):\n", lead.round(3).to_string(), "\n")

    # (c) Covariates vs each other: numeric (Spearman) and categorical (Cramér's V) pairs.
    s = dev[num].corr(method="spearman").where(lambda m: np.triu(np.ones(m.shape, bool), 1)).stack()
    print("Strongest numeric pairs (Spearman):\n", s[s.abs().sort_values(ascending=False).index][:10].round(3).to_string(), "\n")
    pairs = {f"{a} x {b}": cramers_v(dev[a], dev[b]) for i, a in enumerate(cat + ["hotel"])
             for b in (cat + ["hotel"])[i + 1:]}
    print("Categorical pairs (Cramér's V):\n", pd.Series(pairs).sort_values(ascending=False).round(3).to_string(), "\n")
    print("Market segment x distribution channel (share of segment):\n",
          pd.crosstab(dev["market_segment"], dev["distribution_channel"], normalize="index").round(2).to_string(), "\n")
    print("Lead time by customer type (median):", dev.groupby("customer_type")["lead_time"].median().to_dict())
    print("Lead time by market segment (median):", dev.groupby("market_segment")["lead_time"].median().to_dict(), "\n")

    # (d) Subgroups: City vs Resort, and repeat vs new guests.
    cols = [TARGET, "lead_time", "total_nights", "stays_in_weekend_nights", "total_guests", "children",
            "is_repeated_guest", "has_company"]
    sub = dev.groupby("hotel")[cols].mean().T
    sub.loc["no agent"] = dev.groupby("hotel")["agent"].apply(lambda a: (a == "no agent").mean())
    sub.loc["Online TA share"] = dev.groupby("hotel")["market_segment"].apply(lambda m: (m == "Online TA").mean())
    sub.loc["Transient share"] = dev.groupby("hotel")["customer_type"].apply(lambda m: (m == "Transient").mean())
    sub.loc["bookings"] = dev.groupby("hotel").size()
    sub.loc["days"] = raw_dev.groupby("hotel")["arrival_date"].nunique()
    print("City vs Resort (means):\n", sub.round(3).to_string(), "\n")
    print("Repeat vs new guests:\n", dev.groupby("is_repeated_guest")[[TARGET, "lead_time", "previous_cancellations"]]
          .agg(["mean"]).round(3).to_string(), dev["is_repeated_guest"].value_counts().to_dict(), "\n")

    # Figure: four panels on one page.
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    top = corr.reindex(corr["both"].abs().sort_values().index)[["city", "resort"]]
    top.rename(columns={"city": "City", "resort": "Resort"}).plot.barh(ax=ax[0, 0])
    ax[0, 0].axvline(0, color="k", lw=0.8)
    ax[0, 0].set_title("(a) Correlation of numeric covariates with cancellation")
    lead.rename(columns=lambda h: h.split()[0]).plot(marker="o", ax=ax[0, 1])
    ax[0, 1].set(title="(b) Cancel rate by lead time", xlabel="Lead time (days)", ylabel="Cancel rate")
    for a_, col, title in [(ax[1, 0], "market_segment", "(c) Cancel rate by market segment"),
                           (ax[1, 1], "customer_type", "(d) Cancel rate by customer type")]:
        g = dev.groupby([col, "hotel"])[TARGET].agg(["size", "mean"])
        g = g[g["size"] >= 100]["mean"].unstack().rename(columns=lambda h: h.split()[0])
        g.sort_values("City").plot.barh(ax=a_)
        a_.set(title=title, xlabel="Cancel rate", ylabel="")
    fig.suptitle("Development data (80%): cancellation vs covariates")
    fig.tight_layout()
    fig.savefig(OUT / "exploration.png", dpi=150)
    print(f"Saved {OUT / 'exploration.png'}")


if __name__ == "__main__":
    main()
