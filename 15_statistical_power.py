"""
15_statistical_power.py

WHAT THIS DOES
    For every paired comparison in the study it reports, next to the McNemar
    p-value that was already published:
      * the discordant pairs (b, c) -- how many questions actually changed answer
      * the difference in execution accuracy with a 95% confidence interval
      * the smallest difference that COULD have reached significance
      * the difference the design would detect 80% of the time (MDE)

WHY IT EXISTS
    A p-value above 0.05 does not say "there is no effect", it says "this study
    did not detect one". Without knowing how large an effect the study was
    capable of detecting, a null result is uninterpretable. Adding the interval
    and the detection threshold converts every null in this work from
    "we found no difference" into the much stronger, falsifiable claim
    "we can rule out a difference larger than X percentage points".

    No experiment is re-run and no number changes; this is an additional
    statistical description of results already obtained.

METHOD
    All comparisons are paired: the same 1034 dev questions under two
    conditions. Only the discordant pairs carry information -- b questions the
    first condition got right and the second wrong, c the reverse. McNemar's
    exact test asks whether b and c come from a fair coin.
      * confidence interval: Wald interval for paired proportions,
        se = sqrt((b + c) - (b - c)^2 / n) / n
      * minimum significant difference: the smallest |b - c| that an exact
        two-sided binomial test on b + c pairs would call significant
      * MDE (80% power): the true difference at which that test becomes
        significant 80% of the time, found by search over the binomial

INPUT   results/master_results.csv   (produced by 14_build_master_results.py)
OUTPUT  printed table + results/statistical_power.csv

HOW TO RUN
    python 15_statistical_power.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, norm

IN_FILE = os.path.join("results", "master_results.csv")
OUT_FILE = os.path.join("results", "statistical_power.csv")
ALPHA = 0.05

# (label A, column A, label B, column B) -- positive difference means B is better
COMPARISONS = [
    ("Baseline",    "baseline_ok",  "Memory A (raw)",        "memA_ok"),
    ("Baseline",    "baseline_ok",  "Memory B (standard.)",  "memB_ok"),
    ("Baseline",    "baseline_ok",  "Memory B (compact)",    "memB_compact_ok"),
    ("Memory A",    "memA_ok",      "Memory B (compact)",    "memB_compact_ok"),
    ("Full schema", "baseline_ok",  "Retrieved schema",      "retrieved_ok"),
    ("Full schema", "baseline_ok",  "Perfect schema",        "perfect_ok"),
    ("Retrieved",   "retrieved_ok", "Perfect schema",        "perfect_ok"),
]


def to_bool(series):
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def min_significant_difference(m, alpha=ALPHA):
    """Smallest |b - c| that an exact two-sided binomial test on m discordant
    pairs would declare significant. Below this the study cannot show anything,
    no matter what the truth is."""
    if m == 0:
        return None
    for d in range(0, m + 1):
        if (m + d) % 2:
            continue
        b = (m + d) // 2
        if binomtest(min(b, m - b), m, 0.5).pvalue < alpha:
            return d
    return None


def mde_80(m, alpha=ALPHA):
    """Difference (in questions) detected with 80% probability, given m
    discordant pairs."""
    d = min_significant_difference(m, alpha)
    if d is None:
        return None
    for pi in np.arange(0.5, 1.0001, 0.0005):
        lo, hi = (m - d) / 2.0, (m + d) / 2.0
        power = binom.sf(np.floor(hi) - 1e-9, m, pi) + binom.cdf(np.ceil(lo) - 1 + 1e-9, m, pi)
        if power >= 0.80:
            return (2 * pi - 1) * m
    return None


def main():
    df = pd.read_csv(IN_FILE)
    flag_cols = [c for c in df.columns if c.endswith("_ok")]
    for c in flag_cols:
        df[c] = to_bool(df[c])
    n = len(df)

    print(f"Questions: {n}\n")
    print("Execution accuracy per condition")
    for c in flag_cols:
        k = int(df[c].sum())
        print(f"  {c:<20}{k:>5}/{n}  ({100 * k / n:.1f}%)")

    z = norm.ppf(1 - ALPHA / 2)
    out = []
    print("\nPaired comparisons (positive difference = second condition better)")
    header = (f"{'comparison':<40}{'b':>5}{'c':>5}{'b+c':>6}"
              f"{'diff pp':>9}{'95% CI':>20}{'p':>9}{'min sig':>9}{'MDE80':>8}")
    print(header)
    print("-" * len(header))

    for a_name, a_col, b_name, b_col in COMPARISONS:
        a, b_ = df[a_col], df[b_col]
        b = int((a & ~b_).sum())      # A right, B wrong
        c = int((~a & b_).sum())      # B right, A wrong
        m = b + c
        diff = (c - b) / n
        se = np.sqrt(m - (c - b) ** 2 / n) / n if m else float("nan")
        lo, hi = 100 * (diff - z * se), 100 * (diff + z * se)
        p = binomtest(min(b, c), m, 0.5).pvalue if m else float("nan")
        d_min = min_significant_difference(m)
        mde = mde_80(m)
        mde_pp = 100 * mde / n if mde is not None else float("nan")
        label = f"{a_name} -> {b_name}"
        print(f"{label:<40}{b:>5}{c:>5}{m:>6}{100 * diff:>+9.2f}"
              f"   [{lo:>+6.2f}, {hi:>+6.2f}]{p:>9.4f}"
              f"{(d_min if d_min is not None else -1):>9}{mde_pp:>8.2f}")
        out.append({
            "comparison": label, "b": b, "c": c, "discordant": m,
            "diff_pp": round(100 * diff, 2),
            "ci_low_pp": round(lo, 2), "ci_high_pp": round(hi, 2),
            "mcnemar_p": round(p, 4),
            "min_significant_questions": d_min,
            "min_significant_pp": round(100 * d_min / n, 2) if d_min is not None else None,
            "mde80_pp": round(mde_pp, 2),
        })

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(out).to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}")

    ceiling = next(r for r in out if r["comparison"].endswith("Perfect schema")
                   and r["comparison"].startswith("Full"))
    print("\nReading guide")
    print(f"  The largest effect any retrieval-side intervention could have here is "
          f"{ceiling['diff_pp']:+.2f} pp (Full -> Perfect schema).")
    print("  A comparison whose MDE is close to that value can only detect an "
          "intervention that captures nearly all of the available headroom.")


if __name__ == "__main__":
    main()
