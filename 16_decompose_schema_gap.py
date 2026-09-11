"""
16_decompose_schema_gap.py

WHAT THIS DOES
    Explains WHY the retrieved schema (743/1034) falls short of the perfect
    oracle schema (778/1034). It splits the 35-question gap into the two
    possible causes and measures each one.

WHY IT EXISTS
    "Retrieved is significantly worse than Perfect (p = 0.0006)" says that a
    gap exists but not what causes it. There are exactly two ways a schema
    linker can hurt:

      (a) UNDER-SELECTION -- it discarded a table or column the gold query
          needed, so the correct answer became impossible to express.
      (b) OVER-SELECTION  -- it kept everything the query needed, but also a
          lot of irrelevant material that distracts the model.

    The two have opposite remedies. If (a) dominates, the linker should be made
    more permissive. If (b) dominates, it should be made stricter. Without this
    decomposition any recommendation is guesswork.

METHOD
    14_build_master_results.py already stored, per question, the oracle schema
    (extracted from the gold SQL) and the retrieved schema, plus the flag
    sl_recall_complete = "every table and column the gold query needs is
    present in the retrieved schema".

    Questions are split on that flag and the Retrieved-vs-Perfect comparison is
    repeated inside each group. Because the split is defined without reference
    to any accuracy outcome, the two groups are independent of the result being
    measured.

INPUT   results/master_results.csv   (produced by 14_build_master_results.py)
OUTPUT  printed report + results/schema_gap_decomposition.csv

HOW TO RUN
    python 16_decompose_schema_gap.py
"""

import os

import pandas as pd
from scipy.stats import binomtest

IN_FILE = os.path.join("results", "master_results.csv")
OUT_FILE = os.path.join("results", "schema_gap_decomposition.csv")


def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def paired(sub, col_a, col_b):
    """b = A right/B wrong, c = A wrong/B right, and the exact McNemar p."""
    a, b_ = sub[col_a], sub[col_b]
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    m = b + c
    p = binomtest(min(b, c), m, 0.5).pvalue if m else float("nan")
    return b, c, m, p


def main():
    df = pd.read_csv(IN_FILE)
    for c in [x for x in df.columns if x.endswith("_ok")] + ["sl_recall_complete"]:
        df[c] = to_bool(df[c])
    n = len(df)

    full = int(df.baseline_ok.sum())
    retr = int(df.retrieved_ok.sum())
    perf = int(df.perfect_ok.sum())

    print("STEP 1 -- the gap we are explaining")
    print(f"  Full schema      {full}/{n}  ({100 * full / n:.1f}%)")
    print(f"  Retrieved schema {retr}/{n}  ({100 * retr / n:.1f}%)")
    print(f"  Perfect schema   {perf}/{n}  ({100 * perf / n:.1f}%)")
    b, c, m, p = paired(df, "retrieved_ok", "perfect_ok")
    print(f"  Perfect wins on {c} questions, Retrieved wins on {b} -> net gap "
          f"{c - b} questions (McNemar p = {p:.4f})")

    print("\nSTEP 2 -- how often did retrieval actually lose something?")
    incomplete = df[~df.sl_recall_complete]
    complete = df[df.sl_recall_complete]
    print(f"  Retrieval kept everything the gold query needed on "
          f"{len(complete)}/{n} questions ({100 * len(complete) / n:.1f}%)")
    print(f"  It dropped something needed on {len(incomplete)} questions "
          f"({100 * len(incomplete) / n:.1f}%)")

    print("\nSTEP 3 -- the same Retrieved-vs-Perfect comparison inside each group")
    rows = []
    header = (f"  {'group':<34}{'n':>6}{'Retrieved':>11}{'Perfect':>10}"
              f"{'gap':>6}{'p':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, sub in [("(a) something needed was dropped", incomplete),
                       ("(b) nothing needed was dropped", complete)]:
        k = len(sub)
        r = int(sub.retrieved_ok.sum())
        pf = int(sub.perfect_ok.sum())
        bb, cc, mm, pp = paired(sub, "retrieved_ok", "perfect_ok")
        print(f"  {label:<34}{k:>6}{100 * r / k:>10.1f}%{100 * pf / k:>9.1f}%"
              f"{pf - r:>6}{pp:>9.4f}")
        rows.append({"group": label, "n": k, "retrieved": r, "perfect": pf,
                     "gap": pf - r, "perfect_wins": cc, "retrieved_wins": bb,
                     "mcnemar_p": round(pp, 4)})

    total_gap = perf - retr
    from_over = rows[1]["gap"]
    print(f"\nSTEP 4 -- conclusion")
    print(f"  Of the {total_gap}-question gap, {from_over} ({100 * from_over / total_gap:.0f}%) "
          f"come from questions where retrieval lost NOTHING.")
    print(f"  Only {rows[0]['gap']} come from questions where something needed was dropped.")
    print("  The gap is therefore a precision problem, not a recall problem:")
    print("  the linker keeps what is needed and too much besides.")

    print("\nSTEP 5 -- how much surplus does it keep? (questions with complete recall)")
    sub = complete.copy()
    print(f"  columns needed by the gold query : {sub.sl_cols_oracle.mean():.1f} on average")
    print(f"  columns handed to the model      : {sub.sl_cols_retrieved.mean():.1f} on average")
    print(f"  surplus factor                   : x{sub.sl_cols_retrieved.mean() / max(sub.sl_cols_oracle.mean(), 1e-9):.1f}")

    # Descriptive only -- surplus is confounded with query complexity, see below.
    sub["surplus"] = sub.sl_cols_retrieved - sub.sl_cols_oracle
    q33, q66 = sub.surplus.quantile([0.33, 0.66]).values
    sub["surplus_group"] = sub.surplus.map(
        lambda x: "low" if x <= q33 else ("medium" if x <= q66 else "high"))
    print(f"\n  {'surplus group':<16}{'n':>6}{'mean surplus':>14}{'mean complexity':>17}"
          f"{'Retrieved':>11}{'Perfect':>10}{'gap':>6}")
    for g in ["low", "medium", "high"]:
        s = sub[sub.surplus_group == g]
        k = len(s)
        if not k:
            continue
        r = int(s.retrieved_ok.sum())
        pf = int(s.perfect_ok.sum())
        print(f"  {g:<16}{k:>6}{s.surplus.mean():>14.1f}{s.complexity.mean():>17.2f}"
              f"{100 * r / k:>10.1f}%{100 * pf / k:>9.1f}%{pf - r:>6}")
        rows.append({"group": f"complete recall, surplus={g}", "n": k,
                     "retrieved": r, "perfect": pf, "gap": pf - r,
                     "mean_surplus": round(s.surplus.mean(), 1),
                     "mean_complexity": round(s.complexity.mean(), 2)})
    print("\n  CAUTION: surplus rises with query complexity (see the column above),")
    print("  so this last table is a descriptive observation, not a causal claim.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
