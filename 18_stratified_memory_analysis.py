"""
18_stratified_memory_analysis.py

WHAT THIS DOES
    Splits the 1034 dev questions by whether the memory retrieved a
    structurally relevant example, and repeats the Baseline-vs-Memory
    comparison inside each group.

WHY IT EXISTS
    Overall, the memory changes nothing: 737 -> 732 (p = 0.70). But an average
    over 1034 questions can hide an effect that only exists on a subset. The
    retrieval-quality measurement already told us that only about 20% of
    questions receive a top-1 example with the same SQL structure as the gold
    query, and about 32% receive one somewhere in the top-3. If the few-shot
    examples help at all, that is where it must show.

    Two outcomes, both informative:
      * memory helps in the "relevant example" group and not in the other ->
        the overall null is a dilution effect, and the mechanism works when
        retrieval succeeds. The bottleneck is retrieval quality.
      * memory does nothing in either group -> few-shot examples do not move
        this model regardless of their quality. The bottleneck is the model.

    This mirrors 16_decompose_schema_gap.py, which did the same thing for the
    schema-linking thread. The two threads then have symmetric explanations
    rather than one explained result and one bare null.

METHOD AND ITS GUARDRAIL
    The grouping variable (was the retrieved example structurally relevant?)
    is computed from the retrieval alone and never looks at whether the answer
    turned out correct. A split defined by the outcome would guarantee a
    "finding" and mean nothing.

    Primary split uses top-3 relevance, because all three retrieved examples
    go into the prompt: what matters is whether the model saw at least one
    useful example. Top-1 is reported as a secondary, stricter split.

THE INTERACTION TEST
    Comparing the two groups by eye is not enough: each subgroup has its own
    wide confidence interval, and two intervals that look different can easily
    be the same effect. The question "is the memory effect DIFFERENT between
    the two groups?" needs its own test.

    Let delta_i = (memory correct on question i) - (baseline correct on
    question i), a value in {-1, 0, +1}. The quantity of interest is
    mean(delta | relevant example) - mean(delta | no relevant example). Its
    null distribution is obtained by permutation: the relevance labels are
    reshuffled across questions 20,000 times and the same statistic
    recomputed. This assumes nothing about the distribution and is exact up to
    Monte-Carlo error.

INPUT   results/master_results.csv                    (script 14)
        results/retrieval_quality_per_question.csv    (script 17)
OUTPUT  printed report + results/stratified_memory.csv

HOW TO RUN
    python 18_stratified_memory_analysis.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

MASTER = os.path.join("results", "master_results.csv")
RETRIEVAL = os.path.join("results", "retrieval_quality_per_question.csv")
OUT_FILE = os.path.join("results", "stratified_memory.csv")
Z = norm.ppf(0.975)


def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def compare(sub, col_a, col_b):
    """Paired comparison inside one stratum. Returns dict of stats."""
    a, b_ = sub[col_a], sub[col_b]
    b = int((a & ~b_).sum())     # baseline right, memory wrong
    c = int((~a & b_).sum())     # memory right, baseline wrong
    m = b + c
    k = len(sub)
    diff = (c - b) / k if k else float("nan")
    se = np.sqrt(m - (c - b) ** 2 / k) / k if m and k else float("nan")
    p = binomtest(min(b, c), m, 0.5).pvalue if m else float("nan")
    return {"n": k, "a_correct": int(a.sum()), "b_correct": int(b_.sum()),
            "memory_broke": b, "memory_fixed": c, "discordant": m,
            "diff_pp": 100 * diff, "ci_low": 100 * (diff - Z * se),
            "ci_high": 100 * (diff + Z * se), "p": p}


def report(title, df, split_col, col_a, col_b, label_a, label_b, out_rows):
    print(f"\n{title}")
    print(f"  grouping variable: {split_col}")
    header = (f"  {'group':<26}{'n':>6}{label_a:>10}{label_b:>10}"
              f"{'diff pp':>9}{'95% CI':>19}{'p':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for flag, name in [(True, "relevant example"), (False, "no relevant example")]:
        sub = df[df[split_col] == flag]
        if not len(sub):
            continue
        s = compare(sub, col_a, col_b)
        print(f"  {name:<26}{s['n']:>6}{s['a_correct']:>10}{s['b_correct']:>10}"
              f"{s['diff_pp']:>+9.2f}   [{s['ci_low']:>+6.2f},{s['ci_high']:>+6.2f}]{s['p']:>9.4f}")
        out_rows.append({"analysis": title, "split": split_col, "group": name, **s})
    s = compare(df, col_a, col_b)
    print(f"  {'ALL QUESTIONS':<26}{s['n']:>6}{s['a_correct']:>10}{s['b_correct']:>10}"
          f"{s['diff_pp']:>+9.2f}   [{s['ci_low']:>+6.2f},{s['ci_high']:>+6.2f}]{s['p']:>9.4f}")
    out_rows.append({"analysis": title, "split": split_col, "group": "all", **s})


def interaction_test(df, out_rows, n_perm=20000, seed=0):
    """Is the memory effect actually DIFFERENT between the two groups?

    Statistic: mean(delta | relevant) - mean(delta | not relevant), where
    delta = memory_correct - baseline_correct per question. Null distribution
    by reshuffling the relevance labels.
    """
    rng = np.random.default_rng(seed)
    base = df["baseline_ok"].values.astype(int)
    print("\nInteraction test -- is the effect different between the two groups?")
    header = (f"  {'analysis':<22}{'effect|relevant':>17}{'effect|not':>13}"
              f"{'difference':>13}{'perm p':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mem_col, rel_col, name in [
            ("memB_compact_ok", "B_top3_match", "Memory B / top-3"),
            ("memB_compact_ok", "B_top1_match", "Memory B / top-1"),
            ("memA_ok",         "A_top3_match", "Memory A / top-3"),
            ("memA_ok",         "A_top1_match", "Memory A / top-1")]:
        delta = df[mem_col].values.astype(int) - base
        rel = df[rel_col].values
        d1, d0 = delta[rel].mean(), delta[~rel].mean()
        obs = d1 - d0
        n, n1, tot = len(delta), int(rel.sum()), delta.sum()
        # Permute in chunks: a single (n_perm x n) array would need hundreds of
        # megabytes, which is more than a modest laptop should be asked for.
        extreme = 0
        chunk = 1000
        for start in range(0, n_perm, chunk):
            size = min(chunk, n_perm - start)
            idx = np.argsort(rng.random((size, n)), axis=1)[:, :n1]
            sums = delta[idx].sum(axis=1)
            perm = sums / n1 - (tot - sums) / (n - n1)
            extreme += int((np.abs(perm) >= abs(obs) - 1e-12).sum())
        p = extreme / n_perm
        print(f"  {name:<22}{100 * d1:>+16.2f}{100 * d0:>+13.2f}"
              f"{100 * obs:>+13.2f}{p:>9.4f}")
        out_rows.append({"analysis": "interaction " + name, "split": rel_col,
                         "group": "difference of differences",
                         "diff_pp": round(100 * obs, 2), "p": round(p, 4)})


def main():
    master = pd.read_csv(MASTER)
    retr = pd.read_csv(RETRIEVAL)

    for c in [x for x in master.columns if x.endswith("_ok")]:
        master[c] = to_bool(master[c])
    for c in [x for x in retr.columns if x.endswith("_match")]:
        retr[c] = to_bool(retr[c])

    master["question"] = master["question"].astype(str).str.strip()
    retr["question"] = retr["question"].astype(str).str.strip()
    df = master.merge(retr, on=["db_id", "question"], how="inner")
    print(f"Merged {len(df)}/{len(master)} questions "
          f"({'complete' if len(df) == len(master) else 'INCOMPLETE -- check the join'})")

    print("\nHow often did retrieval find a structurally relevant example?")
    for col in ["A_top1_match", "A_top3_match", "B_top1_match", "B_top3_match"]:
        k = int(df[col].sum())
        print(f"  {col:<16}{k:>5}/{len(df)}  ({100 * k / len(df):.1f}%)")

    out_rows = []
    report("Memory B (compact) vs Baseline -- split on top-3 relevance",
           df, "B_top3_match", "baseline_ok", "memB_compact_ok",
           "Baseline", "Memory B", out_rows)
    report("Memory A (raw) vs Baseline -- split on top-3 relevance",
           df, "A_top3_match", "baseline_ok", "memA_ok",
           "Baseline", "Memory A", out_rows)
    report("Memory B (compact) vs Baseline -- split on top-1 relevance (stricter)",
           df, "B_top1_match", "baseline_ok", "memB_compact_ok",
           "Baseline", "Memory B", out_rows)
    report("Memory A (raw) vs Baseline -- split on top-1 relevance (stricter)",
           df, "A_top1_match", "baseline_ok", "memA_ok",
           "Baseline", "Memory A", out_rows)

    interaction_test(df, out_rows)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}")

    print("\nHOW TO READ THIS")
    print("  Dilution would show as a positive effect in the 'relevant example' row,")
    print("  no effect in the other, and a significant interaction p. Model")
    print("  indifference would show as zero in both rows. Judge by the confidence")
    print("  intervals and the interaction p, never by the point estimates alone --")
    print("  the subgroups are small and their intervals are correspondingly wide.")
    print("  A consistent direction across the four analyses is not four independent")
    print("  confirmations: they share questions and largely share retrieved")
    print("  neighbours, so they are four views of the same data.")


if __name__ == "__main__":
    main()
