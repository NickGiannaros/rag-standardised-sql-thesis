"""
21_factorial_analysis.py

WHAT THIS DOES
    Evaluates the two combined conditions produced by 20_generate_combined.py,
    adds them to results/master_results.csv, and analyses the completed
    2 x 3 design:

        schema   in {Full, Retrieved}
        examples in {none, raw (Memory A), standardised (Memory B compact)}

WHY IT EXISTS
    Six separate pairwise tests on six related conditions inflate the chance
    of a false positive and still never answer the question the design was
    built for: do the two mechanisms interact? This script asks the questions
    in the right order.

THREE TESTS, IN ORDER
    1. Cochran's Q -- the omnibus test. It generalises McNemar to k > 2
       related binary samples and asks whether ANY of the six conditions
       differs from the others. If it does not reject, individual pairwise
       comparisons should not be mined for significance.

    2. Pairwise McNemar with Holm-Bonferroni correction -- run only as
       follow-up, and corrected, because six conditions give 15 pairwise
       comparisons; at alpha = 0.05 and no true differences, the chance that
       at least one comes out "significant" is 1 - 0.95^15, roughly 54%.

    3. The interaction -- the actual question. For each question define
           delta_full = memory_correct - baseline_correct     (Full schema)
           delta_retr = combined_correct - retrieved_correct  (Retrieved schema)
       The interaction per question is delta_retr - delta_full, and the
       hypothesis is that its mean is zero: the memory has the same effect
       whichever schema it accompanies. Because the quantity is paired within
       a question, the null distribution is obtained by randomly flipping the
       sign of each question's interaction (sign-flipping permutation), which
       assumes only symmetry under the null.

       Positive mean  -> synergy: memory helps more on a pruned schema.
       Negative mean  -> interference: memory hurts more on a pruned schema.
       Zero           -> the mechanisms are orthogonal.

INPUT   results/master_results.csv, generation_results_combined.json, database/
OUTPUT  results/master_results.csv   (two columns added)
        results/factorial.csv
        printed report

HOW TO RUN
    Run from the working directory containing database/:
        python 21_factorial_analysis.py
"""

import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from statsmodels.stats.contingency_tables import cochrans_q

MASTER = os.path.join("results", "master_results.csv")
COMBINED = "generation_results_combined.json"
OUT_FILE = os.path.join("results", "factorial.csv")
Z = norm.ppf(0.975)

CELLS = [
    ("Full / none",           "baseline_ok"),
    ("Full / raw",            "memA_ok"),
    ("Full / standardised",   "memB_compact_ok"),
    ("Retr / none",           "retrieved_ok"),
    ("Retr / raw",            "combined_A_ok"),
    ("Retr / standardised",   "combined_B_ok"),
]


def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def clean_markdown(sql):
    sql = re.sub(r"```sql\s*", "", sql or "")
    sql = re.sub(r"```", "", sql)
    return sql.strip()


def clean_memory_B(sql, process_sql):
    sql = clean_markdown(sql)
    if "NONE" in sql and process_sql is not None:
        try:
            cleaned = process_sql.SimpleSQL_to_SQL(sql)
            if cleaned and cleaned != "SELECT ERROR":
                return cleaned
        except Exception:
            pass
    return sql


def execute_query(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = sorted(cur.fetchall())
        conn.close()
        return rows
    except Exception as e:
        return "ERROR: " + str(e)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def evaluate_combined(df):
    try:
        from eval import process_sql
    except Exception:
        process_sql = None
        print("WARNING: eval.process_sql unavailable -- Memory B de-standardisation "
              "will be skipped and combined_B numbers will be wrong.")

    with open(COMBINED) as f:
        combined = json.load(f)
    by_key = {(r["db_id"], r["question"].strip()): r for r in combined}
    print(f"Combined generations available for {len(by_key)} questions")

    a_ok, b_ok = [], []
    for _, r in df.iterrows():
        key = (r["db_id"], str(r["question"]).strip())
        c = by_key.get(key)
        if c is None:
            a_ok.append(None)
            b_ok.append(None)
            continue
        db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
        gold = execute_query(db_path, r["gold_sql"])
        a_ok.append(execute_query(db_path, clean_markdown(c["sql_combined_A"])) == gold)
        b_ok.append(execute_query(db_path, clean_memory_B(c["sql_combined_B"], process_sql)) == gold)
    df["combined_A_ok"] = a_ok
    df["combined_B_ok"] = b_ok
    return df


def paired(df, col_a, col_b):
    a, b_ = df[col_a], df[col_b]
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    m = b + c
    n = len(df)
    diff = (c - b) / n
    se = np.sqrt(m - (c - b) ** 2 / n) / n if m else float("nan")
    p = binomtest(min(b, c), m, 0.5).pvalue if m else 1.0
    return diff, diff - Z * se, diff + Z * se, p, m


def main():
    df = pd.read_csv(MASTER)
    for c in [x for x in df.columns if x.endswith("_ok")]:
        df[c] = to_bool(df[c])

    if "combined_A_ok" not in df.columns or df["combined_A_ok"].isna().all():
        print("Evaluating the combined conditions...")
        df = evaluate_combined(df)
        df.to_csv(MASTER, index=False)
        print(f"Added combined_A_ok / combined_B_ok to {MASTER}")

    df = df.dropna(subset=["combined_A_ok", "combined_B_ok"]).copy()
    for c in ["combined_A_ok", "combined_B_ok"]:
        df[c] = df[c].astype(bool)
    n = len(df)

    print(f"\nSTEP 1 -- the completed 2 x 3 design ({n} questions)")
    print(f"  {'':<12}{'no examples':>14}{'raw (A)':>14}{'standardised (B)':>19}")
    for schema, cols in [("Full", ["baseline_ok", "memA_ok", "memB_compact_ok"]),
                         ("Retrieved", ["retrieved_ok", "combined_A_ok", "combined_B_ok"])]:
        cells = "".join(f"{int(df[c].sum()):>8} ({100 * df[c].mean():>4.1f}%)" for c in cols)
        print(f"  {schema:<12}{cells}")

    print("\nSTEP 2 -- Cochran's Q (omnibus: does ANY condition differ?)")
    mat = np.column_stack([df[c].astype(int).values for _, c in CELLS])
    q = cochrans_q(mat)
    print(f"  Q = {q.statistic:.3f}, df = {len(CELLS) - 1}, p = {q.pvalue:.4f}")
    if q.pvalue >= 0.05:
        print("  Not significant: the six conditions are statistically indistinguishable.")
        print("  The pairwise tests below are reported for completeness only.")
    else:
        print("  Significant: at least one condition differs; see the corrected")
        print("  pairwise comparisons below.")

    print("\nSTEP 3 -- pairwise comparisons, Holm-Bonferroni corrected")
    pairs, raw_p, rows = [], [], []
    for i in range(len(CELLS)):
        for j in range(i + 1, len(CELLS)):
            (na, ca), (nb, cb) = CELLS[i], CELLS[j]
            d, lo, hi, p, m = paired(df, ca, cb)
            pairs.append((f"{na} vs {nb}", d, lo, hi, m))
            raw_p.append(p)
    adj = holm(np.array(raw_p))
    print(f"  {'comparison':<44}{'diff pp':>9}{'95% CI':>19}{'p':>9}{'p (Holm)':>11}")
    print("  " + "-" * 92)
    for (label, d, lo, hi, m), p, pa in zip(pairs, raw_p, adj):
        print(f"  {label:<44}{100 * d:>+9.2f}   [{100 * lo:>+6.2f},{100 * hi:>+6.2f}]"
              f"{p:>9.4f}{pa:>11.4f}")
        rows.append({"test": "pairwise", "comparison": label,
                     "diff_pp": round(100 * d, 2),
                     "ci_low_pp": round(100 * lo, 2), "ci_high_pp": round(100 * hi, 2),
                     "discordant": m, "p": round(p, 4), "p_holm": round(pa, 4)})

    print("\nSTEP 4 -- the interaction: does the memory behave differently on a pruned schema?")
    rng = np.random.default_rng(0)
    print(f"  {'memory':<16}{'effect on Full':>16}{'effect on Retr':>16}"
          f"{'interaction':>14}{'perm p':>9}")
    print("  " + "-" * 73)
    for mem_full, mem_comb, label in [("memA_ok", "combined_A_ok", "raw (A)"),
                                      ("memB_compact_ok", "combined_B_ok", "standardised (B)")]:
        d_full = df[mem_full].astype(int).values - df["baseline_ok"].astype(int).values
        d_retr = df[mem_comb].astype(int).values - df["retrieved_ok"].astype(int).values
        inter = d_retr - d_full
        obs = inter.mean()
        signs = rng.choice([-1, 1], size=(20000, len(inter)))
        perm = (signs * inter).mean(axis=1)
        p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
        print(f"  {label:<16}{100 * d_full.mean():>+15.2f}{100 * d_retr.mean():>+16.2f}"
              f"{100 * obs:>+14.2f}{p:>9.4f}")
        rows.append({"test": "interaction", "comparison": label,
                     "effect_full_pp": round(100 * d_full.mean(), 2),
                     "effect_retrieved_pp": round(100 * d_retr.mean(), 2),
                     "interaction_pp": round(100 * obs, 2), "p": round(p, 4)})

    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}")
    print("\nHOW TO READ THIS")
    print("  Read STEP 2 first. If Cochran's Q does not reject, the honest summary is")
    print("  that none of the six configurations separates from the others, and the")
    print("  pairwise table should not be mined for the lowest p-value.")
    print("  STEP 4 is the question this experiment was run to answer: an interaction")
    print("  indistinguishable from zero means the two mechanisms neither reinforce")
    print("  nor obstruct each other, which is a finding in its own right.")


if __name__ == "__main__":
    main()
