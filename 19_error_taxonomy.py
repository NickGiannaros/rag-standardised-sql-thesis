"""
19_error_taxonomy.py

WHAT THIS DOES
    Classifies the questions the baseline gets wrong into failure types, then
    measures -- empirically, not by assertion -- how often each type is
    actually repaired by each of the mechanisms studied.

WHY IT EXISTS
    Every experiment in this work reports how much accuracy moved. None of
    them says what the remaining errors ARE. Without that, the conclusion
    "memory and retrieval do not help much" is incomplete: it does not
    establish whether the errors were of a kind that memory or retrieval
    could ever have fixed.

    A prior observation motivates the question. Of the 1034 dev questions,
    863 (83.5%) are answered correctly by at least one of the six conditions,
    while 171 (16.5%) are wrong under all of them. That 16.5% is an empirical
    upper bound on everything attempted here: even an oracle that picked the
    best condition per question could not exceed 83.5%. This script asks what
    those questions contain.

CATEGORIES (first match wins, applied to the baseline's failures)
    gold_broken        the reference query itself does not execute
    hallucinated_name  prediction references a table or column that does not exist
    not_executable     prediction fails for some other reason (syntax, types)
    wrong_arity        prediction returns a different number of columns
    empty_result       prediction returns no rows where the reference returns some
    unexpected_rows    prediction returns rows where the reference returns none
    too_restrictive    prediction's rows are a strict subset of the reference's
    too_permissive     prediction's rows are a strict superset
    wrong_values       same shape, different content
    other              anything not covered above

    Separately, and independently of the category, the script records which
    SQL constructs differ between prediction and reference (GROUP BY, JOIN,
    DISTINCT, aggregation, ORDER BY/LIMIT, nesting, set operations). A
    category says how the answer was wrong; the construct diff says what the
    query got wrong.

THE POINT OF THE CROSS-TABULATION
    For each category the script reports the proportion repaired by schema
    linking, by memory, and by the perfect-schema oracle. This replaces an
    argument about which failures each mechanism ought to address with a
    measurement of which ones it did.

INPUT   results/master_results.csv, database/
OUTPUT  printed report
        results/error_taxonomy.csv        (per-question categories)
        results/error_taxonomy_summary.csv

HOW TO RUN
    Run from the working directory containing database/:
        python 19_error_taxonomy.py
"""

import os
import re
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())

import pandas as pd

IN_FILE = os.path.join("results", "master_results.csv")
OUT_ROWS = os.path.join("results", "error_taxonomy.csv")
OUT_SUMMARY = os.path.join("results", "error_taxonomy_summary.csv")

AGG_FUNCS = ["count", "avg", "sum", "max", "min"]
CONDITIONS = ["baseline_ok", "memA_ok", "memB_ok", "memB_compact_ok",
              "retrieved_ok", "perfect_ok"]


def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def execute_query(db_path, sql):
    """Returns (rows, error_message). Same execution rules as every other
    script in this project: rows are sorted before comparison."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = sorted(cur.fetchall())
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def features(sql):
    """Which SQL constructs a query uses."""
    s = (sql or "").lower()
    return {
        "WHERE": bool(re.search(r"\bwhere\b", s)),
        "GROUP BY": bool(re.search(r"\bgroup by\b", s)),
        "HAVING": bool(re.search(r"\bhaving\b", s)),
        "ORDER BY": bool(re.search(r"\border by\b", s)),
        "LIMIT": bool(re.search(r"\blimit\b", s)),
        "JOIN": bool(re.search(r"\bjoin\b", s)),
        "DISTINCT": bool(re.search(r"\bdistinct\b", s)),
        "aggregate": any(re.search(r"\b" + f + r"\s*\(", s) for f in AGG_FUNCS),
        "nested": s.count("select") > 1,
        "set op": bool(re.search(r"\b(intersect|except|union)\b", s)),
    }


def classify(gold_rows, gold_err, pred_rows, pred_err):
    if gold_err is not None:
        return "gold_broken"
    if pred_err is not None:
        low = pred_err.lower()
        if "no such column" in low or "no such table" in low:
            return "hallucinated_name"
        return "not_executable"
    if not pred_rows and gold_rows:
        return "empty_result"
    if pred_rows and not gold_rows:
        return "unexpected_rows"
    if not pred_rows and not gold_rows:
        return "other"
    if len(pred_rows[0]) != len(gold_rows[0]):
        return "wrong_arity"
    gset, pset = set(gold_rows), set(pred_rows)
    if pset < gset:
        return "too_restrictive"
    if gset < pset:
        return "too_permissive"
    return "wrong_values"


def main():
    df = pd.read_csv(IN_FILE)
    for c in CONDITIONS:
        df[c] = to_bool(df[c])
    n = len(df)

    solved_somewhere = df[CONDITIONS].any(axis=1)
    print("STEP 1 -- the ceiling of everything attempted")
    print(f"  Baseline correct                    {int(df.baseline_ok.sum())}/{n}"
          f"  ({100 * df.baseline_ok.mean():.1f}%)")
    print(f"  Perfect schema correct              {int(df.perfect_ok.sum())}/{n}"
          f"  ({100 * df.perfect_ok.mean():.1f}%)")
    print(f"  Correct under at least one condition {int(solved_somewhere.sum())}/{n}"
          f"  ({100 * solved_somewhere.mean():.1f}%)")
    print(f"  Wrong under every condition          {int((~solved_somewhere).sum())}/{n}"
          f"  ({100 * (~solved_somewhere).mean():.1f}%)")
    print(f"  mean gold-query complexity: solved {df[solved_somewhere].complexity.mean():.2f}"
          f"  vs never solved {df[~solved_somewhere].complexity.mean():.2f}")

    print("\nSTEP 2 -- classifying the baseline's failures (re-executing queries)")
    failures = df[~df.baseline_ok].copy()
    rows = []
    for i, (_, r) in enumerate(failures.iterrows(), 1):
        db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
        gold_rows, gold_err = execute_query(db_path, r["gold_sql"])
        pred_rows, pred_err = execute_query(db_path, str(r["sql_baseline"]))
        cat = classify(gold_rows, gold_err, pred_rows, pred_err)

        fg, fp = features(r["gold_sql"]), features(r["sql_baseline"])
        missing = [k for k in fg if fg[k] and not fp[k]]
        spurious = [k for k in fg if fp[k] and not fg[k]]

        rows.append({
            "qid": r["qid"], "db_id": r["db_id"], "question": r["question"],
            "complexity": r["complexity"], "difficulty": r["difficulty"],
            "category": cat,
            "missing_constructs": "|".join(missing),
            "spurious_constructs": "|".join(spurious),
            "fixed_by_memA": bool(r["memA_ok"]),
            "fixed_by_memB": bool(r["memB_compact_ok"]),
            "fixed_by_retrieved": bool(r["retrieved_ok"]),
            "fixed_by_perfect": bool(r["perfect_ok"]),
            "fixed_by_anything": bool(r[CONDITIONS].any()),
        })
        if i % 100 == 0:
            print(f"  ...{i}/{len(failures)}")

    tax = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    tax.to_csv(OUT_ROWS, index=False)

    # One condition per column: "memory" must not get two chances where the
    # other mechanisms get one, or the comparison is not a comparison.
    repair_cols = [("fixed_by_memA", "Memory A"), ("fixed_by_memB", "Memory B"),
                   ("fixed_by_retrieved", "Schema link"), ("fixed_by_perfect", "Perfect")]

    print(f"\nSTEP 3 -- failure types among the {len(tax)} baseline errors")
    print("  Repair rate per failure type, one condition at a time.")
    print(f"  {'category':<20}{'n':>6}{'share':>8}"
          + "".join(f"{lbl:>13}" for _, lbl in repair_cols) + f"{'any':>8}")
    print("  " + "-" * 88)
    summary = []
    for cat, k in tax.category.value_counts().items():
        sub = tax[tax.category == cat]
        row = {"category": cat, "n": int(k), "share_pct": round(100 * k / len(tax), 1),
               "fixed_by_anything_pct": round(100 * sub.fixed_by_anything.mean(), 1)}
        for col, lbl in repair_cols:
            row[col + "_pct"] = round(100 * sub[col].mean(), 1)
        print(f"  {cat:<20}{k:>6}{row['share_pct']:>7.1f}%"
              + "".join(f"{row[c + '_pct']:>12.1f}%" for c, _ in repair_cols)
              + f"{row['fixed_by_anything_pct']:>7.1f}%")
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)

    print("\nSTEP 3b -- what each condition does to the whole dev set")
    print("  A null result can mean nothing happened, or that gains and losses")
    print("  cancelled. This separates the two.")
    correct = df[df.baseline_ok]
    print(f"  {'condition':<16}{'repairs':>10}{'breaks':>10}{'net':>8}")
    print("  " + "-" * 44)
    for col, lbl in repair_cols:
        ok_col = {"fixed_by_memA": "memA_ok", "fixed_by_memB": "memB_compact_ok",
                  "fixed_by_retrieved": "retrieved_ok", "fixed_by_perfect": "perfect_ok"}[col]
        repairs = int(tax[col].sum())
        breaks = int((~correct[ok_col]).sum())
        print(f"  {lbl:<16}{repairs:>10}{breaks:>10}{repairs - breaks:>+8}")
        summary.append({"category": f"WHOLE SET: {lbl}", "repairs": repairs,
                        "breaks": breaks, "net": repairs - breaks})
    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)

    print("\nSTEP 4 -- which SQL constructs the failed queries got wrong")
    miss = Counter()
    spur = Counter()
    for _, r in tax.iterrows():
        miss.update([c for c in str(r["missing_constructs"]).split("|") if c])
        spur.update([c for c in str(r["spurious_constructs"]).split("|") if c])
    print(f"  {'construct':<14}{'missing':>10}{'spurious':>10}")
    for c in ["WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "JOIN",
              "DISTINCT", "aggregate", "nested", "set op"]:
        print(f"  {c:<14}{miss.get(c, 0):>10}{spur.get(c, 0):>10}")

    print(f"\nWrote {OUT_ROWS} and {OUT_SUMMARY}")
    print("\nHOW TO READ THIS")
    print("  The repair columns say which failure types each mechanism actually")
    print("  repaired, rather than which ones it was expected to. A category with a")
    print("  high repair rate is one the mechanism addresses; a low rate means the")
    print("  category is out of its reach and belongs in the limitations discussion,")
    print("  because better retrieval of the same kind would not have moved it.")
    print("  Read STEP 3b together with STEP 3: a mechanism with a high repair rate")
    print("  is not necessarily useful, because the same intervention also breaks")
    print("  questions that were previously correct. Only the net column settles it.")


if __name__ == "__main__":
    main()
