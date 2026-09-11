"""
evaluate_value_grounding.py
What this does: measures whether the value_grounding_fix.py corrections
actually move execution accuracy. For each condition (Baseline / Memory A /
Memory B compact), executes both the original and the fixed SQL against the
real database, compares to gold, and reports before/after accuracy plus a
paired McNemar test (same questions, same condition, before vs after the
fix) -- so we can tell whether any change is more than chance, and also
whether the fix ever makes things worse (a previously-correct query broken
by an unwanted "correction").
Input: value_grounding_results.json (from value_grounding_fix.py),
generation_results.json, generation_results_multi.json,
generation_results_remaining.json, generation_results_compact_B.json (for
gold_sql).
Output: printed before/after accuracy per condition, McNemar p-values, and a
breakdown of wrong-to-right vs right-to-wrong flips.
"""
import json, sqlite3, os, sys
sys.path.insert(0, os.getcwd())
from statsmodels.stats.contingency_tables import mcnemar


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


def load_gold():
    all_raw = []
    with open("generation_results.json") as f:
        single = json.load(f)
        for r in single:
            r["db_id"] = "concert_singer"
        all_raw.extend(single)
    with open("generation_results_multi.json") as f:
        all_raw.extend(json.load(f))
    with open("generation_results_remaining.json") as f:
        all_raw.extend(json.load(f))
    return {(r["db_id"], r["question"]): r["gold_sql"] for r in all_raw}


def evaluate_condition(label, rows, gold_by_key):
    before_ok, after_ok = [], []
    for r in rows:
        key = (r["db_id"], r["question"])
        gold_sql = gold_by_key.get(key)
        if gold_sql is None:
            continue
        db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
        gold_result = execute_query(db_path, gold_sql)
        before_result = execute_query(db_path, r["original_sql"])
        after_result = execute_query(db_path, r["fixed_sql"])
        before_ok.append(before_result == gold_result)
        after_ok.append(after_result == gold_result)

    n = len(before_ok)
    n_before = sum(before_ok)
    n_after = sum(after_ok)
    wrong_to_right = sum(1 for b, a in zip(before_ok, after_ok) if (not b) and a)
    right_to_wrong = sum(1 for b, a in zip(before_ok, after_ok) if b and (not a))
    both_right = sum(1 for b, a in zip(before_ok, after_ok) if b and a)
    both_wrong = sum(1 for b, a in zip(before_ok, after_ok) if (not b) and (not a))

    table = [[both_right, right_to_wrong], [wrong_to_right, both_wrong]]
    result = mcnemar(table, exact=True)

    print(f"\n=== {label} (n={n}) ===")
    print(f"Before: {n_before}/{n} ({100*n_before/n:.1f}%)  |  After: {n_after}/{n} ({100*n_after/n:.1f}%)")
    print(f"Wrong->Right (fixed by grounding): {wrong_to_right}  |  Right->Wrong (regressions): {right_to_wrong}")
    print(f"McNemar p-value: {result.pvalue:.4f}")


def main():
    with open("value_grounding_results.json") as f:
        vg = json.load(f)
    gold_by_key = load_gold()

    evaluate_condition("Baseline", vg["baseline"], gold_by_key)
    evaluate_condition("Memory A", vg["memory_A"], gold_by_key)
    evaluate_condition("Memory B (compact)", vg["memory_B_compact"], gold_by_key)


if __name__ == "__main__":
    main()
