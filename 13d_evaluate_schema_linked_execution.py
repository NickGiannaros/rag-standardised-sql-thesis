"""
13d_evaluate_schema_linked_execution.py
What this does: execution accuracy for the schema-linking experiment --
mirrors 6d_evaluate_full_dev_set.py's role, comparing Full schema (no
filtering, already generated for the core RAG experiment), Perfect/oracle
schema (theoretical ceiling), and Retrieved schema (the real bidirectional
mechanism, 13_schema_linking.py) across all 1034 dev questions. Executes
every generated SQL against the real SQLite databases, compares to gold,
and runs paired McNemar tests between Full vs Retrieved and Perfect vs
Retrieved -- same methodology used throughout this project.
Input: generation_results_schema_linked.json.
Output: printed accuracy table + McNemar p-values.
"""
import json, sqlite3, re, sys, os
sys.path.insert(0, os.getcwd())
from statsmodels.stats.contingency_tables import mcnemar

IN_FILE = "generation_results_schema_linked.json"


def clean_markdown(sql):
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```", "", sql)
    return sql.strip()


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


def mcnemar_report(a_name, a_ok, b_name, b_ok):
    a_right_b_wrong = sum(1 for a, b in zip(a_ok, b_ok) if a and not b)
    b_right_a_wrong = sum(1 for a, b in zip(a_ok, b_ok) if b and not a)
    both_right = sum(1 for a, b in zip(a_ok, b_ok) if a and b)
    both_wrong = sum(1 for a, b in zip(a_ok, b_ok) if not a and not b)
    table = [[both_right, a_right_b_wrong], [b_right_a_wrong, both_wrong]]
    result = mcnemar(table, exact=True)
    print(f"\n--- {a_name} vs {b_name} ---")
    print(f"{a_name} correct / {b_name} wrong: {a_right_b_wrong}  |  {b_name} correct / {a_name} wrong: {b_right_a_wrong}")
    print(f"McNemar p-value: {result.pvalue:.4f}")
    return result.pvalue


def main():
    with open(IN_FILE) as f:
        rows = json.load(f)
    print("Total questions:", len(rows))

    full_ok, perfect_ok, retrieved_ok = [], [], []

    for r in rows:
        db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
        gold_result = execute_query(db_path, r["gold_sql"])

        full_ok.append(execute_query(db_path, clean_markdown(r["sql_baseline"])) == gold_result)
        perfect_ok.append(execute_query(db_path, clean_markdown(r["sql_perfect_schema"])) == gold_result)
        retrieved_ok.append(execute_query(db_path, clean_markdown(r["sql_retrieved_schema"])) == gold_result)

    n = len(rows)
    print("\n=== Execution Accuracy ===")
    print(f"Full (no filtering):  {sum(full_ok)}/{n} ({100*sum(full_ok)/n:.1f}%)")
    print(f"Perfect (oracle):     {sum(perfect_ok)}/{n} ({100*sum(perfect_ok)/n:.1f}%)")
    print(f"Retrieved (ours):     {sum(retrieved_ok)}/{n} ({100*sum(retrieved_ok)/n:.1f}%)")

    mcnemar_report("Full", full_ok, "Perfect", perfect_ok)
    mcnemar_report("Full", full_ok, "Retrieved", retrieved_ok)
    mcnemar_report("Perfect", perfect_ok, "Retrieved", retrieved_ok)


if __name__ == "__main__":
    main()
