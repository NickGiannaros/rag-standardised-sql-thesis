"""
13c_measure_schema_linking_quality.py
What this does: retrieval quality on its own, no execution involved --
mirrors 7_measure_retrieval_quality.py's role for Memory A/B, but for schema
linking. Compares the Retrieved schema (retrieve_schema, real mechanism,
no access to gold SQL) against the Perfect/oracle schema (extract_gold_schema,
computed from the gold SQL) for every one of the 1034 dev questions, and
reports:
  - Schema Linking Recall (SLR) and False Positive Rate (FPR), at both table
    level and column level -- of everything actually needed, how much did
    we find; of everything not needed, how much noise did we still include.
  - Schema reduction: average % of tables/columns retained relative to the
    full schema -- how much smaller is what we show the model, compared to
    showing everything.
Directly comparable to Table 8/9/13 in "Rethinking Schema Linking: A
Context-Aware Bidirectional Retrieval Approach for Text-to-SQL".
Input: generation_results_schema_linked.json.
Output: printed SLR/FPR/retention numbers.
"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from eval import process_sql

IN_FILE = "generation_results_schema_linked.json"


def safe_div(a, b):
    return a / b if b else float("nan")


def main():
    with open(IN_FILE) as f:
        rows = json.load(f)
    print("Total questions:", len(rows))

    schema_cache = {}
    col_tp = col_fn = col_fp = col_tn = 0
    tab_tp = tab_fn = tab_fp = tab_tn = 0
    tab_retention, col_retention = [], []

    for r in rows:
        db_id = r["db_id"]
        if db_id not in schema_cache:
            db_path = os.path.join("database", db_id, db_id + ".sqlite")
            schema_cache[db_id] = process_sql.get_schema(db_path)
        full_schema = schema_cache[db_id]

        all_tables = set(full_schema.keys())
        all_cols = {(t, c) for t, cols in full_schema.items() for c in cols}

        gold_tables = set(r["oracle_schema"].keys())
        gold_cols = {(t, c) for t, cols in r["oracle_schema"].items() for c in cols}

        ret_tables = set(r["retrieved_schema"].keys())
        ret_cols = {(t, c) for t, cols in r["retrieved_schema"].items() for c in cols}

        tab_tp += len(gold_tables & ret_tables)
        tab_fn += len(gold_tables - ret_tables)
        tab_fp += len(ret_tables - gold_tables)
        tab_tn += len((all_tables - gold_tables) - ret_tables)

        col_tp += len(gold_cols & ret_cols)
        col_fn += len(gold_cols - ret_cols)
        col_fp += len(ret_cols - gold_cols)
        col_tn += len((all_cols - gold_cols) - ret_cols)

        if all_tables:
            tab_retention.append(len(ret_tables) / len(all_tables))
        if all_cols:
            col_retention.append(len(ret_cols) / len(all_cols))

    print("\n=== Schema Linking Recall (SLR) / False Positive Rate (FPR) ===")
    print(f"Table level  -- SLR: {100*safe_div(tab_tp, tab_tp+tab_fn):.2f}%  FPR: {100*safe_div(tab_fp, tab_fp+tab_tn):.2f}%")
    print(f"Column level -- SLR: {100*safe_div(col_tp, col_tp+col_fn):.2f}%  FPR: {100*safe_div(col_fp, col_fp+col_tn):.2f}%")

    print("\n=== Schema Reduction (average % retained by Retrieved schema) ===")
    print(f"Tables retained:  {100*sum(tab_retention)/len(tab_retention):.2f}%")
    print(f"Columns retained: {100*sum(col_retention)/len(col_retention):.2f}%")


if __name__ == "__main__":
    main()
