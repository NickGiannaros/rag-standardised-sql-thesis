"""
2_standardise_sql.py
What this does: rewrites every raw SQL query in the training set into
Xander's Standardised SQL form (uppercase keywords, every clause spelled out
even when empty, aliases expanded to full table/column names), using
Xander's own sql_tree_to_SimpleSQL() function. A handful of queries
(self-joins) can't be expressed this way and get skipped.
Input: sample_train.json.
Output: sample_train_standardised.json (adds a "standardised_sql" field).
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
from eval import process_sql

with open("sample_train.json") as f:
    train_sample = json.load(f)

results = []
failed = 0
for i, ex in enumerate(train_sample):
    db_id = ex["db_id"]
    db_path = os.path.join("database", db_id, db_id + ".sqlite")
    try:
        schema = process_sql.Schema(process_sql.get_schema(db_path))
        query_ast = process_sql.get_sql(schema, ex["query"])
        standardised = process_sql.sql_tree_to_SimpleSQL(query_ast)
        results.append({"db_id": db_id, "question": ex["question"], "raw_sql": ex["query"], "standardised_sql": standardised})
    except Exception as e:
        failed += 1

print("")
print("Epitycha:", len(results), "/", len(train_sample))
print("Apetycan:", failed)

with open("sample_train_standardised.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved sample_train_standardised.json")