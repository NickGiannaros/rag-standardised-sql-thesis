"""
3_build_texts.py
What this does: builds the text that will get embedded for Memory A (raw
SQL) and Memory B (Standardised SQL) -- one string per training example, in
the form "Question: {q} Schema: {schema} SQL: {sql}". At query time we only
embed "Question + Schema" (see 5_generate_single_db.py), so this is the
"half" of the design that includes the answer.
Input: sample_train_standardised.json.
Output: texts_cache.pkl (not committed -- regenerate by rerunning this script).
"""
import sys, os, json, pickle
sys.path.insert(0, os.getcwd())
from eval import process_sql

with open("sample_train_standardised.json") as f:
    data = json.load(f)

schema_cache = {}
def get_schema_str(db_id):
    if db_id not in schema_cache:
        db_path = os.path.join("database", db_id, db_id + ".sqlite")
        schema_cache[db_id] = str(process_sql.get_schema(db_path))
    return schema_cache[db_id]

texts_A = []
texts_B = []
metadata = []

for ex in data:
    schema_str = get_schema_str(ex["db_id"])
    text_A = "Question: " + ex["question"] + " Schema: " + schema_str + " SQL: " + ex["raw_sql"]
    text_B = "Question: " + ex["question"] + " Schema: " + schema_str + " SQL: " + ex["standardised_sql"]
    texts_A.append(text_A)
    texts_B.append(text_B)
    metadata.append({"question": ex["question"], "db_id": ex["db_id"], "raw_sql": ex["raw_sql"], "standardised_sql": ex["standardised_sql"]})

print("Built texts_A:", len(texts_A))
print("Built texts_B:", len(texts_B))
print("Unique schemas cached:", len(schema_cache))

with open("texts_cache.pkl", "wb") as f:
    pickle.dump({"texts_A": texts_A, "texts_B": texts_B, "metadata": metadata}, f)

print("Saved texts_cache.pkl")