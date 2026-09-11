"""
20_generate_combined.py

WHAT THIS DOES
    Generates SQL under the two conditions that were never tested: the
    retrieved schema AND few-shot memory examples together in the same
    prompt, once with raw-SQL examples (Memory A) and once with Standardised
    SQL examples (Memory B, compact index).

WHY IT EXISTS
    The two mechanisms of this thesis were evaluated separately. Arranged as a
    grid they leave two cells empty:

                        no examples   raw examples   standardised examples
        Full schema        737            731              732
        Retrieved schema   743             ?                ?

    Whether two retrieval mechanisms combine, cancel or interfere is not
    answerable from separate experiments, and it is the first question a
    reader asks after seeing both. Filling the row completes a factorial
    design and lets the interaction be tested rather than assumed.

WHAT IS HELD CONSTANT
    Everything except the schema in the prompt:
      * the prompt template is the one used by 8d_generate_compact_B.py
      * the retrieval query text still uses the FULL schema string, exactly as
        in the memory-only conditions, so the retrieved neighbours are
        identical to those already measured by 17_. Changing it would move two
        variables at once and make the cell uninterpretable.
      * the retrieved schema is read from the stored output of
        13b_generate_schema_linked.py, so no schema linking is re-run and the
        four LLM calls per question it needs are not repeated.

COST
    2 conditions x 1034 questions = 2068 generation calls on gpt-4o-mini.
    Progress is saved every 25 questions and the script resumes where it
    stopped, so it can be interrupted safely.

INPUT   generation_results_schema_linked.json  (retrieved schemas)
        memory_metadata.pkl, memory_A_v2.index, memory_B_compact.index
        Spider dev set, database/
OUTPUT  generation_results_combined.json

HOW TO RUN
    python 20_generate_combined.py
"""

import json
import os
import pickle
import sys

sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv

load_dotenv()

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from eval import process_sql
from open_ai.chat_gpt import ask_chatGPT

SCHEMA_FILE = "generation_results_schema_linked.json"
OUT_FILE = "generation_results_combined.json"
K = 3
SAVE_EVERY = 25


def build_prompt(question, schema_str, examples, sql_field):
    """Identical to 8d_generate_compact_B.py -- only the schema_str differs."""
    prompt = ("You are a SQL expert. Given the database schema and question, "
              "write the correct SQLite query. Return ONLY the SQL query, "
              "nothing else.\n\n")
    prompt += "Schema: " + schema_str + "\n\n"
    if examples:
        prompt += "Here are similar examples:\n"
        for m in examples:
            prompt += "Question: " + m["question"] + "\nSQL: " + m[sql_field] + "\n\n"
    prompt += "Question: " + question + "\nSQL:"
    return prompt


def main():
    with open(SCHEMA_FILE) as f:
        linked = json.load(f)
    retrieved_by_key = {(r["db_id"], r["question"].strip()): r["retrieved_schema"]
                        for r in linked}
    print(f"Retrieved schemas available for {len(retrieved_by_key)} questions")

    with open("memory_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    index_A = faiss.read_index("memory_A_v2.index")
    index_B = faiss.read_index("memory_B_compact.index")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading Spider dev set...")
    spider = load_dataset("xlangai/spider")
    dev_full = list(spider["validation"])

    results = []
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            results = json.load(f)
        done = {(r["db_id"], r["question"].strip()) for r in results}
        print(f"Resuming -- {len(done)} questions already generated")

    schema_cache = {}
    missing_schema = 0

    for n, q in enumerate(dev_full, 1):
        db_id = q["db_id"]
        key = (db_id, q["question"].strip())
        if key in done:
            continue

        if db_id not in schema_cache:
            db_path = os.path.join("database", db_id, db_id + ".sqlite")
            schema_cache[db_id] = str(process_sql.get_schema(db_path))
        full_schema_str = schema_cache[db_id]

        retrieved = retrieved_by_key.get(key)
        if not retrieved:
            missing_schema += 1
            continue
        retrieved_str = str(retrieved)

        # Retrieval query text uses the FULL schema, as in the memory-only runs.
        query_text = "Question: " + q["question"] + " Schema: " + full_schema_str
        emb = np.array(model.encode([query_text], normalize_embeddings=True)).astype("float32")
        _, idx_A = index_A.search(emb, K)
        _, idx_B = index_B.search(emb, K)
        ex_A = [metadata[i] for i in idx_A[0]]
        ex_B = [metadata[i] for i in idx_B[0]]

        sql_A = ask_chatGPT(build_prompt(q["question"], retrieved_str, ex_A, "raw_sql"))
        sql_B = ask_chatGPT(build_prompt(q["question"], retrieved_str, ex_B, "standardised_sql"))

        results.append({
            "db_id": db_id,
            "question": q["question"],
            "gold_sql": q["query"],
            "sql_combined_A": sql_A,
            "sql_combined_B": sql_B,
        })

        if len(results) % SAVE_EVERY == 0:
            with open(OUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  saved -- {len(results)}/{len(dev_full)} done (last db: {db_id})")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)} questions written to {OUT_FILE}")
    if missing_schema:
        print(f"WARNING: {missing_schema} questions had no stored retrieved schema "
              f"and were skipped.")


if __name__ == "__main__":
    main()
