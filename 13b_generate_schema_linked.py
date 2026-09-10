"""
13b_generate_schema_linked.py
What this does: for the same 1034-question dev set used everywhere else in
this project, generates SQL under two new schema conditions on top of the
Full-schema baseline we already have:
  - Perfect: the oracle schema extracted straight from the gold SQL
    (extract_gold_schema) -- the theoretical best case, what if schema
    retrieval were flawless?
  - Retrieved: the real bidirectional schema linking mechanism
    (retrieve_schema, 13_schema_linking.py) -- keyword extraction, question
    decomposition, Table-First retrieval, Column-First retrieval, unioned.

Both use the exact same zero-shot, no-few-shot prompt template as the
Full-schema baseline (build_prompt, no examples) so schema is the only
variable that changes between conditions -- isolates the schema-linking
effect from everything else in the pipeline.

Input: generation_results.json, generation_results_multi.json,
generation_results_remaining.json (already have question/db_id/gold_sql/
sql_baseline -- gold_sql is reused only to compute the oracle schema, never
leaked into what SQL gets generated). Same flat filenames the earlier
generation scripts (5_/5b_/5c_) produce when run in sequence inside Xander/.
Output: generation_results_schema_linked.json -- per question: sql_baseline
(Full, carried over), sql_perfect_schema, sql_retrieved_schema,
oracle_schema, retrieved_schema. Copy the finished file into results/ once
done, same as the other generation outputs in this repo.

Note: this is 5 LLM calls per question for the Retrieved condition alone
(keyword extraction, subquestion decomposition, Table-First, Column-First,
final SQL generation) plus 1 for Perfect -- expect this to take a while and
cost noticeably more than a single-condition run.
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from eval import process_sql
from open_ai.chat_gpt import ask_chatGPT
from importlib import import_module
schema_linking = import_module("13_schema_linking")
extract_gold_schema = schema_linking.extract_gold_schema
retrieve_schema = schema_linking.retrieve_schema

OUT_FILE = "generation_results_schema_linked.json"


def build_prompt(question, schema_str):
    prompt = "You are a SQL expert. Given the database schema and question, write the correct SQLite query. Return ONLY the SQL query, nothing else.\n\n"
    prompt += "Schema: " + schema_str + "\n\n"
    prompt += "Question: " + question + "\nSQL:"
    return prompt


def load_all_raw():
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
    return all_raw


def main():
    all_raw = load_all_raw()
    print("Total questions loaded:", len(all_raw))

    all_results = []
    done_keys = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            all_results = json.load(f)
        done_keys = {(r["db_id"], r["question"]) for r in all_results}
        print("Resuming, already have:", len(done_keys))

    schema_cache = {}

    for i, r in enumerate(all_raw):
        key = (r["db_id"], r["question"])
        if key in done_keys:
            continue

        db_id = r["db_id"]
        if db_id not in schema_cache:
            db_path = os.path.join("database", db_id, db_id + ".sqlite")
            schema_cache[db_id] = process_sql.get_schema(db_path)
        schema = schema_cache[db_id]

        oracle_schema = extract_gold_schema(r["gold_sql"], schema)
        oracle_schema_str = str({t: sorted(c) for t, c in oracle_schema.items()}) if oracle_schema else str(schema)
        sql_perfect = ask_chatGPT(build_prompt(r["question"], oracle_schema_str))

        retrieved = retrieve_schema(r["question"], schema)
        if not retrieved:
            retrieved = schema  # fallback if all 4 retrieval calls returned nothing usable
        sql_retrieved = ask_chatGPT(build_prompt(r["question"], str(retrieved)))

        all_results.append({
            "db_id": db_id,
            "question": r["question"],
            "gold_sql": r["gold_sql"],
            "sql_baseline": r["sql_baseline"],
            "sql_perfect_schema": sql_perfect,
            "sql_retrieved_schema": sql_retrieved,
            "oracle_schema": {t: sorted(c) for t, c in oracle_schema.items()},
            "retrieved_schema": retrieved,
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(all_raw):
            with open(OUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Progress: {len(all_results)} done (last db: {db_id})")

    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nDone. Total questions:", len(all_results))


if __name__ == "__main__":
    main()
