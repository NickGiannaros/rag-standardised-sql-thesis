"""
8d_generate_compact_B.py
What this does: regenerates SQL for the Memory B condition only, this time
retrieving from the compact index, to see if the retrieval-quality gain
actually improves execution accuracy. Baseline and Memory A don't depend on
Memory B's index, so we just reuse their earlier results -- keeps the
comparison to one changed variable and saves 2/3 of the API calls.
Input: memory_metadata.pkl, memory_B_compact.index, full Spider dev set.
Output: generation_results_compact_B.json -- saves progress per database, so
a partial run can pick back up (skips questions already done).
"""
import sys, os, json, pickle
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from eval import process_sql
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import faiss
from open_ai.chat_gpt import ask_chatGPT

with open("memory_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

index_B = faiss.read_index("memory_B_compact.index")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading full Spider dev set...")
spider = load_dataset("xlangai/spider")
dev_full = list(spider["validation"])

db_ids = ["concert_singer", "pets_1", "poker_player", "orchestra", "employee_hire_evaluation",
          "world_1", "car_1", "cre_Doc_Template_Mgt", "dog_kennels", "flight_2",
          "student_transcripts_tracking", "wta_1", "tvshow", "network_1",
          "course_teach", "singer", "museum_visit", "battle_death",
          "voter_1", "real_estate_properties"]
k = 3

def build_prompt(question, schema_str, examples, sql_field):
    prompt = "You are a SQL expert. Given the database schema and question, write the correct SQLite query. Return ONLY the SQL query, nothing else.\n\n"
    prompt += "Schema: " + schema_str + "\n\n"
    if examples:
        prompt += "Here are similar examples:\n"
        for m in examples:
            prompt += "Question: " + m["question"] + "\nSQL: " + m[sql_field] + "\n\n"
    prompt += "Question: " + question + "\nSQL:"
    return prompt

out_file = "generation_results_compact_B.json"
all_results = []
done_keys = set()
if os.path.exists(out_file):
    with open(out_file) as f:
        all_results = json.load(f)
    done_keys = set((r["db_id"], r["question"]) for r in all_results)
    print("Resuming, already have:", len(all_results))

for db_id in db_ids:
    print("\n=== Database:", db_id, "===")
    db_path = os.path.join("database", db_id, db_id + ".sqlite")
    schema_str = str(process_sql.get_schema(db_path))
    test_questions = [q for q in dev_full if q["db_id"] == db_id]
    print("Questions:", len(test_questions))

    for q in test_questions:
        if (db_id, q["question"]) in done_keys:
            continue
        query_text = "Question: " + q["question"] + " Schema: " + schema_str
        query_emb = model.encode([query_text], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        _, idx_B = index_B.search(query_emb, k)
        examples_B = [metadata[i] for i in idx_B[0]]

        prompt_B = build_prompt(q["question"], schema_str, examples_B, "standardised_sql")
        sql_B = ask_chatGPT(prompt_B)

        all_results.append({
            "db_id": db_id,
            "question": q["question"],
            "gold_sql": q["query"],
            "sql_memory_B_compact": sql_B,
        })

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved progress after", db_id, "-- total so far:", len(all_results))

print("\nDone. Total questions:", len(all_results))
