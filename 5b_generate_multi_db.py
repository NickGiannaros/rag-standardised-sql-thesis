"""
5b_generate_multi_db.py
What this does: extends the pilot to 4 more databases (pets_1, poker_player,
orchestra, employee_hire_evaluation -- 160 more questions), so the pilot
result isn't based on a single database. Same prompts and retrieval logic as
5_generate_single_db.py.
Input: full Spider dev set (HuggingFace 'xlangai/spider'), memory_A.index,
memory_B.index, memory_metadata.pkl.
Output: generation_results_multi.json (results/ folder) -- saves progress
after each database, so a partial run can pick back up.
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

index_A = faiss.read_index("memory_A.index")
index_B = faiss.read_index("memory_B.index")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading full Spider dev set...")
spider = load_dataset("xlangai/spider")
dev_full = list(spider["validation"])

db_ids = ["pets_1", "poker_player", "orchestra", "employee_hire_evaluation"]
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

all_results = []

for db_id in db_ids:
    print("\n=== Database:", db_id, "===")
    db_path = os.path.join("database", db_id, db_id + ".sqlite")
    schema_str = str(process_sql.get_schema(db_path))
    test_questions = [q for q in dev_full if q["db_id"] == db_id]
    print("Questions:", len(test_questions))

    for q in test_questions:
        print("Processing:", q["question"])
        query_text = "Question: " + q["question"] + " Schema: " + schema_str
        query_emb = model.encode([query_text], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        scores_A, idx_A = index_A.search(query_emb, k)
        scores_B, idx_B = index_B.search(query_emb, k)
        examples_A = [metadata[i] for i in idx_A[0]]
        examples_B = [metadata[i] for i in idx_B[0]]

        prompt_baseline = build_prompt(q["question"], schema_str, [], "raw_sql")
        prompt_A = build_prompt(q["question"], schema_str, examples_A, "raw_sql")
        prompt_B = build_prompt(q["question"], schema_str, examples_B, "standardised_sql")

        sql_baseline = ask_chatGPT(prompt_baseline)
        sql_A = ask_chatGPT(prompt_A)
        sql_B = ask_chatGPT(prompt_B)

        all_results.append({
            "db_id": db_id,
            "question": q["question"],
            "gold_sql": q["query"],
            "sql_baseline": sql_baseline,
            "sql_memory_A": sql_A,
            "sql_memory_B": sql_B,
        })

    with open("generation_results_multi.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved progress after", db_id)

print("\nDone. Total questions:", len(all_results))