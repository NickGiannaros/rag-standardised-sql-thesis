"""
5_generate_single_db.py
What this does: the first, small pilot run. For each of the 45 dev questions
on concert_singer, retrieves the top-3 closest examples from Memory A and
Memory B separately, builds three prompts (no memory / Memory A / Memory B),
and asks gpt-4o-mini to answer each. At query time we only embed the
question and schema, not the SQL, since we don't know it yet.
Input: sample_dev.json (concert_singer only), memory_A.index, memory_B.index,
memory_metadata.pkl.
Output: generation_results.json (results/ folder).
"""
import sys, os, json, pickle
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()

from eval import process_sql
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from open_ai.chat_gpt import ask_chatGPT

with open("sample_dev.json") as f:
    dev_data = json.load(f)

with open("memory_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

index_A = faiss.read_index("memory_A.index")
index_B = faiss.read_index("memory_B.index")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

db_id = "concert_singer"
db_path = os.path.join("database", db_id, db_id + ".sqlite")
schema_str = str(process_sql.get_schema(db_path))

k = 3
test_questions = dev_data

def build_prompt(question, schema_str, examples, sql_field):
    prompt = "You are a SQL expert. Given the database schema and question, write the correct SQLite query. Return ONLY the SQL query, nothing else.\n\n"
    prompt += "Schema: " + schema_str + "\n\n"
    if examples:
        prompt += "Here are similar examples:\n"
        for m in examples:
            prompt += "Question: " + m["question"] + "\nSQL: " + m[sql_field] + "\n\n"
    prompt += "Question: " + question + "\nSQL:"
    return prompt

results = []

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

    results.append({
        "question": q["question"],
        "gold_sql": q["query"],
        "sql_baseline": sql_baseline,
        "sql_memory_A": sql_A,
        "sql_memory_B": sql_B,
    })

with open("generation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved generation_results.json\n")

for r in results:
    print("="*60)
    print("Question:", r["question"])
    print("Gold:    ", r["gold_sql"])
    print("Baseline:", r["sql_baseline"])
    print("Memory A:", r["sql_memory_A"])
    print("Memory B:", r["sql_memory_B"])