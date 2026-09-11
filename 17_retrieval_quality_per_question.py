"""
17_retrieval_quality_per_question.py

WHAT THIS DOES
    Re-runs the memory retrieval for all 1034 dev questions and saves, PER
    QUESTION, whether the retrieved few-shot examples were structurally
    relevant. Same retrieval, same indices, same skeleton-matching rule as
    7_ / 8b_measure_retrieval_quality*.py -- only the per-question outcome is
    now persisted instead of being aggregated and discarded.

WHY IT EXISTS
    The published retrieval-quality numbers (Memory A 20.8% top-1, Memory B
    compact 20.4% top-1) are averages. They tell us that most retrieved
    examples are structurally irrelevant, but not which questions those were,
    so we cannot ask the obvious follow-up: does the memory help on the
    questions where it retrieved something genuinely relevant?

    That question matters because an effect confined to ~20% of questions is
    diluted roughly fivefold when averaged over all of them, and would be
    invisible in the overall accuracy figures. This script produces the flag
    that 18_stratified_memory_analysis.py needs to test it.

    No API calls: retrieval is local FAISS search over indices that already
    exist. Nothing is re-generated.

NOTE ON THE MATCH CRITERION
    "Skeleton match" means the retrieved example's SQL uses exactly the same
    set of structural features as the gold query: presence of WHERE, GROUP BY,
    HAVING, ORDER BY, LIMIT, JOIN, DISTINCT, and the same set of aggregate
    functions. NONE placeholders are stripped before the check so that
    Standardised SQL is not penalised for its own boilerplate.

INPUT   memory_metadata.pkl, memory_A_v2.index, memory_B_compact.index,
        Spider dev set, database/
OUTPUT  results/retrieval_quality_per_question.csv

HOW TO RUN
    python 17_retrieval_quality_per_question.py
    (a few minutes on CPU -- the embedding model runs over 1034 questions)
"""

import csv
import os
import pickle
import re
import sys

sys.path.insert(0, os.getcwd())

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from eval import process_sql

OUT_FILE = os.path.join("results", "retrieval_quality_per_question.csv")
K = 3
AGG_FUNCS = ["count", "avg", "sum", "max", "min"]


def extract_skeleton(sql):
    """Identical to 7_/8b_: the set of structural features a query uses."""
    s = sql.lower()
    for clause in ["where", "group by", "having", "order by", "limit",
                   "intersect", "except", "union"]:
        s = re.sub(r"\b" + re.escape(clause) + r"\s+none\b", "", s)
    return (
        bool(re.search(r"\bwhere\b", s)),
        bool(re.search(r"\bgroup by\b", s)),
        bool(re.search(r"\bhaving\b", s)),
        bool(re.search(r"\border by\b", s)),
        bool(re.search(r"\blimit\b", s)),
        bool(re.search(r"\bjoin\b", s)),
        bool(re.search(r"\bdistinct\b", s)),
        frozenset(f for f in AGG_FUNCS if re.search(r"\b" + f + r"\s*\(", s)),
    )


def main():
    with open("memory_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    index_A = faiss.read_index("memory_A_v2.index")
    index_B = faiss.read_index("memory_B_compact.index")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading Spider dev set...")
    spider = load_dataset("xlangai/spider")
    dev_full = list(spider["validation"])
    print("Dev questions:", len(dev_full))

    rows = []
    skipped = 0
    for n, q in enumerate(dev_full, 1):
        db_id = q["db_id"]
        db_path = os.path.join("database", db_id, db_id + ".sqlite")
        try:
            schema_str = str(process_sql.get_schema(db_path))
        except Exception:
            skipped += 1
            continue

        gold_skel = extract_skeleton(q["query"])
        query_text = "Question: " + q["question"] + " Schema: " + schema_str
        emb = np.array(model.encode([query_text], normalize_embeddings=True)).astype("float32")

        _, idx_A = index_A.search(emb, K)
        _, idx_B = index_B.search(emb, K)
        ex_A = [metadata[i] for i in idx_A[0]]
        ex_B = [metadata[i] for i in idx_B[0]]

        skels_A = [extract_skeleton(m["raw_sql"]) for m in ex_A]
        skels_B = [extract_skeleton(m["standardised_sql"]) for m in ex_B]

        rows.append({
            "db_id": db_id,
            "question": q["question"].strip(),
            "A_top1_match": skels_A[0] == gold_skel,
            "A_top3_match": any(s == gold_skel for s in skels_A),
            "B_top1_match": skels_B[0] == gold_skel,
            "B_top3_match": any(s == gold_skel for s in skels_B),
            "A_neighbour_ids": "|".join(str(i) for i in idx_A[0]),
            "B_neighbour_ids": "|".join(str(i) for i in idx_B[0]),
        })

        if n % 200 == 0:
            print(f"  ...{n}/{len(dev_full)}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    print(f"\nWrote {OUT_FILE} ({total} rows, {skipped} skipped)")
    print("\n=== VERIFICATION -- must match the published retrieval-quality numbers ===")
    checks = [("Memory A (v2)      top-1", "A_top1_match", 215),
              ("Memory B (compact) top-1", "B_top1_match", 211),
              ("Memory A (v2)      top-3", "A_top3_match", 331),
              ("Memory B (compact) top-3", "B_top3_match", 336)]
    ok_all = True
    for label, col, expected in checks:
        got = sum(1 for r in rows if r[col])
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok_all = False
        print(f"{label:<28}{got:>6}/{total}  ({100 * got / total:.1f}%)   expected {expected}   {status}")
    print("\nRESULT:", "retrieval reproduces exactly."
          if ok_all else "does NOT reproduce -- stop and investigate.")


if __name__ == "__main__":
    main()
