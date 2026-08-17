"""
11_analyze_by_difficulty.py
What this does: an exploratory check -- does the raw-vs-Standardised effect
depend on how complex the query is? The idea: the NONE-boilerplate takes up
proportionally less space in a complex query, so maybe it hurts less (or
even helps) there. Scores each gold query 0-7 based on which clauses/
features it uses, and splits questions into three difficulty groups
(Low/Medium/High). Result: a visible pattern shows up (raw SQL ahead on easy
questions, Standardised SQL ahead on medium/hard ones) -- but see
11b_test_crossover_significance.py, it doesn't hold up statistically.
Input: generation_results.json, generation_results_multi.json,
generation_results_remaining.json, generation_results_compact_B.json.
Output: printed complexity distribution + per-bucket execution accuracy table.
"""
import json, sqlite3, re, sys, os
sys.path.insert(0, os.getcwd())
from eval import process_sql
import numpy as np

AGG_FUNCS = ["count", "avg", "sum", "max", "min"]

def clean_markdown(sql):
    sql = re.sub(r"```sql\s*", "", sql)
    sql = re.sub(r"```", "", sql)
    return sql.strip()

def clean_memory_B(sql):
    sql = clean_markdown(sql)
    if "NONE" in sql:
        try:
            cleaned = process_sql.SimpleSQL_to_SQL(sql)
            if cleaned and cleaned != "SELECT ERROR":
                return cleaned
        except Exception:
            pass
    return sql

def execute_query(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return sorted(rows)
    except Exception as e:
        return "ERROR: " + str(e)

def complexity_score(sql):
    s = sql.lower()
    score = 0
    for pattern in [r"\bwhere\b", r"\bgroup by\b", r"\bhaving\b", r"\border by\b",
                     r"\blimit\b", r"\bjoin\b", r"\bdistinct\b"]:
        if re.search(pattern, s):
            score += 1
    if any(re.search(r"\b" + f + r"\s*\(", s) for f in AGG_FUNCS):
        score += 1
    if s.count("select") > 1:
        score += 1
    if re.search(r"\b(intersect|except|union)\b", s):
        score += 1
    return score

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

old_by_key = {(r["db_id"], r["question"]): r for r in all_raw}

with open("generation_results_compact_B.json") as f:
    new_B = json.load(f)
new_by_key = {(r["db_id"], r["question"]): r for r in new_B}

rows = []
for key, r in old_by_key.items():
    if key not in new_by_key:
        continue
    db_path = os.path.join("database", r["db_id"], r["db_id"] + ".sqlite")
    gold_result = execute_query(db_path, r["gold_sql"])
    baseline_sql = clean_markdown(r["sql_baseline"])
    A_sql = clean_markdown(r["sql_memory_A"])
    Bold_sql = clean_memory_B(r["sql_memory_B"])
    Bnew_sql = clean_memory_B(new_by_key[key]["sql_memory_B_compact"])

    rows.append({
        "complexity": complexity_score(r["gold_sql"]),
        "baseline_ok": execute_query(db_path, baseline_sql) == gold_result,
        "A_ok": execute_query(db_path, A_sql) == gold_result,
        "Bold_ok": execute_query(db_path, Bold_sql) == gold_result,
        "Bnew_ok": execute_query(db_path, Bnew_sql) == gold_result,
    })

scores = [r["complexity"] for r in rows]
low_cut = np.percentile(scores, 33)
high_cut = np.percentile(scores, 66)
print("Complexity score distribution -- min:", min(scores), "max:", max(scores),
      "| tercile cuts: low<=%.1f, high>%.1f" % (low_cut, high_cut))

def bucket(score):
    if score <= low_cut:
        return "Low"
    elif score <= high_cut:
        return "Medium"
    else:
        return "High"

for r in rows:
    r["bucket"] = bucket(r["complexity"])

print("\nPer difficulty bucket:")
for b in ["Low", "Medium", "High"]:
    sub = [r for r in rows if r["bucket"] == b]
    n = len(sub)
    if n == 0:
        continue
    print(f"{b} (n={n}): Baseline {sum(r['baseline_ok'] for r in sub)}/{n} "
          f"({100*sum(r['baseline_ok'] for r in sub)/n:.1f}%) | "
          f"A {sum(r['A_ok'] for r in sub)}/{n} ({100*sum(r['A_ok'] for r in sub)/n:.1f}%) | "
          f"B_old {sum(r['Bold_ok'] for r in sub)}/{n} ({100*sum(r['Bold_ok'] for r in sub)/n:.1f}%) | "
          f"B_new {sum(r['Bnew_ok'] for r in sub)}/{n} ({100*sum(r['Bnew_ok'] for r in sub)/n:.1f}%)")
