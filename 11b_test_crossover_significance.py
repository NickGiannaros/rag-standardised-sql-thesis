"""
11b_test_crossover_significance.py
What this does: follows up on the pattern from 11_analyze_by_difficulty.py
with exactly two McNemar tests -- one for the Low-complexity group, one for
Medium+High combined -- matching the two halves of the pattern we saw. We
picked just these two tests instead of testing every bucket combination, to
avoid the false-positive risk of testing many things and reporting whichever
one happens to look significant. Result: Low p = 0.1996, Medium+High p =
0.2529 -- the pattern doesn't hold up statistically, likely because these
bucket sizes are too small to detect it reliably (39-62 discordant pairs per
test). We report it as an open observation, not a confirmed finding.
Input: generation_results.json, generation_results_multi.json,
generation_results_remaining.json, generation_results_compact_B.json.
Output: printed McNemar p-value for each of the two buckets.
"""
import json, sqlite3, re, sys, os
sys.path.insert(0, os.getcwd())
from eval import process_sql
from statsmodels.stats.contingency_tables import mcnemar
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
    A_sql = clean_markdown(r["sql_memory_A"])
    Bnew_sql = clean_memory_B(new_by_key[key]["sql_memory_B_compact"])

    rows.append({
        "complexity": complexity_score(r["gold_sql"]),
        "A_ok": execute_query(db_path, A_sql) == gold_result,
        "Bnew_ok": execute_query(db_path, Bnew_sql) == gold_result,
    })

scores = [r["complexity"] for r in rows]
low_cut = np.percentile(scores, 33)
high_cut = np.percentile(scores, 66)

def bucket(score):
    return "Low" if score <= low_cut else ("Medium" if score <= high_cut else "High")

for r in rows:
    r["bucket"] = bucket(r["complexity"])

def mcnemar_test(label, sub):
    n = len(sub)
    A_only = sum(1 for r in sub if r["A_ok"] and not r["Bnew_ok"])
    B_only = sum(1 for r in sub if r["Bnew_ok"] and not r["A_ok"])
    both = sum(1 for r in sub if r["A_ok"] and r["Bnew_ok"])
    neither = sum(1 for r in sub if not r["A_ok"] and not r["Bnew_ok"])
    table = [[both, A_only], [B_only, neither]]
    result = mcnemar(table, exact=True)
    print(f"{label} (n={n}): A-only={A_only} | B-only={B_only} | McNemar p-value: {result.pvalue:.4f}")

low_rows = [r for r in rows if r["bucket"] == "Low"]
medhigh_rows = [r for r in rows if r["bucket"] in ("Medium", "High")]

print("A vs Memory B (compact) crossover test:")
mcnemar_test("Low complexity", low_rows)
mcnemar_test("Medium+High complexity", medhigh_rows)
