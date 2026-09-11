"""
14_build_master_results.py

WHAT THIS DOES
    Builds ONE table (results/master_results.csv) with one row per Spider dev
    question and one column per experimental condition, holding the per-question
    execution-correctness flag (True/False).

WHY IT EXISTS
    Until now every evaluation script (6d, 8e, 11, 13d) re-executed all 1034
    questions from scratch, printed a summary, and threw the per-question
    outcome away. That makes any further analysis (statistical power,
    stratification, error taxonomy, factorial tests) impossible without
    re-running everything. This script computes those flags ONCE and persists
    them, so every later analysis is a cheap read of a CSV.

    It changes no experiment and no number. It is a reorganisation of results
    that already exist.

INPUT (found automatically under ./ or ./results/)
    generation_results[_single_db].json        45 rows, concert_singer pilot
    generation_results_multi[_db].json        160 rows
    generation_results_remaining[_dbs].json   829 rows
    generation_results_compact_B.json        1034 rows
    generation_results_schema_linked.json    1034 rows
    database/<db_id>/<db_id>.sqlite          Spider databases

OUTPUT
    results/master_results.csv               1034 rows
    plus a printed verification block that must reproduce the already-published
    numbers: Baseline 737, Memory A 731, Memory B 733, Memory B compact 732,
    Retrieved 743, Perfect 778.

HOW TO RUN
    Run it from the working directory that contains `database/` and `eval/`
    (i.e. the Xander folder):
        python 14_build_master_results.py
"""

import csv
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------- SQL cleaning
# Identical to the helpers used in 6d / 8e / 11 / 13d, so the flags this script
# writes are the same ones those scripts computed on the fly.

try:
    from eval import process_sql
    HAVE_PROCESS_SQL = True
except Exception:
    HAVE_PROCESS_SQL = False
    print("WARNING: could not import eval.process_sql -- Memory B's 'NONE' "
          "de-standardisation will be skipped and the B numbers will be wrong. "
          "Run this script from the Xander working directory.")


def clean_markdown(sql):
    sql = re.sub(r"```sql\s*", "", sql or "")
    sql = re.sub(r"```", "", sql)
    return sql.strip()


def clean_memory_B(sql):
    """Memory B answers may come back in Standardised SQL (with NONE clauses);
    convert them back to executable SQL first."""
    sql = clean_markdown(sql)
    if "NONE" in sql and HAVE_PROCESS_SQL:
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
        cur = conn.cursor()
        cur.execute(sql)
        rows = sorted(cur.fetchall())
        conn.close()
        return rows
    except Exception as e:
        return "ERROR: " + str(e)


# ------------------------------------------------------------ complexity score
# Identical to 11_analyze_by_difficulty.py so difficulty buckets stay comparable.

AGG_FUNCS = ["count", "avg", "sum", "max", "min"]


def complexity_score(sql):
    s = (sql or "").lower()
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


# ------------------------------------------------------------------ file lookup

def find_file(*candidates):
    for name in candidates:
        for folder in ("", "results"):
            p = os.path.join(folder, name) if folder else name
            if os.path.exists(p):
                return p
    raise FileNotFoundError("none of these were found: " + ", ".join(candidates))


def load_json(*candidates):
    path = find_file(*candidates)
    print("  loaded", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------- schema helpers

def norm_schema(schema):
    """{'table': ['col', ...]} -> {'table_lower': set(col_lower)}"""
    out = {}
    for t, cols in (schema or {}).items():
        out[t.lower()] = set(c.lower() for c in (cols or []))
    return out


def schema_recall_complete(oracle, retrieved):
    """True when every table and column the gold query needs is present in the
    retrieved schema (i.e. schema linking lost nothing for this question)."""
    o, r = norm_schema(oracle), norm_schema(retrieved)
    for t, cols in o.items():
        if t not in r:
            return False
        if not cols.issubset(r[t]):
            return False
    return True


def count_cols(schema):
    return sum(len(c or []) for c in (schema or {}).values())


# -------------------------------------------------------------------- main

def main():
    print("Loading generation results...")
    single = load_json("generation_results.json", "generation_results_single_db.json")
    for r in single:
        r.setdefault("db_id", "concert_singer")
    multi = load_json("generation_results_multi.json", "generation_results_multi_db.json")
    remaining = load_json("generation_results_remaining.json", "generation_results_remaining_dbs.json")
    compact = load_json("generation_results_compact_B.json")
    linked = load_json("generation_results_schema_linked.json")

    mem = single + multi + remaining
    key = lambda r: (r["db_id"], r["question"].strip())
    mem_by_key = {key(r): r for r in mem}
    compact_by_key = {key(r): r for r in compact}
    linked_by_key = {key(r): r for r in linked}

    keys = sorted(mem_by_key)
    print(f"\nQuestions: mem={len(mem_by_key)} compact={len(compact_by_key)} "
          f"linked={len(linked_by_key)} | joined on all three = "
          f"{len([k for k in keys if k in compact_by_key and k in linked_by_key])}")

    rows = []
    for qid, k in enumerate(keys):
        m = mem_by_key[k]
        c = compact_by_key.get(k)
        l = linked_by_key.get(k)
        db_id = m["db_id"]
        db_path = os.path.join("database", db_id, db_id + ".sqlite")

        gold_sql = m["gold_sql"]
        gold = execute_query(db_path, gold_sql)

        sql = {
            "baseline": clean_markdown(m["sql_baseline"]),
            "memA": clean_markdown(m["sql_memory_A"]),
            "memB": clean_memory_B(m["sql_memory_B"]),
            "memB_compact": clean_memory_B(c["sql_memory_B_compact"]) if c else "",
            "retrieved": clean_markdown(l["sql_retrieved_schema"]) if l else "",
            "perfect": clean_markdown(l["sql_perfect_schema"]) if l else "",
        }
        ok = {name: (execute_query(db_path, s) == gold if s else "")
              for name, s in sql.items()}

        oracle_schema = (l or {}).get("oracle_schema") or {}
        retrieved_schema = (l or {}).get("retrieved_schema") or {}

        score = complexity_score(gold_sql)
        rows.append({
            "qid": qid,
            "db_id": db_id,
            "question": m["question"].strip(),
            "gold_sql": gold_sql,
            "gold_executes": not (isinstance(gold, str) and gold.startswith("ERROR")),
            "complexity": score,
            # correctness flags -- the whole point of this file
            "baseline_ok": ok["baseline"],
            "memA_ok": ok["memA"],
            "memB_ok": ok["memB"],
            "memB_compact_ok": ok["memB_compact"],
            "retrieved_ok": ok["retrieved"],
            "perfect_ok": ok["perfect"],
            # schema-linking bookkeeping (used later to decompose Retrieved vs Perfect)
            "sl_recall_complete": schema_recall_complete(oracle_schema, retrieved_schema),
            "sl_tables_oracle": len(oracle_schema),
            "sl_tables_retrieved": len(retrieved_schema),
            "sl_cols_oracle": count_cols(oracle_schema),
            "sl_cols_retrieved": count_cols(retrieved_schema),
            # predicted SQL kept so error analysis needs no re-join
            "sql_baseline": sql["baseline"],
            "sql_memA": sql["memA"],
            "sql_memB": sql["memB"],
            "sql_memB_compact": sql["memB_compact"],
            "sql_retrieved": sql["retrieved"],
            "sql_perfect": sql["perfect"],
        })

        if (qid + 1) % 200 == 0:
            print(f"  ...{qid + 1}/{len(keys)}")

    # difficulty buckets (terciles of the complexity score, same rule as script 11)
    scores = sorted(r["complexity"] for r in rows)
    low_cut = scores[int(0.33 * (len(scores) - 1))]
    high_cut = scores[int(0.66 * (len(scores) - 1))]
    for r in rows:
        r["difficulty"] = ("Low" if r["complexity"] <= low_cut
                           else "Medium" if r["complexity"] <= high_cut else "High")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "master_results.csv")
    fields = ["qid", "db_id", "question", "gold_sql", "gold_executes",
              "complexity", "difficulty",
              "baseline_ok", "memA_ok", "memB_ok", "memB_compact_ok",
              "retrieved_ok", "perfect_ok",
              "sl_recall_complete", "sl_tables_oracle", "sl_tables_retrieved",
              "sl_cols_oracle", "sl_cols_retrieved",
              "sql_baseline", "sql_memA", "sql_memB", "sql_memB_compact",
              "sql_retrieved", "sql_perfect"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ verification
    n = len(rows)
    print(f"\nWrote {out_path} ({n} rows)")
    print("\n=== VERIFICATION -- these must match the already-published numbers ===")
    expected = {"baseline_ok": 737, "memA_ok": 731, "memB_ok": 733,
                "memB_compact_ok": 732, "retrieved_ok": 743, "perfect_ok": 778}
    label = {"baseline_ok": "Baseline (full schema)", "memA_ok": "Memory A (raw)",
             "memB_ok": "Memory B (standardised)", "memB_compact_ok": "Memory B (compact)",
             "retrieved_ok": "Retrieved schema", "perfect_ok": "Perfect schema"}
    all_ok = True
    print(f"{'condition':<28}{'got':>8}{'expected':>10}{'':>4}")
    for col, exp in expected.items():
        got = sum(1 for r in rows if r[col] is True)
        mark = "OK" if got == exp else "MISMATCH"
        if got != exp:
            all_ok = False
        print(f"{label[col]:<28}{got:>8}{exp:>10}   {mark}")

    bad_gold = sum(1 for r in rows if not r["gold_executes"])
    if bad_gold:
        print(f"\nNOTE: {bad_gold} gold queries failed to execute.")

    complete = sum(1 for r in rows if r["sl_recall_complete"])
    print(f"\nSchema linking kept everything the gold query needed on "
          f"{complete}/{n} questions ({100 * complete / n:.1f}%).")

    print("\nRESULT:", "all six conditions reproduce exactly."
          if all_ok else "at least one condition does NOT reproduce -- stop and investigate.")


if __name__ == "__main__":
    main()
