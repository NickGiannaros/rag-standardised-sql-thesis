"""
22_worked_example.py

WHAT THIS DOES
    Runs the schema-linking mechanism on ONE question and prints every
    intermediate step: the full schema the model would otherwise receive, the
    keywords it extracts, the sub-questions it decomposes into, what
    Table-First selects, what Column-First selects, the union of the two, and
    the oracle schema for comparison.

WHY IT EXISTS
    The mechanism is four LLM calls whose intermediate outputs are discarded
    once the final schema is assembled. Nothing in the stored results shows
    what actually happens inside it, which makes the whole component hard to
    describe and impossible to verify by reading. This script produces a
    single fully documented trace, to be quoted verbatim in the write-up.

COST
    4 LLM calls (one question). Nothing else is re-run.

INPUT   database/, and the schema-linking module
OUTPUT  printed trace
        results/worked_example.txt   (the same trace, for pasting)
        results/worked_example.json  (structured, for reference)

HOW TO RUN
    Run from the working directory that contains database/ and eval/:
        python 22_worked_example.py
    Optionally pick another question:
        python 22_worked_example.py "concert_singer" "Some other question?"
"""

import io
import json
import os
import sys
from importlib import import_module

sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv

load_dotenv()

from eval import process_sql

DEFAULT_DB = "concert_singer"
DEFAULT_QUESTION = "Show the stadium name and the number of concerts in each stadium."
SCHEMA_FILE = "generation_results_schema_linked.json"


CANDIDATE_MODULES = ("13_schema_linking", "llm_schema_linking", "schema_linking")

# The working directory keeps the mechanism split across two files: the LLM
# retrieval lives in llm_schema_linking.py with an _llm suffix on every name,
# while extract_gold_schema lives in schema_linking.py alongside the earlier
# embedding-only attempt. The clean repository merges them into one module with
# the suffix dropped. Each function is therefore resolved independently, and the
# script reports which file each one came from.
WANTED = {
    "extract_keywords":    ("extract_keywords", "extract_keywords_llm"),
    "decompose_question":  ("decompose_question", "decompose_question_llm"),
    "retrieve_tables":     ("retrieve_tables", "retrieve_tables_llm"),
    "retrieve_columns":    ("retrieve_columns", "retrieve_columns_llm"),
    "extract_gold_schema": ("extract_gold_schema",),
}


def load_schema_linking():
    """Returns (functions dict, provenance dict)."""
    loaded = {}
    for name in CANDIDATE_MODULES:
        try:
            loaded[name] = import_module(name)
        except Exception:
            continue
    if not loaded:
        raise SystemExit("Δεν βρέθηκε κανένα από τα modules: " + ", ".join(CANDIDATE_MODULES))

    funcs, source = {}, {}
    for key, aliases in WANTED.items():
        for mod_name, mod in loaded.items():
            for alias in aliases:
                if hasattr(mod, alias):
                    funcs[key] = getattr(mod, alias)
                    source[key] = f"{mod_name}.{alias}"
                    break
            if key in funcs:
                break
    missing = [k for k in WANTED if k not in funcs]
    if missing:
        raise SystemExit(
            "Λείπουν συναρτήσεις: " + ", ".join(missing)
            + "\nModules που φορτώθηκαν: " + ", ".join(loaded))
    return funcs, source


def fmt_schema(schema, indent="    "):
    if not schema:
        return indent + "(κενό)"
    lines = []
    for t in sorted(schema):
        cols = schema[t]
        cols = sorted(cols) if cols else []
        lines.append(f"{indent}{t} ({len(cols)}): " + (", ".join(cols) if cols else "—"))
    return "\n".join(lines)


def count_cols(schema):
    return sum(len(v or []) for v in (schema or {}).values())


def main():
    db_id = sys.argv[1] if len(sys.argv) > 2 else DEFAULT_DB
    question = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION

    fn, source = load_schema_linking()

    db_path = os.path.join("database", db_id, db_id + ".sqlite")
    full_schema = process_sql.get_schema(db_path)

    gold_sql = None
    stored_retrieved = None
    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE) as f:
            for r in json.load(f):
                if r["db_id"] == db_id and r["question"].strip() == question.strip():
                    gold_sql = r["gold_sql"]
                    stored_retrieved = r.get("retrieved_schema")
                    break

    out = io.StringIO()

    def say(s=""):
        print(s)
        out.write(s + "\n")

    say("=" * 78)
    say("ΠΑΡΑΔΕΙΓΜΑ ΛΕΙΤΟΥΡΓΙΑΣ ΤΟΥ ΜΗΧΑΝΙΣΜΟΥ SCHEMA LINKING")
    say("=" * 78)
    say("\nΠροέλευση κάθε βήματος:")
    for k in ["extract_keywords", "decompose_question", "retrieve_tables",
              "retrieve_columns", "extract_gold_schema"]:
        say(f"  {k:<22} <- {source[k]}")
    say("=" * 78)
    say(f"\nΒάση:    {db_id}")
    say(f"Ερώτηση: {question}")
    if gold_sql:
        say(f"Gold SQL: {' '.join(gold_sql.split())}")

    say(f"\n\n--- ΑΦΕΤΗΡΙΑ: το πλήρες schema ({len(full_schema)} πίνακες, "
        f"{count_cols(full_schema)} στήλες) ---")
    say("Αυτό είναι που έβλεπε το μοντέλο σε κάθε ερώτηση πριν το schema linking.\n")
    say(fmt_schema(full_schema))

    say("\n\n--- ΒΗΜΑ 1: εξαγωγή λέξεων-κλειδιών ---")
    keywords = fn['extract_keywords'](question)
    say(json.dumps(keywords, ensure_ascii=False, indent=2))

    say("\n\n--- ΒΗΜΑ 2: διάσπαση σε υπο-ερωτήσεις ---")
    subquestions = fn['decompose_question'](question)
    say(json.dumps(subquestions, ensure_ascii=False, indent=2))

    say("\n\n--- ΒΗΜΑ 3: Table-First (επιλογή ολόκληρων πινάκων) ---")
    table_first = fn['retrieve_tables'](question, subquestions, keywords, full_schema)
    say(f"{len(table_first)} πίνακες, {count_cols(table_first)} στήλες")
    say(fmt_schema(table_first))

    say("\n\n--- ΒΗΜΑ 4: Column-First (επιλογή μεμονωμένων στηλών) ---")
    column_first = fn['retrieve_columns'](question, subquestions, keywords, full_schema)
    say(f"{len(column_first)} πίνακες, {count_cols(column_first)} στήλες")
    say(fmt_schema(column_first))

    say("\n\n--- ΕΝΩΣΗ: το schema που μπαίνει τελικά στο prompt ---")
    selected = {}
    for t in table_first:
        selected[t] = set(full_schema[t])
    for t, cols in column_first.items():
        selected.setdefault(t, set()).update(cols)
    retrieved = {t: [c for c in full_schema[t] if c in cols] for t, cols in selected.items()}
    say(f"{len(retrieved)} πίνακες, {count_cols(retrieved)} στήλες")
    say(fmt_schema(retrieved))

    oracle = {}
    if gold_sql:
        oracle_raw = fn['extract_gold_schema'](gold_sql, full_schema)
        oracle = {t: sorted(c) for t, c in oracle_raw.items()}
        say("\n\n--- ΓΙΑ ΣΥΓΚΡΙΣΗ: το oracle schema (από το gold SQL) ---")
        say("Αυτό χρειάζεται πραγματικά το ερώτημα. Δεν είναι διαθέσιμο σε πραγματικές συνθήκες.")
        say(f"{len(oracle)} πίνακες, {count_cols(oracle)} στήλες")
        say(fmt_schema(oracle))

    say("\n\n--- ΣΥΝΟΨΗ ---")
    say(f"{'Συνθήκη':<28}{'Πίνακες':>10}{'Στήλες':>10}")
    say("-" * 48)
    say(f"{'Full (πλήρες schema)':<28}{len(full_schema):>10}{count_cols(full_schema):>10}")
    say(f"{'Retrieved (ο μηχανισμός)':<28}{len(retrieved):>10}{count_cols(retrieved):>10}")
    if oracle:
        say(f"{'Perfect (oracle)':<28}{len(oracle):>10}{count_cols(oracle):>10}")
        missing = []
        for t, cs in oracle.items():
            if t not in retrieved:
                missing.append(t)
            else:
                missing += [f"{t}.{c}" for c in cs if c not in retrieved[t]]
        say("\nΈχασε ο μηχανισμός κάτι αναγκαίο; "
            + ("ΟΧΙ" if not missing else "ΝΑΙ -> " + ", ".join(missing)))
        say(f"Περιττές στήλες που κρατήθηκαν: {count_cols(retrieved) - count_cols(oracle)}")

    if stored_retrieved is not None:
        say("\n\n--- ΕΛΕΓΧΟΣ: ταιριάζει με το αποθηκευμένο αποτέλεσμα; ---")
        say("Οι κλήσεις στο μοντέλο δεν είναι ντετερμινιστικές, οπότε μικρές αποκλίσεις")
        say("είναι φυσιολογικές. Μεγάλη απόκλιση σημαίνει ότι φορτώθηκε λάθος module.")
        norm = lambda s: {t.lower(): set(c.lower() for c in (v or []))
                          for t, v in (s or {}).items()}
        a, b = norm(retrieved), norm(stored_retrieved)
        say(f"  αποθηκευμένο: {len(b)} πίνακες, {count_cols(stored_retrieved)} στήλες")
        say(f"  τώρα:         {len(a)} πίνακες, {count_cols(retrieved)} στήλες")
        same_t = set(a) == set(b)
        inter = sum(len(a[t] & b[t]) for t in set(a) & set(b))
        union = sum(len(a.get(t, set()) | b.get(t, set())) for t in set(a) | set(b))
        say(f"  ίδιοι πίνακες: {'ΝΑΙ' if same_t else 'ΟΧΙ'}"
            f"  |  επικάλυψη στηλών: {100 * inter / union if union else 0:.0f}%")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "worked_example.txt"), "w", encoding="utf-8") as f:
        f.write(out.getvalue())
    with open(os.path.join("results", "worked_example.json"), "w", encoding="utf-8") as f:
        json.dump({"db_id": db_id, "question": question, "gold_sql": gold_sql,
                   "full_schema": full_schema, "keywords": keywords,
                   "subquestions": subquestions, "table_first": table_first,
                   "column_first": column_first, "retrieved_schema": retrieved,
                   "oracle_schema": oracle}, f, ensure_ascii=False, indent=2)
    print("\nΓράφτηκαν: results/worked_example.txt και results/worked_example.json")


if __name__ == "__main__":
    main()
