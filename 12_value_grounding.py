"""
value_grounding_fix.py
What this does: the "light version" of the value-grounding idea. Takes the
already-generated SQL for Baseline / Memory A / Memory B (compact) across the
full 1034-question dev set, finds string literals compared against a column
(col = 'value' or col LIKE 'value'), resolves which real table/column that
is, and checks the actual SQLite database for a matching value -- using
normalized (lowercase, punctuation/space-stripped) comparison, since a
same-case-but-differently-spaced literal (e.g. 'North Carolina' vs
'NorthCarolina') is just as broken as a case mismatch. If the column turns
out not to hold string values at all (e.g. it's actually an integer foreign
key -- discovered during prototyping on the car_1/continent example), that's
tagged as a structural/schema-linking issue, not something this fix can
address, and left untouched. No LLM calls -- pure SQL parsing + local SQLite
lookups, so this runs fast and free over the whole dev set.
Input: generation_results.json, generation_results_multi.json,
generation_results_remaining.json, generation_results_compact_B.json.
Output: value_grounding_results.json -- per question/condition: original
SQL, corrected SQL, list of literal fixes applied (with a reason tag for
literals that could NOT be fixed: "no_string_column" vs "no_match").
"""
import json, sqlite3, re, sys, os
sys.path.insert(0, os.getcwd())
from eval import process_sql
import sqlglot
from sqlglot import exp

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


def get_schema_typed(db_path):
    """Schema dict: {table_lower: {"real_name": ..., "cols": {col_lower: real_col_name}}}"""
    schema = {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(t[0]) for t in cur.fetchall()]
    for table in tables:
        cur.execute('PRAGMA table_info("{}")'.format(table))
        schema[table.lower()] = {
            "real_name": table,
            "cols": {str(c[1]).lower(): str(c[1]) for c in cur.fetchall()},
        }
    conn.close()
    return schema


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def scope_aliases(select_node):
    aliases = {}
    from_ = select_node.find(exp.From)
    if from_:
        for t in from_.find_all(exp.Table):
            aliases[(t.alias or t.name).lower()] = t.name
    for j in select_node.args.get("joins") or []:
        for t in j.find_all(exp.Table):
            aliases[(t.alias or t.name).lower()] = t.name
    return aliases


def fix_literals(sql, db_path, schema, value_cache):
    """Returns (corrected_sql, list_of_fix_records, changed_bool)."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:
        return sql, [{"status": "parse_error", "error": str(e)}], False

    changed = False
    records = []
    for lit in list(tree.find_all(exp.Literal)):
        if not lit.is_string:
            continue
        parent = lit.parent
        if not isinstance(parent, (exp.EQ, exp.Like)):
            continue
        col_node = parent.left if parent.right is lit else parent.right
        if not isinstance(col_node, exp.Column):
            continue
        col_name = col_node.name.lower()
        tbl_alias = col_node.table.lower() if col_node.table else None
        sel = parent.find_ancestor(exp.Select)
        if sel is None:
            continue
        aliases = scope_aliases(sel)

        real_table = aliases.get(tbl_alias) if tbl_alias else None
        if not real_table:
            for _, t in aliases.items():
                tl = t.lower()
                if tl in schema and col_name in schema[tl]["cols"]:
                    real_table = t
                    break
        if not real_table or real_table.lower() not in schema:
            continue
        real_col = schema[real_table.lower()]["cols"].get(col_name)
        if not real_col:
            continue

        cache_key = (real_table.lower(), real_col.lower())
        if cache_key not in value_cache:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            try:
                cur.execute(f'SELECT DISTINCT "{real_col}" FROM "{real_table}"')
                rows = cur.fetchall()
            except Exception:
                rows = []
            conn.close()
            str_values = [r[0] for r in rows if isinstance(r[0], str)]
            has_any_rows = len(rows) > 0
            value_cache[cache_key] = (str_values, has_any_rows and not str_values)

        str_values, is_non_string_col = value_cache[cache_key]
        lit_val = lit.this

        if is_non_string_col:
            records.append({
                "status": "no_string_column", "table": real_table, "column": real_col,
                "literal": lit_val,
            })
            continue

        if lit_val in str_values:
            continue  # already exact, nothing to fix

        norm_target = norm(lit_val)
        matches = [v for v in str_values if norm(v) == norm_target]
        if len(matches) == 1 and matches[0] != lit_val:
            lit.set("this", matches[0])
            changed = True
            records.append({
                "status": "fixed", "table": real_table, "column": real_col,
                "from": lit_val, "to": matches[0],
            })
        else:
            records.append({
                "status": "no_match", "table": real_table, "column": real_col,
                "literal": lit_val,
            })

    return tree.sql(dialect="sqlite"), records, changed


def process_condition(rows, sql_field, cleaner, schema_cache, value_cache_by_db):
    out = []
    for r in rows:
        db_id = r["db_id"]
        db_path = os.path.join("database", db_id, db_id + ".sqlite")
        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = get_schema_typed(db_path)
            except Exception:
                schema_cache[db_id] = {}
        if db_id not in value_cache_by_db:
            value_cache_by_db[db_id] = {}

        original_sql = cleaner(r[sql_field])
        fixed_sql, records, changed = fix_literals(
            original_sql, db_path, schema_cache[db_id], value_cache_by_db[db_id]
        )
        out.append({
            "db_id": db_id,
            "question": r["question"],
            "original_sql": original_sql,
            "fixed_sql": fixed_sql,
            "changed": changed,
            "fix_records": records,
        })
    return out


def main():
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

    with open("generation_results_compact_B.json") as f:
        compact_B = json.load(f)

    schema_cache = {}
    value_cache_by_db = {}

    print("Processing Baseline...")
    baseline_out = process_condition(all_raw, "sql_baseline", clean_markdown, schema_cache, value_cache_by_db)
    print("Processing Memory A...")
    a_out = process_condition(all_raw, "sql_memory_A", clean_markdown, schema_cache, value_cache_by_db)
    print("Processing Memory B (compact)...")
    b_out = process_condition(compact_B, "sql_memory_B_compact", clean_memory_B, schema_cache, value_cache_by_db)

    result = {"baseline": baseline_out, "memory_A": a_out, "memory_B_compact": b_out}
    with open("value_grounding_results.json", "w") as f:
        json.dump(result, f, indent=2)

    for label, rows in result.items():
        n_changed = sum(1 for r in rows if r["changed"])
        n_no_string_col = sum(1 for r in rows for fr in r["fix_records"] if fr["status"] == "no_string_column")
        n_no_match = sum(1 for r in rows for fr in r["fix_records"] if fr["status"] == "no_match")
        n_fixed_literals = sum(1 for r in rows for fr in r["fix_records"] if fr["status"] == "fixed")
        print(f"{label}: {n_changed} queries changed | {n_fixed_literals} literals fixed | "
              f"{n_no_string_col} structural (non-string column) | {n_no_match} no match found")


if __name__ == "__main__":
    main()
