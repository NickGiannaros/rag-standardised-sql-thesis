"""
24_column_order_check.py
Τι κάνει: ελέγχει, χωρίς καμία υποκειμενική κρίση, πόσα από τα ερωτήματα που
χρεώνονται ως λάθος επιστρέφουν στην πραγματικότητα τη σωστή πληροφορία με τις
στήλες σε άλλη σειρά.

Η ακρίβεια εκτέλεσης συγκρίνει πλειάδες, και οι πλειάδες είναι διατεταγμένες.
Ένα ερώτημα που γράφει SELECT b, a αντί για SELECT a, b δίνει την ίδια
απάντηση αλλά μετριέται ως λάθος. Το script μετρά πόσο συχνά συμβαίνει αυτό.

Ο έλεγχος: υπάρχει μετάθεση των στηλών του παραγόμενου αποτελέσματος που το
κάνει ταυτόσημο με το gold; Ναι ή όχι. Τίποτα να κριθεί.

Εξετάζονται μόνο οι ερωτήσεις που η αντίστοιχη συνθήκη έχει ήδη χρεωθεί ως
λάθος στο master_results.csv, οπότε οι υπάρχοντες αριθμοί της εργασίας δεν
ξαναϋπολογίζονται -- μόνο συμπληρώνονται.

Είσοδος: results/master_results.csv, database/
Έξοδος : results/column_order_check.csv + σύνοψη στην οθόνη
"""
import os, sys, csv, sqlite3
from collections import Counter
from itertools import permutations
sys.path.insert(0, os.getcwd())
import pandas as pd

MASTER = os.path.join("results", "master_results.csv")
TAX = os.path.join("results", "error_taxonomy.csv")
OUT = os.path.join("results", "column_order_check.csv")

CONDITIONS = [("baseline", "sql_baseline", "baseline_ok"),
              ("Memory A", "sql_memA", "memA_ok"),
              ("Memory B", "sql_memB", "memB_ok"),
              ("Memory B compact", "sql_memB_compact", "memB_compact_ok"),
              ("Retrieved schema", "sql_retrieved", "retrieved_ok"),
              ("Perfect schema", "sql_perfect", "perfect_ok")]

MAX_TIE_PERMS = 5000


def execute_query(db_path, sql):
    """Ίδιοι κανόνες εκτέλεσης με τα υπόλοιπα scripts του έργου."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def col_sigs(rows, arity):
    """Υπογραφή κάθε στήλης: το ταξινομημένο πολυσύνολο των τιμών της."""
    return [tuple(sorted((r[j] for r in rows), key=lambda v: (v is None, str(v))))
            for j in range(arity)]


def is_column_permutation(gold, pred):
    """Υπάρχει μετάθεση στηλών του pred που το κάνει ίσο με το gold;"""
    if gold is None or pred is None:
        return False
    if len(gold) != len(pred) or not gold:
        return False
    a = len(gold[0])
    if a != len(pred[0]) or a < 2:
        return False           # με μία στήλη δεν υπάρχει σειρά να αλλάξει
    g_sorted = sorted(gold, key=repr)
    if sorted(pred, key=repr) == g_sorted:
        return False           # ήδη ίδιο χωρίς μετάθεση: δεν μας αφορά εδώ

    gs, ps = col_sigs(gold, a), col_sigs(pred, a)
    if Counter(map(repr, gs)) != Counter(map(repr, ps)):
        return False           # τα πολυσύνολα των στηλών δεν ταιριάζουν καν

    # υποψήφιες αντιστοιχίσεις: κάθε στήλη του pred σε στήλη του gold με ίδια υπογραφή
    cand = [[j for j in range(a) if repr(gs[j]) == repr(ps[i])] for i in range(a)]
    tried = 0
    for perm in permutations(range(a)):
        if tried > MAX_TIE_PERMS:
            break
        if any(perm[i] not in cand[i] for i in range(a)):
            continue
        tried += 1
        # perm[i] = σε ποια θέση του gold πάει η στήλη i του pred
        rebuilt = []
        for r in pred:
            row = [None] * a
            for i in range(a):
                row[perm[i]] = r[i]
            rebuilt.append(tuple(row))
        if sorted(rebuilt, key=repr) == g_sorted:
            return True
    return False


def main():
    m = pd.read_csv(MASTER)
    tax = pd.read_csv(TAX) if os.path.exists(TAX) else None
    n_total = len(m)
    print(f"Ερωτήσεις: {n_total}\n")

    schema_cache = {}
    gold_cache = {}
    rows_out = []
    summary = []

    for label, sql_col, ok_col in CONDITIONS:
        ok = m[ok_col].astype(str).str.strip().str.lower() == "true"
        wrong = m[~ok]
        flips = 0
        for _, r in wrong.iterrows():
            db_path = os.path.join("database", r.db_id, r.db_id + ".sqlite")
            gkey = (r.db_id, r.gold_sql)
            if gkey not in gold_cache:
                gold_cache[gkey] = execute_query(db_path, r.gold_sql)
            g_rows, g_err = gold_cache[gkey]
            if g_err is not None:
                continue                      # gold που δεν εκτελείται: εκτός
            p_rows, p_err = execute_query(db_path, r[sql_col])
            if p_err is not None:
                continue
            if is_column_permutation(g_rows, p_rows):
                flips += 1
                rows_out.append({"condition": label, "qid": r.qid,
                                 "db_id": r.db_id, "question": r.question,
                                 "gold_sql": r.gold_sql, "pred_sql": r[sql_col]})
        n_ok = int(ok.sum())
        summary.append({
            "condition": label,
            "correct_now": n_ok,
            "accuracy_now_pct": round(100 * n_ok / n_total, 1),
            "column_order_equivalents": flips,
            "correct_if_counted": n_ok + flips,
            "accuracy_if_counted_pct": round(100 * (n_ok + flips) / n_total, 1),
            "difference_pp": round(100 * flips / n_total, 1)})
        print(f"{label:<20} λάθος: {len(wrong):>4}   "
              f"εκ των οποίων μετάθεση στηλών: {flips:>4}   "
              f"{100*n_ok/n_total:>5.1f}% -> {100*(n_ok+flips)/n_total:>5.1f}%")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    det = OUT.replace(".csv", "_cases.csv")
    if rows_out:
        with open(det, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    # ── τι σημαίνει για την ταξινομία του Πίνακα 5.13 ────────────────────
    if tax is not None:
        base = {r["qid"] for r in rows_out if r["condition"] == "baseline"}
        print(f"\nΓια την ταξινομία των {len(tax)} σφαλμάτων του baseline:")
        hit = tax[tax.qid.isin(base)]
        print(f"  μεταθέσεις στηλών συνολικά: {len(hit)} / {len(tax)} "
              f"({100*len(hit)/len(tax):.1f}%)")
        for cat, k in hit.category.value_counts().items():
            tot = int((tax.category == cat).sum())
            print(f"    {cat:<22}{k:>4} / {tot:<4} ({100*k/tot:.0f}% της κατηγορίας)")

    print(f"\nΓράφτηκε: {OUT}")
    if rows_out:
        print(f"Γράφτηκε: {det}  ({len(rows_out)} περιπτώσεις για έλεγχο)")


if __name__ == "__main__":
    main()
