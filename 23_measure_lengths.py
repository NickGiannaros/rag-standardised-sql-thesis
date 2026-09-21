"""
23_measure_lengths.py
Τι κάνει: μετρά το μήκος των τριών μορφών με τις οποίες εμφανίζεται το SQL
ενός παραδείγματος, ώστε η διαφορά μεταξύ Memory A και Memory B να δηλώνεται
με αριθμό και όχι με επιφύλαξη.

Μετρώνται:
  (α) το ίδιο το SQL: raw / standardised / standardised χωρίς τις γραμμές NONE
  (β) ολόκληρο το κείμενο που ενσωματώνεται, ώστε να φανεί πόσο μέρος του
      καταλαμβάνει το SQL — η «αραίωση» της ενότητας 4.4
  (γ) το τμήμα του prompt που διαφέρει μεταξύ των συνθηκών, δηλαδή τα SQL των
      k = 3 ανακτημένων παραδειγμάτων

Τα tokens μετρώνται μόνο αν υπάρχουν οι αντίστοιχες βιβλιοθήκες· διαφορετικά
οι στήλες μένουν κενές και μετρώνται μόνο χαρακτήρες και γραμμές.
  - tiktoken (o200k_base)            -> tokens του gpt-4o-mini, για το prompt
  - transformers / all-MiniLM-L6-v2  -> tokens του μοντέλου ενσωμάτωσης

Είσοδος: texts_cache.pkl (παράγεται από το 3_build_texts.py)
Έξοδος : results/length_comparison.csv
"""
import os, re, pickle, csv, statistics as st

K = 3                      # όσα παραδείγματα μπαίνουν στο prompt
CACHE = "texts_cache.pkl"
OUT = os.path.join("results", "length_comparison.csv")

NONE_LINE = re.compile(
    r"^\s*(GROUP BY|ORDER BY|HAVING|LIMIT|INTERSECT|EXCEPT|UNION|WHERE)\s+NONE\s*$")


def strip_none_clauses(sql):
    """Ίδια συνάρτηση με το 8_rebuild_compact_memories.py."""
    return "\n".join(l for l in sql.split("\n") if not NONE_LINE.match(l)).strip()


# ── προαιρετικοί tokenizers ───────────────────────────────────────────────
def load_tokenizers():
    gpt = mini = None
    try:
        import tiktoken
        gpt = tiktoken.encoding_for_model("gpt-4o-mini").encode
        print("tiktoken: OK")
    except Exception as e:
        print("tiktoken: μη διαθέσιμο ->", type(e).__name__)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        mini = lambda s: tok(s, add_special_tokens=True)["input_ids"]
        print("all-MiniLM-L6-v2 tokenizer: OK")
    except Exception as e:
        print("all-MiniLM-L6-v2 tokenizer: μη διαθέσιμο ->", type(e).__name__)
    return gpt, mini


def main():
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    texts_A, texts_B, metadata = cache["texts_A"], cache["texts_B"], cache["metadata"]
    n = len(metadata)
    print(f"Παραδείγματα μνήμης: {n}")

    gpt_enc, mini_enc = load_tokenizers()

    rows = {name: {"chars": [], "lines": [], "gpt": [], "mini": [], "trunc": 0}
            for name in ("raw_sql", "standardised_sql", "standardised_compact",
                         "text_A", "text_B", "text_B_compact", "schema")}

    for i, m in enumerate(metadata):
        raw = m["raw_sql"]
        std = m["standardised_sql"]
        cmp_ = strip_none_clauses(std)
        # το σχήμα είναι ταυτόσημο στις τρεις συνθήκες — απομονώνεται για σύγκριση
        head, _, tail = texts_A[i].partition(" SQL: ")
        schema = head.partition(" Schema: ")[2]
        tB_compact = texts_B[i].replace(std, cmp_)

        for name, s in (("raw_sql", raw), ("standardised_sql", std),
                        ("standardised_compact", cmp_), ("schema", schema),
                        ("text_A", texts_A[i]), ("text_B", texts_B[i]),
                        ("text_B_compact", tB_compact)):
            r = rows[name]
            r["chars"].append(len(s))
            r["lines"].append(s.count("\n") + 1)
            if gpt_enc:
                r["gpt"].append(len(gpt_enc(s)))
            if mini_enc:
                t = len(mini_enc(s))
                r["mini"].append(t)
                if t > 256:                 # το παράθυρο του all-MiniLM-L6-v2
                    r["trunc"] += 1

    def col(v):
        return (round(st.mean(v), 1), st.median(v), max(v)) if v else ("", "", "")

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["what", "mean_chars", "median_chars", "max_chars",
                    "mean_lines", "mean_gpt4omini_tokens", "mean_minilm_tokens",
                    "over_256_minilm_tokens", "n"])
        for name, r in rows.items():
            mc, dc, xc = col(r["chars"])
            ml, _, _ = col(r["lines"])
            mg, _, _ = col(r["gpt"])
            mm, _, _ = col(r["mini"])
            w.writerow([name, mc, dc, xc, ml, mg, mm,
                        r["trunc"] if r["mini"] else "", n])

    # ── η σύνοψη που ενδιαφέρει το κείμενο ───────────────────────────────
    raw_c = st.mean(rows["raw_sql"]["chars"])
    std_c = st.mean(rows["standardised_sql"]["chars"])
    cmp_c = st.mean(rows["standardised_compact"]["chars"])
    tA = st.mean(rows["text_A"]["chars"])
    tB = st.mean(rows["text_B"]["chars"])

    print(f"\nSQL ανά παράδειγμα (μέσοι χαρακτήρες)")
    print(f"  raw                : {raw_c:7.1f}")
    print(f"  standardised       : {std_c:7.1f}   ({std_c/raw_c:.2f}x του raw)")
    print(f"  standardised compact: {cmp_c:6.1f}   ({cmp_c/std_c*100:.1f}% του standardised)")
    print(f"  οι γραμμές NONE    : {std_c-cmp_c:7.1f} χαρακτήρες, "
          f"{(std_c-cmp_c)/std_c*100:.1f}% της τυποποιημένης μορφής")
    print(f"\nΜερίδιο του SQL μέσα στο κείμενο που ενσωματώνεται")
    print(f"  Memory A: {raw_c/tA*100:.1f}%     Memory B: {std_c/tB*100:.1f}%")
    print(f"\nΔιαφορά στο prompt, k = {K} παραδείγματα")
    print(f"  Memory B - Memory A: {K*(std_c-raw_c):.0f} χαρακτήρες κατά μέσο όρο")
    if rows["text_B"]["mini"]:
        print(f"\nΚείμενα πάνω από το παράθυρο των 256 tokens του all-MiniLM-L6-v2:")
        for k_ in ("text_A", "text_B", "text_B_compact"):
            print(f"  {k_:15}: {rows[k_]['trunc']} / {n}")
    print(f"\nΓράφτηκε: {OUT}")


if __name__ == "__main__":
    main()
