# RAG Memory με Standardised SQL — Πτυχιακή Εργασία

Πειραματική επέκταση του [Xander](https://github.com/henrijsprincis/Xander) — από το paper "Enhancing SQL Query Generation with Neurosymbolic Reasoning" — με έναν εξωτερικό μηχανισμό μνήμης τύπου Retrieval-Augmented Generation (RAG): αντί το μοντέλο να στηρίζεται μόνο στη δική του γνώση, πριν απαντήσει ανακτά παλιά παραδείγματα (ερώτηση, SQL) και τα χρησιμοποιεί ως οδηγό μέσα στο prompt.

## Ερευνητικό ερώτημα

Το Xander εισάγει τη Standardised SQL, μια τυποποιημένη μορφή SQL (κεφαλαία keywords, υποχρεωτικά clauses ακόμα κι όταν είναι κενά, επέκταση aliases σε πλήρη ονόματα πινάκων/στηλών), με στόχο τη μείωση θορύβου κατά την εκπαίδευση μοντέλων. Η παρούσα εργασία εξετάζει αν το ίδιο όφελος εμφανίζεται σε διαφορετικό πλαίσιο: στην ποιότητα ανάκτησης (retrieval quality) μέσα σε ένα σύστημα RAG, όπου παλιά (ερώτηση, SQL) παραδείγματα χρησιμοποιούνται ως few-shot demonstrations (παραδείγματα μέσα στο prompt, πριν το πραγματικό ερώτημα, ώστε το μοντέλο να μάθει το πρότυπο) για ένα LLM (gpt-4o-mini) και, τελικά, στο execution accuracy, δηλαδή αν η τυχόν καλύτερη ανάκτηση μεταφράζεται πράγματι σε περισσότερες σωστές απαντήσεις από το LLM.

## Μεθοδολογία

1. Δύο vector databases πάνω στο [Spider dataset](https://yale-lily.github.io/spider) (7000 training παραδείγματα, 140 databases): Memory A (raw SQL), Memory B (Standardised SQL). Σε ένα vector database κάθε παράδειγμα αποθηκεύεται ως embedding ώστε να μπορούμε να βρίσκουμε γρήγορα παραδείγματα με παρόμοιο νόημα.
2. Asymmetric semantic search: αναζήτηση με βάση το νόημα (όχι λέξη-προς-λέξη), όπου το κείμενο που αποθηκεύεται στο index (Question + Schema + SQL) περιέχει περισσότερη πληροφορία απ' όσο το κείμενο με το οποίο ψάχνουμε τη στιγμή του ερωτήματος (Question + Schema μόνο — δεν ξέρουμε ακόμα το SQL).
3. Retrieval top-k=3 παραδειγμάτων ανά dev ερώτηση (τα 3 πιο κοντινά σημασιολογικά), από κάθε μνήμη ξεχωριστά.
4. Generation SQL μέσω gpt-4o-mini (API), με τα ανακτημένα παραδείγματα ως few-shot prompt.
5. Execution accuracy: εκτέλεση του παραγόμενου SQL στην πραγματική SQLite βάση, σύγκριση των αποτελεσμάτων με το gold (σωστό) αποτέλεσμα — ταυτίζονται ή όχι.
6. Retrieval quality (SQL skeleton matching): το "skeleton" είναι η δομή ενός ερωτήματος χωρίς τις συγκεκριμένες τιμές (ποιοι πίνακες, στήλες, clauses χρησιμοποιούνται). Μετράμε αν τα ανακτημένα παραδείγματα έχουν το ίδιο skeleton με το gold query, καθαρά σε επίπεδο ανάκτησης, πριν καν μπει το LLM στην εικόνα — έτσι απομονώνουμε αν το πρόβλημα είναι στην ανάκτηση ή στην παραγωγή.

Σημείωση: ο νευροσυμβολικός πυρήνας του Xander (Best-First Search, Partial-Query Checker, Query Test-and-Repair) **δεν** ενεργοποιείται σε αυτή τη διαμόρφωση καθώς λειτουργούν πάνω σε token-level πρόσβαση ενός τοπικά fine-tuned μοντέλου, ασύμβατο με τη χρήση API μοντέλου.

## Αποτελέσματα

### Pilot (205 ερωτήσεις, 5 βάσεις δεδομένων)

| Συνθήκη | Execution Accuracy |
|---|---|
| Χωρίς μνήμη (baseline) | 174/205 (85%) |
| Memory A (raw SQL) | 175/205 (85%) |
| Memory B (Standardised SQL) | 179/205 (87%) |

Θετική τάση για τη Standardised SQL, αλλά όχι ακόμα στατιστικά σημαντική σε αυτό το μέγεθος δείγματος.

### Πλήρες Dev Set (1034 ερωτήσεις, 20 βάσεις δεδομένων)

Το αρχικό εύρημα **δεν γενικεύτηκε**:

| Συνθήκη | Execution Accuracy |
|---|---|
| Baseline | 737/1034 (71.3%) |
| Memory A (raw SQL) | 731/1034 (70.7%) |
| Memory B (Standardised SQL) | 733/1034 (70.9%) |

McNemar (Baseline vs B): p = 0.7807. McNemar (A vs B): p = 0.9261 — καμία στατιστικά σημαντική διαφορά.

Το McNemar test είναι ένα στατιστικό τεστ για ζευγαρωτές (paired) δυαδικές μετρήσεις: επειδή οι ίδιες ακριβώς ερωτήσεις απαντήθηκαν και στις δύο συνθήκες, το τεστ κοιτάζει συγκεκριμένα τις περιπτώσεις όπου η μία συνθήκη πέτυχε και η άλλη απέτυχε (και το αντίστροφο), και ελέγχει αν αυτή η ασυμμετρία είναι πιθανό να οφείλεται στην τύχη. Τιμή p < 0.05 θεωρείται στατιστικά σημαντική διαφορά· εδώ p ήταν πολύ πάνω από αυτό το όριο, άρα δεν υπάρχει πραγματική διαφορά.

Ταυτόχρονα, το retrieval quality (skeleton matching) έδειξε την raw SQL ελαφρώς καλύτερη: Memory A 20.8% top-1 / 32.0% top-3, Memory B 19.0% top-1 / 31.2% top-3.

### Διόρθωση: Boilerplate Dilution Fix

Τι παρατηρήσαμε: η Standardised SQL γράφει ρητά κάθε clause ενός ερωτήματος, ακόμα κι όταν είναι κενό (π.χ. ένα απλό SELECT γίνεται "... GROUP BY NONE ORDER BY NONE HAVING NONE ..."). Όταν αυτό το κείμενο μετατρέπεται σε embedding, οι επαναλαμβανόμενες λέξεις "NONE" παίρνουν δυσανάλογο βάρος μέσα στο διάνυσμα και "αραιώνουν" το πραγματικό περιεχόμενο του ερωτήματος. Δηλαδή, δύο εντελώς διαφορετικά ερωτήματα με τα ίδια κενά clauses μπορεί να μοιάζουν πιο κοντινά μεταξύ τους απ' ό,τι θα έπρεπε.

Πώς το διορθώσαμε: αφαιρέσα αυτές τις κενές ("NONE") γραμμές **μόνο** από το κείμενο που μετατρέπεται σε embedding για την αναζήτηση. Η πλήρης Standardised SQL, με όλα τα clauses, παραμένει αυτή που βλέπει το LLM μέσα στο few-shot prompt όταν παράγει την τελική απάντηση. Αρα δεν αλλάζει τι μαθαίνει το μοντέλο, μόνο πώς βρίσκουμε ποια παραδείγματα να του δείξουμε.

Αποτέλεσμα (retrieval quality): Memory B βελτιώθηκε από 19.0% σε 20.4% top-1 (McNemar p = 0.0400, στατιστικά σημαντικό) και από 31.2% σε 32.5% top-3 (πλέον μπροστά από τη Memory A). Δεν μεταφέρθηκε όμως στο execution accuracy (733 → 732, McNemar p = 1.0000).

### Πρόσθετη Αξιολόγηση: Merit Metric

Απλοποιημένη εκδοχή του Merit metric, όπως ορίζεται στο ["Retrieval-Augmented NL2SQL Generation with Data-Centric Query Capsules for Enterprise Applications"](https://dl.acm.org/doi/10.1145/3767695.3769489). Το Merit συνδυάζει δύο σκορ σε ένα F1 (αρμονικός μέσος όρος):
- Structural score: πόσο μοιάζει η δομή δύο SQL ερωτημάτων, μέσω tree edit distance (sqlglot μετατρέπει το SQL σε δέντρο, zss μετράει πόσες αλλαγές — προσθήκες, διαγραφές, αντικαταστάσεις κόμβων —χρειάζονται για να μετατραπεί το ένα δέντρο στο άλλο -- λιγότερες αλλαγές = πιο κοντινά).
- Semantic score: πόσο μοιάζουν σημασιολογικά δύο ερωτήματα, μέσω cosine similarity (πόσο κοντά είναι τα δύο embeddings) πάνω στο ίδιο embedding μοντέλο που χρησιμοποιούμε στο retrieval.

Αποτέλεσμα: Merit A = +0.0189, Merit B (πριν) = -0.0169, Merit B (μετά) = -0.0184 — δεν στηρίζει την υπόθεση.

### Robustness Check (δεύτερο embedding model)

Τα αποτελέσματα παραπάνω βασίζονται σε ένα συγκεκριμένο μοντέλο, οπότε επαναλάβα τη διόρθωση με ένα δεύτερο, διαφορετικό μοντέλο (all-mpnet-base-v2) για να δω αν το εύρημα είναι πραγματικό ή ιδιαιτερότητα του πρώτου μοντέλου. Αποτέλεσμα: Memory B compact 32.9% top-1 (από 30.7%), McNemar p = 0.0469 — επιβεβαιώνει το εύρημα για το retrieval quality σε δεύτερο, ανεξάρτητο embedding model.

### Ανάλυση ανά Δυσκολία Ερωτήματος

Περιγραφικό crossover pattern (raw SQL καλύτερη σε απλά queries, Standardised SQL σε μέτρια/σύνθετα) — **δεν** επιβεβαιώθηκε στατιστικά (p = 0.1996 και p = 0.2529). Αναφέρεται ως ανοιχτή παρατήρηση, όχι εύρημα.

### Ερμηνεία

Η κεντρική υπόθεση (Standardised SQL βελτιώνει γενικά ένα RAG memory σύστημα) δεν επιβεβαιώνεται καθολικά. Το μόνο σταθερό, στατιστικά σημαντικό, δύο φορές επιβεβαιωμένο εύρημα είναι πιο συγκεκριμένο: η Normalized SQL του Xander σχεδιάστηκε για fine-tuning, ποτέ δεν δοκιμάστηκε σε πλαίσιο retrieval — εντοπίσαμε ένα boundary condition (boilerplate dilution) και το διορθώσαμε, με στατιστικά καλύτερο αποτέλεσμα σε δύο ανεξάρτητα embedding models. 

### Προτεινόμενη Επόμενη Φάση (δεν έχει υλοποιηθεί ακόμα)

Επειδή δεν καταφέρα να πετύχουμε καλύτερο execution accuracy με τις παραπάνω διορθώσεις, μια ιδέα που μου ήρθε ήταν να αξιοποιήσουμε το "pipeline approach" (4 βήματα) που περιγράφεται στο ["Retrieval-Augmented NL2SQL Generation with Data-Centric Query Capsules for Enterprise Applications"](https://dl.acm.org/doi/10.1145/3767695.3769489). Σε δειγματοληπτικό έλεγχο των αποτελεσμάτων μας είδα ότι κάτι τέτοιο θα μπορούσε να βοηθήσει: υπήρχαν περιπτώσεις όπου το SQL ήταν δομικά σωστό αλλά απέτυχε επειδή η τιμή δεν ταίριαζε ακριβώς με τη βάση — π.χ. το μοντέλο έγραψε "Europe" ενώ η βάση έχει "europe".

Το pipeline:
1. Παραγωγή ενός SQL template με placeholders αντί για τελικές τιμές (π.χ. "WHERE nationality = [placeholder:string]"), ώστε το μοντέλο να μην "μαντεύει" τιμή στο πρώτο βήμα.
2. Εξαγωγή keywords από την ερώτηση (few-shot), για να εντοπιστεί ποια λέξη της ερώτησης αντιστοιχεί σε ποια τιμή.
3. Αναζήτηση (data retrieval) στην πραγματική βάση δεδομένων για την ακριβή, υπαρκτή τιμή.
4. Συμπλήρωση του template (query completion) με τις ανακτημένες τιμές, ώστε να προκύψει το τελικό, εκτελέσιμο SQL.

Ενδέχεται να χρειαστεί κάποια απλοποίηση αυτού του pipeline για λόγους εφικτότητας (π.χ. λιγότερα βήματα ή πιο ελαφριά υλοποίηση της αναζήτησης τιμών) — κάτι που μπορεί να αποφασιστεί αργότερα.  

## Δομή Repository
!*ΝΟΤΕ: Η υλοποίηση των scripts, η εκτέλεση των πειραμάτων, και μέρος της ανάλυσης/τεκμηρίωσης αυτής της εργασίας έγιναν με σημαντική βοήθεια εργαλείων τεχνητής νοημοσύνης(claude, gemini). Ο ερευνητικός σχεδιασμός, η ερμηνεία των ευρημάτων, και η τελική ευθύνη παραμένουν δικά μου.

**Pilot (205 ερωτήσεις, 5 βάσεις):**
- `1_load_data.py` — λήψη πλήρους training set από το Spider dataset
- `2_standardise_sql.py` — μετατροπή σε Standardised SQL
- `3_build_texts.py` — κατασκευή κειμένου για embedding
- `4_build_memory.py` — δημιουργία των δύο FAISS vector indices (Memory A, Memory B)
- `5_generate_single_db.py` — generation SQL (baseline/A/B) σε 1 βάση (concert_singer, 45 ερωτήσεις)
- `5b_generate_multi_db.py` — generation SQL σε 4 ακόμα βάσεις (160 ερωτήσεις)
- `6_evaluate_single_db.py` — execution accuracy, 1 βάση
- `6b_error_analysis.py` — λεπτομερής ανάλυση λαθών + McNemar test
- `6c_evaluate_multi_db.py` — execution accuracy, πολλαπλές βάσεις

**Πλήρες Dev Set (1034 ερωτήσεις, 20 βάσεις):**
- `5c_generate_remaining_dbs.py` — generation στις υπόλοιπες 15 βάσεις
- `6d_evaluate_full_dev_set.py` — execution accuracy σε όλο το dev set + McNemar tests

**Boilerplate Dilution Fix (retrieval quality):**
- `7_measure_retrieval_quality.py` — μέτρηση retrieval quality (SQL skeleton matching), πριν τη διόρθωση
- `8_rebuild_compact_memories.py` — η διόρθωση: αφαίρεση NONE-boilerplate από το embedding text
- `8b_measure_retrieval_quality_compact.py` — retrieval quality μετά τη διόρθωση
- `8c_measure_retrieval_significance.py` — paired McNemar test (πριν vs μετά)
- `8d_generate_compact_B.py` — επαναπαραγωγή SQL με το compact index
- `8e_evaluate_compact_B.py` — execution accuracy με το compact index

**Merit Metric:**
- `9_compute_merit.py` — απλοποιημένη υλοποίηση του Merit metric (βλ. ενότητα Αποτελέσματα)
- `9b_diagnose_merit.py` — diagnostic (F1 μόνο σε exec-correct απαντήσεις)

**Robustness Check:**
- `10_rebuild_memories_mpnet.py` — rebuild μνημών με all-mpnet-base-v2
- `10b_measure_retrieval_quality_mpnet.py` — retrieval quality με το νέο μοντέλο
- `10c_measure_retrieval_significance_mpnet.py` — paired McNemar test

**Ανάλυση ανά Δυσκολία:**
- `11_analyze_by_difficulty.py` — execution accuracy ανά επίπεδο πολυπλοκότητας query
- `11b_test_crossover_significance.py` — στοχευμένο follow-up significance test

`results/` — αποθηκευμένα αποτελέσματα generation (JSON).

## Setup

Τα scripts εξαρτώνται από το [Xander repository](https://github.com/henrijsprincis/Xander) και τρέχουν μέσα από αυτό (χρησιμοποιούν τα modules `eval.process_sql` και `open_ai.chat_gpt`, καθώς και τις SQLite βάσεις του Spider που περιλαμβάνει).

```bash
git clone https://github.com/henrijsprincis/Xander.git
cd Xander
pip install -r requirements.txt
tar -xf database.zip
```

Αντίγραψε τα scripts αυτού του repository μέσα στον φάκελο `Xander/`, εγκατέστησε τα επιπλέον requirements:

```bash
pip install -r requirements.txt
```

και δημιούργησε `.env` μέσα στον φάκελο `Xander/` με:

```
OPENAI_API_KEY=...
HF_TOKEN=...
```

Τρέξε τα scripts με τη σειρά που φαίνεται στη Δομή Repository παραπάνω. Τα ενδιάμεσα αρχεία (`.index`, `.pkl`, μεγάλα `sample_*.json`) δεν είναι committed (βλ. `.gitignore`) — αναπαράγονται τρέχοντας τα scripts με τη σειρά.

## Πλαίσιο

Πτυχιακή εργασία, στο πλαίσιο του γενικού θέματος "Σχεδίαση και Ανάπτυξη Μηχανισμών Μνήμης και Ανάκτησης Πληροφορίας σε Μεγάλα Γλωσσικά Μοντέλα". 
