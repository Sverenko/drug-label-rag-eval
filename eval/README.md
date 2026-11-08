# Eval questions

[`questions.jsonl`](questions.jsonl) holds 60 questions: 50 that the labels answer and 10 that they don't. They were written with Claude from the label text exactly as indexed (`python -m labelrag.questions show <label> <section>`), and committed before any retrieval or answering ran on them. Each line's `notes` field says why that question is in the set. This page covers the reasoning for each group.

Every question has to pass one test: someone holding only the labels could answer it from one or two passages, and two such people would agree on the answer. Loose wording is allowed, and some questions use it on purpose. Ambiguity about which drug the question means is not: where one brand covers several products, the question names the form.

The questions cover 44 of the 120 labels, across all six themes: diabetes 14, blood thinners 8, urology 8, migraine 8, cholesterol 6, ARBs 6. Blood thinners get more than their 7 labels would suggest because they carry most of the head-to-head trials.

## Where the answers live

| Section | Questions with gold there |
|---|---|
| Dosage and Administration | 19 |
| Warnings and Precautions | 15 |
| Use in Specific Populations | 9 |
| Indications and Usage | 8 |
| Contraindications | 7 |
| Boxed Warning | 6 |
| Drug Interactions | 5 |
| Patient Counseling Information | 5 |
| Clinical Studies | 4 |
| Adverse Reactions | 3 |
| Clinical Pharmacology | 3 |
| How Supplied | 1 |
| Drug Abuse and Dependence | 1 |

When a fact sits in several sections, every section that answers the question is listed as gold, so retrieving any copy counts as a hit.

## What makes a question hard

Eleven questions have no tag. They're plain lookups, the baseline: without them, a hard question failing can't be told apart from a pipeline that fails everything. The other 39 carry one to three tags.

| Tag | What it tests | Questions |
|---|---|---|
| `near_duplicate` (26) | Another label has near-identical text under a different product's name. Gold is only the product the question names, so retrieving a sibling's copy is a miss. | see below |
| `lay_terms` (8) | Everyday words where the label uses clinical ones: "contrast dye", "birth control pills", "kidneys work especially well", "irregular heartbeat", "bowel problems", "severe liver disease". | q02 q06 q14 q15 q16 q27 q37 q49 |
| `knowledge_conflict` (7) | The current label says something different from what's widely repeated, usually because the label changed. An answer from the model's memory is wrong. | q02 q04 q09 q11 q18 q23 q24 |
| `false_premise` (7) | The question assumes a use or fact the label contradicts. The right answer corrects it from the label; declining is wrong. | q03 q05 q07 q11 q18 q32 q44 |
| `full_text` (7) | The answer is only in the full text (a trial table, a pharmacokinetic value, a detail in a long subsection), not in the Highlights summary at the top of the section. | q13 q17 q19 q20 q22 q27 q50 |
| `label_comparison` (5) | How the drug did against another drug in a trial the label itself reports. Comparative questions are answerable when one label does the comparing. | q13 q15 q17 q19 q20 |
| `multi_section` (4) | A complete answer needs facts from more than one section, so retrieving one of them still leaves the answer short. | q16 q18 q23 q29 |
| `brand_vs_generic` (4) | The question uses one name and the passage the other, so only the chunk header ties them. These matter most for the experiment that removes the header. | q26 q31 q33 q46 |

### Label vs memory

`knowledge_conflict` is the group most worth watching. Answering from documents, with citations, is the point of the system, and these questions are where an answer from training data is wrong:

- q23, Lipitor in pregnancy: FDA had the pregnancy contraindication removed from statin labels in 2021, and none of the 8 statin labels lists it.
- q24, liver tests on Lipitor: routine periodic monitoring left statin labels in 2012.
- q11, Invokana's amputation boxed warning: removed in 2020. The label has no boxed warning section at all.
- q18, Brilinta's aspirin boxed warning: dropped from the current label, which keeps only the bleeding-risk warning.
- q02, metformin and contrast imaging: older labels stopped metformin for everyone and withheld it for 48 hours afterward. The current label stops it only for higher-risk patients (eGFR 30 to 60, liver disease, alcoholism, heart failure, intra-arterial contrast) and re-checks eGFR 48 hours after.
- q04, Jardiance's eGFR floor: 30 now, 45 in older labels and much of what's written about the class.
- q09, Ozempic tablets: a new oral product that shares a label with Rybelsus, while the Ozempic most people know is a weekly injection.

### Near-duplicates, by kind

The corpus includes close relatives on purpose, so this is the largest group:

- **Same brand, separate labels**: Pradaxa capsules vs oral pellets (q16); Imitrex injection vs tablets (q43, q44).
- **Same ingredient, different product**: Mounjaro and Zepbound (q05, q06), Saxenda and Victoza (q07), Rybelsus and Ozempic tablets (q09), Jardiance and its combinations (q03, q04), Invokana and Invokamet (q11), Nesina and its combinations (q14), Lipitor, Caduet and Atorvaliq (q23, q24), Zocor and Vytorin (q25), Nexletol and Nexlizet (q28), the tadalafil labels (q29), Proscar and Propecia (q30, q31), Cardura XL and Cardura (q32), Benicar and its combinations (q37).
- **Combination vs component**: Janumet's fixed dose (q01), Opsynvi's macitentan warning (q35), Diovan HCT's hydrochlorothiazide (q39). The combination answers differently from the single drug.
- **Class boilerplate**: different drugs, almost the same words. The metformin contrast rule appears in 15 labels (q02), the GLP-1 thyroid boxed warning in 7 (q08), and the ARB fetal-toxicity boxed warning in all 18 (q42).

## Why the other 10 can't be answered

None of them is answerable from any label; each was checked against the indexed text. They differ in what tempts the system to answer anyway.

| Tag | What it tests | Questions |
|---|---|---|
| `not_in_label` (2) | The drug is indexed, but labels don't cover the topic (insurance coverage, generic availability). Retrieval finds the right label and it has nothing to say. | u01 u02 |
| `off_label` (2) | A real off-label use no label mentions. For u04, Raynaud's appears only as a side effect in migraine labels, so retrieval brings back passages about other drugs. | u03 u04 |
| `absent_drug` (2) | A famous drug that isn't in the corpus while its relatives are (Eliquis next to Xarelto, Savaysa and Pradaxa; Wegovy next to Zepbound, Saxenda and oral semaglutide). The risk is answering with a relative's label. | u05 u06 |
| `mentioned_only` (2) | A drug with no label that other labels mention: warfarin in 173 chunks, rosuvastatin (Crestor) in 21, often with study doses that look like an answer. | u07 u08 |
| `unmentioned_comparison` (2) | Two indexed drugs that no label compares, which invites comparing numbers from separate trials. | u09 u10 |

A question about a drug that isn't in the corpus can still be answerable: Brilinta's interactions section covers rosuvastatin. That's why the unanswerable questions were checked against the text rather than against the manifest.

## Notes for grading

- **Yes/no answers**: for q05, q11, q18, q23 and q32 the key facts don't capture whether the answer says yes or no. An answer that quotes the right passage but gets the yes/no wrong is wrong.
- **Declines**: for the `false_premise` questions the right answer corrects the premise. The fixed decline sentence counts as a wrong decline.
- **Key facts** are copied from the indexed text, where openFDA flattens some symbols ("≥" becomes "?", "m²" becomes "m 2"). No key fact depends on those characters.
- **Validator**: `python -m labelrag.questions` reports one warning, for q05. Mounjaro's Clinical Studies section repeats the indication wording, and it also reports weight loss from the trials, so it was left out of gold on purpose.

Changes after the first eval run go in their own commit with the reason, so the history shows what changed after results were seen.
