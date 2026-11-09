# Failure analysis: the final run

Every failure in [`results/prompt_v3`](../results/prompt_v3/), the final run: prompt v3, 60 questions, each answered from both retrieval orders, so 120 answers. Each failure gets a cause. Answer grades are Claude's (see Metrics in the [README](../README.md#metrics)); retrieval is the same as in every run, since it came from the cache.

## By cause

11 of the 120 answers are wrong or partly right, or are missing altogether:

| Cause | Answers | Which |
|---|---|---|
| Gold section not in the top 50 | 0 | |
| In the top 50, but not the top 5 | 1 | q26, embedding order: declined (rerank ranked it first) |
| Right context, wrong answer | 6 | 3 answers open with "Yes" where the answer is no (q10 in both orders, q34 with embedding only); 2 let a false premise stand (q11 in both); 1 hedges and never answers (q23 with rerank) |
| Answered from memory | 1 | q23, embedding order: says Lipitor is contraindicated in pregnancy, as statin labels did before 2021 |
| Cohere returned no answer (a 422) | 3 | u05, u06 and u10 with rerank, all unanswerable questions |
| Grader wrong | 0 | q18 is borderline: graded right, though it leaves out the aspirin dose the reference includes |

## Retrieval

The gold section was outside the top 5 once: q26 with embedding only ("How often is inclisiran injected?"), where the dosing section names the drug as LEQVIO and ranked 7th; rerank put it first. The gold section was never outside the top 50.

It wasn't ranked first in 24 of the 100 answerable rows (14 embedding order, 10 reranked). What was ranked first instead:

| First chunk | Rows |
|---|---|
| Another section of the right label | 22 |
| A sibling label | 2 (q44 in both orders: Imitrex injection's dosing, for a question about Imitrex tablets) |
| A different drug | 0 |

The first chunk was usually a section on the same topic: Patient Counseling Information restating an instruction (q18, q33, q34, q38, q42), Glucotrol XL's Description of its shell above the counseling note about seeing it in stool (q10), or Effient's geriatric-use paragraph above its boxed warning (q21). None of those 24 first chunks holds every key fact word for word, so the gold section was the better passage each time. But it was in the top 5 every time except q26, the only answer retrieval cost.

So the retrieval errors left are about which section of the right label comes first, not which drug. That's why swapping Embed Fast for Pro, or Rerank Pro for Fast, barely moves the numbers here (see the retrieval experiments in the README).

## Citations

24 answers fail the citation check, which asks whether the cited chunks contain every key fact:

| Why | Answers |
|---|---|
| Cites nothing | 21 |
| Cites the right passage, but the answer leaves out a key fact | 2 (q18 in both orders: the 75 to 100 mg aspirin dose) |
| Cites a chunk that supports only part of the answer | 1 (q31 with rerank: "1 mg" from Adverse Reactions, not "once daily") |

Uncited answers rose from 6 and 5 with v1 to 10 and 11 with v3, and almost all of the rise is in answers that open with "Yes" or "No", which v2 and v3 ask for on yes/no questions:

| Prompt | Opens with Yes or No: uncited | Other answers: uncited |
|---|---|---|
| v1 | 1/19 | 10/67 |
| v2 | 14/42 | 4/47 |
| v3 | 16/41 | 5/58 |

Fisher's exact test puts the v3 gap at p = 0.0004, and Yes/No answers went uncited far more often with v3 than with v1 (p = 0.006). Command A+ attaches citations itself; something about a bare "Yes." or "No." at the start seems to make it skip them for the whole answer. Only one of the 21 uncited answers is graded less than right (q34 with embedding only); the rest are right but unverifiable, and for drug information an answer nobody can check is a failure of its own.

## By tag

Both orders, so each answerable question counts twice:

| Tag | Rows | Not right | Gold not first |
|---|---|---|---|
| `knowledge_conflict` | 14 | 4 | 2 |
| `multi_section` | 8 | 2 | 2 |
| `false_premise` | 14 | 2 | 7 |
| `brand_vs_generic` | 8 | 1 | 4 |
| `near_duplicate` | 52 | 4 | 10 |
| (untagged) | 22 | 3 | 7 |
| `lay_terms` | 16 | 0 | 1 |
| `label_comparison` | 10 | 0 | 1 |
| `full_text` | 14 | 0 | 2 |

Questions where the label changed (`knowledge_conflict`) are the weakest group, as the eval set was built to show: two of the seven still fail in some form (q11, Invokana's removed boxed warning; q23, the removed pregnancy contraindication). Of the unanswerable questions, the two `absent_drug` ones (Eliquis, Wegovy) and one `unmentioned_comparison` hit the 422 with rerank; every other unanswerable answer was a correct decline.

## Three examples

**A rule that fixed one thing and broke another (q10).** "I sometimes see what looks like a whole Glucotrol XL tablet in my stool. Does that mean it isn't working?" v1 answered it correctly in both orders. v2 added "If the question asks yes or no, start with Yes or No", and from then on both orders answered "Yes, the label says patients may occasionally notice in their stool something that looks like a tablet… This does not mean the medication isn't working." Every word after "Yes" is right; the "Yes" answers the reader's worry the wrong way round. The same rule is behind the uncited answers above, so it's the first thing to change.

**The label in front of it, the old label in its answer (q23, embedding order).** "Is Lipitor contraindicated in pregnancy?" The pregnancy section was ranked first. It says to discontinue Lipitor when pregnancy is recognized or, alternatively, consider the patient's ongoing therapeutic needs, and pregnancy isn't among the contraindications: FDA had that contraindication removed from statin labels in 2021. The answer began "Yes." With rerank, the same prompt said the excerpts don't list it as contraindicated, but never said no. This is the failure grounding is supposed to prevent, and a citation wouldn't catch it: the answer cites the right passage and still says the opposite.

**No answer at all (u05, u06, u10 with rerank).** Three unanswerable questions, about Eliquis, Wegovy and Nurtec ODT against Reyvow, got Cohere's 422, "No valid response generated", instead of the decline sentence. The embedding-order versions of the same questions declined correctly. Every 422 in the project so far came on an unanswerable question, which suggests the model was heading for a decline the API then rejected. An application could map this error to the decline message; the eval counts it as wrong, because the system didn't decline.

## What it changes

- Drop or reword the yes/no rule: it caused 3 of the 6 right-context errors and most of the uncited answers. One option is to let the label's own words come first ("The label says to take it at least 1 hour before a meal…") and the Yes or No after.
- Handle the 422 on declines, and check that this doesn't hide real failures.
- Keep the label-change questions in every future eval: they fail in ways a citation check can't see.
- Measure retrieval models on a harder setup before choosing between them: a bigger corpus, questions that don't name the drug, or a smaller candidate pool. On this one, retrieval mostly fails over which section of the right label comes first.
