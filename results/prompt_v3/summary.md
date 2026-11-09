# Eval run: prompt_v3

2026-11-09T03:31:51 UTC. 60 questions (50 answerable, 10 not). Prompt `answer_v3`; models `embed-v5.0-pro`, `embed-v5.0-fast`, `rerank-v5.0-pro`, `command-a-plus-05-2026`.

| | Embed only | Embed + Rerank |
|---|---|---|
| Right section in the top 5 (hit@5) | 49/50 (98%) | 50/50 (100%) |
| Right section ranked first (hit@1) | 36/50 (72%) | 40/50 (80%) |
| Right section anywhere in the top 50 | 50/50 (100%) | — |
| Answer correct (hand grades and key facts) | 33/50 (66%) | 36/50 (72%) |
| Answer correct (Claude's grades) | 45/50 (90%) | 47/50 (94%) |
| Answer has every key fact (automatic) | 33/50 (66%) | 36/50 (72%) |
| Citations that contain the answer | 38/49 (78%) | 37/50 (74%) |
| Unanswerable questions declined | 10/10 (100%) | 7/10 (70%) |
| Answerable questions wrongly declined | 1/50 (2%) | 0/50 (0%) |

Answers graded partly right count as wrong in the rows above: 0/50 (0%) and 0/50 (0%) by hand; 2/50 (4%) and 3/50 (6%) by Claude (embedding only, then rerank).

Near-duplicate questions where a sibling label's copy of a gold section ranked inside the top 5 and above every gold chunk (a Zepbound chunk ahead of Mounjaro's, say): 1/26 (4%) with embedding only, 0/26 (0%) with rerank.

Citations that contain the answer counts the answers that weren't declined, and asks whether the cited chunks contain every key fact; an answer that cites nothing counts as no. Answers that cited nothing: 10/49 (20%) with embedding only, 11/50 (22%) with rerank. Answers Cohere couldn't generate at all (a 422, no valid response generated), counted as wrong: 0/60 (0%) and 3/60 (5%). The five chunks sent held every key fact for 48/50 (96%) of answerable questions with embedding only and 50/50 (100%) with rerank.

McNemar's exact test, embed only vs embed + rerank, on the same questions:

| Metric | Only embed right | Only rerank right | p |
|---|---|---|---|
| hit@1 | 3 | 7 | 0.344 |
| hit@5 | 0 | 1 | 1.000 |
| answer_correct | 2 | 5 | 0.453 |
| answer_correct_claude | 0 | 2 | 0.500 |

Latency per stage (ms), embed + rerank pipeline:

| Stage | p50 | p95 |
|---|---|---|
| Embed query | 83 | 98 |
| Vector search | 1 | 1 |
| Rerank | 374 | 454 |
| Generate answer | 1,337 | 2,428 |
| End to end | 1,797 | 2,913 |
| End to end, embed only | 1,448 | 2,755 |

Where the two configurations disagree:

- hit@5, only embed: none; only rerank: q26
- answer correct, only embed: q41, q44; only rerank: q08, q26, q29, q38, q46

By tag (answerable questions):

| Tag | n | hit@5 embed | hit@5 rerank | correct embed | correct rerank |
|---|---|---|---|---|---|
| brand_vs_generic | 4 | 3 | 4 | 2 | 4 |
| false_premise | 7 | 7 | 7 | 5 | 4 |
| full_text | 7 | 7 | 7 | 4 | 4 |
| knowledge_conflict | 7 | 7 | 7 | 5 | 5 |
| label_comparison | 5 | 5 | 5 | 2 | 2 |
| lay_terms | 8 | 8 | 8 | 7 | 7 |
| multi_section | 4 | 4 | 4 | 0 | 1 |
| near_duplicate | 26 | 26 | 26 | 18 | 19 |
| (untagged) | 11 | 11 | 11 | 8 | 8 |

How often the graders agree, over distinct answers:

- Hand vs Claude, as right, partly right or wrong: 11/11 agree, Cohen's kappa 1.00
- Hand vs Claude, counting partly right as wrong: 11/11 agree, Cohen's kappa 1.00
- Hand vs the key-fact check (answerable questions): 1/1 agree, Cohen's kappa 1.00
- Claude vs the key-fact check (answerable questions): 67/98 agree, Cohen's kappa 0.09
