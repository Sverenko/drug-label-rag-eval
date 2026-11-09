# Eval run: prompt_v2

2026-11-09T03:12:56 UTC. 60 questions (50 answerable, 10 not). Prompt `answer_v2`; models `embed-v5.0-pro`, `embed-v5.0-fast`, `rerank-v5.0-pro`, `command-a-plus-05-2026`.

| | Embed only | Embed + Rerank |
|---|---|---|
| Right section in the top 5 (hit@5) | 49/50 (98%) | 50/50 (100%) |
| Right section ranked first (hit@1) | 36/50 (72%) | 40/50 (80%) |
| Right section anywhere in the top 50 | 50/50 (100%) | — |
| Answer correct (hand grades and key facts) | 35/50 (70%) | 32/50 (64%) |
| Answer correct (Claude's grades) | 42/50 (84%) | 43/50 (86%) |
| Answer has every key fact (automatic) | 35/50 (70%) | 32/50 (64%) |
| Citations that contain the answer | 32/44 (73%) | 36/45 (80%) |
| Unanswerable questions declined | 10/10 (100%) | 8/10 (80%) |
| Answerable questions wrongly declined | 6/50 (12%) | 5/50 (10%) |

Answers graded partly right count as wrong in the rows above: 1/50 (2%) and 0/50 (0%) by hand; 3/50 (6%) and 5/50 (10%) by Claude (embedding only, then rerank).

Near-duplicate questions where a sibling label's copy of a gold section ranked inside the top 5 and above every gold chunk (a Zepbound chunk ahead of Mounjaro's, say): 1/26 (4%) with embedding only, 0/26 (0%) with rerank.

Citations that contain the answer counts the answers that weren't declined, and asks whether the cited chunks contain every key fact; an answer that cites nothing counts as no. Answers that cited nothing: 10/44 (23%) with embedding only, 8/45 (18%) with rerank. Answers Cohere couldn't generate at all (a 422, no valid response generated), counted as wrong: 0/60 (0%) and 2/60 (3%). The five chunks sent held every key fact for 48/50 (96%) of answerable questions with embedding only and 50/50 (100%) with rerank.

McNemar's exact test, embed only vs embed + rerank, on the same questions:

| Metric | Only embed right | Only rerank right | p |
|---|---|---|---|
| hit@1 | 3 | 7 | 0.344 |
| hit@5 | 0 | 1 | 1.000 |
| answer_correct | 5 | 2 | 0.453 |
| answer_correct_claude | 1 | 2 | 1.000 |

Latency per stage (ms), embed + rerank pipeline:

| Stage | p50 | p95 |
|---|---|---|
| Embed query | 83 | 98 |
| Vector search | 1 | 1 |
| Rerank | 374 | 454 |
| Generate answer | 1,531 | 2,726 |
| End to end | 2,026 | 3,183 |
| End to end, embed only | 1,570 | 2,652 |

Where the two configurations disagree:

- hit@5, only embed: none; only rerank: q26
- answer correct, only embed: q06, q13, q19, q27, q46; only rerank: q17, q26

By tag (answerable questions):

| Tag | n | hit@5 embed | hit@5 rerank | correct embed | correct rerank |
|---|---|---|---|---|---|
| brand_vs_generic | 4 | 3 | 4 | 3 | 3 |
| false_premise | 7 | 7 | 7 | 2 | 2 |
| full_text | 7 | 7 | 7 | 6 | 4 |
| knowledge_conflict | 7 | 7 | 7 | 3 | 3 |
| label_comparison | 5 | 5 | 5 | 4 | 3 |
| lay_terms | 8 | 8 | 8 | 7 | 5 |
| multi_section | 4 | 4 | 4 | 0 | 0 |
| near_duplicate | 26 | 26 | 26 | 17 | 16 |
| (untagged) | 11 | 11 | 11 | 8 | 8 |

How often the graders agree, over distinct answers:

- Hand vs Claude, as right, partly right or wrong: 13/14 agree, Cohen's kappa 0.72
- Hand vs Claude, counting partly right as wrong: 14/14 agree, Cohen's kappa 1.00
- Hand vs the key-fact check (answerable questions): 4/4 agree, Cohen's kappa 1.00
- Claude vs the key-fact check (answerable questions): 71/95 agree, Cohen's kappa 0.32
