# Eval run: baseline

2026-11-08T23:44:57 UTC. 60 questions (50 answerable, 10 not). Prompt `answer_v1`; models `embed-v5.0-pro`, `embed-v5.0-fast`, `rerank-v5.0-pro`, `command-a-plus-05-2026`.

| | Embed only | Embed + Rerank |
|---|---|---|
| Right section in the top 5 (hit@5) | 49/50 (98%) | 50/50 (100%) |
| Right section ranked first (hit@1) | 36/50 (72%) | 40/50 (80%) |
| Right section anywhere in the top 50 | 50/50 (100%) | — |
| Answer correct (hand grades) | 42/50 (84%) | 42/50 (84%) |
| Answer correct (Claude's grades) | 40/50 (80%) | 38/50 (76%) |
| Answer has every key fact (automatic) | 32/50 (64%) | 35/50 (70%) |
| Citations that contain the answer | 36/44 (82%) | 36/42 (86%) |
| Unanswerable questions declined | 10/10 (100%) | 9/10 (90%) |
| Answerable questions wrongly declined | 6/50 (12%) | 8/50 (16%) |

Answers graded partly right count as wrong in the rows above: 4/50 (8%) and 3/50 (6%) by hand; 3/50 (6%) and 4/50 (8%) by Claude (embedding only, then rerank).

Near-duplicate questions where a sibling label's copy of a gold section ranked inside the top 5 and above every gold chunk (a Zepbound chunk ahead of Mounjaro's, say): 1/26 (4%) with embedding only, 0/26 (0%) with rerank.

Citations that contain the answer counts the answers that weren't declined, and asks whether the cited chunks contain every key fact; an answer that cites nothing counts as no. Answers that cited nothing: 6/44 (14%) with embedding only, 5/42 (12%) with rerank. Answers Cohere couldn't generate at all (a 422, no valid response generated), counted as wrong: 0/60 (0%) and 0/60 (0%). The five chunks sent held every key fact for 48/50 (96%) of answerable questions with embedding only and 50/50 (100%) with rerank.

McNemar's exact test, embed only vs embed + rerank, on the same questions:

| Metric | Only embed right | Only rerank right | p |
|---|---|---|---|
| hit@1 | 3 | 7 | 0.344 |
| hit@5 | 0 | 1 | 1.000 |
| answer_correct | 2 | 2 | 1.000 |
| answer_correct_claude | 3 | 1 | 0.625 |

Latency per stage (ms), embed + rerank pipeline:

| Stage | p50 | p95 |
|---|---|---|
| Embed query | 83 | 98 |
| Vector search | 0 | 1 |
| Rerank | 374 | 454 |
| Generate answer | 1,209 | 2,426 |
| End to end | 1,684 | 2,943 |
| End to end, embed only | 1,246 | 2,257 |

Where the two configurations disagree:

- hit@5, only embed: none; only rerank: q26
- answer correct, only embed: q15, q45; only rerank: q02, q26

By tag (answerable questions):

| Tag | n | hit@5 embed | hit@5 rerank | correct embed | correct rerank |
|---|---|---|---|---|---|
| brand_vs_generic | 4 | 3 | 4 | 3 | 4 |
| false_premise | 7 | 7 | 7 | 4 | 4 |
| full_text | 7 | 7 | 7 | 7 | 7 |
| knowledge_conflict | 7 | 7 | 7 | 3 | 4 |
| label_comparison | 5 | 5 | 5 | 5 | 4 |
| lay_terms | 8 | 8 | 8 | 7 | 7 |
| multi_section | 4 | 4 | 4 | 2 | 2 |
| near_duplicate | 26 | 26 | 26 | 21 | 22 |
| (untagged) | 11 | 11 | 11 | 10 | 9 |

How often the graders agree, over distinct answers:

- Hand vs Claude, as right, partly right or wrong: 94/102 agree, Cohen's kappa 0.71
- Hand vs Claude, counting partly right as wrong: 97/102 agree, Cohen's kappa 0.81
- Hand vs the key-fact check (answerable questions): 74/91 agree, Cohen's kappa 0.47
- Claude vs the key-fact check (answerable questions): 73/91 agree, Cohen's kappa 0.47
