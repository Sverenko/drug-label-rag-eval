# Eval run: baseline

2026-11-08T23:44:57 UTC. 60 questions (50 answerable, 10 not). Prompt `answer_v1`; models `embed-v5.0-pro`, `embed-v5.0-fast`, `rerank-v5.0-pro`, `command-a-plus-05-2026`.

| | Embed only | Embed + Rerank |
|---|---|---|
| Right section in the top 5 (hit@5) | 49/50 (98%) | 50/50 (100%) |
| Right section ranked first (hit@1) | 36/50 (72%) | 40/50 (80%) |
| Right section anywhere in the top 50 | 50/50 (100%) | — |
| Answer correct (key facts) | 32/50 (64%) | 35/50 (70%) |
| Citations that contain the answer | 36/44 (82%) | 36/42 (86%) |
| Unanswerable questions declined | 10/10 (100%) | 9/10 (90%) |
| Answerable questions wrongly declined | 6/50 (12%) | 8/50 (16%) |

Near-duplicate questions where a sibling label's copy of a gold section ranked inside the top 5 and above every gold chunk (a Zepbound chunk ahead of Mounjaro's, say): 1/26 (4%) with embedding only, 0/26 (0%) with rerank.

Citations that contain the answer counts the answers that weren't declined, and asks whether the cited chunks contain every key fact; an answer that cites nothing counts as no. Answers that cited nothing: 6/44 (14%) with embedding only, 5/42 (12%) with rerank. Answers Cohere couldn't generate at all (a 422, no valid response generated), counted as wrong: 0/60 (0%) and 0/60 (0%). The five chunks sent held every key fact for 48/50 (96%) of answerable questions with embedding only and 50/50 (100%) with rerank.

McNemar's exact test, embed only vs embed + rerank, on the same questions:

| Metric | Only embed right | Only rerank right | p |
|---|---|---|---|
| hit@1 | 3 | 7 | 0.344 |
| hit@5 | 0 | 1 | 1.000 |
| answer_correct | 4 | 7 | 0.549 |

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
- answer correct, only embed: q07, q15, q45, q48; only rerank: q02, q13, q17, q25, q26, q29, q34

By tag (answerable questions):

| Tag | n | hit@5 embed | hit@5 rerank | correct embed | correct rerank |
|---|---|---|---|---|---|
| brand_vs_generic | 4 | 3 | 4 | 3 | 4 |
| false_premise | 7 | 7 | 7 | 3 | 2 |
| full_text | 7 | 7 | 7 | 4 | 6 |
| knowledge_conflict | 7 | 7 | 7 | 2 | 3 |
| label_comparison | 5 | 5 | 5 | 2 | 3 |
| lay_terms | 8 | 8 | 8 | 6 | 6 |
| multi_section | 4 | 4 | 4 | 0 | 1 |
| near_duplicate | 26 | 26 | 26 | 15 | 17 |
| (untagged) | 11 | 11 | 11 | 9 | 8 |
