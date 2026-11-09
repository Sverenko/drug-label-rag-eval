# prompt_v3 vs prompt_v2

Prompt `answer_v2` in prompt_v2, `answer_v3` in prompt_v3; 60 questions. Retrieval is identical, so only the answers differ.

| | prompt_v2, embed | prompt_v3, embed | prompt_v2, rerank | prompt_v3, rerank |
|---|---|---|---|---|
| Answer correct (Claude's grades) | 42/50 (84%) | 45/50 (90%) | 43/50 (86%) | 47/50 (94%) |
| Answer has every key fact (automatic) | 35/50 (70%) | 33/50 (66%) | 32/50 (64%) | 36/50 (72%) |
| Citations that contain the answer | 32/44 (73%) | 38/49 (78%) | 36/45 (80%) | 37/50 (74%) |
| Answers that cited nothing | 10/44 (23%) | 10/49 (20%) | 8/45 (18%) | 11/50 (22%) |
| Unanswerable questions declined | 10/10 (100%) | 10/10 (100%) | 8/10 (80%) | 7/10 (70%) |
| Answerable questions wrongly declined | 6/50 (12%) | 1/50 (2%) | 5/50 (10%) | 0/50 (0%) |

McNemar's exact test, prompt_v2 vs prompt_v3, on the same questions:

| Grades | Configuration | Only prompt_v2 right | Only prompt_v3 right | p | Right only in prompt_v3 | Right only in prompt_v2 |
|---|---|---|---|---|---|---|
| Claude's grades | embed | 1 | 4 | 0.375 | q03, q18, q24, q44 | q34 |
| Claude's grades | rerank | 0 | 4 | 0.125 | q02, q03, q18, q44 | none |

Answer correct by hand grades is compared once both runs are fully graded (6 and 1 of 100 answers so far).
Generating an answer took 1,531 ms at the median in prompt_v2 and 1,337 ms in prompt_v3 (rerank pipeline).
