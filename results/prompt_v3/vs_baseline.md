# prompt_v3 vs baseline

Prompt `answer_v1` in baseline, `answer_v3` in prompt_v3; 60 questions. Retrieval is identical, so only the answers differ.

| | baseline, embed | prompt_v3, embed | baseline, rerank | prompt_v3, rerank |
|---|---|---|---|---|
| Answer correct (Claude's grades) | 40/50 (80%) | 45/50 (90%) | 38/50 (76%) | 47/50 (94%) |
| Answer has every key fact (automatic) | 32/50 (64%) | 33/50 (66%) | 35/50 (70%) | 36/50 (72%) |
| Citations that contain the answer | 36/44 (82%) | 38/49 (78%) | 36/42 (86%) | 37/50 (74%) |
| Answers that cited nothing | 6/44 (14%) | 10/49 (20%) | 5/42 (12%) | 11/50 (22%) |
| Unanswerable questions declined | 10/10 (100%) | 10/10 (100%) | 9/10 (90%) | 7/10 (70%) |
| Answerable questions wrongly declined | 6/50 (12%) | 1/50 (2%) | 8/50 (16%) | 0/50 (0%) |

McNemar's exact test, baseline vs prompt_v3, on the same questions:

| Grades | Configuration | Only baseline right | Only prompt_v3 right | p | Right only in prompt_v3 | Right only in baseline |
|---|---|---|---|---|---|---|
| Claude's grades | embed | 1 | 6 | 0.125 | q02, q03, q05, q18, q24, q44 | q10 |
| Claude's grades | rerank | 1 | 10 | 0.012 | q02, q03, q05, q07, q15, q18, q24, q34, q44, q45 | q10 |

Answer correct by hand grades is compared once both runs are fully graded (100 and 1 of 100 answers so far).
Generating an answer took 1,209 ms at the median in baseline and 1,337 ms in prompt_v3 (rerank pipeline).
