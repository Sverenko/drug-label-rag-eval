# prompt_v2 vs baseline

Prompt `answer_v1` in baseline, `answer_v2` in prompt_v2; 60 questions. Retrieval is identical, so only the answers differ.

| | baseline, embed | prompt_v2, embed | baseline, rerank | prompt_v2, rerank |
|---|---|---|---|---|
| Answer correct (Claude's grades) | 40/50 (80%) | 42/50 (84%) | 38/50 (76%) | 43/50 (86%) |
| Answer has every key fact (automatic) | 32/50 (64%) | 35/50 (70%) | 35/50 (70%) | 32/50 (64%) |
| Citations that contain the answer | 36/44 (82%) | 32/44 (73%) | 36/42 (86%) | 36/45 (80%) |
| Answers that cited nothing | 6/44 (14%) | 10/44 (23%) | 5/42 (12%) | 8/45 (18%) |
| Unanswerable questions declined | 10/10 (100%) | 10/10 (100%) | 9/10 (90%) | 8/10 (80%) |
| Answerable questions wrongly declined | 6/50 (12%) | 6/50 (12%) | 8/50 (16%) | 5/50 (10%) |

McNemar's exact test, baseline vs prompt_v2, on the same questions:

| Grades | Configuration | Only baseline right | Only prompt_v2 right | p | Right only in prompt_v2 | Right only in baseline |
|---|---|---|---|---|---|---|
| Claude's grades | embed | 1 | 3 | 0.625 | q02, q05, q34 | q10 |
| Claude's grades | rerank | 1 | 6 | 0.125 | q05, q07, q15, q24, q34, q45 | q10 |

Answer correct by hand grades is compared once both runs are fully graded (100 and 6 of 100 answers so far).
Generating an answer took 1,209 ms at the median in baseline and 1,531 ms in prompt_v2 (rerank pipeline).
