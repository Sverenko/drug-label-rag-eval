# Retrieval experiment: no_header

Changed: chunks embedded and reranked without their drug-and-section header (a separate index, data/index-no-header/). 50 answerable questions, compared with the retrieval in results/baseline/. No answers are generated.

| | baseline | no_header |
|---|---|---|
| Embedding order: right section first (hit@1) | 36/50 (72%) | 35/50 (70%) |
| Embedding order: in the top 5 (hit@5) | 49/50 (98%) | 48/50 (96%) |
| Embedding order: in the 50 | 50/50 (100%) | 50/50 (100%) |
| Reranked: right section first (hit@1) | 40/50 (80%) | 39/50 (78%) |
| Reranked: in the top 5 (hit@5) | 50/50 (100%) | 48/50 (96%) |
| Query embedding, p50 / p95 ms | 82 / 109 | 82 / 109 |
| Vector search, p50 / p95 ms | 0 / 1 | 1 / 1 |
| Rerank, p50 / p95 ms | 374 / 453 | 362 / 438 |

McNemar's exact test, baseline vs no_header, on the same questions:

| Metric | Only baseline right | Only no_header right | p |
|---|---|---|---|
| embed hit@1 | q08, q19, q31, q46 | q10, q39, q42 | 1.000 |
| embed hit@5 | q31 | none | 1.000 |
| rerank hit@1 | q26, q37, q46 | q10, q38 | 1.000 |
| rerank hit@5 | q26, q31 | none | 0.500 |

Latency in baseline comes from that run's calls and in no_header from this one's; a call served from the cache keeps the latency of the original call. All are single calls from the same machine.
