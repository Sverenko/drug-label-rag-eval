# Retrieval experiment: rerank_fast

Changed: rerank from `rerank-v5.0-pro` to `rerank-v5.0-fast`. 50 answerable questions, compared with the retrieval in results/baseline/. No answers are generated.

| | baseline | rerank_fast |
|---|---|---|
| Embedding order: right section first (hit@1) | 36/50 (72%) | 36/50 (72%) |
| Embedding order: in the top 5 (hit@5) | 49/50 (98%) | 49/50 (98%) |
| Embedding order: in the 50 | 50/50 (100%) | 50/50 (100%) |
| Reranked: right section first (hit@1) | 40/50 (80%) | 41/50 (82%) |
| Reranked: in the top 5 (hit@5) | 50/50 (100%) | 49/50 (98%) |
| Query embedding, p50 / p95 ms | 82 / 109 | 82 / 109 |
| Vector search, p50 / p95 ms | 0 / 1 | 1 / 1 |
| Rerank, p50 / p95 ms | 374 / 453 | 263 / 364 |

McNemar's exact test, baseline vs rerank_fast, on the same questions:

| Metric | Only baseline right | Only rerank_fast right | p |
|---|---|---|---|
| embed hit@1 | none | none | 1.000 |
| embed hit@5 | none | none | 1.000 |
| rerank hit@1 | q05, q07, q16 | q01, q10, q20, q38 | 1.000 |
| rerank hit@5 | q05 | none | 1.000 |

Latency in baseline comes from that run's calls and in rerank_fast from this one's; a call served from the cache keeps the latency of the original call. All are single calls from the same machine.
