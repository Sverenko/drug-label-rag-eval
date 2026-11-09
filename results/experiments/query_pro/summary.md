# Retrieval experiment: query_pro

Changed: query embeddings from `embed-v5.0-fast` to `embed-v5.0-pro`. Rerank skipped: embedding order only. 50 answerable questions, compared with the retrieval in results/baseline/. No answers are generated.

| | baseline | query_pro |
|---|---|---|
| Embedding order: right section first (hit@1) | 36/50 (72%) | 37/50 (74%) |
| Embedding order: in the top 5 (hit@5) | 49/50 (98%) | 48/50 (96%) |
| Embedding order: in the 50 | 50/50 (100%) | 50/50 (100%) |
| Query embedding, p50 / p95 ms | 82 / 109 | 94 / 106 |
| Vector search, p50 / p95 ms | 0 / 1 | 1 / 1 |

McNemar's exact test, baseline vs query_pro, on the same questions:

| Metric | Only baseline right | Only query_pro right | p |
|---|---|---|---|
| embed hit@1 | q31 | q39, q42 | 1.000 |
| embed hit@5 | q07 | none | 1.000 |

Latency in baseline comes from that run's calls and in query_pro from this one's; a call served from the cache keeps the latency of the original call. All are single calls from the same machine.
