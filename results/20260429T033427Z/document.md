# MaxionBench v0.1 Technical Document and Paper Blueprint

This document gives the technical and mathematical specification for **MaxionBench-Portable**, consistent with `project.md`.

## 1) Problem formulation

### 1.1 Objective

MaxionBench evaluates **retrieval infrastructure for agentic applications** under a recorded CPU-only local runtime profile.

Primary question:

> Can a conformance-gated local benchmark produce auditable deployment decisions for agentic retrieval systems?

Evaluation focus:

- the benchmark evaluates retrieval infrastructure
- full agent task success is outside the v0.1 evaluation target

### 1.2 Evaluation universe

Let:

- \( \mathcal{E} \): set of eligible engines
- \( \mathcal{M} \): set of embedding models
- \( \mathcal{S} = \{S1, S2, S3\} \): set of scenarios
- \( \theta \in \Theta_{e,m,s} \): engine / embedding / scenario configuration

Define:

- quality \( Q_{e,m,s}(\theta) \)
- performance vector \( P_{e,m,s}(\theta) = (\mathrm{p50}, \mathrm{p95}, \mathrm{p99}, \mathrm{QPS}) \)
- task cost \( C_{e,m,s}(\theta) \)

For each \((e,m,s)\), the selected benchmark winner is the surviving configuration with minimum task cost:

\[
\theta^*_{e,m,s} = \arg\min_{\theta \in \Theta^{\mathrm{survive}}_{e,m,s}} C_{e,m,s}(\theta)
\]

Tie-break:

1. lower `p99`
2. higher throughput

### 1.3 Budget-level stability

Let \(R_B\) be the ranking of engine / embedding pairs under budget level \(B \in \{B0,B1,B2\}\).

Decision stability is reported with:

- Spearman rank correlation \( \rho(R_{B_i}, R_{B_j}) \)
- top-1 agreement
- top-2 agreement

The paper asks whether rankings stabilize sufficiently by `B2`, and how much is already visible at `B0` and `B1`.

## 2) Systems under test and fairness

### 2.1 Engines

Current paper matrix:

- `faiss-cpu`
- `lancedb-inproc`
- `pgvector`
- `qdrant`

Excluded from the current paper matrix:

- `lancedb-service`

Eligibility:

- an engine must be deployable stably in the benchmark environment and pass conformance
- excluded engines are documented explicitly

### 2.2 Fairness policy

- server engines are measured over local network APIs
- `lancedb-service` is excluded from the current submission because no locally deployable HTTP service binary was available in this repository at evaluation time
- `lancedb-inproc` is included as a secondary embedded reference
- no hidden retries during timed runs
- unsupported features are documented, not silently bypassed

## 3) Adapter contract and conformance

### 3.1 Required interface

Lifecycle:

- `create`, `drop`, `reset`, `healthcheck`

Data operations:

- `bulk_upsert`, `query`, `batch_query`
- `insert`, `update_vectors`, `update_payload`, `delete`
- `flush_or_commit`

Index / control:

- `set_index_params`, `set_search_params`, `optimize_or_compact`

Stats:

- `stats`

Minimum `stats` schema:

- `vector_count`
- `deleted_count`
- `index_size_bytes`
- `ram_usage_bytes`
- `disk_usage_bytes`
- `engine_uptime_s`

### 3.2 Conformance requirements

Conformance must verify:

- CRUD correctness
- visibility after `flush_or_commit`
- filter correctness
- update semantics
- delete semantics
- `S2` freshness semantics using post-ACK visibility

### 3.3 Behavior cards

Each engine must document:

- visibility semantics
- delete model
- update semantics
- freshness behavior
- compaction behavior
- persistence guarantees
- limitations

## 4) Dataset contracts

### 4.1 S1 single-hop corpora

- `scifact`
- `fiqa`

Retrieval uses each dataset's canonical corpus unit.

### 4.2 S2 streaming memory corpus

Static background:

- deterministic `50K` subset from the union of `scifact` and `fiqa`

Online events:

- `CRAG-500`
- each event inserts exactly one supporting passage

Retrieval uses the same static background plus the inserted event passages.

### 4.3 S3 `HotpotQA-portable`

Use the official HotpotQA `dev distractor` split.

Corpus construction:

1. load the official HotpotQA `dev distractor` JSON
2. extract one retrievable paragraph per context title
3. map supporting-fact titles to paragraph-level qrels
4. deduplicate context paragraphs
5. freeze the processed corpus, queries, qrels, manifest, and checksums

One-time preprocessing note:

- downloading the raw HotpotQA file is an offline preprocessing step
- benchmark execution depends only on the frozen `HotpotQA-portable` artifact

## 5) Embedding models

Pinned embedding choices:

- `BAAI/bge-small-en-v1.5` (`384d`)
- `BAAI/bge-base-en-v1.5` (`768d`)

Embedding choice is part of the experimental matrix and affects:

- quality
- latency
- index size
- `task_cost_est`

## 6) Scenario formalization

### 6.1 Concurrency pins

- `S1`: clients `{1, 4, 8}`
- `S2`: static background read clients `8`; freshness event stream `2` write clients, each issuing serialized post-ACK inserts
- `S3`: clients `{1, 4, 8}`

### 6.2 S1 Single-Hop Corpus Retrieval

Datasets:

- `scifact`
- `fiqa`

Primary quality:

- `nDCG@10`

Measure:

- quality
- latency
- throughput
- build / load time
- footprint
- cost

### 6.3 S2 Streaming Memory

Static quality:

- `nDCG@10` on background queries over the current corpus snapshot

Freshness timeline:

- insert event occurs
- adapter executes `insert + flush_or_commit`
- let \(T\) be the time the insert acknowledgment is received
- freshness probes occur at \(T+1s\) and \(T+5s\)

Freshness SLA:

- newly inserted supporting evidence should surface in the top-10 retrieval results within `5s` of ACK

### 6.4 S3 Multi-Hop Evidence Retrieval

Dataset:

- `HotpotQA-portable`

Primary quality:

- `evidence_coverage@10`

Secondary reports:

- `evidence_coverage@5`
- `evidence_coverage@20`

Interpretation:

- `k=10` corresponds to a fast-agent context budget on the order of a few thousand input tokens

## 7) Metrics and formulas

### 7.1 `nDCG@10`

\[
\mathrm{DCG@10}(q)=\sum_{i=1}^{10}\frac{2^{\mathrm{rel}_{q,i}}-1}{\log_2(i+1)}, \qquad
\mathrm{nDCG@10}(q)=\frac{\mathrm{DCG@10}(q)}{\mathrm{IDCG@10}(q)}
\]

### 7.2 `evidence_coverage@k`

Let \(E(q)\) be the gold evidence set for query \(q\), and let \(R_k(q)\) be the top-\(k\) retrieved items.

\[
\mathrm{evidence\_coverage@k}(q)=\frac{|E(q)\cap R_k(q)|}{|E(q)|}
\]

The run-level score is the mean over queries.

### 7.3 Freshness metrics

For each `S2` event, let \(e_q\) be the inserted supporting passage for the paired query \(q\).

Freshness hit:

\[
\mathrm{freshness\_hit@\Delta}(q)=
\begin{cases}
1, & e_q \in R_{10}(q, T+\Delta) \\
0, & \text{otherwise}
\end{cases}
\]

with \(\Delta \in \{1s,5s\}\).

Top-k surfacing latency:

\[
\mathrm{topk\_surfacing\_latency}(q)=\inf \{ \delta \ge 0 \mid e_q \in R_{10}(q,T+\delta) \}
\]

The current report uses fixed probes at `T+1s` and `T+5s`. These probes measure post-insert top-k retrievability, not a direct lower-level index visibility check.

Stale answer rate:

\[
\mathrm{stale\_answer\_rate@5s}=\frac{N_{\mathrm{stale}}}{N_{\mathrm{freshness\_queries}}}
\]

where a query is stale if its supporting passage was inserted before the query and is absent from the top-10 at \(T+5s\).

Current-version note:

- because each event inserts exactly one supporting passage, `stale_answer_rate@5s = 1 - freshness_hit@5s`
- both are still reported because they support different interpretations

### 7.4 Performance

For latency samples \(L=\{l_1,\dots,l_n\}\):

- `p50`, `p95`, `p99` are empirical quantiles

Throughput:

\[
\mathrm{QPS}=\frac{N_{\mathrm{success}}}{T_{\mathrm{wall}}}
\]

### 7.5 Task cost

Define:

\[
\mathrm{task\_cost\_est} = C_{\mathrm{retrieval}} + C_{\mathrm{embedding}} + C_{\mathrm{llm\_context}}
\]

with:

\[
C_{\mathrm{llm\_context}} = c_{\mathrm{llm\_in}} \cdot N_{\mathrm{retrieved\_input\_tokens}}
\]

Rules:

- store `c_llm_in` in config and metadata
- main paper matrix uses local embedding models, so `C_embedding` is measured local compute cost
- a worked appendix example may map token cost to a published API pricing snapshot

### 7.6 Decision stability

For two budget levels \(B_i, B_j\):

- Spearman \( \rho(R_{B_i}, R_{B_j}) \)
- top-1 agreement
- top-2 agreement

## 8) Budget ladder and survivor rules

Budget levels:

- `B0`: `10s` warmup, `10s` measure, `1` repeat
- `B1`: `15s` warmup, `30s` measure, `1` repeat
- `B2`: `30s` warmup, `60s` measure, `2` repeats

Reportable quality floors:

- `S1`: `nDCG@10 >= 0.25`
- `S2`: `nDCG@10 >= 0.25`
- `S3`: `evidence_coverage@10 >= 0.30`

Promotion:

- `B0 -> B1`
  - `S1`: `nDCG@10 >= 0.1875`
  - `S2`: `nDCG@10 >= 0.1875` and `freshness_hit@5s >= 0.6`
  - `S3`: `evidence_coverage@10 >= 0.225`
  - error rate `<= 5%`
- `B1 -> B2`
  - `S1`: `nDCG@10 >= 0.225`
  - `S2`: `nDCG@10 >= 0.225` and `freshness_hit@5s >= 0.8`
  - `S3`: `evidence_coverage@10 >= 0.27`
  - error rate `<= 5%`

If more than `3` configs survive after `B1`, keep the `3` lowest `task_cost_est`.

## 9) Reproducibility protocol

- fixed seeds
- no hidden retries
- conformance before benchmarking
- exact hardware / runtime metadata for the single-node host
- publish the frozen `HotpotQA-portable` manifest and checksums

Required per-run artifacts:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

## 10) Paper blueprint

### 10.1 Core contribution

The paper contribution is **evaluation methodology**:

- portable agentic retrieval benchmarking
- post-ACK top-k freshness measurement for streaming memory
- decision stability across benchmark budgets, distinguishing deployment decisions from full-rank stability

### 10.2 Required figures

- `F1`: ranking stability across `B0`, `B1`, `B2`
- `F2`: `S1` quality / latency / cost by engine and embedding
- `F3`: `S2` freshness comparison (`freshness_hit`, `stale_answer_rate`)
- `F4`: `S3` evidence coverage by engine and embedding

### 10.3 Required table

Minimum viable deployment table:

- workload type
- minimum engine
- recommended embedding tier
- justification grounded in measured quality, freshness, latency, and cost

Minimum viable deployment sensitivity table:

- recompute the recommendation under `p99_max` caps of `100ms`, `200ms`, `500ms`, and no cap
- interpret the main table as the `200ms` deployment-SLA recommendation
- report any winner changes as SLA sensitivity rather than as contradictions

## 11) Threats to validity

- single-node scope limits generalization
- ARM64 local deployment may exclude some engines or narrow some feature sets
- `HotpotQA-portable` is a bounded proxy for full-Wikipedia multi-hop retrieval
- `task_cost_est` depends on the configured `c_llm_in`, though the formula itself is stable

This is acceptable because the paper claim is about **portable decision quality**, not about reproducing a full-cluster leaderboard.
