# MaxionBench v0.1 Portable-Agentic Spec

**Paper title:** **MaxionBench-Portable: CPU-Only Local Evaluation of Retrieval Infrastructure for Agentic AI**

## 0) Positioning and evaluation focus

**MaxionBench v0.1** benchmarks **retrieval infrastructure for agentic applications** under the recorded CPU-only local runtime profile.

Primary research question:

> Can a conformance-gated local benchmark produce auditable deployment decisions for agentic retrieval systems?

Evaluation focus:

- MaxionBench evaluates retrieval infrastructure for agentic applications.
- Full agent task success is outside the v0.1 evaluation target.

Paper-path run profile:

- CPU-only local runtime profile
- fixed B0/B1/B2 budget ladder
- frozen `HotpotQA-portable` preprocessing for S3 multi-evidence
- conformance-gated adapter matrix for reportable engines

One-time preprocessing such as building `HotpotQA-portable` is recorded separately from the B0/B1/B2 benchmark budgets.

## 1) Systems under test

Current paper matrix:

- `faiss-cpu`
- `pgvector`
- `qdrant`

Conditional paper matrix:

- `lancedb-inproc` only if it passes conformance on a local filesystem path and uses native Lance table search for vector retrieval

Exploratory-only / excluded for the current submission:

- `lancedb-service`

Eligibility rule:

- if an engine cannot be deployed stably on ARM64 local or fails conformance, exclude it from the reported matrix and record the exclusion in the support table and behavior cards
- paper-facing tables and figures must fail closed when conformance data is missing

Fairness policy:

- server engines are measured over local network APIs
- `lancedb-service` is excluded from the current paper path because there is no deployable local HTTP service in this repository today
- `lancedb-inproc` is a secondary upper-bound reference only when query execution routes through native Lance table search
- `lancedb-inproc` requires a local filesystem path that supports atomic `rename()` (e.g. `/tmp`); external or network-mounted volumes can return `ENOTSUP` and must not be used
- `pgvector` uses exact sequential scan (`index_method=none`) by default; its higher recall vs. HNSW engines is expected and is a valid finding, not an anomaly
- retries are **off** during timed measurement
- unsupported features are recorded, not hidden

## 2) Adapter contract and conformance

Each adapter must implement:

- lifecycle: `create`, `drop`, `reset`, `healthcheck`
- data ops: `bulk_upsert`, `query`, `batch_query`, `insert`, `update_vectors`, `update_payload`, `delete`, `flush_or_commit`
- index/control: `set_index_params`, `set_search_params`, `optimize_or_compact`
- stats: `stats`

Minimum `stats` fields:

- `vector_count`
- `deleted_count`
- `index_size_bytes`
- `ram_usage_bytes`
- `disk_usage_bytes`
- `engine_uptime_s`

Conformance is mandatory before benchmarking and before report generation. It must verify:

- CRUD correctness
- visibility after `flush_or_commit`
- filter correctness
- vector and payload update semantics
- delete semantics
- freshness semantics for `S2` using post-ACK visibility

Each engine must also have a behavior card under `docs/behavior/`.

Reportability rule:

- `included_in_report=True` is allowed only when conformance passes and the behavior card exists

## 3) Datasets

### 3.1 S1 single-hop corpora

- `scifact`
- `fiqa`

These are the paper-path single-hop retrieval datasets.

### 3.2 S2 static background + event stream

Static background corpus:

- deterministic `50K` passage subset from the union of `scifact` and `fiqa`

Online event stream:

- `CRAG-500`
- one inserted supporting passage per event: the leading chunk (`p0_c0`) of the top-ranked search result for each CRAG event
- all other search result chunks are background distractors and carry no qrel entry
- qrel-referenced event evidence docs are retained before any doc cap is applied
- events whose evidence doc ID already exists in the static background corpus are excluded from freshness evaluation

### 3.3 S3 reasoning corpus

Primary dataset:

- `HotpotQA-portable`

`HotpotQA-portable` construction:

- use the official HotpotQA `dev distractor` split
- corpus = deduplicated context paragraphs from the provided `context` field
- qrels = supporting paragraph titles from `supporting_facts`
- retrieval unit = one context paragraph per title
- freeze the processed corpus, queries, qrels, manifest, and checksums

Distractor source:

- use the distractor paragraphs already shipped in the official HotpotQA release
- do **not** use a live Wikipedia API call

HotpotQA note:

- the HotpotQA raw JSON download is a **one-time offline preprocessing dependency**
- the benchmark itself depends only on the frozen `HotpotQA-portable` manifest and extracted passages

Raw source acquisition for the NeurIPS workflow:

- the workflow downloads the official HotpotQA `dev distractor` JSON automatically into:
  - `dataset/D4/hotpotqa/hotpot_dev_distractor_v1.json`
- do not hand-author these inputs and do not use AI-generated stand-ins

Local preparation steps:

1. run the end-to-end data preparation step:

```bash
maxionbench portable-workflow data --json
```

2. after `dataset/processed/hotpot_portable` exists, run:

```bash
maxionbench submit-portable --budget b0 --json
```

Retrieval unit rule:

- use the dataset's canonical retrievable evidence unit for each scenario

## 4) Embedding models

The paper matrix uses exactly two local embedding models:

- lightweight: `BAAI/bge-small-en-v1.5` (`384d`)
- standard: `BAAI/bge-base-en-v1.5` (`768d`)

Embedding choice is a first-class benchmark axis.

## 5) Scenarios

### S1 Single-Hop Corpus Retrieval

- datasets: `scifact`, `fiqa`
- clients: `{1, 4, 8}`
- primary quality: `nDCG@10`

Goal:

- measure single-hop retrieval quality, latency, throughput, and cost across engines and embedding choices

### S2 Streaming Memory

- background corpus: deterministic `50K` `scifact` + `fiqa` subset
- event stream: `CRAG-500`
- the 50K cap applies to the static background corpus, not to event evidence retention
- clients: read `8`, write `2`
- primary quality: `nDCG@10` on static background queries
- freshness metrics are evaluated in addition to `nDCG@10`

Freshness definition:

- let `T` be the time the insert acknowledgment is received from the engine, after the adapter's `insert + flush_or_commit` path returns
- issue freshness probes at `T+1s` and `T+5s`
- freshness SLA: newly inserted supporting evidence should surface in the top-10 retrieval results within `5s`
- freshness probes must target evidence not already present in the background corpus
- runs are rejected if fewer than `90%` of raw CRAG event queries retain usable evidence after loader filtering

### S3 Multi-Hop Evidence Retrieval

- dataset: `HotpotQA-portable`
- clients: `{1, 4, 8}`
- primary quality: `evidence_coverage@10`
- secondary quality: `evidence_coverage@5`, `evidence_coverage@20`

Goal:

- measure whether a retriever returns the full supporting evidence set within a realistic fast-agent context budget

## 6) Evaluation budgets and sweep protocol

Budget ladder:

- `B0` screening: `10s` warmup, `10s` measure, `1` repeat
- `B1` reduced: `15s` warmup, `30s` measure, `1` repeat
- `B2` full portable: `30s` warmup, `60s` measure, `2` repeats

Reportable quality floors:

- `S1`: `nDCG@10 >= 0.25`
- `S2`: `nDCG@10 >= 0.25`
- `S3`: `evidence_coverage@10 >= 0.30`

Promotion rules:

- `B0 -> B1`:
  - `S1`: `nDCG@10 >= 0.75 x 0.25`
  - `S2`: `nDCG@10 >= 0.75 x 0.25` and `freshness_hit@5s >= 0.6`
  - `S3`: `evidence_coverage@10 >= 0.75 x 0.30`
  - error rate `<= 5%`
- `B1 -> B2`:
  - `S1`: `nDCG@10 >= 0.9 x 0.25`
  - `S2`: `nDCG@10 >= 0.9 x 0.25` and `freshness_hit@5s >= 0.8`
  - `S3`: `evidence_coverage@10 >= 0.9 x 0.30`
  - error rate `<= 5%`

Pruning rule:

- if more than `3` configs survive after `B1`, keep the `3` lowest `task_cost_est`

Final ranking rule:

- for each engine / embedding / scenario cell, rank surviving configs by lowest `task_cost_est`
- tie-break by lower `p99`, then higher throughput

Scenario-specific stability rule:

- `S1` and `S2` are expected to stabilize under lower budgets if the rerun confirms it
- `S3` is allowed to require `B2`; this is a valid reported finding, not a protocol failure
- report decision stability separately from full-rank stability; a stable top-1 recommendation can coexist with noisy lower-rank orderings

## 7) Metrics

### 7.1 Quality

- `S1`: `nDCG@10`
- `S2`: `nDCG@10` on static queries
- `S3`: `evidence_coverage@10` with secondary reports at `k=5,20`

### 7.2 Freshness

- `freshness_hit@1s`
- `freshness_hit@5s`
- `stale_answer_rate@5s`

Current-version note:

- with exactly one inserted supporting passage per CRAG event, `stale_answer_rate@5s = 1 - freshness_hit@5s`
- both are still reported because `freshness_hit` is a success metric and `stale_answer_rate` is a risk metric
- fixed probes measure post-insert top-k retrievability, not direct lower-level index visibility

### 7.3 Systems

- `p50`, `p95`, `p99`
- throughput
- error rate
- build / load time
- RAM / disk footprint

### 7.4 Cost

- `task_cost_est = retrieval_cost + embedding_cost + llm_context_cost`
- `llm_context_cost = c_llm_in x retrieved_input_tokens`
- store `c_llm_in` in config and run metadata

### 7.5 Decision stability

- Spearman `rho` between budget-level rankings
- top-1 agreement
- top-2 agreement

## 8) Reproducibility rules

- fixed seeds everywhere
- no hidden retries
- record exact ARM64 local model, RAM, local OS version, Docker version, and Python environment
- conformance must pass before benchmark execution
- conformance must be run in the same environment state as the benchmark: same env vars, same filesystem paths, same services up
- `MAXIONBENCH_LANCEDB_INPROC_URI` must resolve to a local filesystem that supports `rename()`; the setup command defaults to `/tmp/maxionbench/lancedb/inproc` if the env var is unset
- publish the frozen `HotpotQA-portable` manifest and checksums

## 9) Required artifacts per run

Write:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

Minimum metadata:

- `run_id`, `timestamp_utc`
- `engine`, `engine_version`
- `embedding_model`
- `scenario`, `dataset_bundle`, `dataset_hash`
- `budget_level`
- `seed`
- client counts
- primary quality metric
- freshness metrics when applicable
- overlap-skipped freshness event counts when applicable
- raw and usable `S2` event counts when applicable
- `task_cost_est`
- `c_llm_in`
- hardware / runtime summary

## 10) Critical-path implementation order

1. ✅ update schemas and configs for the portable-agentic profile
2. ✅ add `HotpotQA-portable` preprocessing and manifest generation
3. ✅ implement `S1`
4. ✅ implement `S2` freshness semantics and conformance checks; qrel-first event doc loading; 90% sparse-event guard
5. ✅ implement `S3`
6. ✅ add budget-ladder runner and ranking stability analysis
7. ✅ fix lancedb-inproc to use `/tmp` default URI; add `rename()` preflight check
8. ✅ exclude lancedb-service from paper path; remove delegate-mode config
9. ✅ add reportability gate (conformance pass required; quality floor filter)
10. ✅ run full paper matrix (`B0`, `B1`, `B2`) for all scenarios and engines
11. ✅ generate final figures and the minimum viable deployment table

## 11) Paper outputs

Required results:

- budget-level stability curves (`B0`, `B1`, `B2`)
- scenario-wise engine / embedding rankings
- `S2` freshness behavior comparison
- support table showing reportable vs excluded engines with exclusion reasons
- minimum viable deployment table:
  - workload type
  - minimum engine
  - recommended embedding tier
  - justification grounded in measured quality, freshness, latency, and `task_cost_est`
- minimum viable deployment sensitivity table:
  - recompute the recommendation under `p99_max` caps of `100ms`, `200ms`, `500ms`, and no cap
  - treat the main table as the `200ms` deployment-SLA recommendation
  - report winner changes as SLA sensitivity
