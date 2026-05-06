# MaxionBench v0.1 Execution Prompt

## 0) Mission

Implement **MaxionBench-Portable** as a reproducible CPU-only local benchmark for **retrieval infrastructure for agentic applications** under the recorded ARM64 local runtime profile.

Primary question:

> Can a conformance-gated local benchmark produce auditable deployment decisions for agentic retrieval systems?

Evaluation focus:

- evaluate retrieval infrastructure for agentic applications
- keep full agent task success outside the v0.1 evaluation target

## 1) Source of truth and precedence

Use both docs for every task:

1. `project.md`
2. `prompt.md`

If any conflict appears, follow `project.md` and update this file in a dedicated follow-up change.

## 2) Hard constraints

### 2.1 Runtime profile

- CPU-only local runtime profile
- B0/B1/B2 budget ladder after one-time preprocessing
- frozen preprocessing artifacts for S3 multi-evidence
- conformance-gated adapter matrix for reportable engines

### 2.2 Engine set

Default paper matrix:

- `faiss-cpu`
- `lancedb-inproc`
- `lancedb-service`
- `pgvector`
- `qdrant`

Eligibility:

- if an engine is not stable on ARM64 local or fails conformance, exclude it and record the exclusion explicitly

### 2.3 Embedding pins

- lightweight: `BAAI/bge-small-en-v1.5` (`384d`)
- standard: `BAAI/bge-base-en-v1.5` (`768d`)

### 2.4 Dataset pins

`S1`:

- `scifact`
- `fiqa`

`S2`:

- deterministic `50K` static background subset from `scifact + fiqa`
- `CRAG-500` event stream
- one inserted supporting passage per event

`S3`:

- `FRAMES-portable`
- all FRAMES questions
- corpus = gold evidence passages + `6` same-page non-gold passages + `6` cross-question gold passages
- distractors sourced from the `KILT` Wikipedia dump
- deduplicate by normalized URL + text hash
- freeze seed, manifest, and checksums

KILT note:

- downloading the KILT dump is a **one-time offline preprocessing step**
- the benchmark itself uses only the frozen `FRAMES-portable` extracted passages and manifest

### 2.5 Scenario pins

`S1` single-hop corpus retrieval:

- clients `{1, 4, 8}`
- primary quality `nDCG@10`

`S2` streaming memory:

- static background read clients: `8`
- freshness event stream: `2` write clients, each issuing serialized post-ACK CRAG inserts
- primary quality `nDCG@10` on static background queries
- `T` = insert acknowledgment time after `insert + flush_or_commit` returns
- freshness probes at `T+1s` and `T+5s`

`S3` multi-hop evidence retrieval:

- clients `{1, 4, 8}`
- primary quality `evidence_coverage@10`
- secondary quality `evidence_coverage@5`, `evidence_coverage@20`

### 2.6 Budget pins

- `B0`: warmup `10s`, measure `10s`, repeats `1`
- `B1`: warmup `15s`, measure `30s`, repeats `1`
- `B2`: warmup `30s`, measure `60s`, repeats `2`
- retries are **off** during timed measurement

### 2.7 Promotion rules

Reportable floors:

- `S1`: `nDCG@10 >= 0.25`
- `S2`: `nDCG@10 >= 0.25`
- `S3`: `evidence_coverage@10 >= 0.30`

`B0 -> B1`:

- `S1`: `nDCG@10 >= 0.1875`
- `S2`: `nDCG@10 >= 0.1875` and `freshness_hit@5s >= 0.6`
- `S3`: `evidence_coverage@10 >= 0.225`
- error rate `<= 5%`

`B1 -> B2`:

- `S1`: `nDCG@10 >= 0.225`
- `S2`: `nDCG@10 >= 0.225` and `freshness_hit@5s >= 0.8`
- `S3`: `evidence_coverage@10 >= 0.27`
- error rate `<= 5%`

If more than `3` configs survive after `B1`, keep the `3` lowest `task_cost_est`.

## 3) Measurement protocol

### 3.1 Core metrics

`S1`:

- `nDCG@10`

`S2`:

- `nDCG@10` on static queries
- `freshness_hit@1s`
- `freshness_hit@5s`
- `p95_visibility_latency`
- `stale_answer_rate@5s`

`S3`:

- `evidence_coverage@10`
- `evidence_coverage@5`
- `evidence_coverage@20`

All scenarios:

- `p50`, `p95`, `p99`
- throughput
- error rate
- RAM / disk footprint
- build / load time

### 3.2 Cost metric

Use:

`task_cost_est = retrieval_cost + embedding_cost + llm_context_cost`

with:

`llm_context_cost = c_llm_in x retrieved_input_tokens`

Rules:

- store `c_llm_in` in config and metadata
- main paper matrix uses the two local BGE models above
- if a worked example uses an API-priced LLM, freeze the pricing snapshot date in the appendix

### 3.3 Decision stability

Compare rankings across `B0`, `B1`, and `B2` using:

- Spearman `rho`
- top-1 agreement
- top-2 agreement

## 4) Adapter and behavior requirements

Every adapter must implement:

- `create`, `drop`, `reset`, `healthcheck`
- `bulk_upsert`, `query`, `batch_query`, `insert`, `update_vectors`, `update_payload`, `delete`, `flush_or_commit`
- `set_index_params`, `set_search_params`, `optimize_or_compact`
- `stats`

Conformance must verify:

- CRUD and visibility after `flush_or_commit`
- filter correctness
- update and delete semantics
- `S2` freshness semantics using post-ACK visibility

Behavior cards must document:

- visibility semantics
- delete semantics
- update semantics
- freshness behavior
- compaction behavior
- persistence semantics
- limitations

## 5) Engineering quality bar

- typed public Python interfaces
- deterministic seeds and stable output ordering
- structured logs
- no silent retries or silent feature downgrades
- explicit unsupported-feature reporting
- tests for metric computations
- conformance tests for adapters
- at least one smoke test per scenario family

## 6) Figure and output policy

Figure rules:

- font size `16`
- `600 px` per panel edge
- export `.png` only
- include `*.meta.json` sidecar

Per-run required artifacts:

- `results.parquet`
- `run_metadata.json`
- `config_resolved.yaml`
- logs

Required metadata minimum:

- `run_id`, `timestamp_utc`
- `engine`, `engine_version`
- `embedding_model`
- `scenario`, `dataset_bundle`, `dataset_hash`
- `budget_level`
- client counts
- quality metrics
- freshness metrics when applicable
- `task_cost_est`
- `c_llm_in`
- hardware and runtime summary

## 7) Task completion format

When completing a coding task, respond with:

1. what changed
2. how it was validated
3. assumptions or blockers
4. next concrete task on the critical path

## 8) Ground truth protocol

- S1 (scifact, fiqa): official BEIR qrels
- S2 static queries: official BEIR qrels; S2 freshness GT = the inserted CRAG passage itself
- S3 (FRAMES-portable): FRAMES gold evidence passage sets; `evidence_coverage@k = |E(q) ∩ R_k(q)| / |E(q)|`

## 9) Required paper figures

| ID | Description |
|---|---|
| F1 | S1 nDCG@10 vs p99 Pareto per engine × embedding |
| F2 | S1 task_cost_est at each quality tier |
| F3 | S2 freshness_hit@1s and @5s per engine |
| F4 | S2 p95_visibility_latency CDF across engines |
| F5 | S3 evidence_coverage@k curves per engine × embedding |
| F6 | Decision-stability curves: Spearman ρ and top-1/top-2 agreement across B0/B1/B2 |
| F7 | Minimum viable deployment table (workload → engine → embedding → reason) |
| A1 | rtt_baseline_ms per engine (appendix) |
| A2 | Build/load time + RAM/disk footprint (appendix) |

## 10) Copy-paste task template

```text
Task:
Context:
- Follow project.md + prompt.md exactly.

Scope:
- [exact files/modules to modify]

Acceptance criteria:
1) ...
2) ...
3) ...

Constraints:
- Preserve deterministic outputs.
- Keep retries off during timed phases.
- Do not change schema silently.
- Add or update tests.

Deliverables:
- Code changes
- Tests
- Short verification summary
```

## 11) Copy-paste adapter implementation prompt

```text
Implement adapter <ENGINE> under maxionbench/adapters/<engine>.py.
Must satisfy adapter_contract.py and pass conformance tests including S2 freshness probe.
Create docs/behavior/<engine>.md with:
- visibility/freshness semantics
- delete/update/compaction/persistence semantics
- macOS/ARM-specific notes
Do not change result schema without migration updates.
Return: changed files, test results, engine-specific caveats.
```

## 12) Copy-paste scenario implementation prompt

```text
Implement scenario <Sx> with pinned concurrency and metrics collection.
Input: YAML config. Output: results.parquet + run_metadata.json.
Enforce budget_level (b0/b1/b2) timing and repeat count.
Run RPC baseline before timed phase.
S2: implement freshness probe timing — T = ACK time, probes at T+1s and T+5s, log per-event results.
S3: implement evidence_coverage@k against FRAMES gold evidence sets; report k=5,10,20.
Add one smoke test validating schema and basic invariants.
Return: changed files, tests, sample command.
```
