# MaxionBench Reviewer Artifact

## What this artifact supports

This artifact supports the MaxionBench NeurIPS 2026 E&D submission: a conformance-gated, single-node decision audit for agentic retrieval infrastructure. The supported evidence is the archived `20260429T033427Z` MaxionBench run bundle, generated report tables, conformance metadata, HotpotQA-MaxionBench metadata, and the manuscript PDF.

The supported deployment claim is bounded: under the archived local B2 evidence and a 200 ms p99 rule, FAISS CPU with `BAAI/bge-small-en-v1.5` is the minimum viable deployment for S1, S2, and S3. Objective, conformance, budget, and paired-audit rows are included so reviewers can audit that conclusion.

Latency is measured on precomputed query vectors and times adapter.query plus top-k materialization, including service/container overhead inside the adapter call when applicable; offline embedding is excluded from latency and included only in the task-cost estimate.

## What this artifact does not support

The artifact does not support universal engine rankings, managed-service production latency claims, distributed-cluster conclusions, GPU-track conclusions, or end-to-end generated-answer quality claims.

## Hardware used for the paper run

The archived run metadata records one anonymous single-node ARM64 CPU host with 10 logical CPU cores, 16 GiB RAM, local OS, Python 3.11.14, Docker 29.4.0, and no GPUs.

## Full paper run commands

```bash
maxionbench workflow setup --json
maxionbench workflow data --json
maxionbench submit --budget b0 --json
maxionbench submit --budget b1 --json
maxionbench submit --budget b2 --json
maxionbench workflow finalize --json
python -m maxionbench.cli validate --input results/{ARCHIVE_RUN_ID}/runs --strict-schema --json
```

## Reviewer smoke-test commands

```bash
python -m maxionbench.cli validate --input results/20260429T033427Z/runs --strict-schema --json
python -m pytest -q tests/test_adapter_registry.py tests/test_metrics.py tests/test_quality_metrics.py
```

These checks validate the archived run schema and run small unit tests that do not require rerunning the full benchmark matrix.

## Validation commands and expected outputs

```bash
python -m maxionbench.cli validate --input results/20260429T033427Z/runs --strict-schema --json
```

The expected stored validation output is `results/20260429T033427Z/validation_strict_schema.json` with `"pass": true`, `"error_count": 0`, and `"run_dirs_checked": 72`.

## Data provenance and licenses

See `LICENSES.md` for source citations, exact license/terms sources, and release boundaries. The artifact includes processed evaluation bundles only.

## Croissant and Responsible AI metadata

Core Croissant metadata is in `metadata/hotpot_maxionbench_croissant.json`. Responsible AI metadata is in `metadata/hotpot_maxionbench_croissant_rai.json`. The metadata describes `HOTPOTQA-MAXIONBENCH`, 66,635 documents, 7,405 questions, 14,810 qrels, intended evaluation use, and non-production limitations.

## Anonymization notes

Text files copied into this package replace local workspace and user-home paths with anonymous placeholders. This package is prepared for double-blind review and contains no identifying repository remotes, usernames, or absolute local paths.

## Known runtime limits

The full B0/B1/B2 run is a local single-node run and can take many hours. The smoke test is intentionally bounded and does not rerun the full B2 matrix. pgvector and Qdrant tiny-query checks run only when their local Python dependencies and services are available.
