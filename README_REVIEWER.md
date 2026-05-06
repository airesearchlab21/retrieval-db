# MaxionBench Reviewer README

## Paper claim map

The paper claims that a conformance-gated, single-node retrieval-infrastructure audit can support local deployment decisions. The main archived decision is bounded to `20260429T033427Z`: under a 200 ms worst-case p99 rule, FAISS CPU with `BAAI/bge-small-en-v1.5` is the minimum viable deployment for S1, S2, and S3. The paper also maps objective changes, budget instability, paired S1/S2 audits, an S3 matched-query audit, and an S2 Qdrant competitor check to specific tables.

## Archive ID

The archived paper run is `results/20260429T033427Z`. The package includes the archived MaxionBench run tree, generated paper tables, conformance rows, metadata, and the manuscript PDF used for review.

## Hardware

The archived run metadata records one anonymous single-node ARM64 CPU host with 10 logical CPU cores, 16 GiB RAM, local OS, Python 3.11.14, Docker 29.4.0, and no GPUs.

## Validation

Run:

```bash
python -m maxionbench.cli validate --input results/20260429T033427Z/runs --strict-schema --json
python -m pytest -q tests/test_adapter_registry.py tests/test_metrics.py tests/test_quality_metrics.py
```

The stored strict-schema validation should report `"pass": true`, `"error_count": 0`, and `"run_dirs_checked": 72`.

## Regenerate tables

Run:

```bash
python -m maxionbench.cli report --mode maxionbench --input results/20260429T033427Z/runs --out paper/tables --conformance-matrix artifacts/conformance/conformance_matrix.csv --behavior-dir docs/behavior
```

The numeric consistency checker verifies that main p99 values and strict-margin deltas are derived from the regenerated CSV tables.

## Smoke test

Run:

```bash
python -m maxionbench.cli validate --input results/20260429T033427Z/runs --strict-schema --json
```

This validates the archived run schema without rerunning the full B2 matrix.

## Full runtime

The full local workflow is:

```bash
maxionbench workflow setup --json
maxionbench workflow data --json
maxionbench submit --budget b0 --json
maxionbench submit --budget b1 --json
maxionbench submit --budget b2 --json
maxionbench workflow finalize --json
```

This is a many-hour single-node run; reviewers should use the smoke and validation commands for quick checks.

## Licenses/provenance

See `LICENSES.md` for dataset, model, engine, and release-boundary details. Croissant metadata is in `metadata/hotpot_maxionbench_croissant.json`, Responsible AI metadata is in `metadata/hotpot_maxionbench_croissant_rai.json`, and the source HotpotQA-MaxionBench metadata copy is in `paper/metadata/hotpotqa_portable_croissant.jsonld`.

## LanceDB service exclusion

LanceDB in-process is reportable. LanceDB service mode is intentionally excluded from paper-facing result tables because the archived conformance matrix has no passing local service row.

## Anonymization

The package builder sanitizes local workspace paths, user-home paths, and repository-owner strings in copied text files. Do not add identifying remotes, local absolute paths, or usernames before uploading this package for double-blind review.
