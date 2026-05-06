# Paper Figure Assets

Regenerate these files with:

```bash
python scripts/figures/regenerate_figures.py
```

The script reads `artifacts/runs/portable`, `artifacts/conformance/conformance_matrix.csv`, `docs/behavior/`, and `paper/experiments/s3_paired_quality/summary.json`, stages a full report bundle under `artifacts/figures/paper_refresh`, then copies paper-facing figure assets into this directory.

| Figure | Source data | Script path | Output files | Interpretation |
| --- | --- | --- | --- | --- |
| `maxionbench_decision_audit_conceptual` | Benchmark protocol in `paper/manuscript/sections/02_benchmark.tex` | `maxionbench/reports/portable_exports.py` via `scripts/figures/regenerate_figures.py` | `.svg`, `.meta.json` | Shows how MaxionBench turns leaderboard-style evidence into a conformance-gated decision audit. |
| `portable_decision_surface` | `artifacts/runs/portable/**/results.parquet` | `maxionbench/reports/portable_exports.py` via `scripts/figures/regenerate_figures.py` | `.svg`, `.meta.json` | Shows strict, cost-only/no-p99, and quality-first B2 choices across S1-S3. |
| `s3_paired_audit_forest` | `paper/experiments/s3_paired_quality/summary.json` | `maxionbench/reports/portable_exports.py` via `scripts/figures/regenerate_figures.py` | `.svg`, `.meta.json` | Shows that matched S3 pgvector-minus-FAISS intervals include zero. |
| `portable_task_cost_by_budget` | `artifacts/runs/portable/**/results.parquet` | `maxionbench/reports/portable_exports.py` via `scripts/figures/regenerate_figures.py` | `.svg`, `.meta.json` | Tracks the strict winner context-cost proxy as the run budget increases. |
| `portable_budget_stability` | `artifacts/runs/portable/**/results.parquet` | `maxionbench/reports/portable_exports.py` via `scripts/figures/regenerate_figures.py` | `.svg`, `.meta.json` | Shows that B0 screening does not reliably preserve B2 top-1 decisions for every workload. |
| `portable_s2_post_insert_retrievability` | `artifacts/runs/portable/**/results.parquet` | `maxionbench/reports/portable_exports.py` via `scripts/figures/regenerate_figures.py` | `.svg`, `.meta.json` | Reports fixed 1s and 5s post-insert top-10 retrievability for S2 strict survivors. |
| `portable_mvd_sensitivity` | `artifacts/runs/portable/**/results.parquet` | `maxionbench/reports/portable_exports.py` via `scripts/figures/regenerate_figures.py` | `.svg`, `.meta.json` | Shows how the minimum viable deployment changes as the p99 policy threshold changes. |
