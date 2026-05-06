# NeurIPS Paper Artifacts

This folder stages the manuscript, figures, tables, and result notes for the NeurIPS draft.

Source archive: `results/20260429T033427Z`

## Layout

- `manuscript/`: NeurIPS LaTeX source and `main.pdf`.
- `tables/`: generated CSV/TeX tables from the archived report bundle.
- `figures/`: generated PDF/PNG figures and metadata.
- `results/`: supporting result metadata used by the paper.
- `experiments/`: targeted replication checks and paired analyses used to de-risk the story.
- `archive/`: archive manifest for the exact reproducibility bundle.
- `artifact_card.md`: reviewer-facing claim, scope, and reproducibility summary.
- `submission_readiness.md`: final upload checklist and story guardrails.
- `metadata/`: Croissant-style dataset metadata and evaluation-card metadata.

## Build the manuscript

```bash
cd paper/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c
```

Expected output: `paper/manuscript/main.pdf`.

## Validate the archived bundle

Archived run schema check:

```bash
python -m maxionbench.cli validate \
  --input artifacts/runs/portable \
  --strict-schema \
  --json
```

Expected result used by the paper: pass, 72 run directories checked, 0 errors.

## Key paper-facing evidence

- `figures/maxionbench_decision_audit_conceptual.*`: conceptual comparison between leaderboard evaluation and the MaxionBench decision audit.
- `figures/portable_decision_surface.*`: B2 quality-vs-max-row-p99 decision surface with strict, cost-only/no-p99, and quality-first choices.
- `figures/s3_paired_audit_forest.*`: S3 matched-query paired audit against FAISS exact FlatIP.
- `figures/portable_s2_post_insert_retrievability.*`: S2 post-insert top-10 retrievability diagnostics near the S2 discussion.
- `figures/portable_mvd_sensitivity.*`: p99-threshold sensitivity near the objective-sensitivity discussion.
- `tables/neurips_main_results.*`: strict-latency minimum viable deployment results with confidence intervals.
- `tables/portable_decision_table.*`: MaxionBench deployment decisions under strict max-row p99, cost-only/no-p99, and quality-first objectives.
- `manuscript/tables/portable_support_table.tex`: MaxionBench conformance and behavior-card support table for reportability filtering.
- `manuscript/tables/strict_decision_margins.tex`: decision-margin table for the strict-latency winners.
- `manuscript/tables/strict_faiss_repeat_audit.tex`: same-machine repeat audit for the strict FAISS CPU/bge-small choices.
- `manuscript/tables/s3_paired_quality.tex`: 5,000-query matched S3 audit showing no substantive pgvector quality advantage.
- `manuscript/tables/s2_post_insert_examples.tex`: S2 strict-choice post-insert success/miss counts with archived event/query/document ID examples.
- `manuscript/tables/s2_competitor_check.tex`: larger same-machine paired S2 FAISS/Qdrant competitor check.
- `experiments/s2_larger_same_machine/`: B2 same-machine S2 FAISS/Qdrant rerun with two repeats per engine, 1,788 matched quality observations, and 200 matched post-insert events.
- `experiments/strict_faiss_repeats/`: JSON summary for completed same-machine strict FAISS repeat artifacts.
- `experiments/s2_mini_bundle/`: same-orchestration S2 FAISS/Qdrant mini-bundle with two repeats per engine.
- `manuscript/tables/evidence_strength.tex`: appendix claim/evidence/risk map.

## Current story

The paper is a decision-audit benchmark for retrieval infrastructure. The supported claim is that conformance-gated benchmark evidence can guide local deployment decisions when quality, post-insert top-10 retrievability, max-row p99 latency, normalized context-cost proxy, budget stability, and objective sensitivity are reported together.

Latency is measured on precomputed query vectors and times `adapter.query` plus top-k materialization, including service/container overhead inside the adapter call when applicable; offline embedding is excluded from latency and included only in the normalized context-cost proxy. The reported max-row p99 is the maximum archived p99 across configured B2 client rows and repeats for a candidate.
