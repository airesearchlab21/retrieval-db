# MaxionBench NeurIPS Artifact Card

## Intended Submission Track

NeurIPS 2026 Evaluations & Datasets Track.

The artifact is a benchmark/evaluation package, not a new retrieval model. Its purpose is to support decision-audit claims about agentic retrieval infrastructure under explicit assumptions.

## Supported Claims

- Conformance-gated evaluation changes which engine rows are reportable.
- Strict-latency deployment decisions should report quality, post-insert top-10 retrievability, p99 latency, task cost, budget level, and objective together.
- Under the archived local single-node B2 evidence and a 200 ms p99 rule, FAISS CPU with `BAAI/bge-small-en-v1.5` is the minimum viable deployment for S1, S2, and S3.
- Objective choice changes deployment conclusions, especially for S3.
- The S3 pgvector quality-first row should be interpreted as objective sensitivity, not a substantive quality advantage, after the 5,000-query matched audit.
- S2 Qdrant does not hide a quality/post-insert retrievability win in the paired checks now included in `paper/experiments`.

## Unsupported Claims

- The artifact does not prove universal engine superiority.
- The artifact does not claim production latency across hardware, clusters, or managed services.
- The artifact does not evaluate full end-to-end agent task success.
- The bounded S2 mini-bundle is repeatability evidence only; it is not a replacement for standard uncapped S2 latency evidence.

## Reviewer Entry Points

- Manuscript PDF: `paper/manuscript/main.pdf`
- Main paper source: `paper/manuscript/main.tex`
- Evidence map: `paper/manuscript/tables/evidence_strength.tex`
- Archive manifest: `paper/archive/archive_manifest.json`
- Paper tables: `paper/tables/`
- Paper figures: `paper/figures/`
- Reportability support table: `paper/manuscript/tables/portable_support_table.tex`
- S2 bounded mini-bundle: `paper/experiments/s2_mini_bundle/`
- S3 matched-query audit: `paper/experiments/s3_paired_quality/`
- S3 all-evidence audit support: `paper/experiments/s3_all_evidence/`
- Metadata: `paper/metadata/`

## Reproducibility Checks

Quick local check:

```bash
python -m maxionbench.cli validate --input results/20260429T033427Z/runs --strict-schema --json
```

Full project test check:

```bash
python -m pytest -q
```

Archived run validation:

```bash
python -m maxionbench.cli validate \
  --input results/20260429T033427Z/runs \
  --strict-schema \
  --json
```

Expected archived validation result: 72 run directories checked, 0 errors.

The package preserves the repository layout so reviewers can extract it, install dependencies from `pyproject.toml`, and run validation commands from the package root.

## Data and License Notes

HotpotQA-MaxionBench is a bounded preprocessing of HotpotQA dev distractor for retrieval/evidence-coverage evaluation. HotpotQA is distributed under CC BY-SA 4.0 according to the official HotpotQA/Hugging Face dataset card. The paper artifact records checksums and preprocessing metadata so reviewers can verify the exact local corpus.

Before public submission, confirm that the anonymous reviewer-artifact URL in the Croissant metadata resolves to the uploaded artifact.

## Compute Scope

The reported archive was produced on one local anonymous single-node ARM64 CPU host with 10 logical CPU cores, 16 GB RAM, local OS, Python 3.11.14, Docker 29.4.0, and no GPUs. This is part of the claim boundary: p99 and cost rows are local deployment evidence, not universal infrastructure constants.

## AI/Agent Use Disclosure

Code assistance and manuscript editing assistance were used during preparation. The authors remain responsible for all text, figures, references, and experimental claims. No LLM-generated citation should be accepted without manual verification.
