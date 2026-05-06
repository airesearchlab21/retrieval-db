# MaxionBench NeurIPS Artifact Card

## Intended Submission Track

NeurIPS 2026 Evaluations & Datasets Track.

The artifact is a benchmark/evaluation package for decision-audit claims about agentic retrieval infrastructure under explicit assumptions.

## Supported Claims

- Conformance-gated evaluation changes which engine rows are reportable.
- Strict-latency deployment decisions should report quality, post-insert top-10 retrievability, max-row p99 latency, normalized context-cost proxy, budget level, and objective together.
- Under the archived B2 evidence and a 200 ms max-row p99 rule, FAISS CPU with `BAAI/bge-small-en-v1.5` is the policy-selected strict-latency configuration for S1, S2, and S3.
- Objective choice changes deployment conclusions, especially for S3.
- The S3 pgvector quality-first row should be interpreted as objective sensitivity because the 5,000-query matched audit is indistinguishable from FAISS.
- S2 paired checks find no Qdrant quality/post-insert retrievability advantage in `paper/experiments`.

## Future Extension Claims

- Managed-service, distributed, GPU, and second-machine matrices are natural follow-up rows for the same adapter contract.
- Production-cluster latency studies require a separate hardware and workload matrix.
- End-to-end agent task success can be layered on top of the retrieval-infrastructure evidence.
- High-probe/high-ef service-engine sweeps can extend the targeted checks included under `paper/experiments`.
- The S2 mini-bundle reports repeatability for capped timed phases; the standard S2 rows remain the latency evidence.

## Reviewer Entry Points

- Manuscript PDF: `paper/manuscript/main.pdf`
- Main paper source: `paper/manuscript/main.tex`
- Evidence map: `paper/manuscript/tables/evidence_strength.tex`
- Archive manifest: `paper/archive/archive_manifest.json`
- Paper tables: `paper/tables/`
- Paper figures: `paper/figures/`
- Conceptual decision-audit figure: `paper/figures/maxionbench_decision_audit_conceptual.svg`
- B2 decision-surface figure: `paper/figures/portable_decision_surface.svg`
- S3 paired-audit forest plot: `paper/figures/s3_paired_audit_forest.svg`
- Reportability support table: `paper/manuscript/tables/portable_support_table.tex`
- S2 strict-choice post-insert examples: `paper/manuscript/tables/s2_post_insert_examples.tex`
- Strict FAISS same-machine repeat audit: `paper/manuscript/tables/strict_faiss_repeat_audit.tex`
- S2 same-orchestration mini-bundle: `paper/experiments/s2_mini_bundle/`
- S3 matched-query audit: `paper/experiments/s3_paired_quality/`
- S3 all-evidence audit support: `paper/experiments/s3_all_evidence/`
- Metadata: `paper/metadata/`
- Reviewer package builder: `paper/build_reviewer_package.py`

## Reproducibility Checks

Quick local check:

```bash
python paper/verify_neurips_artifacts.py --json
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

Reviewer package build:

```bash
python paper/build_reviewer_package.py --json
```

The package preserves the repository layout so reviewers can extract it, install dependencies from `pyproject.toml`, and rerun the local artifact verifier from the package root.

## Data and License Notes

HotpotQA-MaxionBench is a frozen preprocessing of HotpotQA dev distractor for retrieval/evidence-coverage evaluation. HotpotQA is distributed under CC BY-SA 4.0 according to the official HotpotQA/Hugging Face dataset card. The paper artifact records checksums and preprocessing metadata so reviewers can verify the exact local corpus.

Before public submission, confirm that the anonymous reviewer-artifact URL in the Croissant metadata resolves to the uploaded artifact.

## Compute Scope

The reported archive was produced on one local anonymous single-node ARM64 CPU host with 10 logical CPU cores, 16 GB RAM, local OS, Python 3.11.14, and Docker 29.4.0. The matrix uses CPU rows throughout, and p99 and cost rows are reported with this runtime profile. Max-row p99 is computed as the maximum archived p99 across configured B2 client rows and repeats for a candidate.

## AI/Agent Use Disclosure

Code assistance and manuscript editing assistance were used during preparation. The authors remain responsible for all text, figures, references, and experimental claims. No LLM-generated citation should be accepted without manual verification.
