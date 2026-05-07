# Corpus-Scale FAISS Stress Check

This directory contains a controlled vector-index scaling check added for the
submission revision. It is not a full MaxionBench workload row and does not run
the adapter conformance gate.

The script uses the real HOTPOTQA-MAXIONBENCH `BAAI/bge-small-en-v1.5`
document and query embeddings, then expands the corpus with deterministic
random unit-vector distractors to 250K and 1M vectors. It compares FAISS exact
`IndexFlatIP` against FAISS `IndexIVFFlat` with `nlist=1024` and `nprobe=32`.
IVF quality is reported as recall@10 against the exact FlatIP top-10 at the
same vector scale.

Primary run:

```bash
python paper/experiments/corpus_scale_stress/run_faiss_scale_stress.py \
  --query-count 512 \
  --output paper/experiments/corpus_scale_stress/faiss_scale_stress_results_q512.json
```

The 512-query run reports:

| vectors | FlatIP p99 ms | IVF p99 ms | IVF recall@10 vs FlatIP |
| ---: | ---: | ---: | ---: |
| 66,635 | 5.06 | 0.37 | 0.9035 |
| 250,000 | 17.49 | 1.72 | 0.9699 |
| 1,000,000 | 76.73 | 9.45 | 0.9994 |

Interpretation: in this isolated vector-only check, exact FlatIP remains below
the 200 ms policy at 1M vectors, but its tail latency grows substantially with
corpus size and IVF gives much lower p99 latency at near-exact recall. This
supports the paper's claim-scoping language: the 50K--66K engine ranking should
not be read as a production-scale ranking.

## Cross-engine Qdrant stress check

The follow-up script adds a non-FAISS vector-only sanity check on the same
vector construction:

```bash
python paper/experiments/corpus_scale_stress/run_cross_engine_scale_stress.py \
  --scales 66635 \
  --query-count 512 \
  --output paper/experiments/corpus_scale_stress/cross_engine_scale_stress_results_q512.json

python paper/experiments/corpus_scale_stress/run_cross_engine_scale_stress.py \
  --scales 250000 \
  --query-count 512 \
  --output paper/experiments/corpus_scale_stress/cross_engine_scale_stress_results_q512_250k.json

python paper/experiments/corpus_scale_stress/run_cross_engine_scale_stress.py \
  --scales 1000000 \
  --query-count 512 \
  --output paper/experiments/corpus_scale_stress/cross_engine_scale_stress_results_q512_1m.json
```

The Qdrant HNSW run uses `m=64`, `ef_construct=128`, `hnsw_ef=256`, and a
10-minute readiness cap. The completed rows report:

| vectors | Qdrant p99 ms | recall@10 vs FlatIP | indexed / points | status |
| ---: | ---: | ---: | ---: | --- |
| 66,635 | 12.74 | 0.9988 | 66,635 / 66,635 | fully indexed |
| 250,000 | 8.91 | 0.9994 | 250,000 / 250,000 | fully indexed |
| 1,000,000 | 10.05 | 0.9994 | 1,000,000 / 1,000,000 | fully indexed |

Interpretation: the cross-engine vector-only check now reaches a fully indexed
1M-vector Qdrant HNSW row. Read against the FAISS scale rows, it supports the
paper's scope statement: exact FAISS remains a strong local 50K--66K baseline,
but approximate service-engine behavior can become favorable as vector count
increases.
