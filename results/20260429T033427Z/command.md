# MaxionBench Commands

Run from the repo root on the operator anonymous single-node ARM64 CPU host. The single-node host is the execution host only, not part of the benchmark storyline.

Target: complete the local workflow within one day. `submit-portable` defaults to a 24-hour execution deadline; reduce `--deadline-hours` if setup/data/embedding time must count inside the day.

## 0. Install and check

```bash
pip install -e ".[dev,engines,reporting,datasets,embeddings]"
maxionbench --help
maxionbench verify-pins --json
maxionbench verify-dataset-manifests --json
maxionbench verify-conformance-configs --json
```

Intel local OS only:

```bash
pip install -e ".[dev,engines,reporting,datasets,embeddings]" "numpy==1.26.4" "transformers<5"
```

## 1. Setup

Unset the env var first so setup uses the correct `/tmp` default for lancedb-inproc (external volumes do not support `rename()`):

```bash
unset MAXIONBENCH_LANCEDB_INPROC_URI
maxionbench portable-workflow setup --json
```

Expected: conformance passes for faiss-cpu, lancedb-inproc, pgvector, and qdrant.

## 2. Data and embeddings

```bash
maxionbench portable-workflow data --json
```

## 3. Run budgets

```bash
maxionbench submit-portable --budget b0 --json 2>&1 | tee artifacts/b0.log
maxionbench submit-portable --budget b1 --json 2>&1 | tee artifacts/b1.log
maxionbench submit-portable --budget b2 --json 2>&1 | tee artifacts/b2.log
```

## 4. Finalize and report

```bash
maxionbench portable-workflow finalize --json
```

## Notes

- **lancedb-inproc**: requires a local filesystem that supports atomic `rename()`. Setup defaults to `/tmp/maxionbench/lancedb/inproc` when `MAXIONBENCH_LANCEDB_INPROC_URI` is unset. Do not point it at an external or network-mounted volume.
- **lancedb-service**: excluded from the paper path. Does not run during `submit-portable`.
- **pgvector**: uses exact sequential scan (`index_method=none`). Higher recall vs. HNSW engines (faiss-cpu, qdrant) is expected and is a valid paper finding.
- **S2 freshness**: requires CRAG-500 events. The loader retains all qrel-referenced evidence docs before applying the 50K background cap. If fewer than 90% of events survive after loading, the run aborts.
- **b1 and b2**: only run configs that cleared the promotion thresholds from the previous budget level.
