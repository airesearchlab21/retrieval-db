"""Controlled FAISS corpus-scale stress test for the NeurIPS submission.

This is not a MaxionBench workload row. It isolates vector-index scale by
combining the HOTPOTQA-MAXIONBENCH bge-small vectors with deterministic random
unit-vector distractors, then compares exact FlatIP search with IVF search.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _synthetic_unit_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dim), dtype=np.float32)
    return _normalize(vectors)


def _build_vectors(base_docs: np.ndarray, target_size: int, seed: int) -> np.ndarray:
    if target_size <= len(base_docs):
        return np.ascontiguousarray(base_docs[:target_size], dtype="float32")
    extra = target_size - len(base_docs)
    distractors = _synthetic_unit_vectors(extra, base_docs.shape[1], seed)
    return np.ascontiguousarray(np.vstack([base_docs, distractors]), dtype="float32")


def _latency_ms(samples: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(samples), dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def _timed_search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[dict[str, float], np.ndarray]:
    latencies = []
    labels = []
    for query in queries:
        start = time.perf_counter()
        _, ids = index.search(query.reshape(1, -1), k)
        latencies.append((time.perf_counter() - start) * 1000.0)
        labels.append(ids[0].copy())
    return _latency_ms(latencies), np.vstack(labels)


def _recall_at_k(candidate: np.ndarray, reference: np.ndarray) -> float:
    hits = 0
    for cand_row, ref_row in zip(candidate, reference):
        hits += len(set(cand_row.tolist()) & set(ref_row.tolist()))
    return hits / float(reference.size)


def _benchmark_scale(
    vectors: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    nlist: int,
    nprobe: int,
    train_size: int,
    seed: int,
) -> dict[str, object]:
    dim = vectors.shape[1]

    flat = faiss.IndexFlatIP(dim)
    start = time.perf_counter()
    flat.add(vectors)
    flat_build_s = time.perf_counter() - start
    flat_latency, flat_ids = _timed_search(flat, queries, k)

    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.nprobe = nprobe

    rng = np.random.default_rng(seed)
    train_n = min(train_size, len(vectors))
    train_idx = rng.choice(len(vectors), size=train_n, replace=False)
    train_vectors = np.ascontiguousarray(vectors[train_idx], dtype="float32")

    start = time.perf_counter()
    ivf.train(train_vectors)
    ivf_train_s = time.perf_counter() - start
    start = time.perf_counter()
    ivf.add(vectors)
    ivf_add_s = time.perf_counter() - start
    ivf_latency, ivf_ids = _timed_search(ivf, queries, k)

    bytes_per_vector = vectors.shape[1] * np.dtype("float32").itemsize
    return {
        "corpus_vectors": int(len(vectors)),
        "dim": int(dim),
        "queries": int(len(queries)),
        "k": int(k),
        "flat": {
            "index": "faiss.IndexFlatIP",
            "build_s": float(flat_build_s),
            "approx_vector_memory_mb": float(len(vectors) * bytes_per_vector / 1_000_000),
            **flat_latency,
        },
        "ivf": {
            "index": "faiss.IndexIVFFlat",
            "nlist": int(nlist),
            "nprobe": int(nprobe),
            "train_vectors": int(train_n),
            "train_s": float(ivf_train_s),
            "add_s": float(ivf_add_s),
            "recall_at_10_vs_flat": float(_recall_at_k(ivf_ids, flat_ids)),
            **ivf_latency,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("reviewer_artifact_maxionbench/dataset"))
    parser.add_argument("--embedding", default="baai-bge-small-en-v1-5")
    parser.add_argument("--output", type=Path, default=Path("paper/experiments/corpus_scale_stress/faiss_scale_stress_results.json"))
    parser.add_argument("--scales", nargs="+", type=int, default=[66_635, 250_000, 1_000_000])
    parser.add_argument("--query-count", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    emb_dir = args.dataset_root / "processed" / "hotpot_portable" / "embeddings" / args.embedding
    docs = _normalize(np.load(emb_dir / "doc_vectors.npy"))
    queries = _normalize(np.load(emb_dir / "query_vectors.npy")[: args.query_count])

    results = {
        "description": (
            "Controlled vector-index scale stress test. Uses real HOTPOTQA-MAXIONBENCH "
            "bge-small document/query embeddings plus deterministic random unit-vector "
            "distractors. This is not a full MaxionBench engine/workload row."
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "base_documents": int(len(docs)),
        "embedding": args.embedding,
        "seed": int(args.seed),
        "results": [],
    }

    for scale in args.scales:
        vectors = _build_vectors(docs, scale, args.seed + scale)
        row = _benchmark_scale(
            vectors,
            queries,
            k=args.k,
            nlist=args.nlist,
            nprobe=args.nprobe,
            train_size=args.train_size,
            seed=args.seed + 17 + scale,
        )
        results["results"].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
