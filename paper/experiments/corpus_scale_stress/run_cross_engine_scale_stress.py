"""Controlled cross-engine vector-scale stress check.

This is not a MaxionBench workload row. It isolates vector-index scale by
combining the HOTPOTQA-MAXIONBENCH bge-small vectors with deterministic random
unit-vector distractors, then compares Qdrant HNSW search against exact FAISS
FlatIP top-10 on the same vector set.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import requests

try:
    import ujson
except ImportError:  # pragma: no cover - optional local speedup
    ujson = None


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


def _timed_faiss_search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[dict[str, float], np.ndarray]:
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


def _json_body(body: dict[str, Any]) -> str:
    if ujson is not None:
        return ujson.dumps(body)
    return json.dumps(body, separators=(",", ":"))


def _qdrant_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout_s: float,
    allow_404: bool = False,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"} if body is not None else None
    data = _json_body(body) if body is not None else None
    response = session.request(method=method, url=url, data=data, headers=headers, timeout=timeout_s)
    if allow_404 and response.status_code == 404:
        return {}
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{method} {url} failed with HTTP {response.status_code}: {response.text[:1000]}") from exc
    if not response.content:
        return {}
    payload = response.json()
    status = payload.get("status")
    if status not in (None, "ok"):
        raise RuntimeError(f"{method} {url} returned non-ok status {status!r}: {response.text[:1000]}")
    return payload


def _create_qdrant_collection(
    session: requests.Session,
    *,
    base_url: str,
    collection: str,
    dim: int,
    hnsw_m: int,
    ef_construct: int,
    full_scan_threshold: int,
    timeout_s: float,
) -> None:
    _qdrant_request(
        session,
        "DELETE",
        f"{base_url}/collections/{collection}",
        timeout_s=timeout_s,
        allow_404=True,
    )
    body = {
        "vectors": {"size": int(dim), "distance": "Dot"},
        "hnsw_config": {
            "m": int(hnsw_m),
            "ef_construct": int(ef_construct),
            "full_scan_threshold": int(max(10, full_scan_threshold)),
        },
    }
    _qdrant_request(session, "PUT", f"{base_url}/collections/{collection}", body=body, timeout_s=timeout_s)


def _wait_for_qdrant_collection(
    session: requests.Session,
    *,
    base_url: str,
    collection: str,
    timeout_s: float,
    max_wait_s: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    last: dict[str, Any] = {}
    while time.perf_counter() - start < max_wait_s:
        payload = _qdrant_request(session, "GET", f"{base_url}/collections/{collection}", timeout_s=timeout_s)
        result = dict(payload.get("result", {}))
        last = result
        status = str(result.get("status", "")).lower()
        optimizer_status = result.get("optimizer_status")
        optimizer_ok = optimizer_status == "ok" or (
            isinstance(optimizer_status, dict) and str(optimizer_status.get("ok", "")).lower() == "true"
        )
        if status == "green" and optimizer_ok:
            return result
        time.sleep(1.0)
    return last


def _qdrant_upsert(
    session: requests.Session,
    *,
    base_url: str,
    collection: str,
    vectors: np.ndarray,
    batch_size: int,
    timeout_s: float,
) -> float:
    start = time.perf_counter()
    total = int(len(vectors))
    for offset in range(0, total, batch_size):
        end = min(offset + batch_size, total)
        body = {
            "batch": {
                "ids": list(range(offset, end)),
                "vectors": vectors[offset:end].tolist(),
            }
        }
        _qdrant_request(
            session,
            "PUT",
            f"{base_url}/collections/{collection}/points?wait=true",
            body=body,
            timeout_s=timeout_s,
        )
        if end == total or end % max(batch_size * 20, batch_size) == 0:
            print(f"qdrant upserted {end:,}/{total:,} vectors", flush=True)
    return time.perf_counter() - start


def _timed_qdrant_search(
    session: requests.Session,
    *,
    base_url: str,
    collection: str,
    queries: np.ndarray,
    k: int,
    hnsw_ef: int,
    timeout_s: float,
) -> tuple[dict[str, float], np.ndarray]:
    latencies = []
    labels: list[list[int]] = []
    for query in queries:
        body = {
            "vector": query.tolist(),
            "limit": int(k),
            "with_payload": False,
            "with_vector": False,
            "params": {"hnsw_ef": int(hnsw_ef)},
        }
        start = time.perf_counter()
        payload = _qdrant_request(
            session,
            "POST",
            f"{base_url}/collections/{collection}/points/search",
            body=body,
            timeout_s=timeout_s,
        )
        latencies.append((time.perf_counter() - start) * 1000.0)
        labels.append([int(item["id"]) for item in payload.get("result", [])])
    return _latency_ms(latencies), np.asarray(labels, dtype=np.int64)


def _benchmark_qdrant_scale(
    vectors: np.ndarray,
    queries: np.ndarray,
    *,
    reference_ids: np.ndarray,
    k: int,
    base_url: str,
    collection_prefix: str,
    batch_size: int,
    hnsw_m: int,
    ef_construct: int,
    hnsw_ef: int,
    full_scan_threshold: int,
    timeout_s: float,
    ready_wait_s: float,
    keep_collection: bool,
) -> dict[str, Any]:
    collection = f"{collection_prefix}_{len(vectors)}"
    session = requests.Session()
    try:
        _create_qdrant_collection(
            session,
            base_url=base_url,
            collection=collection,
            dim=vectors.shape[1],
            hnsw_m=hnsw_m,
            ef_construct=ef_construct,
            full_scan_threshold=full_scan_threshold,
            timeout_s=timeout_s,
        )
        upsert_s = _qdrant_upsert(
            session,
            base_url=base_url,
            collection=collection,
            vectors=vectors,
            batch_size=batch_size,
            timeout_s=timeout_s,
        )
        wait_start = time.perf_counter()
        stats = _wait_for_qdrant_collection(
            session,
            base_url=base_url,
            collection=collection,
            timeout_s=timeout_s,
            max_wait_s=ready_wait_s,
        )
        ready_wait_elapsed_s = time.perf_counter() - wait_start
        latency, qdrant_ids = _timed_qdrant_search(
            session,
            base_url=base_url,
            collection=collection,
            queries=queries,
            k=k,
            hnsw_ef=hnsw_ef,
            timeout_s=timeout_s,
        )
        return {
            "index": "qdrant.HNSW",
            "collection": collection,
            "m": int(hnsw_m),
            "ef_construct": int(ef_construct),
            "hnsw_ef": int(hnsw_ef),
            "full_scan_threshold": int(max(10, full_scan_threshold)),
            "upsert_s": float(upsert_s),
            "ready_wait_s": float(ready_wait_elapsed_s),
            "recall_at_10_vs_flat": float(_recall_at_k(qdrant_ids, reference_ids)),
            "points_count": int(stats.get("points_count") or stats.get("vectors_count") or 0),
            "indexed_vectors_count": int(stats.get("indexed_vectors_count") or 0),
            **latency,
        }
    finally:
        if not keep_collection:
            try:
                _qdrant_request(
                    session,
                    "DELETE",
                    f"{base_url}/collections/{collection}",
                    timeout_s=timeout_s,
                    allow_404=True,
                )
            finally:
                session.close()


def _benchmark_scale(
    base_docs: np.ndarray,
    queries: np.ndarray,
    *,
    scale: int,
    seed: int,
    k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"building vectors for scale={scale:,}", flush=True)
    vectors = _build_vectors(base_docs, scale, seed + scale)
    dim = vectors.shape[1]

    print(f"building FAISS exact baseline for scale={scale:,}", flush=True)
    flat = faiss.IndexFlatIP(dim)
    start = time.perf_counter()
    flat.add(vectors)
    flat_build_s = time.perf_counter() - start
    flat_latency, flat_ids = _timed_faiss_search(flat, queries, k)

    bytes_per_vector = vectors.shape[1] * np.dtype("float32").itemsize
    row: dict[str, Any] = {
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
    }

    print(f"running Qdrant HNSW for scale={scale:,}", flush=True)
    row["qdrant"] = _benchmark_qdrant_scale(
        vectors,
        queries,
        reference_ids=flat_ids,
        k=k,
        base_url=args.qdrant_url.rstrip("/"),
        collection_prefix=args.collection_prefix,
        batch_size=args.qdrant_batch_size,
        hnsw_m=args.qdrant_m,
        ef_construct=args.qdrant_ef_construct,
        hnsw_ef=args.qdrant_hnsw_ef,
        full_scan_threshold=args.qdrant_full_scan_threshold,
        timeout_s=args.qdrant_timeout_s,
        ready_wait_s=args.qdrant_ready_wait_s,
        keep_collection=args.keep_qdrant_collections,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("reviewer_artifact_maxionbench/dataset"))
    parser.add_argument("--embedding", default="baai-bge-small-en-v1-5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/experiments/corpus_scale_stress/cross_engine_scale_stress_results.json"),
    )
    parser.add_argument("--scales", nargs="+", type=int, default=[66_635, 250_000, 1_000_000])
    parser.add_argument("--query-count", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection-prefix", default="maxionbench_cross_scale")
    parser.add_argument("--qdrant-batch-size", type=int, default=2048)
    parser.add_argument("--qdrant-m", type=int, default=64)
    parser.add_argument("--qdrant-ef-construct", type=int, default=128)
    parser.add_argument("--qdrant-hnsw-ef", type=int, default=256)
    parser.add_argument("--qdrant-full-scan-threshold", type=int, default=10)
    parser.add_argument("--qdrant-timeout-s", type=float, default=300.0)
    parser.add_argument("--qdrant-ready-wait-s", type=float, default=600.0)
    parser.add_argument("--keep-qdrant-collections", action="store_true")
    args = parser.parse_args()

    emb_dir = args.dataset_root / "processed" / "hotpot_portable" / "embeddings" / args.embedding
    docs = _normalize(np.load(emb_dir / "doc_vectors.npy"))
    queries = _normalize(np.load(emb_dir / "query_vectors.npy")[: args.query_count])

    results: dict[str, Any] = {
        "description": (
            "Controlled vector-index scale stress test. Uses real HOTPOTQA-MAXIONBENCH "
            "bge-small document/query embeddings plus deterministic random unit-vector "
            "distractors. Qdrant HNSW recall is measured against exact FAISS FlatIP "
            "top-10 at the same vector scale. This is not a full MaxionBench "
            "engine/workload row."
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "base_documents": int(len(docs)),
        "embedding": args.embedding,
        "seed": int(args.seed),
        "qdrant_url": args.qdrant_url,
        "qdrant_config": {
            "m": int(args.qdrant_m),
            "ef_construct": int(args.qdrant_ef_construct),
            "hnsw_ef": int(args.qdrant_hnsw_ef),
            "full_scan_threshold": int(max(10, args.qdrant_full_scan_threshold)),
            "batch_size": int(args.qdrant_batch_size),
        },
        "results": [],
    }

    for scale in args.scales:
        row = _benchmark_scale(
            docs,
            queries,
            scale=scale,
            seed=args.seed,
            k=args.k,
            args=args,
        )
        results["results"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
