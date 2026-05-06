# Artifact Licenses and Release Boundaries

| Asset | Role in paper | Source citation | License or terms file in artifact | Release boundary |
|---|---|---|---|---|
| SciFact | S1/S2 BEIR-format retrieval asset | Wadden et al. 2020 | CC BY-NC 2.0; source metadata: https://huggingface.co/datasets/allenai/scifact/blob/main/README.md | processed evaluation bundle only |
| FiQA | S1/S2 BEIR-format retrieval asset | Maia et al. 2018 | CC BY-SA 4.0; source metadata: https://huggingface.co/datasets/vibrantlabsai/fiqa | processed evaluation bundle only |
| CRAG | S2 event stream source | Facebook Research 2024 | CC BY-NC 4.0; source repository: https://github.com/facebookresearch/CRAG/ | CRAG-500 event-stream derivation only |
| HotpotQA dev distractor | S3 source | Yang et al. 2018 | CC BY-SA 4.0; source metadata: `paper/metadata/hotpotqa_portable_croissant.jsonld` | `HOTPOTQA-MAXIONBENCH` preprocessing only |
| `BAAI/bge-small-en-v1.5` | embedding tier | BAAI 2023 | MIT; source model card: https://huggingface.co/BAAI/bge-small-en-v1.5 | local embedding tier only |
| `BAAI/bge-base-en-v1.5` | embedding tier | BAAI 2023 | MIT; source model card: https://huggingface.co/BAAI/bge-base-en-v1.5 | local embedding tier only |
| FAISS | engine | Johnson et al. 2019 | MIT License; source package metadata: `faiss_cpu-1.8.0.post1.dist-info/LICENSE` | reportable engine |
| LanceDB | engine | LanceDB 2024 | Apache License 2.0; source package metadata: `lancedb-0.25.3.dist-info/licenses/LICENSE` | `lancedb-inproc` reported; service excluded |
| pgvector | engine | pgvector contributors 2024 | PostgreSQL-style license text; source repository: https://github.com/pgvector/pgvector/blob/master/LICENSE | reportable engine |
| Qdrant | engine | Qdrant contributors 2024 | Apache License 2.0; source repository: https://github.com/qdrant/qdrant#license | reportable engine |
