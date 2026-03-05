# MTEB: Massive Text Embedding Benchmark

## 1. Metadata
- Index: 28
- Track: Retrieval & RAG
- Year: 2022
- Venue/Source: NeurIPS Datasets 2022
- Priority: P1
- Authors: Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers
- PDF: [28_mteb_2210.07316.pdf](../paper/28_mteb_2210.07316.pdf)
- Link: https://arxiv.org/abs/2210.07316

## 2. Problem and Contribution
- Problem focus: Massive embedding benchmark for model selection.
- Abstract highlights: Text embeddings are commonly evaluated on a small set of datasets from a single task not covering their possible applications to other tasks. It is unclear whether state-of-the-art embeddings on semantic textual similarity (STS) can be equally well applied to other tasks like clustering or reranking. This makes progress in the field difficult to track, as various models are constantly being proposed without proper evaluation. To solve this problem, we introduce the Massive Text Embedding Benchmark (MTEB). MTEB spans 8 embedding tasks covering a total of 58 datasets and 112 languages. Through the benchmarking of 33 models on MTEB, we establish the most comprehensive benchmark of text embeddings to date. We find that no particular text embedding method dominates across all tasks. This suggests that the field has yet to converge on a universal text embedding method and scale it up sufficiently to provide state-of-the-art results on all embedding tasks. MTEB comes with open-source code and a public leaderboard at this https URL .

## 3. Integration with deepsearch
- Track retrieval metrics (Recall@K, NDCG@K, evidence precision) in utils/pipeline_stats.py.
- Add embedding cache/index next to localdb/insert_mongo.py for fast semantic retrieval.
- Suggested code anchors:
  - `google_search_api/google_search.py`
  - `org_info/google_org_search.py`
  - `org_info/arbitrary_search.py`
  - `pipeline.py`
  - `localdb/insert_mongo.py`

## 4. MVP Implementation Plan
- Build a fixed offline evaluation set (matching accuracy, retrieval recall, evidence coverage).
- Enable strategy in shadow mode and compare quality/cost/latency with rollback switch.
- Store low-confidence cases for iterative data and rule improvements.

## 5. Expected Gains and Risks
- Expected gains: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- Main risks: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. Artifact Paths
- Report path: `paper_reports\28_mteb.md`
- PDF path: `paper\28_mteb_2210.07316.pdf`
