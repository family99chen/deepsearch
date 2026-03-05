# BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models

## 1. Metadata
- Index: 20
- Track: Retrieval & RAG
- Year: 2021
- Venue/Source: NeurIPS Benchmarks 2021
- Priority: P0
- Authors: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych
- PDF: [20_beir_2104.08663.pdf](../paper/20_beir_2104.08663.pdf)
- Link: https://arxiv.org/abs/2104.08663

## 2. Problem and Contribution
- Problem focus: Zero-shot retrieval benchmark for realistic cross-domain evaluation.
- Abstract highlights: Existing neural information retrieval (IR) models have often been studied in homogeneous and narrow settings, which has considerably limited insights into their out-of-distribution (OOD) generalization capabilities. To address this, and to facilitate researchers to broadly evaluate the effectiveness of their models, we introduce Benchmarking-IR (BEIR), a robust and heterogeneous evaluation benchmark for information retrieval. We leverage a careful selection of 18 publicly available datasets from diverse text retrieval tasks and domains and evaluate 10 state-of-the-art retrieval systems including lexical, sparse, dense, late-interaction and re-ranking architectures on the BEIR benchmark. Our results show BM25 is a robust baseline and re-ranking and late-interaction-based models on average achieve the best zero-shot performances, however, at high computational costs. In contrast, dense and sparse-retrieval models are computationally more efficient but often underperform other approaches, highlighting the considerable room for improvement in their generalization capabilities. We hope this framework allows us to better evaluate and understand existing retrieval systems, and contributes to accelerating progress towards better robust and generalizable systems in the future. BEIR is publicly available at this https URL .

## 3. Integration with deepsearch
- Track retrieval metrics (Recall@K, NDCG@K, evidence precision) in utils/pipeline_stats.py.
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
- Report path: `paper_reports\20_beir.md`
- PDF path: `paper\20_beir_2104.08663.pdf`
