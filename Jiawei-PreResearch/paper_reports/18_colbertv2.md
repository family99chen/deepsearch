# ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction

## 1. Metadata
- Index: 18
- Track: Retrieval & RAG
- Year: 2021
- Venue/Source: NAACL 2022
- Priority: P0
- Authors: Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia
- PDF: [18_colbertv2_2112.01488.pdf](../paper/18_colbertv2_2112.01488.pdf)
- Link: https://arxiv.org/abs/2112.01488

## 2. Problem and Contribution
- Problem focus: Compressed late-interaction retriever suited to production IR.
- Abstract highlights: Neural information retrieval (IR) has greatly advanced search and other knowledge-intensive language tasks. While many neural IR methods encode queries and documents into single-vector representations, late interaction models produce multi-vector representations at the granularity of each token and decompose relevance modeling into scalable token-level computations. This decomposition has been shown to make late interaction more effective, but it inflates the space footprint of these models by an order of magnitude. In this work, we introduce ColBERTv2, a retriever that couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction. We evaluate ColBERTv2 across a wide range of benchmarks, establishing state-of-the-art quality within and outside the training domain while reducing the space footprint of late interaction models by 6--10$\times$.

## 3. Integration with deepsearch
- Add ColBERT-style second-stage reranking for candidate URL lists.
- Apply LLM reranking after candidate generation in google_scholar_url/fetch_google_scholar_name_list.py.
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
- Report path: `paper_reports\18_colbertv2.md`
- PDF path: `paper\18_colbertv2_2112.01488.pdf`
