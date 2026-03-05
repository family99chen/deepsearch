# Dense Passage Retrieval for Open-Domain Question Answering

## 1. Metadata
- Index: 14
- Track: Retrieval & RAG
- Year: 2020
- Venue/Source: EMNLP 2020
- Priority: P0
- Authors: Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih
- PDF: [14_dpr_2004.04906.pdf](../paper/14_dpr_2004.04906.pdf)
- Link: https://arxiv.org/abs/2004.04906

## 2. Problem and Contribution
- Problem focus: Bi-encoder dense retrieval baseline for open-domain QA.
- Abstract highlights: Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.

## 3. Integration with deepsearch
- Add dense retrieval stage after Google API recall in google_search_api/google_search.py and org_info/google_org_search.py.
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
- Report path: `paper_reports\14_dpr.md`
- PDF path: `paper\14_dpr_2004.04906.pdf`
