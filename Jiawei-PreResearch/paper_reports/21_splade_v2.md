# SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval

## 1. Metadata
- Index: 21
- Track: Retrieval & RAG
- Year: 2021
- Venue/Source: SIGIR 2022
- Priority: P1
- Authors: Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant
- PDF: [21_splade_v2_2109.10086.pdf](../paper/21_splade_v2_2109.10086.pdf)
- Link: https://arxiv.org/abs/2109.10086

## 2. Problem and Contribution
- Problem focus: Sparse lexical-expansion retriever for first-stage ranking.
- Abstract highlights: In neural Information Retrieval (IR), ongoing research is directed towards improving the first retriever in ranking pipelines. Learning dense embeddings to conduct retrieval using efficient approximate nearest neighbors methods has proven to work well. Meanwhile, there has been a growing interest in learning \emph{sparse} representations for documents and queries, that could inherit from the desirable properties of bag-of-words models such as the exact matching of terms and the efficiency of inverted indexes. Introduced recently, the SPLADE model provides highly sparse representations and competitive results with respect to state-of-the-art dense and sparse approaches. In this paper, we build on SPLADE and propose several significant improvements in terms of effectiveness and/or efficiency. More specifically, we modify the pooling mechanism, benchmark a model solely based on document expansion, and introduce models trained with distillation. We also report results on the BEIR benchmark. Overall, SPLADE is considerably improved with more than $9$\% gains on NDCG@10 on TREC DL 2019, leading to state-of-the-art results on the BEIR benchmark.

## 3. Integration with deepsearch
- Combine sparse lexical features with current search pipeline for long-tail queries.
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
- Report path: `paper_reports\21_splade_v2.md`
- PDF path: `paper\21_splade_v2_2109.10086.pdf`
