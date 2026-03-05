# M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

## 1. Metadata
- Index: 30
- Track: Retrieval & RAG
- Year: 2024
- Venue/Source: arXiv 2024
- Priority: P1
- Authors: Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu
- PDF: [30_m3_embedding_2402.03216.pdf](../paper/30_m3_embedding_2402.03216.pdf)
- Link: https://arxiv.org/abs/2402.03216

## 2. Problem and Contribution
- Problem focus: Unified multilingual embedding model across granularities.
- Abstract highlights: In this paper, we introduce a new embedding model called M3-Embedding, which is distinguished for its versatility in \textit{Multi-Linguality}, \textit{Multi-Functionality}, and \textit{Multi-Granularity}. It provides a uniform support for the semantic retrieval of more than 100 working languages. It can simultaneously accomplish the three common retrieval functionalities: dense retrieval, multi-vector retrieval, and sparse retrieval. Besides, it is also capable of processing inputs of different granularities, spanning from short sentences to long documents of up to 8,192 tokens. The effective training of M3-Embedding presents a series of technical contributions. Notably, we propose a novel self-knowledge distillation approach, where the relevance scores from different retrieval functionalities can be integrated as the teacher signal to enhance the training quality. We also optimize the batching strategy, which enables a large batch size and high training throughput to improve the discriminativeness of embeddings. M3-Embedding exhibits a superior performance in our experiment, leading to new state-of-the-art results on multilingual, cross-lingual, and long-document retrieval benchmarks.

## 3. Integration with deepsearch
- Add embedding cache/index next to localdb/insert_mongo.py for fast semantic retrieval.
- Unify cross-lingual name variants (Chinese/Pinyin/English) in embedding index.
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
- Report path: `paper_reports\30_m3_embedding.md`
- PDF path: `paper\30_m3_embedding_2402.03216.pdf`
