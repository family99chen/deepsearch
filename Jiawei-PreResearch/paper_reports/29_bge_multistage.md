# Towards General Text Embeddings with Multi-stage Contrastive Learning

## 1. Metadata
- Index: 29
- Track: Retrieval & RAG
- Year: 2023
- Venue/Source: arXiv 2023
- Priority: P1
- Authors: Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang
- PDF: [29_bge_multistage_2308.03281.pdf](../paper/29_bge_multistage_2308.03281.pdf)
- Link: https://arxiv.org/abs/2308.03281

## 2. Problem and Contribution
- Problem focus: BGE pipeline for robust embeddings and rerankers.
- Abstract highlights: We present GTE, a general-purpose text embedding model trained with multi-stage contrastive learning. In line with recent advancements in unifying various NLP tasks into a single format, we train a unified text embedding model by employing contrastive learning over a diverse mixture of datasets from multiple sources. By significantly increasing the number of training data during both unsupervised pre-training and supervised fine-tuning stages, we achieve substantial performance gains over existing embedding models. Notably, even with a relatively modest parameter count of 110M, GTE$_\text{base}$ outperforms the black-box embedding API provided by OpenAI and even surpasses 10x larger text embedding models on the massive text embedding benchmark. Furthermore, without additional fine-tuning on each programming language individually, our model outperforms previous best code retrievers of similar size by treating code as text. In summary, our model achieves impressive results by effectively harnessing multi-stage contrastive learning, offering a powerful and efficient text embedding model with broad applicability across various NLP and code-related tasks.

## 3. Integration with deepsearch
- Add embedding cache/index next to localdb/insert_mongo.py for fast semantic retrieval.
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
- Report path: `paper_reports\29_bge_multistage.md`
- PDF path: `paper\29_bge_multistage_2308.03281.pdf`
