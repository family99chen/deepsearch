# Corrective Retrieval Augmented Generation

## 1. Metadata
- Index: 25
- Track: Retrieval & RAG
- Year: 2024
- Venue/Source: arXiv 2024
- Priority: P0
- Authors: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
- PDF: [25_crag_2401.15884.pdf](../paper/25_crag_2401.15884.pdf)
- Link: https://arxiv.org/abs/2401.15884

## 2. Problem and Contribution
- Problem focus: Corrective mechanisms when retrieval quality is poor.
- Abstract highlights: Large language models (LLMs) inevitably exhibit hallucinations since the accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate. Although retrieval-augmented generation (RAG) is a practicable complement to LLMs, it relies heavily on the relevance of retrieved documents, raising concerns about how the model behaves if retrieval goes wrong. To this end, we propose the Corrective Retrieval Augmented Generation (CRAG) to improve the robustness of generation. Specifically, a lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query, returning a confidence degree based on which different knowledge retrieval actions can be triggered. Since retrieval from static and limited corpora can only return sub-optimal documents, large-scale web searches are utilized as an extension for augmenting the retrieval results. Besides, a decompose-then-recompose algorithm is designed for retrieved documents to selectively focus on key information and filter out irrelevant information in them. CRAG is plug-and-play and can be seamlessly coupled with various RAG-based approaches. Experiments on four datasets covering short- and long-form generation tasks show that CRAG can significantly improve the performance of RAG-based approaches.

## 3. Integration with deepsearch
- Add retrieval quality gating in pipeline.py; trigger re-retrieval when evidence is weak.
- Implement correction branch for conflicting or low-confidence evidence.
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
- Report path: `paper_reports\25_crag.md`
- PDF path: `paper\25_crag_2401.15884.pdf`
