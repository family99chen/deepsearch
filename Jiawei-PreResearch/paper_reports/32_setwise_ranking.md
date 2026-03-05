# A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models

## 1. Metadata
- Index: 32
- Track: Retrieval & RAG
- Year: 2023
- Venue/Source: arXiv 2023
- Priority: P1
- Authors: Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, Guido Zuccon
- PDF: [32_setwise_ranking_2310.09497.pdf](../paper/32_setwise_ranking_2310.09497.pdf)
- Link: https://arxiv.org/abs/2310.09497

## 2. Problem and Contribution
- Problem focus: Efficient setwise ranking strategy for LLM reranking.
- Abstract highlights: We propose a novel zero-shot document ranking approach based on Large Language Models (LLMs): the Setwise prompting approach. Our approach complements existing prompting approaches for LLM-based zero-shot ranking: Pointwise, Pairwise, and Listwise. Through the first-of-its-kind comparative evaluation within a consistent experimental framework and considering factors like model size, token consumption, latency, among others, we show that existing approaches are inherently characterised by trade-offs between effectiveness and efficiency. We find that while Pointwise approaches score high on efficiency, they suffer from poor effectiveness. Conversely, Pairwise approaches demonstrate superior effectiveness but incur high computational overhead. Our Setwise approach, instead, reduces the number of LLM inferences and the amount of prompt token consumption during the ranking procedure, compared to previous methods. This significantly improves the efficiency of LLM-based zero-shot ranking, while also retaining high zero-shot ranking effectiveness. We make our code and results publicly available at \url{ this https URL }.

## 3. Integration with deepsearch
- Apply LLM reranking after candidate generation in google_scholar_url/fetch_google_scholar_name_list.py.
- Use setwise/listwise LLM reranking in org_info/google_org_search.py top-k outputs.
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
- Report path: `paper_reports\32_setwise_ranking.md`
- PDF path: `paper\32_setwise_ranking_2310.09497.pdf`
