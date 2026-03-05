# Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering

## 1. Metadata
- Index: 16
- Track: Retrieval & RAG
- Year: 2020
- Venue/Source: arXiv 2020
- Priority: P1
- Authors: Gautier Izacard, Edouard Grave
- PDF: [16_fid_2007.01282.pdf](../paper/16_fid_2007.01282.pdf)
- Link: https://arxiv.org/abs/2007.01282

## 2. Problem and Contribution
- Problem focus: Fusion-in-Decoder architecture for multi-document reasoning.
- Abstract highlights: Generative models for open domain question answering have proven to be competitive, without resorting to external knowledge. While promising, this approach requires to use models with billions of parameters, which are expensive to train and query. In this paper, we investigate how much these models can benefit from retrieving text passages, potentially containing evidence. We obtain state-of-the-art results on the Natural Questions and TriviaQA open benchmarks. Interestingly, we observe that the performance of this method significantly improves when increasing the number of retrieved passages. This is evidence that generative models are good at aggregating and combining evidence from multiple passages.

## 3. Integration with deepsearch
- Split evidence retrieval and answer generation stages in pipeline.py for reliability.
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
- Report path: `paper_reports\16_fid.md`
- PDF path: `paper\16_fid_2007.01282.pdf`
