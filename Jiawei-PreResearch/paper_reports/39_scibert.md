# SciBERT: A Pretrained Language Model for Scientific Text

## 1. Metadata
- Index: 39
- Track: Scholarly Identity
- Year: 2019
- Venue/Source: EMNLP 2019
- Priority: P1
- Authors: Iz Beltagy, Kyle Lo, Arman Cohan
- PDF: [39_scibert_1903.10676.pdf](../paper/39_scibert_1903.10676.pdf)
- Link: https://arxiv.org/abs/1903.10676

## 2. Problem and Contribution
- Problem focus: Domain-specific LM for scientific text understanding.
- Abstract highlights: Obtaining large-scale annotated data for NLP tasks in the scientific domain is challenging and expensive. We release SciBERT, a pretrained language model based on BERT (Devlin et al., 2018) to address the lack of high-quality, large-scale labeled scientific data. SciBERT leverages unsupervised pretraining on a large multi-domain corpus of scientific publications to improve performance on downstream scientific NLP tasks. We evaluate on a suite of tasks including sequence tagging, sentence classification and dependency parsing, with datasets from a variety of scientific domains. We demonstrate statistically significant improvements over BERT and achieve new state-of-the-art results on several of these tasks. The code and pretrained models are available at this https URL .

## 3. Integration with deepsearch
- Use SciBERT embeddings for scientific name/title disambiguation.
- Suggested code anchors:
  - `google_scholar_url/google_account_fetcher_pipeline.py`
  - `google_scholar_url/fetch_author_google_scholar_account.py`
  - `google_scholar_url/verify_author_info.py`
  - `google_scholar_url/name_normalization.py`

## 4. MVP Implementation Plan
- Build a fixed offline evaluation set (matching accuracy, retrieval recall, evidence coverage).
- Enable strategy in shadow mode and compare quality/cost/latency with rollback switch.
- Store low-confidence cases for iterative data and rule improvements.

## 5. Expected Gains and Risks
- Expected gains: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- Main risks: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. Artifact Paths
- Report path: `paper_reports\39_scibert.md`
- PDF path: `paper\39_scibert_1903.10676.pdf`
