# S2AND: A Benchmark and Evaluation System for Author Name Disambiguation

## 1. Metadata
- Index: 33
- Track: Scholarly Identity
- Year: 2021
- Venue/Source: JCDL 2021
- Priority: P0
- Authors: Shivashankar Subramanian, Daniel King, Doug Downey, Sergey Feldman
- PDF: [33_s2and_2103.07534.pdf](../paper/33_s2and_2103.07534.pdf)
- Link: https://arxiv.org/abs/2103.07534

## 2. Problem and Contribution
- Problem focus: Benchmark and system design for author name disambiguation.
- Abstract highlights: Author Name Disambiguation (AND) is the task of resolving which author mentions in a bibliographic database refer to the same real-world person, and is a critical ingredient of digital library applications such as search and citation analysis. While many AND algorithms have been proposed, comparing them is difficult because they often employ distinct features and are evaluated on different datasets. In response to this challenge, we present S2AND, a unified benchmark dataset for AND on scholarly papers, as well as an open-source reference model implementation. Our dataset harmonizes eight disparate AND datasets into a uniform format, with a single rich feature set drawn from the Semantic Scholar (S2) database. Our evaluation suite for S2AND reports performance split by facets like publication year and number of papers, allowing researchers to track both global performance and measures of fairness across facet values. Our experiments show that because previous datasets tend to cover idiosyncratic and biased slices of the literature, algorithms trained to perform well on one on them may generalize poorly to others. By contrast, we show how training on a union of datasets in S2AND results in more robust models that perform well even on datasets unseen in training. The resulting AND model also substantially improves over the production algorithm in S2, reducing error by over 50% in...

## 3. Integration with deepsearch
- Replace rule-only ORCID->Scholar matching with learned scorer in google_account_fetcher_pipeline.py.
- Expand pairwise features in google_scholar_url/verify_author_info.py.
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
- Report path: `paper_reports\33_s2and.md`
- PDF path: `paper\33_s2and_2103.07534.pdf`
