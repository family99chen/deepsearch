# DeepMatcher: A Deep Learning Approach to Entity Matching

## 1. Metadata
- Index: 36
- Track: Scholarly Identity
- Year: 2018
- Venue/Source: SIGMOD 2018
- Priority: P0
- Authors: Sidharth Mudgal et al.
- PDF: [36_deepmatcher_deepmatcher.pdf](../paper/36_deepmatcher_deepmatcher.pdf)
- Link: http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf

## 2. Problem and Contribution
- Problem focus: Classic deep entity matching architecture and feature interactions.
- Abstract highlights: DeepMatcher studies deep learning model variants for entity matching and proposes practical architecture choices for robust record linkage.

## 3. Integration with deepsearch
- Upgrade fetch_author_google_scholar_account.py from heuristics to neural entity matching.
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
- Report path: `paper_reports\36_deepmatcher.md`
- PDF path: `paper\36_deepmatcher_deepmatcher.pdf`
