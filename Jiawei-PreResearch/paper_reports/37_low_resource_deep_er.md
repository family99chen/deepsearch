# Low-resource Deep Entity Resolution with Transfer and Active Learning

## 1. Metadata
- Index: 37
- Track: Scholarly Identity
- Year: 2019
- Venue/Source: ACL 2019
- Priority: P1
- Authors: Yuliang Li et al.
- PDF: [37_low_resource_deep_er_low_resource_deep_er.pdf](../paper/37_low_resource_deep_er_low_resource_deep_er.pdf)
- Link: https://aclanthology.org/P19-1586/

## 2. Problem and Contribution
- Problem focus: Low-resource transfer + active learning for entity resolution.
- Abstract highlights: This paper presents transfer learning and active learning methods to reduce labeling cost for deep entity resolution in low-resource settings.

## 3. Integration with deepsearch
- Upgrade fetch_author_google_scholar_account.py from heuristics to neural entity matching.
- Send low-confidence matches to human review queue and feed back labels.
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
- Report path: `paper_reports\37_low_resource_deep_er.md`
- PDF path: `paper\37_low_resource_deep_er_low_resource_deep_er.pdf`
