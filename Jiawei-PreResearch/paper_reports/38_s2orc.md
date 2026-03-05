# S2ORC: The Semantic Scholar Open Research Corpus

## 1. Metadata
- Index: 38
- Track: Scholarly Identity
- Year: 2019
- Venue/Source: ACL 2020
- Priority: P1
- Authors: Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, Dan S. Weld
- PDF: [38_s2orc_1911.02782.pdf](../paper/38_s2orc_1911.02782.pdf)
- Link: https://arxiv.org/abs/1911.02782

## 2. Problem and Contribution
- Problem focus: Large-scale scholarly corpus for scientific NLP and matching.
- Abstract highlights: We introduce S2ORC, a large corpus of 81.1M English-language academic papers spanning many academic disciplines. The corpus consists of rich metadata, paper abstracts, resolved bibliographic references, as well as structured full text for 8.1M open access papers. Full text is annotated with automatically-detected inline mentions of citations, figures, and tables, each linked to their corresponding paper objects. In S2ORC, we aggregate papers from hundreds of academic publishers and digital archives into a unified source, and create the largest publicly-available collection of machine-readable academic text to date. We hope this resource will facilitate research and development of tools and tasks for text mining over academic text.

## 3. Integration with deepsearch
- Use S2ORC-derived lexical priors for title/abstract matching robustness.
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
- Report path: `paper_reports\38_s2orc.md`
- PDF path: `paper\38_s2orc_1911.02782.pdf`
