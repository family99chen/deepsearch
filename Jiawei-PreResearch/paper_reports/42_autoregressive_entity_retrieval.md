# Autoregressive Entity Retrieval

## 1. Metadata
- Index: 42
- Track: Scholarly Identity
- Year: 2020
- Venue/Source: EMNLP 2020
- Priority: P1
- Authors: Nicola De Cao, Gautier Izacard, Sebastian Riedel, Fabio Petroni
- PDF: [42_autoregressive_entity_retrieval_2010.00904.pdf](../paper/42_autoregressive_entity_retrieval_2010.00904.pdf)
- Link: https://arxiv.org/abs/2010.00904

## 2. Problem and Contribution
- Problem focus: Autoregressive entity retrieval for improved entity linking precision.
- Abstract highlights: Entities are at the center of how we represent and aggregate knowledge. For instance, Encyclopedias such as Wikipedia are structured by entities (e.g., one per Wikipedia article). The ability to retrieve such entities given a query is fundamental for knowledge-intensive tasks such as entity linking and open-domain question answering. Current approaches can be understood as classifiers among atomic labels, one for each entity. Their weight vectors are dense entity representations produced by encoding entity meta information such as their descriptions. This approach has several shortcomings: (i) context and entity affinity is mainly captured through a vector dot product, potentially missing fine-grained interactions; (ii) a large memory footprint is needed to store dense representations when considering large entity sets; (iii) an appropriately hard set of negative data has to be subsampled at training time. In this work, we propose GENRE, the first system that retrieves entities by generating their unique names, left to right, token-by-token in an autoregressive fashion. This mitigates the aforementioned technical issues since: (i) the autoregressive formulation directly captures relations between context and entity name, effectively cross encoding both; (ii) the memory footprint is greatly reduced because the parameters of our encoder-decoder architecture scale with vocabulary ...

## 3. Integration with deepsearch
- Build entity candidate store and apply dense entity linking across sources.
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
- Report path: `paper_reports\42_autoregressive_entity_retrieval.md`
- PDF path: `paper\42_autoregressive_entity_retrieval_2010.00904.pdf`
