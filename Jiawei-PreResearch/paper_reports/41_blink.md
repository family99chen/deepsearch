# Scalable Zero-shot Entity Linking with Dense Entity Retrieval

## 1. Metadata
- Index: 41
- Track: Scholarly Identity
- Year: 2019
- Venue/Source: EMNLP 2020
- Priority: P1
- Authors: Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, Luke Zettlemoyer
- PDF: [41_blink_1911.03814.pdf](../paper/41_blink_1911.03814.pdf)
- Link: https://arxiv.org/abs/1911.03814

## 2. Problem and Contribution
- Problem focus: Scalable dense retrieval for entity linking.
- Abstract highlights: This paper introduces a conceptually simple, scalable, and highly effective BERT-based entity linking model, along with an extensive evaluation of its accuracy-speed trade-off. We present a two-stage zero-shot linking algorithm, where each entity is defined only by a short textual description. The first stage does retrieval in a dense space defined by a bi-encoder that independently embeds the mention context and the entity descriptions. Each candidate is then re-ranked with a cross-encoder, that concatenates the mention and entity text. Experiments demonstrate that this approach is state of the art on recent zero-shot benchmarks (6 point absolute gains) and also on more established non-zero-shot evaluations (e.g. TACKBP-2010), despite its relative simplicity (e.g. no explicit entity embeddings or manually engineered mention tables). We also show that bi-encoder linking is very fast with nearest neighbour search (e.g. linking with 5.9 million candidates in 2 milliseconds), and that much of the accuracy gain from the more expensive cross-encoder can be transferred to the bi-encoder via knowledge distillation. Our code and models are available at this https URL .

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
- Report path: `paper_reports\41_blink.md`
- PDF path: `paper\41_blink_1911.03814.pdf`
