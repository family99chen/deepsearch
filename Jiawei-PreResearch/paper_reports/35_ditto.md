# Deep Entity Matching with Pre-Trained Language Models

## 1. Metadata
- Index: 35
- Track: Scholarly Identity
- Year: 2020
- Venue/Source: VLDB 2020
- Priority: P0
- Authors: Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, Wang-Chiew Tan
- PDF: [35_ditto_2004.00584.pdf](../paper/35_ditto_2004.00584.pdf)
- Link: https://arxiv.org/abs/2004.00584

## 2. Problem and Contribution
- Problem focus: PLM-based entity matching with strong practical performance.
- Abstract highlights: We present Ditto, a novel entity matching system based on pre-trained Transformer-based language models. We fine-tune and cast EM as a sequence-pair classification problem to leverage such models with a simple architecture. Our experiments show that a straightforward application of language models such as BERT, DistilBERT, or RoBERTa pre-trained on large text corpora already significantly improves the matching quality and outperforms previous state-of-the-art (SOTA), by up to 29% of F1 score on benchmark datasets. We also developed three optimization techniques to further improve Ditto's matching capability. Ditto allows domain knowledge to be injected by highlighting important pieces of input information that may be of interest when making matching decisions. Ditto also summarizes strings that are too long so that only the essential information is retained and used for EM. Finally, Ditto adapts a SOTA technique on data augmentation for text to EM to augment the training data with (difficult) examples. This way, Ditto is forced to learn "harder" to improve the model's matching capability. The optimizations we developed further boost the performance of Ditto by up to 9.8%. Perhaps more surprisingly, we establish that Ditto can achieve the previous SOTA results with at most half the number of labeled data. Finally, we demonstrate Ditto's effectiveness on a real-world large-scale ...

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
- Report path: `paper_reports\35_ditto.md`
- PDF path: `paper\35_ditto_2004.00584.pdf`
