# REALM: Retrieval-Augmented Language Model Pre-Training

## 1. Metadata
- Index: 13
- Track: Retrieval & RAG
- Year: 2020
- Venue/Source: ICML 2020
- Priority: P1
- Authors: Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang
- PDF: [13_realm_2002.08909.pdf](../paper/13_realm_2002.08909.pdf)
- Link: https://arxiv.org/abs/2002.08909

## 2. Problem and Contribution
- Problem focus: Retrieval-augmented pretraining for knowledge-intensive tasks.
- Abstract highlights: Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts. To capture knowledge in a more modular and interpretable way, we augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents. We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.

## 3. Integration with deepsearch
- Add dense retrieval stage after Google API recall in google_search_api/google_search.py and org_info/google_org_search.py.
- Train domain-adaptive contrastive retriever on internal person-page corpus.
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
- Report path: `paper_reports\13_realm.md`
- PDF path: `paper\13_realm_2002.08909.pdf`
