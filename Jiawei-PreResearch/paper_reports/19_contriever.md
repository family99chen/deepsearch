# Unsupervised Dense Information Retrieval with Contrastive Learning

## 1. Metadata
- Index: 19
- Track: Retrieval & RAG
- Year: 2021
- Venue/Source: ICLR 2022
- Priority: P1
- Authors: Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave
- PDF: [19_contriever_2112.09118.pdf](../paper/19_contriever_2112.09118.pdf)
- Link: https://arxiv.org/abs/2112.09118

## 2. Problem and Contribution
- Problem focus: Unsupervised dense retriever robust across domains.
- Abstract highlights: Recently, information retrieval has seen the emergence of dense retrievers, using neural networks, as an alternative to classical sparse methods based on term-frequency. These models have obtained state-of-the-art results on datasets and tasks where large training sets are available. However, they do not transfer well to new applications with no training data, and are outperformed by unsupervised term-frequency methods such as BM25. In this work, we explore the limits of contrastive learning as a way to train unsupervised dense retrievers and show that it leads to strong performance in various retrieval settings. On the BEIR benchmark our unsupervised model outperforms BM25 on 11 out of 15 datasets for the Recall@100. When used as pre-training before fine-tuning, either on a few thousands in-domain examples or on the large MS~MARCO dataset, our contrastive model leads to improvements on the BEIR benchmark. Finally, we evaluate our approach for multi-lingual retrieval, where training data is even scarcer than for English, and show that our approach leads to strong unsupervised performance. Our model also exhibits strong cross-lingual transfer when fine-tuned on supervised English data only and evaluated on low resources language such as Swahili. We show that our unsupervised models can perform cross-lingual retrieval between different scripts, such as retrieving English document...

## 3. Integration with deepsearch
- Add dense retrieval stage after Google API recall in google_search_api/google_search.py and org_info/google_org_search.py.
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
- Report path: `paper_reports\19_contriever.md`
- PDF path: `paper\19_contriever_2112.09118.pdf`
