# Precise Zero-Shot Dense Retrieval without Relevance Labels

## 1. Metadata
- Index: 23
- Track: Retrieval & RAG
- Year: 2022
- Venue/Source: ACL 2023
- Priority: P1
- Authors: Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- PDF: [23_hyde_2212.10496.pdf](../paper/23_hyde_2212.10496.pdf)
- Link: https://arxiv.org/abs/2212.10496

## 2. Problem and Contribution
- Problem focus: Hypothetical answer generation for zero-shot retrieval.
- Abstract highlights: While dense retrieval has been shown effective and efficient across tasks and languages, it remains difficult to create effective fully zero-shot dense retrieval systems when no relevance label is available. In this paper, we recognize the difficulty of zero-shot learning and encoding relevance. Instead, we propose to pivot through Hypothetical Document Embeddings~(HyDE). Given a query, HyDE first zero-shot instructs an instruction-following language model (e.g. InstructGPT) to generate a hypothetical document. The document captures relevance patterns but is unreal and may contain false details. Then, an unsupervised contrastively learned encoder~(e.g. Contriever) encodes the document into an embedding vector. This vector identifies a neighborhood in the corpus embedding space, where similar real documents are retrieved based on vector similarity. This second step ground the generated document to the actual corpus, with the encoder's dense bottleneck filtering out the incorrect details. Our experiments show that HyDE significantly outperforms the state-of-the-art unsupervised dense retriever Contriever and shows strong performance comparable to fine-tuned retrievers, across various tasks (e.g. web search, QA, fact verification) and languages~(e.g. sw, ko, ja).

## 3. Integration with deepsearch
- Add hypothetical-answer query expansion in iterative query refinement.
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
- Report path: `paper_reports\23_hyde.md`
- PDF path: `paper\23_hyde_2212.10496.pdf`
