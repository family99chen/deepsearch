# SPECTER: Document-level Representation Learning using Citation-informed Transformers

## 1. Metadata
- Index: 40
- Track: Scholarly Identity
- Year: 2020
- Venue/Source: ACL 2020
- Priority: P1
- Authors: Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel S. Weld
- PDF: [40_specter_2004.07180.pdf](../paper/40_specter_2004.07180.pdf)
- Link: https://arxiv.org/abs/2004.07180

## 2. Problem and Contribution
- Problem focus: Citation-informed document embeddings for paper similarity.
- Abstract highlights: Representation learning is a critical ingredient for natural language processing systems. Recent Transformer language models like BERT learn powerful textual representations, but these models are targeted towards token- and sentence-level training objectives and do not leverage information on inter-document relatedness, which limits their document-level representation power. For applications on scientific documents, such as classification and recommendation, the embeddings power strong performance on end tasks. We propose SPECTER, a new method to generate document-level embedding of scientific documents based on pretraining a Transformer language model on a powerful signal of document-level relatedness: the citation graph. Unlike existing pretrained language models, SPECTER can be easily applied to downstream applications without task-specific fine-tuning. Additionally, to encourage further research on document-level models, we introduce SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. We show that SPECTER outperforms a variety of competitive baselines on the benchmark.

## 3. Integration with deepsearch
- Adopt SPECTER embeddings for document-level similarity in paper matching.
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
- Report path: `paper_reports\40_specter.md`
- PDF path: `paper\40_specter_2004.07180.pdf`
