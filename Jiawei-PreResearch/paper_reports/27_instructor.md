# One Embedder, Any Task: Instruction-Finetuned Text Embeddings

## 1. Metadata
- Index: 27
- Track: Retrieval & RAG
- Year: 2022
- Venue/Source: arXiv 2022
- Priority: P1
- Authors: Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, Tao Yu
- PDF: [27_instructor_2212.09741.pdf](../paper/27_instructor_2212.09741.pdf)
- Link: https://arxiv.org/abs/2212.09741

## 2. Problem and Contribution
- Problem focus: Instruction-tuned embeddings for task-adaptive retrieval.
- Abstract highlights: We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions: every text input is embedded together with instructions explaining the use case (e.g., task and domain descriptions). Unlike encoders from prior work that are more specialized, INSTRUCTOR is a single embedder that can generate text embeddings tailored to different downstream tasks and domains, without any further training. We first annotate instructions for 330 diverse tasks and train INSTRUCTOR on this multitask mixture with a contrastive loss. We evaluate INSTRUCTOR on 70 embedding evaluation tasks (66 of which are unseen during training), ranging from classification and information retrieval to semantic textual similarity and text generation evaluation. INSTRUCTOR, while having an order of magnitude fewer parameters than the previous best model, achieves state-of-the-art performance, with an average improvement of 3.4% compared to the previous best results on the 70 diverse datasets. Our analysis suggests that INSTRUCTOR is robust to changes in instructions, and that instruction finetuning mitigates the challenge of training a single model on diverse datasets. Our model, code, and data are available at this https URL .

## 3. Integration with deepsearch
- Add embedding cache/index next to localdb/insert_mongo.py for fast semantic retrieval.
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
- Report path: `paper_reports\27_instructor.md`
- PDF path: `paper\27_instructor_2212.09741.pdf`
