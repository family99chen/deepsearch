# One Embedder, Any Task: Instruction-Finetuned Text Embeddings

## 1. 基本信息
- 序号: 27
- 研究主线: Retrieval & RAG
- 年份: 2022
- 会议/来源: arXiv 2022
- 优先级: P1
- 作者: Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, Tao Yu
- PDF: [27_instructor_2212.09741.pdf](../paper/27_instructor_2212.09741.pdf)
- 链接: https://arxiv.org/abs/2212.09741

## 2. 研究问题与贡献
- 问题焦点: Instruction-tuned embeddings for task-adaptive retrieval.
- 摘要要点: We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions: every text input is embedded together with instructions explaining the use case (e.g., task and domain descriptions). Unlike encoders from prior work that are more specialized, INSTRUCTOR is a single embedder that can generate text embeddings tailored to different downstream tasks and domains, without any further training. We first annotate instructions for 330 diverse tasks and train INSTRUCTOR on this multitask mixture with a contrastive loss. We evaluate INSTRUCTOR on 70 embedding evaluation tasks (66 of which are unseen during training), ranging from classification and information retrieval to semantic textual similarity and text generation evaluation. INSTRUCTOR, while having an order of magnitude fewer parameters than the previous best model, achieves state-of-the-art performance, with an average improvement of 3.4% compared to the previous best results on the 70 diverse datasets. Our analysis suggests that INSTRUCTOR is robust to changes in instructions, and that instruction finetuning mitigates the challenge of training a single model on diverse datasets. Our model, code, and data are available at this https URL .

## 3. 与 deepsearch 的融合点
- Add embedding cache/index next to localdb/insert_mongo.py for fast semantic retrieval.
- 建议代码锚点:
  - `google_search_api/google_search.py`
  - `org_info/google_org_search.py`
  - `org_info/arbitrary_search.py`
  - `pipeline.py`
  - `localdb/insert_mongo.py`

## 4. MVP 落地计划
- 建立固定离线评测集（匹配准确率、检索召回率、证据覆盖率）。
- 以影子模式启用策略，对比质量/成本/时延并保留回滚开关。
- 沉淀低置信样本，持续迭代数据与规则。

## 5. 预期收益与风险
- 预期收益: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- 主要风险: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. 产物路径
- 报告路径: `paper_reports\27_instructor.md`
- PDF 路径: `paper\27_instructor_2212.09741.pdf`
