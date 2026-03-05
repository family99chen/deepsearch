# Corrective Retrieval Augmented Generation

## 1. 基本信息
- 序号: 25
- 研究主线: Retrieval & RAG
- 年份: 2024
- 会议/来源: arXiv 2024
- 优先级: P0
- 作者: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
- PDF: [25_crag_2401.15884.pdf](../paper/25_crag_2401.15884.pdf)
- 链接: https://arxiv.org/abs/2401.15884

## 2. 研究问题与贡献
- 问题焦点: Corrective mechanisms when retrieval quality is poor.
- 摘要要点: Large language models (LLMs) inevitably exhibit hallucinations since the accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate. Although retrieval-augmented generation (RAG) is a practicable complement to LLMs, it relies heavily on the relevance of retrieved documents, raising concerns about how the model behaves if retrieval goes wrong. To this end, we propose the Corrective Retrieval Augmented Generation (CRAG) to improve the robustness of generation. Specifically, a lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query, returning a confidence degree based on which different knowledge retrieval actions can be triggered. Since retrieval from static and limited corpora can only return sub-optimal documents, large-scale web searches are utilized as an extension for augmenting the retrieval results. Besides, a decompose-then-recompose algorithm is designed for retrieved documents to selectively focus on key information and filter out irrelevant information in them. CRAG is plug-and-play and can be seamlessly coupled with various RAG-based approaches. Experiments on four datasets covering short- and long-form generation tasks show that CRAG can significantly improve the performance of RAG-based approaches.

## 3. 与 deepsearch 的融合点
- Add retrieval quality gating in pipeline.py; trigger re-retrieval when evidence is weak.
- Implement correction branch for conflicting or low-confidence evidence.
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
- 报告路径: `paper_reports\25_crag.md`
- PDF 路径: `paper\25_crag_2401.15884.pdf`
