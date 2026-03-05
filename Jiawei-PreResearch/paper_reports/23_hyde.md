# Precise Zero-Shot Dense Retrieval without Relevance Labels

## 1. 基本信息
- 序号: 23
- 研究主线: Retrieval & RAG
- 年份: 2022
- 会议/来源: ACL 2023
- 优先级: P1
- 作者: Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- PDF: [23_hyde_2212.10496.pdf](../paper/23_hyde_2212.10496.pdf)
- 链接: https://arxiv.org/abs/2212.10496

## 2. 研究问题与贡献
- 问题焦点: Hypothetical answer generation for zero-shot retrieval.
- 摘要要点: While dense retrieval has been shown effective and efficient across tasks and languages, it remains difficult to create effective fully zero-shot dense retrieval systems when no relevance label is available. In this paper, we recognize the difficulty of zero-shot learning and encoding relevance. Instead, we propose to pivot through Hypothetical Document Embeddings~(HyDE). Given a query, HyDE first zero-shot instructs an instruction-following language model (e.g. InstructGPT) to generate a hypothetical document. The document captures relevance patterns but is unreal and may contain false details. Then, an unsupervised contrastively learned encoder~(e.g. Contriever) encodes the document into an embedding vector. This vector identifies a neighborhood in the corpus embedding space, where similar real documents are retrieved based on vector similarity. This second step ground the generated document to the actual corpus, with the encoder's dense bottleneck filtering out the incorrect details. Our experiments show that HyDE significantly outperforms the state-of-the-art unsupervised dense retriever Contriever and shows strong performance comparable to fine-tuned retrievers, across various tasks (e.g. web search, QA, fact verification) and languages~(e.g. sw, ko, ja).

## 3. 与 deepsearch 的融合点
- Add hypothetical-answer query expansion in iterative query refinement.
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
- 报告路径: `paper_reports\23_hyde.md`
- PDF 路径: `paper\23_hyde_2212.10496.pdf`
