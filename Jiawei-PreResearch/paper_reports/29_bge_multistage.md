# Towards General Text Embeddings with Multi-stage Contrastive Learning

## 1. 基本信息
- 序号: 29
- 研究主线: Retrieval & RAG
- 年份: 2023
- 会议/来源: arXiv 2023
- 优先级: P1
- 作者: Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang
- PDF: [29_bge_multistage_2308.03281.pdf](../paper/29_bge_multistage_2308.03281.pdf)
- 链接: https://arxiv.org/abs/2308.03281

## 2. 研究问题与贡献
- 问题焦点: BGE pipeline for robust embeddings and rerankers.
- 摘要要点: We present GTE, a general-purpose text embedding model trained with multi-stage contrastive learning. In line with recent advancements in unifying various NLP tasks into a single format, we train a unified text embedding model by employing contrastive learning over a diverse mixture of datasets from multiple sources. By significantly increasing the number of training data during both unsupervised pre-training and supervised fine-tuning stages, we achieve substantial performance gains over existing embedding models. Notably, even with a relatively modest parameter count of 110M, GTE$_\text{base}$ outperforms the black-box embedding API provided by OpenAI and even surpasses 10x larger text embedding models on the massive text embedding benchmark. Furthermore, without additional fine-tuning on each programming language individually, our model outperforms previous best code retrievers of similar size by treating code as text. In summary, our model achieves impressive results by effectively harnessing multi-stage contrastive learning, offering a powerful and efficient text embedding model with broad applicability across various NLP and code-related tasks.

## 3. 与 deepsearch 的融合点
- Add embedding cache/index next to localdb/insert_mongo.py for fast semantic retrieval.
- Apply LLM reranking after candidate generation in google_scholar_url/fetch_google_scholar_name_list.py.
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
- 报告路径: `paper_reports\29_bge_multistage.md`
- PDF 路径: `paper\29_bge_multistage_2308.03281.pdf`
