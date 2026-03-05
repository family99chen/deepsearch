# SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval

## 1. 基本信息
- 序号: 21
- 研究主线: Retrieval & RAG
- 年份: 2021
- 会议/来源: SIGIR 2022
- 优先级: P1
- 作者: Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant
- PDF: [21_splade_v2_2109.10086.pdf](../paper/21_splade_v2_2109.10086.pdf)
- 链接: https://arxiv.org/abs/2109.10086

## 2. 研究问题与贡献
- 问题焦点: Sparse lexical-expansion retriever for first-stage ranking.
- 摘要要点: In neural Information Retrieval (IR), ongoing research is directed towards improving the first retriever in ranking pipelines. Learning dense embeddings to conduct retrieval using efficient approximate nearest neighbors methods has proven to work well. Meanwhile, there has been a growing interest in learning \emph{sparse} representations for documents and queries, that could inherit from the desirable properties of bag-of-words models such as the exact matching of terms and the efficiency of inverted indexes. Introduced recently, the SPLADE model provides highly sparse representations and competitive results with respect to state-of-the-art dense and sparse approaches. In this paper, we build on SPLADE and propose several significant improvements in terms of effectiveness and/or efficiency. More specifically, we modify the pooling mechanism, benchmark a model solely based on document expansion, and introduce models trained with distillation. We also report results on the BEIR benchmark. Overall, SPLADE is considerably improved with more than $9$\% gains on NDCG@10 on TREC DL 2019, leading to state-of-the-art results on the BEIR benchmark.

## 3. 与 deepsearch 的融合点
- Combine sparse lexical features with current search pipeline for long-tail queries.
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
- 报告路径: `paper_reports\21_splade_v2.md`
- PDF 路径: `paper\21_splade_v2_2109.10086.pdf`
