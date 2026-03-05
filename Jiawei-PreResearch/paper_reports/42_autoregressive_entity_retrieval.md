# Autoregressive Entity Retrieval

## 1. 基本信息
- 序号: 42
- 研究主线: Scholarly Identity
- 年份: 2020
- 会议/来源: EMNLP 2020
- 优先级: P1
- 作者: Nicola De Cao, Gautier Izacard, Sebastian Riedel, Fabio Petroni
- PDF: [42_autoregressive_entity_retrieval_2010.00904.pdf](../paper/42_autoregressive_entity_retrieval_2010.00904.pdf)
- 链接: https://arxiv.org/abs/2010.00904

## 2. 研究问题与贡献
- 问题焦点: Autoregressive entity retrieval for improved entity linking precision.
- 摘要要点: Entities are at the center of how we represent and aggregate knowledge. For instance, Encyclopedias such as Wikipedia are structured by entities (e.g., one per Wikipedia article). The ability to retrieve such entities given a query is fundamental for knowledge-intensive tasks such as entity linking and open-domain question answering. Current approaches can be understood as classifiers among atomic labels, one for each entity. Their weight vectors are dense entity representations produced by encoding entity meta information such as their descriptions. This approach has several shortcomings: (i) context and entity affinity is mainly captured through a vector dot product, potentially missing fine-grained interactions; (ii) a large memory footprint is needed to store dense representations when considering large entity sets; (iii) an appropriately hard set of negative data has to be subsampled at training time. In this work, we propose GENRE, the first system that retrieves entities by generating their unique names, left to right, token-by-token in an autoregressive fashion. This mitigates the aforementioned technical issues since: (i) the autoregressive formulation directly captures relations between context and entity name, effectively cross encoding both; (ii) the memory footprint is greatly reduced because the parameters of our encoder-decoder architecture scale with vocabulary ...

## 3. 与 deepsearch 的融合点
- Build entity candidate store and apply dense entity linking across sources.
- 建议代码锚点:
  - `google_scholar_url/google_account_fetcher_pipeline.py`
  - `google_scholar_url/fetch_author_google_scholar_account.py`
  - `google_scholar_url/verify_author_info.py`
  - `google_scholar_url/name_normalization.py`

## 4. MVP 落地计划
- 建立固定离线评测集（匹配准确率、检索召回率、证据覆盖率）。
- 以影子模式启用策略，对比质量/成本/时延并保留回滚开关。
- 沉淀低置信样本，持续迭代数据与规则。

## 5. 预期收益与风险
- 预期收益: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- 主要风险: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. 产物路径
- 报告路径: `paper_reports\42_autoregressive_entity_retrieval.md`
- PDF 路径: `paper\42_autoregressive_entity_retrieval_2010.00904.pdf`
