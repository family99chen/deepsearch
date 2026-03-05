# BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models

## 1. 基本信息
- 序号: 20
- 研究主线: Retrieval & RAG
- 年份: 2021
- 会议/来源: NeurIPS Benchmarks 2021
- 优先级: P0
- 作者: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych
- PDF: [20_beir_2104.08663.pdf](../paper/20_beir_2104.08663.pdf)
- 链接: https://arxiv.org/abs/2104.08663

## 2. 研究问题与贡献
- 问题焦点: Zero-shot retrieval benchmark for realistic cross-domain evaluation.
- 摘要要点: Existing neural information retrieval (IR) models have often been studied in homogeneous and narrow settings, which has considerably limited insights into their out-of-distribution (OOD) generalization capabilities. To address this, and to facilitate researchers to broadly evaluate the effectiveness of their models, we introduce Benchmarking-IR (BEIR), a robust and heterogeneous evaluation benchmark for information retrieval. We leverage a careful selection of 18 publicly available datasets from diverse text retrieval tasks and domains and evaluate 10 state-of-the-art retrieval systems including lexical, sparse, dense, late-interaction and re-ranking architectures on the BEIR benchmark. Our results show BM25 is a robust baseline and re-ranking and late-interaction-based models on average achieve the best zero-shot performances, however, at high computational costs. In contrast, dense and sparse-retrieval models are computationally more efficient but often underperform other approaches, highlighting the considerable room for improvement in their generalization capabilities. We hope this framework allows us to better evaluate and understand existing retrieval systems, and contributes to accelerating progress towards better robust and generalizable systems in the future. BEIR is publicly available at this https URL .

## 3. 与 deepsearch 的融合点
- Track retrieval metrics (Recall@K, NDCG@K, evidence precision) in utils/pipeline_stats.py.
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
- 报告路径: `paper_reports\20_beir.md`
- PDF 路径: `paper\20_beir_2104.08663.pdf`
