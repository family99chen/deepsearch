# MTEB: Massive Text Embedding Benchmark

## 1. 基本信息
- 序号: 28
- 研究主线: Retrieval & RAG
- 年份: 2022
- 会议/来源: NeurIPS Datasets 2022
- 优先级: P1
- 作者: Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers
- PDF: [28_mteb_2210.07316.pdf](../paper/28_mteb_2210.07316.pdf)
- 链接: https://arxiv.org/abs/2210.07316

## 2. 研究问题与贡献
- 问题焦点: Massive embedding benchmark for model selection.
- 摘要要点: Text embeddings are commonly evaluated on a small set of datasets from a single task not covering their possible applications to other tasks. It is unclear whether state-of-the-art embeddings on semantic textual similarity (STS) can be equally well applied to other tasks like clustering or reranking. This makes progress in the field difficult to track, as various models are constantly being proposed without proper evaluation. To solve this problem, we introduce the Massive Text Embedding Benchmark (MTEB). MTEB spans 8 embedding tasks covering a total of 58 datasets and 112 languages. Through the benchmarking of 33 models on MTEB, we establish the most comprehensive benchmark of text embeddings to date. We find that no particular text embedding method dominates across all tasks. This suggests that the field has yet to converge on a universal text embedding method and scale it up sufficiently to provide state-of-the-art results on all embedding tasks. MTEB comes with open-source code and a public leaderboard at this https URL .

## 3. 与 deepsearch 的融合点
- Track retrieval metrics (Recall@K, NDCG@K, evidence precision) in utils/pipeline_stats.py.
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
- 报告路径: `paper_reports\28_mteb.md`
- PDF 路径: `paper\28_mteb_2210.07316.pdf`
