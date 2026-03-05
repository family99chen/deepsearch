# A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models

## 1. 基本信息
- 序号: 32
- 研究主线: Retrieval & RAG
- 年份: 2023
- 会议/来源: arXiv 2023
- 优先级: P1
- 作者: Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, Guido Zuccon
- PDF: [32_setwise_ranking_2310.09497.pdf](../paper/32_setwise_ranking_2310.09497.pdf)
- 链接: https://arxiv.org/abs/2310.09497

## 2. 研究问题与贡献
- 问题焦点: Efficient setwise ranking strategy for LLM reranking.
- 摘要要点: We propose a novel zero-shot document ranking approach based on Large Language Models (LLMs): the Setwise prompting approach. Our approach complements existing prompting approaches for LLM-based zero-shot ranking: Pointwise, Pairwise, and Listwise. Through the first-of-its-kind comparative evaluation within a consistent experimental framework and considering factors like model size, token consumption, latency, among others, we show that existing approaches are inherently characterised by trade-offs between effectiveness and efficiency. We find that while Pointwise approaches score high on efficiency, they suffer from poor effectiveness. Conversely, Pairwise approaches demonstrate superior effectiveness but incur high computational overhead. Our Setwise approach, instead, reduces the number of LLM inferences and the amount of prompt token consumption during the ranking procedure, compared to previous methods. This significantly improves the efficiency of LLM-based zero-shot ranking, while also retaining high zero-shot ranking effectiveness. We make our code and results publicly available at \url{ this https URL }.

## 3. 与 deepsearch 的融合点
- Apply LLM reranking after candidate generation in google_scholar_url/fetch_google_scholar_name_list.py.
- Use setwise/listwise LLM reranking in org_info/google_org_search.py top-k outputs.
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
- 报告路径: `paper_reports\32_setwise_ranking.md`
- PDF 路径: `paper\32_setwise_ranking_2310.09497.pdf`
