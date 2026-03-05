# S2AND: A Benchmark and Evaluation System for Author Name Disambiguation

## 1. 基本信息
- 序号: 33
- 研究主线: Scholarly Identity
- 年份: 2021
- 会议/来源: JCDL 2021
- 优先级: P0
- 作者: Shivashankar Subramanian, Daniel King, Doug Downey, Sergey Feldman
- PDF: [33_s2and_2103.07534.pdf](../paper/33_s2and_2103.07534.pdf)
- 链接: https://arxiv.org/abs/2103.07534

## 2. 研究问题与贡献
- 问题焦点: Benchmark and system design for author name disambiguation.
- 摘要要点: Author Name Disambiguation (AND) is the task of resolving which author mentions in a bibliographic database refer to the same real-world person, and is a critical ingredient of digital library applications such as search and citation analysis. While many AND algorithms have been proposed, comparing them is difficult because they often employ distinct features and are evaluated on different datasets. In response to this challenge, we present S2AND, a unified benchmark dataset for AND on scholarly papers, as well as an open-source reference model implementation. Our dataset harmonizes eight disparate AND datasets into a uniform format, with a single rich feature set drawn from the Semantic Scholar (S2) database. Our evaluation suite for S2AND reports performance split by facets like publication year and number of papers, allowing researchers to track both global performance and measures of fairness across facet values. Our experiments show that because previous datasets tend to cover idiosyncratic and biased slices of the literature, algorithms trained to perform well on one on them may generalize poorly to others. By contrast, we show how training on a union of datasets in S2AND results in more robust models that perform well even on datasets unseen in training. The resulting AND model also substantially improves over the production algorithm in S2, reducing error by over 50% in...

## 3. 与 deepsearch 的融合点
- Replace rule-only ORCID->Scholar matching with learned scorer in google_account_fetcher_pipeline.py.
- Expand pairwise features in google_scholar_url/verify_author_info.py.
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
- 报告路径: `paper_reports\33_s2and.md`
- PDF 路径: `paper\33_s2and_2103.07534.pdf`
