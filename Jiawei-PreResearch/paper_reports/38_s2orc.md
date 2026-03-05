# S2ORC: The Semantic Scholar Open Research Corpus

## 1. 基本信息
- 序号: 38
- 研究主线: Scholarly Identity
- 年份: 2019
- 会议/来源: ACL 2020
- 优先级: P1
- 作者: Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, Dan S. Weld
- PDF: [38_s2orc_1911.02782.pdf](../paper/38_s2orc_1911.02782.pdf)
- 链接: https://arxiv.org/abs/1911.02782

## 2. 研究问题与贡献
- 问题焦点: Large-scale scholarly corpus for scientific NLP and matching.
- 摘要要点: We introduce S2ORC, a large corpus of 81.1M English-language academic papers spanning many academic disciplines. The corpus consists of rich metadata, paper abstracts, resolved bibliographic references, as well as structured full text for 8.1M open access papers. Full text is annotated with automatically-detected inline mentions of citations, figures, and tables, each linked to their corresponding paper objects. In S2ORC, we aggregate papers from hundreds of academic publishers and digital archives into a unified source, and create the largest publicly-available collection of machine-readable academic text to date. We hope this resource will facilitate research and development of tools and tasks for text mining over academic text.

## 3. 与 deepsearch 的融合点
- Use S2ORC-derived lexical priors for title/abstract matching robustness.
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
- 报告路径: `paper_reports\38_s2orc.md`
- PDF 路径: `paper\38_s2orc_1911.02782.pdf`
