# DeepMatcher: A Deep Learning Approach to Entity Matching

## 1. 基本信息
- 序号: 36
- 研究主线: Scholarly Identity
- 年份: 2018
- 会议/来源: SIGMOD 2018
- 优先级: P0
- 作者: Sidharth Mudgal et al.
- PDF: [36_deepmatcher_deepmatcher.pdf](../paper/36_deepmatcher_deepmatcher.pdf)
- 链接: http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf

## 2. 研究问题与贡献
- 问题焦点: Classic deep entity matching architecture and feature interactions.
- 摘要要点: DeepMatcher studies deep learning model variants for entity matching and proposes practical architecture choices for robust record linkage.

## 3. 与 deepsearch 的融合点
- Upgrade fetch_author_google_scholar_account.py from heuristics to neural entity matching.
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
- 报告路径: `paper_reports\36_deepmatcher.md`
- PDF 路径: `paper\36_deepmatcher_deepmatcher.pdf`
