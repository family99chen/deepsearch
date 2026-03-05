# Low-resource Deep Entity Resolution with Transfer and Active Learning

## 1. 基本信息
- 序号: 37
- 研究主线: Scholarly Identity
- 年份: 2019
- 会议/来源: ACL 2019
- 优先级: P1
- 作者: Yuliang Li et al.
- PDF: [37_low_resource_deep_er_low_resource_deep_er.pdf](../paper/37_low_resource_deep_er_low_resource_deep_er.pdf)
- 链接: https://aclanthology.org/P19-1586/

## 2. 研究问题与贡献
- 问题焦点: Low-resource transfer + active learning for entity resolution.
- 摘要要点: This paper presents transfer learning and active learning methods to reduce labeling cost for deep entity resolution in low-resource settings.

## 3. 与 deepsearch 的融合点
- Upgrade fetch_author_google_scholar_account.py from heuristics to neural entity matching.
- Send low-confidence matches to human review queue and feed back labels.
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
- 报告路径: `paper_reports\37_low_resource_deep_er.md`
- PDF 路径: `paper\37_low_resource_deep_er_low_resource_deep_er.pdf`
