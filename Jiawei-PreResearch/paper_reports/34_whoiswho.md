# Web-Scale Academic Name Disambiguation: the WhoIsWho Benchmark, Leaderboard, and Toolkit

## 1. 基本信息
- 序号: 34
- 研究主线: Scholarly Identity
- 年份: 2023
- 会议/来源: KDD 2023
- 优先级: P0
- 作者: Bo Chen, Jing Zhang, Fanjin Zhang, Tianyi Han, Yuqing Cheng, Xiaoyan Li, Yuxiao Dong, Jie Tang
- PDF: [34_whoiswho_2302.11848.pdf](../paper/34_whoiswho_2302.11848.pdf)
- 链接: https://arxiv.org/abs/2302.11848

## 2. 研究问题与贡献
- 问题焦点: Web-scale academic name disambiguation benchmark/toolkit.
- 摘要要点: Name disambiguation -- a fundamental problem in online academic systems -- is now facing greater challenges with the increasing growth of research papers. For example, on AMiner, an online academic search platform, about 10% of names own more than 100 authors. Such real-world challenging cases have not been effectively addressed by existing researches due to the small-scale or low-quality datasets that they have used. The development of effective algorithms is further hampered by a variety of tasks and evaluation protocols designed on top of diverse datasets. To this end, we present WhoIsWho owning, a large-scale benchmark with over 1,000,000 papers built using an interactive annotation process, a regular leaderboard with comprehensive tasks, and an easy-to-use toolkit encapsulating the entire pipeline as well as the most powerful features and baseline models for tackling the tasks. Our developed strong baseline has already been deployed online in the AMiner system to enable daily arXiv paper assignments. The public leaderboard is available at this http URL . The toolkit is at this https URL . The online demo of daily arXiv paper assignments is at this https URL .

## 3. 与 deepsearch 的融合点
- Replace rule-only ORCID->Scholar matching with learned scorer in google_account_fetcher_pipeline.py.
- Expand pairwise features in google_scholar_url/verify_author_info.py.
- Create offline labeled set for ORCID->Scholar regression tests.
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
- 报告路径: `paper_reports\34_whoiswho.md`
- PDF 路径: `paper\34_whoiswho_2302.11848.pdf`
