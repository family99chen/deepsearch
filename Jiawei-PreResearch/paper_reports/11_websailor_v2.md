# WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning

## 1. 基本信息
- 序号: 11
- 研究主线: Agentic Web Research
- 年份: 2025
- 会议/来源: arXiv 2025
- 优先级: P1
- 作者: Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Jialong Wu, Xinyu Wang, Zile Qiao, ...
- PDF: [11_websailor_v2_2509.13305.pdf](../paper/11_websailor_v2_2509.13305.pdf)
- 链接: https://arxiv.org/abs/2509.13305

## 2. 研究问题与贡献
- 问题焦点: Scalable RL and synthetic trajectories for web agents.
- 摘要要点: Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all open-source agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap.

## 3. 与 deepsearch 的融合点
- Leverage logs/ and total_usage/ traces to build preference or policy tuning datasets.
- Add offline benchmark harness for org_info/iter_agent and org_info/iteragent_advanced.
- Track trajectory metrics in utils/pipeline_stats.py (success, steps, retries).
- 建议代码锚点:
  - `pipeline.py`
  - `org_info/iteragent_advanced/brain.py`
  - `org_info/iteragent_advanced/pageexecuter.py`
  - `utils/pipeline_stats.py`

## 4. MVP 落地计划
- 建立固定离线评测集（匹配准确率、检索召回率、证据覆盖率）。
- 以影子模式启用策略，对比质量/成本/时延并保留回滚开关。
- 沉淀低置信样本，持续迭代数据与规则。

## 5. 预期收益与风险
- 预期收益: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- 主要风险: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. 产物路径
- 报告路径: `paper_reports\11_websailor_v2.md`
- PDF 路径: `paper\11_websailor_v2_2509.13305.pdf`
