# Universal Deep Research: Bring Your Own Model and Strategy

## 1. 基本信息
- 序号: 12
- 研究主线: Agentic Web Research
- 年份: 2025
- 会议/来源: arXiv 2025
- 优先级: P1
- 作者: Peter Belcak, Pavlo Molchanov
- PDF: [12_universal_deep_research_2509.00244.pdf](../paper/12_universal_deep_research_2509.00244.pdf)
- 链接: https://arxiv.org/abs/2509.00244

## 2. 研究问题与贡献
- 问题焦点: Model-agnostic deep research architecture and strategy framework.
- 摘要要点: Deep research tools are among the most impactful and most commonly encountered agentic systems today. We observe, however, that each deep research agent introduced so far is hard-coded to carry out a particular research strategy using a fixed choice of tools. We introduce Universal Deep Research (UDR), a generalist agentic system that wraps around any language model and enables the user to create, edit, and refine their own entirely custom deep research strategies without any need for additional training or finetuning. To showcase the generality of our system, we equip UDR with example minimal, expansive, and intensive research strategies, and provide a user interface to facilitate experimentation with the system.

## 3. 与 deepsearch 的融合点
- Integrate structured Thought-Action-Observation traces in org_info/iteragent_advanced/brain.py.
- Unify action schema and recovery paths in org_info/iteragent_advanced/pageexecuter.py.
- Insert explicit query planning phase before _analyze_sources in pipeline.py.
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
- 报告路径: `paper_reports\12_universal_deep_research.md`
- PDF 路径: `paper\12_universal_deep_research_2509.00244.pdf`
