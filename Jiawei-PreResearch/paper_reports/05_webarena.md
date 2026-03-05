# WebArena: A Realistic Web Environment for Building Autonomous Agents

## 1. 基本信息
- 序号: 05
- 研究主线: Agentic Web Research
- 年份: 2023
- 会议/来源: NeurIPS Datasets/Benchmarks 2024
- 优先级: P0
- 作者: Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig
- PDF: [05_webarena_2307.13854.pdf](../paper/05_webarena_2307.13854.pdf)
- 链接: https://arxiv.org/abs/2307.13854

## 2. 研究问题与贡献
- 问题焦点: Realistic web environments for end-to-end autonomous agents.
- 摘要要点: With advances in generative AI, there is now potential for autonomous agents to manage daily tasks via natural language commands. However, current agents are primarily created and tested in simplified synthetic environments, leading to a disconnect with real-world scenarios. In this paper, we build an environment for language-guided agents that is highly realistic and reproducible. Specifically, we focus on agents that perform tasks on the web, and create an environment with fully functional websites from four common domains: e-commerce, social forum discussions, collaborative software development, and content management. Our environment is enriched with tools (e.g., a map) and external knowledge bases (e.g., user manuals) to encourage human-like task-solving. Building upon our environment, we release a set of benchmark tasks focusing on evaluating the functional correctness of task completions. The tasks in our benchmark are diverse, long-horizon, and designed to emulate tasks that humans routinely perform on the internet. We experiment with several baseline agents, integrating recent techniques such as reasoning before acting. The results demonstrate that solving complex tasks is challenging: our best GPT-4-based agent only achieves an end-to-end task success rate of 14.41%, significantly lower than the human performance of 78.24%. These results highlight the need for further...

## 3. 与 deepsearch 的融合点
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
- 报告路径: `paper_reports\05_webarena.md`
- PDF 路径: `paper\05_webarena_2307.13854.pdf`
