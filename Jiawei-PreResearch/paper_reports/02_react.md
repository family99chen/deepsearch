# ReAct: Synergizing Reasoning and Acting in Language Models

## 1. 基本信息
- 序号: 02
- 研究主线: Agentic Web Research
- 年份: 2022
- 会议/来源: ICLR 2023
- 优先级: P0
- 作者: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- PDF: [02_react_2210.03629.pdf](../paper/02_react_2210.03629.pdf)
- 链接: https://arxiv.org/abs/2210.03629

## 2. 研究问题与贡献
- 问题焦点: Reason+Act paradigm for multi-step tool interaction.
- 摘要要点: While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReA...

## 3. 与 deepsearch 的融合点
- Integrate structured Thought-Action-Observation traces in org_info/iteragent_advanced/brain.py.
- Unify action schema and recovery paths in org_info/iteragent_advanced/pageexecuter.py.
- Add explicit tool-selection policy for search, navigation, and extraction in iteragent_advanced modules.
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
- 报告路径: `paper_reports\02_react.md`
- PDF 路径: `paper\02_react_2210.03629.pdf`
