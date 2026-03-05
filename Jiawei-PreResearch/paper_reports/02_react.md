# ReAct: Synergizing Reasoning and Acting in Language Models

## 1. Metadata
- Index: 02
- Track: Agentic Web Research
- Year: 2022
- Venue/Source: ICLR 2023
- Priority: P0
- Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- PDF: [02_react_2210.03629.pdf](../paper/02_react_2210.03629.pdf)
- Link: https://arxiv.org/abs/2210.03629

## 2. Problem and Contribution
- Problem focus: Reason+Act paradigm for multi-step tool interaction.
- Abstract highlights: While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReA...

## 3. Integration with deepsearch
- Integrate structured Thought-Action-Observation traces in org_info/iteragent_advanced/brain.py.
- Unify action schema and recovery paths in org_info/iteragent_advanced/pageexecuter.py.
- Add explicit tool-selection policy for search, navigation, and extraction in iteragent_advanced modules.
- Suggested code anchors:
  - `pipeline.py`
  - `org_info/iteragent_advanced/brain.py`
  - `org_info/iteragent_advanced/pageexecuter.py`
  - `utils/pipeline_stats.py`

## 4. MVP Implementation Plan
- Build a fixed offline evaluation set (matching accuracy, retrieval recall, evidence coverage).
- Enable strategy in shadow mode and compare quality/cost/latency with rollback switch.
- Store low-confidence cases for iterative data and rule improvements.

## 5. Expected Gains and Risks
- Expected gains: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- Main risks: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. Artifact Paths
- Report path: `paper_reports\02_react.md`
- PDF path: `paper\02_react_2210.03629.pdf`
