# WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning

## 1. Metadata
- Index: 11
- Track: Agentic Web Research
- Year: 2025
- Venue/Source: arXiv 2025
- Priority: P1
- Authors: Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Jialong Wu, Xinyu Wang, Zile Qiao, ...
- PDF: [11_websailor_v2_2509.13305.pdf](../paper/11_websailor_v2_2509.13305.pdf)
- Link: https://arxiv.org/abs/2509.13305

## 2. Problem and Contribution
- Problem focus: Scalable RL and synthetic trajectories for web agents.
- Abstract highlights: Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all open-source agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap.

## 3. Integration with deepsearch
- Leverage logs/ and total_usage/ traces to build preference or policy tuning datasets.
- Add offline benchmark harness for org_info/iter_agent and org_info/iteragent_advanced.
- Track trajectory metrics in utils/pipeline_stats.py (success, steps, retries).
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
- Report path: `paper_reports\11_websailor_v2.md`
- PDF path: `paper\11_websailor_v2_2509.13305.pdf`
