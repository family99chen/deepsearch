# DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments

## 1. Metadata
- Index: 10
- Track: Agentic Web Research
- Year: 2025
- Venue/Source: EMNLP 2025 (paper + arXiv)
- Priority: P0
- Authors: Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, Pengfei Liu
- PDF: [10_deepresearcher_2504.03160.pdf](../paper/10_deepresearcher_2504.03160.pdf)
- Link: https://arxiv.org/abs/2504.03160

## 2. Problem and Contribution
- Problem focus: Deep research agent trained on real/simulated search environments.
- Abstract highlights: Large Language Models (LLMs) equipped with web search capabilities have demonstrated impressive potential for deep research tasks. However, current approaches predominantly rely on either manually engineered prompts (prompt engineering-based) with brittle performance or reinforcement learning within controlled Retrieval-Augmented Generation (RAG) environments (RAG-based) that fail to capture the complexities of real-world interaction. In this paper, we introduce DeepResearcher, the first comprehensive framework for end-to-end training of LLM-based deep research agents through scaling reinforcement learning (RL) in real-world environments with authentic web search interactions. Unlike RAG-based approaches that assume all necessary information exists within a fixed corpus, our method trains agents to navigate the noisy, unstructured, and dynamic nature of the open web. We implement a specialized multi-agent architecture where browsing agents extract relevant information from various webpage structures and overcoming significant technical challenges. Extensive experiments on open-domain research tasks demonstrate that DeepResearcher achieves substantial improvements of up to 28.9 points over prompt engineering-based baselines and up to 7.2 points over RAG-based RL agents. Our qualitative analysis reveals emergent cognitive behaviors from end-to-end RL training, including the abili...

## 3. Integration with deepsearch
- Leverage logs/ and total_usage/ traces to build preference or policy tuning datasets.
- Integrate structured Thought-Action-Observation traces in org_info/iteragent_advanced/brain.py.
- Unify action schema and recovery paths in org_info/iteragent_advanced/pageexecuter.py.
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
- Report path: `paper_reports\10_deepresearcher.md`
- PDF path: `paper\10_deepresearcher_2504.03160.pdf`
