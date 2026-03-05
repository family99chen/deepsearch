# Universal Deep Research: Bring Your Own Model and Strategy

## 1. Metadata
- Index: 12
- Track: Agentic Web Research
- Year: 2025
- Venue/Source: arXiv 2025
- Priority: P1
- Authors: Peter Belcak, Pavlo Molchanov
- PDF: [12_universal_deep_research_2509.00244.pdf](../paper/12_universal_deep_research_2509.00244.pdf)
- Link: https://arxiv.org/abs/2509.00244

## 2. Problem and Contribution
- Problem focus: Model-agnostic deep research architecture and strategy framework.
- Abstract highlights: Deep research tools are among the most impactful and most commonly encountered agentic systems today. We observe, however, that each deep research agent introduced so far is hard-coded to carry out a particular research strategy using a fixed choice of tools. We introduce Universal Deep Research (UDR), a generalist agentic system that wraps around any language model and enables the user to create, edit, and refine their own entirely custom deep research strategies without any need for additional training or finetuning. To showcase the generality of our system, we equip UDR with example minimal, expansive, and intensive research strategies, and provide a user interface to facilitate experimentation with the system.

## 3. Integration with deepsearch
- Integrate structured Thought-Action-Observation traces in org_info/iteragent_advanced/brain.py.
- Unify action schema and recovery paths in org_info/iteragent_advanced/pageexecuter.py.
- Insert explicit query planning phase before _analyze_sources in pipeline.py.
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
- Report path: `paper_reports\12_universal_deep_research.md`
- PDF path: `paper\12_universal_deep_research_2509.00244.pdf`
