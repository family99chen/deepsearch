# Mind2Web: Towards a Generalist Agent for the Web

## 1. Metadata
- Index: 04
- Track: Agentic Web Research
- Year: 2023
- Venue/Source: NeurIPS 2023
- Priority: P1
- Authors: Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, Yu Su
- PDF: [04_mind2web_2306.06070.pdf](../paper/04_mind2web_2306.06070.pdf)
- Link: https://arxiv.org/abs/2306.06070

## 2. Problem and Contribution
- Problem focus: Large-scale benchmark for real-world web agent grounding.
- Abstract highlights: We introduce Mind2Web, the first dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website. Existing datasets for web agents either use simulated websites or only cover a limited set of websites and tasks, thus not suitable for generalist web agents. With over 2,000 open-ended tasks collected from 137 websites spanning 31 domains and crowdsourced action sequences for the tasks, Mind2Web provides three necessary ingredients for building generalist web agents: 1) diverse domains, websites, and tasks, 2) use of real-world websites instead of simulated and simplified ones, and 3) a broad spectrum of user interaction patterns. Based on Mind2Web, we conduct an initial exploration of using large language models (LLMs) for building generalist web agents. While the raw HTML of real-world websites are often too large to be fed to LLMs, we show that first filtering it with a small LM significantly improves the effectiveness and efficiency of LLMs. Our solution demonstrates a decent level of performance, even on websites or entire domains the model has never seen before, but there is still a substantial room to improve towards truly generalizable agents. We open-source our dataset, model implementation, and trained models ( this https URL ) to facilitate further research on building a generalist agent ...

## 3. Integration with deepsearch
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
- Report path: `paper_reports\04_mind2web.md`
- PDF path: `paper\04_mind2web_2306.06070.pdf`
