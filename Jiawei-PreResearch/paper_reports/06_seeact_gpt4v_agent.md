# GPT-4V(ision) is a Generalist Web Agent, if Grounded

## 1. Metadata
- Index: 06
- Track: Agentic Web Research
- Year: 2024
- Venue/Source: arXiv 2024
- Priority: P1
- Authors: Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, Yu Su
- PDF: [06_seeact_gpt4v_agent_2401.01614.pdf](../paper/06_seeact_gpt4v_agent_2401.01614.pdf)
- Link: https://arxiv.org/abs/2401.01614

## 2. Problem and Contribution
- Problem focus: Grounded visual-web agent design for robust page operations.
- Abstract highlights: The recent development on large multimodal models (LMMs), especially GPT-4V(ision) and Gemini, has been quickly expanding the capability boundaries of multimodal models beyond traditional tasks like image captioning and visual question answering. In this work, we explore the potential of LMMs like GPT-4V as a generalist web agent that can follow natural language instructions to complete tasks on any given website. We propose SEEACT, a generalist web agent that harnesses the power of LMMs for integrated visual understanding and acting on the web. We evaluate on the recent MIND2WEB benchmark. In addition to standard offline evaluation on cached websites, we enable a new online evaluation setting by developing a tool that allows running web agents on live websites. We show that GPT-4V presents a great potential for web agents -- it can successfully complete 51.1 of the tasks on live websites if we manually ground its textual plans into actions on the websites. This substantially outperforms text-only LLMs like GPT-4 or smaller models (FLAN-T5 and BLIP-2) specifically fine-tuned for web agents. However, grounding still remains a major challenge. Existing LMM grounding strategies like set-of-mark prompting turns out to be not effective for web agents, and the best grounding strategy we develop in this paper leverages both the HTML structure and visuals. Yet, there is still a substan...

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
- Report path: `paper_reports\06_seeact_gpt4v_agent.md`
- PDF path: `paper\06_seeact_gpt4v_agent_2401.01614.pdf`
