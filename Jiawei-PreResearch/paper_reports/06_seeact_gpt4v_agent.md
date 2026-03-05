# GPT-4V(ision) is a Generalist Web Agent, if Grounded

## 1. 基本信息
- 序号: 06
- 研究主线: Agentic Web Research
- 年份: 2024
- 会议/来源: arXiv 2024
- 优先级: P1
- 作者: Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, Yu Su
- PDF: [06_seeact_gpt4v_agent_2401.01614.pdf](../paper/06_seeact_gpt4v_agent_2401.01614.pdf)
- 链接: https://arxiv.org/abs/2401.01614

## 2. 研究问题与贡献
- 问题焦点: Grounded visual-web agent design for robust page operations.
- 摘要要点: The recent development on large multimodal models (LMMs), especially GPT-4V(ision) and Gemini, has been quickly expanding the capability boundaries of multimodal models beyond traditional tasks like image captioning and visual question answering. In this work, we explore the potential of LMMs like GPT-4V as a generalist web agent that can follow natural language instructions to complete tasks on any given website. We propose SEEACT, a generalist web agent that harnesses the power of LMMs for integrated visual understanding and acting on the web. We evaluate on the recent MIND2WEB benchmark. In addition to standard offline evaluation on cached websites, we enable a new online evaluation setting by developing a tool that allows running web agents on live websites. We show that GPT-4V presents a great potential for web agents -- it can successfully complete 51.1 of the tasks on live websites if we manually ground its textual plans into actions on the websites. This substantially outperforms text-only LLMs like GPT-4 or smaller models (FLAN-T5 and BLIP-2) specifically fine-tuned for web agents. However, grounding still remains a major challenge. Existing LMM grounding strategies like set-of-mark prompting turns out to be not effective for web agents, and the best grounding strategy we develop in this paper leverages both the HTML structure and visuals. Yet, there is still a substan...

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
- 报告路径: `paper_reports\06_seeact_gpt4v_agent.md`
- PDF 路径: `paper\06_seeact_gpt4v_agent_2401.01614.pdf`
