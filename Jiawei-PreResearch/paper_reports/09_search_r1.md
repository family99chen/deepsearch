# Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

## 1. 基本信息
- 序号: 09
- 研究主线: Agentic Web Research
- 年份: 2025
- 会议/来源: arXiv 2025
- 优先级: P0
- 作者: Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, Jiawei Han
- PDF: [09_search_r1_2503.09516.pdf](../paper/09_search_r1_2503.09516.pdf)
- 链接: https://arxiv.org/abs/2503.09516

## 2. 研究问题与贡献
- 问题焦点: RL training recipe for search-enabled reasoning models.
- 摘要要点: Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at this https URL .

## 3. 与 deepsearch 的融合点
- Leverage logs/ and total_usage/ traces to build preference or policy tuning datasets.
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
- 报告路径: `paper_reports\09_search_r1.md`
- PDF 路径: `paper\09_search_r1_2503.09516.pdf`
