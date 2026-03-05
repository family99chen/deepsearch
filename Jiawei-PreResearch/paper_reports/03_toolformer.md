# Toolformer: Language Models Can Teach Themselves to Use Tools

## 1. 基本信息
- 序号: 03
- 研究主线: Agentic Web Research
- 年份: 2023
- 会议/来源: NeurIPS 2023
- 优先级: P1
- 作者: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- PDF: [03_toolformer_2302.04761.pdf](../paper/03_toolformer_2302.04761.pdf)
- 链接: https://arxiv.org/abs/2302.04761

## 2. 研究问题与贡献
- 问题焦点: Self-supervised tool-use data generation for LLMs.
- 摘要要点: Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.

## 3. 与 deepsearch 的融合点
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
- 报告路径: `paper_reports\03_toolformer.md`
- PDF 路径: `paper\03_toolformer_2302.04761.pdf`
