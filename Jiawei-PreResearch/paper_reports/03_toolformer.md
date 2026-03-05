# Toolformer: Language Models Can Teach Themselves to Use Tools

## 1. Metadata
- Index: 03
- Track: Agentic Web Research
- Year: 2023
- Venue/Source: NeurIPS 2023
- Priority: P1
- Authors: Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- PDF: [03_toolformer_2302.04761.pdf](../paper/03_toolformer_2302.04761.pdf)
- Link: https://arxiv.org/abs/2302.04761

## 2. Problem and Contribution
- Problem focus: Self-supervised tool-use data generation for LLMs.
- Abstract highlights: Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.

## 3. Integration with deepsearch
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
- Report path: `paper_reports\03_toolformer.md`
- PDF path: `paper\03_toolformer_2302.04761.pdf`
