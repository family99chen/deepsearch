# WebGPT: Browser-assisted question-answering with human feedback

## 1. Metadata
- Index: 01
- Track: Agentic Web Research
- Year: 2021
- Venue/Source: OpenAI / arXiv 2021
- Priority: P1
- Authors: Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, ...
- PDF: [01_webgpt_2112.09332.pdf](../paper/01_webgpt_2112.09332.pdf)
- Link: https://arxiv.org/abs/2112.09332

## 2. Problem and Contribution
- Problem focus: Browser-grounded QA with human feedback and citation-oriented generation.
- Abstract highlights: We fine-tune GPT-3 to answer long-form questions using a text-based web-browsing environment, which allows the model to search and navigate the web. By setting up the task so that it can be performed by humans, we are able to train models on the task using imitation learning, and then optimize answer quality with human feedback. To make human evaluation of factual accuracy easier, models must collect references while browsing in support of their answers. We train and evaluate our models on ELI5, a dataset of questions asked by Reddit users. Our best model is obtained by fine-tuning GPT-3 using behavior cloning, and then performing rejection sampling against a reward model trained to predict human preferences. This model's answers are preferred by humans 56% of the time to those of our human demonstrators, and 69% of the time to the highest-voted answer from Reddit.

## 3. Integration with deepsearch
- Integrate structured Thought-Action-Observation traces in org_info/iteragent_advanced/brain.py.
- Unify action schema and recovery paths in org_info/iteragent_advanced/pageexecuter.py.
- Bind report conclusions to source URLs in pipeline.py final report generation.
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
- Report path: `paper_reports\01_webgpt.md`
- PDF path: `paper\01_webgpt_2112.09332.pdf`
