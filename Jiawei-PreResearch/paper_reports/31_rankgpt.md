# Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents

## 1. Metadata
- Index: 31
- Track: Retrieval & RAG
- Year: 2023
- Venue/Source: arXiv 2023
- Priority: P0
- Authors: Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren
- PDF: [31_rankgpt_2304.09542.pdf](../paper/31_rankgpt_2304.09542.pdf)
- Link: https://arxiv.org/abs/2304.09542

## 2. Problem and Contribution
- Problem focus: LLM-as-reranker paradigm for better final ranking quality.
- Abstract highlights: Large Language Models (LLMs) have demonstrated remarkable zero-shot generalization across various language-related tasks, including search engines. However, existing work utilizes the generative ability of LLMs for Information Retrieval (IR) rather than direct passage ranking. The discrepancy between the pre-training objectives of LLMs and the ranking objective poses another challenge. In this paper, we first investigate generative LLMs such as ChatGPT and GPT-4 for relevance ranking in IR. Surprisingly, our experiments reveal that properly instructed LLMs can deliver competitive, even superior results to state-of-the-art supervised methods on popular IR benchmarks. Furthermore, to address concerns about data contamination of LLMs, we collect a new test set called NovelEval, based on the latest knowledge and aiming to verify the model's ability to rank unknown knowledge. Finally, to improve efficiency in real-world applications, we delve into the potential for distilling the ranking capabilities of ChatGPT into small specialized models using a permutation distillation scheme. Our evaluation results turn out that a distilled 440M model outperforms a 3B supervised model on the BEIR benchmark. The code to reproduce our results is available at this http URL .

## 3. Integration with deepsearch
- Apply LLM reranking after candidate generation in google_scholar_url/fetch_google_scholar_name_list.py.
- Use setwise/listwise LLM reranking in org_info/google_org_search.py top-k outputs.
- Suggested code anchors:
  - `google_search_api/google_search.py`
  - `org_info/google_org_search.py`
  - `org_info/arbitrary_search.py`
  - `pipeline.py`
  - `localdb/insert_mongo.py`

## 4. MVP Implementation Plan
- Build a fixed offline evaluation set (matching accuracy, retrieval recall, evidence coverage).
- Enable strategy in shadow mode and compare quality/cost/latency with rollback switch.
- Store low-confidence cases for iterative data and rule improvements.

## 5. Expected Gains and Risks
- Expected gains: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- Main risks: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. Artifact Paths
- Report path: `paper_reports\31_rankgpt.md`
- PDF path: `paper\31_rankgpt_2304.09542.pdf`
