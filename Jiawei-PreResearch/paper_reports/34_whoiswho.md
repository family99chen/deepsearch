# Web-Scale Academic Name Disambiguation: the WhoIsWho Benchmark, Leaderboard, and Toolkit

## 1. Metadata
- Index: 34
- Track: Scholarly Identity
- Year: 2023
- Venue/Source: KDD 2023
- Priority: P0
- Authors: Bo Chen, Jing Zhang, Fanjin Zhang, Tianyi Han, Yuqing Cheng, Xiaoyan Li, Yuxiao Dong, Jie Tang
- PDF: [34_whoiswho_2302.11848.pdf](../paper/34_whoiswho_2302.11848.pdf)
- Link: https://arxiv.org/abs/2302.11848

## 2. Problem and Contribution
- Problem focus: Web-scale academic name disambiguation benchmark/toolkit.
- Abstract highlights: Name disambiguation -- a fundamental problem in online academic systems -- is now facing greater challenges with the increasing growth of research papers. For example, on AMiner, an online academic search platform, about 10% of names own more than 100 authors. Such real-world challenging cases have not been effectively addressed by existing researches due to the small-scale or low-quality datasets that they have used. The development of effective algorithms is further hampered by a variety of tasks and evaluation protocols designed on top of diverse datasets. To this end, we present WhoIsWho owning, a large-scale benchmark with over 1,000,000 papers built using an interactive annotation process, a regular leaderboard with comprehensive tasks, and an easy-to-use toolkit encapsulating the entire pipeline as well as the most powerful features and baseline models for tackling the tasks. Our developed strong baseline has already been deployed online in the AMiner system to enable daily arXiv paper assignments. The public leaderboard is available at this http URL . The toolkit is at this https URL . The online demo of daily arXiv paper assignments is at this https URL .

## 3. Integration with deepsearch
- Replace rule-only ORCID->Scholar matching with learned scorer in google_account_fetcher_pipeline.py.
- Expand pairwise features in google_scholar_url/verify_author_info.py.
- Create offline labeled set for ORCID->Scholar regression tests.
- Suggested code anchors:
  - `google_scholar_url/google_account_fetcher_pipeline.py`
  - `google_scholar_url/fetch_author_google_scholar_account.py`
  - `google_scholar_url/verify_author_info.py`
  - `google_scholar_url/name_normalization.py`

## 4. MVP Implementation Plan
- Build a fixed offline evaluation set (matching accuracy, retrieval recall, evidence coverage).
- Enable strategy in shadow mode and compare quality/cost/latency with rollback switch.
- Store low-confidence cases for iterative data and rule improvements.

## 5. Expected Gains and Risks
- Expected gains: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- Main risks: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. Artifact Paths
- Report path: `paper_reports\34_whoiswho.md`
- PDF path: `paper\34_whoiswho_2302.11848.pdf`
