# DeepSearch 论文预研总报告

- 生成时间: 2026-03-06 01:05:29
- 论文总数: 42
- 成功下载: 42
- 下载失败: 0

## 1. 预研范围
- 智能体网页深研: 聚焦网页浏览智能体的规划、执行与评估。
- 检索与 RAG: 聚焦召回、重排、检索校正与证据化生成。
- 学者身份识别与实体匹配: 聚焦 ORCID/Scholar 映射与同名消歧。

## 2. 与当前项目的融合方向
- 将 ORCID -> Google Scholar 映射从规则阈值升级为学习式消歧打分。
- 构建多阶段检索管线: 关键词召回 + 语义召回 + LLM 重排。
- 在 iteragent_advanced 中增强规划能力、失败恢复与可观测性。
- 在 pipeline 中引入证据绑定、冲突处理、低置信重检索。
- 在 pipeline_stats 中补齐质量/成本/时延统计指标。

## 3. 分阶段落地路线
### 阶段 A（1-2 周）
- 在候选结果阶段引入 LLM 重排。
- 增加检索质量门控与重检索回退策略。
- 建立最小离线评测集与统计看板。

### 阶段 B（2-4 周）
- 引入作者消歧特征与学习式匹配器。
- 引入 SciBERT/SPECTER/E5 向量提升语义匹配。
- 建立低置信样本人工复核回流机制。

### 阶段 C（4-8 周）
- 建立网页智能体任务评测体系。
- 基于日志数据做策略蒸馏或强化优化。
- 建立质量/时延/成本 KPI 持续监控。

## 4. P0 优先论文
1. ReAct: Synergizing Reasoning and Acting in Language Models（2022，智能体网页深研）
2. WebArena: A Realistic Web Environment for Building Autonomous Agents（2023，智能体网页深研）
3. The BrowserGym Ecosystem for Web Agent Research（2024，智能体网页深研）
4. Search-o1: Agentic Search-Enhanced Large Reasoning Models（2025，智能体网页深研）
5. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning（2025，智能体网页深研）
6. DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments（2025，智能体网页深研）
7. Dense Passage Retrieval for Open-Domain Question Answering（2020，检索与 RAG）
8. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2020，检索与 RAG）
9. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction（2021，检索与 RAG）
10. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models（2021，检索与 RAG）
11. Text Embeddings by Weakly-Supervised Contrastive Pre-training（2022，检索与 RAG）
12. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection（2023，检索与 RAG）
13. Corrective Retrieval Augmented Generation（2024，检索与 RAG）
14. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents（2023，检索与 RAG）
15. S2AND: A Benchmark and Evaluation System for Author Name Disambiguation（2021，学者身份识别与实体匹配）
16. Web-Scale Academic Name Disambiguation: the WhoIsWho Benchmark, Leaderboard, and Toolkit（2023，学者身份识别与实体匹配）
17. Deep Entity Matching with Pre-Trained Language Models（2020，学者身份识别与实体匹配）
18. DeepMatcher: A Deep Learning Approach to Entity Matching（2018，学者身份识别与实体匹配）

## 5. 全量索引
### 智能体网页深研
- [01] WebGPT: Browser-assisted question-answering with human feedback（2021） | 通过 | [PDF](./paper/01_webgpt_2112.09332.pdf) | [报告](./paper_reports/01_webgpt.md)
- [02] ReAct: Synergizing Reasoning and Acting in Language Models（2022） | 通过 | [PDF](./paper/02_react_2210.03629.pdf) | [报告](./paper_reports/02_react.md)
- [03] Toolformer: Language Models Can Teach Themselves to Use Tools（2023） | 通过 | [PDF](./paper/03_toolformer_2302.04761.pdf) | [报告](./paper_reports/03_toolformer.md)
- [04] Mind2Web: Towards a Generalist Agent for the Web（2023） | 通过 | [PDF](./paper/04_mind2web_2306.06070.pdf) | [报告](./paper_reports/04_mind2web.md)
- [05] WebArena: A Realistic Web Environment for Building Autonomous Agents（2023） | 通过 | [PDF](./paper/05_webarena_2307.13854.pdf) | [报告](./paper_reports/05_webarena.md)
- [06] GPT-4V(ision) is a Generalist Web Agent, if Grounded（2024） | 通过 | [PDF](./paper/06_seeact_gpt4v_agent_2401.01614.pdf) | [报告](./paper_reports/06_seeact_gpt4v_agent.md)
- [07] The BrowserGym Ecosystem for Web Agent Research（2024） | 通过 | [PDF](./paper/07_browsergym_2412.05467.pdf) | [报告](./paper_reports/07_browsergym.md)
- [08] Search-o1: Agentic Search-Enhanced Large Reasoning Models（2025） | 通过 | [PDF](./paper/08_search_o1_2501.05366.pdf) | [报告](./paper_reports/08_search_o1.md)
- [09] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning（2025） | 通过 | [PDF](./paper/09_search_r1_2503.09516.pdf) | [报告](./paper_reports/09_search_r1.md)
- [10] DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments（2025） | 通过 | [PDF](./paper/10_deepresearcher_2504.03160.pdf) | [报告](./paper_reports/10_deepresearcher.md)
- [11] WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning（2025） | 通过 | [PDF](./paper/11_websailor_v2_2509.13305.pdf) | [报告](./paper_reports/11_websailor_v2.md)
- [12] Universal Deep Research: Bring Your Own Model and Strategy（2025） | 通过 | [PDF](./paper/12_universal_deep_research_2509.00244.pdf) | [报告](./paper_reports/12_universal_deep_research.md)

### 检索与 RAG
- [13] REALM: Retrieval-Augmented Language Model Pre-Training（2020） | 通过 | [PDF](./paper/13_realm_2002.08909.pdf) | [报告](./paper_reports/13_realm.md)
- [14] Dense Passage Retrieval for Open-Domain Question Answering（2020） | 通过 | [PDF](./paper/14_dpr_2004.04906.pdf) | [报告](./paper_reports/14_dpr.md)
- [15] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2020） | 通过 | [PDF](./paper/15_rag_2005.11401.pdf) | [报告](./paper_reports/15_rag.md)
- [16] Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering（2020） | 通过 | [PDF](./paper/16_fid_2007.01282.pdf) | [报告](./paper_reports/16_fid.md)
- [17] ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（2020） | 通过 | [PDF](./paper/17_colbert_2004.12832.pdf) | [报告](./paper_reports/17_colbert.md)
- [18] ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction（2021） | 通过 | [PDF](./paper/18_colbertv2_2112.01488.pdf) | [报告](./paper_reports/18_colbertv2.md)
- [19] Unsupervised Dense Information Retrieval with Contrastive Learning（2021） | 通过 | [PDF](./paper/19_contriever_2112.09118.pdf) | [报告](./paper_reports/19_contriever.md)
- [20] BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models（2021） | 通过 | [PDF](./paper/20_beir_2104.08663.pdf) | [报告](./paper_reports/20_beir.md)
- [21] SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval（2021） | 通过 | [PDF](./paper/21_splade_v2_2109.10086.pdf) | [报告](./paper_reports/21_splade_v2.md)
- [22] Text Embeddings by Weakly-Supervised Contrastive Pre-training（2022） | 通过 | [PDF](./paper/22_e5_2212.03533.pdf) | [报告](./paper_reports/22_e5.md)
- [23] Precise Zero-Shot Dense Retrieval without Relevance Labels（2022） | 通过 | [PDF](./paper/23_hyde_2212.10496.pdf) | [报告](./paper_reports/23_hyde.md)
- [24] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection（2023） | 通过 | [PDF](./paper/24_self_rag_2310.11511.pdf) | [报告](./paper_reports/24_self_rag.md)
- [25] Corrective Retrieval Augmented Generation（2024） | 通过 | [PDF](./paper/25_crag_2401.15884.pdf) | [报告](./paper_reports/25_crag.md)
- [26] From Local to Global: A Graph RAG Approach to Query-Focused Summarization（2024） | 通过 | [PDF](./paper/26_graphrag_2404.16130.pdf) | [报告](./paper_reports/26_graphrag.md)
- [27] One Embedder, Any Task: Instruction-Finetuned Text Embeddings（2022） | 通过 | [PDF](./paper/27_instructor_2212.09741.pdf) | [报告](./paper_reports/27_instructor.md)
- [28] MTEB: Massive Text Embedding Benchmark（2022） | 通过 | [PDF](./paper/28_mteb_2210.07316.pdf) | [报告](./paper_reports/28_mteb.md)
- [29] Towards General Text Embeddings with Multi-stage Contrastive Learning（2023） | 通过 | [PDF](./paper/29_bge_multistage_2308.03281.pdf) | [报告](./paper_reports/29_bge_multistage.md)
- [30] M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation（2024） | 通过 | [PDF](./paper/30_m3_embedding_2402.03216.pdf) | [报告](./paper_reports/30_m3_embedding.md)
- [31] Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents（2023） | 通过 | [PDF](./paper/31_rankgpt_2304.09542.pdf) | [报告](./paper_reports/31_rankgpt.md)
- [32] A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models（2023） | 通过 | [PDF](./paper/32_setwise_ranking_2310.09497.pdf) | [报告](./paper_reports/32_setwise_ranking.md)

### 学者身份识别与实体匹配
- [33] S2AND: A Benchmark and Evaluation System for Author Name Disambiguation（2021） | 通过 | [PDF](./paper/33_s2and_2103.07534.pdf) | [报告](./paper_reports/33_s2and.md)
- [34] Web-Scale Academic Name Disambiguation: the WhoIsWho Benchmark, Leaderboard, and Toolkit（2023） | 通过 | [PDF](./paper/34_whoiswho_2302.11848.pdf) | [报告](./paper_reports/34_whoiswho.md)
- [35] Deep Entity Matching with Pre-Trained Language Models（2020） | 通过 | [PDF](./paper/35_ditto_2004.00584.pdf) | [报告](./paper_reports/35_ditto.md)
- [36] DeepMatcher: A Deep Learning Approach to Entity Matching（2018） | 通过 | [PDF](./paper/36_deepmatcher_deepmatcher.pdf) | [报告](./paper_reports/36_deepmatcher.md)
- [37] Low-resource Deep Entity Resolution with Transfer and Active Learning（2019） | 通过 | [PDF](./paper/37_low_resource_deep_er_low_resource_deep_er.pdf) | [报告](./paper_reports/37_low_resource_deep_er.md)
- [38] S2ORC: The Semantic Scholar Open Research Corpus（2019） | 通过 | [PDF](./paper/38_s2orc_1911.02782.pdf) | [报告](./paper_reports/38_s2orc.md)
- [39] SciBERT: A Pretrained Language Model for Scientific Text（2019） | 通过 | [PDF](./paper/39_scibert_1903.10676.pdf) | [报告](./paper_reports/39_scibert.md)
- [40] SPECTER: Document-level Representation Learning using Citation-informed Transformers（2020） | 通过 | [PDF](./paper/40_specter_2004.07180.pdf) | [报告](./paper_reports/40_specter.md)
- [41] Scalable Zero-shot Entity Linking with Dense Entity Retrieval（2019） | 通过 | [PDF](./paper/41_blink_1911.03814.pdf) | [报告](./paper_reports/41_blink.md)
- [42] Autoregressive Entity Retrieval（2020） | 通过 | [PDF](./paper/42_autoregressive_entity_retrieval_2010.00904.pdf) | [报告](./paper_reports/42_autoregressive_entity_retrieval.md)

## 6. 备注
- 建议滚动更新论文池，持续跟进新成果。
- 若引入外部 API，建议补齐限流、重试、审计与合规策略。