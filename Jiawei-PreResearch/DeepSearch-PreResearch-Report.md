# DeepSearch Paper Pre-Research Report

- Generated at: 2026-03-05 23:17:50
- Total papers: 42
- Successful downloads: 42
- Failed downloads: 0

## 1. Scope
- Agentic Web Research: deep research agents, web interaction, search reasoning.
- Retrieval & RAG: recall, reranking, RAG control/correction, evaluation.
- Scholarly Identity: author disambiguation, entity matching, scientific text representation.

## 2. Integration Opportunities for Current Project
- Upgrade ORCID->Scholar mapping with learned disambiguation beyond static thresholds.
- Build multi-stage retrieval: search API recall + embedding recall + LLM rerank.
- Strengthen agent loop in iteragent_advanced with explicit planning and recovery.
- Add retrieval quality gating and evidence-grounded report generation in pipeline.py.
- Expand metrics in utils/pipeline_stats.py for quality/cost/latency governance.

## 3. Phased Roadmap
### Phase A (1-2 weeks)
- Add reranking in candidate selection (RankGPT / setwise ranking pattern).
- Add retrieval quality gate and re-retrieval fallback branch.
- Build minimal offline benchmark and reporting dashboard.

### Phase B (2-4 weeks)
- Integrate learned disambiguation features (S2AND/WhoIsWho style).
- Add scientific embeddings (SciBERT/SPECTER/E5) for paper-level matching.
- Create human-in-the-loop loop for low-confidence cases.

### Phase C (4-8 weeks)
- Evaluate agent strategies with BrowserGym/WebArena style tasks.
- Distill policy from logs and optionally run RL-style optimization.
- Finalize KPI monitoring for quality, latency, and API cost.

## 4. P0 Priority Papers
1. ReAct: Synergizing Reasoning and Acting in Language Models (2022, Agentic Web Research)
2. WebArena: A Realistic Web Environment for Building Autonomous Agents (2023, Agentic Web Research)
3. The BrowserGym Ecosystem for Web Agent Research (2024, Agentic Web Research)
4. Search-o1: Agentic Search-Enhanced Large Reasoning Models (2025, Agentic Web Research)
5. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2025, Agentic Web Research)
6. DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments (2025, Agentic Web Research)
7. Dense Passage Retrieval for Open-Domain Question Answering (2020, Retrieval & RAG)
8. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020, Retrieval & RAG)
9. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (2021, Retrieval & RAG)
10. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models (2021, Retrieval & RAG)
11. Text Embeddings by Weakly-Supervised Contrastive Pre-training (2022, Retrieval & RAG)
12. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (2023, Retrieval & RAG)
13. Corrective Retrieval Augmented Generation (2024, Retrieval & RAG)
14. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents (2023, Retrieval & RAG)
15. S2AND: A Benchmark and Evaluation System for Author Name Disambiguation (2021, Scholarly Identity)
16. Web-Scale Academic Name Disambiguation: the WhoIsWho Benchmark, Leaderboard, and Toolkit (2023, Scholarly Identity)
17. Deep Entity Matching with Pre-Trained Language Models (2020, Scholarly Identity)
18. DeepMatcher: A Deep Learning Approach to Entity Matching (2018, Scholarly Identity)

## 5. Full Index
### Agentic Web Research
- [01] WebGPT: Browser-assisted question-answering with human feedback (2021) | OK | [PDF](./paper/01_webgpt_2112.09332.pdf) | [Report](./paper_reports/01_webgpt.md)
- [02] ReAct: Synergizing Reasoning and Acting in Language Models (2022) | OK | [PDF](./paper/02_react_2210.03629.pdf) | [Report](./paper_reports/02_react.md)
- [03] Toolformer: Language Models Can Teach Themselves to Use Tools (2023) | OK | [PDF](./paper/03_toolformer_2302.04761.pdf) | [Report](./paper_reports/03_toolformer.md)
- [04] Mind2Web: Towards a Generalist Agent for the Web (2023) | OK | [PDF](./paper/04_mind2web_2306.06070.pdf) | [Report](./paper_reports/04_mind2web.md)
- [05] WebArena: A Realistic Web Environment for Building Autonomous Agents (2023) | OK | [PDF](./paper/05_webarena_2307.13854.pdf) | [Report](./paper_reports/05_webarena.md)
- [06] GPT-4V(ision) is a Generalist Web Agent, if Grounded (2024) | OK | [PDF](./paper/06_seeact_gpt4v_agent_2401.01614.pdf) | [Report](./paper_reports/06_seeact_gpt4v_agent.md)
- [07] The BrowserGym Ecosystem for Web Agent Research (2024) | OK | [PDF](./paper/07_browsergym_2412.05467.pdf) | [Report](./paper_reports/07_browsergym.md)
- [08] Search-o1: Agentic Search-Enhanced Large Reasoning Models (2025) | OK | [PDF](./paper/08_search_o1_2501.05366.pdf) | [Report](./paper_reports/08_search_o1.md)
- [09] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (2025) | OK | [PDF](./paper/09_search_r1_2503.09516.pdf) | [Report](./paper_reports/09_search_r1.md)
- [10] DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments (2025) | OK | [PDF](./paper/10_deepresearcher_2504.03160.pdf) | [Report](./paper_reports/10_deepresearcher.md)
- [11] WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning (2025) | OK | [PDF](./paper/11_websailor_v2_2509.13305.pdf) | [Report](./paper_reports/11_websailor_v2.md)
- [12] Universal Deep Research: Bring Your Own Model and Strategy (2025) | OK | [PDF](./paper/12_universal_deep_research_2509.00244.pdf) | [Report](./paper_reports/12_universal_deep_research.md)

### Retrieval & RAG
- [13] REALM: Retrieval-Augmented Language Model Pre-Training (2020) | OK | [PDF](./paper/13_realm_2002.08909.pdf) | [Report](./paper_reports/13_realm.md)
- [14] Dense Passage Retrieval for Open-Domain Question Answering (2020) | OK | [PDF](./paper/14_dpr_2004.04906.pdf) | [Report](./paper_reports/14_dpr.md)
- [15] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020) | OK | [PDF](./paper/15_rag_2005.11401.pdf) | [Report](./paper_reports/15_rag.md)
- [16] Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (2020) | OK | [PDF](./paper/16_fid_2007.01282.pdf) | [Report](./paper_reports/16_fid.md)
- [17] ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (2020) | OK | [PDF](./paper/17_colbert_2004.12832.pdf) | [Report](./paper_reports/17_colbert.md)
- [18] ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (2021) | OK | [PDF](./paper/18_colbertv2_2112.01488.pdf) | [Report](./paper_reports/18_colbertv2.md)
- [19] Unsupervised Dense Information Retrieval with Contrastive Learning (2021) | OK | [PDF](./paper/19_contriever_2112.09118.pdf) | [Report](./paper_reports/19_contriever.md)
- [20] BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models (2021) | OK | [PDF](./paper/20_beir_2104.08663.pdf) | [Report](./paper_reports/20_beir.md)
- [21] SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (2021) | OK | [PDF](./paper/21_splade_v2_2109.10086.pdf) | [Report](./paper_reports/21_splade_v2.md)
- [22] Text Embeddings by Weakly-Supervised Contrastive Pre-training (2022) | OK | [PDF](./paper/22_e5_2212.03533.pdf) | [Report](./paper_reports/22_e5.md)
- [23] Precise Zero-Shot Dense Retrieval without Relevance Labels (2022) | OK | [PDF](./paper/23_hyde_2212.10496.pdf) | [Report](./paper_reports/23_hyde.md)
- [24] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (2023) | OK | [PDF](./paper/24_self_rag_2310.11511.pdf) | [Report](./paper_reports/24_self_rag.md)
- [25] Corrective Retrieval Augmented Generation (2024) | OK | [PDF](./paper/25_crag_2401.15884.pdf) | [Report](./paper_reports/25_crag.md)
- [26] From Local to Global: A Graph RAG Approach to Query-Focused Summarization (2024) | OK | [PDF](./paper/26_graphrag_2404.16130.pdf) | [Report](./paper_reports/26_graphrag.md)
- [27] One Embedder, Any Task: Instruction-Finetuned Text Embeddings (2022) | OK | [PDF](./paper/27_instructor_2212.09741.pdf) | [Report](./paper_reports/27_instructor.md)
- [28] MTEB: Massive Text Embedding Benchmark (2022) | OK | [PDF](./paper/28_mteb_2210.07316.pdf) | [Report](./paper_reports/28_mteb.md)
- [29] Towards General Text Embeddings with Multi-stage Contrastive Learning (2023) | OK | [PDF](./paper/29_bge_multistage_2308.03281.pdf) | [Report](./paper_reports/29_bge_multistage.md)
- [30] M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation (2024) | OK | [PDF](./paper/30_m3_embedding_2402.03216.pdf) | [Report](./paper_reports/30_m3_embedding.md)
- [31] Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents (2023) | OK | [PDF](./paper/31_rankgpt_2304.09542.pdf) | [Report](./paper_reports/31_rankgpt.md)
- [32] A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models (2023) | OK | [PDF](./paper/32_setwise_ranking_2310.09497.pdf) | [Report](./paper_reports/32_setwise_ranking.md)

### Scholarly Identity
- [33] S2AND: A Benchmark and Evaluation System for Author Name Disambiguation (2021) | OK | [PDF](./paper/33_s2and_2103.07534.pdf) | [Report](./paper_reports/33_s2and.md)
- [34] Web-Scale Academic Name Disambiguation: the WhoIsWho Benchmark, Leaderboard, and Toolkit (2023) | OK | [PDF](./paper/34_whoiswho_2302.11848.pdf) | [Report](./paper_reports/34_whoiswho.md)
- [35] Deep Entity Matching with Pre-Trained Language Models (2020) | OK | [PDF](./paper/35_ditto_2004.00584.pdf) | [Report](./paper_reports/35_ditto.md)
- [36] DeepMatcher: A Deep Learning Approach to Entity Matching (2018) | OK | [PDF](./paper/36_deepmatcher_deepmatcher.pdf) | [Report](./paper_reports/36_deepmatcher.md)
- [37] Low-resource Deep Entity Resolution with Transfer and Active Learning (2019) | OK | [PDF](./paper/37_low_resource_deep_er_low_resource_deep_er.pdf) | [Report](./paper_reports/37_low_resource_deep_er.md)
- [38] S2ORC: The Semantic Scholar Open Research Corpus (2019) | OK | [PDF](./paper/38_s2orc_1911.02782.pdf) | [Report](./paper_reports/38_s2orc.md)
- [39] SciBERT: A Pretrained Language Model for Scientific Text (2019) | OK | [PDF](./paper/39_scibert_1903.10676.pdf) | [Report](./paper_reports/39_scibert.md)
- [40] SPECTER: Document-level Representation Learning using Citation-informed Transformers (2020) | OK | [PDF](./paper/40_specter_2004.07180.pdf) | [Report](./paper_reports/40_specter.md)
- [41] Scalable Zero-shot Entity Linking with Dense Entity Retrieval (2019) | OK | [PDF](./paper/41_blink_1911.03814.pdf) | [Report](./paper_reports/41_blink.md)
- [42] Autoregressive Entity Retrieval (2020) | OK | [PDF](./paper/42_autoregressive_entity_retrieval_2010.00904.pdf) | [Report](./paper_reports/42_autoregressive_entity_retrieval.md)

## 6. Notes
- For production rollout, include strict retry/rate-limit/audit policies for external APIs.
- For personally identifiable data, define data minimization and retention policies.
- Refresh this paper set periodically to include newer 2026+ work.
