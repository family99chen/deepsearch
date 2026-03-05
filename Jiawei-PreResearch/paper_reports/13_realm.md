# REALM: Retrieval-Augmented Language Model Pre-Training

## 1. 基本信息
- 序号: 13
- 研究主线: Retrieval & RAG
- 年份: 2020
- 会议/来源: ICML 2020
- 优先级: P1
- 作者: Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang
- PDF: [13_realm_2002.08909.pdf](../paper/13_realm_2002.08909.pdf)
- 链接: https://arxiv.org/abs/2002.08909

## 2. 研究问题与贡献
- 问题焦点: Retrieval-augmented pretraining for knowledge-intensive tasks.
- 摘要要点: Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts. To capture knowledge in a more modular and interpretable way, we augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents. We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.

## 3. 与 deepsearch 的融合点
- Add dense retrieval stage after Google API recall in google_search_api/google_search.py and org_info/google_org_search.py.
- Train domain-adaptive contrastive retriever on internal person-page corpus.
- 建议代码锚点:
  - `google_search_api/google_search.py`
  - `org_info/google_org_search.py`
  - `org_info/arbitrary_search.py`
  - `pipeline.py`
  - `localdb/insert_mongo.py`

## 4. MVP 落地计划
- 建立固定离线评测集（匹配准确率、检索召回率、证据覆盖率）。
- 以影子模式启用策略，对比质量/成本/时延并保留回滚开关。
- 沉淀低置信样本，持续迭代数据与规则。

## 5. 预期收益与风险
- 预期收益: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- 主要风险: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. 产物路径
- 报告路径: `paper_reports\13_realm.md`
- PDF 路径: `paper\13_realm_2002.08909.pdf`
