# Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering

## 1. 基本信息
- 序号: 16
- 研究主线: Retrieval & RAG
- 年份: 2020
- 会议/来源: arXiv 2020
- 优先级: P1
- 作者: Gautier Izacard, Edouard Grave
- PDF: [16_fid_2007.01282.pdf](../paper/16_fid_2007.01282.pdf)
- 链接: https://arxiv.org/abs/2007.01282

## 2. 研究问题与贡献
- 问题焦点: Fusion-in-Decoder architecture for multi-document reasoning.
- 摘要要点: Generative models for open domain question answering have proven to be competitive, without resorting to external knowledge. While promising, this approach requires to use models with billions of parameters, which are expensive to train and query. In this paper, we investigate how much these models can benefit from retrieving text passages, potentially containing evidence. We obtain state-of-the-art results on the Natural Questions and TriviaQA open benchmarks. Interestingly, we observe that the performance of this method significantly improves when increasing the number of retrieved passages. This is evidence that generative models are good at aggregating and combining evidence from multiple passages.

## 3. 与 deepsearch 的融合点
- Split evidence retrieval and answer generation stages in pipeline.py for reliability.
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
- 报告路径: `paper_reports\16_fid.md`
- PDF 路径: `paper\16_fid_2007.01282.pdf`
