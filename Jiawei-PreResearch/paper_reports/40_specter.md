# SPECTER: Document-level Representation Learning using Citation-informed Transformers

## 1. 基本信息
- 序号: 40
- 研究主线: Scholarly Identity
- 年份: 2020
- 会议/来源: ACL 2020
- 优先级: P1
- 作者: Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel S. Weld
- PDF: [40_specter_2004.07180.pdf](../paper/40_specter_2004.07180.pdf)
- 链接: https://arxiv.org/abs/2004.07180

## 2. 研究问题与贡献
- 问题焦点: Citation-informed document embeddings for paper similarity.
- 摘要要点: Representation learning is a critical ingredient for natural language processing systems. Recent Transformer language models like BERT learn powerful textual representations, but these models are targeted towards token- and sentence-level training objectives and do not leverage information on inter-document relatedness, which limits their document-level representation power. For applications on scientific documents, such as classification and recommendation, the embeddings power strong performance on end tasks. We propose SPECTER, a new method to generate document-level embedding of scientific documents based on pretraining a Transformer language model on a powerful signal of document-level relatedness: the citation graph. Unlike existing pretrained language models, SPECTER can be easily applied to downstream applications without task-specific fine-tuning. Additionally, to encourage further research on document-level models, we introduce SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. We show that SPECTER outperforms a variety of competitive baselines on the benchmark.

## 3. 与 deepsearch 的融合点
- Adopt SPECTER embeddings for document-level similarity in paper matching.
- 建议代码锚点:
  - `google_scholar_url/google_account_fetcher_pipeline.py`
  - `google_scholar_url/fetch_author_google_scholar_account.py`
  - `google_scholar_url/verify_author_info.py`
  - `google_scholar_url/name_normalization.py`

## 4. MVP 落地计划
- 建立固定离线评测集（匹配准确率、检索召回率、证据覆盖率）。
- 以影子模式启用策略，对比质量/成本/时延并保留回滚开关。
- 沉淀低置信样本，持续迭代数据与规则。

## 5. 预期收益与风险
- 预期收益: Better ORCID-Scholar matching quality, stronger retrieval precision/recall, and more grounded reports.
- 主要风险: Added complexity, runtime overhead, and benchmark maintenance cost.

## 6. 产物路径
- 报告路径: `paper_reports\40_specter.md`
- PDF 路径: `paper\40_specter_2004.07180.pdf`
