# SciBERT: A Pretrained Language Model for Scientific Text

## 1. 基本信息
- 序号: 39
- 研究主线: Scholarly Identity
- 年份: 2019
- 会议/来源: EMNLP 2019
- 优先级: P1
- 作者: Iz Beltagy, Kyle Lo, Arman Cohan
- PDF: [39_scibert_1903.10676.pdf](../paper/39_scibert_1903.10676.pdf)
- 链接: https://arxiv.org/abs/1903.10676

## 2. 研究问题与贡献
- 问题焦点: Domain-specific LM for scientific text understanding.
- 摘要要点: Obtaining large-scale annotated data for NLP tasks in the scientific domain is challenging and expensive. We release SciBERT, a pretrained language model based on BERT (Devlin et al., 2018) to address the lack of high-quality, large-scale labeled scientific data. SciBERT leverages unsupervised pretraining on a large multi-domain corpus of scientific publications to improve performance on downstream scientific NLP tasks. We evaluate on a suite of tasks including sequence tagging, sentence classification and dependency parsing, with datasets from a variety of scientific domains. We demonstrate statistically significant improvements over BERT and achieve new state-of-the-art results on several of these tasks. The code and pretrained models are available at this https URL .

## 3. 与 deepsearch 的融合点
- Use SciBERT embeddings for scientific name/title disambiguation.
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
- 报告路径: `paper_reports\39_scibert.md`
- PDF 路径: `paper\39_scibert_1903.10676.pdf`
