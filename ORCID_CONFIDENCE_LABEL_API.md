# ORCID Confidence Label API Guide

本文说明新版 ORCID -> Google Scholar 匹配接口中的 `confidence_label` 字段怎么用、哪些接口会返回、以及如何解释返回值。

## 1. 字段说明

新版 API 在 ORCID 入口相关结果里增加了一个可空字段：

```json
"confidence_label": "high risk"
```

可取值如下：

- `high risk`
- `low risk`
- `no risk`
- `null`

说明：

- `high risk`：ORCID 人名与最终匹配到的 Google Scholar 人名在规范化后不完全一致。
- `low risk`：两边名字一致，但参与 ORCID 重合验证的 Scholar 论文里，作者列表出现了两次同名作者，存在同名歧义风险。
- `no risk`：名字一致，且没有触发上述同名歧义规则。
- `null`：通常表示命中了旧缓存，旧数据里还没有这个字段；系统不会对旧缓存做回填。

## 2. 适用范围

`confidence_label` 只会出现在 **ORCID 入口** 的结果里。

包含该字段的接口：

- `GET /find/sync`
- `GET /find` 的最终 SSE `[RESULT]`
- `POST /find/job` 提交后，通过 `GET /jobs/{job_id}` 读取最终结果
- `POST /person/report/orcid` 提交后，通过 `GET /jobs/{job_id}` 读取最终结果
- `GET /person/report/orcid/stream` 的最终 SSE `[RESULT]`

不保证返回该字段的接口：

- `POST /person/report`
- `GET /person/report/stream`

也就是说，**直接用 Google Scholar URL / user_id 走 DeepSearch 报告时，不以 `confidence_label` 作为契约字段**。

## 3. 返回位置

### 3.1 ORCID -> Google Scholar 匹配接口

`GET /find/sync?orcid_id=<ORCID>`

返回结构：

```json
{
  "success": true,
  "orcid_id": "0000-0002-1825-0097",
  "google_scholar_url": "https://scholar.google.com/citations?user=xxxx",
  "author_name": "John Smith",
  "affiliation": "Example University",
  "match_count": 12,
  "confidence_label": "no risk",
  "error": null
}
```

### 3.2 ORCID DeepSearch 报告接口

`POST /person/report/orcid?orcid_id=<ORCID>`

该接口本身先返回 job 提交信息：

```json
{
  "job_id": "job_xxx",
  "status": "pending",
  "job_url": "http://host/jobs/job_xxx",
  "stream_url": "http://host/jobs/job_xxx/stream"
}
```

然后通过 `GET /jobs/{job_id}` 或 stream 拿最终结果。最终结果中会包含：

```json
{
  "person_name": "John Smith",
  "organization": "Example University",
  "report": "...",
  "iterations": 1,
  "queries": [],
  "sources": [],
  "confidence_label": "low risk"
}
```

## 4. 典型调用方式

### 4.1 同步查 ORCID -> Google Scholar

```bash
curl "http://127.0.0.1:8080/find/sync?orcid_id=0000-0002-1825-0097"
```

### 4.2 异步查 ORCID -> Google Scholar

提交任务：

```bash
curl -X POST "http://127.0.0.1:8080/find/job?orcid_id=0000-0002-1825-0097"
```

查询任务：

```bash
curl "http://127.0.0.1:8080/jobs/<job_id>"
```

如果成功，`result` 部分会带：

```json
{
  "success": true,
  "orcid_id": "0000-0002-1825-0097",
  "google_scholar_url": "https://scholar.google.com/citations?user=xxxx",
  "author_name": "John Smith",
  "affiliation": "Example University",
  "match_count": 12,
  "confidence_label": "high risk",
  "error": null
}
```

### 4.3 ORCID DeepSearch 报告

提交任务：

```bash
curl -X POST "http://127.0.0.1:8080/person/report/orcid?orcid_id=0000-0002-1825-0097"
```

读取结果：

```bash
curl "http://127.0.0.1:8080/jobs/<job_id>"
```

或直接看流式输出：

```bash
curl -N "http://127.0.0.1:8080/person/report/orcid/stream?orcid_id=0000-0002-1825-0097"
```

最终 `[RESULT]` 中会包含 `confidence_label`。

## 5. 风险标签判定规则

系统当前按以下顺序判定：

1. 先把 ORCID 名字和最终匹配到的 Google Scholar 名字做规范化。
2. 如果规范化后名字不完全一样，返回 `high risk`。
3. 如果名字一样，再看参与 ORCID 论文重合验证的 Scholar 匹配论文。
4. 如果这些匹配论文里有任意一篇的作者列表中出现两次同名作者，而且这个名字就是最终匹配到的人，返回 `low risk`。
5. 否则返回 `no risk`。

这里的 `high / low / no risk` 是 **匹配风险标签**，不是系统错误码，也不是人物真实性结论。

## 6. `null` 的解释

`confidence_label = null` 最常见的原因不是新逻辑失败，而是：

- 命中了旧版 `orcid_googleaccount_map` 缓存；
- 或命中了旧版 `person_pipeline_cache`；
- 这些旧文档创建时还没有 `confidence_label` 字段。

兼容策略是：

- 旧缓存不补写；
- 旧缓存继续可读；
- API 直接返回 `null`；
- 只有新生成的数据才会带 `high risk` / `low risk` / `no risk`。

因此：

- `null` 不等于 `no risk`
- `null` 只表示“当前这条缓存没有标签”

## 7. 缓存落点

该字段会写入两个地方：

### 7.1 ORCID -> Google Scholar 映射缓存

Mongo collection：

`orcid_googleaccount_map`

字段位置：

```json
{
  "key": "<orcid_id>",
  "value": {
    "google_scholar_url": "...",
    "name": "...",
    "match_count": 12,
    "confidence_label": "high risk"
  }
}
```

### 7.2 ORCID DeepSearch 报告缓存

Mongo collection：

`person_pipeline_cache`

字段位置：

```json
{
  "key": "gs:...|iter:...|links:...|workers:...|model:...|backend:...|ver:v2_no_extra",
  "value": {
    "final": {
      "person_name": "...",
      "organization": "...",
      "report": "...",
      "confidence_label": "low risk"
    }
  }
}
```

## 8. 使用建议

如果你把这个字段给下游系统使用，建议按下面方式处理：

- `high risk`：提示人工复核，特别是同名多作者、缩写名、拼音名场景。
- `low risk`：允许继续使用，但在 UI 或日志中标为“同名歧义风险”。
- `no risk`：可按默认正常结果处理。
- `null`：不要当作 `no risk`；这通常只是旧缓存未升级。

## 9. Python 示例

```python
import requests

resp = requests.get(
    "http://127.0.0.1:8080/find/sync",
    params={"orcid_id": "0000-0002-1825-0097"},
    timeout=120,
)
resp.raise_for_status()
data = resp.json()

print(data["google_scholar_url"])
print(data.get("confidence_label"))
```

异步 job 方式：

```python
import requests

submit = requests.post(
    "http://127.0.0.1:8080/person/report/orcid",
    params={"orcid_id": "0000-0002-1825-0097"},
    timeout=30,
).json()

job = requests.get(submit["job_url"], timeout=30).json()
result = job.get("result") or {}

print(result.get("person_name"))
print(result.get("confidence_label"))
```

## 10. 版本兼容说明

这次改动是 **加字段**，不是改协议版本：

- 没有改现有 API 路径；
- 没有改现有缓存 key；
- 没有要求旧缓存迁移；
- 老客户端忽略 `confidence_label` 即可；
- 新客户端要把 `null` 当作“未标注”，不要误判成 `no risk`。
