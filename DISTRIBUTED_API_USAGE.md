# DeepSearch 分布式部署与 API 使用说明

本文档说明如何把 `deepsearch` 部署成 API + 多台 worker 的分布式服务，以及调用方如何使用 `find`、按 ORCID 生成 deepsearch 报告、按 Google Scholar 生成 deepsearch 报告。

## 1. 架构

推荐部署结构：

```text
Client
  -> API Server: FastAPI / uvicorn / pm2
      -> Redis: 任务队列、任务状态、流式日志
          -> Worker Server A: Celery worker
          -> Worker Server B: Celery worker
          -> Worker Server C: Celery worker
      -> MongoDB: 共享缓存
```

组件职责：

- API Server：只对外暴露 HTTP API。
- Redis：Celery broker、结果状态、任务日志流。
- Worker Server：真正执行 ORCID 查找、Serper 搜索、Scholar 验证、deepsearch 报告生成。
- MongoDB：缓存 ORCID 信息、Scholar 映射、页面分析和报告结果。

## 2. MongoDB 缓存能否共享

可以共享，而且建议共享。

所有 API 和 worker 连接同一个 MongoDB 后，可以复用这些缓存：

- `orcid_googleaccount_map`：ORCID -> Google Scholar URL 映射。
- `orcid_info`：ORCID 姓名、论文、组织信息。
- `google_scholar_person` / `google_scholar_paper`：Scholar 候选人和论文搜索缓存。
- `google_scholar_person_detail`：Scholar 作者页论文列表缓存。
- `person_pipeline_cache`：deepsearch 最终报告缓存。

推荐用环境变量覆盖 Mongo 连接：

```bash
export MONGODB_URL="mongodb://<mongo-private-ip>:27018/"
export MONGODB_DB_NAME="deepsearch_cache"
```

生产注意事项：

- MongoDB 不要暴露到公网。
- 只允许 API/worker 所在机器的内网 IP 访问。
- 多台 worker 不要各自使用独立 MongoDB，否则缓存不会共享。

## 3. Redis 部署

Redis 是分布式任务队列，不是业务缓存。

用途：

- API 把任务写入 Redis。
- Celery worker 从 Redis 拉任务。
- worker 把任务日志和最终结果写回 Redis。
- `/jobs/{job_id}` 和 `/jobs/{job_id}/stream` 从 Redis 读取状态和日志。

API Server 上安装并启动 Redis：

```bash
apt update
apt install -y redis-server
systemctl enable redis-server
systemctl start redis-server
redis-cli ping
```

看到 `PONG` 表示 Redis 正常。

如果 worker 在其他服务器，Redis 需要监听内网地址，并通过安全组/防火墙只允许 worker 访问。不要把 Redis 直接开放到公网。

worker 连接 Redis：

```bash
export REDIS_URL="redis://<redis-private-ip>:6379/0"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
```

## 4. Worker 服务器部署

每台 worker 服务器需要：

1. 部署同一份 `/root/deepsearch` 代码。
2. 安装同一个 Python 环境依赖。
3. 配置同一个 Redis 地址。
4. 配置同一个 MongoDB 地址。
5. 启动 Celery worker。

示例：

```bash
cd /root/deepsearch

/root/miniconda3/envs/webprof/bin/pip install -r requirements.txt

export REDIS_URL="redis://<redis-private-ip>:6379/0"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
export MONGODB_URL="mongodb://<mongo-private-ip>:27018/"

pm2 start /root/miniconda3/envs/webprof/bin/celery \
  --name deepsearch-worker \
  --interpreter none \
  -- -A tasks:celery_app worker --loglevel=INFO --concurrency=1
```

检查：

```bash
pm2 status
pm2 logs deepsearch-worker --lines 100 --nostream
```

看到类似内容表示 worker 正常：

```text
Connected to redis://...
celery@... ready
```

并发建议：

- 初始用 `--concurrency=1`。
- 当前每个任务内部还会调用 Serper、SerpAPI、Chromium、Mongo、LLM，不建议一开始开太大。
- 多台 CPU 服务器更适合“多机器每台低并发”，而不是单机高并发。

## 5. API Server 启动

API Server 启动 FastAPI：

```bash
cd /root/deepsearch

export REDIS_URL="redis://127.0.0.1:6379/0"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
export MONGODB_URL="mongodb://<mongo-private-ip-or-localhost>:27018/"

pm2 start /root/miniconda3/envs/webprof/bin/uvicorn \
  --name deepsearch \
  --interpreter none \
  -- main:app --host 0.0.0.0 --port 8080
```

重启：

```bash
pm2 restart deepsearch --update-env
```

## 6. API 使用方式

下面假设 API 地址为：

```text
http://47.250.116.163:8080
```

### 6.1 ORCID -> Google Scholar URL：旧版同步接口

接口：

```text
GET /find/sync?orcid_id=<ORCID>
```

示例：

```bash
curl "http://47.250.116.163:8080/find/sync?orcid_id=0000-0002-1429-026X"
```

返回成功：

```json
{
  "success": true,
  "orcid_id": "0000-0002-1429-026X",
  "google_scholar_url": "https://scholar.google.com/citations?user=xxxx",
  "author_name": "Name",
  "affiliation": "Organization",
  "match_count": 1,
  "error": null
}
```

返回未找到：

```json
{
  "success": false,
  "orcid_id": "0000-0002-1429-026X",
  "google_scholar_url": null,
  "author_name": null,
  "affiliation": null,
  "match_count": null,
  "error": "未找到匹配的 Google Scholar 账号"
}
```

注意：`/find/sync` 当前是旧版同步执行，不走 Redis/Celery 分布式 worker。它适合兼容旧调用方。

### 6.2 ORCID -> Google Scholar URL：分布式流式接口

接口：

```text
GET /find?orcid_id=<ORCID>
```

示例：

```bash
curl -N "http://47.250.116.163:8080/find?orcid_id=0000-0002-1429-026X"
```

返回是 SSE 文本流：

```text
data: [LOG] 任务已提交: <job_id>
id: ...
data: [START] 开始查找 ORCID: ...
id: ...
data: [LOG] ...
id: ...
data: [RESULT] {"success": true, ...}
data: [END]
```

这个接口会提交 Redis/Celery 任务，由 worker 执行，适合分布式。

### 6.3 DeepSearch by ORCID

接口：

```text
POST /person/report/orcid?orcid_id=<ORCID>
```

示例：

```bash
curl -X POST "http://47.250.116.163:8080/person/report/orcid?orcid_id=0000-0002-1429-026X"
```

返回任务：

```json
{
  "job_id": "abc123",
  "status": "pending",
  "job_url": "http://47.250.116.163:8080/jobs/abc123",
  "stream_url": "http://47.250.116.163:8080/jobs/abc123/stream"
}
```

查询结果：

```bash
curl "http://47.250.116.163:8080/jobs/abc123"
```

任务完成后的返回：

```json
{
  "job_id": "abc123",
  "job_type": "person_report_orcid",
  "status": "success",
  "payload": {
    "orcid_id": "0000-0002-1429-026X"
  },
  "result": {
    "person_name": "Name",
    "organization": "Organization",
    "report": "...",
    "iterations": 1,
    "queries": [],
    "sources": []
  },
  "error": null
}
```

流式日志：

```bash
curl -N "http://47.250.116.163:8080/jobs/abc123/stream"
```

或者直接使用 ORCID report stream 接口：

```bash
curl -N "http://47.250.116.163:8080/person/report/orcid/stream?orcid_id=0000-0002-1429-026X"
```

### 6.4 DeepSearch by Google Scholar URL

接口：

```text
POST /person/report?google_scholar_url=<Google Scholar URL>
```

示例：

```bash
curl -X POST --get "http://47.250.116.163:8080/person/report" \
  --data-urlencode "google_scholar_url=https://scholar.google.com/citations?user=xxxx"
```

也可以只传 Scholar user id：

```bash
curl -X POST "http://47.250.116.163:8080/person/report?user_id=xxxx"
```

返回任务：

```json
{
  "job_id": "def456",
  "status": "pending",
  "job_url": "http://47.250.116.163:8080/jobs/def456",
  "stream_url": "http://47.250.116.163:8080/jobs/def456/stream"
}
```

查询结果：

```bash
curl "http://47.250.116.163:8080/jobs/def456"
```

任务完成后的 `result`：

```json
{
  "person_name": "Name",
  "organization": "Organization",
  "report": "...",
  "iterations": 1,
  "queries": [],
  "sources": []
}
```

流式版本：

```bash
curl -N --get "http://47.250.116.163:8080/person/report/stream" \
  --data-urlencode "google_scholar_url=https://scholar.google.com/citations?user=xxxx"
```

## 7. 任务状态说明

`GET /jobs/{job_id}` 的状态：

```text
pending  -> 已提交，等待 worker
running  -> worker 正在执行
success  -> 任务执行完成
failed   -> 任务执行异常
```

注意：

- `status=success` 只代表任务执行完成。
- 业务上是否找到结果，要看 `result.success` 或报告内容。
- 例如 ORCID find 可能返回 `status=success` 但 `result.success=false`，表示任务正常跑完，但没有找到 Scholar。

## 8. 分布式部署注意事项

- 所有 worker 必须运行同一份代码版本。
- 所有 worker 必须连接同一个 Redis。
- 所有 worker 和 API 建议连接同一个 MongoDB。
- Redis 和 MongoDB 不要公网裸奔。
- worker 的 `config.yaml` 里的 API key、Serper、SerpAPI、LLM 配置要一致。
- `scholar_author_fetch.use_requests: false` 时，验证 Scholar 作者论文列表会直接走 Chromium。
- 每台 worker 都会启动自己的 Chromium，注意内存和 CPU。
- 建议先每台 worker `--concurrency=1`，稳定后再逐步增加。

## 9. 常见排查命令

API 状态：

```bash
pm2 status
pm2 logs deepsearch --lines 100 --nostream
```

Worker 状态：

```bash
pm2 logs deepsearch-worker --lines 100 --nostream
```

Redis：

```bash
redis-cli ping
```

MongoDB：

```bash
mongosh --port 27018
```

测试 API：

```bash
curl "http://47.250.116.163:8080/"
curl "http://47.250.116.163:8080/find/sync?orcid_id=0000-0002-1429-026X"
```
