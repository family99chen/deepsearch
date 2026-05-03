# DeepSearch Worker 分布式部署说明

这份文档给 worker 服务器部署使用。当前主服务器是：

```text
公网 IP: 47.250.116.163
内网 IP: 172.26.2.150
API: 47.250.116.163:8080
Redis: 172.26.2.150:6379，无密码
MongoDB: 172.26.2.150:27018，无认证
```

## 1. 当前主服务器状态

主服务器负责：

```text
FastAPI API
Redis 任务队列
MongoDB 共享缓存
本机 Celery worker
```

Redis 已配置为：

```text
bind 0.0.0.0 -::*
protected-mode no
requirepass 空
port 6379
```

MongoDB 当前通过 Docker 暴露：

```text
0.0.0.0:27018 -> mongod --bind_ip_all
```

所有 worker 服务器都应该连同一个 Redis 和 MongoDB。

## 2. 安全组要求

如果三台 worker 和主服务器在同一个内网，优先使用：

```text
172.26.2.150
```

主服务器安全组/防火墙只允许 worker 服务器访问：

```text
6379  Redis
27018 MongoDB
```

不要把 Redis/MongoDB 对全公网开放给所有 IP。当前 Redis 无密码，为了效率优先，必须靠安全组白名单保护。

如果三台 worker 不在同一个内网，只能用公网：

```text
47.250.116.163:6379
47.250.116.163:27018
```

这种情况下更要做安全组白名单，只放行三台 worker 的公网 IP。

## 3. Worker 服务器要做什么

worker 服务器不需要启动 FastAPI，也不需要开放 8080。

它只需要：

```text
1. 放一份 deepsearch 代码
2. 安装 Python 依赖
3. 配置 REDIS_URL
4. 配置 MONGODB_URL
5. 启动 Celery worker
```

## 4. 部署代码

方式 A：从 Git 拉代码：

```bash
cd /root
git clone <你的仓库地址> deepsearch
cd /root/deepsearch
```

方式 B：从主服务器复制代码：

```bash
scp -r root@47.250.116.163:/root/deepsearch /root/deepsearch
cd /root/deepsearch
```

确保三台 worker 使用同一份代码版本。

## 5. 安装 Python 环境

如果服务器已经有 conda：

```bash
conda create -n webprof python=3.11 -y
conda activate webprof
cd /root/deepsearch
pip install -r requirements.txt
```

如果你们统一使用主服务器一样的路径，后续命令默认使用：

```text
/root/miniconda3/envs/webprof/bin/python
/root/miniconda3/envs/webprof/bin/celery
```

如果路径不同，把启动命令里的 celery 路径换成实际路径。

## 6. 配置环境变量

同内网推荐：

```bash
export REDIS_URL="redis://172.26.2.150:6379/0"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
export MONGODB_URL="mongodb://172.26.2.150:27018/"
```

如果只能走公网：

```bash
export REDIS_URL="redis://47.250.116.163:6379/0"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
export MONGODB_URL="mongodb://47.250.116.163:27018/"
```

测试 Redis：

```bash
redis-cli -h 172.26.2.150 -p 6379 ping
```

应该返回：

```text
PONG
```

测试 Mongo：

```bash
mongosh --host 172.26.2.150 --port 27018 --quiet --eval 'db.adminCommand({ping:1})'
```

应该返回：

```text
{ ok: 1 }
```

## 7. 启动 Worker

建议每台 worker 先从 `concurrency=10` 开始：

```bash
cd /root/deepsearch

export REDIS_URL="redis://172.26.2.150:6379/0"
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"
export MONGODB_URL="mongodb://172.26.2.150:27018/"

pm2 start /root/miniconda3/envs/webprof/bin/celery \
  --name deepsearch-worker \
  --interpreter none \
  -- -A tasks:celery_app worker --loglevel=INFO --concurrency=10
```

如果这台 worker 配置和主服务器一样，稳定后可以尝试：

```bash
pm2 delete deepsearch-worker
pm2 start /root/miniconda3/envs/webprof/bin/celery \
  --name deepsearch-worker \
  --interpreter none \
  -- -A tasks:celery_app worker --loglevel=INFO --concurrency=15
```

不建议单台一开始就上 20 以上，因为 Chromium 会吃 CPU，Google Scholar 也可能反爬。

保存 PM2 进程：

```bash
pm2 save
```

## 8. 检查 Worker 是否接入成功

```bash
pm2 status
pm2 logs deepsearch-worker --lines 100 --nostream
```

看到这些表示成功：

```text
Connected to redis://172.26.2.150:6379/0
celery@... ready
[tasks]
  . deepsearch.orcid_find
  . deepsearch.person_report_gs
  . deepsearch.person_report_orcid
```

在主服务器可以看 Redis 队列：

```bash
redis-cli llen celery
```

也可以看 Celery 活跃任务：

```bash
/root/miniconda3/envs/webprof/bin/celery -A tasks:celery_app inspect active --timeout=5
```

## 9. 调用方怎么用

批量任务推荐调用主服务器 API：

```bash
curl -X POST "http://47.250.116.163:8080/find/job?orcid_id=0000-0001-5063-2645"
```

返回：

```json
{
  "job_id": "...",
  "status": "pending",
  "job_url": "http://47.250.116.163:8080/jobs/...",
  "stream_url": "http://47.250.116.163:8080/jobs/.../stream"
}
```

轮询结果：

```bash
curl "http://47.250.116.163:8080/jobs/<job_id>"
```

结果完成后：

```json
{
  "status": "success",
  "result": {
    "success": true,
    "orcid_id": "0000-0001-5063-2645",
    "google_scholar_url": "https://scholar.google.com/citations?user=...",
    "author_name": "...",
    "affiliation": "...",
    "match_count": null,
    "error": null
  }
}
```

不要用 `/find` 做大批量并发，因为它是 SSE 长连接，客户端容易断流。

推荐：

```text
POST /find/job
GET  /jobs/{job_id}
```

## 10. MongoDB 缓存共享说明

MongoDB 可以共享读写，并且推荐共享。

所有 worker 连接：

```text
mongodb://172.26.2.150:27018/
```

共享的缓存包括：

```text
orcid_googleaccount_map       ORCID -> Google Scholar URL
orcid_info                    ORCID 姓名/论文/组织
google_scholar_person         Scholar 候选人
google_scholar_paper          论文搜索缓存
google_scholar_person_detail  Scholar 作者论文列表缓存
person_pipeline_cache         deepsearch 报告缓存
```

任何一台 worker 写入缓存，其他 worker 都能读到。

## 11. 注意事项

1. 当前主服务器 IP `47.250.116.163` 已被 Google Scholar 判定 unusual traffic。新 worker 如果有不同出口 IP，会更有帮助。
2. `scholar_author_fetch.use_requests: false` 时，会直接用 Chromium 获取 Scholar 作者页。
3. 单机 worker 并发不是越高越好。建议每台 `10-15` 起步。
4. Redis 无密码，必须靠安全组白名单保护。
5. MongoDB 无认证，必须靠安全组白名单保护。
6. 所有 worker 的 `config.yaml` 里的 Serper、SerpAPI、LLM 配置要和主服务器一致。

## 12. 一句话总结

主服务器：

```text
API + Redis + MongoDB
```

三台 worker：

```text
只跑 Celery worker，连接主服务器 Redis 和 MongoDB
```

调用方：

```text
只调用主服务器 API，不直接调用 worker
```
