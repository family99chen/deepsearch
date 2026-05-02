# DeepSearch 分布式队列部署

## 组件

- API 节点：运行 `uvicorn main:app`，只负责提交任务、查询任务和转发 SSE。
- Redis：作为 Celery broker、result backend 和任务事件存储。
- Worker 节点：运行 `celery -A tasks:celery_app worker`，执行 Chromium/搜索/report pipeline。
- MongoDB：继续作为业务缓存，所有 API/worker 应连接同一个 MongoDB。容器部署可用 `MONGODB_URL` 覆盖 `config.yaml`。

## 本机验证

```bash
cp .env.example .env
docker compose up --build
```

调用后会返回 `job_id`，再用 `/jobs/{job_id}` 查询结果，或用 `/jobs/{job_id}/stream` 读取日志。

## 三台 CPU Worker 部署

1. 在一台机器上部署 API 和 Redis，Redis 只暴露内网地址。
2. 三台 CPU 服务器使用同一个镜像启动 worker，并设置相同的 `REDIS_URL`。
3. 每台 worker 挂载自己的 `config.yaml`，或用 `MONGODB_URL` 等环境变量覆盖部署差异；按机器能力设置：
   - `browser_scheduler.global_slots`
   - `browser_scheduler.default_domain_slots`
   - `person_pipeline.max_workers`
   - `organization_pipeline.max_concurrent`
4. 建议初始值：
   - `CELERY_WORKER_CONCURRENCY=1`
   - `person_pipeline.max_workers=2`
   - `organization_pipeline.max_concurrent=2`
   - `browser_scheduler.global_slots=2`
5. 稳定后再逐台增加 `global_slots` 和 `max_workers`，观察 Chrome 崩溃率、目标站限流、任务失败率和平均耗时。

## Worker 启动示例

```bash
docker run -d --name deepsearch-worker \
  --shm-size=1g \
  -e REDIS_URL=redis://<redis-internal-host>:6379/0 \
  -e CELERY_BROKER_URL=redis://<redis-internal-host>:6379/0 \
  -e CELERY_RESULT_BACKEND=redis://<redis-internal-host>:6379/0 \
  -e MONGODB_URL=mongodb://<mongo-internal-host>:27018/ \
  -e CELERY_WORKER_PREFETCH_MULTIPLIER=1 \
  -v /opt/deepsearch/config.yaml:/app/config.yaml:ro \
  -v /opt/deepsearch/logs:/app/logs \
  -v /opt/deepsearch/total_usage:/app/total_usage \
  deepsearch:latest \
  celery -A tasks:celery_app worker --loglevel=INFO --concurrency=1
```

## 注意事项

- 不要把真实 `config.yaml` 烧进镜像；生产用只读 volume 或密钥系统挂载。
- 多台机器不要共享同一个 NFS `total_usage/` 目录；当前 JSON 统计适合单机，本方案只保留兼容。
- 如果某台 worker 的出口 IP 被限流，先降低该节点的 `global_slots` 和域名并发。
- 容器运行 Chromium 时必须给足 `/dev/shm`，建议从 `1g` 起步。
