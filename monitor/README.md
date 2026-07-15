# DeepSearch 运维监控面板

独立项目，不修改 `/root/deepsearch` 任何代码。通过只读方式采集 Redis、MongoDB、Celery、SSH Worker 节点，以及 `deepsearch/total_usage/*.json` 用量文件。

## 目录结构

```
deepsearch-monitor/
├── backend/          # FastAPI，监听 127.0.0.1:8091
├── frontend/         # 单页 Dashboard（拷贝到本地运行）
└── README.md
```

## 1. 启动后端（服务器上）

```bash
cd /root/deepsearch-monitor/backend
/root/miniconda3/envs/webprof/bin/pip install -r requirements.txt
/root/miniconda3/envs/webprof/bin/python main.py
```

或使用 uvicorn：

```bash
cd /root/deepsearch-monitor/backend
/root/miniconda3/envs/webprof/bin/uvicorn main:app --host 127.0.0.1 --port 8091
```

验证：

```bash
curl http://127.0.0.1:8091/api/health
```

### 可选：PM2 托管

```bash
pm2 start /root/miniconda3/envs/webprof/bin/uvicorn \
  --name deepsearch-monitor \
  --interpreter none \
  --cwd /root/deepsearch-monitor/backend \
  -- main:app --host 127.0.0.1 --port 8091
pm2 save
```

## 2. SSH 隧道（你本地电脑）

```bash
ssh -L 8091:127.0.0.1:8091 root@47.250.116.163
```

保持该终端不要关闭。

## 3. 启动前端（你本地电脑）

将 `frontend/` 文件夹拷贝到本地后：

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`。

Vite 会把 `/api/*` 代理到 `http://127.0.0.1:8091`（经 SSH 隧道）。

若不使用 Vite，也可在 API 地址栏填 `http://127.0.0.1:8091`，用任意静态服务器托管 `frontend/`（需后端 CORS 已开启）。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/infra` | 节点 + Celery + Redis + Mongo + **本地 LLM 探活** |
| GET | `/api/usage/meta` | 用量文件元信息 |
| GET | `/api/usage/batch?date=YYYY-MM-DD` | 7 个用量文件指定日期 |
| GET | `/api/usage?file=xxx&date=YYYY-MM-DD` | 单个用量文件 |

`date` 仅限最近 7 天。

## 面板内容

- 总览卡片：Celery 在线数、活跃任务、Redis 内存、Mongo 大小、**本地 LLM 健康**
- **本地大模型**面板：探活 URL、HTTP 状态、延迟（打开页面或刷新时 GET `/v1/models`，200 为健康）
- Celery 集群表、四节点状态表
- Redis / MongoDB 详情
- 7 个用量 JSON 卡片，每张可独立选最近 7 天中某一天

数据仅在打开页面、点击「刷新全部」、切换全局日期或单卡片日期下拉时请求，无自动轮询。
