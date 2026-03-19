

```md
## 部署 DeepSearch 到新服务器

本文说明如何将 `deepsearch` 应用和 MongoDB 缓存服务部署到一台新的 Linux 服务器上。

### 1. 部署目标

本项目依赖两部分服务：

1. `DeepSearch API`
2. `MongoDB` 缓存服务

其中：

- API 默认监听 `8080`
- MongoDB 默认监听 `27018`
- 默认数据库名为 `deepsearch_cache`

项目当前使用 `FastAPI + uvicorn` 运行，配置文件为 `config.yaml`。

---

## 2. 服务器要求

建议服务器环境：

- Ubuntu 20.04 / 22.04 / 24.04
- Python 3.10+
- MongoDB 6.x 或 7.x
- 至少 2 核 CPU
- 至少 4GB 内存
- 能访问外网
- 能正常启动 Chrome / Selenium 所需依赖

如果需要启用 `org_info` 相关网页深挖能力，还需要图形相关依赖：

- `xvfb`
- `google-chrome` 或兼容 Chromium
- Selenium 运行依赖库

---

## 3. 拉取代码

```bash
cd /opt
git clone <你的仓库地址> deepsearch
cd /opt/deepsearch
```

如果你不是通过 Git 部署，也可以直接把当前项目目录整体复制到新服务器，例如放在：

```bash
/opt/deepsearch
```

---

## 4. 创建 Python 虚拟环境

```bash
cd /opt/deepsearch
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pymongo
pip install undetected-chromedriver
```

说明：

- `requirements.txt` 里当前没有显式包含 `pymongo`
- 但项目缓存功能依赖 `pymongo`
- `org_info` 的浏览器自动化能力依赖 `undetected-chromedriver`

---

## 5. 安装系统依赖

先安装常用运行依赖：

```bash
apt update
apt install -y \
  python3 python3-venv python3-pip \
  xvfb curl wget git unzip ca-certificates \
  libnss3 libxss1 libasound2 libatk-bridge2.0-0 libgtk-3-0 \
  libgbm1 libxdamage1 libxrandr2 libu2f-udev libvulkan1
```

如果服务器上还没有 Chrome，建议安装 Google Chrome：

```bash
wget -O /tmp/google-chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt install -y /tmp/google-chrome.deb
```

安装完成后检查：

```bash
google-chrome --version
```

---

## 6. 安装 MongoDB

### 方案 A：直接安装 MongoDB 服务

以 Ubuntu 为例：

```bash
apt update
apt install -y gnupg curl
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" > /etc/apt/sources.list.d/mongodb-org-7.0.list
apt update
apt install -y mongodb-org
```

### 修改 MongoDB 端口为 `27018`

编辑：

```bash
/etc/mongod.conf
```

建议至少改成：

```yaml
storage:
  dbPath: /var/lib/mongodb

systemLog:
  destination: file
  path: /var/log/mongodb/mongod.log
  logAppend: true

net:
  port: 27018
  bindIp: 127.0.0.1

processManagement:
  timeZoneInfo: /usr/share/zoneinfo
```

启动并设为开机自启：

```bash
systemctl enable mongod
systemctl restart mongod
systemctl status mongod
```

### 检查 MongoDB

```bash
mongosh --port 27018
```

进入后可执行：

```javascript
show dbs
use deepsearch_cache
db.runCommand({ ping: 1 })
```

---

## 7. 配置 `config.yaml`

项目核心配置文件：

```bash
/opt/deepsearch/config.yaml
```

你至少需要检查这些字段：

### MongoDB

```yaml
mongodb:
  host: "localhost"
  port: 27018
  db_name: "deepsearch_cache"
  collection: "api_cache"
```

### Google Custom Search

```yaml
google:
  api_key: "你的 Google API Key"
  cx: "你的 Custom Search Engine ID"
```

### SerpAPI

```yaml
serpapi:
  api_key: "你的 SerpAPI Key"
```

### ORCID

```yaml
orcid:
  access_token: "你的 ORCID token"
```

### LLM

```yaml
llm:
  backend: "api"
  api:
    api_key: "你的 LLM API Key"
    api_base: "你的 OpenAI 兼容接口地址"
    model: "gpt-4o-mini"
    timeout: 120
```

### Person Pipeline

```yaml
person_pipeline:
  backend: "api"
  model: "gpt-5.4"
  max_iterations: 1
  max_links: 10
  max_workers: 3
```

---

## 8. 强烈建议：不要把密钥直接写进仓库

当前项目的 `config.yaml` 是明文配置形式。部署到新服务器时，建议：

1. 不要直接提交线上密钥到 Git
2. 只保留 `config.example.yaml`
3. 实际部署机器上手动写 `config.yaml`
4. 或者改造成读取环境变量

例如：

- `GOOGLE_API_KEY`
- `GOOGLE_CX`
- `SERPAPI_API_KEY`
- `LLM_API_KEY`
- `ORCID_ACCESS_TOKEN`

如果暂时不改代码，至少要保证：

- 仓库是私有的
- 服务器权限收紧
- `config.yaml` 不被误提交

---

## 9. 启动 API 服务

进入项目目录：

```bash
cd /opt/deepsearch
source .venv/bin/activate
```

开发测试方式：

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

如果你只是想快速验证服务是否正常，访问：

```bash
curl http://127.0.0.1:8080/
```

预期能看到：

- 服务名
- 版本号
- `/docs`

Swagger 文档地址：

```text
http://<服务器IP>:8080/docs
```

---

## 10. 推荐使用 systemd 托管 API

创建文件：

```bash
/etc/systemd/system/deepsearch.service
```

内容示例：

```ini
[Unit]
Description=DeepSearch FastAPI Service
After=network.target mongod.service
Wants=mongod.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/deepsearch
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/deepsearch/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

生效并启动：

```bash
systemctl daemon-reload
systemctl enable deepsearch
systemctl restart deepsearch
systemctl status deepsearch
```

查看日志：

```bash
journalctl -u deepsearch -f
```

---

## 11. 推荐使用 Nginx 反向代理

如果你想对外提供稳定访问，建议用 Nginx 转发到 `127.0.0.1:8080`。

创建站点配置：

```bash
/etc/nginx/sites-available/deepsearch
```

内容示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 20m;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_buffering off;
        proxy_cache off;
    }
}
```

启用：

```bash
ln -s /etc/nginx/sites-available/deepsearch /etc/nginx/sites-enabled/deepsearch
nginx -t
systemctl reload nginx
```

如果你需要 HTTPS，再配合 `certbot` 即可。

---

## 12. MongoDB 缓存说明

项目会自动在 MongoDB 中创建缓存集合和索引，不需要手动建表。

当前会用到的缓存包括但不限于：

- ORCID -> Google Scholar 映射缓存
- Google Search / SerpAPI 调用缓存
- `person_pipeline_cache`
- `page_analysis_cache`
- 其他 `org_info` 相关缓存集合

Mongo 缓存模块会自动创建：

- `key` 唯一索引
- `expire_at` TTL 索引

因此只要 MongoDB 服务正常，应用首次写入时会自动初始化索引。

---

## 13. 部署后建议的验证步骤

### 1. 检查首页

```bash
curl http://127.0.0.1:8080/
```

### 2. 检查 API 文档

浏览器访问：

```text
http://<服务器IP>:8080/docs
```

### 3. 测试 ORCID -> Scholar 查找

```bash
curl -N "http://127.0.0.1:8080/find?orcid_id=0000-0002-1825-0097"
```

### 4. 测试 ORCID 直接生成报告

```bash
curl -N "http://127.0.0.1:8080/person/report/orcid/stream?orcid_id=0000-0002-1825-0097"
```

### 5. 测试 Scholar 用户 ID 直接生成报告

```bash
curl -N "http://127.0.0.1:8080/person/report/stream?user_id=iWykd1cAAAAJ"
```

### 6. 检查 Mongo 是否写入缓存

```bash
mongosh --port 27018
```

进入后：

```javascript
use deepsearch_cache
show collections
```

---

## 14. 常见问题

### 1. `MongoDB 连接失败`

检查：

- `mongod` 是否启动
- 端口是否为 `27018`
- `config.yaml` 是否一致
- 是否只绑定了 `127.0.0.1`

### 2. `404 Not Found`

常见原因：

- 请求路径写错
- 把完整 URL 错误编码成 path
- 反向代理没有正确转发

例如，正确的是：

```text
/person/report/orcid/stream?orcid_id=xxxx
```

不是：

```text
http%3A//host:8080/person/report/orcid/stream?orcid_id=xxxx
```

### 3. Selenium / Chrome 启动失败

检查：

- `xvfb` 是否安装
- `google-chrome` 是否安装
- 缺失的系统动态库是否已安装
- 服务器内存是否足够

### 4. LLM 接口 `400 Bad Request`

检查：

- `config.yaml` 中的 `llm.api.api_base`
- `api_key`
- `model`
- prompt 是否过长
- provider 是否支持当前请求格式

---

## 15. 生产环境建议

建议至少做下面几件事：

1. 使用 `systemd` 托管 `deepsearch`
2. 使用 `Nginx` 做反向代理
3. `MongoDB` 仅监听 `127.0.0.1`
4. 不要把真实密钥提交到仓库
5. 为日志目录和项目目录做好备份
6. 部署后先用少量真实请求验证缓存、LLM、Google Search、SerpAPI 是否都正常

---

## 16. 一套最小可用部署流程

如果你想快速在新服务器上跑起来，可以按下面顺序执行：

```bash
apt update
apt install -y python3 python3-venv python3-pip git xvfb wget curl unzip ca-certificates \
  libnss3 libxss1 libasound2 libatk-bridge2.0-0 libgtk-3-0 libgbm1 libxdamage1 libxrandr2 libu2f-udev libvulkan1

cd /opt
git clone <你的仓库地址> deepsearch
cd /opt/deepsearch

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pymongo undetected-chromedriver

# 安装 MongoDB，并改到 27018
# 配置 /opt/deepsearch/config.yaml

uvicorn main:app --host 0.0.0.0 --port 8080
```

然后访问：

```text
http://<服务器IP>:8080/docs
```

即可开始测试。
```
