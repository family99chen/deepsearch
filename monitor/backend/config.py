"""Monitor service configuration (read-only, no changes to deepsearch)."""

from pathlib import Path

MONITOR_ROOT = Path(__file__).resolve().parent.parent
DEEPSEARCH_ROOT = Path("/root/deepsearch")
TOTAL_USAGE_DIR = DEEPSEARCH_ROOT / "total_usage"

HOST = "127.0.0.1"
PORT = 8091

REDIS_URL = "redis://127.0.0.1:6379/0"
MONGO_URL = "mongodb://127.0.0.1:27018/"
MONGO_DB = "deepsearch_cache"

CELERY_BIN = "/root/miniconda3/envs/webprof/bin/celery"
CELERY_APP = "tasks:celery_app"
CELERY_CWD = str(DEEPSEARCH_ROOT)
CELERY_ENV = {"REDIS_URL": "redis://127.0.0.1:6379/0"}

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=8",
    "-o", "StrictHostKeyChecking=accept-new",
]

WORKERS = [
    {"id": "main", "label": "主节点", "ssh_host": None},
    {"id": "worker1", "label": "Worker 1", "ssh_host": "worker1"},
    {"id": "worker2", "label": "Worker 2", "ssh_host": "worker2"},
    {"id": "worker3", "label": "Worker 3", "ssh_host": "worker3"},
    {"id": "worker4", "label": "Worker 4", "ssh_host": "worker4"},
    {"id": "worker5", "label": "Worker 5", "ssh_host": "worker5"},
    {"id": "worker6", "label": "Worker 6", "ssh_host": "worker6"},
    {"id": "worker7", "label": "Worker 7", "ssh_host": "worker7"},
]

USAGE_FILES = {
    "mapping_api": "mapping_api.json",
    "pipeline_stats": "pipeline_stats.json",
    "llm": "llm.json",
    "org_pipeline_stats": "org_pipeline_stats.json",
    "patent_pipeline_stats": "patent_pipeline_stats.json",
    "google_search": "google_search.json",
    "serpapi": "serpapi.json",
}

USAGE_LABELS = {
    "mapping_api": "API 调用 (mapping_api)",
    "pipeline_stats": "ORCID 查找 (pipeline_stats)",
    "llm": "LLM 调用 (llm)",
    "org_pipeline_stats": "人物报告 (org_pipeline_stats)",
    "patent_pipeline_stats": "专利查询 (patent_pipeline_stats)",
    "google_search": "Google 搜索 (google_search)",
    "serpapi": "SerpAPI (serpapi)",
}
