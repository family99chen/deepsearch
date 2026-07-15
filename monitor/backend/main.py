"""DeepSearch ops monitor API — listens on 127.0.0.1:8091 only."""

from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from collectors.celery_collector import collect_celery
from collectors.llm_collector import collect_llm_local
from collectors.mongo_collector import collect_mongo
from collectors.nodes_collector import collect_nodes
from collectors.redis_collector import collect_redis
from collectors.usage_collector import (
    get_usage_batch,
    get_usage_for_date,
    last_n_days,
    list_usage_meta,
)
from config import HOST, PORT

app = FastAPI(title="DeepSearch Monitor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "service": "deepsearch-monitor"}


@app.get("/api/infra")
def infra():
    """Infrastructure snapshot: nodes, celery, redis, mongo, local LLM probe."""
    llm_local = collect_llm_local()
    return {
        "ok": True,
        "fetched_at": date.today().isoformat(),
        "nodes": collect_nodes(),
        "celery": collect_celery(),
        "redis": collect_redis(),
        "mongo": collect_mongo(),
        "llm_local": llm_local,
    }


@app.get("/api/usage/meta")
def usage_meta():
    return list_usage_meta()


@app.get("/api/usage")
def usage_one(
    file: str = Query(..., description="Usage file key"),
    date_str: Optional[str] = Query(None, alias="date"),
):
    day = date_str or date.today().isoformat()
    allowed = last_n_days(7)
    if day not in allowed:
        raise HTTPException(status_code=400, detail=f"date must be one of {allowed}")
    result = get_usage_for_date(file, day)
    if not result.get("ok") and "unknown file" in result.get("error", ""):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/usage/batch")
def usage_batch(date_str: Optional[str] = Query(None, alias="date")):
    day = date_str or date.today().isoformat()
    allowed = last_n_days(7)
    if day not in allowed:
        raise HTTPException(status_code=400, detail=f"date must be one of {allowed}")
    return get_usage_batch(day)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
