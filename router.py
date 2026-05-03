"""
Google Scholar 账号查找路由
提供 ORCID → Google Scholar URL 的查找服务
支持流式日志输出
"""

import sys
import os
import json
import asyncio
from datetime import date
from typing import Optional, AsyncGenerator
import concurrent.futures
from urllib.parse import quote

from fastapi import APIRouter, Query, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 添加子目录到 path
sys.path.insert(0, "google_scholar_url")

from google_scholar_url.google_account_fetcher_pipeline import find_google_scholar_by_orcid
from pipeline import run_person_pipeline, run_person_pipeline_by_orcid, is_failure_report
from job_store import get_job_store
from tasks import (
    submit_orcid_find,
    submit_person_report_gs,
    submit_person_report_orcid,
)
from utils.usage_tracker import record_api_call, get_tracker
from utils.org_pipeline_stats import get_org_pipeline_stats
from utils.pipeline_stats import get_pipeline_stats
from utils.stream_capture import (
    ThreadSafeConsoleBuffer,
    capture_console_output,
    drain_console_buffer,
)

# 创建路由器
router = APIRouter(tags=["Google Scholar"])

# Pipeline 并发控制（避免资源争抢）
PIPELINE_MAX_CONCURRENT = int(os.getenv("PIPELINE_MAX_CONCURRENT", "10"))
PIPELINE_SEMAPHORE = asyncio.Semaphore(PIPELINE_MAX_CONCURRENT)


def _format_sse_json(tag: str, payload: dict) -> str:
    return f"data: {tag} {json.dumps(payload, ensure_ascii=False)}\n\n"


def _read_log_lines(
    log_buffer: ThreadSafeConsoleBuffer,
    position: int,
    remainder: str,
):
    return drain_console_buffer(log_buffer, position, remainder)


# ============ 使用统计依赖 ============

async def track_usage(request: Request):
    """
    依赖注入：自动记录 API 调用
    在路由函数执行前调用
    """
    endpoint = request.url.path
    record_api_call(endpoint)
    return endpoint


def _resolve_google_scholar_url(
    google_scholar_url: Optional[str],
    user_id: Optional[str],
) -> str:
    """兼容完整 URL 和 Scholar user_id 两种输入。"""
    if google_scholar_url:
        return google_scholar_url.strip()

    if user_id:
        return f"https://scholar.google.com/citations?user={quote(user_id.strip())}"

    raise HTTPException(
        status_code=400,
        detail="请提供 google_scholar_url 或 user_id",
    )


class SearchResult(BaseModel):
    """搜索结果模型"""
    success: bool
    orcid_id: str
    google_scholar_url: Optional[str] = None
    author_name: Optional[str] = None
    affiliation: Optional[str] = None
    match_count: Optional[int] = None
    error: Optional[str] = None


class PersonPipelineResult(BaseModel):
    """Person Pipeline 结果模型"""
    person_name: str
    organization: Optional[str]
    report: str
    iterations: int
    queries: list
    sources: list


class JobSubmission(BaseModel):
    """异步任务提交结果"""
    job_id: str
    status: str
    job_url: str
    stream_url: str


class JobStatus(BaseModel):
    """异步任务状态"""
    job_id: str
    job_type: Optional[str] = None
    status: str
    payload: dict = Field(default_factory=dict)
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    updated_at: Optional[str] = None
    celery_task_id: Optional[str] = None


def _build_job_submission(request: Request, job_id: str) -> JobSubmission:
    return JobSubmission(
        job_id=job_id,
        status="pending",
        job_url=str(request.url_for("get_job_status", job_id=job_id)),
        stream_url=str(request.url_for("stream_job_status", job_id=job_id)),
    )


def _submit_job_or_503(submitter, *args) -> str:
    try:
        return submitter(*args)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"任务队列不可用或提交失败: {exc}",
        ) from exc


def _format_stored_sse_event(event_id: str, fields: dict) -> str:
    tag = fields.get("tag") or "[LOG]"
    payload_json = fields.get("payload_json")
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            payload = {"raw": payload_json}
        return f"id: {event_id}\n" + _format_sse_json(tag, payload)

    message = fields.get("message") or ""
    suffix = f" {message}" if message else ""
    return f"id: {event_id}\ndata: {tag}{suffix}\n\n"


async def stream_job_events(
    job_id: str,
    last_event_id: str = "0-0",
    emit_queued_log: bool = False,
) -> AsyncGenerator[str, None]:
    store = get_job_store()
    if not store.get_job(job_id):
        yield _format_sse_json("[ERROR]", {"success": False, "error": "任务不存在"})
        yield "data: [END]\n\n"
        return

    if emit_queued_log:
        yield f"data: [LOG] 任务已提交: {job_id}\n\n"

    current_event_id = last_event_id
    while True:
        events, current_event_id = await asyncio.to_thread(
            store.read_events,
            job_id,
            current_event_id,
            5000,
            100,
        )
        for event_id, fields in events:
            yield _format_stored_sse_event(event_id, fields)
            if fields.get("tag") == "[END]":
                return

        job = await asyncio.to_thread(store.get_job, job_id)
        if job and job.get("status") in {"success", "failed"}:
            result = job.get("result") or {"success": False, "error": job.get("error")}
            tag = "[RESULT]" if job.get("status") == "success" else "[ERROR]"
            yield _format_sse_json(tag, result)
            yield "data: [END]\n\n"
            return


async def run_person_pipeline_with_logs(mode: str, identifier: str) -> AsyncGenerator[str, None]:
    """
    运行 person pipeline 并流式输出日志
    
    Args:
        mode: "gs" 或 "orcid"
        identifier: google_scholar_url 或 orcid_id
    """
    if mode == "orcid":
        yield f"data: [START] 开始生成 ORCID 报告: {identifier}\n\n"
    else:
        yield f"data: [START] 开始生成 Google Scholar 报告: {identifier}\n\n"
    
    log_buffer = ThreadSafeConsoleBuffer()
    result_obj = None
    error_msg = None
    
    try:
        def run_sync():
            nonlocal result_obj
            with capture_console_output(log_buffer):
                if mode == "orcid":
                    result_obj = run_person_pipeline_by_orcid(orcid_id=identifier)
                else:
                    result_obj = run_person_pipeline(google_scholar_url=identifier)
        
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_sync)
        
        last_pos = 0
        remainder = ""
        while not future.done():
            await asyncio.sleep(0.5)
            lines, last_pos, remainder = _read_log_lines(log_buffer, last_pos, remainder)
            for line in lines:
                yield f"data: [LOG] {line}\n\n"
        
        await future
        
        lines, last_pos, remainder = _read_log_lines(log_buffer, last_pos, remainder)
        for line in lines:
            yield f"data: [LOG] {line}\n\n"
        if remainder:
            yield f"data: [LOG] {remainder}\n\n"
        
    except Exception as e:
        error_msg = str(e)
    
    if result_obj:
        result = PersonPipelineResult(
            person_name=result_obj.person_name,
            organization=result_obj.organization,
            report=result_obj.report,
            iterations=result_obj.iterations,
            queries=result_obj.queries,
            sources=result_obj.sources,
        )
        event_tag = "[ERROR]" if is_failure_report(result.report) else "[RESULT]"
        yield _format_sse_json(event_tag, result.model_dump())
    else:
        yield _format_sse_json(
            "[ERROR]",
            {"success": False, "error": error_msg or "unknown error"},
        )
    
    yield "data: [END]\n\n"


async def run_pipeline_with_logs(orcid_id: str) -> AsyncGenerator[str, None]:
    """
    运行 pipeline 并流式输出日志
    
    Args:
        orcid_id: ORCID ID
        
    Yields:
        日志行和最终结果
    
    Note:
        其他参数（max_iterations, match_threshold, max_matches）
        由 pipeline 从 config.yaml 读取
    """
    yield f"data: [START] 开始查找 ORCID: {orcid_id}\n\n"
    
    # 用于捕获 print 输出
    log_buffer = ThreadSafeConsoleBuffer()
    result_url = None
    result_author = None
    error_msg = None
    
    try:
        # 在线程池中运行同步的 pipeline 函数
        def run_sync():
            nonlocal result_url, result_author
            with capture_console_output(log_buffer):
                url, author = find_google_scholar_by_orcid(
                    orcid_id=orcid_id,
                    verbose=True
                )
                result_url = url
                result_author = author
        
        # 使用 asyncio 在线程中运行
        loop = asyncio.get_event_loop()
        
        # 启动任务
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_sync)
        
        # 持续读取日志并输出
        last_pos = 0
        remainder = ""
        while not future.done():
            await asyncio.sleep(0.5)  # 每 0.5 秒检查一次
            lines, last_pos, remainder = _read_log_lines(log_buffer, last_pos, remainder)
            for line in lines:
                yield f"data: [LOG] {line}\n\n"
        
        # 等待任务完成
        await future
        
        # 读取剩余的日志
        lines, last_pos, remainder = _read_log_lines(log_buffer, last_pos, remainder)
        for line in lines:
            yield f"data: [LOG] {line}\n\n"
        if remainder:
            yield f"data: [LOG] {remainder}\n\n"
        
    except Exception as e:
        error_msg = str(e)
    
    # 输出最终结果
    if result_url:
        result = SearchResult(
            success=True,
            orcid_id=orcid_id,
            google_scholar_url=result_url,
            author_name=result_author.get('name') if result_author else None,
            affiliation=result_author.get('affiliation') if result_author else None,
            match_count=result_author.get('match_count') if result_author else None
        )
    else:
        result = SearchResult(
            success=False,
            orcid_id=orcid_id,
            error=error_msg or "未找到匹配的 Google Scholar 账号"
        )
    
    if result.success:
        yield _format_sse_json("[RESULT]", result.model_dump())
    else:
        yield _format_sse_json("[ERROR]", result.model_dump())
    yield "data: [END]\n\n"


@router.get("/find")
async def find_google_scholar_account_stream(
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage)  # 自动记录调用
):
    """
    根据 ORCID ID 查找对应的 Google Scholar 账号（流式输出）
    
    参数配置从 config.yaml 读取（max_iterations, match_threshold, max_matches）
    
    返回 Server-Sent Events (SSE) 流式响应：
    - [START] 开始处理
    - [LOG] 处理日志
    - [RESULT] 成功结果（JSON 格式）
    - [ERROR] 失败结果（JSON 格式）
    - [END] 处理结束
    """
    job_id = _submit_job_or_503(submit_orcid_find, orcid_id)
    return StreamingResponse(
        stream_job_events(job_id=job_id, emit_queued_log=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲
        }
    )


@router.post("/find/job", response_model=JobSubmission)
async def submit_google_scholar_account_job(
    request: Request,
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage),
):
    """
    根据 ORCID ID 提交 Google Scholar 查找任务。

    批量调用推荐使用该接口：立即返回 job_id，不保持 SSE 长连接。
    """
    job_id = _submit_job_or_503(submit_orcid_find, orcid_id)
    return _build_job_submission(request, job_id)


@router.get("/find/sync", response_model=SearchResult)
async def find_google_scholar_account_sync(
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage)  # 自动记录调用
):
    """
    根据 ORCID ID 查找对应的 Google Scholar 账号（同步返回）
    """
    result_url = None
    result_author = None
    error_msg = None

    def run_sync():
        nonlocal result_url, result_author
        url, author = find_google_scholar_by_orcid(
            orcid_id=orcid_id,
            verbose=False
        )
        result_url = url
        result_author = author

    try:
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        await loop.run_in_executor(executor, run_sync)
    except Exception as e:
        error_msg = str(e)

    if result_url:
        return SearchResult(
            success=True,
            orcid_id=orcid_id,
            google_scholar_url=result_url,
            author_name=result_author.get("name") if result_author else None,
            affiliation=result_author.get("affiliation") if result_author else None,
            match_count=result_author.get("match_count") if result_author else None,
        )

    return SearchResult(
        success=False,
        orcid_id=orcid_id,
        error=error_msg or "未找到匹配的 Google Scholar 账号",
    )


@router.post("/person/report", response_model=JobSubmission, tags=["Person Pipeline"])
async def person_report_by_google_scholar(
    request: Request,
    google_scholar_url: Optional[str] = Query(None, description="Google Scholar 个人主页 URL"),
    user_id: Optional[str] = Query(None, description="Google Scholar 账号 ID，如 iWykd1cAAAAJ"),
    _tracked: str = Depends(track_usage),
):
    """
    输入 Google Scholar URL 或账号 ID 生成个人完整报告
    """
    resolved_google_scholar_url = _resolve_google_scholar_url(google_scholar_url, user_id)
    job_id = _submit_job_or_503(submit_person_report_gs, resolved_google_scholar_url)
    return _build_job_submission(request, job_id)


@router.get("/person/report/stream", tags=["Person Pipeline"])
async def person_report_by_google_scholar_stream(
    google_scholar_url: Optional[str] = Query(None, description="Google Scholar 个人主页 URL"),
    user_id: Optional[str] = Query(None, description="Google Scholar 账号 ID，如 iWykd1cAAAAJ"),
    _tracked: str = Depends(track_usage),
):
    """
    输入 Google Scholar URL 或账号 ID，流式生成报告
    """
    resolved_google_scholar_url = _resolve_google_scholar_url(google_scholar_url, user_id)
    job_id = _submit_job_or_503(submit_person_report_gs, resolved_google_scholar_url)

    return StreamingResponse(
        stream_job_events(job_id=job_id, emit_queued_log=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/person/report/orcid", response_model=JobSubmission, tags=["Person Pipeline"])
async def person_report_by_orcid(
    request: Request,
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage),
):
    """
    输入 ORCID ID 生成个人完整报告
    """
    job_id = _submit_job_or_503(submit_person_report_orcid, orcid_id)
    return _build_job_submission(request, job_id)


@router.get("/person/report/orcid/stream", tags=["Person Pipeline"])
async def person_report_by_orcid_stream(
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage),
):
    """
    ORCID 报告（流式输出）
    """
    job_id = _submit_job_or_503(submit_person_report_orcid, orcid_id)
    return StreamingResponse(
        stream_job_events(job_id=job_id, emit_queued_log=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str):
    """获取异步任务状态和最终结果。"""
    job = await asyncio.to_thread(get_job_store().get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JobStatus(**job)


@router.get("/jobs/{job_id}/stream", tags=["Jobs"])
async def stream_job_status(
    job_id: str,
    after: str = Query("0-0", description="Redis Stream event id，用于断线续传"),
):
    """从 Redis Stream 读取任务日志和最终结果。"""
    return StreamingResponse(
        stream_job_events(job_id=job_id, last_event_id=after),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============ 使用统计端点 ============

@router.get("/usage", tags=["Usage"])
async def get_usage_stats():
    """
    获取 API 使用统计
    
    返回：
    - total: 总调用次数
    - daily: 按日期分组的调用统计
    - endpoints: 按端点分组的调用统计
    """
    tracker = get_tracker()
    return tracker.get_stats()


@router.get("/usage/today", tags=["Usage"])
async def get_today_usage():
    """
    获取今日 API 使用统计
    
    返回：
    - date: 今日日期
    - total: 今日总调用次数
    - endpoints: 各端点调用次数
    - google_search_api: Google Search API 使用估算
    """
    tracker = get_tracker()
    today_stats = tracker.get_daily_stats()
    today_total = today_stats.get("total", 0)
    
    # Google Search API 每日免费额度 100 次
    # 根据之前分析，每次查找最多消耗 3 次 Google Search
    google_search_daily_limit = 100
    estimated_google_usage = today_total * 3  # 估算
    
    return {
        "date": date.today().isoformat(),
        "total": today_total,
        "endpoints": today_stats.get("endpoints", {}),
        "first_request": today_stats.get("first_request"),
        "last_request": today_stats.get("last_request"),
        "google_search_api": {
            "daily_limit": google_search_daily_limit,
            "estimated_usage": estimated_google_usage,
            "estimated_remaining": max(0, google_search_daily_limit - estimated_google_usage)
        }
    }


@router.get("/usage/pipeline", tags=["Usage"])
async def get_pipeline_usage():
    """
    获取 Pipeline 详细统计
    
    返回详细的查找统计：
    - total_requests: 总请求次数
    - cache_hits: 缓存命中次数（直接返回已缓存的结果）
    - success: 成功获取作者次数
    - error: 错误次数
    - not_found: 未找到次数
    - name_search: 名字搜索统计
      - total: 通过名字搜索成功的总数
      - by_iterations: 按 Google Search 调用次数分类 {"1": 10, "2": 5, ...}
    - paper_search: 论文搜索统计
      - total: 通过论文搜索成功的总数
      - by_papers: 按搜索论文数分类 {"1": 8, "2": 2, ...}
    """
    stats = get_pipeline_stats()
    return stats.get_stats()


@router.get("/usage/pipeline/today", tags=["Usage"])
async def get_pipeline_today_usage():
    """
    获取今日 Pipeline 详细统计
    """
    stats = get_pipeline_stats()
    return stats.get_today_stats()


@router.get("/usage/org-pipeline", tags=["Usage"])
async def get_org_pipeline_usage():
    """
    获取 deepsearch 调用统计

    返回：
    - total_requests: deepsearch 总调用次数
    - cache_hits: person_pipeline_cache 命中次数
    - success / not_found / error: 互斥终态统计
    """
    stats = get_org_pipeline_stats()
    return stats.get_stats()


@router.get("/usage/org-pipeline/today", tags=["Usage"])
async def get_org_pipeline_today_usage():
    """
    获取今日 deepsearch 调用统计
    """
    stats = get_org_pipeline_stats()
    return stats.get_today_stats()
