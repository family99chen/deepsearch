"""
Google Scholar 账号查找路由
提供 ORCID → Google Scholar URL 的查找服务
支持流式日志输出
"""

import sys
import os
import asyncio
from io import StringIO
from datetime import date
from typing import Optional, AsyncGenerator
import concurrent.futures

from fastapi import APIRouter, Query, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 添加子目录到 path
sys.path.insert(0, "google_scholar_url")

from google_scholar_url.google_account_fetcher_pipeline import find_google_scholar_by_orcid
from pipeline import run_person_pipeline, run_person_pipeline_by_orcid
from utils.usage_tracker import record_api_call, get_tracker
from utils.pipeline_stats import get_pipeline_stats

# 创建路由器
router = APIRouter(tags=["Google Scholar"])

# Pipeline 并发控制（避免资源争抢）
PIPELINE_MAX_CONCURRENT = int(os.getenv("PIPELINE_MAX_CONCURRENT", "5"))
PIPELINE_SEMAPHORE = asyncio.Semaphore(PIPELINE_MAX_CONCURRENT)


# ============ 使用统计依赖 ============

async def track_usage(request: Request):
    """
    依赖注入：自动记录 API 调用
    在路由函数执行前调用
    """
    endpoint = request.url.path
    record_api_call(endpoint)
    return endpoint


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
    
    log_buffer = StringIO()
    result_obj = None
    error_msg = None
    
    try:
        def run_sync():
            nonlocal result_obj
            old_stdout = sys.stdout
            sys.stdout = log_buffer
            try:
                if mode == "orcid":
                    result_obj = run_person_pipeline_by_orcid(orcid_id=identifier)
                else:
                    result_obj = run_person_pipeline(google_scholar_url=identifier)
            finally:
                sys.stdout = old_stdout
        
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_sync)
        
        last_pos = 0
        while not future.done():
            await asyncio.sleep(0.5)
            log_buffer.seek(last_pos)
            new_logs = log_buffer.read()
            last_pos = log_buffer.tell()
            if new_logs:
                for line in new_logs.strip().split('\n'):
                    if line:
                        yield f"data: [LOG] {line}\n\n"
        
        await future
        
        log_buffer.seek(last_pos)
        remaining_logs = log_buffer.read()
        if remaining_logs:
            for line in remaining_logs.strip().split('\n'):
                if line:
                    yield f"data: [LOG] {line}\n\n"
        
    except Exception as e:
        error_msg = str(e)
        yield f"data: [ERROR] {error_msg}\n\n"
    
    if result_obj:
        result = PersonPipelineResult(
            person_name=result_obj.person_name,
            organization=result_obj.organization,
            report=result_obj.report,
            iterations=result_obj.iterations,
            queries=result_obj.queries,
            sources=result_obj.sources,
        )
        yield f"data: [RESULT] {result.model_dump_json()}\n\n"
    else:
        yield f"data: [RESULT] {{\"success\": false, \"error\": \"{error_msg or 'unknown error'}\"}}\n\n"
    
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
    log_buffer = StringIO()
    result_url = None
    result_author = None
    error_msg = None
    
    try:
        # 在线程池中运行同步的 pipeline 函数
        def run_sync():
            nonlocal result_url, result_author
            # 重定向 stdout 到 buffer
            old_stdout = sys.stdout
            sys.stdout = log_buffer
            try:
                url, author = find_google_scholar_by_orcid(
                    orcid_id=orcid_id,
                    verbose=True
                )
                result_url = url
                result_author = author
            finally:
                sys.stdout = old_stdout
        
        # 使用 asyncio 在线程中运行
        loop = asyncio.get_event_loop()
        
        # 启动任务
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_sync)
        
        # 持续读取日志并输出
        last_pos = 0
        while not future.done():
            await asyncio.sleep(0.5)  # 每 0.5 秒检查一次
            
            # 读取新的日志
            log_buffer.seek(last_pos)
            new_logs = log_buffer.read()
            last_pos = log_buffer.tell()
            
            if new_logs:
                for line in new_logs.strip().split('\n'):
                    if line:
                        yield f"data: [LOG] {line}\n\n"
        
        # 等待任务完成
        await future
        
        # 读取剩余的日志
        log_buffer.seek(last_pos)
        remaining_logs = log_buffer.read()
        if remaining_logs:
            for line in remaining_logs.strip().split('\n'):
                if line:
                    yield f"data: [LOG] {line}\n\n"
        
    except Exception as e:
        error_msg = str(e)
        yield f"data: [ERROR] {error_msg}\n\n"
    
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
    
    yield f"data: [RESULT] {result.model_dump_json()}\n\n"
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
    - [RESULT] 最终结果（JSON 格式）
    - [END] 处理结束
    """
    return StreamingResponse(
        run_pipeline_with_logs(orcid_id=orcid_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲
        }
    )


@router.get("/find/sync", response_model=SearchResult)
async def find_google_scholar_account_sync(
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage)  # 自动记录调用
):
    """
    根据 ORCID ID 查找对应的 Google Scholar 账号（同步返回）
    
    参数配置从 config.yaml 读取（max_iterations, match_threshold, max_matches）
    
    注意：此接口可能需要较长时间，建议使用 /find 流式接口
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
            author_name=result_author.get('name') if result_author else None,
            affiliation=result_author.get('affiliation') if result_author else None
        )
    else:
        return SearchResult(
            success=False,
            orcid_id=orcid_id,
            error=error_msg or "未找到匹配的 Google Scholar 账号"
        )


@router.post("/person/report", response_model=PersonPipelineResult, tags=["Person Pipeline"])
async def person_report_by_google_scholar(
    google_scholar_url: str = Query(..., description="Google Scholar 个人主页 URL"),
    _tracked: str = Depends(track_usage),
):
    """
    输入 Google Scholar URL 生成个人完整报告
    """
    async with PIPELINE_SEMAPHORE:
        def run_sync():
            return run_person_pipeline(google_scholar_url=google_scholar_url)

        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        result = await loop.run_in_executor(executor, run_sync)
        return PersonPipelineResult(
            person_name=result.person_name,
            organization=result.organization,
            report=result.report,
            iterations=result.iterations,
            queries=result.queries,
            sources=result.sources,
        )


@router.get("/person/report/stream", tags=["Person Pipeline"])
async def person_report_by_google_scholar_stream(
    google_scholar_url: str = Query(..., description="Google Scholar 个人主页 URL"),
    _tracked: str = Depends(track_usage),
):
    """
    Google Scholar 报告（流式输出）
    """
    return StreamingResponse(
        run_person_pipeline_with_logs(mode="gs", identifier=google_scholar_url),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/person/report/orcid", response_model=PersonPipelineResult, tags=["Person Pipeline"])
async def person_report_by_orcid(
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage),
):
    """
    输入 ORCID ID 生成个人完整报告
    """
    async with PIPELINE_SEMAPHORE:
        def run_sync():
            return run_person_pipeline_by_orcid(orcid_id=orcid_id)

        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        result = await loop.run_in_executor(executor, run_sync)
        return PersonPipelineResult(
            person_name=result.person_name,
            organization=result.organization,
            report=result.report,
            iterations=result.iterations,
            queries=result.queries,
            sources=result.sources,
        )


@router.get("/person/report/orcid/stream", tags=["Person Pipeline"])
async def person_report_by_orcid_stream(
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    _tracked: str = Depends(track_usage),
):
    """
    ORCID 报告（流式输出）
    """
    return StreamingResponse(
        run_person_pipeline_with_logs(mode="orcid", identifier=orcid_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
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
