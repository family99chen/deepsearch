"""
Google Scholar 账号查找路由
提供 ORCID → Google Scholar URL 的查找服务
支持流式日志输出
"""

import sys
import asyncio
from io import StringIO
from typing import Optional, AsyncGenerator
import concurrent.futures

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 添加子目录到 path
sys.path.insert(0, "google_scholar_url")

from google_scholar_url.google_account_fetcher_pipeline import find_google_scholar_by_orcid

# 创建路由器
router = APIRouter(tags=["Google Scholar"])


class SearchResult(BaseModel):
    """搜索结果模型"""
    success: bool
    orcid_id: str
    google_scholar_url: Optional[str] = None
    author_name: Optional[str] = None
    affiliation: Optional[str] = None
    match_count: Optional[int] = None
    error: Optional[str] = None


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
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097")
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
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097")
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
