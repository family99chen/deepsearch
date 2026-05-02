"""Celery tasks for distributed DeepSearch workers."""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from celery import Celery

# 兼容 google_scholar_url 内部模块的绝对导入。
sys.path.insert(0, str(Path(__file__).parent / "google_scholar_url"))

from google_scholar_url.google_account_fetcher_pipeline import find_google_scholar_by_orcid
from job_store import get_job_store, get_redis_url
from pipeline import is_failure_report, run_person_pipeline, run_person_pipeline_by_orcid
from utils.stream_capture import (
    ThreadSafeConsoleBuffer,
    capture_console_output,
    drain_console_buffer,
)


REDIS_URL = get_redis_url()
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

celery_app = Celery("deepsearch", broker=REDIS_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1")),
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "86400")),
    timezone=os.getenv("TZ", "UTC"),
)


def _serialize_person_result(result: Any) -> Dict[str, Any]:
    return {
        "person_name": result.person_name,
        "organization": result.organization,
        "report": result.report,
        "iterations": result.iterations,
        "queries": result.queries,
        "sources": result.sources,
    }


def _serialize_search_result(
    orcid_id: str,
    result_url: Optional[str],
    result_author: Optional[Dict[str, Any]],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    if result_url:
        return {
            "success": True,
            "orcid_id": orcid_id,
            "google_scholar_url": result_url,
            "author_name": result_author.get("name") if result_author else None,
            "affiliation": result_author.get("affiliation") if result_author else None,
            "match_count": result_author.get("match_count") if result_author else None,
            "error": None,
        }
    return {
        "success": False,
        "orcid_id": orcid_id,
        "google_scholar_url": None,
        "author_name": None,
        "affiliation": None,
        "match_count": None,
        "error": error or "未找到匹配的 Google Scholar 账号",
    }


def _drain_logs(job_id: str, log_buffer: ThreadSafeConsoleBuffer, state: Dict[str, Any]) -> None:
    store = get_job_store()
    lines, state["last_pos"], state["remainder"] = drain_console_buffer(
        log_buffer,
        state["last_pos"],
        state["remainder"],
    )
    for line in lines:
        store.append_log(job_id, line)


def _run_with_streamed_logs(
    job_id: str,
    celery_task_id: Optional[str],
    start_message: str,
    runner: Callable[[], Dict[str, Any]],
    result_tag: Callable[[Dict[str, Any]], str],
) -> Dict[str, Any]:
    store = get_job_store()
    store.set_running(job_id, celery_task_id=celery_task_id)
    store.append_event(job_id, "[START]", message=start_message)

    log_buffer = ThreadSafeConsoleBuffer()
    state = {"last_pos": 0, "remainder": ""}

    try:
        def wrapped_runner():
            with capture_console_output(log_buffer):
                return runner()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wrapped_runner)
            while not future.done():
                time.sleep(0.5)
                _drain_logs(job_id, log_buffer, state)
            result = future.result()

        _drain_logs(job_id, log_buffer, state)
        if state["remainder"]:
            store.append_log(job_id, state["remainder"])

        store.set_success(job_id, result)
        store.append_event(job_id, result_tag(result), payload=result)
        store.append_event(job_id, "[END]")
        return result
    except Exception as exc:
        _drain_logs(job_id, log_buffer, state)
        if state["remainder"]:
            store.append_log(job_id, state["remainder"])
        error_result = {"success": False, "error": str(exc)}
        store.set_failed(job_id, str(exc), result=error_result)
        store.append_event(job_id, "[ERROR]", payload=error_result)
        store.append_event(job_id, "[END]")
        raise


@celery_app.task(name="deepsearch.orcid_find", bind=True)
def orcid_find_task(self, job_id: str, orcid_id: str) -> Dict[str, Any]:
    def runner() -> Dict[str, Any]:
        result_url, result_author = find_google_scholar_by_orcid(
            orcid_id=orcid_id,
            verbose=True,
        )
        return _serialize_search_result(orcid_id, result_url, result_author)

    return _run_with_streamed_logs(
        job_id=job_id,
        celery_task_id=self.request.id,
        start_message=f"开始查找 ORCID: {orcid_id}",
        runner=runner,
        result_tag=lambda result: "[RESULT]" if result.get("success") else "[ERROR]",
    )


@celery_app.task(name="deepsearch.person_report_gs", bind=True)
def person_report_gs_task(self, job_id: str, google_scholar_url: str) -> Dict[str, Any]:
    def runner() -> Dict[str, Any]:
        return _serialize_person_result(
            run_person_pipeline(google_scholar_url=google_scholar_url)
        )

    return _run_with_streamed_logs(
        job_id=job_id,
        celery_task_id=self.request.id,
        start_message=f"开始生成 Google Scholar 报告: {google_scholar_url}",
        runner=runner,
        result_tag=lambda result: "[ERROR]" if is_failure_report(result.get("report")) else "[RESULT]",
    )


@celery_app.task(name="deepsearch.person_report_orcid", bind=True)
def person_report_orcid_task(self, job_id: str, orcid_id: str) -> Dict[str, Any]:
    def runner() -> Dict[str, Any]:
        return _serialize_person_result(run_person_pipeline_by_orcid(orcid_id=orcid_id))

    return _run_with_streamed_logs(
        job_id=job_id,
        celery_task_id=self.request.id,
        start_message=f"开始生成 ORCID 报告: {orcid_id}",
        runner=runner,
        result_tag=lambda result: "[ERROR]" if is_failure_report(result.get("report")) else "[RESULT]",
    )


def submit_orcid_find(orcid_id: str) -> str:
    store = get_job_store()
    job_id = store.create_job("orcid_find", {"orcid_id": orcid_id})
    try:
        async_result = orcid_find_task.delay(job_id, orcid_id)
    except Exception as exc:
        store.set_failed(job_id, f"任务提交失败: {exc}")
        raise
    store.set_dispatched(job_id, celery_task_id=async_result.id)
    return job_id


def submit_person_report_gs(google_scholar_url: str) -> str:
    store = get_job_store()
    job_id = store.create_job(
        "person_report_gs",
        {"google_scholar_url": google_scholar_url},
    )
    try:
        async_result = person_report_gs_task.delay(job_id, google_scholar_url)
    except Exception as exc:
        store.set_failed(job_id, f"任务提交失败: {exc}")
        raise
    store.set_dispatched(job_id, celery_task_id=async_result.id)
    return job_id


def submit_person_report_orcid(orcid_id: str) -> str:
    store = get_job_store()
    job_id = store.create_job("person_report_orcid", {"orcid_id": orcid_id})
    try:
        async_result = person_report_orcid_task.delay(job_id, orcid_id)
    except Exception as exc:
        store.set_failed(job_id, f"任务提交失败: {exc}")
        raise
    store.set_dispatched(job_id, celery_task_id=async_result.id)
    return job_id
