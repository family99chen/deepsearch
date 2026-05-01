"""
浏览器资源调度器。

目标：
- 限制全站同时运行的浏览器型 worker 数量，避免瞬时爆发
- 对同一域名做额外并发限制，降低单域名抖动和反爬压力

说明：
- 当前 deepsearch 以单主进程方式在 PM2 下运行，因此这里使用进程内
  线程同步原语即可覆盖所有请求发起的 worker 子进程调度。
- 这里调度的是“启动 worker 子进程的时机”，不改变 worker 内 fresh
  Chrome / fresh 预热 / 新窗口等抓取策略。
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from urllib.parse import urlparse

import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
WAIT_POLL_SECONDS = 0.25


def _load_scheduler_config() -> Tuple[int, int, Dict[str, int]]:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}

    scheduler = data.get("browser_scheduler", {}) or {}
    global_slots = int(scheduler.get("global_slots", 8) or 0)
    default_domain_slots = int(scheduler.get("default_domain_slots", 2) or 0)
    raw_limits = scheduler.get("domain_limits", {}) or {}

    domain_limits: Dict[str, int] = {}
    if isinstance(raw_limits, dict):
        for domain, value in raw_limits.items():
            try:
                limit = int(value)
            except (TypeError, ValueError):
                continue
            domain_key = str(domain).strip().lower()
            if domain_key:
                domain_limits[domain_key] = max(limit, 0)

    return max(global_slots, 0), max(default_domain_slots, 0), domain_limits


def _normalize_domain(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.netloc or "").strip().lower()


class BrowserResourceController:
    def __init__(self):
        self.global_slots, self.default_domain_slots, self.domain_limits = _load_scheduler_config()
        self._global_semaphore = (
            threading.Semaphore(self.global_slots) if self.global_slots > 0 else None
        )
        self._domain_lock = threading.Lock()
        self._domain_semaphores: Dict[str, threading.Semaphore] = {}
        self._domain_capacities: Dict[str, int] = {}

    def _emit(self, message: str, verbose: bool):
        if verbose:
            print(message, flush=True)

    def _resolve_domain_limit(self, domain: str) -> int:
        if not domain:
            return 0

        if domain in self.domain_limits:
            return self.domain_limits[domain]

        for configured_domain, limit in sorted(
            self.domain_limits.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if domain == configured_domain or domain.endswith(f".{configured_domain}"):
                return limit

        return max(self.default_domain_slots, 0)

    def _get_domain_semaphore(self, domain: str) -> Tuple[Optional[threading.Semaphore], int]:
        limit = self._resolve_domain_limit(domain)
        if limit <= 0:
            return None, 0

        with self._domain_lock:
            semaphore = self._domain_semaphores.get(domain)
            if semaphore is None:
                semaphore = threading.Semaphore(limit)
                self._domain_semaphores[domain] = semaphore
                self._domain_capacities[domain] = limit
            return semaphore, self._domain_capacities[domain]

    def _acquire_with_wait_log(
        self,
        semaphore: Optional[threading.Semaphore],
        label: str,
        limit: int,
        verbose: bool,
    ) -> bool:
        if semaphore is None or limit <= 0:
            return False

        wait_started_at = time.monotonic()
        wait_logged = False

        while True:
            if semaphore.acquire(timeout=WAIT_POLL_SECONDS):
                waited = time.monotonic() - wait_started_at
                if wait_logged:
                    self._emit(
                        f"[BrowserScheduler] 获得 {label} (limit={limit})，等待 {waited:.1f}s",
                        verbose,
                    )
                return True

            if not wait_logged:
                self._emit(
                    f"[BrowserScheduler] 等待 {label} (limit={limit})",
                    verbose,
                )
                wait_logged = True

    @contextmanager
    def acquire(self, url: str, verbose: bool = False) -> Iterator[None]:
        domain = _normalize_domain(url)
        domain_semaphore, domain_limit = self._get_domain_semaphore(domain)
        acquired_domain = self._acquire_with_wait_log(
            domain_semaphore,
            f"域名槽位 {domain}" if domain else "域名槽位",
            domain_limit,
            verbose,
        )

        acquired_global = False
        try:
            acquired_global = self._acquire_with_wait_log(
                self._global_semaphore,
                "全局 browser 槽位",
                self.global_slots,
                verbose,
            )
            yield
        finally:
            if acquired_global and self._global_semaphore is not None:
                self._global_semaphore.release()
            if acquired_domain and domain_semaphore is not None:
                domain_semaphore.release()


_CONTROLLER = BrowserResourceController()


@contextmanager
def browser_resource_context(url: str, verbose: bool = False) -> Iterator[None]:
    with _CONTROLLER.acquire(url, verbose=verbose):
        yield
