"""Probe local LLM health via OpenAI-compatible /models endpoint."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any, Dict

import yaml

from config import DEEPSEARCH_ROOT


def _load_llm_local_config() -> Dict[str, Any]:
    config_path = DEEPSEARCH_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    llm_cfg = data.get("llm", {}) or {}
    local_cfg = llm_cfg.get("local", {}) or {}
    return {
        "backend": llm_cfg.get("backend", "api"),
        "api_base": local_cfg.get("api_base", "http://localhost:11434/v1"),
        "model": local_cfg.get("model", "unknown"),
    }


def collect_llm_local(timeout: float = 5.0) -> Dict[str, Any]:
    """
    Manual health probe: GET {api_base}/models, HTTP 200 => healthy.
    Reads api_base/model from deepsearch/config.yaml (llm.local).
    """
    cfg = _load_llm_local_config()
    api_base = str(cfg.get("api_base", "http://localhost:11434/v1")).rstrip("/")
    model = cfg.get("model", "unknown")
    backend = cfg.get("backend", "api")
    probe_url = f"{api_base}/models"

    base = {
        "backend": backend,
        "api_base": api_base,
        "model": model,
        "probe_url": probe_url,
    }

    start = time.monotonic()
    try:
        req = urllib.request.Request(probe_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            healthy = status == 200
            return {
                **base,
                "ok": healthy,
                "healthy": healthy,
                "http_status": status,
                "latency_ms": latency_ms,
            }
    except urllib.error.HTTPError as exc:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {
            **base,
            "ok": False,
            "healthy": False,
            "http_status": exc.code,
            "latency_ms": latency_ms,
            "error": str(exc.reason or exc),
        }
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {
            **base,
            "ok": False,
            "healthy": False,
            "http_status": None,
            "latency_ms": latency_ms,
            "error": str(exc),
        }
