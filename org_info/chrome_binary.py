"""
Chrome binary resolution helpers.

Resolution order:
1. Explicit environment variables
2. Common system browser locations in PATH
3. Selenium Manager managed Chrome for Testing download/cache
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

try:
    from utils.logger import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


CHROME_ENV_VARS = (
    "CHROME_BINARY_PATH",
    "GOOGLE_CHROME_BIN",
    "CHROME_BIN",
)

SYSTEM_CHROME_CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)

_resolved_chrome_binary: Optional[str] = None
_resolve_lock = threading.Lock()


def _normalize_browser_path(path: Optional[object]) -> Optional[str]:
    if path is None:
        return None

    value = str(path).strip()
    if not value:
        return None

    browser_path = Path(value)
    if browser_path.is_file() and os.access(browser_path, os.X_OK):
        return str(browser_path)
    return None


def _resolve_from_env() -> Optional[Tuple[str, str]]:
    for env_var in CHROME_ENV_VARS:
        raw_value = os.environ.get(env_var)
        if not raw_value:
            continue

        browser_path = _normalize_browser_path(raw_value)
        if browser_path:
            return browser_path, f"env:{env_var}"

        logger.warning("环境变量 %s 指向的 Chrome 路径无效: %s", env_var, raw_value)

    return None


def _resolve_from_system_path() -> Optional[Tuple[str, str]]:
    for candidate in SYSTEM_CHROME_CANDIDATES:
        browser_path = _normalize_browser_path(shutil.which(candidate))
        if browser_path:
            return browser_path, f"system:{candidate}"

    return None


def _resolve_from_selenium_manager() -> Optional[Tuple[str, str]]:
    try:
        from selenium.webdriver.common.selenium_manager import SeleniumManager

        result = SeleniumManager().binary_paths(["--browser", "chrome"])
        browser_path = _normalize_browser_path(result.get("browser_path"))
        if browser_path:
            return browser_path, "selenium-manager"

        logger.warning("Selenium Manager 返回了无效的 browser_path: %r", result.get("browser_path"))
    except Exception as exc:
        logger.warning("Selenium Manager 获取 Chrome 失败: %s", exc)

    return None


def resolve_chrome_binary_path() -> str:
    """Resolve a usable Chrome binary path, downloading if needed."""
    global _resolved_chrome_binary

    with _resolve_lock:
        cached_path = _normalize_browser_path(_resolved_chrome_binary)
        if cached_path:
            return cached_path

        for resolver in (
            _resolve_from_env,
            _resolve_from_system_path,
            _resolve_from_selenium_manager,
        ):
            resolved = resolver()
            if resolved:
                browser_path, source = resolved
                _resolved_chrome_binary = browser_path
                logger.info("使用 Chrome 二进制: %s (%s)", browser_path, source)
                return browser_path

    raise RuntimeError(
        "未找到可用的 Chrome/Chromium 浏览器。"
        "已检查环境变量 CHROME_BINARY_PATH/GOOGLE_CHROME_BIN/CHROME_BIN、"
        "系统 PATH 中的浏览器，以及 Selenium Manager 自动下载的 Chrome for Testing。"
        "请安装 google-chrome/chromium，或配置 CHROME_BINARY_PATH。"
    )


def get_chrome_version_main(browser_path: Optional[str] = None) -> Optional[int]:
    """Read Chrome major version from a resolved executable."""
    chrome_binary = _normalize_browser_path(browser_path) if browser_path else None
    if chrome_binary is None:
        try:
            chrome_binary = resolve_chrome_binary_path()
        except Exception:
            return None

    try:
        result = subprocess.run(
            [chrome_binary, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = (result.stdout or result.stderr).strip()
        for token in output.split():
            if token and token[0].isdigit():
                return int(token.split(".", 1)[0])
    except Exception:
        return None

    return None
