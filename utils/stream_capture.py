"""
按请求隔离控制台输出，避免并发流式请求互相串日志。
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import io
import sys
import threading
from typing import Iterator, List, Optional, Tuple


_CAPTURED_STREAM: contextvars.ContextVar[Optional["ThreadSafeConsoleBuffer"]] = (
    contextvars.ContextVar("deepsearch_captured_stream", default=None)
)
_PATCH_LOCK = threading.Lock()
_THREADPOOL_PATCHED = False
_ORIGINAL_THREADPOOL_SUBMIT = concurrent.futures.ThreadPoolExecutor.submit
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class ThreadSafeConsoleBuffer(io.TextIOBase):
    """线程安全的文本缓冲区，支持按位置增量读取。"""

    def __init__(self):
        super().__init__()
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        text = str(s)
        if not text:
            return 0
        with self._lock:
            self._buffer.seek(0, io.SEEK_END)
            self._buffer.write(text)
        return len(text)

    def flush(self) -> None:
        return None

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def read_from(self, position: int) -> Tuple[str, int]:
        with self._lock:
            self._buffer.seek(position)
            data = self._buffer.read()
            return data, self._buffer.tell()


class RoutedTextStream(io.TextIOBase):
    """根据当前上下文把 stdout/stderr 路由到对应缓冲区。"""

    def __init__(self, fallback: io.TextIOBase):
        super().__init__()
        self._fallback = fallback

    def _current_stream(self) -> io.TextIOBase:
        return _CAPTURED_STREAM.get() or self._fallback

    def write(self, s: str) -> int:
        return self._current_stream().write(s)

    def flush(self) -> None:
        self._current_stream().flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._fallback, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._fallback, "errors", "strict")

    @property
    def buffer(self):
        return getattr(self._fallback, "buffer", None)

    def fileno(self) -> int:
        return self._fallback.fileno()


STDOUT_ROUTER = RoutedTextStream(_ORIGINAL_STDOUT)
STDERR_ROUTER = RoutedTextStream(_ORIGINAL_STDERR)


def _run_with_context(
    ctx: contextvars.Context,
    fn,
    args: tuple,
    kwargs: dict,
):
    return ctx.run(fn, *args, **kwargs)


def _submit_with_context(self, fn, /, *args, **kwargs):
    ctx = contextvars.copy_context()
    return _ORIGINAL_THREADPOOL_SUBMIT(
        self,
        _run_with_context,
        ctx,
        fn,
        args,
        kwargs,
    )


def install_stream_capture() -> None:
    """安装全局 stdout/stderr 路由和线程池上下文继承。"""

    global _THREADPOOL_PATCHED

    with _PATCH_LOCK:
        if sys.stdout is not STDOUT_ROUTER:
            sys.stdout = STDOUT_ROUTER
        if sys.stderr is not STDERR_ROUTER:
            sys.stderr = STDERR_ROUTER

        if not _THREADPOOL_PATCHED:
            concurrent.futures.ThreadPoolExecutor.submit = _submit_with_context
            _THREADPOOL_PATCHED = True


@contextlib.contextmanager
def capture_console_output(
    buffer: Optional[ThreadSafeConsoleBuffer] = None,
) -> Iterator[ThreadSafeConsoleBuffer]:
    """在当前请求上下文中捕获 stdout/stderr。"""

    install_stream_capture()
    target = buffer or ThreadSafeConsoleBuffer()
    token = _CAPTURED_STREAM.set(target)
    try:
        yield target
    finally:
        _CAPTURED_STREAM.reset(token)


def drain_console_buffer(
    buffer: ThreadSafeConsoleBuffer,
    position: int,
    remainder: str = "",
) -> Tuple[List[str], int, str]:
    """从缓冲区按位置读取新增内容，并按行拆分。"""

    new_text, next_position = buffer.read_from(position)
    if not new_text:
        return [], next_position, remainder

    combined = remainder + new_text
    parts = combined.splitlines(keepends=True)
    lines: List[str] = []
    next_remainder = ""

    for part in parts:
        if part.endswith("\n") or part.endswith("\r"):
            line = part.rstrip("\r\n")
            if line:
                lines.append(line)
        else:
            next_remainder = part

    return lines, next_position, next_remainder
