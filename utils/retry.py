"""
指数退避重试工具模块
提供通用的重试装饰器和函数
"""

import time
import random
import functools
from typing import Callable, Tuple, Type, Optional, Any
import requests


# 默认可重试的异常类型
DEFAULT_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
    ConnectionError,
    TimeoutError,
)

# 默认可重试的 HTTP 状态码
DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


def exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    retryable_status_codes: Tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
):
    """
    指数退避重试装饰器
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数基数
        jitter: 是否添加随机抖动（防止雷鸣群效应）
        retryable_exceptions: 可重试的异常类型元组
        retryable_status_codes: 可重试的 HTTP 状态码元组
        on_retry: 重试时的回调函数 (exception, attempt, delay)
        
    Returns:
        装饰器函数
        
    Usage:
        @exponential_backoff(max_retries=3)
        def fetch_data():
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # 检查是否是 HTTP 错误且状态码可重试
                    if isinstance(e, requests.exceptions.HTTPError):
                        if hasattr(e, 'response') and e.response is not None:
                            if e.response.status_code not in retryable_status_codes:
                                # 不可重试的状态码，直接抛出
                                raise
                    
                    # 如果已经是最后一次尝试，抛出异常
                    if attempt >= max_retries:
                        raise
                    
                    # 计算延迟时间
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # 添加随机抖动
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    # 调用重试回调
                    if on_retry:
                        on_retry(e, attempt + 1, delay)
                    else:
                        print(f"[RETRY] 第 {attempt + 1}/{max_retries} 次重试，"
                              f"{delay:.1f}秒后重试... 错误: {type(e).__name__}: {e}")
                    
                    time.sleep(delay)
            
            # 不应该到达这里，但以防万一
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_request(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    retryable_status_codes: Tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    **kwargs
) -> Any:
    """
    指数退避重试函数（非装饰器版本）
    
    Args:
        func: 要执行的函数
        *args: 传递给函数的位置参数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数基数
        jitter: 是否添加随机抖动
        retryable_exceptions: 可重试的异常类型元组
        retryable_status_codes: 可重试的 HTTP 状态码元组
        on_retry: 重试时的回调函数
        **kwargs: 传递给函数的关键字参数
        
    Returns:
        函数执行结果
        
    Usage:
        response = retry_request(
            session.get, 
            url, 
            timeout=10,
            max_retries=3
        )
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
            
        except retryable_exceptions as e:
            last_exception = e
            
            # 检查是否是 HTTP 错误且状态码可重试
            if isinstance(e, requests.exceptions.HTTPError):
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code not in retryable_status_codes:
                        raise
            
            if attempt >= max_retries:
                raise
            
            delay = min(
                base_delay * (exponential_base ** attempt),
                max_delay
            )
            
            if jitter:
                delay = delay * (0.5 + random.random())
            
            if on_retry:
                on_retry(e, attempt + 1, delay)
            else:
                print(f"[RETRY] 第 {attempt + 1}/{max_retries} 次重试，"
                      f"{delay:.1f}秒后重试... 错误: {type(e).__name__}: {e}")
            
            time.sleep(delay)
    
    if last_exception:
        raise last_exception


class RetrySession:
    """
    带重试功能的 requests.Session 包装器
    
    Usage:
        session = RetrySession(max_retries=3)
        response = session.get(url, timeout=10)
    """
    
    def __init__(
        self,
        session: Optional[requests.Session] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_status_codes: Tuple[int, ...] = DEFAULT_RETRYABLE_STATUS_CODES,
    ):
        """
        初始化重试 Session
        
        Args:
            session: 已有的 requests.Session，如果不提供则创建新的
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_base: 指数基数
            jitter: 是否添加随机抖动
            retryable_status_codes: 可重试的 HTTP 状态码
        """
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes
    
    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """执行带重试的请求"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = getattr(self.session, method)(url, **kwargs)
                
                # 检查状态码是否需要重试
                if response.status_code in self.retryable_status_codes:
                    if attempt < self.max_retries:
                        raise requests.exceptions.HTTPError(
                            f"HTTP {response.status_code}",
                            response=response
                        )
                
                return response
                
            except DEFAULT_RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                
                if attempt >= self.max_retries:
                    raise
                
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                if self.jitter:
                    delay = delay * (0.5 + random.random())
                
                print(f"[RETRY] 第 {attempt + 1}/{self.max_retries} 次重试，"
                      f"{delay:.1f}秒后重试... 错误: {type(e).__name__}")
                
                time.sleep(delay)
        
        if last_exception:
            raise last_exception
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET 请求"""
        return self._request_with_retry('get', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """POST 请求"""
        return self._request_with_retry('post', url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """PUT 请求"""
        return self._request_with_retry('put', url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """DELETE 请求"""
        return self._request_with_retry('delete', url, **kwargs)
    
    # 代理 session 的其他常用属性和方法
    @property
    def headers(self):
        return self.session.headers
    
    @property
    def cookies(self):
        return self.session.cookies
    
    @property
    def proxies(self):
        return self.session.proxies
    
    @proxies.setter
    def proxies(self, value):
        self.session.proxies = value


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("指数退避重试工具测试")
    print("=" * 60)
    
    # 测试装饰器
    @exponential_backoff(max_retries=3, base_delay=0.5)
    def test_func():
        print("尝试请求...")
        raise requests.exceptions.ConnectionError("模拟连接错误")
    
    try:
        test_func()
    except requests.exceptions.ConnectionError:
        print("[预期] 最终失败")
    
    print()
    print("测试完成")

