"""
LLM API 调用模块

调用云端 LLM API（OpenAI、Claude 等）
内部使用流式请求防止超时，外部返回完整文本

支持同步和异步两种模式
"""

import sys
import json
import yaml
import asyncio
import requests
import aiohttp
import time
from pathlib import Path
from typing import Optional, Dict, Any, Generator, List, AsyncGenerator

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入日志
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_chat_completion_payload(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> Dict[str, Any]:
    """
    构建 OpenAI 兼容 chat/completions 请求体。

    GPT-5 系列要求使用 `max_completion_tokens`，老模型仍保持 `max_tokens`
    以兼容更多 OpenAI 兼容中转站。
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }

    token_field = "max_completion_tokens" if (model or "").strip().lower().startswith("gpt-5") else "max_tokens"
    payload[token_field] = max_tokens
    return payload


def _content_length(content: Any) -> int:
    """估算消息内容长度（按字符数）"""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += _content_length(item.get("text") or item.get("content"))
            else:
                total += _content_length(item)
        return total
    return len(str(content))


def _messages_length(messages: List[Dict[str, Any]]) -> int:
    """估算 messages 总长度（按字符数）"""
    return sum(_content_length(message.get("content")) for message in messages if isinstance(message, dict))


class LLMApiClient:
    """
    云端 LLM API 客户端
    
    支持 OpenAI 兼容的 API 格式
    内部流式请求，外部返回完整文本
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        初始化 API 客户端
        
        Args:
            api_key: API Key（不提供则从配置读取）
            api_base: API Base URL（不提供则从配置读取）
            model: 模型名称（不提供则从配置读取）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            verbose: 是否打印详细日志
        """
        config = _load_config()
        llm_config = config.get("llm", {}).get("api", {})
        
        self.api_key = api_key or llm_config.get("api_key", "")
        self.api_base = api_base or llm_config.get("api_base", "https://api.openai.com/v1")
        self.model = model or llm_config.get("model", "gpt-4o-mini")
        self.timeout = timeout if timeout is not None else llm_config.get("timeout", 120)
        self.stream_gap_timeout = llm_config.get("stream_gap_timeout", self.timeout)
        self.max_retries = max_retries if max_retries is not None else llm_config.get("max_retries", 3)
        self.verbose = verbose
        
        # 移除末尾的斜杠
        self.api_base = self.api_base.rstrip("/")
        
        if self.verbose:
            print(f"[LLM API] 初始化: {self.api_base}, 模型: {self.model}")
    
    def _stream_request(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 20480,
    ) -> Generator[str, None, None]:
        """
        流式请求（内部使用）
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            
        Yields:
            文本片段
        """
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = _build_chat_completion_payload(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        if self.verbose:
            print(f"[LLM API] 请求: {url}")
            print(f"[LLM API] 模型: {self.model}")
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_text = line.decode("utf-8")
                
                # 跳过非数据行
                if not line_text.startswith("data: "):
                    continue
                
                data_str = line_text[6:]  # 去掉 "data: " 前缀
                
                # 结束标记
                if data_str == "[DONE]":
                    break
                
                try:
                    import json
                    chunk = json.loads(data_str)
                    
                    # 提取内容
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                            
                except json.JSONDecodeError:
                    continue
                    
        except requests.exceptions.Timeout:
            logger.error(f"LLM API 请求超时: {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API 请求失败: {e}")
            raise
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 20480,
    ) -> str:
        """
        发送查询并返回完整响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            
        Returns:
            完整的响应文本
        """
        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.verbose:
            print(f"[LLM API] 查询: {prompt[:50]}...")
        
        # 流式请求并收集响应
        full_response = []
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                for chunk in self._stream_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    full_response.append(chunk)
                    if self.verbose:
                        print(chunk, end="", flush=True)
                
                if self.verbose:
                    print()  # 换行
                
                result = "".join(full_response)
                logger.info(
                    "LLM API 查询成功: model=%s, request_chars=%s, response_chars=%s",
                    self.model,
                    _messages_length(messages),
                    len(result),
                )
                return result
                
            except Exception as e:
                last_exception = e
                full_response = []  # 重置
                
                if attempt < self.max_retries - 1:
                    if self.verbose:
                        print(f"[LLM API] 重试 {attempt + 1}/{self.max_retries}: {e}")
                    logger.warning(f"LLM API 重试 {attempt + 1}: {e}")
                    continue
                else:
                    logger.error(f"LLM API 查询失败: {e}")
                    raise last_exception
        
        return ""
    
    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 20480,
    ) -> str:
        """
        多轮对话
        
        Args:
            messages: 消息列表，格式 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            
        Returns:
            完整的响应文本
        """
        if self.verbose:
            print(f"[LLM API] 对话: {len(messages)} 条消息")
        
        # 流式请求并收集响应
        full_response = []
        
        for chunk in self._stream_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            full_response.append(chunk)
            if self.verbose:
                print(chunk, end="", flush=True)
        
        if self.verbose:
            print()
        
        result = "".join(full_response)
        logger.info(
            "LLM API 对话成功: model=%s, request_chars=%s, response_chars=%s",
            self.model,
            _messages_length(messages),
            len(result),
        )
        return result


# ============ 异步客户端 ============

class LLMApiClientAsync:
    """
    云端 LLM API 异步客户端
    
    支持高并发请求
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        verbose: bool = False,
    ):
        """初始化异步 API 客户端"""
        config = _load_config()
        llm_config = config.get("llm", {}).get("api", {})
        
        self.api_key = api_key or llm_config.get("api_key", "")
        self.api_base = api_base or llm_config.get("api_base", "https://api.openai.com/v1")
        self.model = model or llm_config.get("model", "gpt-4o-mini")
        self.timeout = timeout if timeout is not None else llm_config.get("timeout", 120)
        self.stream_gap_timeout = llm_config.get("stream_gap_timeout", self.timeout)
        self.max_retries = max_retries if max_retries is not None else llm_config.get("max_retries", 3)
        self.verbose = verbose
        
        self.api_base = self.api_base.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP session"""
        if self._session is None or self._session.closed:
            # Streaming requests can legitimately run for several minutes on
            # long final-report prompts. Do not cap total response time; only
            # time out connection setup or long gaps between streamed chunks.
            timeout = aiohttp.ClientTimeout(
                total=None,
                sock_connect=self.timeout,
                sock_read=None,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """关闭 session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _stream_request(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 20480,
    ) -> AsyncGenerator[str, None]:
        """异步流式请求"""
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = _build_chat_completion_payload(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        if self.verbose:
            print(f"[LLM API Async] 请求: {url}")
        
        session = await self._get_session()
        
        started_at = time.time()
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            first_chunk_at = None
            first_reasoning_at = None
            first_content_at = None
            reasoning_chars = 0
            content_chars = 0
            data_lines = 0
            last_progress_at = started_at
            logger.info(
                "LLM API Async 响应 headers: model=%s, headers_seconds=%.2f",
                self.model,
                time.time() - started_at,
            )
            
            while True:
                read_timeout = self.timeout if first_chunk_at is None else self.stream_gap_timeout
                if first_chunk_at is None:
                    phase = "首个流式 chunk"
                else:
                    phase = "下一个流式 chunk"

                try:
                    line = await asyncio.wait_for(
                        response.content.readline(),
                        timeout=read_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"LLM API Async 等待{phase}超时: {read_timeout}s"
                    ) from exc

                if not line:
                    break
                if first_chunk_at is None:
                    first_chunk_at = time.time()
                    logger.info(
                        "LLM API Async 首个流式 chunk: model=%s, first_chunk_seconds=%.2f",
                        self.model,
                        first_chunk_at - started_at,
                    )
                
                line_text = line.decode("utf-8").strip()
                
                if not line_text.startswith("data: "):
                    continue
                
                data_str = line_text[6:]
                
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        data_lines += 1
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        content = delta.get("content", "")
                        if reasoning:
                            reasoning_chars += len(reasoning)
                            if first_reasoning_at is None:
                                first_reasoning_at = time.time()
                                logger.info(
                                    "LLM API Async 首个推理 token: model=%s, first_reasoning_seconds=%.2f",
                                    self.model,
                                    first_reasoning_at - started_at,
                                )
                        if content:
                            content_chars += len(content)
                            if first_content_at is None:
                                first_content_at = time.time()
                                logger.info(
                                    "LLM API Async 首个内容 token: model=%s, first_content_seconds=%.2f",
                                    self.model,
                                    first_content_at - started_at,
                                )
                            yield content
                        now = time.time()
                        if data_lines % 100 == 0 or now - last_progress_at >= 60:
                            last_progress_at = now
                            logger.info(
                                "LLM API Async 流式进度: model=%s, elapsed=%.2f, data_lines=%s, reasoning_chars=%s, content_chars=%s",
                                self.model,
                                now - started_at,
                                data_lines,
                                reasoning_chars,
                                content_chars,
                            )
                except json.JSONDecodeError:
                    continue
    
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 20480,
    ) -> str:
        """异步查询"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.verbose:
            print(f"[LLM API Async] 查询: {prompt[:50]}...")
        
        full_response = []
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                async for chunk in self._stream_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    full_response.append(chunk)
                    if self.verbose:
                        print(chunk, end="", flush=True)
                
                if self.verbose:
                    print()
                
                result = "".join(full_response)
                logger.info(
                    "LLM API Async 查询成功: model=%s, request_chars=%s, response_chars=%s",
                    self.model,
                    _messages_length(messages),
                    len(result),
                )
                return result
                
            except Exception as e:
                last_exception = e
                full_response = []
                
                if attempt < self.max_retries - 1:
                    await self.close()
                    logger.warning(
                        "LLM API Async 重试 %s/%s: %s: %r",
                        attempt + 1,
                        self.max_retries,
                        type(e).__name__,
                        e,
                    )
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(
                        "LLM API Async 查询失败: %s: %r",
                        type(e).__name__,
                        e,
                        exc_info=True,
                    )
                    raise last_exception
        
        return ""
    
    async def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 20480,
    ) -> str:
        """异步多轮对话"""
        full_response = []
        
        async for chunk in self._stream_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            full_response.append(chunk)
            if self.verbose:
                print(chunk, end="", flush=True)
        
        if self.verbose:
            print()
        
        result = "".join(full_response)
        logger.info(
            "LLM API Async 对话成功: model=%s, request_chars=%s, response_chars=%s",
            self.model,
            _messages_length(messages),
            len(result),
        )
        return result
    
    async def batch_query(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 20480,
        max_concurrency: int = 5,
    ) -> List[str]:
        """
        批量并发查询
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示
            temperature: 温度
            max_tokens: 最大 token
            max_concurrency: 最大并发数
            
        Returns:
            响应列表（顺序与输入一致）
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def query_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.query(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        
        if self.verbose:
            print(f"[LLM API Async] 批量查询: {len(prompts)} 个, 并发: {max_concurrency}")
        
        tasks = [query_with_semaphore(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量查询第 {i} 个失败: {result}")
                final_results.append("")
            else:
                final_results.append(result)
        
        logger.info(f"LLM API Async 批量查询完成: {len(prompts)} 个")
        return final_results


# ============ 便捷函数 ============

def query(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    verbose: bool = False,
) -> str:
    """便捷函数：同步查询"""
    client = LLMApiClient(verbose=verbose)
    return client.query(prompt, system_prompt=system_prompt, temperature=temperature)


async def query_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    verbose: bool = False,
) -> str:
    """便捷函数：异步查询"""
    async with LLMApiClientAsync(verbose=verbose) as client:
        return await client.query(prompt, system_prompt=system_prompt, temperature=temperature)


async def batch_query_async(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_concurrency: int = 5,
    verbose: bool = False,
) -> List[str]:
    """便捷函数：批量并发查询"""
    async with LLMApiClientAsync(verbose=verbose) as client:
        return await client.batch_query(
            prompts=prompts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_concurrency=max_concurrency,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("LLM API 测试")
    print("=" * 60)
    
    # 测试
    try:
        response = query(
            prompt="你好，请简单介绍一下你自己。",
            verbose=True,
        )
        print(f"\n响应长度: {len(response)} 字符")
    except Exception as e:
        print(f"测试失败: {e}")

