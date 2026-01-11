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
        timeout: int = 120,
        max_retries: int = 3,
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
        self.timeout = timeout or llm_config.get("timeout", 120)
        self.max_retries = max_retries
        self.verbose = verbose
        
        # 移除末尾的斜杠
        self.api_base = self.api_base.rstrip("/")
        
        if self.verbose:
            print(f"[LLM API] 初始化: {self.api_base}, 模型: {self.model}")
    
    def _stream_request(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
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
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
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
        max_tokens: int = 4096,
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
                logger.info(f"LLM API 查询成功: {len(result)} 字符")
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
        max_tokens: int = 4096,
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
        logger.info(f"LLM API 对话成功: {len(result)} 字符")
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
        timeout: int = 120,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        """初始化异步 API 客户端"""
        config = _load_config()
        llm_config = config.get("llm", {}).get("api", {})
        
        self.api_key = api_key or llm_config.get("api_key", "")
        self.api_base = api_base or llm_config.get("api_base", "https://api.openai.com/v1")
        self.model = model or llm_config.get("model", "gpt-4o-mini")
        self.timeout = timeout or llm_config.get("timeout", 120)
        self.max_retries = max_retries
        self.verbose = verbose
        
        self.api_base = self.api_base.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
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
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """异步流式请求"""
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        if self.verbose:
            print(f"[LLM API Async] 请求: {url}")
        
        session = await self._get_session()
        
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if not line:
                    continue
                
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
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue
    
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
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
                logger.info(f"LLM API Async 查询成功: {len(result)} 字符")
                return result
                
            except Exception as e:
                last_exception = e
                full_response = []
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"LLM API Async 重试 {attempt + 1}: {e}")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"LLM API Async 查询失败: {e}")
                    raise last_exception
        
        return ""
    
    async def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
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
        
        return "".join(full_response)
    
    async def batch_query(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
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

