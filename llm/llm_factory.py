"""
LLM 工厂模块

提供统一的 LLM 调用接口
根据配置自动选择使用云端 API 或本地模型

支持同步和异步两种模式
"""

import sys
import yaml
import asyncio
from pathlib import Path
from typing import Optional, Literal, List

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入具体实现
from llm.api import LLMApiClient, LLMApiClientAsync
from llm.local import LLMLocalClient, LLMLocalClientAsync

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


# LLM 类型
LLMType = Literal["api", "local", "auto"]


class LLMFactory:
    """
    LLM 工厂类
    
    提供统一的接口，根据配置自动选择后端
    """
    
    def __init__(
        self,
        backend: Optional[LLMType] = None,
        model: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        初始化 LLM 工厂
        
        Args:
            backend: 后端类型
                - "api": 使用云端 API
                - "local": 使用本地模型
                - "auto": 自动选择（优先本地，本地不可用时使用 API）
                - None: 从配置文件读取
            verbose: 是否打印详细日志
        """
        config = _load_config()
        llm_config = config.get("llm", {})
        
        self.backend = backend or llm_config.get("backend", "api")
        self.model = model
        self.verbose = verbose
        
        # 初始化客户端
        self._api_client: Optional[LLMApiClient] = None
        self._local_client: Optional[LLMLocalClient] = None
        
        if self.verbose:
            print(f"[LLM Factory] 后端: {self.backend}")
        
        logger.info(f"LLM Factory 初始化, 后端: {self.backend}")
    
    def _get_api_client(self) -> LLMApiClient:
        """获取 API 客户端（懒加载）"""
        if self._api_client is None:
            self._api_client = LLMApiClient(model=self.model, verbose=self.verbose)
        return self._api_client
    
    def _get_local_client(self) -> LLMLocalClient:
        """获取本地客户端（懒加载）"""
        if self._local_client is None:
            self._local_client = LLMLocalClient(model=self.model, verbose=self.verbose)
        return self._local_client
    
    def _get_client(self):
        """根据后端类型获取客户端"""
        if self.backend == "api":
            return self._get_api_client()
        elif self.backend == "local":
            return self._get_local_client()
        elif self.backend == "auto":
            # 自动选择：优先本地
            local_client = self._get_local_client()
            if local_client.is_available():
                if self.verbose:
                    print("[LLM Factory] 自动选择: 本地模型")
                return local_client
            else:
                if self.verbose:
                    print("[LLM Factory] 本地不可用，使用 API")
                return self._get_api_client()
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
    
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
        client = self._get_client()
        return client.query(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        多轮对话
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            
        Returns:
            完整的响应文本
        """
        client = self._get_client()
        return client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ============ 异步工厂类 ============

class LLMFactoryAsync:
    """
    LLM 异步工厂类
    
    提供统一的异步接口，支持高并发
    """
    
    def __init__(
        self,
        backend: Optional[LLMType] = None,
        model: Optional[str] = None,
        verbose: bool = False,
    ):
        """初始化异步 LLM 工厂"""
        config = _load_config()
        llm_config = config.get("llm", {})
        
        self.backend = backend or llm_config.get("backend", "api")
        self.model = model
        self.verbose = verbose
        
        self._api_client: Optional[LLMApiClientAsync] = None
        self._local_client: Optional[LLMLocalClientAsync] = None
        self._active_client = None
        
        if self.verbose:
            print(f"[LLM Factory Async] 后端: {self.backend}")
        
        logger.info(f"LLM Factory Async 初始化, 后端: {self.backend}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """关闭所有客户端"""
        if self._api_client:
            await self._api_client.close()
        if self._local_client:
            await self._local_client.close()
    
    def _get_api_client(self) -> LLMApiClientAsync:
        """获取 API 客户端"""
        if self._api_client is None:
            self._api_client = LLMApiClientAsync(model=self.model, verbose=self.verbose)
        return self._api_client
    
    def _get_local_client(self) -> LLMLocalClientAsync:
        """获取本地客户端"""
        if self._local_client is None:
            self._local_client = LLMLocalClientAsync(model=self.model, verbose=self.verbose)
        return self._local_client
    
    async def _get_client(self):
        """根据后端类型获取客户端"""
        if self.backend == "api":
            return self._get_api_client()
        elif self.backend == "local":
            return self._get_local_client()
        elif self.backend == "auto":
            local_client = self._get_local_client()
            if await local_client.is_available():
                if self.verbose:
                    print("[LLM Factory Async] 自动选择: 本地模型")
                return local_client
            else:
                if self.verbose:
                    print("[LLM Factory Async] 本地不可用，使用 API")
                return self._get_api_client()
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
    
    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """异步查询"""
        client = await self._get_client()
        return await client.query(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    async def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """异步多轮对话"""
        client = await self._get_client()
        return await client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
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
            响应列表
        """
        client = await self._get_client()
        return await client.batch_query(
            prompts=prompts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
        )


# ============ 便捷函数 ============

# 默认工厂实例（懒加载）
_default_factory: Optional[LLMFactory] = None


def _get_default_factory(verbose: bool = False) -> LLMFactory:
    """获取默认工厂实例"""
    global _default_factory
    if _default_factory is None:
        _default_factory = LLMFactory(verbose=verbose)
    return _default_factory


def query(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    backend: Optional[LLMType] = None,
    verbose: bool = False,
) -> str:
    """
    便捷函数：发送查询
    
    Args:
        prompt: 用户提示
        system_prompt: 系统提示（可选）
        temperature: 温度参数
        max_tokens: 最大生成 token 数
        backend: 后端类型（不提供则使用默认配置）
        verbose: 是否打印详细日志
        
    Returns:
        完整的响应文本
        
    Example:
        >>> from llm import query
        >>> response = query("请介绍一下 Python")
        >>> print(response)
    """
    if backend or model:
        factory = LLMFactory(backend=backend, model=model, verbose=verbose)
    else:
        factory = _get_default_factory(verbose=verbose)
    
    return factory.query(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def chat(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    backend: Optional[LLMType] = None,
    verbose: bool = False,
) -> str:
    """
    便捷函数：多轮对话
    
    Args:
        messages: 消息列表
        temperature: 温度参数
        max_tokens: 最大生成 token 数
        backend: 后端类型（不提供则使用默认配置）
        verbose: 是否打印详细日志
        
    Returns:
        完整的响应文本
        
    Example:
        >>> from llm import chat
        >>> messages = [
        ...     {"role": "system", "content": "你是一个有帮助的助手"},
        ...     {"role": "user", "content": "你好"},
        ... ]
        >>> response = chat(messages)
    """
    if backend or model:
        factory = LLMFactory(backend=backend, model=model, verbose=verbose)
    else:
        factory = _get_default_factory(verbose=verbose)
    
    return factory.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ============ 异步便捷函数 ============

async def query_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    backend: Optional[LLMType] = None,
    verbose: bool = False,
) -> str:
    """
    异步查询
    
    Example:
        >>> response = await query_async("请介绍一下 Python")
    """
    async with LLMFactoryAsync(backend=backend, model=model, verbose=verbose) as factory:
        return await factory.query(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


async def chat_async(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    backend: Optional[LLMType] = None,
    verbose: bool = False,
) -> str:
    """异步多轮对话"""
    async with LLMFactoryAsync(backend=backend, model=model, verbose=verbose) as factory:
        return await factory.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


async def batch_query_async(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_concurrency: int = 5,
    model: Optional[str] = None,
    backend: Optional[LLMType] = None,
    verbose: bool = False,
) -> List[str]:
    """
    批量并发查询
    
    Args:
        prompts: 提示列表
        system_prompt: 系统提示
        temperature: 温度
        max_tokens: 最大 token
        max_concurrency: 最大并发数
        backend: 后端类型
        verbose: 详细日志
        
    Returns:
        响应列表（顺序与输入一致）
        
    Example:
        >>> prompts = ["介绍 Python", "介绍 JavaScript", "介绍 Go"]
        >>> responses = await batch_query_async(prompts, max_concurrency=3)
    """
    async with LLMFactoryAsync(backend=backend, model=model, verbose=verbose) as factory:
        return await factory.batch_query(
            prompts=prompts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Factory 测试")
    print("=" * 60)
    
    # 测试不同后端
    test_prompt = "请用一句话介绍一下你自己。"
    
    # 使用默认配置
    print("\n[测试] 使用默认配置")
    print("-" * 40)
    try:
        response = query(test_prompt, verbose=True)
        print(f"\n响应: {response}")
    except Exception as e:
        print(f"失败: {e}")
    
    # 测试 auto 模式
    print("\n[测试] Auto 模式")
    print("-" * 40)
    try:
        response = query(test_prompt, backend="auto", verbose=True)
        print(f"\n响应: {response}")
    except Exception as e:
        print(f"失败: {e}")

