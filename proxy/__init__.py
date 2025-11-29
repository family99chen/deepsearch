"""
代理配置模块
提供全局代理设置，供整个项目使用
"""

from .proxy_config import (
    get_proxy,
    get_proxy_dict,
    configure_session,
    get_selenium_proxy_args,
    is_proxy_enabled,
    get_current_ip,
)

__all__ = [
    'get_proxy',
    'get_proxy_dict',
    'configure_session',
    'get_selenium_proxy_args',
    'is_proxy_enabled',
    'get_current_ip',
]

