"""
代理配置管理模块
从 config.yaml 或环境变量读取代理设置
"""

import os
import re
import yaml
import requests
from pathlib import Path
from typing import Optional, Dict, List


def _load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _get_proxy_config() -> dict:
    """获取代理配置"""
    config = _load_config()
    return config.get("proxy", {})


def is_proxy_enabled() -> bool:
    """
    检查代理是否启用
    
    优先级：环境变量 > config.yaml
    """
    # 环境变量优先
    env_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
    if env_proxy:
        return True
    
    # 检查配置文件
    proxy_config = _get_proxy_config()
    return proxy_config.get("enabled", False) and proxy_config.get("url")


def get_proxy() -> Optional[str]:
    """
    获取代理 URL
    
    优先级：环境变量 > config.yaml
    
    Returns:
        代理 URL 或 None
    """
    # 环境变量优先
    env_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if env_proxy:
        return env_proxy
    
    # 从配置文件读取
    proxy_config = _get_proxy_config()
    if proxy_config.get("enabled", False):
        return proxy_config.get("url") or None
    
    return None


def get_proxy_dict() -> Optional[Dict[str, str]]:
    """
    获取 requests 使用的代理字典
    
    Returns:
        {"http": "...", "https": "..."} 或 None
    """
    proxy_url = get_proxy()
    if proxy_url:
        return {
            "http": proxy_url,
            "https": proxy_url
        }
    return None


def configure_session(session: requests.Session) -> requests.Session:
    """
    配置 requests Session 的代理
    
    Args:
        session: requests.Session 对象
        
    Returns:
        配置好的 session
    """
    proxy_dict = get_proxy_dict()
    if proxy_dict:
        session.proxies = proxy_dict
        print(f"[PROXY] 已配置代理: {mask_proxy_url(get_proxy())}")
    return session


def get_selenium_proxy_args() -> List[str]:
    """
    获取 Selenium Chrome 的代理参数
    
    Returns:
        Chrome 启动参数列表
    """
    proxy_url = get_proxy()
    if proxy_url:
        return [f"--proxy-server={proxy_url}"]
    return []


def mask_proxy_url(proxy_url: str) -> str:
    """
    隐藏代理 URL 中的密码
    
    Args:
        proxy_url: 代理 URL
        
    Returns:
        隐藏密码后的 URL
    """
    if not proxy_url:
        return ""
    return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', proxy_url)


def get_current_ip(use_proxy: bool = True) -> Optional[str]:
    """
    获取当前公网 IP
    
    Args:
        use_proxy: 是否使用代理获取
        
    Returns:
        公网 IP 地址
    """
    try:
        proxies = get_proxy_dict() if use_proxy else None
        response = requests.get(
            "https://api.ipify.org",
            proxies=proxies,
            timeout=10
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] 获取 IP 失败: {e}")
        return None


def verify_proxy() -> bool:
    """
    验证代理是否正常工作
    
    Returns:
        代理是否可用
    """
    proxy_url = get_proxy()
    if not proxy_url:
        print("[INFO] 未配置代理")
        return True
    
    print(f"[INFO] 验证代理: {mask_proxy_url(proxy_url)}")
    
    try:
        # 不使用代理获取真实 IP
        real_ip = get_current_ip(use_proxy=False)
        print(f"[INFO] 真实 IP: {real_ip}")
        
        # 使用代理获取 IP
        proxy_ip = get_current_ip(use_proxy=True)
        print(f"[INFO] 代理 IP: {proxy_ip}")
        
        if proxy_ip and proxy_ip != real_ip:
            print("[SUCCESS] 代理验证成功！IP 已更改")
            return True
        elif proxy_ip == real_ip:
            print("[WARNING] 代理未生效，IP 未改变")
            return False
        else:
            print("[ERROR] 无法通过代理连接")
            return False
            
    except Exception as e:
        print(f"[ERROR] 代理验证失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("代理配置测试")
    print("=" * 50)
    print()
    
    print(f"代理启用: {is_proxy_enabled()}")
    print(f"代理 URL: {mask_proxy_url(get_proxy() or '未配置')}")
    print()
    
    print("验证代理...")
    verify_proxy()

