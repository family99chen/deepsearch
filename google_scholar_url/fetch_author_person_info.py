"""
通过 ORCID API 获取作者的个人信息（姓名等）

支持 MongoDB 缓存，避免重复 API 调用
"""

import sys
import requests
import yaml
from pathlib import Path
from typing import Optional

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入重试模块
try:
    from utils.retry import exponential_backoff, DEFAULT_RETRYABLE_EXCEPTIONS
except ImportError:
    def exponential_backoff(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    DEFAULT_RETRYABLE_EXCEPTIONS = (requests.exceptions.RequestException,)

# 导入缓存模块
try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

# 缓存配置
CACHE_COLLECTION = "orcid_info"
CACHE_TTL = 180 * 24 * 3600  # 6个月（秒）


def load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_cache() -> Optional["MongoCache"]:
    """获取缓存实例"""
    if not HAS_CACHE:
        return None
    try:
        cache = MongoCache(collection_name=CACHE_COLLECTION)
        return cache if cache.is_connected() else None
    except Exception:
        return None


@exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=DEFAULT_RETRYABLE_EXCEPTIONS + (requests.exceptions.RequestException,)
)
def fetch_author_info(orcid_id: str, access_token: Optional[str] = None) -> dict:
    """
    通过 ORCID ID 获取作者的个人信息
    
    Args:
        orcid_id: 作者的 ORCID ID，格式如 "0000-0002-1825-0097"
        access_token: ORCID API access token，如果不提供则从配置文件读取
    
    Returns:
        包含作者个人信息的字典
        
    Raises:
        requests.exceptions.HTTPError: API 请求失败（重试后仍失败）
    """
    config = load_config()
    
    if access_token is None:
        access_token = config["orcid"]["access_token"]
    
    api_base_url = config["orcid"]["api_base_url"]
    url = f"{api_base_url}/{orcid_id}/person"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    return response.json()


def parse_author_name(person_data: dict) -> dict:
    """
    解析作者姓名信息
    
    Args:
        person_data: ORCID API /person 端点返回的数据
    
    Returns:
        包含作者姓名的字典
    """
    name_info = person_data.get("name") or {}
    
    given_names = name_info.get("given-names") or {}
    family_name = name_info.get("family-name") or {}
    credit_name = name_info.get("credit-name") or {}
    
    result = {
        "given_name": given_names.get("value") if isinstance(given_names, dict) else None,
        "family_name": family_name.get("value") if isinstance(family_name, dict) else None,
        "credit_name": credit_name.get("value") if isinstance(credit_name, dict) else None,
        "full_name": None
    }
    
    # 构建完整姓名
    if result["credit_name"]:
        result["full_name"] = result["credit_name"]
    elif result["given_name"] and result["family_name"]:
        result["full_name"] = f"{result['given_name']} {result['family_name']}"
    elif result["family_name"]:
        result["full_name"] = result["family_name"]
    elif result["given_name"]:
        result["full_name"] = result["given_name"]
    
    return result


def get_author_name(orcid_id: str, use_cache: bool = True) -> dict:
    """
    获取作者姓名（支持缓存）
    
    Args:
        orcid_id: 作者的 ORCID ID
        use_cache: 是否使用缓存，默认 True
    
    Returns:
        包含作者姓名的字典:
        - given_name: 名
        - family_name: 姓
        - credit_name: 署名（作者自定义的显示名称）
        - full_name: 完整姓名
    """
    cache_key = orcid_id
    cache = _get_cache() if use_cache else None
    
    # 尝试从缓存读取
    if cache:
        cached_name = cache.get_field(cache_key, "full_name")
        if cached_name is not None:
            # 缓存命中，返回完整数据
            print(f"[INFO] ORCID {orcid_id} 个人信息命中缓存")
            return {
                "given_name": cache.get_field(cache_key, "given_name"),
                "family_name": cache.get_field(cache_key, "family_name"),
                "credit_name": cache.get_field(cache_key, "credit_name"),
                "full_name": cached_name,
            }
    
    # 调用 API 获取数据
    person_data = fetch_author_info(orcid_id)
    result = parse_author_name(person_data)
    
    # 写入缓存（扁平化存储每个字段）
    if cache:
        for field, value in result.items():
            cache.set_field(cache_key, field, value, ttl=CACHE_TTL)
        print(f"[INFO] ORCID {orcid_id} 个人信息已写入缓存")
    
    return result


if __name__ == "__main__":
    # 示例用法
    test_orcid = "0000-0003-3701-8119"
    
    print(f"正在获取 ORCID {test_orcid} 的作者信息...")
    
    cache = _get_cache()
    cache_key = test_orcid
    
    if cache:
        has_cache = cache.has_field(cache_key, "full_name")
        print(f"[缓存] {'命中' if has_cache else '未命中'}")
    
    try:
        author_name = get_author_name(test_orcid)
        
        print("\n=== 作者信息 ===")
        print(f"完整姓名: {author_name['full_name']}")
        print(f"名 (Given Name): {author_name['given_name']}")
        print(f"姓 (Family Name): {author_name['family_name']}")
        if author_name.get('credit_name'):
            print(f"署名 (Credit Name): {author_name['credit_name']}")
        
        if cache:
            print(f"\n[缓存] 数据已写入 collection: {CACHE_COLLECTION}")
            
    except requests.exceptions.HTTPError as e:
        print(f"API 请求失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
