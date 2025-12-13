"""
通过 ORCID API 获取作者的论文列表

支持 MongoDB 缓存，避免重复 API 调用
"""

import sys
import requests
import yaml
from pathlib import Path
from typing import Optional, List

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
def fetch_author_works(orcid_id: str, access_token: Optional[str] = None) -> dict:
    """
    通过 ORCID ID 获取作者的论文列表
    
    Args:
        orcid_id: 作者的 ORCID ID，格式如 "0000-0002-1825-0097"
        access_token: ORCID API access token，如果不提供则从配置文件读取
    
    Returns:
        包含作者论文信息的字典
        
    Raises:
        requests.exceptions.HTTPError: API 请求失败（重试后仍失败）
    """
    config = load_config()
    
    if access_token is None:
        access_token = config["orcid"]["access_token"]
    
    api_base_url = config["orcid"]["api_base_url"]
    url = f"{api_base_url}/{orcid_id}/works"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    return response.json()


def parse_paper_list(works_data: dict) -> List[dict]:
    """
    解析 ORCID API 返回的论文数据
    
    Args:
        works_data: ORCID API 返回的原始数据
    
    Returns:
        解析后的论文列表，每个论文包含标题、DOI、发表年份等信息
    """
    papers = []
    
    groups = works_data.get("group", []) or []
    
    for group in groups:
        work_summaries = group.get("work-summary", []) or []
        if not work_summaries:
            continue
        
        # 取第一个 work-summary 作为代表
        work = work_summaries[0]
        
        paper = {
            "title": None,
            "doi": None,
            "publication_year": None,
            "journal": None,
            "external_ids": [],
            "put_code": work.get("put-code")
        }
        
        # 提取标题 (安全处理 None 值)
        title_info = work.get("title") or {}
        title_obj = title_info.get("title") or {}
        if isinstance(title_obj, dict):
            paper["title"] = title_obj.get("value")
        
        # 提取发表年份 (安全处理 None 值)
        pub_date = work.get("publication-date") or {}
        year_obj = pub_date.get("year") or {}
        if isinstance(year_obj, dict):
            paper["publication_year"] = year_obj.get("value")
        
        # 提取期刊名称 (安全处理 None 值)
        journal_title = work.get("journal-title") or {}
        if isinstance(journal_title, dict):
            paper["journal"] = journal_title.get("value")
        
        # 提取外部标识符 (DOI, PMID 等)
        external_ids_info = work.get("external-ids") or {}
        external_ids = external_ids_info.get("external-id", []) or []
        
        for ext_id in external_ids:
            if not ext_id:
                continue
            id_type = ext_id.get("external-id-type")
            id_value = ext_id.get("external-id-value")
            
            paper["external_ids"].append({
                "type": id_type,
                "value": id_value
            })
            
            if id_type == "doi":
                paper["doi"] = id_value
        
        papers.append(paper)
    
    return papers


def get_author_papers(orcid_id: str, use_cache: bool = True) -> List[dict]:
    """
    获取并解析作者的论文列表（支持缓存）
    
    Args:
        orcid_id: 作者的 ORCID ID
        use_cache: 是否使用缓存，默认 True
    
    Returns:
        解析后的论文列表
    """
    cache_key = orcid_id
    cache = _get_cache() if use_cache else None
    
    # 尝试从缓存读取
    if cache:
        cached_papers = cache.get_field(cache_key, "papers")
        if cached_papers is not None:
            print(f"[INFO] ORCID {orcid_id} 论文列表命中缓存 ({len(cached_papers)} 篇)")
            return cached_papers
    
    # 调用 API 获取数据
    works_data = fetch_author_works(orcid_id)
    papers = parse_paper_list(works_data)
    
    # 写入缓存
    if cache:
        cache.set_field(cache_key, "papers", papers, ttl=CACHE_TTL)
        print(f"[INFO] ORCID {orcid_id} 论文列表已写入缓存 ({len(papers)} 篇)")
    
    return papers


if __name__ == "__main__":
    # 示例用法
    import json
    
    test_orcid = "0000-0003-3701-8119"
    
    print(f"正在获取 ORCID {test_orcid} 的论文列表...")
    
    cache = _get_cache()
    cache_key = test_orcid
    
    if cache:
        has_cache = cache.has_field(cache_key, "papers")
        print(f"[缓存] {'命中' if has_cache else '未命中'}")
    
    try:
        papers = get_author_papers(test_orcid)
        print(f"\n共获取到 {len(papers)} 篇论文\n")
        
        for i, paper in enumerate(papers[:5], 1):  # 只显示前5篇
            print(f"--- 论文 {i} ---")
            print(f"标题: {paper['title']}")
            print(f"DOI: {paper['doi']}")
            print(f"发表年份: {paper['publication_year']}")
            print(f"期刊: {paper['journal']}")
            print()
        
        if len(papers) > 5:
            print(f"... 还有 {len(papers) - 5} 篇论文未显示")
        
        if cache:
            print(f"\n[缓存] 数据已写入 collection: {CACHE_COLLECTION}")
            
    except requests.exceptions.HTTPError as e:
        print(f"API 请求失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
