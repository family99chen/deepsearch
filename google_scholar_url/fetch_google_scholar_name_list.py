"""
Google Scholar 作者搜索模块
使用 Google Custom Search API 搜索 Google Scholar 个人页面

特性：
- 使用官方 Google Search API，无需维护 cookies
- 支持指数退避重试
- 支持异步批量搜索
"""

import os
import sys
import re
import asyncio
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 Google Search API
from google_search_api.google_search import GoogleSearchAPI, GoogleSearchAPISync

# 导入作者名字过滤器
from google_scholar_url.author_name_filter import is_same_author

# 导入缓存模块
try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

# 缓存配置
CACHE_COLLECTION = "google_scholar_person"
CACHE_TTL = 180 * 24 * 3600  # 6个月


def _get_cache() -> Optional["MongoCache"]:
    """获取缓存实例"""
    if not HAS_CACHE:
        return None
    try:
        cache = MongoCache(collection_name=CACHE_COLLECTION)
        return cache if cache.is_connected() else None
    except Exception:
        return None


def _normalize_name(name: str) -> str:
    """
    标准化作者名字（去除特殊字符、小写）
    """
    if not name:
        return ""
    # 去除 Unicode 零宽字符和控制字符
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', name)
    # 小写并合并空格
    return " ".join(name.lower().split())


class GoogleScholarAuthorScraper:
    """
    Google Scholar 作者搜索（使用 Google Custom Search API）
    
    通过 Google 搜索 site:scholar.google.com/citations 来查找学者主页
    """
    
    # Google Scholar 个人页面的 URL 模式
    SCHOLAR_PROFILE_PATTERN = re.compile(
        r'https?://scholar\.google\.com[^/]*/citations\?[^"]*user=([a-zA-Z0-9_-]+)'
    )
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        max_retries: int = 3,
        **kwargs  # 兼容旧接口的参数
    ):
        """
        初始化搜索器
        
        Args:
            api_key: Google API Key（不提供则从配置文件读取）
            cx: Custom Search Engine ID（不提供则从配置文件读取）
            max_retries: 最大重试次数（API 内部处理）
        """
        self.api_key = api_key
        self.cx = cx
        self.max_retries = max_retries
        
        # 兼容旧接口
        self.cookies_valid = True  # API 方式始终有效
        self.cookies_path = kwargs.get('cookies_path')  # 忽略，保持兼容
        
        # 缓存（只写入，供后续环节使用）
        self._cache = _get_cache()
    
    def search_author(
        self, 
        author_name: str, 
        start: int = 1, 
        num: int = 10,
        organization: Optional[str] = None
    ) -> List[Dict]:
        """
        搜索作者并返回作者信息列表（单页）
        
        Args:
            author_name: 作者名字
            start: 起始位置（1-100），用于分页
                   - start=1: 返回第 1-10 条
                   - start=11: 返回第 11-20 条
                   - start=21: 返回第 21-30 条
                   - 以此类推...
            num: 返回数量（1-10，Google API 限制）
            organization: 作者所属机构（可选），用于缩小搜索范围
            
        Returns:
            包含作者信息的字典列表，每个字典包含:
            - name: 作者名字（从搜索结果标题提取）
            - url: Google Scholar 个人页面 URL
            - user_id: Google Scholar user ID
            - affiliation: 机构（从搜索结果摘要提取）
            - snippet: 搜索结果摘要
            
        Usage:
            # 第一次搜索（第 1-10 条）
            authors = scraper.search_author("Jun Li", start=1)
            
            # 带组织搜索（更精确）
            authors = scraper.search_author("Jun Li", organization="MIT")
            
            # 如果没找到，继续搜索（第 11-20 条）
            if not found:
                authors = scraper.search_author("Jun Li", start=11)
        """
        # 参数校验
        start = max(1, min(start, 100))  # Google API 限制 1-100
        num = max(1, min(num, 10))  # Google API 限制最多 10 条
        
        print(f"[INFO] 正在搜索作者: {author_name}")
        if organization:
            print(f"[INFO] 限定机构: {organization}")
        print(f"[INFO] 起始位置: {start}, 数量: {num}")
        print(f"[INFO] 使用 Google Custom Search API")
        
        # 构建搜索查询：限定在 Google Scholar 个人页面
        if organization:
            # 如果有组织信息，添加到搜索条件中
            query = f'site:scholar.google.com/citations {author_name} {organization}'
        else:
            query = f'site:scholar.google.com/citations {author_name}'
        
        print(f"[DEBUG] 查询: {query}")
        
        try:
            # 使用同步 API
            api = GoogleSearchAPISync(api_key=self.api_key, cx=self.cx)
            
            result = api.search(
                query=query,
                num=num,
                start=start
            )
            
            if not result.success:
                print(f"[ERROR] API 请求失败: {result.error}")
                return []
            
            if not result.items:
                print("[INFO] 没有搜索结果")
                return []
            
            # 解析搜索结果
            all_authors = []
            seen_user_ids = set()
            
            for item in result.items:
                author_info = self._parse_search_result(item, author_name)
                
                if author_info and author_info['user_id'] not in seen_user_ids:
                    result_name = author_info['name']
                    
                    # 将原始结果写入缓存（用 user_id 去重，供后续环节使用）
                    if self._cache:
                        cache_key = author_info['user_id']  # 用 user_id 作为唯一 key
                        # 添加标准化后的名字字段
                        author_info_to_cache = author_info.copy()
                        author_info_to_cache['normalized_name'] = _normalize_name(result_name)
                        self._cache.set(cache_key, author_info_to_cache, ttl=CACHE_TTL)
                    
                    # 使用名字过滤器检查是否是同一人
                    is_match, _ = is_same_author(author_name, result_name, verbose=False)
                    
                    if is_match:
                        seen_user_ids.add(author_info['user_id'])
                        all_authors.append(author_info)
                        print(f"  - 找到: {author_info['name']} | {author_info.get('affiliation', 'N/A')}")
                    else:
                        print(f"  - 跳过（名字不匹配）: {result_name}")
            
            print(f"[INFO] 本页获取 {len(all_authors)} 个作者 (start={start})")
            
            return all_authors
            
        except Exception as e:
            print(f"[ERROR] 搜索失败: {e}")
            return []
    
    async def search_author_async(
        self, 
        author_name: str, 
        start: int = 1, 
        num: int = 10,
        organization: Optional[str] = None
    ) -> List[Dict]:
        """
        异步搜索作者（单页）
        
        Args:
            author_name: 作者名字
            start: 起始位置（1-100）
            num: 返回数量（1-10）
            organization: 作者所属机构（可选）
            
        Returns:
            包含作者信息的字典列表
        """
        start = max(1, min(start, 100))
        num = max(1, min(num, 10))
        
        print(f"[INFO] 正在搜索作者: {author_name}")
        if organization:
            print(f"[INFO] 限定机构: {organization}")
        print(f"[INFO] 起始位置: {start}, 数量: {num}")
        print(f"[INFO] 使用 Google Custom Search API (异步)")
        
        if organization:
            query = f'site:scholar.google.com/citations "{author_name}" "{organization}"'
        else:
            query = f'site:scholar.google.com/citations "{author_name}"'
        
        try:
            async with GoogleSearchAPI(api_key=self.api_key, cx=self.cx) as api:
                result = await api.search(
                    query=query,
                    num=num,
                    start=start
                )
                
                if not result.success:
                    print(f"[ERROR] API 请求失败: {result.error}")
                    return []
                
                if not result.items:
                    print("[INFO] 没有搜索结果")
                    return []
                
                all_authors = []
                seen_user_ids = set()
                
                for item in result.items:
                    author_info = self._parse_search_result(item, author_name)
                    
                    if author_info and author_info['user_id'] not in seen_user_ids:
                        # 使用名字过滤器检查是否是同一人
                        result_name = author_info['name']
                        is_match, _ = is_same_author(author_name, result_name, verbose=False)
                        
                        if is_match:
                            seen_user_ids.add(author_info['user_id'])
                            all_authors.append(author_info)
                            print(f"  - 找到: {author_info['name']} | {author_info.get('affiliation', 'N/A')}")
                        else:
                            print(f"  - 跳过（名字不匹配）: {result_name}")
                
                print(f"[INFO] 本页获取 {len(all_authors)} 个作者 (start={start})")
                
                return all_authors
                
        except Exception as e:
            print(f"[ERROR] 搜索失败: {e}")
            return []
    
    def _parse_search_result(self, item, search_name: str) -> Optional[Dict]:
        """
        解析单个搜索结果
        
        Args:
            item: SearchResult 对象
            search_name: 原始搜索的名字
            
        Returns:
            作者信息字典，如果不是有效的 Scholar 页面则返回 None
        """
        url = item.link
        
        # 检查是否是 Google Scholar 个人页面
        if 'scholar.google.com' not in url or 'citations' not in url:
            return None
        
        # 提取 user ID
        user_id = self._extract_user_id(url)
        if not user_id:
            return None
        
        # 标准化 URL
        clean_url = f"https://scholar.google.com/citations?user={user_id}&hl=zh-CN"
        
        # 从标题提取作者名字
        # 标题格式通常是 "作者名 - Google Scholar" 或 "作者名- Google 学术搜索"
        title = item.title
        name = self._extract_name_from_title(title)
        
        # 从摘要提取机构信息
        snippet = item.snippet or ""
        affiliation = self._extract_affiliation(snippet)
        
        # 提取研究兴趣（如果摘要中有）
        interests = self._extract_interests(snippet)
        
        return {
            "name": name or search_name,
            "url": clean_url,
            "user_id": user_id,
            "affiliation": affiliation,
            "snippet": snippet,
            "interests": interests,
            "email": "",  # API 无法获取
            "cited_by": "",  # API 无法获取
        }
    
    def _extract_user_id(self, url: str) -> Optional[str]:
        """从 URL 提取 Google Scholar user ID"""
        # 方法1：从 URL 参数提取
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'user' in params:
                return params['user'][0]
        except:
            pass
        
        # 方法2：用正则表达式提取
        match = self.SCHOLAR_PROFILE_PATTERN.search(url)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_name_from_title(self, title: str) -> str:
        """从搜索结果标题提取作者名字"""
        if not title:
            return ""
        
        # 移除 " - Google Scholar" 或类似后缀
        patterns = [
            r'\s*[-–—]\s*Google\s*(Scholar|学术搜索|學術搜尋).*$',
            r'\s*[-–—]\s*Google.*$',
            r'\s*\|\s*Google.*$',
        ]
        
        name = title
        for pattern in patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _extract_affiliation(self, snippet: str) -> str:
        """从摘要提取机构信息"""
        if not snippet:
            return ""
        
        # 机构通常在摘要开头，以逗号或句号分隔
        # 例如："Professor, MIT · Computer Science · Machine Learning"
        
        # 尝试提取第一个主要部分
        parts = re.split(r'[·•|]', snippet)
        if parts:
            # 取第一部分，通常是机构
            first_part = parts[0].strip()
            # 如果太长，可能不是机构
            if len(first_part) < 100:
                return first_part
        
        return ""
    
    def _extract_interests(self, snippet: str) -> List[str]:
        """从摘要提取研究兴趣"""
        if not snippet:
            return []
        
        # 研究兴趣通常用 · 或 | 分隔
        parts = re.split(r'[·•|]', snippet)
        
        interests = []
        for part in parts[1:]:  # 跳过第一部分（通常是机构）
            part = part.strip()
            # 过滤掉太长或太短的
            if 3 < len(part) < 50:
                interests.append(part)
        
        return interests[:5]  # 最多返回 5 个
    
    def get_author_urls(
        self, 
        author_name: str, 
        start: int = 1, 
        num: int = 10,
        organization: Optional[str] = None
    ) -> List[str]:
        """只返回作者主页 URL 列表"""
        authors = self.search_author(author_name, start=start, num=num, organization=organization)
        return [a["url"] for a in authors if a.get("url")]
    
    def search_author_all(
        self, 
        author_name: str, 
        max_results: int = 20,
        organization: Optional[str] = None
    ) -> List[Dict]:
        """
        搜索作者并返回多页结果（便捷方法）
        
        Args:
            author_name: 作者名字
            max_results: 最大返回结果数（最多 100）
            organization: 作者所属机构（可选）
            
        Returns:
            包含作者信息的字典列表
        """
        all_authors = []
        seen_user_ids = set()
        
        for start in range(1, min(max_results, 100) + 1, 10):
            authors = self.search_author(author_name, start=start, num=10, organization=organization)
            
            for author in authors:
                if author['user_id'] not in seen_user_ids:
                    seen_user_ids.add(author['user_id'])
                    all_authors.append(author)
            
            if len(all_authors) >= max_results:
                break
            
            if len(authors) < 10:
                break
        
        return all_authors[:max_results]
    
    # ============ 兼容旧接口的方法 ============
    
    def _load_cookies(self) -> bool:
        """兼容旧接口，始终返回 True"""
        return True
    
    def login_and_save_cookies(self, wait_time: int = 60):
        """兼容旧接口，API 方式无需登录"""
        print("[INFO] 使用 Google Custom Search API，无需登录")
        print("[INFO] 请确保 config.yaml 中配置了 google.api_key 和 google.cx")
        return True


def main():
    """主函数"""
    scraper = GoogleScholarAuthorScraper()
    
    # 搜索作者
    test_name = "Alexandra Ramadan"
    
    print("\n" + "=" * 60)
    print(f"搜索作者: {test_name}")
    print("=" * 60)
    
    # 方式1：单页搜索（默认返回第 1-10 条）
    print("\n[测试1] 第一页搜索 (start=1)")
    authors = scraper.search_author(test_name, start=1)
    
    if authors:
        print(f"找到 {len(authors)} 个作者")
        for i, author in enumerate(authors, 1):
            print(f"  [{i}] {author['name']} | {author.get('affiliation', 'N/A')[:30]}")
    
    # 方式2：继续搜索下一页
    print("\n[测试2] 第二页搜索 (start=11)")
    authors_page2 = scraper.search_author(test_name, start=11)
    
    if authors_page2:
        print(f"找到 {len(authors_page2)} 个作者")
        for i, author in enumerate(authors_page2, 1):
            print(f"  [{i}] {author['name']} | {author.get('affiliation', 'N/A')[:30]}")
    
    # 方式3：使用便捷方法获取多页
    # print("\n[测试3] 获取前20条 (search_author_all)")
    # all_authors = scraper.search_author_all(test_name, max_results=20)
    # print(f"共找到 {len(all_authors)} 个作者")
    
    if not authors:
        print("\n[WARNING] 未获取到结果")
        print("[提示] 请检查 config.yaml 中的 Google API 配置")


if __name__ == "__main__":
    main()
