"""
通过 Google Search 搜索组织中的人员链接

功能：
- 输入组织名 + 个人名字
- 搜索这个人在该组织官网的页面链接
- 返回相关链接列表
- 支持 MongoDB 缓存，避免重复 API 调用（6个月过期）

使用 Google Custom Search API
"""

import sys
import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 Google Search API
from google_search_api.google_search import GoogleSearchAPISync, SearchResult

# 导入 MongoDB 缓存
try:
    from localdb.insert_mongo import MongoCache
    HAS_MONGO_CACHE = True
except ImportError:
    HAS_MONGO_CACHE = False

# 导入日志
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ============ 缓存配置 ============
# 缓存 Collection 名称
CACHE_COLLECTION = "person_relevant_link"
# 缓存过期时间：6个月 = 6 * 30 * 24 * 60 * 60 秒
CACHE_TTL_SECONDS = 6 * 30 * 24 * 60 * 60  # 15552000 秒


@dataclass
class OrgPersonLink:
    """组织人员链接"""
    title: str                  # 页面标题
    url: str                    # 链接地址
    domain: str                 # 域名
    snippet: str                # 摘要
    link_type: str              # 链接类型（profile/faculty/staff/research/other）
    relevance_score: float      # 相关性评分 (0-1)


class GoogleOrgSearch:
    """
    Google 组织人员搜索器
    
    搜索某人在特定组织中的官方页面链接
    支持 MongoDB 缓存，避免重复 API 调用
    """
    
    # 常见的人员页面 URL 模式
    PROFILE_PATTERNS = [
        r'/people/',
        r'/faculty/',
        r'/staff/',
        r'/profile/',
        r'/researcher/',
        r'/team/',
        r'/members/',
        r'/directory/',
        r'/author/',
        r'/person/',
        r'/users/',
    ]
    
    # 需要过滤的页面类型
    FILTER_PATTERNS = [
        r'/search\?',
        r'/login',
        r'/signin',
        r'/news/',
        r'/events/',
        r'/publications/',  # 论文列表页，不是个人页
        r'/job',
        r'/career',
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = True,
        use_cache: bool = True,
    ):
        """
        初始化搜索器
        
        Args:
            api_key: Google API Key（不提供则从配置读取）
            cx: Custom Search Engine ID（不提供则从配置读取）
            max_retries: 最大重试次数
            verbose: 是否打印详细日志
            use_cache: 是否使用 MongoDB 缓存（默认 True）
        """
        self.api = GoogleSearchAPISync(
            api_key=api_key,
            cx=cx,
            max_retries=max_retries,
        )
        self.verbose = verbose
        self.use_cache = use_cache and HAS_MONGO_CACHE
        
        # 初始化缓存
        self._cache = None
        if self.use_cache:
            try:
                self._cache = MongoCache(collection_name=CACHE_COLLECTION)
                if not self._cache.is_connected():
                    if self.verbose:
                        print("[WARNING] MongoDB 缓存连接失败，将不使用缓存")
                    logger.warning("MongoDB 缓存连接失败")
                    self._cache = None
                    self.use_cache = False
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] 初始化缓存失败: {e}")
                logger.warning(f"初始化缓存失败: {e}")
                self._cache = None
                self.use_cache = False
    
    def _get_cache_key(self, person_name: str, google_scholar_url: Optional[str]) -> str:
        """
        生成缓存键（优先使用 Google Scholar URL 去重）
        
        Args:
            person_name: 人员名字
            
        Returns:
            缓存键
        """
        if google_scholar_url:
            return f"gs:{google_scholar_url.strip().lower()}"
        # 回退到人员名字
        return f"person:{person_name.strip().lower()}"
    
    def _get_cached_links(
        self, 
        person_name: str, 
        organization: str,
        google_scholar_url: Optional[str] = None,
    ) -> Optional[List[OrgPersonLink]]:
        """
        从缓存获取链接
        
        缓存结构:
        {
            "key": "person:emmy tay",
            "value": {
                "organization": {
                    "links": [...],
                    "expire_at": datetime
                },
                "social_media": {  # 将来可扩展
                    "links": [...],
                    "expire_at": datetime
                }
            }
        }
        
        Args:
            person_name: 人员名字
            organization: 组织名称
            
        Returns:
            缓存的链接列表，如果没有或已过期则返回 None
        """
        if not self._cache:
            return None
        
        cache_key = self._get_cache_key(person_name, google_scholar_url)
        
        try:
            # 获取文档
            doc = self._cache.collection.find_one({"key": cache_key})
            
            if not doc:
                return None
            
            value = doc.get("value", {})
            if not isinstance(value, dict):
                return None
            
            # 获取 organization 字段数据
            org_data = value.get("organization")
            if not org_data:
                return None
            
            # 检查 organization 数据是否过期
            expire_at = org_data.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                if self.verbose:
                    print(f"[CACHE] {person_name} 的 organization 链接已过期，将重新搜索")
                return None
            
            # 解析链接数据
            links_data = org_data.get("links", [])
            links = []
            for link_data in links_data:
                links.append(OrgPersonLink(
                    title=link_data.get("title", ""),
                    url=link_data.get("url", ""),
                    domain=link_data.get("domain", ""),
                    snippet=link_data.get("snippet", ""),
                    link_type=link_data.get("link_type", "other"),
                    relevance_score=link_data.get("relevance_score", 0.0),
                ))
            
            if self.verbose:
                print(f"[CACHE HIT] 从缓存读取 {len(links)} 个 organization 链接")
            key_hint = google_scholar_url or person_name
            logger.info(f"缓存命中: {key_hint}, {len(links)} 个链接")
            
            return links
            
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 读取缓存失败: {e}")
            logger.warning(f"读取缓存失败: {person_name}, {e}")
            return None
    
    def _save_to_cache(
        self, 
        person_name: str, 
        organization: str, 
        links: List[OrgPersonLink],
        google_scholar_url: Optional[str] = None,
    ) -> bool:
        """
        保存链接到缓存（更新时替换 organization 字段的旧数据）
        
        缓存结构:
        {
            "key": "person:emmy tay",
            "value": {
                "organization": {
                    "links": [...],
                    "updated_at": datetime,
                    "expire_at": datetime
                }
            }
        }
        
        Args:
            person_name: 人员名字
            organization: 组织名称（用于记录来源）
            links: 链接列表
            
        Returns:
            是否成功
        """
        if not self._cache:
            return False
        
        cache_key = self._get_cache_key(person_name, google_scholar_url)
        
        try:
            # 将链接转换为可序列化的字典列表
            links_data = [asdict(link) for link in links]
            
            from datetime import timedelta
            # organization 字段数据（包含链接列表和过期时间）
            org_data = {
                "links": links_data,
                "source_organization": organization,  # 记录搜索时使用的组织名
                "google_scholar_url": google_scholar_url,
                "updated_at": datetime.utcnow(),
                "expire_at": datetime.utcnow() + timedelta(seconds=CACHE_TTL_SECONDS),
            }
            
            # 使用 $set 更新 organization 字段（会自动替换旧数据）
            self._cache.collection.update_one(
                {"key": cache_key},
                {
                    "$set": {
                        "value.organization": org_data,
                        "updated_at": datetime.utcnow(),
                    },
                    "$setOnInsert": {
                        "key": cache_key,
                        "created_at": datetime.utcnow(),
                    }
                },
                upsert=True
            )
            
            if self.verbose:
                print(f"[CACHE SAVE] 已缓存 {len(links)} 个 organization 链接")
            key_hint = google_scholar_url or person_name
            logger.info(f"缓存写入: {key_hint} @ {organization}, {len(links)} 个链接")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 保存缓存失败: {e}")
            logger.warning(f"保存缓存失败: {person_name}, {e}")
            return False
    
    def search_person_in_org(
        self,
        person_name: str,
        organization: str,
        max_results: int = 20,
        site_restrict: Optional[str] = None,
        force_refresh: bool = False,
        google_scholar_url: Optional[str] = None,
    ) -> List[OrgPersonLink]:
        """
        搜索某人在组织中的链接
        
        Args:
            person_name: 人员名字
            organization: 组织名称
            max_results: 最大返回结果数
            site_restrict: 限定搜索站点（如 "mit.edu"）
            force_refresh: 强制刷新缓存（忽略缓存，重新搜索）
            
        Returns:
            OrgPersonLink 列表
        """
        if self.verbose:
            print(f"[INFO] 搜索: {person_name} @ {organization}")
        logger.info(f"开始搜索组织链接: {person_name} @ {organization}")
        
        # 检查缓存（除非强制刷新）
        if self.use_cache and not force_refresh:
            cached_links = self._get_cached_links(person_name, organization, google_scholar_url)
            if cached_links is not None:
                return cached_links[:max_results]
        
        # 构建搜索查询
        if site_restrict:
            query = f'site:{site_restrict} "{person_name}"'
        else:
            query = f'"{person_name}" "{organization}"'
        
        if self.verbose:
            print(f"[DEBUG] 查询: {query}")
        
        # 执行搜索
        result = self.api.search(query, num=min(max_results, 10))
        
        if not result.success:
            print(f"[ERROR] 搜索失败: {result.error}")
            logger.error(f"Google 搜索失败: {person_name} @ {organization}, {result.error}")
            return []
        
        if not result.items:
            if self.verbose:
                print("[INFO] 没有搜索结果")
            # 空结果不缓存，下次会重新搜索
            return []
        
        # 解析结果
        links = []
        for item in result.items:
            link = self._parse_search_result(item, person_name, organization)
            if link:
                links.append(link)
        
        # 按相关性排序
        links.sort(key=lambda x: x.relevance_score, reverse=True)
        
        if self.verbose:
            print(f"[INFO] 找到 {len(links)} 个相关链接")
        logger.info(f"搜索完成: {person_name} @ {organization}, 找到 {len(links)} 个链接")
        
        # 保存到缓存（空结果不缓存）
        if self.use_cache and links:
            self._save_to_cache(person_name, organization, links, google_scholar_url)
        
        return links[:max_results]

    def search_person_in_org_with_raw(
        self,
        person_name: str,
        organization: str,
        max_results: int = 20,
        site_restrict: Optional[str] = None,
        force_refresh: bool = False,
        google_scholar_url: Optional[str] = None,
    ) -> Tuple[List[OrgPersonLink], Dict[str, Any]]:
        """
        搜索某人在组织中的链接，并返回原始搜索内容（包含摘要）
        """
        if self.verbose:
            print(f"[INFO] 搜索: {person_name} @ {organization}")
        logger.info(f"开始搜索组织链接: {person_name} @ {organization}")

        # 构建搜索查询
        if site_restrict:
            query = f'site:{site_restrict} "{person_name}"'
        else:
            query = f'"{person_name}" "{organization}"'

        # 检查缓存（除非强制刷新）
        if self.use_cache and not force_refresh:
            cached_links = self._get_cached_links(person_name, organization, google_scholar_url)
            if cached_links is not None:
                raw = {
                    "from_cache": True,
                    "success": True,
                    "query": query,
                    "total_results": len(cached_links),
                    "error": None,
                    "items": [
                        {
                            "title": link.title,
                            "link": link.url,
                            "snippet": link.snippet,
                            "display_link": link.domain,
                            "html_title": None,
                            "html_snippet": None,
                            "formatted_url": None,
                        }
                        for link in cached_links[:max_results]
                    ],
                    "google_scholar_url": google_scholar_url,
                }
                return cached_links[:max_results], raw

        if self.verbose:
            print(f"[DEBUG] 查询: {query}")

        # 执行搜索
        result = self.api.search(query, num=min(max_results, 10))

        raw = {
            "from_cache": False,
            "success": result.success,
            "query": result.query,
            "total_results": result.total_results,
            "error": result.error,
            "items": [
                {
                    "title": item.title,
                    "link": item.link,
                    "snippet": item.snippet,
                    "display_link": item.display_link,
                    "html_title": item.html_title,
                    "html_snippet": item.html_snippet,
                    "formatted_url": item.formatted_url,
                }
                for item in result.items
            ],
            "google_scholar_url": google_scholar_url,
        }

        if not result.success:
            print(f"[ERROR] 搜索失败: {result.error}")
            logger.error(f"Google 搜索失败: {person_name} @ {organization}, {result.error}")
            return [], raw

        if not result.items:
            if self.verbose:
                print("[INFO] 没有搜索结果")
            return [], raw

        # 解析结果
        links = []
        for item in result.items:
            link = self._parse_search_result(item, person_name, organization)
            if link:
                links.append(link)

        # 按相关性排序
        links.sort(key=lambda x: x.relevance_score, reverse=True)

        if self.verbose:
            print(f"[INFO] 找到 {len(links)} 个相关链接")
        logger.info(f"搜索完成: {person_name} @ {organization}, 找到 {len(links)} 个链接")

        # 保存到缓存（空结果不缓存）
        if self.use_cache and links:
            self._save_to_cache(person_name, organization, links, google_scholar_url)

        return links[:max_results], raw
    
    def search_person_in_org_multi_query(
        self,
        person_name: str,
        organization: str,
        max_results: int = 20,
        force_refresh: bool = False,
    ) -> List[OrgPersonLink]:
        """
        使用多种查询策略搜索某人在组织中的链接
        
        Args:
            person_name: 人员名字
            organization: 组织名称
            max_results: 最大返回结果数
            force_refresh: 强制刷新缓存（忽略缓存，重新搜索）
            
        Returns:
            OrgPersonLink 列表（去重后）
        """
        if self.verbose:
            print(f"[INFO] 多策略搜索: {person_name} @ {organization}")
        
        # 检查缓存（除非强制刷新）
        if self.use_cache and not force_refresh:
            cached_links = self._get_cached_links(person_name, organization)
            if cached_links is not None:
                return cached_links[:max_results]
        
        all_links = []
        seen_urls = set()
        
        # 策略1: 基本搜索
        queries = [
            f'"{person_name}" "{organization}"',
            f'"{person_name}" {organization} profile',
            f'"{person_name}" {organization} faculty',
            f'"{person_name}" site:{self._guess_domain(organization)}' if self._guess_domain(organization) else None,
        ]
        
        for query in queries:
            if not query:
                continue
            
            if self.verbose:
                print(f"[DEBUG] 查询: {query}")
            
            result = self.api.search(query, num=10)
            
            if not result.success:
                if self.verbose:
                    print(f"[WARNING] 查询失败: {result.error}")
                continue
            
            for item in result.items:
                if item.link in seen_urls:
                    continue
                
                link = self._parse_search_result(item, person_name, organization)
                if link:
                    seen_urls.add(item.link)
                    all_links.append(link)
        
        # 按相关性排序
        all_links.sort(key=lambda x: x.relevance_score, reverse=True)
        
        if self.verbose:
            print(f"[INFO] 共找到 {len(all_links)} 个不重复链接")
        logger.info(f"多策略搜索完成: {person_name} @ {organization}, 找到 {len(all_links)} 个链接")
        
        # 保存到缓存（空结果不缓存）
        if self.use_cache and all_links:
            self._save_to_cache(person_name, organization, all_links)
        
        return all_links[:max_results]
    
    def _parse_search_result(
        self,
        item: SearchResult,
        person_name: str,
        organization: str,
    ) -> Optional[OrgPersonLink]:
        """
        解析搜索结果
        
        Args:
            item: 搜索结果项
            person_name: 人员名字
            organization: 组织名称
            
        Returns:
            OrgPersonLink 或 None
        """
        url = item.link
        title = item.title
        snippet = item.snippet
        
        # 过滤不相关的页面
        for pattern in self.FILTER_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return None
        
        # 提取域名
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            domain = ""
        
        # 判断链接类型
        link_type = self._classify_link_type(url, title)
        
        # 计算相关性评分
        relevance_score = self._calculate_relevance(
            url, title, snippet, person_name, organization
        )
        
        return OrgPersonLink(
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            link_type=link_type,
            relevance_score=relevance_score,
        )
    
    def _classify_link_type(self, url: str, title: str) -> str:
        """分类链接类型"""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # 检查 URL 模式
        for pattern in self.PROFILE_PATTERNS:
            if re.search(pattern, url_lower):
                if 'faculty' in pattern:
                    return 'faculty'
                if 'staff' in pattern:
                    return 'staff'
                if 'research' in pattern:
                    return 'research'
                return 'profile'
        
        # 检查标题关键词
        if any(kw in title_lower for kw in ['profile', 'personal', '个人']):
            return 'profile'
        if any(kw in title_lower for kw in ['faculty', '教授', '教师']):
            return 'faculty'
        if any(kw in title_lower for kw in ['researcher', '研究员']):
            return 'research'
        
        return 'other'
    
    def _calculate_relevance(
        self,
        url: str,
        title: str,
        snippet: str,
        person_name: str,
        organization: str,
    ) -> float:
        """
        计算相关性评分
        
        Args:
            url: 页面 URL
            title: 页面标题
            snippet: 页面摘要
            person_name: 人员名字
            organization: 组织名称
            
        Returns:
            相关性评分 (0-1)
        """
        score = 0.0
        
        # 名字出现在标题中 (+0.3)
        if person_name.lower() in title.lower():
            score += 0.3
        
        # 名字出现在 URL 中 (+0.2)
        name_parts = person_name.lower().split()
        url_lower = url.lower()
        if any(part in url_lower for part in name_parts if len(part) > 2):
            score += 0.2
        
        # 组织名出现在域名中 (+0.2)
        org_parts = organization.lower().split()
        if any(part in url_lower for part in org_parts if len(part) > 2):
            score += 0.2
        
        # 是个人页面类型 (+0.2)
        for pattern in self.PROFILE_PATTERNS:
            if re.search(pattern, url_lower):
                score += 0.2
                break
        
        # 摘要中包含名字 (+0.1)
        if person_name.lower() in snippet.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def _guess_domain(self, organization: str) -> Optional[str]:
        """
        根据组织名猜测域名
        
        Args:
            organization: 组织名称
            
        Returns:
            可能的域名或 None
        """
        org_lower = organization.lower()
        
        # 常见大学域名映射
        university_domains = {
            'mit': 'mit.edu',
            'massachusetts institute of technology': 'mit.edu',
            'stanford': 'stanford.edu',
            'stanford university': 'stanford.edu',
            'harvard': 'harvard.edu',
            'harvard university': 'harvard.edu',
            'berkeley': 'berkeley.edu',
            'uc berkeley': 'berkeley.edu',
            'carnegie mellon': 'cmu.edu',
            'cmu': 'cmu.edu',
            'caltech': 'caltech.edu',
            'princeton': 'princeton.edu',
            'yale': 'yale.edu',
            'columbia': 'columbia.edu',
            'oxford': 'ox.ac.uk',
            'cambridge': 'cam.ac.uk',
            'tsinghua': 'tsinghua.edu.cn',
            '清华': 'tsinghua.edu.cn',
            '清华大学': 'tsinghua.edu.cn',
            'peking': 'pku.edu.cn',
            '北大': 'pku.edu.cn',
            '北京大学': 'pku.edu.cn',
            'fudan': 'fudan.edu.cn',
            '复旦': 'fudan.edu.cn',
            '复旦大学': 'fudan.edu.cn',
            'zhejiang': 'zju.edu.cn',
            '浙大': 'zju.edu.cn',
            '浙江大学': 'zju.edu.cn',
            'sjtu': 'sjtu.edu.cn',
            '上交': 'sjtu.edu.cn',
            '上海交通大学': 'sjtu.edu.cn',
            'nus': 'nus.edu.sg',
            'national university of singapore': 'nus.edu.sg',
            'ntu': 'ntu.edu.sg',
            'nanyang': 'ntu.edu.sg',
            'eth': 'ethz.ch',
            'eth zurich': 'ethz.ch',
            'google': 'google.com',
            'microsoft': 'microsoft.com',
            'facebook': 'facebook.com',
            'meta': 'meta.com',
            'apple': 'apple.com',
            'amazon': 'amazon.com',
            'alibaba': 'alibaba.com',
            '阿里巴巴': 'alibaba.com',
            'tencent': 'tencent.com',
            '腾讯': 'tencent.com',
            'baidu': 'baidu.com',
            '百度': 'baidu.com',
        }
        
        for key, domain in university_domains.items():
            if key in org_lower:
                return domain
        
        return None


# ============ 便捷函数 ============

def search_person_in_org(
    person_name: str,
    organization: str,
    max_results: int = 20,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    google_scholar_url: Optional[str] = None,
) -> List[OrgPersonLink]:
    """
    便捷函数：搜索某人在组织中的链接
    
    Args:
        person_name: 人员名字
        organization: 组织名称
        max_results: 最大返回结果数
        verbose: 是否打印详细日志
        use_cache: 是否使用 MongoDB 缓存（默认 True）
        force_refresh: 强制刷新缓存（忽略缓存，重新搜索）
        
    Returns:
        OrgPersonLink 列表
    """
    searcher = GoogleOrgSearch(verbose=verbose, use_cache=use_cache)
    return searcher.search_person_in_org(
        person_name=person_name,
        organization=organization,
        max_results=max_results,
        force_refresh=force_refresh,
        google_scholar_url=google_scholar_url,
    )


def search_person_in_org_with_raw(
    person_name: str,
    organization: str,
    max_results: int = 20,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    google_scholar_url: Optional[str] = None,
) -> Tuple[List[OrgPersonLink], Dict[str, Any]]:
    """
    便捷函数：搜索某人在组织中的链接，并返回原始搜索内容（包含摘要）
    """
    searcher = GoogleOrgSearch(verbose=verbose, use_cache=use_cache)
    return searcher.search_person_in_org_with_raw(
        person_name=person_name,
        organization=organization,
        max_results=max_results,
        force_refresh=force_refresh,
        google_scholar_url=google_scholar_url,
    )


def search_person_in_org_multi(
    person_name: str,
    organization: str,
    max_results: int = 20,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> List[OrgPersonLink]:
    """
    便捷函数：使用多策略搜索某人在组织中的链接
    
    Args:
        person_name: 人员名字
        organization: 组织名称
        max_results: 最大返回结果数
        verbose: 是否打印详细日志
        use_cache: 是否使用 MongoDB 缓存（默认 True）
        force_refresh: 强制刷新缓存（忽略缓存，重新搜索）
        
    Returns:
        OrgPersonLink 列表（去重后）
    """
    searcher = GoogleOrgSearch(verbose=verbose, use_cache=use_cache)
    return searcher.search_person_in_org_multi_query(
        person_name=person_name,
        organization=organization,
        max_results=max_results,
        force_refresh=force_refresh,
    )


def get_org_profile_url(
    person_name: str,
    organization: str,
    verbose: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Optional[str]:
    """
    便捷函数：获取最可能的个人主页 URL
    
    Args:
        person_name: 人员名字
        organization: 组织名称
        verbose: 是否打印详细日志
        use_cache: 是否使用 MongoDB 缓存（默认 True）
        force_refresh: 强制刷新缓存（忽略缓存，重新搜索）
        
    Returns:
        最可能的个人主页 URL 或 None
    """
    links = search_person_in_org(
        person_name, organization, 
        max_results=5, verbose=verbose,
        use_cache=use_cache, force_refresh=force_refresh
    )
    
    if not links:
        return None
    
    # 优先返回 profile 类型的链接
    for link in links:
        if link.link_type in ('profile', 'faculty'):
            return link.url
    
    # 否则返回评分最高的
    return links[0].url


def clear_person_org_cache(
    person_name: str,
    clear_all: bool = False,
    verbose: bool = True,
    google_scholar_url: Optional[str] = None,
) -> bool:
    """
    便捷函数：清除某人的 organization 链接缓存
    
    Args:
        person_name: 人员名字
        clear_all: 是否清除该人所有来源的缓存（包括 organization、social_media 等）
        verbose: 是否打印详细日志
        
    Returns:
        是否成功
    """
    if not HAS_MONGO_CACHE:
        if verbose:
            print("[WARNING] MongoDB 缓存模块不可用")
        return False
    
    try:
        cache = MongoCache(collection_name=CACHE_COLLECTION)
        if not cache.is_connected():
            if verbose:
                print("[WARNING] MongoDB 连接失败")
            return False
        
        if google_scholar_url:
            cache_key = f"gs:{google_scholar_url.strip().lower()}"
        else:
            cache_key = f"person:{person_name.strip().lower()}"
        
        if clear_all:
            # 清除该人所有缓存
            cache.delete(cache_key)
            if verbose:
                print(f"[CACHE] 已清除 {person_name} 的所有链接缓存")
            logger.info(f"清除缓存(全部): {person_name}")
        else:
            # 只清除 organization 字段
            cache.collection.update_one(
                {"key": cache_key},
                {"$unset": {"value.organization": ""}}
            )
            if verbose:
                print(f"[CACHE] 已清除 {person_name} 的 organization 链接缓存")
            logger.info(f"清除缓存(organization): {person_name}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] 清除缓存失败: {e}")
        logger.error(f"清除缓存失败: {person_name}, {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Google 组织人员搜索工具")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        ("Emmy Tay", "University of California, Davis"),
        ("Ju Li", "MIT"),
        # ("Andrew Ng", "Stanford"),
    ]
    
    for person_name, organization in test_cases:
        print()
        print(f"{'=' * 60}")
        print(f"搜索: {person_name} @ {organization}")
        print(f"{'=' * 60}")
        
        links = search_person_in_org(person_name, organization, max_results=10)
        
        if links:
            print(f"\n找到 {len(links)} 个链接:")
            for i, link in enumerate(links, 1):
                print(f"\n  [{i}] {link.title}")
                print(f"      URL: {link.url}")
                print(f"      类型: {link.link_type}")
                print(f"      评分: {link.relevance_score:.2f}")
                if link.snippet:
                    snippet = link.snippet[:100] + "..." if len(link.snippet) > 100 else link.snippet
                    print(f"      摘要: {snippet}")
        else:
            print("\n未找到相关链接")
        
        print()

