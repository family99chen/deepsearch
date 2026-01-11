"""
从 Google Scholar 账号 URL 提取组织信息

特性：
- 优先从缓存读取
- 缓存未命中时访问网页获取
- 提取组织名、作者名、研究兴趣等信息
- 写入 google_scholar_person_detail collection

独立模块，只复用底层缓存和日志模块
"""

import sys
import time
import random
import string
import re
import requests
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入日志模块
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 导入代理模块
try:
    from proxy import configure_session
except ImportError:
    def configure_session(session): return session

# 导入重试配置
try:
    from utils.retry import DEFAULT_RETRYABLE_EXCEPTIONS, DEFAULT_RETRYABLE_STATUS_CODES
except ImportError:
    DEFAULT_RETRYABLE_EXCEPTIONS = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )
    DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

# 导入缓存模块
HAS_CACHE = False
try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    pass

# 缓存配置
CACHE_COLLECTION = "google_scholar_person_detail"
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


def _generate_random_string(length: int, chars: str = None) -> str:
    """生成随机字符串"""
    if chars is None:
        chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def _generate_fake_cookies() -> Dict[str, str]:
    """
    生成假的 Google cookies
    让请求看起来像是登录用户
    """
    timestamp = int(time.time())
    
    fake_cookies = {
        'SID': _generate_random_string(68, string.ascii_letters + string.digits + '-_'),
        'HSID': _generate_random_string(11, string.ascii_letters + string.digits),
        'SSID': _generate_random_string(11, string.ascii_letters + string.digits),
        'APISID': _generate_random_string(32, string.ascii_letters + string.digits + '-_'),
        'SAPISID': _generate_random_string(43, string.ascii_letters + string.digits + '-_/'),
        'GSP': f'LM={timestamp}:S={_generate_random_string(16)}',
        'NID': f'{random.randint(500, 520)}=' + _generate_random_string(170, string.ascii_letters + string.digits + '-_='),
        'SEARCH_SAMESITE': 'CgQIz5sB',
        'AEC': _generate_random_string(70, string.ascii_letters + string.digits + '-_'),
        '1P_JAR': time.strftime('%Y-%m-%d-%H', time.gmtime()),
        'SIDCC': _generate_random_string(76, string.ascii_letters + string.digits + '-_'),
        '__Secure-1PSID': _generate_random_string(68, string.ascii_letters + string.digits + '-_.'),
        '__Secure-3PSID': _generate_random_string(68, string.ascii_letters + string.digits + '-_.'),
        'CONSENT': f'PENDING+{random.randint(100, 999)}',
        'GOOGLE_ABUSE_EXEMPTION': _generate_random_string(80, string.ascii_letters + string.digits + '-_='),
    }
    
    return fake_cookies


def _extract_user_id(profile_url: str) -> Optional[str]:
    """从 URL 中提取 user_id"""
    try:
        parsed = urlparse(profile_url)
        params = parse_qs(parsed.query)
        return params.get('user', [None])[0]
    except Exception:
        return None


class GoogleScholarOrgExtractor:
    """
    Google Scholar 组织信息提取器
    
    从 Google Scholar 个人页面提取组织/机构信息
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        """
        初始化提取器
        
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            use_cache: 是否使用缓存
            verbose: 是否打印详细日志
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_cache = use_cache
        self.verbose = verbose
        
        # 缓存实例
        self._cache = _get_cache() if use_cache else None
        
        # 创建 session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        
        # 配置代理
        configure_session(self.session)
        
        # 加载假 cookies
        self._load_fake_cookies()
    
    def _load_fake_cookies(self):
        """加载假 cookies"""
        fake_cookies = _generate_fake_cookies()
        for name, value in fake_cookies.items():
            self.session.cookies.set(name, value, domain='.google.com')
        if self.verbose:
            print(f"[INFO] 已加载假 cookies ({len(fake_cookies)} 个)")
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算指数退避延迟时间"""
        delay = min(
            self.base_delay * (2 ** attempt),
            self.max_delay
        )
        # 添加随机抖动 (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter
    
    def _request_with_retry(self, url: str, timeout: int = 15) -> Optional[requests.Response]:
        """带指数退避重试的请求"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=timeout)
                
                if response.status_code in DEFAULT_RETRYABLE_STATUS_CODES:
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        print(f"[RETRY] HTTP {response.status_code}，{delay:.1f}秒后重试 "
                              f"({attempt + 1}/{self.max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[ERROR] HTTP {response.status_code}，重试次数已用尽")
                        return None
                
                response.raise_for_status()
                return response
                
            except DEFAULT_RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                
                if attempt >= self.max_retries:
                    print(f"[ERROR] 请求失败，重试次数已用尽: {type(e).__name__}: {e}")
                    return None
                
                delay = self._calculate_delay(attempt)
                print(f"[RETRY] {type(e).__name__}，{delay:.1f}秒后重试 "
                      f"({attempt + 1}/{self.max_retries})...")
                time.sleep(delay)
                
            except requests.RequestException as e:
                print(f"[ERROR] 请求失败: {e}")
                return None
        
        return None
    
    def _check_blocked(self, html_text: str) -> bool:
        """检查是否被 Google Scholar 封锁"""
        if 'captcha' in html_text.lower() or 'recaptcha' in html_text.lower():
            return True
        if 'unusual traffic' in html_text.lower():
            return True
        if 'accounts.google.com' in html_text and 'gsc_prf_in' not in html_text:
            return True
        return False
    
    def _safe_int(self, text: str) -> int:
        """安全转换为整数"""
        try:
            return int(text.strip().replace(',', ''))
        except (ValueError, AttributeError):
            return 0
    
    def _parse_profile_info(self, soup: BeautifulSoup, user_id: str) -> Dict[str, Any]:
        """
        从页面解析作者个人信息
        
        Args:
            soup: BeautifulSoup 对象
            user_id: Google Scholar user ID
            
        Returns:
            包含个人信息的字典
        """
        profile = {
            "user_id": user_id,
            "url": f"https://scholar.google.com/citations?user={user_id}",
        }
        
        # 作者名字
        name_elem = soup.select_one('div#gsc_prf_in')
        if name_elem:
            profile["name"] = name_elem.text.strip()
        
        # 组织/机构（核心字段）
        affiliation_elem = soup.select_one('div.gsc_prf_il')
        if affiliation_elem:
            profile["affiliation"] = affiliation_elem.get_text(strip=True)
        
        # 邮箱验证信息
        email_elem = soup.select_one('div#gsc_prf_ivh')
        if email_elem:
            profile["verified_email"] = email_elem.get_text(strip=True)
        
        # 研究兴趣/领域
        interests = []
        interest_elems = soup.select('a.gsc_prf_inta')
        for elem in interest_elems:
            interests.append(elem.text.strip())
        profile["interests"] = interests
        
        # 头像 URL
        photo_elem = soup.select_one('img#gsc_prf_pup-img')
        if photo_elem:
            profile["photo_url"] = photo_elem.get('src', '')
        
        # 引用统计
        stats_table = soup.select('td.gsc_rsb_std')
        if len(stats_table) >= 6:
            profile["citations"] = {
                "total": self._safe_int(stats_table[0].text),
                "since_year": self._safe_int(stats_table[1].text),
            }
            profile["h_index"] = {
                "total": self._safe_int(stats_table[2].text),
                "since_year": self._safe_int(stats_table[3].text),
            }
            profile["i10_index"] = {
                "total": self._safe_int(stats_table[4].text),
                "since_year": self._safe_int(stats_table[5].text),
            }
        
        return profile
    
    def _parse_coauthors(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """解析共同作者列表"""
        coauthors = []
        coauthor_elems = soup.select('li.gsc_rsb_aa')
        
        for elem in coauthor_elems:
            coauthor = {}
            
            name_link = elem.select_one('a')
            if name_link:
                coauthor["name"] = name_link.text.strip()
                href = name_link.get('href', '')
                if 'user=' in href:
                    parsed = parse_qs(urlparse(href).query)
                    coauthor["user_id"] = parsed.get('user', [''])[0]
                    coauthor["url"] = f"https://scholar.google.com{href}"
            
            affiliation_elem = elem.select_one('span.gsc_rsb_a_ext')
            if affiliation_elem:
                coauthor["affiliation"] = affiliation_elem.text.strip()
            
            if coauthor.get("name"):
                coauthors.append(coauthor)
        
        return coauthors
    
    def get_organization(self, profile_url: str) -> Optional[str]:
        """
        获取 Google Scholar 作者的组织名
        
        优先从缓存读取，缓存未命中则访问网页
        
        Args:
            profile_url: Google Scholar 个人页面 URL
            
        Returns:
            组织名，如果无法获取则返回 None
        """
        user_id = _extract_user_id(profile_url)
        if not user_id:
            print(f"[ERROR] 无法从 URL 中提取 user ID: {profile_url}")
            return None
        
        if self.verbose:
            print(f"[INFO] 获取组织信息: user_id={user_id}")
        
        # 尝试从缓存读取
        if self.use_cache and self._cache:
            cached_data = self._cache.get(user_id)
            if cached_data:
                affiliation = cached_data.get("affiliation")
                if affiliation:
                    if self.verbose:
                        print(f"[CACHE] 命中缓存: {affiliation}")
                    return affiliation
                elif self.verbose:
                    print(f"[CACHE] 缓存中无组织信息")
        
        # 访问网页获取
        profile = self._fetch_and_cache_profile(user_id)
        if profile:
            return profile.get("affiliation")
        
        return None
    
    def get_profile_info(self, profile_url: str) -> Dict[str, Any]:
        """
        获取 Google Scholar 作者的完整个人信息
        
        优先从缓存读取，缓存未命中则访问网页
        
        Args:
            profile_url: Google Scholar 个人页面 URL
            
        Returns:
            包含个人信息的字典
        """
        user_id = _extract_user_id(profile_url)
        if not user_id:
            print(f"[ERROR] 无法从 URL 中提取 user ID: {profile_url}")
            return {}
        
        if self.verbose:
            print(f"[INFO] 获取个人信息: user_id={user_id}")
        
        # 尝试从缓存读取
        if self.use_cache and self._cache:
            cached_data = self._cache.get(user_id)
            if cached_data:
                if self.verbose:
                    print(f"[CACHE] 命中缓存: {cached_data.get('name', 'N/A')}")
                return cached_data
        
        # 访问网页获取
        return self._fetch_and_cache_profile(user_id)
    
    def _fetch_and_cache_profile(self, user_id: str) -> Dict[str, Any]:
        """
        访问网页获取 profile 并写入缓存
        
        Args:
            user_id: Google Scholar user ID
            
        Returns:
            个人信息字典
        """
        if self.verbose:
            print(f"[INFO] 正在访问 Google Scholar 页面...")
        
        url = f"https://scholar.google.com/citations?user={user_id}&hl=zh-CN"
        response = self._request_with_retry(url)
        
        if response is None:
            print(f"[ERROR] 无法获取作者页面: {user_id}")
            return {}
        
        # 检查是否被封锁
        if self._check_blocked(response.text):
            print("[ERROR] 访问被限制")
            # 尝试刷新 cookies 重试一次
            self._load_fake_cookies()
            response = self._request_with_retry(url)
            if response is None or self._check_blocked(response.text):
                print("[ERROR] 刷新 cookies 后仍被限制")
                return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 解析个人信息
        profile = self._parse_profile_info(soup, user_id)
        
        # 解析共同作者
        profile["coauthors"] = self._parse_coauthors(soup)
        
        # 添加元数据
        profile["fetched_at"] = datetime.now().isoformat()
        
        if self.verbose:
            print(f"[INFO] 作者: {profile.get('name', 'N/A')}")
            print(f"[INFO] 组织: {profile.get('affiliation', 'N/A')}")
        
        # 写入缓存
        if self.use_cache and self._cache:
            self._cache.set(user_id, profile, ttl=CACHE_TTL)
            if self.verbose:
                print(f"[CACHE] 已写入缓存: {user_id}")
        
        logger.info(f"获取 Google Scholar 组织信息: {user_id} -> {profile.get('affiliation', 'N/A')}")
        
        return profile


# ============ 便捷函数 ============

def get_organization(
    profile_url: str,
    use_cache: bool = True,
    verbose: bool = True,
) -> Optional[str]:
    """
    便捷函数：获取 Google Scholar 作者的组织名
    
    Args:
        profile_url: Google Scholar 个人页面 URL
        use_cache: 是否使用缓存
        verbose: 是否打印详细日志
        
    Returns:
        组织名，如果无法获取则返回 None
    """
    extractor = GoogleScholarOrgExtractor(
        use_cache=use_cache,
        verbose=verbose,
    )
    return extractor.get_organization(profile_url)


def get_profile_info(
    profile_url: str,
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    便捷函数：获取 Google Scholar 作者的完整个人信息
    
    Args:
        profile_url: Google Scholar 个人页面 URL
        use_cache: 是否使用缓存
        verbose: 是否打印详细日志
        
    Returns:
        包含个人信息的字典:
        - user_id: Google Scholar ID
        - name: 作者名字
        - affiliation: 组织/机构
        - verified_email: 验证邮箱信息
        - interests: 研究兴趣列表
        - citations: {"total": int, "since_year": int}
        - h_index: {"total": int, "since_year": int}
        - i10_index: {"total": int, "since_year": int}
        - coauthors: 共同作者列表
        - fetched_at: 抓取时间
    """
    extractor = GoogleScholarOrgExtractor(
        use_cache=use_cache,
        verbose=verbose,
    )
    return extractor.get_profile_info(profile_url)


def get_organization_by_user_id(
    user_id: str,
    use_cache: bool = True,
    verbose: bool = True,
) -> Optional[str]:
    """
    便捷函数：通过 user_id 获取组织名
    
    Args:
        user_id: Google Scholar user ID
        use_cache: 是否使用缓存
        verbose: 是否打印详细日志
        
    Returns:
        组织名
    """
    url = f"https://scholar.google.com/citations?user={user_id}"
    return get_organization(url, use_cache=use_cache, verbose=verbose)


if __name__ == "__main__":
    print("=" * 60)
    print("Google Scholar 组织信息提取工具")
    print("=" * 60)
    
    # 测试 URL
    test_url = "https://scholar.google.com/citations?user=GUaAHccAAAAJ&hl=zh-CN"
    
    print(f"\n测试 URL: {test_url}")
    print()
    
    # 测试1: 获取组织名
    print("[测试1] 获取组织名")
    print("-" * 40)
    org = get_organization(test_url)
    print(f"结果: {org}")
    
    # 测试2: 获取完整信息
    print()
    print("[测试2] 获取完整个人信息")
    print("-" * 40)
    profile = get_profile_info(test_url)
    
    if profile:
        print(f"  姓名: {profile.get('name', 'N/A')}")
        print(f"  组织: {profile.get('affiliation', 'N/A')}")
        print(f"  邮箱验证: {profile.get('verified_email', 'N/A')}")
        print(f"  研究兴趣: {', '.join(profile.get('interests', []))}")
        
        if 'citations' in profile:
            print(f"  总引用: {profile['citations'].get('total', 0)}")
        if 'h_index' in profile:
            print(f"  H-Index: {profile['h_index'].get('total', 0)}")
        if 'i10_index' in profile:
            print(f"  i10-Index: {profile['i10_index'].get('total', 0)}")
        
        coauthors = profile.get('coauthors', [])
        if coauthors:
            print(f"  共同作者数: {len(coauthors)}")
            for i, ca in enumerate(coauthors[:3], 1):
                print(f"    {i}. {ca.get('name', 'N/A')} - {ca.get('affiliation', 'N/A')}")
            if len(coauthors) > 3:
                print(f"    ... 还有 {len(coauthors) - 3} 位")
    
    # 测试3: 通过 user_id 获取
    print()
    print("[测试3] 通过 user_id 获取")
    print("-" * 40)
    org2 = get_organization_by_user_id("GUaAHccAAAAJ")
    print(f"结果: {org2}")

