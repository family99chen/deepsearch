"""
从 Google Scholar 个人页面获取作者的论文列表

特性：
- 支持指数退避重试
- 支持代理配置
- 自动分页获取所有论文
"""

import os
import sys
import json
import time
import random
import requests
from typing import List, Optional
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from pathlib import Path

# 添加项目根目录到 path，以便导入 proxy 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入代理模块
try:
    from proxy import configure_session
except ImportError:
    def configure_session(session): return session

# 导入重试模块
try:
    from utils.retry import DEFAULT_RETRYABLE_EXCEPTIONS, DEFAULT_RETRYABLE_STATUS_CODES
except ImportError:
    DEFAULT_RETRYABLE_EXCEPTIONS = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )
    DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


class GoogleScholarProfileScraper:
    """Google Scholar 个人主页文章爬虫（带指数退避重试）"""
    
    # 默认 cookies 路径：脚本所在目录
    DEFAULT_COOKIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_cookies.json")
    
    def __init__(
        self, 
        cookies_path: str = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
    ):
        """
        初始化爬虫
        
        Args:
            cookies_path: cookies 文件路径，默认为脚本所在目录下的 google_cookies.json
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
        """
        self.cookies_path = cookies_path or self.DEFAULT_COOKIES_PATH
        
        # 重试配置
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        
        # 配置代理（如果启用）
        configure_session(self.session)
        
        # 加载 cookies
        self._load_cookies()
    
    def _load_cookies(self):
        """加载 cookies"""
        if os.path.exists(self.cookies_path):
            try:
                with open(self.cookies_path, 'r', encoding='utf-8') as f:
                    cookies = json.load(f)
                for cookie in cookies:
                    self.session.cookies.set(
                        cookie['name'],
                        cookie['value'],
                        domain=cookie.get('domain', '.google.com')
                    )
                print(f"[INFO] 已加载 cookies: {self.cookies_path}")
            except Exception as e:
                print(f"[WARNING] 加载 cookies 失败: {e}")
    
    def _request_with_retry(self, url: str, timeout: int = 15) -> Optional[requests.Response]:
        """
        带指数退避重试的请求
        
        Args:
            url: 请求 URL
            timeout: 超时时间
            
        Returns:
            Response 对象，如果失败返回 None
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=timeout)
                
                # 检查是否需要重试的状态码
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
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        计算指数退避延迟时间
        
        Args:
            attempt: 当前尝试次数（从0开始）
            
        Returns:
            延迟时间（秒）
        """
        delay = min(
            self.base_delay * (2 ** attempt),
            self.max_delay
        )
        # 添加随机抖动 (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter
    
    def get_all_papers(self, profile_url: str) -> List[dict]:
        """
        获取作者主页的所有论文（带重试）
        
        Args:
            profile_url: 作者 Google Scholar 主页 URL
            
        Returns:
            论文列表，每个论文包含 title, year, citations, authors, venue 等
        """
        # 解析 URL 获取 user ID
        parsed = urlparse(profile_url)
        params = parse_qs(parsed.query)
        user_id = params.get('user', [None])[0]
        
        if not user_id:
            print(f"[ERROR] 无法从 URL 中提取 user ID: {profile_url}")
            return []
        
        print(f"[INFO] 正在获取作者文章列表, user_id: {user_id}")
        
        all_papers = []
        cstart = 0  # 分页起始位置
        pagesize = 100  # 每页数量
        consecutive_failures = 0  # 连续失败次数
        max_consecutive_failures = 2  # 最大连续失败次数
        
        while True:
            # 构建请求 URL
            url = (f"https://scholar.google.com/citations?"
                   f"user={user_id}&hl=zh-CN&cstart={cstart}&pagesize={pagesize}")
            
            print(f"[INFO] 正在获取第 {cstart // pagesize + 1} 页 (cstart={cstart})...")
            
            # 使用带重试的请求
            response = self._request_with_retry(url)
            
            if response is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"[ERROR] 连续 {consecutive_failures} 次请求失败，停止获取")
                    break
                print(f"[WARNING] 请求失败，跳过当前页 ({consecutive_failures}/{max_consecutive_failures})")
                cstart += pagesize
                continue
            
            # 重置连续失败计数
            consecutive_failures = 0
            
            # 礼貌性延迟
            time.sleep(2)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 检查是否被封锁
            if self._check_blocked(response.text):
                print("[ERROR] 访问被限制，请重新登录获取 cookies")
                break
            
            # 解析论文列表
            paper_rows = soup.select('tr.gsc_a_tr')
            
            if not paper_rows:
                print("[INFO] 没有更多论文")
                break
            
            for row in paper_rows:
                paper = self._parse_paper_row(row)
                if paper:
                    all_papers.append(paper)
            
            print(f"[INFO] 已获取 {len(all_papers)} 篇论文")
            
            # 检查是否还有下一页
            if len(paper_rows) < pagesize:
                print("[INFO] 已获取全部论文")
                break
            
            # 检查是否有"显示更多"按钮被禁用
            show_more = soup.select_one('button#gsc_bpf_more')
            if show_more and show_more.get('disabled'):
                print("[INFO] 已到最后一页")
                break
            
            cstart += pagesize
            time.sleep(1)  # 额外延迟
        
        return all_papers
    
    def _check_blocked(self, html_text: str) -> bool:
        """
        检查是否被 Google Scholar 封锁
        
        Args:
            html_text: HTML 文本
            
        Returns:
            是否被封锁
        """
        # 检查是否有验证码
        if 'captcha' in html_text.lower() or 'recaptcha' in html_text.lower():
            return True
        
        # 检查是否是异常流量页面
        if 'unusual traffic' in html_text.lower():
            return True
        
        # 检查是否是登录页面
        if 'accounts.google.com' in html_text:
            return True
        
        return False
    
    def _parse_paper_row(self, row) -> Optional[dict]:
        """
        解析单行论文信息
        
        Args:
            row: BeautifulSoup 的 tr 元素
            
        Returns:
            论文字典或 None
        """
        paper = {}
        
        # 标题
        title_elem = row.select_one('a.gsc_a_at')
        if title_elem:
            paper['title'] = title_elem.text.strip()
            # 获取论文详情链接
            href = title_elem.get('href', '')
            if href:
                paper['detail_url'] = f"https://scholar.google.com{href}"
        else:
            return None
        
        # 作者和期刊信息
        gray_text = row.select('div.gs_gray')
        if len(gray_text) >= 1:
            paper['authors'] = gray_text[0].text.strip()
        if len(gray_text) >= 2:
            paper['venue'] = gray_text[1].text.strip()
        
        # 引用次数
        cite_elem = row.select_one('a.gsc_a_ac')
        if cite_elem and cite_elem.text.strip():
            paper['citations'] = cite_elem.text.strip()
        else:
            paper['citations'] = '0'
        
        # 年份
        year_elem = row.select_one('span.gsc_a_h.gsc_a_hc.gs_ibl')
        if year_elem:
            paper['year'] = year_elem.text.strip()
        
        return paper


def get_google_scholar_papers(
    profile_url: str, 
    cookies_path: str = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> List[dict]:
    """
    便捷函数：获取 Google Scholar 作者的所有论文（带重试）
    
    Args:
        profile_url: 作者 Google Scholar 主页 URL
        cookies_path: cookies 文件路径
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        
    Returns:
        论文列表
    """
    scraper = GoogleScholarProfileScraper(
        cookies_path=cookies_path,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    return scraper.get_all_papers(profile_url)


if __name__ == "__main__":
    # 示例用法
    test_url = "https://scholar.google.com/citations?hl=zh-CN&user=DsUCHdUAAAAJ"
    
    print("=" * 60)
    print("Google Scholar 论文列表获取工具")
    print("=" * 60)
    print(f"目标 URL: {test_url}")
    print()
    
    papers = get_google_scholar_papers(test_url)
    
    print()
    print("=" * 60)
    print(f"共获取 {len(papers)} 篇论文")
    print("=" * 60)
    
    # 显示前 10 篇
    for i, paper in enumerate(papers[:10], 1):
        print(f"\n[{i}] {paper.get('title', 'N/A')}")
        print(f"    作者: {paper.get('authors', 'N/A')}")
        print(f"    期刊: {paper.get('venue', 'N/A')}")
        print(f"    年份: {paper.get('year', 'N/A')}")
        print(f"    引用: {paper.get('citations', '0')}")
    
    if len(papers) > 10:
        print(f"\n... 还有 {len(papers) - 10} 篇论文未显示")
