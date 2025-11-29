"""
从 Google Scholar 个人页面获取作者的论文列表
"""

import os
import sys
import json
import time
import requests
from typing import List, Optional
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

# 添加项目根目录到 path，以便导入 proxy 模块
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/google_scholar_url/', 1)[0])

# 导入代理模块
try:
    from proxy import configure_session
except ImportError:
    def configure_session(session): return session


class GoogleScholarProfileScraper:
    """Google Scholar 个人主页文章爬虫"""
    
    # 默认 cookies 路径：脚本所在目录
    DEFAULT_COOKIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_cookies.json")
    
    def __init__(self, cookies_path: str = None):
        """
        初始化爬虫
        
        Args:
            cookies_path: cookies 文件路径，默认为脚本所在目录下的 google_cookies.json
        """
        self.cookies_path = cookies_path or self.DEFAULT_COOKIES_PATH
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
    
    def get_all_papers(self, profile_url: str) -> List[dict]:
        """
        获取作者主页的所有论文
        
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
        
        while True:
            # 构建请求 URL
            url = (f"https://scholar.google.com/citations?"
                   f"user={user_id}&hl=zh-CN&cstart={cstart}&pagesize={pagesize}")
            
            print(f"[INFO] 正在获取第 {cstart // pagesize + 1} 页 (cstart={cstart})...")
            
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                time.sleep(2)  # 礼貌性延迟
                
            except requests.RequestException as e:
                print(f"[ERROR] 请求失败: {e}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
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


def get_google_scholar_papers(profile_url: str, cookies_path: str = "google_cookies.json") -> List[dict]:
    """
    便捷函数：获取 Google Scholar 作者的所有论文
    
    Args:
        profile_url: 作者 Google Scholar 主页 URL
        cookies_path: cookies 文件路径
        
    Returns:
        论文列表
    """
    scraper = GoogleScholarProfileScraper(cookies_path=cookies_path)
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

