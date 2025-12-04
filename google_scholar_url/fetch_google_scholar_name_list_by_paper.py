"""
通过论文标题搜索 Google Scholar 作者

使用 SerpAPI Google Scholar 搜索论文，从搜索结果中提取作者信息

特性：
- 通过论文标题精确搜索
- 从论文结果中提取有 Scholar 主页的作者
- 支持作者名字过滤
"""

import sys
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 SerpAPI
from google_search_api.serpapi_google_scholar import SerpAPIScholar

# 导入作者名字过滤器
from google_scholar_url.author_name_filter import is_same_author


class GoogleScholarAuthorByPaper:
    """
    通过论文搜索 Google Scholar 作者
    
    从论文搜索结果中提取作者的 Scholar 主页 URL
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        """
        初始化
        
        Args:
            api_key: SerpAPI API Key（不提供则从配置读取）
            max_retries: 最大重试次数
            verbose: 是否打印详细信息
        """
        self.client = SerpAPIScholar(
            api_key=api_key,
            max_retries=max_retries,
            verbose=verbose
        )
        self.verbose = verbose
    
    def search_author_by_paper(
        self,
        paper_title: str,
        author_name: Optional[str] = None,
        num: int = 20,
        filter_by_name: bool = True,
    ) -> List[Dict]:
        """
        通过论文标题搜索作者
        
        Args:
            paper_title: 论文标题
            author_name: 作者名字（可选，用于过滤和组合搜索）
            num: 搜索结果数量
            filter_by_name: 是否用作者名字过滤结果
            
        Returns:
            作者信息列表，每个包含:
            - name: 作者名
            - author_id: Google Scholar ID
            - url: Scholar 主页 URL
        """
        # 构建查询
        if author_name:
            query = f'author:"{author_name}" "{paper_title}"'
        else:
            query = f'"{paper_title}"'
        
        print(f"[INFO] 搜索论文: {paper_title[:50]}...")
        if author_name:
            print(f"[INFO] 限定作者: {author_name}")
        
        # 搜索
        result = self.client.search(query, num=num)
        
        if not result.success:
            print(f"[ERROR] 搜索失败: {result.error}")
            return []
        
        if not result.items:
            print("[INFO] 没有搜索结果")
            return []
        
        # 提取作者
        all_authors = []
        seen_ids = set()
        
        for paper in result.items:
            # 从论文的 authors 字段提取作者信息
            for author in paper.authors:
                author_id = author.get("author_id")
                
                # 必须有 author_id 才说明有 Scholar 主页
                if not author_id:
                    continue
                
                # 去重
                if author_id in seen_ids:
                    continue
                
                author_name_result = author.get("name", "")
                author_link = author.get("link", "")
                
                # 如果没有 link，用 author_id 构建
                if not author_link and author_id:
                    author_link = f"https://scholar.google.com/citations?user={author_id}&hl=zh-CN"
                
                # 名字过滤
                if filter_by_name and author_name:
                    is_match, _ = is_same_author(author_name, author_name_result, verbose=False)
                    if not is_match:
                        if self.verbose:
                            print(f"  - 跳过（名字不匹配）: {author_name_result}")
                        continue
                
                seen_ids.add(author_id)
                author_info = {
                    "name": author_name_result,
                    "author_id": author_id,
                    "url": author_link,
                }
                all_authors.append(author_info)
                print(f"  - 找到作者: {author_name_result} (ID: {author_id})")
        
        print(f"[INFO] 共找到 {len(all_authors)} 个有 Scholar 主页的作者")
        return all_authors
    
    def search_author_by_papers(
        self,
        paper_titles: List[str],
        author_name: Optional[str] = None,
        filter_by_name: bool = True,
    ) -> List[Dict]:
        """
        通过多篇论文搜索作者（去重合并）
        
        Args:
            paper_titles: 论文标题列表
            author_name: 作者名字（可选）
            filter_by_name: 是否用作者名字过滤
            
        Returns:
            去重后的作者列表
        """
        all_authors = []
        seen_ids = set()
        
        for i, title in enumerate(paper_titles, 1):
            print(f"\n[{i}/{len(paper_titles)}] 搜索论文...")
            authors = self.search_author_by_paper(
                paper_title=title,
                author_name=author_name,
                filter_by_name=filter_by_name
            )
            
            for author in authors:
                if author["author_id"] not in seen_ids:
                    seen_ids.add(author["author_id"])
                    all_authors.append(author)
        
        print(f"\n[总计] 从 {len(paper_titles)} 篇论文中找到 {len(all_authors)} 个不同作者")
        return all_authors
    
    def get_author_urls(
        self,
        paper_title: str,
        author_name: Optional[str] = None,
        filter_by_name: bool = True,
    ) -> List[str]:
        """
        只返回作者 URL 列表
        
        Args:
            paper_title: 论文标题
            author_name: 作者名字（可选）
            filter_by_name: 是否用作者名字过滤
            
        Returns:
            Scholar 主页 URL 列表
        """
        authors = self.search_author_by_paper(
            paper_title=paper_title,
            author_name=author_name,
            filter_by_name=filter_by_name
        )
        return [a["url"] for a in authors]


# 便捷函数
def search_author_by_paper(
    paper_title: str,
    author_name: Optional[str] = None,
    filter_by_name: bool = True,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """通过论文标题搜索作者"""
    searcher = GoogleScholarAuthorByPaper(api_key=api_key)
    return searcher.search_author_by_paper(
        paper_title=paper_title,
        author_name=author_name,
        filter_by_name=filter_by_name
    )


def get_author_urls_by_paper(
    paper_title: str,
    author_name: Optional[str] = None,
    filter_by_name: bool = True,
    api_key: Optional[str] = None,
) -> List[str]:
    """通过论文标题获取作者 URL 列表"""
    searcher = GoogleScholarAuthorByPaper(api_key=api_key)
    return searcher.get_author_urls(
        paper_title=paper_title,
        author_name=author_name,
        filter_by_name=filter_by_name
    )


if __name__ == "__main__":
    print("=" * 60)
    print("通过论文搜索 Google Scholar 作者")
    print("=" * 60)
    
    # 测试
    searcher = GoogleScholarAuthorByPaper(verbose=True)
    
    # 测试1: 只用论文标题
    print("\n[测试1] 只用论文标题搜索")
    paper = "Quantum spin Hall effect in two-dimensional transition metal dichalcogenides"
    authors = searcher.search_author_by_paper(paper)
    
    if authors:
        print("\n找到的作者:")
        for i, a in enumerate(authors, 1):
            print(f"  {i}. {a['name']}")
            print(f"     URL: {a['url']}")
    
    # 测试2: 论文标题 + 作者名
    print("\n" + "=" * 60)
    print("[测试2] 论文标题 + 作者名搜索")
    authors = searcher.search_author_by_paper(
        paper_title=paper,
        author_name="Ju Li",
        filter_by_name=True
    )
    
    if authors:
        print("\n找到的作者:")
        for i, a in enumerate(authors, 1):
            print(f"  {i}. {a['name']}")
            print(f"     URL: {a['url']}")

