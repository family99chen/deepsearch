"""
根据 ORCID 信息查找对应的 Google Scholar 账号
通过论文标题匹配来验证身份
"""

from typing import List, Dict, Optional, Tuple

# 导入已有的模块
from fetch_google_scholar_name_list import GoogleScholarAuthorScraper
from fetch_author_paper_list import get_author_papers
from verify_author_info import verify_author_identity


def find_google_scholar_account(
    author_name: str,
    orcid_id: str = None,
    orcid_papers: List[dict] = None,
    max_candidates: int = 20,
    match_threshold: float = 0.85,
    max_matches: int = 10,
    verbose: bool = True
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    根据作者名字和 ORCID 信息查找对应的 Google Scholar 账号
    
    Args:
        author_name: 作者名字（用于在 Google Scholar 搜索）
        orcid_id: ORCID ID（如果未提供 orcid_papers，则必须提供）
        orcid_papers: 已获取的 ORCID 论文列表（可选，传入后不再调用 ORCID API）
        max_candidates: 最大搜索候选人数量
        match_threshold: 标题匹配相似度阈值
        max_matches: 每个候选人最大匹配论文数量
        verbose: 是否打印详细信息
        
    Returns:
        (匹配的 Google Scholar URL, 匹配的作者信息字典)
        如果未找到，返回 (None, None)
    """
    # 参数验证
    if orcid_papers is None and orcid_id is None:
        raise ValueError("必须提供 orcid_id 或 orcid_papers 其中之一")
    
    if verbose:
        print("=" * 60)
        print("Google Scholar 账号查找工具")
        print("=" * 60)
        print(f"作者名字: {author_name}")
        if orcid_id:
            print(f"ORCID ID: {orcid_id}")
        print()
    
    # 1. 获取 ORCID 论文列表（如果未传入）
    if orcid_papers is None:
        if verbose:
            print("[STEP 1] 获取 ORCID 论文列表...")
        try:
            orcid_papers = get_author_papers(orcid_id)
            if verbose:
                print(f"[INFO] ORCID 共 {len(orcid_papers)} 篇论文\n")
        except Exception as e:
            print(f"[ERROR] 获取 ORCID 论文失败: {e}")
            return None, None
        
        if not orcid_papers:
            print("[WARNING] ORCID 论文列表为空，无法进行匹配")
            return None, None
    else:
        if verbose:
            print(f"[STEP 1] 使用传入的 ORCID 论文列表 ({len(orcid_papers)} 篇)\n")
    
    # 2. 搜索 Google Scholar 候选人
    if verbose:
        print("[STEP 2] 搜索 Google Scholar 候选人...")
    
    scraper = GoogleScholarAuthorScraper()
    
    if not scraper.cookies_valid:
        print("[ERROR] Cookies 无效，请先运行登录流程")
        return None, None
    
    candidates = scraper.search_author(author_name, max_results=max_candidates)
    
    if not candidates:
        print("[WARNING] 未找到任何候选人")
        return None, None
    
    if verbose:
        print(f"[INFO] 找到 {len(candidates)} 个候选人\n")
    
    # 3. 遍历候选人进行验证
    if verbose:
        print("[STEP 3] 验证候选人身份...")
        print("-" * 40)
    
    for i, candidate in enumerate(candidates, 1):
        candidate_url = candidate.get('url')
        candidate_name = candidate.get('name', 'Unknown')
        candidate_affiliation = candidate.get('affiliation', '')
        
        if not candidate_url:
            continue
        
        if verbose:
            print(f"\n[{i}/{len(candidates)}] 验证: {candidate_name}")
            print(f"    机构: {candidate_affiliation}")
            print(f"    URL: {candidate_url}")
        
        try:
            # 调用验证函数
            is_same, matches = verify_author_identity(
                scholar_profile_url=candidate_url,
                orcid_papers=orcid_papers,
                match_threshold=match_threshold,
                max_matches=max_matches,
                verbose=False  # 验证时不打印详细信息
            )
            
            if is_same:
                if verbose:
                    print(f"    ✓ 匹配成功！找到 {len(matches)} 篇相同论文")
                    print()
                    print("=" * 60)
                    print("查找结果")
                    print("=" * 60)
                    print(f"找到匹配的 Google Scholar 账号：")
                    print(f"  姓名: {candidate_name}")
                    print(f"  机构: {candidate_affiliation}")
                    print(f"  URL: {candidate_url}")
                    print(f"  匹配论文数: {len(matches)}")
                
                return candidate_url, candidate
            else:
                if verbose:
                    print(f"    ✗ 不匹配")
                    
        except Exception as e:
            if verbose:
                print(f"    ✗ 验证出错: {e}")
            continue
    
    # 未找到匹配
    if verbose:
        print()
        print("=" * 60)
        print("查找结果")
        print("=" * 60)
        print("未找到匹配的 Google Scholar 账号")
    
    return None, None


def find_google_scholar_account_from_candidates(
    candidates: List[Dict],
    orcid_papers: List[dict],
    match_threshold: float = 0.85,
    max_matches: int = 10,
    verbose: bool = True
) -> Tuple[Optional[str], Optional[Dict], int]:
    """
    从已有的候选人列表中查找匹配的 Google Scholar 账号
    
    Args:
        candidates: 候选人列表，每个候选人是一个字典，包含 url, name, affiliation 等
        orcid_papers: ORCID 论文列表
        match_threshold: 标题匹配相似度阈值
        max_matches: 每个候选人最大匹配论文数量
        verbose: 是否打印详细信息
        
    Returns:
        (匹配的 Google Scholar URL, 匹配的候选人信息, 匹配的论文数量)
        如果未找到，返回 (None, None, 0)
    """
    if not candidates:
        if verbose:
            print("[WARNING] 候选人列表为空")
        return None, None, 0
    
    if not orcid_papers:
        if verbose:
            print("[WARNING] ORCID 论文列表为空，无法进行匹配")
        return None, None, 0
    
    if verbose:
        print(f"[INFO] 开始验证 {len(candidates)} 个候选人...")
        print(f"[INFO] ORCID 论文数量: {len(orcid_papers)}")
        print("-" * 40)
    
    for i, candidate in enumerate(candidates, 1):
        # 支持两种格式：字典或纯 URL 字符串
        if isinstance(candidate, str):
            url = candidate
            name = "Unknown"
            affiliation = ""
        else:
            url = candidate.get('url', '')
            name = candidate.get('name', 'Unknown')
            affiliation = candidate.get('affiliation', '')
        
        if not url:
            continue
        
        if verbose:
            print(f"\n[{i}/{len(candidates)}] 验证: {name}")
            print(f"    机构: {affiliation}")
            print(f"    URL: {url}")
        
        try:
            is_same, matches = verify_author_identity(
                scholar_profile_url=url,
                orcid_papers=orcid_papers,
                match_threshold=match_threshold,
                max_matches=max_matches,
                verbose=False
            )
            
            if is_same:
                if verbose:
                    print(f"    ✓ 匹配成功！找到 {len(matches)} 篇相同论文")
                return url, candidate if isinstance(candidate, dict) else {'url': url}, len(matches)
            else:
                if verbose:
                    print(f"    ✗ 不匹配")
                    
        except Exception as e:
            if verbose:
                print(f"    ✗ 验证出错: {e}")
            continue
    
    if verbose:
        print("\n[INFO] 未找到匹配的账号")
    
    return None, None, 0


if __name__ == "__main__":
    # 示例用法
    
    # 测试数据
    test_author_name = "Kenji Watanabe"
    test_orcid = "0000-0003-3701-8119"
    
    print("=" * 60)
    print("Google Scholar 账号查找工具 - 示例")
    print("=" * 60)
    print()
    
    # 方式1：直接传入作者名字和 ORCID ID
    matched_url, matched_author = find_google_scholar_account(
        author_name=test_author_name,
        orcid_id=test_orcid,
        max_candidates=10,
        match_threshold=0.85,
        max_matches=10,
        verbose=True
    )
    
    print()
    if matched_url:
        print(f"✓ 找到 Google Scholar 账号: {matched_url}")
    else:
        print("✗ 未找到匹配的 Google Scholar 账号")
    
    # 方式2：先获取论文列表，然后复用
    # orcid_papers = get_author_papers(test_orcid)
    # matched_url, matched_author = find_google_scholar_account(
    #     author_name=test_author_name,
    #     orcid_papers=orcid_papers,  # 传入已获取的论文列表
    #     verbose=True
    # )

