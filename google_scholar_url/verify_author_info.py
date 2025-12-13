"""
验证 Google Scholar 作者与 ORCID 作者是否为同一人
通过模糊匹配论文标题来判断
"""

import re
import math
from typing import List, Tuple
from difflib import SequenceMatcher

# 导入已有的模块
from fetch_author_paper_list import get_author_papers
from fetch_author_google_paper_list import get_google_scholar_papers


def normalize_title(title: str) -> str:
    """
    标准化论文标题用于比较
    
    Args:
        title: 原始标题
        
    Returns:
        标准化后的标题（小写、去除特殊字符）
    """
    if not title:
        return ""
    
    # 转小写
    title = title.lower()
    
    # 去除 HTML 标签
    title = re.sub(r'<[^>]+>', '', title)
    
    # 去除特殊字符，只保留字母数字和空格
    title = re.sub(r'[^a-z0-9\s]', ' ', title)
    
    # 合并多个空格
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title


def fuzzy_match_title(title1: str, title2: str, threshold: float = 0.85) -> bool:
    """
    模糊匹配两个标题
    
    Args:
        title1: 第一个标题
        title2: 第二个标题
        threshold: 相似度阈值 (0-1)
        
    Returns:
        是否匹配
    """
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    
    if not norm1 or not norm2:
        return False
    
    # 完全匹配
    if norm1 == norm2:
        return True
    
    # 包含关系（一个是另一个的子串）
    if norm1 in norm2 or norm2 in norm1:
        # 如果较短的至少有20个字符，认为匹配
        if min(len(norm1), len(norm2)) >= 20:
            return True
    
    # 使用 SequenceMatcher 计算相似度
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    return similarity >= threshold


def verify_author_identity(
    scholar_profile_url: str,
    orcid_id: str = None,
    orcid_papers: List[dict] = None,
    match_threshold: float = 0.85,
    max_matches: int = 10,
    verbose: bool = True
) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    验证 Google Scholar 作者与 ORCID 作者是否为同一人
    
    Args:
        scholar_profile_url: Google Scholar 个人主页 URL
        orcid_id: ORCID ID（如果未提供 orcid_papers，则必须提供）
        orcid_papers: 已获取的 ORCID 论文列表（可选，传入后不再调用 ORCID API）
        match_threshold: 标题匹配相似度阈值
        max_matches: 最大匹配数量，达到后停止匹配以节约时间
        verbose: 是否打印详细信息
        
    Returns:
        (是否为同一人, 匹配的论文列表[(scholar_title, orcid_title), ...])
    """
    # 参数验证
    if orcid_papers is None and orcid_id is None:
        raise ValueError("必须提供 orcid_id 或 orcid_papers 其中之一")
    
    if verbose:
        print("=" * 60)
        print("开始验证作者身份")
        print("=" * 60)
        print(f"Google Scholar: {scholar_profile_url}")
        if orcid_id:
            print(f"ORCID: {orcid_id}")
        if orcid_papers:
            print(f"ORCID 论文列表: 已传入 {len(orcid_papers)} 篇")
        print()
    
    # 1. 获取 Google Scholar 论文列表
    if verbose:
        print("[STEP 1] 获取 Google Scholar 论文列表...")
    
    scholar_papers = get_google_scholar_papers(scholar_profile_url)
    scholar_titles = [p.get('title', '') for p in scholar_papers if p.get('title')]
    
    if verbose:
        print(f"[INFO] Google Scholar 共 {len(scholar_titles)} 篇论文\n")
    
    if not scholar_titles:
        print("[WARNING] 未获取到 Google Scholar 论文")
        return False, []
    
    # 2. 获取 ORCID 论文列表（如果未传入）
    if orcid_papers is not None:
        # 使用传入的论文列表
        if verbose:
            print("[STEP 2] 使用传入的 ORCID 论文列表...")
        orcid_titles = [p.get('title', '') for p in orcid_papers if p.get('title')]
        if verbose:
            print(f"[INFO] ORCID 共 {len(orcid_titles)} 篇论文\n")
    else:
        # 调用 API 获取
        if verbose:
            print("[STEP 2] 获取 ORCID 论文列表...")
        try:
            orcid_papers = get_author_papers(orcid_id)
            orcid_titles = [p.get('title', '') for p in orcid_papers if p.get('title')]
            
            if verbose:
                print(f"[INFO] ORCID 共 {len(orcid_titles)} 篇论文\n")
        except Exception as e:
            print(f"[ERROR] 获取 ORCID 论文失败: {e}")
            return False, []
    
    if not orcid_titles:
        print("[WARNING] 未获取到 ORCID 论文")
        return False, []
    
    # 3. 计算所需匹配数量
    base_count = min(len(scholar_titles), len(orcid_titles))
    
    # 如果双方论文数都超过 1000，只需要 10% 重合；否则需要 30%
    if len(scholar_titles) >= 1000 and len(orcid_titles) >= 1000:
        required_ratio = 0.10
    else:
        required_ratio = 0.30
    
    required_matches = math.ceil(base_count * required_ratio)  # 向上取整，1篇也必须匹配
    # 目标匹配数：取 max_matches 和 required_matches 中较小的，达到即停止
    target_matches = min(max_matches, required_matches)
    
    # 4. 匹配论文标题
    if verbose:
        print(f"[STEP 3] 匹配论文标题（目标 {target_matches} 篇，阈值 {required_matches}，上限 {max_matches}）...")
    
    matched_papers = []
    
    for scholar_title in scholar_titles:
        # 达到目标匹配数即停止
        if len(matched_papers) >= target_matches:
            if verbose:
                print(f"[INFO] 已达到目标匹配数 {target_matches}，停止匹配")
            break
        
        for orcid_title in orcid_titles:
            if fuzzy_match_title(scholar_title, orcid_title, match_threshold):
                matched_papers.append((scholar_title, orcid_title))
                if verbose:
                    print(f"  ✓ 匹配成功 ({len(matched_papers)}/{target_matches}):")
                    print(f"    Scholar: {scholar_title[:60]}...")
                    print(f"    ORCID:   {orcid_title[:60]}...")
                break  # 一篇 scholar 论文只匹配一次
    
    # 5. 判断结果
    # 达到目标匹配数即为同一人
    is_same_person = len(matched_papers) >= target_matches
    
    if verbose:
        print()
        print("=" * 60)
        print("验证结果")
        print("=" * 60)
        print(f"基数（较少一方）: {base_count} 篇")
        print(f"要求重合比例: {required_ratio:.0%} = {required_matches} 篇")
        print(f"目标匹配数量: {target_matches} 篇 (min(阈值{required_matches}, 上限{max_matches}))")
        print(f"实际匹配数量: {len(matched_papers)} 篇")
        print(f"是否为同一人: {'✓ 是' if is_same_person else '✗ 否'}")
    
    return is_same_person, matched_papers


if __name__ == "__main__":
    # 示例用法
    # 替换为实际的 Google Scholar 主页 URL 和 ORCID ID
    
    test_scholar_url = "https://scholar.google.com/citations?hl=zh-CN&user=DsUCHdUAAAAJ"  # kenji watanabe
    test_orcid = "0000-0003-3701-8119"  # 示例 ORCID
    
    print("=" * 60)
    print("作者身份验证工具")
    print("=" * 60)
    print()
    
    # 方式1：传入 orcid_id，自动获取论文列表
    # is_same, matches = verify_author_identity(
    #     scholar_profile_url=test_scholar_url,
    #     orcid_id=test_orcid,
    #     match_threshold=0.85,
    #     max_matches=10,
    #     verbose=True
    # )
    
    # 方式2：先获取 ORCID 论文列表，然后传入（推荐，可复用）
    print("[预处理] 先获取 ORCID 论文列表...")
    orcid_papers = get_author_papers(test_orcid)
    print(f"[预处理] 获取到 {len(orcid_papers)} 篇论文\n")
    
    # 验证时直接传入论文列表，不再重复调用 API
    is_same, matches = verify_author_identity(
        scholar_profile_url=test_scholar_url,
        orcid_papers=orcid_papers,  # 直接传入论文列表
        match_threshold=0.85,
        max_matches=10,
        verbose=True
    )
    
    print()
    if is_same:
        print(f"结论: 确认为同一人，共有 {len(matches)} 篇论文匹配")
    else:
        print("结论: 无法确认为同一人，没有找到匹配的论文")
