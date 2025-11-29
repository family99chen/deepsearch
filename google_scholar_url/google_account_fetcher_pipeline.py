"""
Google Scholar 账号查找 Pipeline
输入 ORCID ID，输出对应的 Google Scholar 账号 URL

完整流程：
1. 通过 ORCID API 获取作者姓名
2. 用姓名在 Google Scholar 搜索候选人
3. 通过 ORCID API 获取作者论文列表
4. 遍历候选人，匹配论文标题，找到正确的 Google Scholar 账号
"""

import sys
from typing import Optional, Tuple, Dict

# 添加项目根目录到 path
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/google_scholar_url/', 1)[0])

# 导入日志模块
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 导入子模块
from fetch_author_person_info import get_author_name
from fetch_author_paper_list import get_author_papers
from fetch_google_scholar_name_list import GoogleScholarAuthorScraper
from fetch_author_google_scholar_account import find_google_scholar_account_from_candidates


def log_print(message: str, level: str = "info"):
    """同时打印和记录日志"""
    print(message)
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)


def find_google_scholar_by_orcid(
    orcid_id: str,
    max_candidates: int = 20,
    match_threshold: float = 0.85,
    max_matches: int = 10,
    verbose: bool = True
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    根据 ORCID ID 查找对应的 Google Scholar 账号
    
    Args:
        orcid_id: ORCID ID，格式如 "0000-0002-1825-0097"
        max_candidates: 最大搜索候选人数量
        match_threshold: 标题匹配相似度阈值
        max_matches: 每个候选人最大匹配论文数量
        verbose: 是否打印详细信息
        
    Returns:
        (匹配的 Google Scholar URL, 匹配的作者信息字典)
        如果未找到，返回 (None, None)
    """
    logger.info(f"开始查找 ORCID: {orcid_id}")
    
    if verbose:
        print("=" * 60)
        print("Google Scholar 账号查找 Pipeline")
        print("=" * 60)
        print(f"输入 ORCID: {orcid_id}")
        print()
    
    # ========== STEP 1: 获取作者姓名 ==========
    if verbose:
        print("[STEP 1/4] 获取作者姓名...")
    
    try:
        author_name_info = get_author_name(orcid_id)
        author_name = author_name_info.get('full_name')
        
        if not author_name:
            log_print("[ERROR] 无法获取作者姓名", "error")
            return None, None
        
        logger.info(f"作者姓名: {author_name}")
        if verbose:
            print(f"[INFO] 作者姓名: {author_name}")
            if author_name_info.get('credit_name'):
                print(f"[INFO] 署名: {author_name_info['credit_name']}")
            print()
            
    except Exception as e:
        log_print(f"[ERROR] 获取作者姓名失败: {e}", "error")
        return None, None
    
    # ========== STEP 2: 获取 ORCID 论文列表 ==========
    if verbose:
        print("[STEP 2/4] 获取 ORCID 论文列表...")
    
    try:
        orcid_papers = get_author_papers(orcid_id)
        
        if not orcid_papers:
            log_print("[WARNING] ORCID 论文列表为空，无法进行匹配", "warning")
            return None, None
        
        logger.info(f"获取到 {len(orcid_papers)} 篇 ORCID 论文")
        if verbose:
            print(f"[INFO] 获取到 {len(orcid_papers)} 篇论文")
            # 显示前3篇作为示例
            for i, paper in enumerate(orcid_papers[:3], 1):
                title = paper.get('title', 'N/A')
                print(f"    {i}. {title[:50]}...")
            if len(orcid_papers) > 3:
                print(f"    ... 还有 {len(orcid_papers) - 3} 篇")
            print()
            
    except Exception as e:
        log_print(f"[ERROR] 获取 ORCID 论文失败: {e}", "error")
        return None, None
    
    # ========== STEP 3: 搜索 Google Scholar 候选人 ==========
    if verbose:
        print("[STEP 3/4] 搜索 Google Scholar 候选人...")
    
    scraper = GoogleScholarAuthorScraper()
    
    if not scraper.cookies_valid:
        log_print("[ERROR] Google Scholar cookies 无效，请先运行登录流程", "error")
        print("[提示] 运行: python fetch_google_scholar_name_list.py")
        return None, None
    
    try:
        candidates = scraper.search_author(author_name, max_results=max_candidates)
        
        if not candidates:
            # 尝试使用 credit_name 搜索
            credit_name = author_name_info.get('credit_name')
            if credit_name and credit_name != author_name:
                logger.info(f"使用署名重新搜索: {credit_name}")
                if verbose:
                    print(f"[INFO] 使用署名重新搜索: {credit_name}")
                candidates = scraper.search_author(credit_name, max_results=max_candidates)
        
        if not candidates:
            log_print("[WARNING] 未找到任何候选人", "warning")
            return None, None
        
        logger.info(f"找到 {len(candidates)} 个候选人")
        if verbose:
            print(f"[INFO] 找到 {len(candidates)} 个候选人")
            print()
            
    except Exception as e:
        log_print(f"[ERROR] 搜索候选人失败: {e}", "error")
        return None, None
    
    # ========== STEP 4: 验证候选人 ==========
    if verbose:
        print("[STEP 4/4] 验证候选人身份...")
    
    matched_url, matched_candidate, match_count = find_google_scholar_account_from_candidates(
        candidates=candidates,
        orcid_papers=orcid_papers,
        match_threshold=match_threshold,
        max_matches=max_matches,
        verbose=verbose
    )
    
    # ========== 输出结果 ==========
    if verbose:
        print()
        print("=" * 60)
        print("查找结果")
        print("=" * 60)
    
    if matched_url:
        logger.info(f"找到匹配: {matched_url}")
        if verbose:
            print(f"✓ 找到匹配的 Google Scholar 账号")
            print(f"  URL: {matched_url}")
            if matched_candidate:
                print(f"  姓名: {matched_candidate.get('name', 'N/A')}")
                print(f"  机构: {matched_candidate.get('affiliation', 'N/A')}")
            print(f"  匹配论文数: {match_count}")
        return matched_url, matched_candidate
    else:
        logger.warning(f"未找到匹配的 Google Scholar 账号: {orcid_id}")
        if verbose:
            print("✗ 未找到匹配的 Google Scholar 账号")
        return None, None


if __name__ == "__main__":
    # 示例用法
    
    # 测试 ORCID ID
    test_orcid = "0000-0002-7841-8058"
    
    print()
    print("=" * 60)
    print("Google Scholar 账号查找工具")
    print("=" * 60)
    print()
    print(f"测试 ORCID: {test_orcid}")
    print()
    
    # 执行查找
    url, author_info = find_google_scholar_by_orcid(
        orcid_id=test_orcid,
        max_candidates=15,
        match_threshold=0.85,
        max_matches=10,
        verbose=True
    )
    
    print()
    print("-" * 60)
    if url:
        print(f"最终结果: {url}")
    else:
        print("最终结果: Not Available")
