"""
Google Scholar 账号查找 Pipeline
输入 ORCID ID，输出对应的 Google Scholar 账号 URL

完整流程：
1. 检查缓存（如果有直接返回）
2. 通过 ORCID API 获取作者姓名
3. 通过 ORCID API 获取作者论文列表
4. 用姓名在 Google Scholar 搜索候选人（迭代搜索）
5. 遍历候选人，匹配论文标题，找到正确的 Google Scholar 账号
6. 如果未找到，继续搜索下一批候选人，直到找到或达到迭代上限
7. 找到后存入缓存
"""

import sys
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

# 添加项目根目录到 path
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/google_scholar_url/', 1)[0])

# 导入日志模块
try:
    from utils.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 导入缓存模块
HAS_CACHE = False
try:
    from localdb.insert_mongo import MongoCache
    HAS_CACHE = True
except ImportError:
    pass

# 导入统计模块
HAS_STATS = False
try:
    from utils.pipeline_stats import (
        record_request,
        record_cache_hit,
        record_error,
        record_name_search_success,
        record_paper_search_success,
        record_not_found
    )
    HAS_STATS = True
except ImportError:
    # 如果统计模块不可用，使用空函数
    def record_request(): pass
    def record_cache_hit(): pass
    def record_error(): pass
    def record_name_search_success(iterations): pass
    def record_paper_search_success(papers_searched): pass
    def record_not_found(): pass

# 缓存配置
CACHE_COLLECTION = "orcid_googleaccount_map"
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

# 导入子模块
from fetch_author_person_info import get_author_name
from fetch_author_paper_list import get_author_papers
from fetch_google_scholar_name_list import GoogleScholarAuthorScraper
from fetch_author_google_scholar_account import find_google_scholar_account_from_candidates
from fetch_person_organization import get_author_organization
from fetch_google_scholar_name_list_by_paper import GoogleScholarAuthorByPaper
from name_normalization import normalize_name


def load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


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
    max_iterations: int = None,
    match_threshold: float = None,
    max_matches: int = None,
    verbose: bool = True,
    use_cache: bool = True
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    根据 ORCID ID 查找对应的 Google Scholar 账号（迭代搜索）
    
    Args:
        orcid_id: ORCID ID，格式如 "0000-0002-1825-0097"
        max_iterations: 最大迭代次数（每次搜索 10 个候选人）
                       默认从 config.yaml 读取，如果未配置则为 5
        match_threshold: 标题匹配相似度阈值，默认从配置读取
        max_matches: 每个候选人最大匹配论文数量，默认从配置读取
        verbose: 是否打印详细信息
        use_cache: 是否使用缓存（默认 True）
        
    Returns:
        (匹配的 Google Scholar URL, 匹配的作者信息字典)
        如果未找到，返回 (None, None)
    """
    # 加载配置
    config = load_config()
    pipeline_config = config.get("pipeline", {})
    
    # 使用传入参数或配置文件中的默认值
    if max_iterations is None:
        max_iterations = pipeline_config.get("max_iterations", 5)
    if match_threshold is None:
        match_threshold = pipeline_config.get("match_threshold", 0.85)
    if max_matches is None:
        max_matches = pipeline_config.get("max_matches", 10)
    
    # ========== 记录请求 ==========
    record_request()
    
    # ========== 缓存读取 ==========
    cache = _get_cache() if use_cache else None
    
    if cache:
        cached_data = cache.get(orcid_id)
        if cached_data and cached_data.get("google_scholar_url"):
            # 记录缓存命中
            record_cache_hit()
            
            if verbose:
                print("=" * 60)
                print("Google Scholar 账号查找 Pipeline")
                print("=" * 60)
                print(f"输入 ORCID: {orcid_id}")
                print()
                print(f"[CACHE] 命中缓存")
                print(f"  URL: {cached_data['google_scholar_url']}")
                print(f"  姓名: {cached_data.get('name', 'N/A')}")
                print(f"  机构: {cached_data.get('affiliation', 'N/A')}")
                print(f"  缓存时间: {cached_data.get('cached_at', 'N/A')}")
            
            logger.info(f"缓存命中: {orcid_id} -> {cached_data['google_scholar_url']}")
            
            # 返回格式与正常查找一致
            matched_candidate = {
                "name": cached_data.get("name"),
                "url": cached_data.get("google_scholar_url"),
                "user_id": cached_data.get("user_id"),
                "affiliation": cached_data.get("affiliation"),
            }
            return cached_data["google_scholar_url"], matched_candidate
    
    logger.info(f"开始查找 ORCID: {orcid_id}")
    
    if verbose:
        print("=" * 60)
        print("Google Scholar 账号查找 Pipeline")
        print("=" * 60)
        print(f"输入 ORCID: {orcid_id}")
        print(f"最大迭代次数: {max_iterations}")
        print(f"匹配阈值: {match_threshold}")
        print()
    
    # ========== STEP 1: 获取作者姓名和组织 ==========
    if verbose:
        print("[STEP 1/4] 获取作者姓名和组织...")
    
    try:
        author_name_info = get_author_name(orcid_id)
        author_name = author_name_info.get('full_name')
        
        if not author_name:
            log_print("[ERROR] 无法获取作者姓名", "error")
            record_error()
            return None, None
        
        # 规范化姓名（中文转拼音等）
        author_name_original = author_name
        author_name = normalize_name(author_name)
        
        logger.info(f"作者姓名: {author_name_original} -> {author_name}")
        if verbose:
            print(f"[INFO] 作者姓名: {author_name_original}")
            if author_name != author_name_original.lower().strip():
                print(f"[INFO] 规范化: {author_name}")
            if author_name_info.get('credit_name'):
                print(f"[INFO] 署名: {author_name_info['credit_name']}")
        
        # 获取作者组织（可选，失败不影响主流程）
        author_organization = None
        try:
            author_organization = get_author_organization(orcid_id)
            if author_organization:
                logger.info(f"作者组织: {author_organization}")
                if verbose:
                    print(f"[INFO] 作者组织: {author_organization}")
            else:
                if verbose:
                    print("[INFO] 未获取到组织信息，将仅使用姓名搜索")
        except Exception as org_e:
            if verbose:
                print(f"[INFO] 获取组织信息失败: {org_e}，将仅使用姓名搜索")
        
        if verbose:
            print()
            
    except Exception as e:
        log_print(f"[ERROR] 获取作者姓名失败: {e}", "error")
        record_error()
        return None, None
    
    # ========== STEP 2: 获取 ORCID 论文列表 ==========
    if verbose:
        print("[STEP 2/4] 获取 ORCID 论文列表...")
    
    try:
        orcid_papers = get_author_papers(orcid_id)
        
        if not orcid_papers:
            log_print("[WARNING] ORCID 论文列表为空，无法进行匹配", "warning")
            record_error()
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
        record_error()
        return None, None
    
    # ========== STEP 3 & 4: 迭代搜索候选人并验证 ==========
    scraper = GoogleScholarAuthorScraper()
    
    # 统计：记录总的 Google Search API 调用次数
    total_google_search_calls = 0
    
    # 准备搜索名字列表（先用 full_name，如果没结果再用 credit_name）
    search_names = [author_name]
    #credit_name = author_name_info.get('credit_name')
    #if credit_name:
    #    credit_name = normalize_name(credit_name)  # 规范化署名
    #    if credit_name and credit_name != author_name:
    #        search_names.append(credit_name)
    
    # 搜索策略：
    # 1. 如果有组织信息，先用 "名字 + 组织" 搜索
    # 2. 如果没找到或没有组织，用 "仅名字" 搜索
    search_configs = []
    
    if author_organization:
        # 优先使用组织限制搜索
        for name in search_names:
            search_configs.append({
                'name': name,
                'organization': author_organization,
                'desc': f'"{name}" + 组织 "{author_organization}"'
            })
        # 然后是不带组织的搜索（作为回退）
        for name in search_names:
            search_configs.append({
                'name': name,
                'organization': None,
                'desc': f'仅 "{name}"'
            })
    else:
        # 没有组织信息，仅使用姓名
        for name in search_names:
            search_configs.append({
                'name': name,
                'organization': None,
                'desc': f'仅 "{name}"'
            })
    
    for config in search_configs:
        search_name = config['name']
        search_org = config['organization']
        search_desc = config['desc']
        
        if verbose:
            print(f"\n[INFO] 搜索策略: {search_desc}")
        
        # 迭代搜索
        for iteration in range(max_iterations):
            start = iteration * 10 + 1
            
            if start > 100:  # Google API 限制
                if verbose:
                    print(f"[INFO] 已达到 Google API 搜索上限 (100 条)")
                break
            
            if verbose:
                print()
                print(f"[STEP 3/4] 搜索候选人 (第 {iteration + 1}/{max_iterations} 批, start={start})...")
            
            try:
                candidates = scraper.search_author(
                    search_name, 
                    start=start, 
                    num=10,
                    organization=search_org
                )
                
                # 统计：每次搜索增加计数（不管是否有结果，因为 API 已调用）
                total_google_search_calls += 1
                
                if not candidates:
                    if verbose:
                        print(f"[INFO] 第 {iteration + 1} 批没有更多候选人")
                    break
                
                logger.info(f"第 {iteration + 1} 批找到 {len(candidates)} 个候选人")
                if verbose:
                    print(f"[INFO] 找到 {len(candidates)} 个候选人")
                    
            except Exception as e:
                log_print(f"[ERROR] 搜索候选人失败: {e}", "error")
                continue
            
            # ========== STEP 4: 验证当前批次的候选人 ==========
            if verbose:
                print(f"[STEP 4/4] 验证第 {iteration + 1} 批候选人...")
            
            matched_url, matched_candidate, match_count = find_google_scholar_account_from_candidates(
                candidates=candidates,
                orcid_papers=orcid_papers,
                match_threshold=match_threshold,
                max_matches=max_matches,
                verbose=verbose
            )
            
            # 如果找到匹配，写入缓存并返回
            if matched_url:
                # 记录统计：名字搜索成功，记录使用了多少次 Google Search
                record_name_search_success(total_google_search_calls)
                
                # 写入缓存
                if cache and matched_candidate:
                    cache_data = {
                        "orcid_id": orcid_id,
                        "google_scholar_url": matched_url,
                        "user_id": matched_candidate.get("user_id", ""),
                        "name": matched_candidate.get("name", ""),
                        "affiliation": matched_candidate.get("affiliation", ""),
                        "search_method": "name_search",
                        "search_strategy": search_desc,
                        "match_count": match_count,
                        "google_search_calls": total_google_search_calls,
                        "cached_at": datetime.now().isoformat(),
                    }
                    cache.set(orcid_id, cache_data, ttl=CACHE_TTL)
                    logger.info(f"已写入缓存: {orcid_id} -> {matched_url}")
                
                if verbose:
                    print()
                    print("=" * 60)
                    print("查找结果")
                    print("=" * 60)
                    print(f"✓ 找到匹配的 Google Scholar 账号")
                    print(f"  URL: {matched_url}")
                    if matched_candidate:
                        print(f"  姓名: {matched_candidate.get('name', 'N/A')}")
                        print(f"  机构: {matched_candidate.get('affiliation', 'N/A')}")
                    print(f"  匹配论文数: {match_count}")
                    print(f"  搜索策略: {search_desc}")
                    print(f"  搜索迭代次数: {iteration + 1}")
                    print(f"  Google Search 调用次数: {total_google_search_calls}")
                    if cache:
                        print(f"  [CACHE] 已写入缓存")
                
                logger.info(f"找到匹配: {matched_url} (策略: {search_desc}, 迭代 {iteration + 1} 次, Google Search {total_google_search_calls} 次)")
                return matched_url, matched_candidate
            
            if verbose:
                print(f"[INFO] 第 {iteration + 1} 批未找到匹配，继续搜索下一批...")
            
            # 如果返回的候选人少于 10 个，说明没有更多结果了
            if len(candidates) < 10:
                if verbose:
                    print(f"[INFO] 当前策略没有更多候选人了")
                break
        
        # 当前策略搜索完毕，继续下一个策略
    
    # ========== 方法2: 通过论文标题搜索（逐篇搜索验证，节省 API 调用） ==========
    if verbose:
        print()
        print("=" * 60)
        print("[方法2] 名字搜索未找到，尝试通过论文标题搜索...")
        print("=" * 60)
    
    logger.info("名字搜索未找到，尝试论文标题搜索")
    
    # 使用论文搜索器
    paper_searcher = GoogleScholarAuthorByPaper(verbose=False)  # 减少输出
    
    # 选择几篇论文进行搜索（取前2篇，避免 API 调用过多）
    max_papers_to_search = 1
    papers_to_search = orcid_papers[:max_papers_to_search]
    
    # 记录已验证过的候选人 ID，避免重复验证
    verified_ids = set()
    
    # 统计：记录搜索了多少篇论文
    papers_searched_count = 0
    
    for i, paper in enumerate(papers_to_search, 1):
        paper_title = paper.get('title', '')
        if not paper_title:
            continue
        
        if verbose:
            print()
            print(f"[STEP 3/4] 论文搜索 ({i}/{len(papers_to_search)}): {paper_title[:50]}...")
        
        try:
            # 搜索论文，用作者名过滤
            authors = paper_searcher.search_author_by_paper(
                paper_title=paper_title,
                author_name=author_name,
                filter_by_name=True
            )
            
            # 统计：每次搜索增加计数（不管是否有结果，因为 API 已调用）
            papers_searched_count += 1
            
            if not authors:
                if verbose:
                    print(f"[INFO] 该论文未找到候选人，尝试下一篇...")
                continue
            
            # 转换为候选人格式，并过滤已验证过的
            paper_candidates = []
            for author in authors:
                if author['author_id'] not in verified_ids:
                    verified_ids.add(author['author_id'])
                    candidate = {
                        'name': author['name'],
                        'url': author['url'],
                        'user_id': author['author_id'],
                        'affiliation': '',  # 论文搜索不提供机构信息
                    }
                    paper_candidates.append(candidate)
            
            if not paper_candidates:
                if verbose:
                    print(f"[INFO] 候选人已在之前验证过，跳过...")
                continue
            
            if verbose:
                print(f"[INFO] 找到 {len(paper_candidates)} 个新候选人")
                print()
                print(f"[STEP 4/4] 验证候选人...")
            
            logger.info(f"论文 '{paper_title[:30]}...' 找到 {len(paper_candidates)} 个候选人")
            
            # 立即验证该论文找到的候选人
            matched_url, matched_candidate, match_count = find_google_scholar_account_from_candidates(
                candidates=paper_candidates,
                orcid_papers=orcid_papers,
                match_threshold=match_threshold,
                max_matches=max_matches,
                verbose=verbose
            )
            
            # 如果找到匹配，写入缓存并返回
            if matched_url:
                # 记录统计：论文搜索成功，记录搜索了多少篇论文
                record_paper_search_success(papers_searched_count)
                
                # 写入缓存
                if cache and matched_candidate:
                    cache_data = {
                        "orcid_id": orcid_id,
                        "google_scholar_url": matched_url,
                        "user_id": matched_candidate.get("user_id", ""),
                        "name": matched_candidate.get("name", ""),
                        "affiliation": matched_candidate.get("affiliation", ""),
                        "search_method": "paper_search",
                        "search_paper": paper_title[:100],
                        "match_count": match_count,
                        "papers_searched": papers_searched_count,
                        "cached_at": datetime.now().isoformat(),
                    }
                    cache.set(orcid_id, cache_data, ttl=CACHE_TTL)
                    logger.info(f"已写入缓存: {orcid_id} -> {matched_url}")
                
                if verbose:
                    print()
                    print("=" * 60)
                    print("查找结果")
                    print("=" * 60)
                    print(f"✓ 找到匹配的 Google Scholar 账号（通过论文搜索）")
                    print(f"  URL: {matched_url}")
                    if matched_candidate:
                        print(f"  姓名: {matched_candidate.get('name', 'N/A')}")
                    print(f"  匹配论文数: {match_count}")
                    print(f"  搜索论文: {paper_title[:50]}...")
                    print(f"  SerpAPI 搜索次数: {papers_searched_count}")
                    if cache:
                        print(f"  [CACHE] 已写入缓存")
                
                logger.info(f"通过论文搜索找到匹配: {matched_url} (搜索 {papers_searched_count} 篇论文)")
                return matched_url, matched_candidate
            
            if verbose:
                print(f"[INFO] 该论文的候选人未匹配，尝试下一篇...")
                    
        except Exception as e:
            if verbose:
                print(f"[WARNING] 搜索论文失败: {e}")
            continue
    
    if verbose:
        print(f"\n[INFO] 所有论文搜索完毕，未找到匹配")
    
    # 所有方法都没找到
    # 记录统计：未找到
    record_not_found()
    
    if verbose:
        print()
        print("=" * 60)
        print("查找结果")
        print("=" * 60)
        print("✗ 未找到匹配的 Google Scholar 账号")
        print(f"  方法1 - 名字搜索: 已搜索 {len(search_configs)} 种策略，每种最多 {max_iterations} 次迭代")
        print(f"  方法2 - 论文搜索: 已搜索 {len(papers_to_search)} 篇论文")
        print(f"  搜索名字: {search_names}")
    
    logger.warning(f"未找到匹配的 Google Scholar 账号: {orcid_id}")
    return None, None


if __name__ == "__main__":
    # 示例用法
    
    # 测试 ORCID ID
    test_orcid = "0000-0002-5790-5164"
    
    print()
    print("=" * 60)
    print("Google Scholar 账号查找工具")
    print("=" * 60)
    print()
    print(f"测试 ORCID: {test_orcid}")
    print()
    
    # 执行查找（使用配置文件中的默认参数）
    url, author_info = find_google_scholar_by_orcid(
        orcid_id=test_orcid,
        verbose=True
    )
    
    print()
    print("-" * 60)
    if url:
        print(f"最终结果: {url}")
    else:
        print("最终结果: Not Available")
