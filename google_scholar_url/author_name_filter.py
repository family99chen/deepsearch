"""
作者姓名一致性匹配算法

基于匈牙利算法和最大可能值一致性检验，判断两个名字是否为同一人。
不依赖阈值，通过比对 best match 和 max possible 的一致性来判断。

算法原理：
1. 将两个名字分词（tokenize）
2. 构建 token 间的相似度矩阵（使用 Jaro-Winkler）
3. 使用匈牙利算法找到全局最优匹配（best match）
4. 对每列计算局部最大可能值（max possible）
5. 如果 best match == max possible，则认为是同一人
"""

import re
import unicodedata
from typing import List, Tuple, Optional
import numpy as np

# 尝试导入 scipy，如果没有则使用自己实现的匈牙利算法
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def jaro_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的 Jaro 相似度
    
    Args:
        s1: 第一个字符串
        s2: 第二个字符串
    
    Returns:
        Jaro 相似度 (0-1)
    """
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # 匹配窗口
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    # 找匹配字符
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # 计算换位
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len1 + matches / len2 + 
            (matches - transpositions / 2) / matches) / 3
    
    return jaro


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    计算两个字符串的 Jaro-Winkler 相似度
    
    Jaro-Winkler 在 Jaro 基础上，对共同前缀给予额外权重
    
    Args:
        s1: 第一个字符串
        s2: 第二个字符串
        p: 前缀权重因子 (默认 0.1，最大 0.25)
    
    Returns:
        Jaro-Winkler 相似度 (0-1)
    """
    jaro = jaro_similarity(s1, s2)
    
    # 计算共同前缀长度（最多4个字符）
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    return jaro + prefix_len * p * (1 - jaro)


def normalize_name(name: str) -> str:
    """
    规范化名字字符串
    
    - 转小写
    - 移除重音符号
    - 移除非字母字符（保留空格）
    """
    # 转小写
    name = name.lower()
    
    # 移除重音符号 (é -> e, ü -> u, etc.)
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    
    # 移除非字母字符，保留空格
    name = re.sub(r'[^a-z\s]', ' ', name)
    
    # 合并多个空格
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def tokenize_name(name: str) -> List[str]:
    """
    将名字分词
    
    Args:
        name: 原始名字字符串
    
    Returns:
        token 列表
    """
    normalized = normalize_name(name)
    tokens = normalized.split()
    
    # 过滤掉太短的 token（单个字符可能是中间名缩写）
    # 但保留单字符如果它是唯一的 token
    if len(tokens) > 1:
        tokens = [t for t in tokens if len(t) > 0]
    
    return tokens


def build_similarity_matrix(tokens1: List[str], tokens2: List[str]) -> np.ndarray:
    """
    构建 token 间的相似度矩阵
    
    Args:
        tokens1: 第一组 tokens（行）
        tokens2: 第二组 tokens（列）
    
    Returns:
        相似度矩阵 (len(tokens1) x len(tokens2))
    """
    n, m = len(tokens1), len(tokens2)
    matrix = np.zeros((n, m))
    
    for i, t1 in enumerate(tokens1):
        for j, t2 in enumerate(tokens2):
            matrix[i, j] = jaro_winkler_similarity(t1, t2)
    
    return matrix


def hungarian_algorithm_simple(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    简单实现的匈牙利算法（用于没有 scipy 的情况）
    
    对于小矩阵使用贪心近似
    """
    n, m = cost_matrix.shape
    assignments = []
    used_rows = set()
    used_cols = set()
    
    # 贪心选择最大值
    flat_indices = np.argsort(cost_matrix.flatten())[::-1]
    
    for idx in flat_indices:
        row = idx // m
        col = idx % m
        
        if row not in used_rows and col not in used_cols:
            assignments.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
            
            if len(assignments) == min(n, m):
                break
    
    return assignments


def find_best_match(similarity_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """
    使用匈牙利算法找到最优匹配
    
    Args:
        similarity_matrix: 相似度矩阵
    
    Returns:
        (匹配对列表, 总相似度)
    """
    if similarity_matrix.size == 0:
        return [], 0.0
    
    # 匈牙利算法求最小化，所以用 1 - similarity 作为代价
    cost_matrix = 1 - similarity_matrix
    
    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = list(zip(row_ind, col_ind))
    else:
        assignments = hungarian_algorithm_simple(similarity_matrix)
    
    # 计算总相似度
    total_similarity = sum(similarity_matrix[r, c] for r, c in assignments)
    
    return assignments, total_similarity


def find_max_possible_by_col(similarity_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """
    对每列找到最大可能值（列方向局部最优）
    
    Args:
        similarity_matrix: 相似度矩阵 (rows x cols)
    
    Returns:
        (每列的最大匹配对, 总最大可能值)
    """
    if similarity_matrix.size == 0:
        return [], 0.0
    
    n_cols = similarity_matrix.shape[1]
    max_assignments = []
    total_max = 0.0
    
    for col in range(n_cols):
        col_values = similarity_matrix[:, col]
        max_row = np.argmax(col_values)
        max_val = col_values[max_row]
        max_assignments.append((max_row, col))
        total_max += max_val
    
    return max_assignments, total_max


def find_max_possible_by_row(similarity_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """
    对每行找到最大可能值（行方向局部最优）
    
    注意：这里返回的是每个被匹配行的最优列
    
    Args:
        similarity_matrix: 相似度矩阵 (rows x cols)
    
    Returns:
        (每行的最大匹配对, 总最大可能值)
    """
    if similarity_matrix.size == 0:
        return [], 0.0
    
    n_rows = similarity_matrix.shape[0]
    max_assignments = []
    total_max = 0.0
    
    for row in range(n_rows):
        row_values = similarity_matrix[row, :]
        max_col = np.argmax(row_values)
        max_val = row_values[max_col]
        max_assignments.append((row, max_col))
        total_max += max_val
    
    return max_assignments, total_max


def check_bidirectional_consistency(
    similarity_matrix: np.ndarray, 
    best_assignments: List[Tuple[int, int]]
) -> Tuple[bool, dict]:
    """
    双向一致性检验：检查每个匹配是否同时是行最优和列最优
    
    Args:
        similarity_matrix: 相似度矩阵
        best_assignments: 匈牙利算法的匹配结果
    
    Returns:
        (是否一致, 详细信息)
    """
    inconsistencies = []
    
    for row, col in best_assignments:
        matched_val = similarity_matrix[row, col]
        
        # 检查列方向：该行是否是该列的最大值
        col_max_row = np.argmax(similarity_matrix[:, col])
        col_max_val = similarity_matrix[col_max_row, col]
        is_col_optimal = (row == col_max_row)
        
        # 检查行方向：该列是否是该行的最大值
        row_max_col = np.argmax(similarity_matrix[row, :])
        row_max_val = similarity_matrix[row, row_max_col]
        is_row_optimal = (col == row_max_col)
        
        if not is_col_optimal or not is_row_optimal:
            inconsistencies.append({
                'row': row,
                'col': col,
                'matched_val': matched_val,
                'is_col_optimal': is_col_optimal,
                'col_best': (col_max_row, col_max_val),
                'is_row_optimal': is_row_optimal,
                'row_best': (row_max_col, row_max_val)
            })
    
    is_consistent = len(inconsistencies) == 0
    
    return is_consistent, {'inconsistencies': inconsistencies}


# 无效名字列表（这些名字无法判断真实身份，宽松处理视为匹配）
INVALID_NAMES = {
    '无主题', '无标题', 'untitled', 'no title', 'no name',
    '未知', 'unknown', 'n/a', 'na', 'none', '暂无'
}


def is_invalid_name(name: str) -> bool:
    """
    检查是否是无效名字
    
    Args:
        name: 名字字符串
    
    Returns:
        是否是无效名字
    """
    if not name or not name.strip():
        return True
    
    normalized = name.strip().lower()
    return normalized in {n.lower() for n in INVALID_NAMES}


def is_same_author(name1: str, name2: str, 
                   similarity_tolerance: float = 0.01,
                   verbose: bool = False) -> Tuple[bool, dict]:
    """
    判断两个名字是否为同一人
    
    算法：
    1. 分词并构建相似度矩阵
    2. 比较匈牙利算法的 best match 和每列的 max possible
    3. 如果两者一致（总分相等），认为是同一人
    
    特殊规则：
    - 如果任一名字是无效内容（如"无主题"），视为匹配（宽松处理）
    
    Args:
        name1: 第一个名字
        name2: 第二个名字
        similarity_tolerance: 相似度容差（处理浮点误差）
        verbose: 是否输出详细信息
    
    Returns:
        (是否同一人, 详细信息字典)
    """
    # 特殊规则：无效名字视为匹配（无法确认真实身份，宽松处理）
    if is_invalid_name(name1) or is_invalid_name(name2):
        invalid_name = name1 if is_invalid_name(name1) else name2
        if verbose:
            print(f"\n⚠️ 检测到无效名字 '{invalid_name}'，视为匹配（宽松处理）")
        return True, {
            'name1': name1,
            'name2': name2,
            'is_same_author': True,
            'reason': 'invalid_name_passthrough',
            'invalid_name': invalid_name
        }
    
    # 分词
    tokens1 = tokenize_name(name1)
    tokens2 = tokenize_name(name2)
    
    if not tokens1 or not tokens2:
        return False, {'error': 'Empty tokens', 'tokens1': tokens1, 'tokens2': tokens2}
    
    # 确保较短的在列（作为被匹配的目标）
    if len(tokens1) < len(tokens2):
        tokens_row = tokens2  # 长的作为行
        tokens_col = tokens1  # 短的作为列
        swapped = True
    else:
        tokens_row = tokens1
        tokens_col = tokens2
        swapped = False
    
    # 构建相似度矩阵
    sim_matrix = build_similarity_matrix(tokens_row, tokens_col)
    
    # 计算 best match（匈牙利算法）
    best_assignments, best_score = find_best_match(sim_matrix)
    
    # 计算 max possible（每列最大值之和）- 旧方法，保留用于参考
    max_col_assignments, max_col_score = find_max_possible_by_col(sim_matrix)
    
    # 双向一致性检验：每个匹配必须同时是行最优和列最优
    is_bidirectional_consistent, consistency_details = check_bidirectional_consistency(
        sim_matrix, best_assignments
    )
    
    # 额外检查：如果匹配分数太低，即使一致也不认为是同一人
    # 这里使用一个基于 token 数量的动态检查
    min_acceptable_avg = 0.7  # 平均每个匹配至少 0.7
    avg_score = best_score / len(tokens_col) if tokens_col else 0
    has_good_matches = avg_score >= min_acceptable_avg
    
    # 使用双向一致性检验
    is_same = is_bidirectional_consistent and has_good_matches
    
    # 构建详细信息
    details = {
        'name1': name1,
        'name2': name2,
        'tokens_row': tokens_row,
        'tokens_col': tokens_col,
        'swapped': swapped,
        'similarity_matrix': sim_matrix.tolist(),
        'best_match': {
            'assignments': [(tokens_row[r], tokens_col[c], sim_matrix[r, c]) 
                          for r, c in best_assignments],
            'score': best_score
        },
        'max_possible_by_col': {
            'assignments': [(tokens_row[r], tokens_col[c], sim_matrix[r, c]) 
                          for r, c in max_col_assignments],
            'score': max_col_score
        },
        'is_bidirectional_consistent': is_bidirectional_consistent,
        'consistency_details': consistency_details,
        'avg_match_score': avg_score,
        'has_good_matches': has_good_matches,
        'is_same_author': is_same
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"名字比较: '{name1}' vs '{name2}'")
        print(f"{'='*60}")
        print(f"Tokens (行): {tokens_row}")
        print(f"Tokens (列): {tokens_col}")
        print(f"\n相似度矩阵:")
        print(f"{'':12}", end='')
        for t in tokens_col:
            print(f"{t:12}", end='')
        print()
        for i, t in enumerate(tokens_row):
            print(f"{t:12}", end='')
            for j in range(len(tokens_col)):
                print(f"{sim_matrix[i,j]:12.3f}", end='')
            print()
        print(f"\nBest Match (匈牙利算法):")
        for r, c in best_assignments:
            # 检查是否是双向最优
            is_col_best = (r == np.argmax(sim_matrix[:, c]))
            is_row_best = (c == np.argmax(sim_matrix[r, :]))
            markers = []
            if not is_col_best:
                markers.append("非列最优")
            if not is_row_best:
                markers.append("非行最优")
            marker_str = f" ⚠️ {', '.join(markers)}" if markers else " ✓"
            print(f"  {tokens_row[r]} -> {tokens_col[c]} ({sim_matrix[r,c]:.3f}){marker_str}")
        print(f"  总分: {best_score:.3f}")
        
        # 显示不一致的详情
        if not is_bidirectional_consistent:
            print(f"\n⚠️ 双向一致性检验失败:")
            for inc in consistency_details['inconsistencies']:
                row, col = inc['row'], inc['col']
                print(f"  {tokens_row[row]} -> {tokens_col[col]}:")
                if not inc['is_row_optimal']:
                    better_col = inc['row_best'][0]
                    better_val = inc['row_best'][1]
                    print(f"    行最优应是: {tokens_col[better_col]} ({better_val:.3f})")
                if not inc['is_col_optimal']:
                    better_row = inc['col_best'][0]
                    better_val = inc['col_best'][1]
                    print(f"    列最优应是: {tokens_row[better_row]} ({better_val:.3f})")
        
        print(f"\n双向一致性: {'✓ 通过' if is_bidirectional_consistent else '✗ 不通过'}")
        print(f"平均匹配分: {avg_score:.3f} ({'✓ 足够高' if has_good_matches else '✗ 太低'})")
        print(f"\n结论: {'✓ 同一人' if is_same else '✗ 不是同一人'}")
    
    return is_same, details


def filter_candidates_by_name(target_name: str, 
                              candidates: List[dict],
                              name_field: str = 'name',
                              verbose: bool = False) -> List[dict]:
    """
    从候选人列表中筛选出名字匹配的候选人
    
    Args:
        target_name: 目标作者名字
        candidates: 候选人列表，每个元素是包含名字的字典
        name_field: 候选人字典中名字字段的键名
        verbose: 是否输出详细信息
    
    Returns:
        筛选后的候选人列表
    """
    filtered = []
    
    for candidate in candidates:
        candidate_name = candidate.get(name_field, '')
        if not candidate_name:
            continue
        
        is_same, details = is_same_author(target_name, candidate_name, verbose=verbose)
        
        if is_same:
            # 添加匹配信息到候选人
            candidate['_name_match_score'] = details['best_match']['score']
            candidate['_name_match_details'] = details
            filtered.append(candidate)
        elif verbose:
            print(f"  [过滤] {candidate_name} - 不匹配")
    
    return filtered


# ============== 测试代码 ==============

if __name__ == "__main__":
    print("\n" + "="*70)
    print("作者姓名一致性匹配算法测试")
    print("="*70)
    
    # 测试用例
    test_cases = [
        # 应该匹配的情况
        ("Emmy Tay", "Emmy Xue Yun Tay", True),
        ("Ju Li", "Ju Li", True),
        ("J. Li", "Ju Li", True),
        ("Li Ju", "Ju Li", True),  # 顺序不同
        ("John Smith", "John A. Smith", True),
        ("Michael Johnson", "Mike Johnson", True),  # 昵称
        ("Zhang Wei", "Wei Zhang", True),  # 中文名顺序
        ("María García", "Maria Garcia", True),  # 重音符号
        
        # 不应该匹配的情况
        ("Ju Li", "John Smith", False),
        ("Emmy Tay", "Emma Taylor", False),
        ("Zhang Wei", "Li Wei", False),  # 姓不同
        ("Michael Brown", "Michael Green", False),
        ("Ju Li", "Li Jun", False),  # 相似但不同
        ("Xie Weihao", "Yi Zhengyuan", False),  # 完全不同的中文名
        
        # 一致性检验的边界情况
        ("Mike Michael", "Michael Smith", False),  # mike/michael 竞争 michael 列
        ("Li Wei", "Wei Li Chen", False),  # wei 可能有竞争
    ]
    
    print("\n测试结果:")
    print("-"*70)
    
    correct = 0
    total = len(test_cases)
    
    for name1, name2, expected in test_cases:
        is_same, details = is_same_author(name1, name2, verbose=False)
        
        status = "✓" if is_same == expected else "✗"
        result = "同一人" if is_same else "不同人"
        expected_str = "同一人" if expected else "不同人"
        
        if is_same == expected:
            correct += 1
        
        print(f"{status} '{name1}' vs '{name2}'")
        print(f"    结果: {result} (期望: {expected_str})")
        print(f"    Best: {details['best_match']['score']:.3f}, "
              f"双向一致: {details['is_bidirectional_consistent']}, "
              f"平均: {details['avg_match_score']:.3f}")
        print()
    
    print("-"*70)
    print(f"准确率: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # 详细展示一个例子
    print("\n\n" + "="*70)
    print("详细示例 1: Emmy Tay vs Emmy Xue Yun Tay (应该匹配)")
    print("="*70)
    is_same_author("Emmy Tay", "Emmy Xue Yun Tay", verbose=True)
    
    print("\n\n" + "="*70)
    print("详细示例 2: Ju Li vs John Smith (平均分太低)")
    print("="*70)
    is_same_author("Ju Li", "John Smith", verbose=True)
    
    print("\n\n" + "="*70)
    print("详细示例 3: Xie Weihao vs Yi Zhengyuan (完全不同)")
    print("="*70)
    is_same_author("Xie Weihao", "Yi Zhengyuan", verbose=True)
    
    print("\n\n" + "="*70)
    print("详细示例 4: Mike Michael vs Michael Smith (一致性检验)")
    print("="*70)
    is_same_author("Mike Michael", "Michael Smith", verbose=True)

