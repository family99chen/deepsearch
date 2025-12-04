"""
姓名规范化模块

使用 unidecode 自动将任意语言姓名转换为拉丁字母
无需手动规则，完全依赖 unidecode 的语言学数据

安装：
    pip install unidecode
"""

import re

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False
    print("[WARNING] unidecode 未安装")
    print("[TIP] 安装: pip install unidecode")


def normalize_name(name: str) -> str:
    """
    将任意语言的姓名转换为小写拉丁字母
    
    Args:
        name: 原始姓名（任意语言）
    
    Returns:
        规范化后的姓名（小写拉丁字母，空格分隔）
    """
    if not name:
        return ""
    
    # unidecode 处理所有 Unicode -> ASCII
    if HAS_UNIDECODE:
        name = unidecode(name)
    
    # 转小写，只保留字母和空格
    name = name.lower()
    name = re.sub(r'[^a-z\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def are_names_equivalent(name1: str, name2: str) -> bool:
    """
    检查两个名字是否等价（忽略顺序）
    """
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    if n1 == n2:
        return True
    
    return sorted(n1.split()) == sorted(n2.split())


if __name__ == "__main__":
    print("=" * 70)
    print("姓名规范化测试 (unidecode)")
    print("=" * 70)
    print(f"unidecode: {'✓ 已安装' if HAS_UNIDECODE else '✗ 未安装'}")
    
    tests = [
        # 中文
        "侯阳",
        "张三",
        "李小明",
        
        # 日文
        "山田太郎",
        "やまだ たろう",
        "ヤマダ タロウ",
        "佐藤花子",
        
        # 泰语
        "สมชาย",
        "ประยุทธ์",
        
        # 西方语言
        "María García",
        "Müller",
        "O'Brien",
        "François Côté",
        
        # 俄文
        "Владимир Путин",
        "Иван Петров",
        
        # 韩文
        "김철수",
        "박지성",
        
        # 阿拉伯文
        "محمد",
        
        # 希腊文
        "Αλέξανδρος",
    ]
    
    print(f"\n{'原始':<25} {'规范化结果':<30}")
    print("-" * 60)
    
    for name in tests:
        result = normalize_name(name)
        print(f"{name:<25} {result:<30}")
    
    print("\n" + "=" * 70)
    print("跨语言等价测试")
    print("=" * 70)
    
    equiv_tests = [
        ("侯阳", "Hou Yang"),
        ("张三", "Zhang San"),
        ("Владимир", "Vladimir"),
        ("François", "Francois"),
    ]
    
    for n1, n2 in equiv_tests:
        norm1 = normalize_name(n1)
        norm2 = normalize_name(n2)
        equiv = are_names_equivalent(n1, n2)
        print(f"{n1} -> {norm1}")
        print(f"{n2} -> {norm2}")
        print(f"等价: {'✓' if equiv else '✗'}")
        print()
