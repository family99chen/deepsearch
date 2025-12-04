"""
通过 ORCID API 获取作者的组织/机构信息
"""

import sys
import requests
import yaml
from pathlib import Path
from typing import Optional, List, Dict

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入重试模块
try:
    from utils.retry import exponential_backoff, DEFAULT_RETRYABLE_EXCEPTIONS
except ImportError:
    def exponential_backoff(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    DEFAULT_RETRYABLE_EXCEPTIONS = (requests.exceptions.RequestException,)


def load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=DEFAULT_RETRYABLE_EXCEPTIONS + (requests.exceptions.RequestException,)
)
def fetch_employments(orcid_id: str, access_token: Optional[str] = None) -> dict:
    """
    通过 ORCID ID 获取作者的雇佣信息（机构）
    
    Args:
        orcid_id: 作者的 ORCID ID
        access_token: ORCID API access token
    
    Returns:
        ORCID API 返回的雇佣信息
    """
    config = load_config()
    
    if access_token is None:
        access_token = config["orcid"]["access_token"]
    
    api_base_url = config["orcid"]["api_base_url"]
    url = f"{api_base_url}/{orcid_id}/employments"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    return response.json()


def parse_organizations(employments_data: dict) -> List[Dict]:
    """
    解析雇佣数据，提取组织信息
    
    Args:
        employments_data: ORCID API /employments 端点返回的数据
    
    Returns:
        组织列表，按结束日期排序（当前工作优先）
    """
    organizations = []
    
    # 获取 affiliation-group 列表
    affiliation_groups = employments_data.get("affiliation-group", [])
    
    for group in affiliation_groups:
        summaries = group.get("summaries", [])
        for summary_item in summaries:
            employment = summary_item.get("employment-summary", {})
            
            org_info = employment.get("organization", {})
            org_name = org_info.get("name", "")
            
            if not org_name:
                continue
            
            # 获取地址信息
            address = org_info.get("address", {})
            city = address.get("city", "")
            country = address.get("country", "")
            
            # 获取时间信息
            start_date = employment.get("start-date")
            end_date = employment.get("end-date")
            
            # 判断是否是当前工作
            is_current = end_date is None
            
            # 获取角色/职位
            role = employment.get("role-title", "")
            department = employment.get("department-name", "")
            
            org = {
                "name": org_name,
                "city": city,
                "country": country,
                "role": role,
                "department": department,
                "is_current": is_current,
                "start_date": start_date,
                "end_date": end_date,
            }
            
            organizations.append(org)
    
    # 按当前工作优先排序
    organizations.sort(key=lambda x: (not x["is_current"], x["name"]))
    
    return organizations


def get_author_organization(orcid_id: str) -> Optional[str]:
    """
    获取作者的主要组织/机构名称
    
    优先返回当前工作的机构，如果没有则返回最近的机构
    
    Args:
        orcid_id: 作者的 ORCID ID
    
    Returns:
        机构名称，如果没有则返回 None
    """
    try:
        employments_data = fetch_employments(orcid_id)
        organizations = parse_organizations(employments_data)
        
        if organizations:
            # 返回第一个（当前或最近的）组织名称
            return organizations[0]["name"]
        
        return None
        
    except Exception as e:
        print(f"[WARNING] 获取组织信息失败: {e}")
        return None


def get_all_organizations(orcid_id: str) -> List[Dict]:
    """
    获取作者的所有组织/机构信息
    
    Args:
        orcid_id: 作者的 ORCID ID
    
    Returns:
        组织列表
    """
    try:
        employments_data = fetch_employments(orcid_id)
        return parse_organizations(employments_data)
    except Exception as e:
        print(f"[WARNING] 获取组织信息失败: {e}")
        return []


if __name__ == "__main__":
    # 示例用法
    test_orcid = "0000-0003-3701-8119"
    
    print(f"正在获取 ORCID {test_orcid} 的组织信息...\n")
    
    try:
        # 获取主要组织
        primary_org = get_author_organization(test_orcid)
        print(f"主要组织: {primary_org or '无'}")
        
        # 获取所有组织
        print("\n=== 所有组织 ===")
        all_orgs = get_all_organizations(test_orcid)
        
        if all_orgs:
            for i, org in enumerate(all_orgs, 1):
                status = "当前" if org["is_current"] else "历史"
                print(f"\n[{i}] {org['name']} ({status})")
                if org["role"]:
                    print(f"    职位: {org['role']}")
                if org["department"]:
                    print(f"    部门: {org['department']}")
                if org["city"] or org["country"]:
                    print(f"    地点: {org['city']}, {org['country']}")
        else:
            print("未找到组织信息")
            
    except requests.exceptions.HTTPError as e:
        print(f"API 请求失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

