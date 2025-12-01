#!/usr/bin/env python3
"""
Cookies 格式转换工具
将 EditThisCookie 导出格式转换为项目使用的格式
"""

import json
import os
from pathlib import Path

# 文件路径
COOKIES_FILE = Path(__file__).parent / "google_cookies.json"


def convert_cookies(input_file: str = None, output_file: str = None):
    """
    转换 EditThisCookie 格式的 cookies 为项目格式
    
    EditThisCookie 格式:
        expirationDate, hostOnly, storeId, id 等字段
        
    项目格式:
        expiry, httpOnly, secure, sameSite 等字段
    """
    input_path = Path(input_file) if input_file else COOKIES_FILE
    output_path = Path(output_file) if output_file else COOKIES_FILE
    
    print(f"[INFO] 读取文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        cookies = json.load(f)
    
    print(f"[INFO] 原始 cookies 数量: {len(cookies)}")
    
    converted = []
    for cookie in cookies:
        new_cookie = {
            "name": cookie.get("name"),
            "value": cookie.get("value"),
            "domain": cookie.get("domain", ".google.com"),
            "path": cookie.get("path", "/"),
            # 关键：expirationDate -> expiry
            "expiry": cookie.get("expirationDate") or cookie.get("expiry"),
            "secure": cookie.get("secure", False),
            "httpOnly": cookie.get("httpOnly", False),
        }
        
        # 处理 sameSite
        same_site = cookie.get("sameSite", "")
        if same_site == "no_restriction":
            new_cookie["sameSite"] = "None"
        elif same_site == "lax":
            new_cookie["sameSite"] = "Lax"
        elif same_site == "strict":
            new_cookie["sameSite"] = "Strict"
        else:
            new_cookie["sameSite"] = "Lax"
        
        converted.append(new_cookie)
    
    # 保存转换后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] 已转换并保存到: {output_path}")
    print(f"[INFO] 转换后 cookies 数量: {len(converted)}")
    
    # 检查关键 cookies
    key_cookies = ['SID', 'HSID', 'SSID', 'APISID', 'SAPISID']
    found = [c['name'] for c in converted if c['name'] in key_cookies]
    print(f"[INFO] 关键 cookies: {found}")
    
    return converted


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="转换 EditThisCookie 格式的 cookies")
    parser.add_argument('-i', '--input', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    convert_cookies(args.input, args.output)


if __name__ == "__main__":
    main()

