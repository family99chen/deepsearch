#!/usr/bin/env python3
"""
调试脚本：查看 Google Scholar 返回的具体内容
"""

import os
import json
import requests
from pathlib import Path

COOKIES_FILE = Path(__file__).parent / "google_cookies.json"

def debug_request():
    # 创建 session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    
    # 加载 cookies
    with open(COOKIES_FILE, 'r') as f:
        cookies = json.load(f)
    
    for cookie in cookies:
        session.cookies.set(
            cookie['name'],
            cookie['value'],
            domain=cookie.get('domain', '.google.com')
        )
    
    print(f"[INFO] 已加载 {len(cookies)} 个 cookies")
    
    # 测试请求
    url = "https://scholar.google.com/citations?view_op=search_authors&mauthors=test&hl=zh-CN"
    
    print(f"[INFO] 请求 URL: {url}")
    print()
    
    response = session.get(url, timeout=15)
    
    print(f"[INFO] 状态码: {response.status_code}")
    print(f"[INFO] 响应长度: {len(response.text)} 字符")
    print()
    
    # 检查关键内容
    text = response.text.lower()
    
    print("=" * 60)
    print("检查结果:")
    print("=" * 60)
    
    if 'captcha' in text or 'recaptcha' in text:
        print("❌ 检测到验证码 (CAPTCHA)")
    else:
        print("✓ 无验证码")
    
    if 'unusual traffic' in text:
        print("❌ 检测到异常流量提示")
    else:
        print("✓ 无异常流量提示")
    
    if 'accounts.google.com' in text:
        print("❌ 被重定向到登录页面")
    else:
        print("✓ 未重定向到登录")
    
    if 'gsc_1usr' in text or 'gs_ai_name' in text:
        print("✓ 找到作者搜索结果区域")
    else:
        print("❌ 未找到作者搜索结果区域")
    
    # 保存响应内容用于分析
    debug_file = Path(__file__).parent / "debug_response.html"
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print()
    print(f"[INFO] 完整响应已保存到: {debug_file}")
    print("[INFO] 你可以用浏览器打开这个文件查看具体内容")
    
    # 显示部分响应内容
    print()
    print("=" * 60)
    print("响应内容前 2000 字符:")
    print("=" * 60)
    print(response.text[:2000])


if __name__ == "__main__":
    debug_request()

