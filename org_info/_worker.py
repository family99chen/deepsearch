"""Worker script - 在完全独立的进程中运行"""
import sys
import os
import json
import asyncio
from pathlib import Path

# 设置独立的 chromedriver 缓存目录（基于 PID）
os.environ["UC_CACHE_DIR"] = f"/tmp/uc_cache_{os.getpid()}"

sys.path.insert(0, str(Path(__file__).parent.parent))

def run_task(url: str, person_name: str, verbose: bool):
    """运行单个任务"""
    result = {
        "url": url,
        "success": False,
        "mode": "failed",
        "report": "",
        "info_count": 0,
        "error": None,
    }
    
    try:
        # 先尝试 iter_agent
        from org_info.iter_agent.iteragent import search_person_in_org as iter_agent_search
        from org_info.shared_driver import close_shared_driver
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if verbose:
                print(f"[Worker {os.getpid()}] iter_agent: {url[:50]}...", file=sys.stderr)
            
            res = loop.run_until_complete(
                iter_agent_search(
                    start_url=url,
                    person_name=person_name,
                    max_links_per_page=2,
                    max_concurrent=1,
                    verbose=False,
                )
            )
            
            if res.success and res.collected_info:
                result["success"] = True
                result["mode"] = "iter_agent"
                result["report"] = res.final_report
                result["info_count"] = len(res.collected_info)
                close_shared_driver()
                return result
        except Exception as e:
            if verbose:
                print(f"[Worker {os.getpid()}] iter_agent 失败: {e}", file=sys.stderr)
        
        # 在尝试 brain 模式前，只关闭 driver 但保持 Xvfb 运行
        # 这样 brain 创建新 driver 时可以复用 Xvfb
        try:
            from org_info.shared_driver import reset_shared_driver
            reset_shared_driver()
        except:
            pass
        
        # 健康检查：测试能否创建新的 driver
        try:
            from org_info.shared_driver import get_shared_driver
            if verbose:
                print(f"[Worker {os.getpid()}] 测试 driver 健康状态...", file=sys.stderr)
            test_driver = get_shared_driver()
            test_driver.get("about:blank")
            if verbose:
                print(f"[Worker {os.getpid()}] ✓ driver 健康检查通过", file=sys.stderr)
        except Exception as e:
            if verbose:
                print(f"[Worker {os.getpid()}] ✗ driver 健康检查失败: {e}", file=sys.stderr)
            # 尝试完全重置
            try:
                close_shared_driver()
            except:
                pass
        
        # 尝试 brain 模式（会创建新的 driver，复用已有的 Xvfb）
        try:
            from org_info.iteragent_advanced.brain import search_person as brain_search
            
            if verbose:
                print(f"[Worker {os.getpid()}] brain: {url[:50]}...", file=sys.stderr)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            res = loop.run_until_complete(
                brain_search(
                    start_url=url,
                    person_name=person_name,
                    verbose=False,
                )
            )
            
            if res.success and res.collected_info:
                result["success"] = True
                result["mode"] = "brain"
                result["report"] = res.collected_info
                result["info_count"] = 1
            else:
                result["error"] = res.final_reason
                if verbose:
                    print(f"[Worker {os.getpid()}] brain 未找到信息: {res.final_reason}", file=sys.stderr)
        except Exception as e:
            result["error"] = str(e)
            if verbose:
                print(f"[Worker {os.getpid()}] brain 异常: {e}", file=sys.stderr)
    
    except Exception as e:
        result["error"] = str(e)
    
    finally:
        try:
            from org_info.shared_driver import close_shared_driver
            close_shared_driver()
        except:
            pass
        
        # 清理缓存目录
        import shutil
        cache_dir = f"/tmp/uc_cache_{os.getpid()}"
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except:
            pass
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--person", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    result = run_task(args.url, args.person, args.verbose)
    print(json.dumps(result))  # 输出到 stdout

