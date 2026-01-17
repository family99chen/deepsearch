"""
Organization Pipeline - subprocess 版本（真正模拟 tmux）

使用 subprocess 启动完全独立的 Python 进程，和 tmux 开多个窗口一样：
- 完全独立的 Python 解释器
- 没有任何状态共享
- 每个进程有自己的 chromedriver 缓存
"""

import sys
import os
import json
import subprocess
import tempfile
import yaml
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入子模块（仅用于搜索链接，不需要 Chrome）
from org_info.extract_org_info import get_organization
from org_info.google_org_search import search_person_in_org as google_search, OrgPersonLink


# ============ 配置 ============

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
WORKER_SCRIPT = Path(__file__).parent / "_worker.py"

def _load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ============ 数据结构 ============

@dataclass
class LinkResult:
    url: str
    success: bool
    mode: str
    report: str
    info_count: int
    error: Optional[str] = None


@dataclass
class PipelineResult:
    person_name: str
    organization: Optional[str]
    links_found: int
    links_processed: int
    success_count: int
    merged_report: str
    link_results: List[LinkResult] = field(default_factory=list)
    elapsed_time: float = 0.0


# ============ Worker 脚本 ============

WORKER_CODE = '''
"""Worker script - 在完全独立的进程中运行"""
import sys
import os
import json
import asyncio
from pathlib import Path

# 设置独立的 chromedriver 缓存目录（基于 PID）
os.environ["UC_CACHE_DIR"] = f"/tmp/uc_cache_{os.getpid()}"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        
        # 尝试 brain 模式
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
        except Exception as e:
            result["error"] = str(e)
            if verbose:
                print(f"[Worker {os.getpid()}] brain 失败: {e}", file=sys.stderr)
    
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
'''


def _ensure_worker_script():
    """确保 worker 脚本存在"""
    if not WORKER_SCRIPT.exists():
        WORKER_SCRIPT.write_text(WORKER_CODE)
    return WORKER_SCRIPT


# ============ Pipeline ============

class OrganizationPipelineSubprocess:
    """
    subprocess 版本的 Pipeline（真正模拟 tmux）
    
    每个链接在完全独立的 Python 进程中处理。
    """
    
    def __init__(
        self,
        max_links: int = None,
        max_workers: int = None,
        verbose: bool = True,
    ):
        config = _load_config().get("organization_pipeline", {})
        
        self.max_links = max_links or config.get("max_links", 10)
        self.max_workers = max_workers or config.get("max_concurrent", 3)
        self.verbose = verbose
        
        # 确保 worker 脚本存在
        _ensure_worker_script()
        
        if self.verbose:
            print(f"[Pipeline-Subprocess] 完全进程隔离模式（类似 tmux）")
            print(f"[Pipeline-Subprocess] max_links={self.max_links}, max_workers={self.max_workers}")
    
    def _run_worker(self, url: str, person_name: str) -> LinkResult:
        """在子进程中运行 worker"""
        try:
            # 使用 subprocess 启动完全独立的 Python 进程
            cmd = [
                sys.executable,  # 当前 Python 解释器
                str(WORKER_SCRIPT),
                "--url", url,
                "--person", person_name,
            ]
            if self.verbose:
                cmd.append("--verbose")
            
            if self.verbose:
                print(f"[Subprocess] 启动: {url[:50]}...")
            
            # 运行子进程，超时 5 分钟
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent),  # 在项目根目录运行
            )
            
            # 打印子进程的 stderr（调试信息）
            if result.stderr and self.verbose:
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        print(f"  {line}")
            
            # 解析 stdout 中的 JSON 结果
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip().split('\n')[-1])
                return LinkResult(**data)
            else:
                return LinkResult(
                    url=url,
                    success=False,
                    mode="failed",
                    report="",
                    info_count=0,
                    error=f"Worker 退出码: {result.returncode}",
                )
        
        except subprocess.TimeoutExpired:
            return LinkResult(
                url=url,
                success=False,
                mode="failed",
                report="",
                info_count=0,
                error="超时 (5分钟)",
            )
        except Exception as e:
            return LinkResult(
                url=url,
                success=False,
                mode="failed",
                report="",
                info_count=0,
                error=str(e),
            )
    
    def run(
        self,
        person_name: str,
        organization: str = None,
        google_scholar_url: str = None,
    ) -> PipelineResult:
        """运行 Pipeline"""
        start_time = time.time()
        
        if self.verbose:
            print("=" * 60)
            print(f"[Pipeline-Subprocess] 开始搜索: {person_name}")
            print("=" * 60)
        
        # 1. 提取组织信息
        org = organization
        if not org and google_scholar_url:
            if self.verbose:
                print(f"\n[Step 1] 从 Google Scholar 提取组织信息...")
            org = get_organization(google_scholar_url)
            if self.verbose:
                print(f"  组织: {org or '未找到'}")
        
        # 2. 搜索链接
        if self.verbose:
            print(f"\n[Step 2] 搜索 {person_name} + {org or 'N/A'} 的相关链接...")
        
        links = google_search(person_name, org, max_results=self.max_links)
        
        if self.verbose:
            print(f"  找到 {len(links)} 个链接")
            for i, link in enumerate(links[:5]):
                print(f"    {i+1}. {link.url[:60]}...")
        
        if not links:
            return PipelineResult(
                person_name=person_name,
                organization=org,
                links_found=0,
                links_processed=0,
                success_count=0,
                merged_report="未找到相关链接",
            )
        
        # 3. 使用线程池并行启动子进程（线程只是启动和等待子进程，实际工作在子进程中）
        if self.verbose:
            print(f"\n[Step 3] 并行启动 {len(links)} 个独立进程...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_worker, link.url, person_name): link.url
                for link in links
            }
            
            for future in as_completed(futures):
                url = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if self.verbose:
                        status = "✓" if result.success else "✗"
                        print(f"  [{status}] {url[:50]}... ({result.mode})")
                except Exception as e:
                    results.append(LinkResult(
                        url=url,
                        success=False,
                        mode="failed",
                        report="",
                        info_count=0,
                        error=str(e),
                    ))
        
        # 4. 合并结果
        success_results = [r for r in results if r.success]
        reports = [r.report for r in success_results if r.report]
        merged = self._merge_reports(reports, person_name)
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"\n[完成] 成功: {len(success_results)}/{len(results)}, 耗时: {elapsed:.1f}s")
        
        return PipelineResult(
            person_name=person_name,
            organization=org,
            links_found=len(links),
            links_processed=len(results),
            success_count=len(success_results),
            merged_report=merged,
            link_results=results,
            elapsed_time=elapsed,
        )
    
    def _merge_reports(self, reports: List[str], person_name: str) -> str:
        if not reports:
            return f"未找到关于 {person_name} 的有效信息"
        
        merged = f"# {person_name} 信息汇总\n\n"
        merged += f"共从 {len(reports)} 个来源收集到信息：\n\n"
        
        for i, report in enumerate(reports, 1):
            merged += f"---\n## 来源 {i}\n\n{report}\n\n"
        
        return merged


# ============ 便捷函数 ============

def run_pipeline(
    person_name: str,
    organization: str = None,
    google_scholar_url: str = None,
    max_links: int = None,
    max_workers: int = None,
    verbose: bool = True,
) -> PipelineResult:
    """运行 subprocess 版 Pipeline"""
    pipeline = OrganizationPipelineSubprocess(
        max_links=max_links,
        max_workers=max_workers,
        verbose=verbose,
    )
    return pipeline.run(person_name, organization, google_scholar_url)


# ============ 测试 ============

if __name__ == "__main__":
    result = run_pipeline(
        person_name="Wang Shiqi",
        organization="City University of Hong Kong",
        max_links=10,
        max_workers=3,  # 可以开满，因为是真正的进程隔离
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Pipeline 结果")
    print("=" * 60)
    print(f"人名: {result.person_name}")
    print(f"组织: {result.organization}")
    print(f"链接: {result.links_found} 找到, {result.success_count} 成功")
    print(f"耗时: {result.elapsed_time:.1f}s")
    print("\n报告:")
    print(result.merged_report[:2000])

