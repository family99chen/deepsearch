"""
Social Media Pipeline - subprocess 版本（真正模拟 tmux）

使用 subprocess 启动完全独立的 Python 进程：
- 完全独立的 Python 解释器
- 没有任何状态共享
- 每个进程有自己的 chromedriver 缓存
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入社交媒体搜索模块
from org_info.social_media_search import (
    search_person_in_social_media_with_raw as social_search_with_raw,
    SocialMediaLink,
)

# 复用已有 worker
WORKER_SCRIPT = Path(__file__).parent / "_worker.py"


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
    social_search_raw: Optional[Dict[str, Any]] = None
    elapsed_time: float = 0.0


class SocialMediaPipelineSubprocess:
    """
    subprocess 版本的 Social Media Pipeline（真正模拟 tmux）

    每个链接在完全独立的 Python 进程中处理。
    """

    def __init__(
        self,
        max_links: int = 10,
        max_workers: int = 3,
        verbose: bool = True,
    ):
        self.max_links = max_links
        self.max_workers = max_workers
        self.verbose = verbose

        if self.verbose:
            print("[SocialMedia-Pipeline] 完全进程隔离模式（类似 tmux）")
            print(f"[SocialMedia-Pipeline] max_links={self.max_links}, max_workers={self.max_workers}")

    def _run_worker(self, url: str, person_name: str) -> LinkResult:
        """在子进程中运行 worker"""
        try:
            cmd = [
                sys.executable,
                str(WORKER_SCRIPT),
                "--url", url,
                "--person", person_name,
            ]
            if self.verbose:
                cmd.append("--verbose")

            if self.verbose:
                print(f"[Subprocess] 启动: {url[:50]}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent),
            )

            if result.stderr and self.verbose:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        print(f"  {line}")

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip().split("\n")[-1])
                return LinkResult(**data)
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
        organization: Optional[str],
        google_scholar_url: Optional[str] = None,
    ) -> PipelineResult:
        start_time = time.time()

        if self.verbose:
            print("=" * 60)
            print(f"[SocialMedia-Pipeline] 开始搜索: {person_name}")
            print("=" * 60)

        # 1. 搜索社交媒体链接
        if self.verbose:
            print(f"\n[Step 1] 搜索 {person_name} + {organization or 'N/A'} 的社交媒体链接...")

        links, social_raw = social_search_with_raw(
            person_name,
            organization,
            max_results=self.max_links,
            google_scholar_url=google_scholar_url,
        )

        if self.verbose:
            print(f"  找到 {len(links)} 个链接")
            for i, link in enumerate(links[:5]):
                print(f"    {i+1}. {link.url[:60]}...")

        if not links:
            merged = self._merge_reports([], person_name)
            merged += "\n---\n## Social Media Search 原始内容\n\n"
            merged += "```json\n"
            merged += json.dumps(social_raw, ensure_ascii=True, indent=2)
            merged += "\n```\n"
            return PipelineResult(
                person_name=person_name,
                organization=organization,
                links_found=0,
                links_processed=0,
                success_count=0,
                merged_report=merged,
                social_search_raw=social_raw,
            )

        # 2. 并行启动子进程处理链接
        if self.verbose:
            print(f"\n[Step 2] 并行启动 {len(links)} 个独立进程...")

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

        # 3. 合并结果
        success_results = [r for r in results if r.success]
        reports = [r.report for r in success_results if r.report]
        merged = self._merge_reports(reports, person_name)
        merged += "\n---\n## Social Media Search 原始内容\n\n"
        merged += "```json\n"
        merged += json.dumps(social_raw, ensure_ascii=True, indent=2)
        merged += "\n```\n"

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\n[完成] 成功: {len(success_results)}/{len(results)}, 耗时: {elapsed:.1f}s")

        return PipelineResult(
            person_name=person_name,
            organization=organization,
            links_found=len(links),
            links_processed=len(results),
            success_count=len(success_results),
            merged_report=merged,
            link_results=results,
            social_search_raw=social_raw,
            elapsed_time=elapsed,
        )

    def _merge_reports(self, reports: List[str], person_name: str) -> str:
        if not reports:
            return f"未找到关于 {person_name} 的有效社交媒体信息"

        merged = f"# {person_name} 社交媒体信息汇总\n\n"
        merged += f"共从 {len(reports)} 个来源收集到信息：\n\n"
        for i, report in enumerate(reports, 1):
            merged += f"---\n## 来源 {i}\n\n{report}\n\n"
        return merged


def run_social_media_pipeline(
    person_name: str,
    organization: str,
    google_scholar_url: Optional[str] = None,
    max_links: int = 10,
    max_workers: int = 3,
    verbose: bool = True,
) -> PipelineResult:
    pipeline = SocialMediaPipelineSubprocess(
        max_links=max_links,
        max_workers=max_workers,
        verbose=verbose,
    )
    return pipeline.run(person_name, organization, google_scholar_url)


if __name__ == "__main__":
    result = run_social_media_pipeline(
        person_name="Wang Shiqi",
        organization="City University of Hong Kong",
        google_scholar_url="https://scholar.google.com/citations?user=Pr7s2VUAAAAJ&hl=en",
        max_links=10,
        max_workers=3,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Social Media Pipeline 结果")
    print("=" * 60)
    print(f"人名: {result.person_name}")
    print(f"组织: {result.organization}")
    print(f"链接: {result.links_found} 找到, {result.success_count} 成功")
    print(f"耗时: {result.elapsed_time:.1f}s")
    print("\n报告:")
    print(result.merged_report[:5000])

