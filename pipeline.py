"""
总 Pipeline：
- 输入 Google Scholar URL
- 提取人名与组织
- 依次运行 organization_pipeline 与 social_media_pipeline
- 交给分析 AI 判断是否信息足够
- 不足则生成 query，走 arbitrary_pipeline 迭代补充
"""

import sys
import re
import json
import asyncio
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# 兼容 google_scholar_url 内部的相对导入
sys.path.insert(0, str(project_root / "google_scholar_url"))

from org_info.extract_org_info import get_profile_info
from google_scholar_url.google_account_fetcher_pipeline import (
    find_google_scholar_by_orcid,
    _get_cache as _get_orcid_cache,
)
from google_scholar_url.fetch_author_person_info import _get_cache as _get_orcid_person_cache
from google_scholar_url.fetch_person_organization import _get_cache as _get_orcid_org_cache
from org_info.organization_pipeline import run_pipeline as run_org_pipeline
from org_info.social_media_pipeline import run_social_media_pipeline
from org_info.arbitrary_pipeline import run_arbitrary_pipeline
from llm import query_async


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


async def _query_strong_async(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    backend: str,
) -> str:
    return await query_async(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.3,
        model=model,
        backend=backend,
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


@dataclass
class FinalResult:
    person_name: str
    organization: Optional[str]
    report: str
    iterations: int
    queries: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


class PersonPipeline:
    def __init__(
        self,
        max_iterations: int = 3,
        max_links: int = 10,
        max_workers: int = 3,
        verbose: bool = True,
        model: str = "gpt-5.2",
        backend: str = "api",
    ):
        self.max_iterations = max_iterations
        self.max_links = max_links
        self.max_workers = max_workers
        self.verbose = verbose
        self.model = model
        self.backend = backend

    def run(self, google_scholar_url: str, extra_sources: Optional[List[str]] = None) -> FinalResult:
        if self.verbose:
            print("=" * 60)
            print("[Pipeline] 从 Google Scholar 开始")
            print("=" * 60)

        profile = get_profile_info(google_scholar_url)
        person_name = profile.get("name")
        organization = profile.get("affiliation")
        if not person_name:
            raise ValueError("无法从 Google Scholar 获取姓名")

        if self.verbose:
            print(f"[Pipeline] 姓名: {person_name}")
            print(f"[Pipeline] 组织: {organization or '未找到'}")

        # 1) 组织信息 pipeline
        org_result = run_org_pipeline(
            person_name=person_name,
            organization=organization,
            google_scholar_url=google_scholar_url,
            max_links=self.max_links,
            max_workers=self.max_workers,
            verbose=self.verbose,
        )

        # 2) 社交媒体 pipeline
        social_result = run_social_media_pipeline(
            person_name=person_name,
            organization=organization,
            google_scholar_url=google_scholar_url,
            max_links=self.max_links,
            max_workers=self.max_workers,
            verbose=self.verbose,
        )

        scholar_detail = {
            "name": profile.get("name"),
            "affiliation": profile.get("affiliation"),
            "verified_email": profile.get("verified_email"),
            "interests": profile.get("interests"),
            "citations": profile.get("citations"),
            "h_index": profile.get("h_index"),
            "i10_index": profile.get("i10_index"),
            "coauthors": profile.get("coauthors"),
            "profile_url": profile.get("url"),
        }
        sources = [
            "## Google Scholar Person Detail (authoritative)\n"
            + json.dumps(scholar_detail, ensure_ascii=True, indent=2),
            org_result.merged_report,
            social_result.merged_report,
        ]
        if extra_sources:
            sources = extra_sources + sources
            if self.verbose:
                print("[Pipeline] extra_sources 内容:")
                for idx, item in enumerate(extra_sources, 1):
                    print(f"[extra_source {idx}] {item}")
        queries: List[str] = []

        # 3) 分析 AI 判断是否足够
        for i in range(self.max_iterations):
            try:
                decision = self._analyze_sources(
                    person_name,
                    organization,
                    sources,
                    queries,
                    iteration=i + 1,
                )
            except Exception:
                # LLM 出错时直接返回已有内容，确保不中止
                return FinalResult(
                    person_name=person_name,
                    organization=organization,
                    report="\n\n".join(sources),
                    iterations=i + 1,
                    queries=queries,
                    sources=sources,
                )
            if decision.get("sufficient"):
                report = decision.get("final_report") or self._final_report(
                    person_name,
                    organization,
                    sources,
                )
                return FinalResult(
                    person_name=person_name,
                    organization=organization,
                    report=report,
                    iterations=i + 1,
                    queries=queries,
                    sources=sources,
                )

            next_query = decision.get("next_query", "").strip()
            if not next_query:
                break
            queries.append(next_query)

            ai_person_name = decision.get("person_name")
            if self.verbose:
                if ai_person_name:
                    print(f"[Pipeline] AI 提供人名: {ai_person_name}")
                else:
                    print("[Pipeline] AI 未提供人名，仅使用 query")
            arbitrary_result = run_arbitrary_pipeline(
                query=next_query,
                google_scholar_url=google_scholar_url,
                max_links=self.max_links,
                max_workers=self.max_workers,
                verbose=self.verbose,
                person_name=ai_person_name,
                organization=organization,
            )
            sources.append(arbitrary_result.merged_report)

        # 达到最大迭代次数，仍生成最终报告
        final_report = self._final_report(person_name, organization, sources)
        return FinalResult(
            person_name=person_name,
            organization=organization,
            report=final_report,
            iterations=self.max_iterations,
            queries=queries,
            sources=sources,
        )

    def _analyze_sources(
        self,
        person_name: str,
        organization: Optional[str],
        sources: List[str],
        queries: List[str],
        iteration: int,
    ) -> Dict[str, Any]:
        system_prompt = (
            "你是信息分析助手，负责判断信息是否足够生成完整报告，并提出下一步查询。"
            "输出必须是 JSON。"
        )
        prompt = f"""
人名: {person_name}
组织: {organization or "N/A"}
迭代: {iteration}
历史查询: {queries}

已收集的来源文本（包含 org search / social media search / arbitrary search 的完整结果）:
{chr(10).join(sources)}

规则：
1. 如果信息冲突，以 Google Scholar Person Detail 为准。
2. Scholar 信息优先级最高，不得被其他来源覆盖。

完整报告参考：
1. 基本信息：姓名、职称/职位、所属机构
2. 个人主页：官网/主页链接（若无必须说明原因）
3. 社交媒体账号：LinkedIn、X/Twitter、ResearchGate、ORCID、GitHub 等（可为空但需说明检索不足）
4. 学术与研究：研究方向、代表成果/论文、学术影响（如引用、奖项）
5. 联系方式：邮箱/电话/办公地址（若无说明缺失）
6. 关系网络：合作作者、同领域伙伴或相关人物
7. 背景信息：教育经历/履历/出生地（若无说明缺失）
8. 各种相关信息

任务：
- 判断当前信息是否是完整报告。
- 如果满足：返回 sufficient=true，必须给出 final_report（完整综述）。
- 如果不满足：返回 sufficient=false，仅生成下一条 query（用于 Google 搜索），不要输出报告内容（final_report 置空）。
- 允许在 JSON 中提供可选的 person_name 字段，用于指示是否把人名包含进 query。
- 如果提供 person_name，后续爬虫会带这个人名进行定向分析；如果不提供，则只按 query 泛化搜索（可能返回相关人物信息）。

输出 JSON 格式:
{{
  "sufficient": true/false,
  "final_report": "...",
  "next_query": "...",
  "person_name": "...",
  "missing": ["..."],
  "reasoning": "..."
}}
"""
        response = _run_async(
            _query_strong_async(
                prompt,
                system_prompt,
                model=self.model,
                backend=self.backend,
            )
        )
        parsed = _extract_json(response)
        if not parsed:
            return {"sufficient": False, "next_query": "", "reasoning": "解析失败"}
        return parsed

    def _final_report(
        self,
        person_name: str,
        organization: Optional[str],
        sources: List[str],
    ) -> str:
        system_prompt = "你是信息分析助手，请基于来源内容生成完整报告。"
        prompt = f"""
人名: {person_name}
组织: {organization or "N/A"}

已收集的来源文本（包含 org search / social media search / arbitrary search 的完整结果）:
{chr(10).join(sources)}

规则：
1. 如果信息冲突，以 Google Scholar Person Detail 为准。
2. Scholar 信息优先级最高，不得被其他来源覆盖。
3. 直接输出事实性报告内容
4. 信息来源不权威没关系，也可作为报告内容，主要是不能冲突

完整报告参考：
1. 基本信息：姓名、职称/职位、所属机构
2. 个人主页：官网/主页链接（若无必须说明原因）
3. 社交媒体账号：LinkedIn、X/Twitter、ResearchGate、ORCID、GitHub 等（可为空但需说明检索不足）
4. 学术与研究：研究方向、代表成果/论文、学术影响（如引用、奖项）
5. 联系方式：邮箱/电话/办公地址（若无说明缺失）
6. 关系网络：合作作者、同领域伙伴或相关人物
7. 背景信息：教育经历/履历/出生地（若无说明缺失）
8. 各种相关信息

请生成详细报告。
"""
        try:
            response = _run_async(
                _query_strong_async(
                    prompt,
                    system_prompt,
                    model=self.model,
                    backend=self.backend,
                )
            )
            return response.strip() or "\n\n".join(sources)
        except Exception:
            return "\n\n".join(sources)

def run_person_pipeline(
    google_scholar_url: str,
    max_iterations: int = 3,
    max_links: int = 10,
    max_workers: int = 3,
    verbose: bool = True,
    model: str = "gpt-5.2",
    backend: str = "api",
) -> FinalResult:
    config = _load_config().get("person_pipeline", {})
    max_iterations = config.get("max_iterations", max_iterations)
    max_links = config.get("max_links", max_links)
    max_workers = config.get("max_workers", max_workers)
    model = config.get("model", model)
    backend = config.get("backend", backend)
    pipeline = PersonPipeline(
        max_iterations=max_iterations,
        max_links=max_links,
        max_workers=max_workers,
        verbose=verbose,
        model=model,
        backend=backend,
    )
    return pipeline.run(google_scholar_url)


def run_person_pipeline_by_orcid(
    orcid_id: str,
    max_iterations: int = 3,
    max_links: int = 10,
    max_workers: int = 3,
    verbose: bool = True,
    model: str = "gpt-5.2",
    backend: str = "api",
) -> FinalResult:
    config = _load_config().get("person_pipeline", {})
    max_iterations = config.get("max_iterations", max_iterations)
    max_links = config.get("max_links", max_links)
    max_workers = config.get("max_workers", max_workers)
    model = config.get("model", model)
    backend = config.get("backend", backend)

    gs_url, matched_candidate = find_google_scholar_by_orcid(
        orcid_id=orcid_id,
        verbose=verbose,
        use_cache=True,
    )
    if not gs_url:
        raise ValueError(f"无法通过 ORCID 找到 Google Scholar 账号: {orcid_id}")

    orcid_cached = None
    cache = _get_orcid_cache()
    if cache:
        orcid_cached = cache.get(orcid_id)

    person_cache = _get_orcid_person_cache()
    org_cache = _get_orcid_org_cache()
    orcid_person_info = {}
    orcid_org_info = {}
    if person_cache:
        orcid_person_info = {
            "given_name": person_cache.get_field(orcid_id, "given_name"),
            "family_name": person_cache.get_field(orcid_id, "family_name"),
            "credit_name": person_cache.get_field(orcid_id, "credit_name"),
            "full_name": person_cache.get_field(orcid_id, "full_name"),
        }
    if org_cache:
        orcid_org_info = {
            "primary_organization": org_cache.get_field(orcid_id, "primary_organization"),
            "organizations": org_cache.get_field(orcid_id, "organizations"),
        }

    orcid_detail = {
        "orcid_id": orcid_id,
        "google_scholar_url": gs_url,
        "matched_candidate": matched_candidate,
        "cache": orcid_cached,
        "orcid_person_cache": orcid_person_info,
        "orcid_org_cache": orcid_org_info,
    }
    extra_sources = [
        "## ORCID -> Google Scholar Mapping (authoritative)\n"
        + json.dumps(orcid_detail, ensure_ascii=True, indent=2),
    ]

    pipeline = PersonPipeline(
        max_iterations=max_iterations,
        max_links=max_links,
        max_workers=max_workers,
        verbose=verbose,
        model=model,
        backend=backend,
    )
    return pipeline.run(gs_url, extra_sources=extra_sources)


if __name__ == "__main__":
    config = _load_config().get("person_pipeline", {})
    result = run_person_pipeline_by_orcid(
        orcid_id="0000-0003-3397-3725",
        max_iterations=config.get("max_iterations", 3),
        max_links=config.get("max_links", 10),
        max_workers=config.get("max_workers", 3),
        model=config.get("model", "gpt-5.2"),
        backend=config.get("backend", "api"),
        verbose=True,
    )
    print("\n" + "=" * 60)
    print("总 Pipeline 结果")
    print("=" * 60)
    print(f"人名: {result.person_name}")
    print(f"组织: {result.organization}")
    print(f"迭代次数: {result.iterations}")
    print(f"查询历史: {result.queries}")
    print("\n报告:")
    print(result.report[:10000])

