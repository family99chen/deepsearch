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
from dataclasses import dataclass, field, asdict
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
from localdb.deepsearch_cache import get_person_pipeline_cache
from utils.org_pipeline_stats import (
    record_cache_hit as record_org_pipeline_cache_hit,
    record_error as record_org_pipeline_error,
    record_not_found as record_org_pipeline_not_found,
    record_request as record_org_pipeline_request,
    record_success as record_org_pipeline_success,
)


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


def _safe_record_stats(callback):
    try:
        callback()
    except Exception:
        # 统计是附属能力，不能影响主流程
        pass


def _is_not_found_exception(exc: Exception) -> bool:
    if isinstance(exc, ValueError):
        message = str(exc).lower()
        markers = ("未找到", "not found", "无法从 google scholar 获取姓名")
        return any(marker in message for marker in markers)
    return False


def _is_missing_organization(organization: Optional[str]) -> bool:
    value = (organization or "").strip()
    if not value:
        return True

    normalized = value.lower()
    missing_markers = [
        "未知所在单位机构",
        "unknown affiliation",
        "unknown organization",
        "unknown institution",
        "n/a",
        "none",
    ]
    return any(marker in normalized for marker in missing_markers)


def _pick_fallback_organization(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        if isinstance(candidate, str):
            value = candidate.strip()
            if value and not _is_missing_organization(value):
                return value
        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str):
                    value = item.strip()
                    if value and not _is_missing_organization(value):
                        return value
                elif isinstance(item, dict):
                    for key in ("name", "organization", "department-name", "value"):
                        raw = item.get(key)
                        if isinstance(raw, str):
                            value = raw.strip()
                            if value and not _is_missing_organization(value):
                                return value
    return None


def _extract_domain_from_verified_email(verified_email: Optional[str]) -> Optional[str]:
    text = (verified_email or "").strip()
    if not text:
        return None

    match = re.search(r"([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    if not match:
        return None

    domain = match.group(1).lower().strip(" .,:;)")
    return domain or None


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
    
    @staticmethod
    def _to_cache_payload(result: FinalResult) -> Dict[str, Any]:
        return {
            "person_name": result.person_name,
            "organization": result.organization,
            "report": result.report,
            "iterations": result.iterations,
            "queries": result.queries,
            "sources": result.sources,
        }

    @staticmethod
    def _normalize_report_text(report: Optional[str]) -> Optional[str]:
        if not report:
            return None
        value = report.strip()
        return value or None

    @staticmethod
    def _build_failure_report(person_name: str) -> str:
        return f"DeepSearch failed: unable to generate the final report for {person_name}. Please try again later."

    def run(
        self,
        google_scholar_url: str,
        extra_sources: Optional[List[str]] = None,
        fallback_organization: Optional[str] = None,
    ) -> FinalResult:
        _safe_record_stats(record_org_pipeline_request)
        pipeline_cache = get_person_pipeline_cache()
        try:
            cached_result = pipeline_cache.get_final_result(
                google_scholar_url=google_scholar_url,
                max_iterations=self.max_iterations,
                max_links=self.max_links,
                max_workers=self.max_workers,
                model=self.model,
                backend=self.backend,
                extra_sources=extra_sources,
            )
            if cached_result:
                final = cached_result["final"]
                _safe_record_stats(record_org_pipeline_cache_hit)
                if self.verbose:
                    print("=" * 60)
                    print("[Pipeline] 命中 person_pipeline_cache")
                    print("=" * 60)
                    print(f"[CACHE] Google Scholar URL: {google_scholar_url}")
                result = FinalResult(
                    person_name=final.get("person_name", ""),
                    organization=final.get("organization"),
                    report=final.get("report", ""),
                    iterations=final.get("iterations", 0),
                    queries=final.get("queries", []),
                    sources=final.get("sources", []),
                )
                _safe_record_stats(record_org_pipeline_success)
                return result

            if self.verbose:
                print("=" * 60)
                print("[Pipeline] 从 Google Scholar 开始")
                print("=" * 60)

            profile = get_profile_info(google_scholar_url)
            person_name = profile.get("name")
            organization = profile.get("affiliation")
            original_organization = organization
            scholar_email_domain = _extract_domain_from_verified_email(profile.get("verified_email"))
            resolved_fallback_organization = _pick_fallback_organization(
                fallback_organization,
                scholar_email_domain,
            )
            if _is_missing_organization(organization) and resolved_fallback_organization:
                organization = resolved_fallback_organization
            if not person_name:
                raise ValueError("无法从 Google Scholar 获取姓名")

            if self.verbose:
                print(f"[Pipeline] 姓名: {person_name}")
                print(f"[Pipeline] 组织: {organization or '未找到'}")
                if organization != original_organization and resolved_fallback_organization:
                    print(f"[Pipeline] 组织回退来源: {resolved_fallback_organization}")

            stage_data: Dict[str, Any] = {
                "profile": {
                    "name": profile.get("name"),
                    "affiliation": profile.get("affiliation"),
                    "resolved_organization": organization,
                    "verified_email_domain": scholar_email_domain,
                    "verified_email": profile.get("verified_email"),
                    "interests": profile.get("interests"),
                    "citations": profile.get("citations"),
                    "h_index": profile.get("h_index"),
                    "i10_index": profile.get("i10_index"),
                    "coauthors": profile.get("coauthors"),
                    "profile_url": profile.get("url"),
                },
                "arbitrary_pipeline": {},
            }

            def _save_and_return(result: FinalResult) -> FinalResult:
                pipeline_cache.set_final_result(
                    google_scholar_url=google_scholar_url,
                    max_iterations=self.max_iterations,
                    max_links=self.max_links,
                    max_workers=self.max_workers,
                    model=self.model,
                    backend=self.backend,
                    result=self._to_cache_payload(result),
                    extra_sources=extra_sources,
                    stages=stage_data,
                )
                return result

            # 1) 组织信息 pipeline
            org_result = run_org_pipeline(
                person_name=person_name,
                organization=organization,
                google_scholar_url=google_scholar_url,
                max_links=self.max_links,
                max_workers=self.max_workers,
                verbose=self.verbose,
            )
            stage_data["organization_pipeline"] = asdict(org_result)

            # 2) 社交媒体 pipeline
            social_result = run_social_media_pipeline(
                person_name=person_name,
                organization=organization,
                google_scholar_url=google_scholar_url,
                max_links=self.max_links,
                max_workers=self.max_workers,
                verbose=self.verbose,
            )
            stage_data["social_media_pipeline"] = asdict(social_result)

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

            def _return_failed_result(iterations: int) -> FinalResult:
                _safe_record_stats(record_org_pipeline_not_found)
                return FinalResult(
                    person_name=person_name,
                    organization=organization,
                    report=self._build_failure_report(person_name),
                    iterations=iterations,
                    queries=queries.copy(),
                    sources=[],
                )

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
                    report = self._final_report(person_name, organization, sources)
                    normalized_report = self._normalize_report_text(report)
                    if normalized_report:
                        result = _save_and_return(FinalResult(
                            person_name=person_name,
                            organization=organization,
                            report=normalized_report,
                            iterations=i + 1,
                            queries=queries,
                            sources=sources,
                        ))
                        _safe_record_stats(record_org_pipeline_success)
                        return result
                    return _return_failed_result(i + 1)
                if decision.get("sufficient"):
                    report = self._normalize_report_text(decision.get("final_report")) or self._final_report(
                        person_name,
                        organization,
                        sources,
                    )
                    normalized_report = self._normalize_report_text(report)
                    if not normalized_report:
                        return _return_failed_result(i + 1)
                    stage_data["final_decision"] = decision
                    result = _save_and_return(FinalResult(
                        person_name=person_name,
                        organization=organization,
                        report=normalized_report,
                        iterations=i + 1,
                        queries=queries,
                        sources=sources,
                    ))
                    _safe_record_stats(record_org_pipeline_success)
                    return result

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
                stage_data["arbitrary_pipeline"][next_query] = asdict(arbitrary_result)
                sources.append(arbitrary_result.merged_report)

            # 达到最大迭代次数，仍生成最终报告
            final_report = self._final_report(person_name, organization, sources)
            normalized_report = self._normalize_report_text(final_report)
            if not normalized_report:
                return _return_failed_result(self.max_iterations)
            result = _save_and_return(FinalResult(
                person_name=person_name,
                organization=organization,
                report=normalized_report,
                iterations=self.max_iterations,
                queries=queries,
                sources=sources,
            ))
            _safe_record_stats(record_org_pipeline_success)
            return result
        except Exception as exc:
            if _is_not_found_exception(exc):
                _safe_record_stats(record_org_pipeline_not_found)
            else:
                _safe_record_stats(record_org_pipeline_error)
            raise

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
5. 个人成就：重要荣誉、奖项、头衔、入选计划、里程碑成果等（若无说明缺失）
6. 联系方式：邮箱/电话/办公地址（若无说明缺失）
7. 关系网络：合作作者、同领域伙伴或相关人物
8. 背景信息：教育经历/履历/出生地（若无说明缺失）
9. 各种相关信息

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
    ) -> Optional[str]:
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
5. 个人成就：重要荣誉、奖项、头衔、入选计划、里程碑成果等（若无说明缺失）
6. 联系方式：邮箱/电话/办公地址（若无说明缺失）
7. 关系网络：合作作者、同领域伙伴或相关人物
8. 背景信息：教育经历/履历/出生地（若无说明缺失）
9. 各种相关信息

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
            return self._normalize_report_text(response)
        except Exception:
            return None

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
    fallback_organization = _pick_fallback_organization(
        matched_candidate.get("affiliation") if isinstance(matched_candidate, dict) else None,
        orcid_cached.get("affiliation") if isinstance(orcid_cached, dict) else None,
        orcid_org_info.get("primary_organization"),
        orcid_org_info.get("organizations"),
    )

    pipeline = PersonPipeline(
        max_iterations=max_iterations,
        max_links=max_links,
        max_workers=max_workers,
        verbose=verbose,
        model=model,
        backend=backend,
    )
    return pipeline.run(
        gs_url,
        extra_sources=extra_sources,
        fallback_organization=fallback_organization,
    )


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

