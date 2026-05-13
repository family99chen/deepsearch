from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from router import JobSubmission, _build_job_submission, track_usage
from tasks import (
    submit_patent_search_direct,
    submit_patent_search_gs,
    submit_patent_search_orcid,
)

router = APIRouter(prefix="/patents", tags=["Patent Pipeline"])


def _submit_or_503(submitter, *args, **kwargs) -> str:
    try:
        return submitter(*args, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"任务队列不可用或提交失败: {exc}") from exc


@router.post("/orcid", response_model=JobSubmission)
async def patents_by_orcid(
    request: Request,
    orcid_id: str = Query(..., description="ORCID ID，格式如 0000-0002-1825-0097"),
    use_cache: bool = Query(True, description="是否使用缓存"),
    _tracked: str = Depends(track_usage),
):
    job_id = _submit_or_503(submit_patent_search_orcid, orcid_id, use_cache)
    return _build_job_submission(request, job_id)


@router.post("/gs", response_model=JobSubmission)
async def patents_by_google_scholar(
    request: Request,
    google_scholar_url: Optional[str] = Query(None, description="Google Scholar 个人主页 URL"),
    user_id: Optional[str] = Query(None, description="Google Scholar user ID"),
    use_cache: bool = Query(True, description="是否使用缓存"),
    _tracked: str = Depends(track_usage),
):
    if not google_scholar_url and not user_id:
        raise HTTPException(status_code=400, detail="请提供 google_scholar_url 或 user_id")
    job_id = _submit_or_503(submit_patent_search_gs, google_scholar_url, user_id, use_cache)
    return _build_job_submission(request, job_id)


@router.post("/search", response_model=JobSubmission)
async def patents_by_name_org(
    request: Request,
    person_name: str = Query(..., description="姓名"),
    organization: str = Query(..., description="组织/机构"),
    use_cache: bool = Query(True, description="是否使用缓存"),
    _tracked: str = Depends(track_usage),
):
    job_id = _submit_or_503(submit_patent_search_direct, person_name, organization, use_cache)
    return _build_job_submission(request, job_id)
