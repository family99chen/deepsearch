from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import TOTAL_USAGE_DIR, USAGE_FILES, USAGE_LABELS


def last_n_days(n: int = 7, end: Optional[date] = None) -> List[str]:
    end = end or date.today()
    return [(end - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def available_dates_for_file(file_key: str, n: int = 7) -> List[str]:
    path = TOTAL_USAGE_DIR / USAGE_FILES[file_key]
    if not path.exists():
        return last_n_days(n)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        daily = data.get("daily", {})
        candidates = last_n_days(n)
        return [d for d in candidates if d in daily] or candidates
    except Exception:
        return last_n_days(n)


def get_usage_for_date(file_key: str, day: str) -> Dict[str, Any]:
    if file_key not in USAGE_FILES:
        return {"ok": False, "error": f"unknown file key: {file_key}"}

    path = TOTAL_USAGE_DIR / USAGE_FILES[file_key]
    label = USAGE_LABELS.get(file_key, file_key)

    if not path.exists():
        return {"ok": False, "file_key": file_key, "label": label, "date": day, "error": "file not found"}

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {"ok": False, "file_key": file_key, "label": label, "date": day, "error": str(exc)}

    daily = data.get("daily", {})
    day_data = daily.get(day)
    has_data = day_data is not None

    return {
        "ok": True,
        "file_key": file_key,
        "label": label,
        "filename": USAGE_FILES[file_key],
        "date": day,
        "has_data": has_data,
        "data": day_data,
        "last_updated": data.get("last_updated"),
        "available_dates": [d for d in last_n_days(7) if d in daily] or last_n_days(7),
    }


def get_usage_batch(day: str) -> Dict[str, Any]:
    items = [get_usage_for_date(key, day) for key in USAGE_FILES]
    return {
        "ok": True,
        "date": day,
        "items": items,
        "default_date": day,
    }


def list_usage_meta() -> Dict[str, Any]:
    files = []
    for key, filename in USAGE_FILES.items():
        path = TOTAL_USAGE_DIR / filename
        files.append({
            "file_key": key,
            "label": USAGE_LABELS.get(key, key),
            "filename": filename,
            "exists": path.exists(),
            "available_dates": available_dates_for_file(key, 7),
        })
    return {
        "ok": True,
        "today": date.today().isoformat(),
        "dates": last_n_days(7),
        "files": files,
    }
