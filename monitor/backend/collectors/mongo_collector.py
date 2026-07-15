from __future__ import annotations

from typing import Any, Dict, List

from pymongo import MongoClient

from config import MONGO_DB, MONGO_URL


def collect_mongo() -> Dict[str, Any]:
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[MONGO_DB]
        dbstats = db.command("dbStats")

        collections: List[Dict[str, Any]] = []
        for name in sorted(db.list_collection_names()):
            stats = db.command("collStats", name)
            collections.append({
                "name": name,
                "count": stats.get("count", 0),
                "size_mb": round(stats.get("size", 0) / 1024 / 1024, 1),
                "storage_mb": round(stats.get("storageSize", 0) / 1024 / 1024, 1),
            })

        collections.sort(key=lambda x: x["size_mb"], reverse=True)
        return {
            "ok": True,
            "db": MONGO_DB,
            "data_size_gb": round(dbstats.get("dataSize", 0) / 1024 / 1024 / 1024, 2),
            "storage_size_gb": round(dbstats.get("storageSize", 0) / 1024 / 1024 / 1024, 2),
            "index_size_mb": round(dbstats.get("indexSize", 0) / 1024 / 1024, 1),
            "collections": collections,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
