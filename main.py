"""
DeepSearch API 服务
主入口文件
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import router

app = FastAPI(
    title="DeepSearch API",
    description="根据 ORCID ID 查找对应的 Google Scholar 账号",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "service": "DeepSearch API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "/find": "提交 ORCID 查找任务并流式输出",
            "/find/sync": "提交 ORCID 查找任务",
            "/person/report": "提交 Google Scholar 人物报告任务",
            "/person/report/orcid": "提交 ORCID 人物报告任务",
            "/jobs/{job_id}": "查询任务状态和结果",
            "/jobs/{job_id}/stream": "流式读取任务日志和结果",
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True  # 开发模式，支持热重载
    )

