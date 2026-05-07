#!/usr/bin/env python3
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.config import ConfigManager
from core.token_manager import TokenManager
from core.log_store import LogStore
from routes import openai_compat, admin_api, claude_api

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("tabbit2openai")

# ── 初始化核心组件 ──
cfg = ConfigManager()
token_manager = TokenManager(cfg)
log_store = LogStore(max_entries=cfg.get("logging", "max_entries", default=500))

# ── 初始化路由模块 ──
openai_compat.init(token_manager, cfg, log_store)
admin_api.init(cfg, token_manager, log_store)
claude_api.init(token_manager, cfg, log_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Tabbit2API started — tokens: %d, port: %d",
        len(cfg.get("tokens", default=[])),
        cfg.get("server", "port", default=8800),
    )
    yield
    await token_manager.close_all()


app = FastAPI(lifespan=lifespan)


# ── OpenAI 标准错误格式中间件 ──
from fastapi.exceptions import RequestValidationError
from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(HTTPException)
async def openai_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "code": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        },
    )


# ── 挂载路由 ──
app.include_router(claude_api.router)  # Claude Messages API（/v1/messages）
app.include_router(openai_compat.router)  # OpenAI 兼容（/v1/chat/completions）
app.include_router(admin_api.router)

# ── 静态文件 & 管理面板入口 ──
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/admin")
async def admin_page():
    return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    uvicorn.run(
        app,
        host=cfg.get("server", "host", default="0.0.0.0"),
        port=cfg.get("server", "port", default=8800),
    )
