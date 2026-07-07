# -*- coding: utf-8 -*-
"""
===================================
Daily Stock Analysis - FastAPI 后端服务入口
===================================

职责：
1. 提供 RESTful API 服务
2. 配置 CORS 跨域支持
3. 健康检查接口
4. 托管前端静态文件（生产模式）

启动方式：
    uvicorn server:app --reload --host 127.0.0.1 --port 8000
    
    或使用 main.py:
    python main.py --serve-only      # 仅启动 API 服务
    python main.py --serve           # API 服务 + 执行分析
"""

import logging
import os
from pathlib import Path
import sys

from src.config import setup_env, get_config
from src.logging_config import setup_logging
from src.web_bind import infer_uvicorn_cli_host, publish_bind_host

# 初始化环境变量与日志
setup_env()

config = get_config()
level_name = (config.log_level or "INFO").upper()
level = getattr(logging, level_name, logging.INFO)

setup_logging(
    log_prefix="api_server",
    console_level=level,
    extra_quiet_loggers=['uvicorn', 'fastapi'],
)

_declared_host = (
    os.getenv("DSA_BIND_HOST")
    or os.getenv("UVICORN_HOST")
    or infer_uvicorn_cli_host()
)
if not _declared_host and Path(sys.argv[0]).name == "server.py":
    _declared_host = os.getenv("WEBUI_HOST", os.getenv("API_HOST", "127.0.0.1"))
if _declared_host:
    publish_bind_host(_declared_host)

# 从 api.app 导入应用实例
from api.app import app  # noqa: E402

# 导出 app 供 uvicorn 使用
__all__ = ['app']


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("WEBUI_HOST", os.getenv("API_HOST", "127.0.0.1"))
    port = int(os.getenv("WEBUI_PORT", os.getenv("API_PORT", "8000")))
    publish_bind_host(host)
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
    )
