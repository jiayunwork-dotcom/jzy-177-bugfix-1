"""``python -m app`` —— uvicorn 监听容器内 8000 端口。"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=os.environ.get("DOA_HOST", "0.0.0.0"),
        port=int(os.environ.get("DOA_PORT", "8000")),
        reload=False,
    )
