FROM python:3.11-slim

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DOA_DATA_DIR=/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 容器内由 uvicorn 监听 8000；会话数据目录 /data 可挂载
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
