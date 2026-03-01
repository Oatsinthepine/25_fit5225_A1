# 核心设计思路

# 1️ 基础镜像
FROM python:3.10-slim

# 2 环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3 设置工作目录
WORKDIR /app

# 4 复制 requirements
COPY cloudpose/requirements.txt .

# 5 安装系统依赖（opencv 需要）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 6 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 7 复制项目代码
COPY cloudpose/app ./app
# COPY cloudpose/app/model ./model

# 8 暴露端口
EXPOSE 60000

# 9 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "60000"]