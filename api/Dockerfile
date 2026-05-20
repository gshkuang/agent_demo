# Agent Service - 生产级 Docker 镜像
# 多阶段构建，减小镜像体积

# ============ 构建阶段 ============
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY api/requirements-api.txt ./

# 安装 Python 依赖到虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-api.txt

# ============ 运行阶段 ============
FROM python:3.12-slim AS runtime

WORKDIR /app

# 安全：创建非 root 用户
RUN groupadd -r agent && useradd -r -g agent agent

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY api/main.py ./
COPY memory/ ./memory/
COPY planning/ ./planning/
COPY tool_safety/ ./tool_safety/
COPY observability/ ./observability/

# 创建数据目录
RUN mkdir -p /app/data && chown -R agent:agent /app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 切换到非 root 用户
USER agent

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
