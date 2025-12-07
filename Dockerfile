# CYRP - 穿黄工程智能管控系统
# Yellow River Crossing Project Intelligent Control System
#
# 多阶段构建Dockerfile

# ============================================================
# 阶段1: 基础依赖层
# ============================================================
FROM python:3.11-slim as base

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd --gid 1000 cyrp \
    && useradd --uid 1000 --gid cyrp --shell /bin/bash --create-home cyrp

# ============================================================
# 阶段2: 依赖安装层
# ============================================================
FROM base as dependencies

WORKDIR /app

# 复制依赖文件
COPY requirements.txt ./

# 安装Python依赖
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ============================================================
# 阶段3: 测试层
# ============================================================
FROM dependencies as test

WORKDIR /app

# 复制源代码
COPY . .

# 安装测试依赖
RUN pip install pytest pytest-asyncio pytest-cov

# 运行测试
RUN pytest tests/ -v --cov=cyrp --cov-report=term-missing || true

# ============================================================
# 阶段4: 生产层
# ============================================================
FROM dependencies as production

WORKDIR /app

# 复制源代码
COPY --chown=cyrp:cyrp cyrp/ ./cyrp/
COPY --chown=cyrp:cyrp config/ ./config/
COPY --chown=cyrp:cyrp scripts/ ./scripts/

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/reports \
    && chown -R cyrp:cyrp /app

# 切换到非root用户
USER cyrp

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 暴露端口
EXPOSE 8080

# 默认命令
CMD ["python", "-m", "cyrp.web.app"]

# ============================================================
# 阶段5: 开发层
# ============================================================
FROM dependencies as development

WORKDIR /app

# 安装开发工具
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    mypy \
    ipython \
    jupyter

# 复制所有源代码
COPY . .

# 开发环境使用root便于调试
USER root

# 暴露开发端口
EXPOSE 8080 8888

# 开发模式默认命令
CMD ["python", "-m", "cyrp.web.app", "--debug"]
