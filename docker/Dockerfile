# Multi-stage Dockerfile for AI-Powered Video Advertisement Placement System
# Base: nvidia/cuda:11.8-devel-ubuntu22.04

# =============================================================================
# Stage 1: Base Image with System Dependencies
# =============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # Graphics and video processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # FFmpeg with hardware acceleration
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    # OpenGL and rendering
    libglfw3-dev \
    libglew-dev \
    libopengl0 \
    libgl1-mesa-dev \
    # Additional dependencies
    pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    # System utilities
    htop \
    nano \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN 8
RUN wget https://developer.download.nvidia.com/compute/cudnn/8.9.7/local_installers/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz \
    && tar -xf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz \
    && cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/include/cudnn*.h /usr/local/cuda/include \
    && cp -P cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64 \
    && chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* \
    && rm -rf cudnn-linux-x86_64-8.9.7.29_cuda11-archive*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app /data /models /cache /tmp/video_processing && \
    chown -R appuser:appuser /app /data /models /cache /tmp/video_processing

# =============================================================================
# Stage 2: Python Dependencies Installation
# =============================================================================
FROM base as python-deps

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install additional production dependencies
RUN pip install \
    gunicorn==21.2.0 \
    redis==4.6.0 \
    psycopg2-binary==2.9.7 \
    celery==5.3.1 \
    prometheus-client==0.17.1 \
    structlog==23.1.0

# =============================================================================
# Stage 3: Development Stage
# =============================================================================
FROM python-deps as development

# Install development tools
RUN pip install \
    jupyter==1.0.0 \
    ipdb==0.13.13 \
    pytest-xdist==3.3.1 \
    coverage==7.3.0

# Set development environment
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG

USER appuser
WORKDIR /app

# Default command for development
CMD ["python3", "-m", "video_ad_placement.api.main"]

# =============================================================================
# Stage 4: Production Stage
# =============================================================================
FROM python-deps as production

# Copy application code
COPY --chown=appuser:appuser . /app

# Install the package
WORKDIR /app
USER appuser
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache

# Set production environment
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO
ENV MODEL_CACHE_DIR=/models
ENV DATA_DIR=/data
ENV CACHE_DIR=/cache
ENV TMP_DIR=/tmp/video_processing

# Health check script
COPY --chown=appuser:appuser docker/scripts/healthcheck.py /app/healthcheck.py
RUN chmod +x /app/healthcheck.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /app/healthcheck.py || exit 1

# Signal handling script
COPY --chown=appuser:appuser docker/scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "video_ad_placement.api.main:app"]

# =============================================================================
# Stage 5: Worker Stage (for Celery workers)
# =============================================================================
FROM production as worker

# Worker-specific environment
ENV WORKER_TYPE=video_processor
ENV CELERY_CONCURRENCY=2

# Worker command
CMD ["celery", "worker", "-A", "video_ad_placement.worker.celery_app", \
     "--loglevel=info", "--concurrency=2", "--pool=prefork"]

# =============================================================================
# Stage 6: Model Downloader Stage
# =============================================================================
FROM python-deps as model-downloader

USER appuser
WORKDIR /models

# Copy model download script
COPY --chown=appuser:appuser docker/scripts/download_models.py /app/download_models.py

# Download and cache models
RUN python3 /app/download_models.py

CMD ["echo", "Models downloaded successfully"] 