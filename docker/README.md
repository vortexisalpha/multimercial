# Docker Deployment for Video Advertisement Placement System

## Quick Start

### Development Environment

```bash
# Clone repository and navigate to project
cd video-ad-placement

# Start development environment with hot reload
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Check services status
docker-compose ps

# View logs
docker-compose logs -f app
```

**Access Points:**
- API: http://localhost:8000
- Jupyter Lab: http://localhost:8888 (token in logs)
- Grafana: http://localhost:3001 (admin/admin)
- Prometheus: http://localhost:9091

### Production Environment

```bash
# Build production image
docker build -f docker/Dockerfile -t video-ad-placement:latest .

# Start production stack
docker-compose --profile production up -d

# Scale workers
docker-compose up -d --scale worker=5
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   App Server    │    │     Worker      │
│  Load Balancer  │────│   (Gunicorn)    │    │   (Celery)      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         │              │     Redis       │─────────────┘
         │              │  Cache/Queue    │
         │              └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         └──────────────│   PostgreSQL    │
                        │    Database     │
                        └─────────────────┘
```

## Container Details

### Main Application (`app`)
- **Base**: nvidia/cuda:11.8-devel-ubuntu22.04
- **GPU**: 1 GPU per container
- **Memory**: 8-16GB
- **Purpose**: API server and request handling

### Worker (`worker`)
- **Base**: Same as app
- **GPU**: 1 GPU per container  
- **Memory**: 6-12GB
- **Purpose**: Video processing tasks

### Database (`postgres`)
- **Image**: postgres:15-alpine
- **Memory**: 2-4GB
- **Storage**: Persistent volume for data

### Cache (`redis`)
- **Image**: redis:7-alpine
- **Memory**: 1-2GB
- **Purpose**: Job queue and caching

### Monitoring (`prometheus`, `grafana`)
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards

## GPU Requirements

### Supported GPUs
- NVIDIA Tesla V100, A100
- NVIDIA RTX 30xx/40xx series
- Minimum 8GB VRAM

### Setup
```bash
# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

## Environment Variables

### Required Variables
```bash
# Database
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=video_ad_placement

# Application
SECRET_KEY=your_secret_key_here
ENVIRONMENT=production

# GPU
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Optional Variables
```bash
# Performance
GUNICORN_WORKERS=4
CELERY_CONCURRENCY=2
PRELOAD_MODELS=true

# Logging
LOG_LEVEL=INFO
```

## Volume Mounts

```yaml
volumes:
  # Model cache (persistent, read-only for workers)
  models_cache:/models

  # Video data (persistent, read-write)
  video_data:/data

  # Database data (persistent)
  postgres_data:/var/lib/postgresql/data

  # Redis data (persistent)
  redis_data:/data

  # Temporary processing (ephemeral)
  tmp_processing:/tmp/video_processing
```

## Health Checks

### Application Health
```bash
# API health check
curl http://localhost:8000/health

# Worker health (Celery)
docker-compose exec worker celery inspect ping
```

### GPU Health
```bash
# Check GPU utilization
docker-compose exec app nvidia-smi

# Monitor GPU memory
watch -n 1 'docker-compose exec app nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

## Scaling

### Horizontal Scaling
```bash
# Scale application servers
docker-compose up -d --scale app=3

# Scale workers
docker-compose up -d --scale worker=5

# Scale with specific GPU allocation
docker-compose up -d --scale worker=2
```

### Resource Limits
```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          memory: 12G
          cpus: '3.0'
        reservations:
          memory: 6G
          cpus: '1.5'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Development Features

### Hot Reload
- Source code mounted as volume
- Automatic restart on changes
- Jupyter Lab for experimentation

### Development Tools
```bash
# Access development container
docker-compose exec dev_tools bash

# Run tests
docker-compose exec dev_tools pytest /app/tests

# Code formatting
docker-compose exec dev_tools black /app/src

# Type checking
docker-compose exec dev_tools mypy /app/src
```

## Production Features

### Multi-stage Build
- Optimized base image
- Minimal runtime dependencies
- Security hardening

### Health Checks
- Application readiness probes
- GPU availability checks
- Database connectivity

### Resource Management
- GPU memory limits
- CPU/memory constraints
- Restart policies

## Monitoring

### Metrics Endpoints
- App metrics: http://localhost:8000/metrics
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3001

### Key Metrics
- GPU utilization and memory
- Processing queue length
- API response times
- Error rates

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Check NVIDIA runtime
   docker info | grep nvidia
   
   # Verify GPU in container
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

2. **Out of Memory**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Check GPU memory
   docker-compose exec app nvidia-smi
   ```

3. **Model Download Issues**
   ```bash
   # Check model download logs
   docker-compose logs model_downloader
   
   # Manual download
   docker-compose exec app python3 /app/docker/scripts/download_models.py
   ```

### Debug Commands
```bash
# Check all container status
docker-compose ps

# View container logs
docker-compose logs -f <service_name>

# Execute commands in container
docker-compose exec <service_name> bash

# Check resource usage
docker stats $(docker-compose ps -q)
```

## Security

### Image Security
- Non-root user (uid: 1000)
- Minimal base image
- Regular security updates

### Network Security
- Internal networks for services
- No unnecessary port exposure
- TLS for external connections

### Secret Management
- Environment variables for secrets
- Docker secrets support
- No hardcoded credentials

## Performance Optimization

### Model Optimization
- Model caching and preloading
- Batch processing
- GPU memory pooling

### Infrastructure
- SSD storage for models
- Optimized Docker layers
- Efficient GPU scheduling

For detailed production deployment, see [Deployment Guide](../docs/DEPLOYMENT.md). 