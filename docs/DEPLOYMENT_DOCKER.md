# Deployment Guide

## AI-Powered Video Advertisement Placement System

This guide covers the complete deployment process for the video advertisement placement system using Docker and Kubernetes with NVIDIA GPU support.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Production Scaling](#production-scaling)
5. [Monitoring & Observability](#monitoring--observability)
6. [Security Considerations](#security-considerations)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **GPU Nodes**: NVIDIA Tesla V100, A100, or RTX series with 8GB+ VRAM
- **CPU**: 8+ cores per GPU node
- **Memory**: 32GB+ RAM per GPU node
- **Storage**: 
  - Fast SSD for model cache (100GB+)
  - Network storage for video data (1TB+)

### Software Requirements

- Docker 20.10+ with NVIDIA Container Runtime
- Kubernetes 1.24+ with GPU Operator
- kubectl configured for cluster access
- Helm 3.0+ (optional, for easier deployments)

### GPU Setup

1. **Install NVIDIA Drivers**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-driver-535
   ```

2. **Install Docker with NVIDIA Runtime**:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install NVIDIA Container Runtime
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Verify GPU Access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

## Docker Deployment

### Quick Start (Development)

1. **Clone and Build**:
   ```bash
   git clone <repository-url>
   cd video-ad-placement
   
   # Build development image
   docker build -f docker/Dockerfile.dev -t video-ad-placement:dev .
   ```

2. **Start Development Environment**:
   ```bash
   # Start with hot reload
   docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
   
   # Check services
   docker-compose ps
   ```

3. **Access Services**:
   - API: http://localhost:8000
   - Jupyter Lab: http://localhost:8888
   - Grafana: http://localhost:3001 (admin/admin)
   - Prometheus: http://localhost:9091

### Production Deployment

1. **Build Production Image**:
   ```bash
   # Multi-stage production build
   docker build -f docker/Dockerfile -t video-ad-placement:latest .
   
   # Tag for registry
   docker tag video-ad-placement:latest your-registry.com/video-ad-placement:v1.0.0
   docker push your-registry.com/video-ad-placement:v1.0.0
   ```

2. **Production Deployment**:
   ```bash
   # Update environment variables
   cp .env.example .env
   # Edit .env with production values
   
   # Deploy production stack
   docker-compose --profile production up -d
   ```

3. **Scale Services**:
   ```bash
   # Scale workers based on load
   docker-compose up -d --scale worker=5
   
   # Scale application instances
   docker-compose up -d --scale app=3
   ```

### Environment Configuration

Create `.env` file:
```bash
# Database
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=video_ad_placement

# Redis
REDIS_PASSWORD=your_redis_password

# Application
SECRET_KEY=your_secret_key_here
API_TOKEN=your_api_token_here
ENVIRONMENT=production
LOG_LEVEL=INFO

# GPU Settings
CUDA_VISIBLE_DEVICES=0,1,2,3
NVIDIA_VISIBLE_DEVICES=all

# Model Settings
MODEL_CACHE_DIR=/models
PRELOAD_MODELS=true

# Performance
GUNICORN_WORKERS=4
CELERY_CONCURRENCY=2
```

## Kubernetes Deployment

### Cluster Setup

1. **Install GPU Operator**:
   ```bash
   # Add NVIDIA Helm repository
   helm repo add nvidia https://nvidia.github.io/gpu-operator
   helm repo update
   
   # Install GPU Operator
   helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator
   ```

2. **Verify GPU Nodes**:
   ```bash
   kubectl get nodes -l accelerator=nvidia-tesla-gpu
   kubectl describe node <gpu-node-name>
   ```

### Application Deployment

1. **Create Namespace and Secrets**:
   ```bash
   # Create namespace
   kubectl apply -f k8s/namespace.yaml
   
   # Update secrets with base64 encoded values
   echo -n "your_password" | base64
   # Update k8s/secrets.yaml with encoded values
   kubectl apply -f k8s/secrets.yaml
   ```

2. **Deploy Storage**:
   ```bash
   # Create persistent volumes
   kubectl apply -f k8s/persistent-volumes.yaml
   
   # Verify PVCs are bound
   kubectl get pvc -n video-ad-placement
   ```

3. **Deploy Configuration**:
   ```bash
   # Apply ConfigMaps
   kubectl apply -f k8s/configmap.yaml
   
   # Apply RBAC
   kubectl apply -f k8s/rbac.yaml
   ```

4. **Deploy Applications**:
   ```bash
   # Deploy all services
   kubectl apply -f k8s/deployments.yaml
   kubectl apply -f k8s/services.yaml
   
   # Deploy autoscaling
   kubectl apply -f k8s/hpa.yaml
   ```

5. **Verify Deployment**:
   ```bash
   # Check all pods are running
   kubectl get pods -n video-ad-placement
   
   # Check services
   kubectl get svc -n video-ad-placement
   
   # Check GPU allocation
   kubectl describe node <gpu-node> | grep nvidia.com/gpu
   ```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: video-ad-placement-ingress
  namespace: video-ad-placement
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/client-body-buffer-size: 100M
    nginx.ingress.kubernetes.io/proxy-body-size: 100M
spec:
  tls:
    - hosts:
        - api.yourdomain.com
      secretName: video-ad-placement-tls
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: nginx
                port:
                  number: 80
```

## Production Scaling

### Horizontal Scaling

1. **Auto-scaling Configuration**:
   ```bash
   # Monitor HPA status
   kubectl get hpa -n video-ad-placement
   
   # Scale manually if needed
   kubectl scale deployment video-ad-placement-app --replicas=5 -n video-ad-placement
   ```

2. **GPU Node Scaling**:
   ```bash
   # Label new GPU nodes
   kubectl label nodes <new-gpu-node> accelerator=nvidia-tesla-gpu
   
   # Verify GPU resources
   kubectl describe node <new-gpu-node> | grep nvidia.com/gpu
   ```

### Vertical Scaling

Update resource requests in deployments:
```yaml
resources:
  limits:
    memory: 32Gi
    cpu: 8000m
    nvidia.com/gpu: 2
  requests:
    memory: 16Gi
    cpu: 4000m
    nvidia.com/gpu: 2
```

### Multi-Region Deployment

```bash
# Deploy to multiple clusters
for region in us-east us-west eu-central; do
  kubectl config use-context $region
  kubectl apply -f k8s/
done
```

## Monitoring & Observability

### Metrics Collection

1. **Prometheus Metrics**:
   - GPU utilization and memory
   - Processing queue lengths
   - API response times
   - Model inference latency

2. **Custom Metrics**:
   ```python
   # Add to application code
   from prometheus_client import Counter, Histogram, Gauge
   
   video_processing_total = Counter('video_processing_total', 'Total videos processed')
   gpu_memory_usage = Gauge('gpu_memory_usage_bytes', 'GPU memory usage')
   inference_duration = Histogram('model_inference_duration_seconds', 'Model inference time')
   ```

### Grafana Dashboards

Access Grafana at http://grafana.yourdomain.com with configured dashboards for:
- System metrics (CPU, Memory, GPU)
- Application metrics (API latency, queue depth)
- Business metrics (videos processed, error rates)

### Logging

Logs are centralized using Loki with the following structure:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "video-processor",
  "gpu_id": 0,
  "video_id": "abc123",
  "processing_time": 45.2,
  "message": "Video processing completed successfully"
}
```

### Alerts

Key alerts configured in Prometheus:
- High GPU memory usage (>90%)
- Processing queue backup (>100 jobs)
- API error rate (>5%)
- Model inference failures

## Security Considerations

### Network Security

1. **Service Mesh**: Use Istio for secure service-to-service communication
2. **Network Policies**: Restrict pod-to-pod communication
3. **TLS**: Enable TLS for all external communications

### Secret Management

```bash
# Use Kubernetes secrets or external secret management
kubectl create secret generic model-registry-creds \
  --from-literal=username=your_username \
  --from-literal=password=your_password \
  -n video-ad-placement
```

### RBAC

Minimal privilege access configured in `k8s/rbac.yaml`:
- Application pods: Read-only access to ConfigMaps
- Prometheus: Read access to metrics endpoints
- Workers: No cluster-level permissions

### Image Security

```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image video-ad-placement:latest
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   # Check GPU operator status
   kubectl get pods -n gpu-operator
   
   # Verify node labels
   kubectl get nodes --show-labels | grep accelerator
   
   # Check NVIDIA device plugin
   kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
   ```

2. **Out of GPU Memory**:
   ```bash
   # Check GPU usage
   kubectl exec -it <pod-name> -n video-ad-placement -- nvidia-smi
   
   # Reduce batch size or model size in configuration
   kubectl edit configmap video-ad-placement-config -n video-ad-placement
   ```

3. **Model Download Issues**:
   ```bash
   # Check init container logs
   kubectl logs <pod-name> -c model-downloader -n video-ad-placement
   
   # Manually download models
   kubectl exec -it <pod-name> -n video-ad-placement -- python3 /app/docker/scripts/download_models.py
   ```

4. **Performance Issues**:
   ```bash
   # Check resource utilization
   kubectl top pods -n video-ad-placement
   
   # Review metrics in Grafana
   # Check queue lengths in Redis
   kubectl exec -it redis-<pod-id> -n video-ad-placement -- redis-cli info
   ```

### Debug Commands

```bash
# Get all resources
kubectl get all -n video-ad-placement

# Check pod logs
kubectl logs -f deployment/video-ad-placement-app -n video-ad-placement

# Debug failing pods
kubectl describe pod <pod-name> -n video-ad-placement

# Port forward for debugging
kubectl port-forward svc/video-ad-placement-app 8000:8000 -n video-ad-placement

# Execute commands in pods
kubectl exec -it <pod-name> -n video-ad-placement -- /bin/bash
```

### Performance Tuning

1. **GPU Optimization**:
   - Use mixed precision training
   - Optimize batch sizes
   - Enable GPU memory pooling

2. **Model Optimization**:
   - Use TensorRT for inference optimization
   - Implement model quantization
   - Cache frequently used models

3. **Infrastructure**:
   - Use node affinity for GPU scheduling
   - Implement resource quotas
   - Configure proper CPU/memory ratios

## Health Checks

### Application Health

```bash
# Check application health
curl http://your-domain.com/health

# Check internal services
kubectl exec -it <pod-name> -n video-ad-placement -- curl http://localhost:8000/health
```

### System Health

```bash
# Check cluster health
kubectl get nodes
kubectl get pods --all-namespaces

# Check GPU health
kubectl exec -it <gpu-pod> -n video-ad-placement -- nvidia-smi

# Check storage
kubectl get pv,pvc -n video-ad-placement
```

This deployment guide provides a comprehensive foundation for deploying the AI-powered video advertisement placement system in production environments. Adjust configurations based on your specific infrastructure and requirements. 