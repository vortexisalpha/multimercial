# Video Advertisement Placement Service

An AI-powered video advertisement placement service that automatically integrates advertisements into video content using advanced computer vision and machine learning techniques.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with Python 3.11)
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd multimercial
```

2. **Install dependencies**
```bash
pip install fastapi uvicorn pydantic hydra-core omegaconf
pip install "python-jose[cryptography]" "passlib[bcrypt]" PyJWT
pip install psutil aiofiles python-multipart httpx
pip install transformers opencv-python
```

3. **Create required directories**
```bash
mkdir -p logs uploads/advertisements
```

### Running the Service

**Start the API server:**
```bash
python run_server.py
```

The server will start on `http://localhost:8000` with the following endpoints available:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health
- **Service Info**: http://localhost:8000/api/v1/info

## 📋 API Usage

### 1. Get API Keys (Development)
```bash
curl http://localhost:8000/api/v1/test-auth
```

Response:
```json
{
    "admin_api_key": "admin_adde73956e80fa10e12dbd47",
    "user_api_key": "user_b5db4f67df2070f1f40b5e6e",
    "usage": "Add 'X-API-Key: <key>' header to authenticate requests"
}
```

### 2. Check System Health
```bash
curl http://localhost:8000/api/v1/health
```

### 3. Process a Video
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: user_b5db4f67df2070f1f40b5e6e" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "advertisement_config": {
      "ad_type": "image",
      "ad_url": "https://example.com/ad.jpg",
      "width": 1.0,
      "height": 0.6
    },
    "placement_config": {
      "strategy": "automatic",
      "quality_threshold": 0.7
    }
  }' \
  http://localhost:8000/api/v1/process-video
```

Response:
```json
{
    "job_id": "job_1749430240_api_user",
    "status": "queued",
    "message": "Video processing started",
    "estimated_completion_time": 1749430540.9882061
}
```

### 4. Check Processing Status
```bash
curl -H "X-API-Key: user_b5db4f67df2070f1f40b5e6e" \
  http://localhost:8000/api/v1/status/{job_id}
```

### 5. Get System Metrics (Admin Only)
```bash
curl -H "X-API-Key: admin_adde73956e80fa10e12dbd47" \
  http://localhost:8000/api/v1/metrics
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_api.py
```

This will test:
- ✅ API connectivity
- ✅ Authentication system
- ✅ Health monitoring
- ✅ Rate limiting
- ✅ Video processing endpoints
- ✅ System metrics

## 🏗️ Architecture

### Core Components

1. **FastAPI Server** - REST API with async support
2. **Authentication** - JWT + API key authentication
3. **Database** - SQLite for development (configurable)
4. **Rate Limiting** - Memory-based request throttling
5. **WebSocket** - Real-time job progress updates
6. **Monitoring** - System health and metrics collection

### Current Status

- ✅ **API Infrastructure**: Fully functional REST API
- ✅ **Authentication**: Role-based access control
- ✅ **Database**: SQLite storage with health monitoring
- ✅ **Monitoring**: Real-time system metrics
- ⚠️ **ML Pipeline**: Currently mocked (see "Full ML Setup" below)

## 🔧 Configuration

The service uses a hierarchical configuration system. Key settings:

- **Environment**: Development/Production modes
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Security**: JWT secrets and API key management
- **Processing**: Quality levels and resource limits

Configuration is loaded from `conf/config.yaml` or can be set via environment variables.

## 📊 API Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/v1/info` | GET | Service information | No |
| `/api/v1/health` | GET | System health check | No |
| `/api/v1/test-auth` | GET | Get test API keys | No |
| `/api/v1/process-video` | POST | Submit video for processing | Yes |
| `/api/v1/status/{job_id}` | GET | Check processing status | Yes |
| `/api/v1/metrics` | GET | System metrics | Admin |
| `/api/v1/upload-advertisement` | POST | Upload ad content | Yes |
| `/api/v1/batch-process` | POST | Batch process videos | Yes |
| `/ws/{job_id}` | WebSocket | Real-time progress | Yes |

## 🔄 Development Workflow

1. **Start the server**: `python run_server.py`
2. **Get API keys**: Visit http://localhost:8000/api/v1/test-auth
3. **Test endpoints**: Use curl or the interactive docs at http://localhost:8000/docs
4. **Monitor health**: Check http://localhost:8000/api/v1/health
5. **View logs**: Check the `logs/` directory

## 🚀 Full ML Pipeline Setup (Optional)

To enable the complete video processing pipeline with AI features:

### 1. Install ML Dependencies
```bash
# Deep learning frameworks
pip install torch torchvision torchaudio
pip install diffusers transformers

# Computer vision
pip install ultralytics opencv-python-headless

# Optional: GPU acceleration
pip install torch-tensorrt  # For NVIDIA GPUs
```

### 2. Enable Pipeline
In `src/video_ad_placement/api/main.py`, uncomment the pipeline initialization:
```python
# Uncomment these lines:
from ..pipeline import VideoAdPlacementPipeline
# ... pipeline initialization code
```

### 3. Configure GPU (Optional)
Update configuration to enable GPU processing:
```yaml
video_processing:
  use_gpu: true
  gpu_devices: [0]
```

## 🐳 Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_server.py"]
```

Build and run:
```bash
docker build -t video-ad-placement .
docker run -p 8000:8000 video-ad-placement
```

## 🔒 Production Deployment

For production deployment:

1. **Set environment variables**:
```bash
export ENVIRONMENT=production
export JWT_SECRET_KEY=your-secure-secret-key
export DATABASE_URL=postgresql://user:pass@host:port/db
```

2. **Use a production WSGI server**:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.video_ad_placement.api.main:app
```

3. **Set up reverse proxy** (nginx/Apache)
4. **Configure SSL certificates**
5. **Set up monitoring and logging**

## 📈 Monitoring & Metrics

The service provides comprehensive monitoring:

- **Health Checks**: Component-level health monitoring
- **Metrics**: Request rates, processing times, resource usage
- **Logging**: Structured JSON logging
- **WebSocket**: Real-time job progress updates

Access metrics at: http://localhost:8000/api/v1/metrics (admin required)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_api.py`
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues

1. **Server won't start**:
   - Check Python version (3.8+ required)
   - Verify all dependencies are installed
   - Check port 8000 availability

2. **Authentication errors**:
   - Get fresh API keys from `/api/v1/test-auth`
   - Ensure `X-API-Key` header is included

3. **Import errors**:
   - Run from project root directory
   - Verify Python path includes `src/`

### Getting Help

- Check server logs in `logs/` directory
- Visit http://localhost:8000/docs for interactive API documentation
- Check health status at http://localhost:8000/api/v1/health

---

**🎉 You now have a fully functional Video Advertisement Placement Service!**

The API is ready to accept video processing requests and can be extended with the full ML pipeline when needed.

## 🎬 Video Processing Demo

### Real Advertisement Placement

The service includes a **working video advertisement placement pipeline** that actually processes videos and overlays advertisements:

**Quick Demo:**
```bash
python quick_demo.py
```

**Direct Processing:**
```bash
python process_video_with_ad.py
```

**Features:**
- ✅ **Real video processing** with OpenCV
- ✅ **Advertisement overlay** with styling and transparency
- ✅ **Multiple placement strategies** (top_left, top_right, bottom_left, bottom_right, center)
- ✅ **Configurable timing** (start time, duration)
- ✅ **Quality control** with borders and shadows
- ✅ **Progress tracking** and detailed reporting

**Example Configuration:**
```python
placement_config = {
    "strategy": "bottom_right",     # Position on screen
    "opacity": 0.85,               # Transparency (0.0-1.0)
    "scale_factor": 0.3,           # Size relative to video width
    "start_time": 3.0,             # Start after 3 seconds
    "duration": 15.0               # Show for 15 seconds
}
```

**Output:**
- Processed video with advertisement overlay
- Detailed processing report (JSON)
- Performance metrics and statistics
