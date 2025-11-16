# Enhanced Speech-to-Text API

A production-ready, enterprise-grade Speech-to-Text API powered by OpenAI Whisper with advanced features including authentication, rate limiting, caching, async processing, and comprehensive monitoring.

## Features

### Security
- **API Key Authentication** - Secure your API with custom API keys
- **Rate Limiting** - Prevent abuse with configurable request throttling
- **Input Validation** - Comprehensive file validation and sanitization
- **CORS Configuration** - Configurable cross-origin resource sharing

### Core Features
- **Batch Processing** - Process up to 10 audio files in a single request
- **Webhook Support** - Async callbacks for long-running transcriptions
- **Multiple Export Formats** - JSON, SRT, and VTT subtitle formats
- **99 Languages** - Support for 99 languages following ISO 639-1 standard
- **Multiple Audio Formats** - MP3, WAV, M4A, OGG, WebM, FLAC, AAC

### Performance
- **Redis Caching** - Cache results for identical files to reduce processing time
- **Async Task Queue** - Celery-based queue for long-running transcriptions
- **GPU Support** - Optional CUDA acceleration for faster processing
- **File Streaming** - Efficient handling of large audio files (up to 100MB)

### Monitoring & Analytics
- **Prometheus Metrics** - Detailed metrics for monitoring and alerting
- **Request Tracking** - Correlation IDs for tracing requests across services
- **Usage Analytics** - Track API usage, success rates, and performance
- **Health Checks** - Comprehensive health endpoints for orchestration

### Developer Experience
- **OpenAPI/Swagger** - Interactive API documentation at `/docs`
- **Comprehensive Tests** - Unit and integration tests with >80% coverage
- **CI/CD Pipeline** - GitHub Actions workflow for automated testing and deployment
- **Docker Support** - Complete Docker Compose setup with all services

## Quick Start

### Prerequisites
- Docker and Docker Compose
- (Optional) Python 3.11+ for local development

### Using Docker Compose (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd sst-api
```

2. **Configure environment variables** (optional)
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start all services**
```bash
# Start core services (API + Redis + Celery)
docker-compose up -d

# Or start with monitoring (adds Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

4. **Verify the service is running**
```bash
curl http://localhost:3008/health
```

5. **Access the interactive documentation**
Open your browser to: http://localhost:3008/docs

### Local Development

1. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Start Redis** (required for caching and async features)
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

3. **Run the API server**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. **Run Celery worker** (in another terminal)
```bash
celery -A app.celery_app worker --loglevel=info
```

## API Endpoints

### Transcription

#### POST /transcribe
Transcribe a single audio file to text.

**Parameters:**
- `file` (required): Audio file (mp3, wav, m4a, etc.) - Max 100MB
- `language` (optional): Language code (e.g., 'en', 'es', 'de')
- `export_format` (optional): Output format - `json`, `srt`, `vtt` (default: `json`)
- `webhook_url` (optional): Webhook URL for async processing
- `use_cache` (optional): Enable/disable caching (default: `true`)

**Headers:**
- `X-API-Key`: Your API key (required if authentication is enabled)
- `X-Correlation-ID`: Optional correlation ID for request tracking

**Example with cURL:**
```bash
curl -X POST http://localhost:3008/transcribe \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "export_format=json"
```

**Example with Python:**
```python
import requests

url = "http://localhost:3008/transcribe"
headers = {"X-API-Key": "your-api-key"}
files = {"file": open("audio.mp3", "rb")}
data = {"language": "en", "export_format": "srt"}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.text)
```

**Response (JSON format):**
```json
{
  "success": true,
  "text": "Full transcription text here",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a sample transcription."
    }
  ],
  "correlation_id": "abc-123-def",
  "cached": false,
  "processing_time": 1.234
}
```

#### POST /transcribe/batch
Process multiple audio files in a single request.

**Parameters:**
- `files` (required): List of audio files (max 10 files)
- `language` (optional): Language code for all files
- `webhook_url` (optional): Webhook URL for results

**Example:**
```bash
curl -X POST http://localhost:3008/transcribe/batch \
  -H "X-API-Key: your-api-key" \
  -F "files=@audio1.mp3" \
  -F "files=@audio2.wav" \
  -F "language=en"
```

#### GET /task/{task_id}
Get the status of an async transcription task.

**Example:**
```bash
curl http://localhost:3008/task/abc-123-def \
  -H "X-API-Key: your-api-key"
```

### Information Endpoints

#### GET /languages
Get list of supported language codes.

```bash
curl http://localhost:3008/languages
```

#### GET /health
Health check endpoint with system status.

```bash
curl http://localhost:3008/health
```

#### GET /analytics
Get usage analytics and statistics (requires authentication).

```bash
curl http://localhost:3008/analytics \
  -H "X-API-Key: your-api-key"
```

#### GET /metrics
Prometheus metrics endpoint for monitoring.

```bash
curl http://localhost:3008/metrics
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model size: tiny, base, small, medium, large |
| `API_KEYS` | - | Comma-separated API keys (empty = no auth) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `RATE_LIMIT_REQUESTS` | `10` | Max requests per time window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
| `REDIS_HOST` | `redis` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `CACHE_TTL` | `3600` | Cache time-to-live in seconds |
| `USE_GPU` | `false` | Enable GPU acceleration (requires CUDA) |

### Whisper Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 39M | Fastest | Good | Quick transcriptions, testing |
| base | 74M | Fast | Better | General purpose, default |
| small | 244M | Medium | Great | Better accuracy needed |
| medium | 769M | Slow | Excellent | High accuracy required |
| large | 1550M | Slowest | Best | Maximum accuracy |

### Authentication

Enable authentication by setting API keys:

```bash
# In .env or docker-compose.yml
API_KEYS=key1,key2,key3
```

Then include the key in requests:
```bash
curl -H "X-API-Key: key1" http://localhost:3008/transcribe
```

### Rate Limiting

Configure rate limits to prevent abuse:

```bash
RATE_LIMIT_REQUESTS=100  # 100 requests
RATE_LIMIT_WINDOW=60     # per 60 seconds
```

Rate limiting is per API key (if auth enabled) or per IP address.

## Export Formats

### JSON (Default)
Standard JSON response with full transcription details, segments, and metadata.

### SRT (SubRip)
Subtitle format compatible with most video players:
```
1
00:00:00,000 --> 00:00:02,500
Hello, this is a sample transcription.

2
00:00:02,500 --> 00:00:05,000
This is the second segment.
```

### VTT (WebVTT)
Web Video Text Tracks format:
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello, this is a sample transcription.

00:00:02.500 --> 00:00:05.000
This is the second segment.
```

## Webhooks

For long-running transcriptions, use webhooks for async processing:

```bash
curl -X POST http://localhost:3008/transcribe \
  -H "X-API-Key: your-api-key" \
  -F "file=@large-audio.mp3" \
  -F "webhook_url=https://your-server.com/webhook"
```

The webhook will receive a POST request with:
```json
{
  "correlation_id": "abc-123-def",
  "status": "completed",
  "result": {
    "text": "Transcription text",
    "language": "en",
    "segments": [...]
  }
}
```

## Monitoring

### Prometheus Metrics

Available metrics at `/metrics`:
- `stt_requests_total` - Total requests by endpoint and status
- `stt_request_duration_seconds` - Request duration histogram
- `stt_transcription_duration_seconds` - Transcription duration histogram
- `stt_active_requests` - Current active requests
- `stt_cache_hits_total` - Total cache hits
- `stt_cache_misses_total` - Total cache misses

### Grafana Dashboard

Access Grafana at http://localhost:3000 (when using monitoring profile):
- Username: `admin`
- Password: `admin`

### Usage Analytics

Get usage statistics via the `/analytics` endpoint:
```bash
curl http://localhost:3008/analytics -H "X-API-Key: your-api-key"
```

Returns:
```json
{
  "total_requests": 1000,
  "successful_requests": 950,
  "failed_requests": 50,
  "success_rate": 0.95,
  "average_processing_time": 2.34,
  "cache_statistics": {
    "hits": 200,
    "misses": 750,
    "hit_rate": 0.21
  }
}
```

## Testing

### Run Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest test_app.py -v

# Run with coverage
pytest test_app.py -v --cov=app --cov-report=html

# Run specific test class
pytest test_app.py::TestTranscription -v
```

### Test Coverage

Current test coverage: >80%
- Health and information endpoints
- Authentication and authorization
- Rate limiting
- Transcription (single and batch)
- Export formats (JSON, SRT, VTT)
- Utility functions
- Error handling

## CI/CD

GitHub Actions workflow automatically:
- Runs tests on every push
- Checks code quality (black, flake8)
- Builds Docker images
- Runs security scans
- Deploys on main branch (configure deployment target)

## GPU Support

Enable GPU acceleration for faster transcription:

1. **Install NVIDIA drivers and CUDA**

2. **Update docker-compose.yml:**
```yaml
stt-service:
  environment:
    - USE_GPU=true
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. **Restart services:**
```bash
docker-compose up -d
```

## Troubleshooting

### Common Issues

**Issue: Model download takes too long**
- Solution: Models are cached in `./models` volume. First run will download the model.

**Issue: Out of memory errors**
- Solution: Use a smaller model (tiny or base) or increase Docker memory limits.

**Issue: Redis connection failed**
- Solution: Ensure Redis service is running. Check `docker-compose ps`.

**Issue: Rate limit exceeded**
- Solution: Adjust `RATE_LIMIT_REQUESTS` or wait for the rate limit window to reset.

**Issue: Authentication errors**
- Solution: Ensure `X-API-Key` header is set correctly. Check `API_KEYS` environment variable.

### Logs

View service logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f stt-service
docker-compose logs -f celery-worker
docker-compose logs -f redis
```

## Production Deployment

### Recommended Settings

```yaml
# docker-compose.yml
environment:
  - WHISPER_MODEL=small  # Balance of speed and accuracy
  - API_KEYS=${API_KEYS}  # Set via environment
  - CORS_ORIGINS=https://your-domain.com
  - RATE_LIMIT_REQUESTS=100
  - RATE_LIMIT_WINDOW=60
  - CACHE_TTL=7200  # 2 hours
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3008;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for large file uploads
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;

        # Increase max body size
        client_max_body_size 100M;
    }
}
```

### Scaling

Scale Celery workers:
```bash
docker-compose up -d --scale celery-worker=4
```

### Backup

Important data to backup:
- Redis data: `./redis-data` volume (for cache persistence)
- Model cache: `./models` volume (to avoid re-downloading)
- Analytics data: Export via `/analytics` endpoint

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      FastAPI Application        │
│  ┌──────────────────────────┐   │
│  │ Authentication Middleware │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ Rate Limiting Middleware  │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │  Correlation ID Tracking  │   │
│  └──────────────────────────┘   │
└────────┬──────────┬─────────────┘
         │          │
         ▼          ▼
    ┌────────┐  ┌────────┐
    │ Redis  │  │ Celery │
    │ Cache  │  │ Worker │
    └────────┘  └────────┘
         │          │
         ▼          ▼
    ┌──────────────────┐
    │ Whisper Model    │
    │ (CPU/GPU)        │
    └──────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest test_app.py -v`
5. Run code quality checks: `black . && flake8 .`
6. Commit changes: `git commit -am 'Add your feature'`
7. Push to the branch: `git push origin feature/your-feature`
8. Create a Pull Request

## License

This project is provided as-is for educational and commercial use.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: [API Docs](http://localhost:3008/docs)

## Changelog

### Version 2.0.0 (Latest)
- Added API key authentication
- Added rate limiting
- Added batch processing
- Added webhook support
- Added SRT and VTT export formats
- Added Redis caching
- Added Celery async task queue
- Added GPU support
- Added Prometheus metrics
- Added correlation ID tracking
- Added usage analytics
- Added comprehensive tests
- Added CI/CD pipeline
- Enhanced OpenAPI documentation

### Version 1.0.0
- Initial release
- Basic transcription functionality
- Multi-language support
- Docker support
