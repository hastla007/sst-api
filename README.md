# Enterprise Speech-to-Text API v3.0

<div align="center">

**🚀 Production-Ready | 🎯 Enterprise-Grade | 🤖 AI-Powered**

An enterprise-grade Speech-to-Text API powered by OpenAI Whisper with advanced AI features including speaker diarization, translation, sentiment analysis, and real-time streaming.

[Features](#features) • [Quick Start](#quick-start) • [API Documentation](#api-documentation) • [Configuration](#configuration) • [Deployment](#deployment)

</div>

---

## ✨ What's New in v3.0

- **🎯 Speaker Diarization** - Identify who spoke when with AI-powered speaker separation
- **🌍 Auto Language Detection** - Automatic language identification for 99+ languages
- **📊 Confidence Scores** - Per-segment confidence metrics for quality assurance
- **💬 Custom Vocabulary** - Domain-specific prompts for improved accuracy
- **🔄 Translation** - Translate transcriptions to multiple languages
- **🧠 Sentiment Analysis** - AI-powered sentiment detection (positive/negative)
- **⚡ Real-Time Streaming** - WebSocket-based live transcription
- **☁️ Cloud Storage Integration** - Direct support for S3, GCS, and Azure Blob
- **📝 Enhanced Export Formats** - Added DOCX, PDF, and TXT export options
- **💾 Persistent Database** - PostgreSQL storage for transcription history
- **🔍 Full-Text Search** - Search across all transcriptions
- **🏢 Multi-Tenant Support** - Organizations, projects, and team management
- **💰 Usage Quotas & Billing** - Built-in quota tracking and cost management
- **🔔 Webhook Retry Logic** - Automatic retry with exponential backoff
- **🖥️ Web Dashboard** - Beautiful UI for easy interaction

---

## 🎯 Features

### 🔐 Enterprise Security
- **API Key Authentication** - Multi-key support with organization management
- **Rate Limiting** - Per-organization configurable limits
- **Input Validation** - Comprehensive file and URL validation
- **CORS Configuration** - Flexible cross-origin resource sharing
- **Webhook Validation** - SSRF protection for webhook URLs

### 🎤 Advanced Transcription
- **Auto Language Detection** - No need to specify language manually
- **Speaker Diarization** - Identify and separate different speakers (powered by pyannote.audio)
- **Confidence Scores** - Per-word and per-segment confidence metrics
- **Custom Vocabulary** - Provide context prompts for domain-specific terminology
- **99+ Languages** - Full support for ISO 639-1 language codes
- **Multiple Audio Formats** - MP3, WAV, M4A, OGG, WebM, FLAC, AAC

### 🌐 AI-Powered Features
- **Translation** - Translate transcriptions to multiple languages (powered by Marian NMT)
- **Sentiment Analysis** - Detect positive/negative sentiment (powered by DistilBERT)
- **Speaker Identification** - Who spoke when and for how long
- **Confidence Analysis** - Quality metrics for each transcription segment

### 📤 Export & Integration
- **Multiple Export Formats** - JSON, SRT, VTT, TXT, DOCX, PDF
- **Cloud Storage Support** - Direct integration with AWS S3, Google Cloud Storage, Azure Blob
- **Webhook Support** - Async callbacks with automatic retry logic (exponential backoff)
- **Batch Processing** - Process up to 10 files simultaneously
- **Real-Time Streaming** - WebSocket endpoint for live transcription

### 💾 Data Management
- **PostgreSQL Database** - Persistent storage for all transcriptions
- **Full-Text Search** - Search transcriptions by content or filename
- **Organization Management** - Multi-tenant with projects and teams
- **Usage Tracking** - Detailed analytics per organization
- **Quota Management** - Automatic quota enforcement with billing support

### ⚡ Performance & Scalability
- **Redis Caching** - Hash-based caching with configurable TTL
- **Async Task Queue** - Celery-based processing for long-running jobs
- **GPU Support** - Optional CUDA acceleration
- **Connection Pooling** - Optimized database connections
- **File Streaming** - Efficient handling of files up to 100MB

### 📊 Monitoring & Observability
- **Prometheus Metrics** - Comprehensive metrics for all operations
- **Request Tracking** - Correlation IDs across all services
- **Usage Analytics** - Detailed statistics and reporting
- **Health Checks** - Multi-service health monitoring
- **Grafana Dashboards** - Optional visualization (included in monitoring profile)

### 👨‍💻 Developer Experience
- **Web Dashboard** - Beautiful UI at `/dashboard` for easy testing
- **OpenAPI/Swagger** - Interactive documentation at `/docs`
- **ReDoc** - Alternative documentation at `/redoc`
- **Comprehensive Tests** - >80% code coverage
- **CI/CD Pipeline** - GitHub Actions automated testing
- **Docker Support** - Complete multi-service Docker Compose setup

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- (Optional) Python 3.11+ for local development
- (Optional) HuggingFace token for speaker diarization

### Using Docker Compose (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd sst-api
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
# Minimal config works out of the box!
```

3. **Start all services**
```bash
# Start core services (API + PostgreSQL + Redis + Celery)
docker-compose up -d

# Or start with monitoring (adds Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

4. **Verify the service is running**
```bash
curl http://localhost:3008/health
```

5. **Access the Web Dashboard**
Open your browser to: **http://localhost:3008/dashboard**

6. **Or use the Interactive API Documentation**
Open: **http://localhost:3008/docs**

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| STT API | 3008 | Main API endpoint |
| PostgreSQL | 5432 | Database for persistent storage |
| Redis | 6379 | Cache and message broker |
| Prometheus | 9090 | Metrics (optional, monitoring profile) |
| Grafana | 3000 | Dashboards (optional, monitoring profile) |

---

## 📖 API Documentation

### Core Endpoints

#### `POST /transcribe` - Advanced Transcription

Transcribe audio files with all enterprise features.

**Parameters:**
- `file` - Audio file upload (OR use `file_url`)
- `file_url` - URL to audio file (supports S3, GCS, Azure, HTTP/HTTPS)
- `language` - Language code (optional, auto-detected if not provided)
- `initial_prompt` - Custom vocabulary/context for better accuracy
- `export_format` - Output format: `json`, `srt`, `vtt`, `txt`, `docx`, `pdf`
- `enable_diarization` - Enable speaker diarization (boolean)
- `enable_translation` - Enable translation (boolean)
- `target_language` - Target language for translation
- `enable_sentiment` - Enable sentiment analysis (boolean)
- `return_confidence` - Return confidence scores (boolean, default: true)
- `webhook_url` - Webhook URL for async processing
- `use_cache` - Use cached results (boolean, default: true)
- `project_id` - Project ID for organization (optional)

**Example using curl:**
```bash
curl -X POST "http://localhost:3008/transcribe" \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true" \
  -F "enable_sentiment=true" \
  -F "export_format=json"
```

**Example response:**
```json
{
  "success": true,
  "transcription_id": "uuid-here",
  "text": "Full transcription text...",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, welcome to the meeting.",
      "speaker": "SPEAKER_00"
    }
  ],
  "confidence_scores": [
    {
      "segment_id": 0,
      "start": 0.0,
      "end": 2.5,
      "confidence": 0.95
    }
  ],
  "diarization": {
    "speakers": [...]
  },
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.92
  },
  "processing_time": 1.23,
  "audio_duration": 120.5
}
```

#### `POST /transcribe/batch` - Batch Processing

Process multiple audio files in a single request (up to 10 files).

#### `GET /ws/transcribe` - Real-Time Streaming

WebSocket endpoint for live transcription. Send audio chunks and receive real-time results.

#### `GET /transcriptions/search` - Full-Text Search

Search across all transcriptions.

**Parameters:**
- `query` - Search query
- `project_id` - Filter by project (optional)
- `limit` - Results limit (default: 10)
- `offset` - Results offset (default: 0)

#### `GET /transcriptions/{id}` - Get Transcription

Retrieve a specific transcription by ID.

### Organization Management

#### `POST /organizations` - Create Organization

Create a new organization with subscription tier.

**Parameters:**
- `name` - Organization name
- `email` - Organization email
- `subscription_tier` - Tier: `free`, `starter`, `professional`, `enterprise`

**Quotas by Tier:**
- **Free**: 1,000 minutes/month
- **Starter**: 10,000 minutes/month
- **Professional**: 50,000 minutes/month
- **Enterprise**: 1,000,000 minutes/month

#### `POST /projects` - Create Project

Create a new project within your organization.

#### `GET /usage/analytics` - Usage Analytics

Get detailed usage statistics for your organization.

### Utility Endpoints

- `GET /languages` - Get list of all 99 supported languages
- `GET /health` - Health check with feature status
- `GET /metrics` - Prometheus metrics
- `GET /analytics` - System analytics
- `GET /dashboard` - Web Dashboard UI

---

## ⚙️ Configuration

### Environment Variables

See `.env.example` for all configuration options. Key variables:

#### Core Configuration
```bash
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
USE_GPU=false       # Set to true for CUDA acceleration
```

#### Database
```bash
DATABASE_URL=postgresql://user:password@postgres:5432/stt_db
```

#### Authentication
```bash
API_KEYS=key1,key2,key3  # Comma-separated API keys
RATE_LIMIT_REQUESTS=10   # Requests per window
RATE_LIMIT_WINDOW=60     # Window in seconds
```

#### Cloud Storage
```bash
# AWS S3
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_BUCKET=stt-audio-files

# Google Cloud Storage
GCS_BUCKET=stt-audio-files
GCS_CREDENTIALS_FILE=/path/to/credentials.json

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
AZURE_CONTAINER=stt-audio-files
```

#### Speaker Diarization
```bash
HUGGINGFACE_TOKEN=your-hf-token  # Required for pyannote models
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
```

> **Note:** If you previously set `pyannote/speaker-diarization` (or `-3.0`),
> the app now upgrades that value automatically to `pyannote/speaker-diarization-3.1`
> to avoid loading the deprecated pipeline that still depends on `speechbrain`.

Get your token from: https://huggingface.co/settings/tokens
Accept model licenses:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/embedder-3.0

---

## 🏗️ Architecture

```
┌─────────────┐
│ Web Dashboard│
│   (React)   │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│         FastAPI Application (v3.0)          │
│  ┌────────────────────────────────────────┐ │
│  │  Core Features                         │ │
│  │  • Transcription (Whisper)            │ │
│  │  • Speaker Diarization (pyannote)     │ │
│  │  • Translation (Marian NMT)           │ │
│  │  • Sentiment Analysis (DistilBERT)    │ │
│  └────────────────────────────────────────┘ │
└──┬────────┬────────┬────────┬──────────┬───┘
   │        │        │        │          │
┌──▼──┐ ┌──▼──┐ ┌───▼───┐ ┌─▼────┐ ┌───▼────┐
│Redis│ │Celery│ │Postgres│ │S3/GCS│ │Webhooks│
│Cache│ │Queue │ │  DB    │ │Azure │ │ Retry  │
└─────┘ └──────┘ └────────┘ └──────┘ └────────┘
```

---

## 🧪 Testing

### Run Tests
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest test_app.py -v
```

### Manual Testing
```bash
python manual_test.py
```

---

## 📦 Deployment

### Docker Deployment (Production)

1. **Build the image**
```bash
docker build -t stt-api:v3.0 .
```

2. **Run with docker-compose**
```bash
docker-compose -f docker-compose.yml up -d
```

3. **Scale workers**
```bash
docker-compose up -d --scale celery-worker=3
```

### Kubernetes Deployment

Example Kubernetes manifests are available in the `k8s/` directory (if applicable).

### Environment-Specific Configuration

- **Development**: Use `docker-compose.yml` as-is
- **Staging**: Add `--profile monitoring` for observability
- **Production**: Use external managed services for PostgreSQL and Redis

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: Speaker diarization not working
**Solution**: Ensure you have a valid HuggingFace token and have accepted the pyannote model license.

**Issue**: Translation fails
**Solution**: Translation models are downloaded on first use. Ensure you have internet connectivity and sufficient disk space.

**Issue**: Database connection fails
**Solution**: Ensure PostgreSQL container is healthy: `docker-compose ps`

**Issue**: High memory usage
**Solution**: Use smaller Whisper model (`tiny` or `base`) or disable transformers features.

---

## 📊 Performance Benchmarks

| Model | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------------|-------------|----------|
| tiny | 32x realtime | 100x realtime | Good |
| base | 16x realtime | 70x realtime | Better |
| small | 6x realtime | 30x realtime | Great |
| medium | 2x realtime | 12x realtime | Excellent |
| large | 1x realtime | 6x realtime | Best |

*Benchmarks on Intel i7 CPU and NVIDIA RTX 3080 GPU*

---

## 🛣️ Roadmap

### Planned Features (v4.0)
- [ ] Video transcription support
- [ ] Custom model fine-tuning
- [ ] Advanced speaker identification (name assignment)
- [ ] Meeting summarization
- [ ] Keyword extraction
- [ ] Multi-language translation in single request
- [ ] GraphQL API
- [ ] gRPC support

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenAI Whisper** - Core transcription engine
- **pyannote.audio** - Speaker diarization
- **Hugging Face Transformers** - Translation and sentiment analysis
- **FastAPI** - Modern Python web framework
- **PostgreSQL, Redis, Celery** - Infrastructure components

---

## 📞 Support

For issues, questions, or feature requests:
- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Documentation**: http://localhost:3008/docs
- **Email**: support@your-domain.com

---

<div align="center">

**Made with ❤️ by the STT API Team**

⭐ Star us on GitHub if this project helped you!

</div>
