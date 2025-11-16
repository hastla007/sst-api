# Speech-to-Text API Service

A production-ready Speech-to-Text (STT) API service powered by OpenAI Whisper and FastAPI. This service provides accurate audio transcription with support for 99+ languages.

## Features

- **Multi-language Support**: 99+ languages supported (ISO 639-1 codes)
- **High Accuracy**: Powered by OpenAI Whisper model
- **File Size Validation**: Up to 100MB audio files
- **Multiple Audio Formats**: MP3, WAV, M4A, OGG, FLAC, AAC, WebM
- **RESTful API**: Clean, well-documented API endpoints
- **Docker Support**: Easy deployment with Docker Compose
- **Health Checks**: Built-in health monitoring
- **CORS Enabled**: Ready for web integrations
- **Auto-generated Docs**: Interactive API documentation at `/docs`

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd sst-api

# Start the service
docker-compose up -d

# Check service health
curl http://localhost:3008/health
```

The service will be available at `http://localhost:3008`

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /transcribe

Transcribe an audio file to text.

**Parameters:**
- `file` (required): Audio file (multipart/form-data)
- `language` (optional): Language code (e.g., 'en', 'es', 'de')

**Example using cURL:**

```bash
# Transcribe with auto-detection
curl -X POST http://localhost:3008/transcribe \
  -F "file=@audio.mp3"

# Transcribe with specific language
curl -X POST http://localhost:3008/transcribe \
  -F "file=@audio.mp3" \
  -F "language=en"
```

**Example using Python:**

```python
import requests

url = "http://localhost:3008/transcribe"
files = {"file": open("audio.mp3", "rb")}
data = {"language": "en"}  # Optional

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Transcription: {result['text']}")
print(f"Language: {result['language']}")
```

**Response:**

```json
{
  "success": true,
  "text": "This is the transcribed text from your audio file.",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "This is the transcribed text"
    }
  ]
}
```

### GET /languages

Get list of all supported language codes.

```bash
curl http://localhost:3008/languages
```

**Response:**

```json
{
  "supported_languages": ["af", "ar", "as", "az", ...],
  "count": 99,
  "note": "Language codes follow ISO 639-1 standard"
}
```

### GET /health

Check service health status.

```bash
curl http://localhost:3008/health
```

**Response:**

```json
{
  "status": "healthy",
  "model": "base",
  "service": "STT API",
  "max_file_size_mb": 100.0,
  "supported_languages_count": 99
}
```

### GET /

Get API information and available endpoints.

```bash
curl http://localhost:3008/
```

### GET /docs

Interactive API documentation (Swagger UI) - visit in your browser:
```
http://localhost:3008/docs
```

## Configuration

### Environment Variables

Configure the service using environment variables in `docker-compose.yml`:

```yaml
environment:
  - WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
```

**Available Models:**

| Model  | Size | Speed | Accuracy |
|--------|------|-------|----------|
| tiny   | 39M  | Fast  | Good     |
| base   | 74M  | Fast  | Better   |
| small  | 244M | Med   | Great    |
| medium | 769M | Slow  | Excellent|
| large  | 1550M| Slowest| Best    |

### Port Configuration

Change the exposed port in `docker-compose.yml`:

```yaml
ports:
  - "3008:8000"  # Change 3008 to your preferred port
```

### Resource Limits

Adjust memory limits based on your model choice:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

## Supported Audio Formats

- MP3 (audio/mpeg)
- WAV (audio/wav, audio/wave)
- M4A (audio/m4a, audio/mp4)
- OGG (audio/ogg)
- WebM (audio/webm)
- FLAC (audio/flac)
- AAC (audio/aac)

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid language code, empty file)
- `413` - Payload too large (file > 100MB)
- `422` - Validation error
- `500` - Server error

**Example Error Response:**

```json
{
  "detail": "Unsupported language code: xyz. Use /languages endpoint to see supported languages."
}
```

## Integration Examples

### n8n Workflow

1. Add HTTP Request node
2. Set Method to POST
3. URL: `http://stt-service:8000/transcribe`
4. Body: Form-Data
5. Add file field and language (optional)

### Python Script

```python
import requests
import json

def transcribe_audio(file_path, language=None):
    url = "http://localhost:3008/transcribe"

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'language': language} if language else {}

        response = requests.post(url, files=files, data=data)
        response.raise_for_status()

        return response.json()

# Usage
result = transcribe_audio('meeting.mp3', language='en')
print(result['text'])
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribeAudio(filePath, language = null) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));

  if (language) {
    form.append('language', language);
  }

  const response = await axios.post(
    'http://localhost:3008/transcribe',
    form,
    { headers: form.getHeaders() }
  );

  return response.data;
}

// Usage
transcribeAudio('audio.mp3', 'en')
  .then(result => console.log(result.text));
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest httpx

# Run tests
pytest
```

### Viewing Logs

```bash
# Docker logs
docker-compose logs -f stt-service

# Follow logs
docker logs -f stt-service
```

## Troubleshooting

### Service won't start

1. Check if port 3008 is available: `lsof -i :3008`
2. Check Docker logs: `docker-compose logs stt-service`
3. Ensure sufficient disk space for model download

### Transcription fails

1. Verify audio file format is supported
2. Check file size (must be < 100MB)
3. Try with a different Whisper model
4. Check service logs for detailed error messages

### Out of memory errors

1. Reduce Whisper model size (use `tiny` or `base`)
2. Increase Docker memory limits in `docker-compose.yml`
3. Process smaller audio files

## Performance Tips

1. **Model Selection**: Use `base` model for good balance of speed/accuracy
2. **Caching**: Models are cached in `./models` volume to avoid re-downloads
3. **File Size**: Smaller files process faster; consider splitting large files
4. **Language Hint**: Providing language code improves accuracy and speed

## Production Deployment

### Security Considerations

1. **Add Authentication**: Implement API key or JWT authentication
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **HTTPS**: Use reverse proxy (nginx/traefik) with SSL
4. **File Validation**: Additional validation for production use

### Recommended nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name stt.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3008;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 100M;
    }
}
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
- Open an issue on GitHub
- Check `/docs` endpoint for API documentation
- Review logs for debugging information
