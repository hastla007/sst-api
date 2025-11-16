from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import whisper
import tempfile
import os
import logging
import hashlib
import json
import uuid
import time
from collections import defaultdict, deque
from functools import wraps
import asyncio
import aiohttp
from urllib.parse import urlparse

# Third-party imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Caching and advanced features will be disabled.")

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logging.warning("Celery not available. Async queue features will be disabled.")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Metrics will be disabled.")

# Setup logging with custom filter for correlation_id
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s'
)
logger = logging.getLogger(__name__)
logger.addFilter(CorrelationIdFilter())

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
SUPPORTED_LANGUAGES = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
    "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
    "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
    "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
    "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
    "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
    "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
    "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
}
SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/ogg", "audio/webm",
    "audio/flac", "audio/aac"
}

# Security Configuration
API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()
ENABLE_AUTH = len(API_KEYS) > 0
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default

# Celery Configuration
CELERY_BROKER = os.getenv("CELERY_BROKER", f"redis://{REDIS_HOST}:{REDIS_PORT}/1")
CELERY_BACKEND = os.getenv("CELERY_BACKEND", f"redis://{REDIS_HOST}:{REDIS_PORT}/2")

# GPU Configuration
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
DEVICE = "cuda" if USE_GPU else "cpu"

# Initialize Redis client
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5
        )
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        redis_client = None

# Initialize Celery
celery_app = None
if CELERY_AVAILABLE and redis_client:
    celery_app = Celery(
        "stt_tasks",
        broker=CELERY_BROKER,
        backend=CELERY_BACKEND
    )
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
    )

# Prometheus Metrics
if PROMETHEUS_AVAILABLE:
    request_counter = Counter('stt_requests_total', 'Total requests', ['endpoint', 'status'])
    request_duration = Histogram('stt_request_duration_seconds', 'Request duration', ['endpoint'])
    transcription_duration = Histogram('stt_transcription_duration_seconds', 'Transcription duration')
    active_requests = Gauge('stt_active_requests', 'Number of active requests')
    cache_hits = Counter('stt_cache_hits_total', 'Total cache hits')
    cache_misses = Counter('stt_cache_misses_total', 'Total cache misses')

# In-memory rate limiting (fallback when Redis is not available)
rate_limit_storage = defaultdict(lambda: deque())

# Usage analytics storage
usage_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_duration": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
}

app = FastAPI(
    title="Enhanced STT Service API",
    version="2.0.0",
    description="""
    ## Advanced Speech-to-Text API with Enterprise Features

    ### Features:
    - 🔐 API Key Authentication
    - ⚡ Rate Limiting
    - 📦 Batch Processing
    - 🔔 Webhook Support
    - 📝 SRT/VTT Subtitle Export
    - 🚀 Async Queue Processing
    - 💾 Redis Caching
    - 🎯 GPU Support
    - 📊 Prometheus Metrics
    - 🔍 Request Tracking
    - 📈 Usage Analytics

    ### Authentication:
    Pass your API key in the `X-API-Key` header.

    ### Rate Limits:
    - Default: 10 requests per 60 seconds
    - Configurable via environment variables
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
logger.info(f"Loading Whisper model: {MODEL_SIZE} on device: {DEVICE}")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
logger.info("Model loaded successfully")


# Middleware for correlation ID
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id

    # Add correlation ID to logger context
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.correlation_id = getattr(record, 'correlation_id', correlation_id)
        return record

    logging.setLogRecordFactory(record_factory)

    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id

    logging.setLogRecordFactory(old_factory)
    return response


# Authentication dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not ENABLE_AUTH:
        return True

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please provide X-API-Key header."
        )

    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return x_api_key


# Rate limiting dependency
async def rate_limiter(request: Request, api_key: str = Depends(verify_api_key)):
    identifier = api_key if ENABLE_AUTH else request.client.host
    current_time = time.time()

    if redis_client:
        # Use Redis for distributed rate limiting
        key = f"rate_limit:{identifier}"
        try:
            pipe = redis_client.pipeline()
            pipe.zadd(key, {str(current_time): current_time})
            pipe.zremrangebyscore(key, 0, current_time - RATE_LIMIT_WINDOW)
            pipe.zcard(key)
            pipe.expire(key, RATE_LIMIT_WINDOW)
            results = pipe.execute()
            request_count = results[2]
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            request_count = 0
    else:
        # Fallback to in-memory rate limiting
        requests = rate_limit_storage[identifier]
        # Remove old requests
        while requests and requests[0] < current_time - RATE_LIMIT_WINDOW:
            requests.popleft()
        requests.append(current_time)
        request_count = len(requests)

    if request_count > RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )

    return True


# Utility functions
def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()


def validate_webhook_url(url: str) -> bool:
    """Validate webhook URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False


def get_cached_result(cache_key: str) -> Optional[Dict]:
    """Get cached transcription result"""
    if not redis_client:
        return None

    try:
        cached = redis_client.get(f"transcription:{cache_key}")
        if cached:
            if PROMETHEUS_AVAILABLE:
                cache_hits.inc()
            usage_stats["cache_hits"] += 1
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")

    if PROMETHEUS_AVAILABLE:
        cache_misses.inc()
    usage_stats["cache_misses"] += 1
    return None


def set_cached_result(cache_key: str, result: Dict):
    """Cache transcription result"""
    if not redis_client:
        return

    try:
        redis_client.setex(
            f"transcription:{cache_key}",
            CACHE_TTL,
            json.dumps(result)
        )
    except Exception as e:
        logger.error(f"Cache storage error: {e}")


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT/VTT timestamp"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: List[Dict]) -> str:
    """Convert segments to SRT format"""
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_content)


def segments_to_vtt(segments: List[Dict]) -> str:
    """Convert segments to WebVTT format"""
    vtt_content = ["WEBVTT\n"]
    for segment in segments:
        start = format_timestamp(segment['start']).replace(',', '.')
        end = format_timestamp(segment['end']).replace(',', '.')
        text = segment['text'].strip()
        vtt_content.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(vtt_content)


async def send_webhook(webhook_url: str, data: Dict):
    """Send webhook notification"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status >= 400:
                    logger.error(f"Webhook failed with status {response.status}")
                else:
                    logger.info(f"Webhook sent successfully to {webhook_url}")
    except Exception as e:
        logger.error(f"Webhook error: {e}")


def transcribe_file(file_path: str, language: Optional[str] = None) -> Dict:
    """Core transcription function"""
    transcribe_options = {}
    if language:
        transcribe_options["language"] = language

    start_time = time.time()
    result = model.transcribe(file_path, **transcribe_options)
    duration = time.time() - start_time

    if PROMETHEUS_AVAILABLE:
        transcription_duration.observe(duration)

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": result["segments"],
        "duration": duration
    }


# Celery task for async processing
if celery_app:
    @celery_app.task(name="transcribe_async")
    def transcribe_async_task(file_content: bytes, filename: str, language: Optional[str], webhook_url: Optional[str], correlation_id: str):
        """Async transcription task"""
        temp_file_path = None
        try:
            # Save file temporarily
            suffix = os.path.splitext(filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Transcribe
            result = transcribe_file(temp_file_path, language)

            # Clean up
            os.unlink(temp_file_path)

            # Send webhook if provided
            if webhook_url:
                webhook_data = {
                    "correlation_id": correlation_id,
                    "status": "completed",
                    "result": result
                }
                # Note: Celery tasks are synchronous, so we use requests for webhook
                import requests
                try:
                    requests.post(webhook_url, json=webhook_data, timeout=30)
                except Exception as e:
                    logger.error(f"Webhook error in async task: {e}")

            return result

        except Exception as e:
            logger.error(f"Async transcription error: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise


@app.post(
    "/transcribe",
    dependencies=[Depends(rate_limiter)],
    response_model=None,
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "text": "Hello, this is a sample transcription.",
                        "language": "en",
                        "segments": [
                            {"start": 0.0, "end": 2.5, "text": "Hello, this is a sample transcription."}
                        ],
                        "correlation_id": "abc-123-def",
                        "cached": False
                    }
                }
            }
        }
    }
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (mp3, wav, m4a, etc.) - Max 100MB"),
    language: Optional[str] = Form(None, description="Optional language code (e.g., 'en', 'es', 'de')"),
    export_format: Optional[str] = Form("json", description="Export format: json, srt, vtt"),
    webhook_url: Optional[str] = Form(None, description="Webhook URL for async processing"),
    use_cache: bool = Form(True, description="Use cached results if available"),
    api_key: str = Depends(verify_api_key)
):
    """
    Transcribe audio file to text with advanced options

    **Parameters:**
    - **file**: Audio file (mp3, wav, m4a, etc.) - Max 100MB
    - **language**: Optional language code (e.g., 'en', 'es', 'de')
    - **export_format**: Output format - json (default), srt, vtt
    - **webhook_url**: Optional webhook URL for async processing
    - **use_cache**: Enable/disable caching (default: true)

    **Returns:**
    - Transcribed text with segments and metadata
    - Or task ID if webhook is specified (async processing)
    """
    correlation_id = request.state.correlation_id
    temp_file_path = None
    start_time = time.time()

    try:
        if PROMETHEUS_AVAILABLE:
            active_requests.inc()

        usage_stats["total_requests"] += 1

        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")

        # Validate language code
        if language and language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language code: {language}. Use /languages endpoint to see supported languages."
            )

        # Validate export format
        if export_format not in ["json", "srt", "vtt"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid export_format. Must be one of: json, srt, vtt"
            )

        # Validate webhook URL if provided
        if webhook_url and not validate_webhook_url(webhook_url):
            raise HTTPException(
                status_code=400,
                detail="Invalid webhook URL. Must be a valid HTTP or HTTPS URL."
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Validate file content (basic magic number check)
        if content[:4] not in [b'ID3\x03', b'ID3\x04', b'RIFF', b'fLaC', b'OggS'] and content[:2] not in [b'\xff\xfb', b'\xff\xf3', b'\xff\xf2']:
            logger.warning(f"Suspicious file content, may not be a valid audio file")

        logger.info(f"File size: {file_size / (1024*1024):.2f}MB")

        # Calculate file hash for caching
        file_hash = calculate_file_hash(content)
        cache_key = f"{file_hash}:{language or 'auto'}"

        # Check cache
        cached_result = None
        if use_cache:
            cached_result = get_cached_result(cache_key)

        if cached_result:
            logger.info("Returning cached result")
            cached_result["correlation_id"] = correlation_id
            cached_result["cached"] = True

            if export_format == "srt":
                return PlainTextResponse(
                    content=segments_to_srt(cached_result["segments"]),
                    media_type="text/plain"
                )
            elif export_format == "vtt":
                return PlainTextResponse(
                    content=segments_to_vtt(cached_result["segments"]),
                    media_type="text/vtt"
                )

            return JSONResponse(content=cached_result)

        # Async processing with webhook
        if webhook_url and celery_app:
            task = transcribe_async_task.delay(
                content,
                file.filename,
                language,
                webhook_url,
                correlation_id
            )
            logger.info(f"Queued async task: {task.id}")
            return JSONResponse(content={
                "success": True,
                "task_id": task.id,
                "correlation_id": correlation_id,
                "status": "processing",
                "message": "Transcription queued. Results will be sent to webhook URL."
            })

        # Synchronous processing
        filename = file.filename or "audio.wav"
        suffix = os.path.splitext(filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Transcribing file: {temp_file_path}")

        # Transcribe
        result = transcribe_file(temp_file_path, language)

        # Clean up temp file
        os.unlink(temp_file_path)
        temp_file_path = None

        # Cache result
        if use_cache:
            set_cached_result(cache_key, result)

        logger.info("Transcription completed successfully")

        # Update stats
        usage_stats["successful_requests"] += 1
        request_duration_time = time.time() - start_time
        usage_stats["total_duration"] += request_duration_time

        if PROMETHEUS_AVAILABLE:
            request_counter.labels(endpoint='transcribe', status='success').inc()
            request_duration.labels(endpoint='transcribe').observe(request_duration_time)

        # Send webhook in background if specified
        if webhook_url:
            webhook_data = {
                "correlation_id": correlation_id,
                "status": "completed",
                "result": result
            }
            background_tasks.add_task(send_webhook, webhook_url, webhook_data)

        # Prepare response
        response_data = {
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"],
            "correlation_id": correlation_id,
            "cached": False,
            "processing_time": result["duration"]
        }

        # Export in requested format
        if export_format == "srt":
            return PlainTextResponse(
                content=segments_to_srt(result["segments"]),
                media_type="text/plain",
                headers={"X-Correlation-ID": correlation_id}
            )
        elif export_format == "vtt":
            return PlainTextResponse(
                content=segments_to_vtt(result["segments"]),
                media_type="text/vtt",
                headers={"X-Correlation-ID": correlation_id}
            )

        return JSONResponse(content=response_data)

    except HTTPException:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp file: {cleanup_error}")
        raise

    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            request_counter.labels(endpoint='transcribe', status='error').inc()

        usage_stats["failed_requests"] += 1
        logger.error(f"Transcription error: {str(e)}", exc_info=True)

        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp file: {cleanup_error}")

        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if PROMETHEUS_AVAILABLE:
            active_requests.dec()


@app.post(
    "/transcribe/batch",
    dependencies=[Depends(rate_limiter)],
    response_model=None
)
async def transcribe_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple audio files"),
    language: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key)
):
    """
    Batch process multiple audio files

    **Parameters:**
    - **files**: List of audio files to transcribe
    - **language**: Optional language code for all files
    - **webhook_url**: Optional webhook URL for results

    **Returns:**
    - List of task IDs for async processing
    - Or immediate results if webhook not specified and Celery not available
    """
    correlation_id = request.state.correlation_id

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch request")

    # Validate language code
    if language and language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language code: {language}. Use /languages endpoint to see supported languages."
        )

    # Validate webhook URL if provided
    if webhook_url and not validate_webhook_url(webhook_url):
        raise HTTPException(
            status_code=400,
            detail="Invalid webhook URL. Must be a valid HTTP or HTTPS URL."
        )

    results = []

    for file in files:
        try:
            content = await file.read()
            file_size = len(content)

            if file_size > MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB"
                })
                continue

            # Queue for async processing if available
            if celery_app:
                task = transcribe_async_task.delay(
                    content,
                    file.filename,
                    language,
                    webhook_url,
                    f"{correlation_id}:{file.filename}"
                )
                results.append({
                    "filename": file.filename,
                    "task_id": task.id,
                    "status": "queued"
                })
            else:
                # Immediate processing
                filename = file.filename or "audio.wav"
                suffix = os.path.splitext(filename)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                result = transcribe_file(temp_file_path, language)
                os.unlink(temp_file_path)

                results.append({
                    "filename": file.filename,
                    "status": "completed",
                    "result": result
                })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    return JSONResponse(content={
        "success": True,
        "correlation_id": correlation_id,
        "total_files": len(files),
        "results": results
    })


@app.get("/task/{task_id}")
async def get_task_status(task_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get status of async transcription task

    **Parameters:**
    - **task_id**: Task ID returned from async transcription

    **Returns:**
    - Task status and result if completed
    """
    if not celery_app:
        raise HTTPException(
            status_code=503,
            detail="Async processing not available. Celery is not configured."
        )

    if not task_id or not task_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid task_id provided"
        )

    try:
        from celery.result import AsyncResult
        task = AsyncResult(task_id, app=celery_app)

        if task.ready():
            if task.successful():
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task.result
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(task.info)
                }
        else:
            return {
                "task_id": task_id,
                "status": "processing"
            }
    except Exception as e:
        logger.error(f"Error retrieving task status for {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving task status: {str(e)}"
        )


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported language codes"""
    return {
        "supported_languages": sorted(list(SUPPORTED_LANGUAGES)),
        "count": len(SUPPORTED_LANGUAGES),
        "note": "Language codes follow ISO 639-1 standard"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    redis_status = "connected" if redis_client else "disconnected"
    celery_status = "available" if celery_app else "unavailable"

    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "service": "Enhanced STT API",
        "version": "2.0.0",
        "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
        "supported_languages_count": len(SUPPORTED_LANGUAGES),
        "features": {
            "authentication": ENABLE_AUTH,
            "rate_limiting": True,
            "caching": redis_status,
            "async_processing": celery_status,
            "metrics": PROMETHEUS_AVAILABLE,
            "gpu_support": USE_GPU
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Metrics not available. Prometheus client is not installed."
        )

    from fastapi import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/analytics")
async def get_analytics(api_key: str = Depends(verify_api_key)):
    """
    Get usage analytics

    **Returns:**
    - Request statistics
    - Cache performance
    - Average processing time
    """
    avg_duration = (
        usage_stats["total_duration"] / usage_stats["successful_requests"]
        if usage_stats["successful_requests"] > 0
        else 0
    )

    cache_hit_rate = (
        usage_stats["cache_hits"] / (usage_stats["cache_hits"] + usage_stats["cache_misses"])
        if (usage_stats["cache_hits"] + usage_stats["cache_misses"]) > 0
        else 0
    )

    return {
        "total_requests": usage_stats["total_requests"],
        "successful_requests": usage_stats["successful_requests"],
        "failed_requests": usage_stats["failed_requests"],
        "success_rate": (
            usage_stats["successful_requests"] / usage_stats["total_requests"]
            if usage_stats["total_requests"] > 0
            else 0
        ),
        "average_processing_time": avg_duration,
        "cache_statistics": {
            "hits": usage_stats["cache_hits"],
            "misses": usage_stats["cache_misses"],
            "hit_rate": cache_hit_rate
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Enhanced Speech-to-Text API",
        "version": "2.0.0",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
        "features": {
            "authentication": ENABLE_AUTH,
            "rate_limiting": f"{RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s",
            "batch_processing": True,
            "webhooks": True,
            "export_formats": ["json", "srt", "vtt"],
            "caching": redis_client is not None,
            "async_queue": celery_app is not None,
            "gpu_support": USE_GPU,
            "metrics": PROMETHEUS_AVAILABLE
        },
        "endpoints": {
            "transcribe": "/transcribe (POST) - Transcribe single audio file",
            "transcribe_batch": "/transcribe/batch (POST) - Batch process multiple files",
            "task_status": "/task/{task_id} (GET) - Get async task status",
            "languages": "/languages (GET) - Get supported language codes",
            "health": "/health (GET) - Health check",
            "metrics": "/metrics (GET) - Prometheus metrics",
            "analytics": "/analytics (GET) - Usage analytics",
            "docs": "/docs - Interactive API documentation",
            "redoc": "/redoc - Alternative API documentation"
        },
        "documentation": "/docs"
    }
