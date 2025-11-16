from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import whisper
import tempfile
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

app = FastAPI(title="STT Service API", version="1.0.0")

# Add CORS middleware to allow n8n to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
logger.info(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)
logger.info("Model loaded successfully")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Transcribe audio file to text

    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.) - Max 100MB
    - language: Optional language code (e.g., 'en', 'es', 'de')

    Returns:
    - success: Boolean indicating success
    - text: Transcribed text
    - language: Detected/specified language
    - segments: Detailed transcription segments with timestamps
    """
    temp_file_path = None
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")

        # Validate language code if provided
        if language and language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language code: {language}. Use /languages endpoint to see supported languages."
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

        logger.info(f"File size: {file_size / (1024*1024):.2f}MB")

        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Transcribing file: {temp_file_path}")
        
        # Transcribe audio
        transcribe_options = {}
        if language:
            transcribe_options["language"] = language
            
        result = model.transcribe(temp_file_path, **transcribe_options)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        logger.info("Transcription completed successfully")
        
        return JSONResponse(content={
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"]
        })
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp file: {cleanup_error}")
        raise

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)

        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp file: {cleanup_error}")

        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "service": "STT API",
        "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
        "supported_languages_count": len(SUPPORTED_LANGUAGES)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Speech-to-Text API",
        "version": "1.0.0",
        "model": MODEL_SIZE,
        "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
        "endpoints": {
            "transcribe": "/transcribe (POST) - Transcribe audio to text",
            "languages": "/languages (GET) - Get supported language codes",
            "health": "/health (GET) - Health check",
            "docs": "/docs - Interactive API documentation"
        }
    }