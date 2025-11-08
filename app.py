from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    language: str = None
):
    """
    Transcribe audio file to text
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.)
    - language: Optional language code (e.g., 'en', 'es', 'de')
    """
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
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
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "service": "STT API"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Speech-to-Text API",
        "version": "1.0.0",
        "model": MODEL_SIZE,
        "endpoints": {
            "transcribe": "/transcribe (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }
```

**stt-service/.dockerignore**:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.env
.venv
venv/
