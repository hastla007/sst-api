from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os

app = FastAPI()

# Load Whisper model (you can choose: tiny, base, small, medium, large)
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe audio
        result = model.transcribe(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return JSONResponse(content={
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"]
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
