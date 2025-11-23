FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies + Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python/pip aliases
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip + install CUDA-compatible PyTorch
RUN pip install --upgrade pip setuptools wheel


# Install CUDA PyTorch first
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2  --extra-index-url https://download.pytorch.org/whl/cu121


RUN pip install --no-cache-dir \
    torchmetrics==1.2.0 \
    lightning==2.1.3 \
    transformers==4.34.0 \
    pyannote.audio==3.1.1

# THEN install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



# Copy application
COPY . .

# Pre-download Whisper model
RUN python -c "import whisper; whisper.load_model('base')"

# Expose ports
EXPOSE 8004
EXPOSE 8005

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
