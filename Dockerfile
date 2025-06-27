#FROM nvcr.io/nvidia/l4t-pytorch:r36.3.0-py3
FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04
WORKDIR /workspace/wyoming-f5-tts

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git sox libsox-fmt-all libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone F5-TTS repository
RUN git clone https://github.com/SWivid/F5-TTS.git && \
    cd F5-TTS && pip install -e .

# Copy Wyoming server code
COPY wyoming_f5_tts/ wyoming_f5_tts/
#COPY main.py .

# Expose Wyoming TTS port
EXPOSE 10200

# Run the Wyoming server
CMD ["python", "wyoming_f5_tts/__main__.py", "--uri", "tcp://0.0.0.0:10200", "--streaming", "--debug"]

