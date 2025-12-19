FROM python:3.10-slim

# Install ffmpeg and curl
RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple/

# Pre-download the model during build
RUN python -c "from speechbrain.pretrained import EncoderClassifier; EncoderClassifier.from_hparams(source='Jzuluaga/accent-id-commonaccent_ecapa', savedir='/app/model_cache')"

COPY app.py .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
