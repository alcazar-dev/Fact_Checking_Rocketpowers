FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch CPU >= 2.6, sin CUDA/NVIDIA
RUN pip install --no-cache-dir torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY cascade_router.py .
COPY train_phobert_evidence.py .

COPY phobert_evidence_checkpoints/ ./phobert_evidence_checkpoints/
COPY phobert_v2_evidence_checkpoints/ ./phobert_v2_evidence_checkpoints/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]