FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV QWEN_MODEL_ID=Qwen/Qwen-Image-2512 \
    MODEL_STORAGE_PATH=/runpod-volume/qwen-image-2512 \
    MIN_STORAGE_FREE_GB=40 \
    RUNPOD_INIT_TIMEOUT=1800

CMD ["python", "-u", "/app/handler.py"]
