# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS base

ARG APP_NAME=xtts-finetune-webui
ARG CUDA_VER=cu121
ARG GID=966
ARG UID=966
ARG WHISPER_MODEL="large-v3"

# Environment
ENV APP_NAME=$APP_NAME \
    CUDA_VER=$CUDA_VER \
    WHISPER_MODEL=$WHISPER_MODEL

# User configuration
ENV HOME /app/$APP_NAME
RUN groupadd -r app -g $GID && \
    useradd --no-log-init -m -r -g app app -u $UID

# Prepare file-system
RUN mkdir -p /app/server && chown -R $UID:$GID /app
COPY --chown=$UID:$GID *.py *.sh *.txt *.md /app/server/
ADD --chown=$UID:$GID utils /app/server/utils

# Enter environment and install dependencies
WORKDIR /app/server

USER $UID:$GID

ENV NVIDIA_VISIBLE_DEVICES=all PATH=$PATH:$HOME/.local/bin
# Install nvidia-pyindex & nvidia-cudnn for libcudnn_ops_infer.so.8
# See: https://github.com/SYSTRAN/faster-whisper/issues/516
RUN pip3 install --user --no-cache-dir nvidia-pyindex && \
    pip3 install --user --no-cache-dir nvidia-cudnn && \
    pip3 install --user --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/$CUDA_VER && \
    pip3 install --user --no-cache-dir -r requirements.txt --no-cache-dir && \
    python3 -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8')"

# Ports and servername
EXPOSE 5003
ENV GRADIO_ANALYTICS_ENABLED="False"

CMD [ "bash", "start-container.sh"]
