# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG CAPTIONER_REPO=https://github.com/FurkanGozukara/Ultimate_Image_Captioner_Pro.git
ARG CAPTIONER_REF=main
ARG IMAGE_VERSION=1.2

LABEL org.opencontainers.image.title="Ultimate Image Captioner" \
      org.opencontainers.image.description="CUDA 13 / PyTorch 2.11 runtime for Ultimate Image Captioner Pro" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.source="https://github.com/markwelshboy/src-secourses"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PIP_PREFER_BINARY=1 \
    UV_SKIP_WHEEL_FILENAME_CHECK=1 \
    UV_LINK_MODE=copy \
    VENV=/opt/venv \
    CAPTIONER_IMAGE_DIR=/opt/Ultimate_Image_Captioner_Pro \
    CAPTIONER_WORKSPACE_DIR=/workspace/Ultimate_Image_Captioner_Pro \
    CAPTIONER_DOWNLOADER_SOURCE=/opt/ultimate-image-captioner-tools/HF_model_downloader.py \
    CAPTIONER_DOWNLOADER=/workspace/HF_model_downloader.py \
    CAPTIONER_LOG_FILE=/workspace/logs/captioner.log \
    HF_HOME=/workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Match the general-purpose Linux/tooling layer used by the ComfyUI image,
# but use Python 3.11 because the supplied flash-attention wheel is cp311.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
       software-properties-common \
       ca-certificates curl wget gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 python3.11-venv python3.11-dev \
       git git-lfs curl ca-certificates jq \
       build-essential gcc g++ cmake ninja-build pkg-config \
       ffmpeg aria2 rsync tmux unzip wget vim less nano \
       bash-completion net-tools iproute2 iputils-ping procps lsof \
       libgl1 libglib2.0-0 libsm6 libxext6 \
       openssh-server supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /var/run/sshd /workspace /workspace/logs \
    && git lfs install --system \
    && python3.11 -m venv "${VENV}"

ENV PATH="${VENV}/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel uv

COPY requirements_premium_caption.txt /tmp/requirements_premium_caption.txt

# Install the stack from Runpod_Ultimate_Caption_Install.sh. Models remain a
# runtime download, but the downloader itself is included in the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
      --python "${VENV}/bin/python" \
      --index-strategy unsafe-best-match \
      -r /tmp/requirements_premium_caption.txt \
    && uv pip install \
      --python "${VENV}/bin/python" \
      --no-deps \
      --index-url https://download.pytorch.org/whl/cu130 \
      xformers==0.0.35

RUN git clone --depth 1 "${CAPTIONER_REPO}" "${CAPTIONER_IMAGE_DIR}" \
    && if [[ "${CAPTIONER_REF}" != "main" ]]; then \
         git -C "${CAPTIONER_IMAGE_DIR}" fetch --depth 1 origin "${CAPTIONER_REF}"; \
         git -C "${CAPTIONER_IMAGE_DIR}" checkout --detach FETCH_HEAD; \
       fi \
    && rm -rf "${CAPTIONER_IMAGE_DIR}/.git" \
    && mkdir -p \
       "${CAPTIONER_IMAGE_DIR}/outputs" \
       "${CAPTIONER_IMAGE_DIR}/presets" \
       "${CAPTIONER_IMAGE_DIR}/model_files_beta_one" \
       "${CAPTIONER_IMAGE_DIR}/model_files_qwen3_vl3_8b_instruct" \
       /opt/ultimate-image-captioner-tools

COPY HF_model_downloader.py /opt/ultimate-image-captioner-tools/HF_model_downloader.py

RUN python - <<'PY'
import sys
import torch
import torchvision
import gradio
import transformers
import xformers

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("gradio:", gradio.__version__)
print("transformers:", transformers.__version__)
print("xformers:", xformers.__version__)
PY

RUN python -m py_compile /opt/ultimate-image-captioner-tools/HF_model_downloader.py

COPY start.sh /usr/local/bin/start-captioner-container
COPY launch-captioner.sh /usr/local/bin/launch-captioner
COPY captionerctl.sh /usr/local/bin/captionerctl
COPY download-models.sh /usr/local/bin/download-captioner-models
RUN chmod 0755 \
      /usr/local/bin/start-captioner-container \
      /usr/local/bin/launch-captioner \
      /usr/local/bin/captionerctl \
      /usr/local/bin/download-captioner-models

WORKDIR /workspace
EXPOSE 22 7861
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/start-captioner-container"]
