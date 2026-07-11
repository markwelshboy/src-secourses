# Ultimate Image Captioner 1.2 Docker image

This build follows the same general container pattern as `comfyui-inference-headless-to-desktop`: an NVIDIA CUDA/Ubuntu base, a dedicated virtual environment, common pod tools, OpenSSH, a persistent `/workspace`, and a small entrypoint that pulls `markwelshboy/pod-runtime` at startup.

The supplied captioner stack is installed during the image build. `HF_model_downloader.py` is **not** run in the build or entrypoint; model downloads remain a runtime operation.

## Image names

The build publishes both tags:

```text
markwelshboy/ultimate_image_captioner:1.2
markwelshboy/ultimate_image_captioner:latest
```

## Build and push locally

From this directory:

```bash
docker login
bash build-and-push.sh
```

For a local non-pushed build:

```bash
PUSH=false bash build-and-push.sh
```

The image is intentionally `linux/amd64`, because the supplied flash-attention wheel is a CPython 3.11 Linux x86-64 wheel.

## RunPod / SimplePod settings

Use the image above, mount persistent storage at `/workspace`, expose HTTP port `7861`, and expose a TCP port mapped to container port `22` when direct SSH is required.

Recommended environment variables:

```text
HF_TOKEN=<runtime secret, when needed>
GRADIO_SHARE=false
CAPTIONER_AUTO_START=true
SSH_PUBLIC_KEY=<contents of your .pub file>
```

Only the **public** SSH key is needed. Store it in the pod provider's secret/environment-variable facility. Never place a private key in this repository, a Docker build argument, or the image.

The entrypoint also accepts `SSH_PUBLIC_KEY_B64`, `SSH_PUBLIC_KEY_FILE`, `RUNPOD_SSH_PUBLIC_KEY`, `PUBLIC_KEY`, or `SSH_AUTHORIZED_KEYS`. If the platform has already populated `/root/.ssh/authorized_keys`, it is preserved when no key variable is supplied.

## Runtime layout

Image-owned application source is baked into:

```text
/opt/Ultimate_Image_Captioner_Pro
```

At startup it is synchronized to:

```text
/workspace/Ultimate_Image_Captioner_Pro
```

The synchronization preserves these persistent runtime directories:

```text
outputs/
presets/
model_files_*/
```

This keeps the path expected by the original instructions and by `HF_model_downloader.py`, while allowing an image update to refresh application code. Newly shipped preset files are seeded without replacing existing user files.

## Download models at runtime

Place the existing downloader at `/workspace/HF_model_downloader.py`, then run:

```bash
cd /workspace
export HF_HOME=/workspace
/opt/venv/bin/python HF_model_downloader.py
```

Because the script is located in `/workspace`, its default target becomes `/workspace/Ultimate_Image_Captioner_Pro`.

## Launching

The container starts the app automatically with the equivalent container-safe command:

```bash
cd /workspace/Ultimate_Image_Captioner_Pro
export HF_HOME=/workspace
export PYTHONWARNINGS=ignore
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
/opt/venv/bin/python app.py \
  --server-name 0.0.0.0 \
  --server-port 7861 \
  --no-inbrowser
```

Set `GRADIO_SHARE=true` to add `--share`. The manual launcher is:

```bash
launch-captioner
```

Set `CAPTIONER_AUTO_START=false` for an SSH-only troubleshooting pod.

## GitHub Actions publishing

The repository workflow `.github/workflows/ultimate-image-captioner-docker.yml` builds this directory and pushes both Docker Hub tags on a manual dispatch or when relevant files are merged to `main`.

Configure these repository secrets first:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```
