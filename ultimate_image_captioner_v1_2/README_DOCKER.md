# Ultimate Image Captioner 1.2 Docker image

This build follows the same general container pattern as `comfyui-inference-headless-to-desktop`: an NVIDIA CUDA/Ubuntu base, a dedicated virtual environment, common pod tools, OpenSSH, a persistent `/workspace`, and a small entrypoint that pulls `markwelshboy/pod-runtime` at startup.

The supplied captioner stack and `HF_model_downloader.py` are included in the image. Model weights are not baked into the image; they are downloaded into persistent `/workspace` storage after the application becomes available.

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

Mount persistent storage at `/workspace`, expose HTTP port `7861`, and expose a TCP port mapped to container port `22` when direct SSH is required.

Recommended environment variables:

```text
HF_TOKEN=<runtime secret, when needed>
GRADIO_SHARE=false
CAPTIONER_AUTO_START=true
CAPTIONER_AUTO_DOWNLOAD_MODELS=true
CAPTIONER_RESTART_AFTER_MODEL_DOWNLOAD=true
SSH_PUBLIC_KEY=<contents of your .pub file>

TELEGRAM_BOT_TOKEN=<Telegram bot token>
TELEGRAM_CHAT_ID=<Telegram chat id>
TELEGRAM_NAME=ultimate-image-captioner::runpod
TELEGRAM_NAG_ENABLED=true
TELEGRAM_NAG_INTERVAL_SECONDS=1800
```

Only the **public** SSH key is needed. Store it in the pod provider's secret/environment-variable facility. Never place a private key, Telegram bot token, or Hugging Face token in this repository, a Docker build argument, or the image.

The entrypoint also accepts `SSH_PUBLIC_KEY_B64`, `SSH_PUBLIC_KEY_FILE`, `RUNPOD_SSH_PUBLIC_KEY`, `PUBLIC_KEY`, or `SSH_AUTHORIZED_KEYS`. If the platform has already populated `/root/.ssh/authorized_keys`, it is preserved when no key variable is supplied.

Telegram notifications are best effort. A missing token, invalid chat, or temporary Telegram outage does not stop the application or model downloader.

## Automatic startup sequence

With the default settings, the container performs this sequence:

1. Synchronizes the baked application into `/workspace/Ultimate_Image_Captioner_Pro` while preserving models, outputs, and presets.
2. Installs `/workspace/HF_model_downloader.py`.
3. Pulls `pod-runtime`, configures SSH, and starts Supervisor.
4. Starts the Gradio application.
5. Waits until port `7861` responds.
6. Sends a Telegram message that the pod has started and the app is up.
7. Runs `download-captioner-models` automatically.
8. Sends Telegram messages when model downloading starts and ends.
9. Restarts the application after a successful model download or verification pass.
10. Sends a reminder every 30 minutes while the pod remains active.

The downloader is idempotent and skips files that are already present and verified. Automatic downloading can be disabled with:

```text
CAPTIONER_AUTO_DOWNLOAD_MODELS=false
```

The post-download restart can be disabled independently with:

```text
CAPTIONER_RESTART_AFTER_MODEL_DOWNLOAD=false
```

The 30-minute reminder can be disabled or adjusted with:

```text
TELEGRAM_NAG_ENABLED=false
TELEGRAM_NAG_INTERVAL_SECONDS=1800
```

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

The downloader is baked into the image and installed at startup as:

```text
/workspace/HF_model_downloader.py
```

Keeping it at `/workspace` is important because the downloader calculates its default target relative to its own file location. Its target therefore becomes `/workspace/Ultimate_Image_Captioner_Pro`, rather than creating a nested application directory.

## Manual model download

Automatic downloading is enabled by default. To run it manually:

```bash
captionerctl download
captionerctl doctor
captionerctl restart
```

The wrapper sends the same Telegram start/end notifications as the automatic process.

## Application control and logs

The application runs under Supervisor, independently of SSH. Available commands are:

```bash
captionerctl status
captionerctl services
captionerctl start
captionerctl stop
captionerctl restart
captionerctl logs
captionerctl bootstrap-logs
captionerctl nag-logs
captionerctl download
captionerctl telegram-test
captionerctl doctor
```

Persistent logs are:

```text
/workspace/logs/captioner.log
/workspace/logs/captioner-bootstrap.log
/workspace/logs/captioner-nag.log
/workspace/logs/supervisord.log
```

The automatic downloader's console output is written to `captioner-bootstrap.log`.

To inspect all supervised processes directly:

```bash
supervisorctl -c /etc/supervisor/conf.d/ultimate-captioner.conf status
```

To test Telegram manually:

```bash
captionerctl telegram-test "Test message from the captioner pod"
```

`captionerctl doctor` checks CUDA availability, inspects the expected Qwen model directory, detects the common nested-downloader layout, and attempts a local `AutoProcessor` load.

If an older image was used to run `HF_model_downloader.py` from inside the application directory, it may have created:

```text
/workspace/Ultimate_Image_Captioner_Pro/Ultimate_Image_Captioner_Pro
```

Repair that layout with:

```bash
captionerctl repair-download-layout
captionerctl doctor
captionerctl restart
```

The repair merges missing files into the correct application directory but deliberately leaves the nested source directory in place until it is manually verified and removed.

## Launching

The container starts the app automatically through Supervisor with the equivalent container-safe command:

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

Set `GRADIO_SHARE=true` to add `--share`. The low-level manual launcher is:

```bash
launch-captioner
```

Set `CAPTIONER_AUTO_START=false` for an SSH-first troubleshooting pod, then start it with `captionerctl start`. The automatic downloader bootstrap is only started when the application is auto-started.

## GitHub Actions publishing

The repository workflow `.github/workflows/ultimate-image-captioner-docker.yml` builds this directory and pushes both Docker Hub tags on a manual dispatch or when relevant files are merged to `main`.

Configure these repository secrets first:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```
