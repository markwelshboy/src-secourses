# Ultimate Image Captioner 1.2 Docker image

This build follows the same general container pattern as `comfyui-inference-headless-to-desktop`: an NVIDIA CUDA/Ubuntu base, a dedicated virtual environment, common pod tools, OpenSSH, a persistent `/workspace`, and a small entrypoint that pulls `markwelshboy/pod-runtime` at startup.

The supplied captioner stack and `HF_model_downloader.py` are included in the image. Model weights are not baked into the image; they remain a runtime download into persistent `/workspace` storage.

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

The downloader is baked into the image and installed at startup as:

```text
/workspace/HF_model_downloader.py
```

Keeping it at `/workspace` is important because the downloader calculates its default target relative to its own file location. Its target therefore becomes `/workspace/Ultimate_Image_Captioner_Pro`, rather than creating a nested application directory.

## Download models at runtime

Use the included wrapper:

```bash
download-captioner-models
```

That command runs `/workspace/HF_model_downloader.py` with the image's Python environment and downloads into:

```text
/workspace/Ultimate_Image_Captioner_Pro
```

After a first-time model download, restart the application so any failed lazy model state is cleared:

```bash
captionerctl restart
```

## Application control and logs

The application runs under Supervisor, independently of SSH. Available commands are:

```bash
captionerctl status
captionerctl start
captionerctl stop
captionerctl restart
captionerctl logs
captionerctl doctor
```

The persistent application log is:

```text
/workspace/logs/captioner.log
```

Supervisor's own log is:

```text
/workspace/logs/supervisord.log
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

Set `CAPTIONER_AUTO_START=false` for an SSH-first troubleshooting pod, then start it with `captionerctl start`.

## GitHub Actions publishing

The repository workflow `.github/workflows/ultimate-image-captioner-docker.yml` builds this directory and pushes both Docker Hub tags on a manual dispatch or when relevant files are merged to `main`.

Configure these repository secrets first:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```
