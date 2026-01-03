@echo off

if exist venv_downloader (
    echo Virtual environment already exists. Skipping creation...
) else (
    echo Creating virtual environment...
    python -m venv venv_downloader
)


set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy

call .\venv_downloader\Scripts\activate.bat

python -m pip install --upgrade pip

pip install uv

uv pip install -U gradio==6.2.0 huggingface_hub hf_transfer hf_xet requests

set HF_XET_CHUNK_CACHE_SIZE_BYTES=90737418240

set HUGGING_FACE_HUB_TOKEN=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
python -W ignore Downloader_Gradio_App.py

pause