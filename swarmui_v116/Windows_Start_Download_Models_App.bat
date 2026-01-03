@echo off

if exist venv_downloader (
    echo Virtual environment already exists. Skipping creation...
) else (
    echo Creating virtual environment...
    python -m venv venv_downloader
)

call .\venv_downloader\Scripts\activate.bat

python -m pip install --upgrade pip

pip install uv

uv pip install -U gradio==6.2.0 huggingface_hub hf_transfer hf_xet requests

set HF_XET_CHUNK_CACHE_SIZE_BYTES=90737418240

set HUGGING_FACE_HUB_TOKEN=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
python -W ignore Downloader_Gradio_App.py

pause