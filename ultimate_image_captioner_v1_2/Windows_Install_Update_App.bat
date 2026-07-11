@echo off

echo WARNING. For this auto installer to work you need to have installed Python 3.11.10+, Git, FFmpeg, CUDA 13.0+ and C++ tools 
echo This tutorial shows all step by step : https://youtu.be/DrhUHnYfwC0

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy

git clone https://github.com/FurkanGozukara/Ultimate_Image_Captioner_Pro

cd Ultimate_Image_Captioner_Pro

git reset --hard

git pull

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo Python launcher is available. Generating Python 3.11 VENV
    py -3.11 -m venv venv
) else (
    echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.11
    python -m venv venv
)

call .\venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_premium_caption.txt  --index-strategy unsafe-best-match
uv pip install xformers==0.0.35 --no-deps --index-url https://download.pytorch.org/whl/cu130

echo downloading models starting, if you get download error, use download models bat file

python HF_model_downloader.py

REM Show completion message
echo Ultimate Image Captioner Pro installed check out all messages and save them before close to verify any errors or not later

REM Pause to keep the command prompt open
pause
