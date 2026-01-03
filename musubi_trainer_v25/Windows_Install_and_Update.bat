@echo off

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy

echo WARNING. For this auto installer to work you need to have installed Python 3.10.11, Git, FFmpeg, cuDNN 9.4+, CUDA 12.8, MSVC and C++ tools 
echo This tutorial shows all step by step : https://youtu.be/DrhUHnYfwC0?si=UAAVyZ8_QUPAjy7a

git clone --depth 1 https://github.com/FurkanGozukara/SECourses_Musubi_Trainer

cd SECourses_Musubi_Trainer

git reset --hard

git pull

git clone https://github.com/kohya-ss/musubi-tuner

cd musubi-tuner

git reset --hard

git pull

git checkout main

git pull

cd ..

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo Python launcher is available. Generating Python 3.10 VENV
    py -3.10 -m venv venv
) else (
    echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.10
    python -m venv venv
)

call .\venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_trainer.txt

cd SECourses_Musubi_Trainer

cd musubi-tuner

uv pip install -e .

echo installation completed check for errors

REM Pause to keep the command prompt open
pause