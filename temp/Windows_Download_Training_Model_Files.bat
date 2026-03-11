@echo off

cd SECourses_Musubi_Trainer

call .\venv\Scripts\activate.bat

cd ..

echo Installing/upgrading required packages...

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy

python.exe -m pip install --upgrade pip
pip install uv
uv pip install huggingface_hub hf_xet ipywidgets hf_transfer

echo.
echo ====================================================
echo Training Models Download
echo ====================================================
echo.
echo Please select which models to download:
echo.
python Download_Train_Models.py --list
echo.

set /p choice="Enter your choice (1-12): "

if "%choice%"=="1" (
    echo.
    echo Downloading Qwen Image Training Models...
    python Download_Train_Models.py --model qwen_image
) else if "%choice%"=="2" (
    echo.
    echo Downloading Qwen Image ^(2512^) Training Models...
    python Download_Train_Models.py --model qwen_image_2512
) else if "%choice%"=="3" (
    echo.
    echo Downloading Qwen Image Edit Plus ^(2509^) Training Models...
    python Download_Train_Models.py --model qwen_image_edit_plus
) else if "%choice%"=="4" (
    echo.
    echo Downloading Qwen Image Edit ^(2511^) Training Models...
    python Download_Train_Models.py --model qwen_image_edit_2511
) else if "%choice%"=="5" (
    echo.
    echo Downloading Wan 2.1 Text to Video Training Models...
    python Download_Train_Models.py --model wan21_t2v
) else if "%choice%"=="6" (
    echo.
    echo Downloading Wan 2.2 Text to Video Training Models...
    python Download_Train_Models.py --model wan22_t2v
) else if "%choice%"=="7" (
    echo.
    echo Downloading Wan 2.2 Image to Video Training Models...
    python Download_Train_Models.py --model wan22_i2v
) else if "%choice%"=="8" (
    echo.
    echo Downloading FLUX 2 Dev Training Models...
    python Download_Train_Models.py --model flux2_dev
) else if "%choice%"=="9" (
    echo.
    echo Downloading FLUX Klein 9B Training Models...
    python Download_Train_Models.py --model flux_klein_9b
) else if "%choice%"=="10" (
    echo.
    echo Downloading FLUX Klein 4B Training Models...
    python Download_Train_Models.py --model flux_klein_4b
) else if "%choice%"=="11" (
    echo.
    echo Downloading Z-Image Base Training Models - includes Z_Image_Training_Text_Encoder.safetensors...
    python Download_Train_Models.py --model z_image_base
) else if "%choice%"=="12" (
    echo.
    echo Downloading Z-Image Turbo Training Models - includes Z_Image_Training_Text_Encoder.safetensors...
    python Download_Train_Models.py --model z_image_turbo
) else (
    echo.
    echo Invalid choice. Please run the script again and select a number from 1 to 12.
    pause
    exit /b 1
)

REM Pause to keep the command prompt open
pause
