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
echo 1. Qwen Image Training Models
echo    - qwen_2.5_vl_7b_bf16.safetensors
echo    - qwen_train_vae.safetensors
echo    - qwen_image_bf16.safetensors
echo    Default directory: Training_Models_Qwen
echo.
echo 2. Qwen Image ^(2512^) Training Models
echo    - qwen_2.5_vl_7b_bf16.safetensors
echo    - qwen_train_vae.safetensors
echo    - Qwen_Image_2512_BF16.safetensors
echo    Default directory: Training_Models_Qwen
echo.
echo 3. Qwen Image Edit Plus ^(2509^) Training Models
echo    - qwen_2.5_vl_7b_bf16.safetensors
echo    - qwen_train_vae.safetensors
echo    - Qwen_Image_Edit_Plus_2509_bf16.safetensors
echo    Default directory: Training_Models_Qwen
echo.
echo 4. Qwen Image Edit ^(2511^) Training Models
echo    - qwen_2.5_vl_7b_bf16.safetensors
echo    - qwen_train_vae.safetensors
echo    - Qwen_Image_Edit_2511_BF16.safetensors
echo    Default directory: Training_Models_Qwen
echo.
echo 5. Wan 2.1 Text to Video Training Models
echo    - Wan2_1_VAE_bf16.safetensors
echo    - models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors
echo    - umt5-xxl-enc-bf16.safetensors
echo    - wan2.1_t2v_14B_bf16.safetensors
echo    Default directory: Training_Models_Wan
echo.
echo 6. Wan 2.2 Text to Video Training Models
echo    - Wan2_1_VAE_bf16.safetensors
echo    - umt5-xxl-enc-bf16.safetensors
echo    - Wan-2.2-T2V-High-Noise-BF16.safetensors
echo    - Wan-2.2-T2V-Low-Noise-BF16.safetensors
echo    Default directory: Training_Models_Wan
echo.
echo 7. Wan 2.2 Image to Video Training Models
echo    - Wan2_1_VAE_bf16.safetensors
echo    - umt5-xxl-enc-bf16.safetensors
echo    - Wan-2.2-I2V-Low-Noise-BF16.safetensors
echo    - Wan-2.2-I2V-High-Noise-BF16.safetensors
echo    Default directory: Training_Models_Wan
echo.

set /p choice="Enter your choice (1-7): "

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
) else (
    echo.
    echo Invalid choice. Please run the script again and select 1, 2, 3, 4, 5, 6, or 7.
    pause
    exit /b 1
)

REM Pause to keep the command prompt open
pause