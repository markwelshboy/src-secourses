@echo off

cd ComfyUI

call .\venv\Scripts\activate.bat

REM set CUDA_VISIBLE_DEVICES=0

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
set NVIDIA_TF32_OVERRIDE=1
set CUDA_MODULE_LOADING=LAZY

python.exe -s main.py --windows-standalone-build --use-sage-attention --auto-launch

pause
