@echo off

cd Ultimate_Image_Captioner_Pro

call .\venv\Scripts\activate.bat

REM SET CUDA_VISIBLE_DEVICES=0  - this is used to set certain CUDA device visible only used
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore

python app.py

pause
