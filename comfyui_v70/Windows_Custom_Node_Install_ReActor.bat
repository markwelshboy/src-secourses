@echo off

cd ComfyUI

call .\venv\Scripts\activate.bat

cd custom_nodes

git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor

git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack

cd ComfyUI-ReActor

git stash

git reset --hard

git pull --force

python install.py

uv pip install -r requirements.txt

cd ..

cd ComfyUI-Impact-Pack

git stash

git reset --hard

git pull --force

uv pip install -r requirements.txt

cd ..

cd ..

echo Installing requirements...

uv pip install -r requirements.txt

uv pip install onnxruntime-gpu==1.22.0

REM Pause to keep the command prompt open
pause