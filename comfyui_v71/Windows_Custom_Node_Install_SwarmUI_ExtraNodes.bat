@echo off

cd ComfyUI

call .\venv\Scripts\activate.bat

cd custom_nodes

REM Remove existing SwarmComfyCommon if it exists
if exist SwarmComfyCommon (
    echo Removing existing SwarmComfyCommon...
    rmdir /s /q SwarmComfyCommon
)

REM Remove existing SwarmKSampler if it exists
if exist SwarmKSampler (
    echo Removing existing SwarmKSampler...
    rmdir /s /q SwarmKSampler
)

echo Downloading SwarmUI ExtraNodes (SwarmComfyCommon and SwarmKSampler)...

REM Clone SwarmComfyCommon
git clone --depth 1 --filter=blob:none --sparse https://github.com/mcmonkeyprojects/SwarmUI
cd SwarmUI
git sparse-checkout set src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon
xcopy /E /I /Y "src\BuiltinExtensions\ComfyUIBackend\ExtraNodes\SwarmComfyCommon" "..\SwarmComfyCommon"
cd ..

cd SwarmUI
git sparse-checkout set src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyExtra
xcopy /E /I /Y "src\BuiltinExtensions\ComfyUIBackend\ExtraNodes\SwarmComfyExtra" "..\SwarmComfyExtra"
cd ..

REM Clone SwarmKSampler
cd SwarmUI
git sparse-checkout set src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmKSampler
xcopy /E /I /Y "src\BuiltinExtensions\ComfyUIBackend\ExtraNodes\SwarmKSampler" "..\SwarmKSampler"
cd ..

REM Clean up temporary SwarmUI folder
rmdir /s /q SwarmUI

echo SwarmUI ExtraNodes installed successfully!

cd ..

cd ..

REM Pause to keep the command prompt open
pause

