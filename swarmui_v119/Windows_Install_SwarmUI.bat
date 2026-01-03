@echo off

cd /d "%~dp0"

if exist SwarmUI (
    echo SwarmUI is already installed in this folder. If this is incorrect, delete the 'SwarmUI' folder and try again.
    pause
    exit /b
)

if exist SwarmUI.sln (
    echo SwarmUI is already installed in this folder. If this is incorrect, delete 'SwarmUI.sln' and try again.
    pause
    exit /b
)

set "tempfile=%TEMP%\swarm_dotnet_sdklist.tmp"
dotnet --list-sdks > "%tempfile%"
findstr "8.0." "%tempfile%" > nul
if %ERRORLEVEL% neq 0 (
    echo DotNet SDK 8 is not installed, will install from WinGet...
    winget install Microsoft.DotNet.SDK.8 --accept-source-agreements --accept-package-agreements
)
del "%tempfile%"

WHERE git
IF %ERRORLEVEL% NEQ 0 (
    winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
)

git clone --depth 1 https://github.com/mcmonkeyprojects/SwarmUI
git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-Frame-Interpolation
git clone --depth 1 https://github.com/welltop-cn/ComfyUI-TeaCache SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-TeaCache
git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/comfyui_controlnet_aux

cd SwarmUI

cmd /c .\launch-windows.bat --launch_mode webinstall

IF %ERRORLEVEL% NEQ 0 ( pause )
