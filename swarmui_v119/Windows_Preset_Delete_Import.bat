@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo    SwarmUI Preset Reset and Import
echo ============================================
echo.
echo This will:
echo 1. Backup your current presets to 'presets_backups' folder
echo 2. Delete ALL existing presets from SwarmUI
echo 3. Import the latest Amazing_SwarmUI_Presets_v*.json file
echo.
echo REQUIREMENTS:
echo    - SwarmUI must be running (http://localhost:7861)
echo    - Amazing_SwarmUI_Presets_v*.json file in this folder
echo.

:confirm
set /p choice="Do you want to proceed? (y/N): "
if /i "%choice%"=="y" goto :proceed
if /i "%choice%"=="yes" goto :proceed
if "%choice%"=="" goto :abort
if /i "%choice%"=="n" goto :abort
if /i "%choice%"=="no" goto :abort
echo Invalid choice. Please enter 'y' for yes or 'n' for no.
goto :confirm

:abort
echo.
echo Operation cancelled by user.
echo.
pause
exit /b 0

:proceed
echo.
echo Starting preset reset and import process...
echo.

REM Install required Python package if not present
echo CHECKING: Python requirements...
python -c "import requests" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing required package: requests
    pip install requests
    if %ERRORLEVEL% neq 0 (
        echo.
        echo Failed to install required Python package 'requests'
        echo Please install it manually: pip install requests
        echo.
        pause
        exit /b 1
    )
)

echo.
echo RUNNING: Preset manager...
cd utilities
python swarmui_preset_manager.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Preset management failed!
    echo Check the error messages above.
    echo.
) else (
    echo.
    echo Preset management completed!
    echo.
    echo Next steps:
    echo 1. Check your SwarmUI web interface
    echo 2. Go to the Presets tab
    echo 3. Your new presets should be available
    echo.
    echo Your original presets are backed up in the 'presets_backups' folder
)

echo.
pause