@echo off

cd SwarmUI
echo Triggering SwarmUI rebuild...

rem Create the must_rebuild flag file
echo. 2>src\bin\must_rebuild

echo Rebuild flag created. Next time you run launch-windows.bat, it will rebuild automatically.
echo.
echo Alternatively, you can run launch-windows.bat now to trigger the rebuild immediately.
pause
