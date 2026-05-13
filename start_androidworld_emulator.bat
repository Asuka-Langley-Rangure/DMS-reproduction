@echo off
set EMULATOR_PATH=D:\Android\Sdk\emulator\emulator.exe
set AVD_NAME=AndroidWorldAvd

echo Checking emulator path...

if not exist "%EMULATOR_PATH%" (
    echo ERROR: emulator.exe not found at: %EMULATOR_PATH%
    pause
    exit /b 1
)

echo.
echo Available AVDs:
"%EMULATOR_PATH%" -list-avds

echo.
echo Starting Android emulator: %AVD_NAME%
echo Command: "%EMULATOR_PATH%" -avd %AVD_NAME% -no-snapshot -grpc 8554
echo.

"%EMULATOR_PATH%" -avd %AVD_NAME% -no-snapshot -grpc 8554

pause