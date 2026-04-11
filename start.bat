@echo off
:: ══════════════════════════════════════════════════════════════════
::  WRAPPER: Ensures window NEVER closes unexpectedly.
::  The script restarts itself inside "cmd /k" which keeps the
::  window open no matter what. On success, "exit" closes it.
:: ══════════════════════════════════════════════════════════════════
if "%~1"=="__run__" goto :MAIN
cmd /k "%~f0" __run__
exit /b 0

:MAIN
setlocal enabledelayedexpansion
title NeuralCensor Setup
mode con cols=80 lines=50

echo.
echo  ================================================================
echo    NeuralCensor - Automatic Image Anonymization
echo    Powered by SAM3 + Ollama
echo  ================================================================
echo.

cd /d "%~dp0"
echo [INFO] Working directory: %cd%
echo.

:: ─── If already set up, skip to launch ────────────────────────────
if exist ".setup_complete" (
    echo [OK] Setup already complete - skipping installation.
    echo      Launching NeuralCensor directly ...
    echo.
    if not exist "venv\Scripts\activate.bat" (
        echo  [ERROR] Virtual environment is missing!
        echo          Delete ".setup_complete" and run start.bat again.
        echo.
        echo  Type "exit" to close this window.
        goto :EOF
    )
    call venv\Scripts\activate.bat
    goto :OLLAMA
)

echo  IMPORTANT: Do NOT close this window during setup!
echo  First-time setup can take 15-30 minutes depending on your
echo  internet connection.
echo.
echo  ================================================================
echo.
echo  === FIRST-TIME SETUP - this only runs once ===
echo.
echo    Step 1/7    Check Python
echo    Step 2/7    Create virtual environment
echo    Step 3/7    Install PyTorch with CUDA
echo    Step 3.5/7  Install FFmpeg (for video audio)
echo    Step 4/7    Install base dependencies
echo    Step 5/7    Install SAM3 from GitHub
echo    Step 6/7    Download SAM3 checkpoint
echo    Step 7/7    Check Ollama + model
echo.
echo  ----------------------------------------------------------------
echo   Setup starting in 5 seconds ...
echo  ----------------------------------------------------------------
ping -n 6 127.0.0.1 >nul 2>nul
echo.
echo  Setup starting now ...
echo.

:: ─── Step 1: Check Python ─────────────────────────────────────────
echo.
echo  [Step 1/7] Checking Python installation ...
echo  ----------------------------------------------------------------
python --version 2>nul
if errorlevel 1 (
    echo.
    echo  [ERROR] Python was not found!
    echo          Install Python 3.12: https://www.python.org/downloads/
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)

for /f "tokens=2" %%v in ('python --version 2^>nul') do set PYVER=%%v
echo  [OK] Python !PYVER! found.

:: ─── Step 2: Virtual environment ──────────────────────────────────
echo.
echo  [Step 2/7] Setting up virtual environment ...
echo  ----------------------------------------------------------------
if exist "venv\Scripts\activate.bat" (
    echo  [OK] Virtual environment already exists.
) else (
    echo  [INFO] Creating virtual environment ...
    python -m venv venv
    if not exist "venv\Scripts\activate.bat" (
        echo  [ERROR] Could not create virtual environment.
        echo  Type "exit" to close this window.
        goto :EOF
    )
    echo  [OK] Virtual environment created.
)

echo  [INFO] Activating virtual environment ...
call venv\Scripts\activate.bat
echo  [OK] Virtual environment activated.

echo  [INFO] Upgrading pip ...
python -m pip install --upgrade pip >nul 2>nul
echo  [OK] pip upgraded.

:: ─── Step 3: Install PyTorch with CUDA ────────────────────────────
echo.
echo  [Step 3/7] Installing PyTorch 2.10 with CUDA 12.8 ...
echo  ----------------------------------------------------------------
echo  [INFO] Download size: approximately 3 GB
echo         This can take 5-20 minutes. DO NOT close this window!
echo.

cmd /c "pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --progress-bar on"
if errorlevel 1 (
    echo.
    echo  [WARNING] CUDA version failed. Trying CPU-only ...
    cmd /c "pip install torch torchvision --progress-bar on"
    if errorlevel 1 (
        echo  [ERROR] PyTorch installation failed completely.
        echo  Type "exit" to close this window.
        goto :EOF
    )
    echo  [OK] PyTorch CPU-only installed.
) else (
    echo  [OK] PyTorch with CUDA installed successfully.
)

:: ─── Step 3.5: Install FFmpeg ─────────────────────────────────────
echo.
echo  [Step 3.5/7] Checking for FFmpeg ...
echo  ----------------------------------------------------------------
:: Check system ffmpeg first, then venv-local ffmpeg
ffmpeg -version >nul 2>nul
if not errorlevel 1 (
    echo  [OK] FFmpeg is already installed system-wide.
    goto :FFMPEG_OK
)
:: Also check if ffmpeg is already installed inside the venv
if exist "%CD%\venv\Scripts\ffmpeg.exe" (
    echo  [OK] FFmpeg is already installed in venv.
    goto :FFMPEG_OK
)
echo  [INFO] FFmpeg not found. Downloading and installing FFmpeg ...
echo         This might take a minute...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile '%TEMP%\ffmpeg_nc.zip' -UseBasicParsing"
if errorlevel 1 (
    echo  [WARN] Failed to download FFmpeg. Videos will be saved without audio.
    goto :FFMPEG_OK
)
powershell -Command "Expand-Archive -Path '%TEMP%\ffmpeg_nc.zip' -DestinationPath '%TEMP%\ffmpeg_nc_ext' -Force"
if errorlevel 1 (
    echo  [WARN] Failed to extract FFmpeg archive. Videos will be saved without audio.
    del /f /q "%TEMP%\ffmpeg_nc.zip" >nul 2>nul
    goto :FFMPEG_OK
)
:: Copy ffmpeg.exe, ffprobe.exe, ffplay.exe into venv\Scripts
xcopy /s /y "%TEMP%\ffmpeg_nc_ext\ffmpeg-master-latest-win64-gpl\bin\*.exe" "%CD%\venv\Scripts\" >nul 2>nul
if errorlevel 1 (
    echo  [WARN] Could not copy FFmpeg to venv. Videos will be saved without audio.
) else (
    echo  [OK] FFmpeg installed successfully into venv\Scripts.
)
del /f /q "%TEMP%\ffmpeg_nc.zip" >nul 2>nul
rmdir /s /q "%TEMP%\ffmpeg_nc_ext" >nul 2>nul
:FFMPEG_OK

:: ─── Step 4: Install base dependencies ────────────────────────────
echo.
echo  [Step 4/7] Installing base dependencies ...
echo  ----------------------------------------------------------------
cmd /c "pip install -r requirements.txt --progress-bar on"
if errorlevel 1 (
    echo  [ERROR] Could not install dependencies.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] Base dependencies installed.

:: ─── Step 5: Install SAM3 ─────────────────────────────────────────
echo.
echo  [Step 5/7] Installing SAM3 from GitHub ...
echo  ----------------------------------------------------------------
echo  [INFO] Requires Git: https://git-scm.com/downloads
echo         This may take 2-5 minutes.
echo.
cmd /c "pip install git+https://github.com/facebookresearch/sam3.git"
if errorlevel 1 (
    echo  [ERROR] SAM3 installation failed.
    echo          Make sure Git is installed: https://git-scm.com/downloads
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] SAM3 installed.
echo.
echo  [INFO] Installing SAM3 extras ...
cmd /c "pip install einops pycocotools triton-windows"
echo  [OK] SAM3 dependencies ready.

:: ─── Step 6: HuggingFace Login and SAM3 Checkpoint ────────────────
if exist "checkpoints\sam3\model.safetensors" goto :SAM3_OK
if exist "checkpoints\sam3\config.json" goto :SAM3_OK

echo.
echo  [Step 6/7] Downloading SAM3 model checkpoint ...
echo  ----------------------------------------------------------------
echo.
echo  ================================================================
echo   HUGGINGFACE ACCESS REQUIRED
echo  ================================================================
echo.
echo   1. Create a free account at: https://huggingface.co
echo.
echo   2. Request model access - one time, usually instant:
echo      https://huggingface.co/facebook/sam3
echo      Click "Agree and access repository"
echo.
echo   3. Create an access token:
echo      https://huggingface.co/settings/tokens
echo      Click "Create new token" and enable ALL 3 checkboxes:
echo        [x] Read access to contents of all public gated repos
echo        [x] Read access to contents of all repos you can access
echo        [x] Make calls to inference providers
echo      Then click "Create token" and copy it.
echo.
echo   4. Paste the token below and press ENTER
echo.
echo  ----------------------------------------------------------------
echo   PRIVACY: Your token is stored ONLY locally on this PC at:
echo   %USERPROFILE%\.cache\huggingface\token
echo   It is NEVER sent anywhere except to huggingface.co
echo  ----------------------------------------------------------------
echo.
set /p HF_TOKEN="  Your HuggingFace Token: "
echo.

if "!HF_TOKEN!"=="" (
    echo  [ERROR] No token entered.
    echo  Type "exit" to close this window.
    goto :EOF
)

echo  [INFO] Logging in to HuggingFace ...
cmd /c "venv\Scripts\hf.exe auth login --token !HF_TOKEN!"
if errorlevel 1 (
    echo  [ERROR] HuggingFace login failed. Check your token.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] HuggingFace login successful.
echo.

mkdir checkpoints\sam3 2>nul
echo  [DOWNLOAD] Downloading SAM3 checkpoint - about 5 GB ...
echo             DO NOT close this window!
echo.
cmd /c "venv\Scripts\hf.exe download facebook/sam3 --local-dir checkpoints\sam3 --token !HF_TOKEN!"
if errorlevel 1 (
    echo  [ERROR] SAM3 download failed. Check access and token.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] SAM3 checkpoint downloaded.

:SAM3_OK
echo.
echo  [OK] SAM3 checkpoint is present.

:: Mark setup as complete
echo done > .setup_complete
echo  [OK] Setup complete! Next start will skip installation.
echo.

:OLLAMA
:: ─── Step 7: Check Ollama ─────────────────────────────────────────
echo.
echo  [Step 7/7] Checking Ollama + Model ...
echo  ----------------------------------------------------------------
curl -s --max-time 5 http://localhost:11434/api/tags >nul 2>nul
if errorlevel 1 (
    echo.
    echo  [ERROR] Ollama is not running!
    echo          1. Download: https://ollama.com/download
    echo          2. Start Ollama
    echo          3. Run start.bat again
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)
echo  [OK] Ollama is running.

echo.
echo  [INFO] Checking gemma4:e4b model ...
ollama list 2>nul | findstr /i "gemma4:e4b" >nul 2>nul
if not errorlevel 1 goto :MODEL_OK

echo  [DOWNLOAD] Pulling gemma4:e4b ...
ollama pull gemma4:e4b
if not errorlevel 1 (
    echo  [OK] gemma4:e4b ready.
    goto :LAUNCH
)

:: ── Pull failed → try automatic Ollama update ──────────────────────
echo.
echo  [WARNING] Pull failed. Attempting automatic Ollama update ...
echo  ----------------------------------------------------------------
echo.
echo  [DOWNLOAD] Downloading latest Ollama installer ...
echo             (from https://ollama.com/download/OllamaSetup.exe)
echo.
curl -L --progress-bar -o "%TEMP%\OllamaSetup.exe" "https://ollama.com/download/OllamaSetup.exe"
if errorlevel 1 (
    echo.
    echo  [ERROR] Could not download Ollama installer.
    echo          Please update manually: https://ollama.com/download
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)

echo.
echo  [INFO] Installing Ollama update silently ...
echo         Please wait, this may take 30-60 seconds ...
echo.
"%TEMP%\OllamaSetup.exe" /SILENT /NORESTART
if errorlevel 1 (
    echo  [WARNING] Silent install may have failed. Trying /S flag ...
    "%TEMP%\OllamaSetup.exe" /S
)

echo.
echo  [INFO] Waiting for Ollama to restart ...
ping -n 10 127.0.0.1 >nul 2>nul

:: Wait until Ollama API responds again (up to ~30s) - goto workaround for for-loop
set OLLAMA_READY=0
set WAIT_COUNT=0
:WAIT_LOOP
if !WAIT_COUNT! GEQ 10 goto :CHECK_READY
curl -s --max-time 3 http://localhost:11434/api/tags >nul 2>nul
if not errorlevel 1 (
    set OLLAMA_READY=1
    goto :CHECK_READY
)
ping -n 4 127.0.0.1 >nul 2>nul
set /a WAIT_COUNT+=1
goto :WAIT_LOOP

:CHECK_READY
if !OLLAMA_READY!==0 (
    echo.
    echo  [WARNING] Ollama did not restart automatically.
    echo           Please start Ollama manually, then run start.bat again.
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)

echo  [OK] Ollama is running again after update.
echo.
echo  [DOWNLOAD] Retrying: pulling gemma4:e4b ...
ollama pull gemma4:e4b
if errorlevel 1 (
    echo.
    echo  [ERROR] Could not pull gemma4:e4b even after Ollama update.
    echo.
    echo  ================================================================
    echo   MANUAL STEPS:
    echo  ================================================================
    echo.
    echo   1. Open a new terminal (cmd or PowerShell) and run:
    echo.
    echo        ollama pull gemma4:e4b
    echo.
    echo   2. Run start.bat again after the download is complete.
    echo.
    echo  ================================================================
    echo   Alternative models you can try:
    echo  ================================================================
    echo.
    echo     ollama pull gemma3:4b
    echo     ollama pull llama3.2:3b
    echo     ollama pull phi4-mini
    echo.
    echo  ================================================================
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)

echo.
echo  [OK] gemma4:e4b successfully downloaded!
echo.
echo  ================================================================
echo   Ollama was updated and the model is ready.
echo   Please close this window and run start.bat again
echo   to launch NeuralCensor.
echo  ================================================================
echo.
echo  Type "exit" to close this window.
goto :EOF

:MODEL_OK
echo  [OK] gemma4:e4b is available.

:LAUNCH
:: ─── Launch NeuralCensor ──────────────────────────────────────────
echo.
echo  ================================================================
echo    All checks passed! Launching NeuralCensor ...
echo  ================================================================
echo.

python neuralcensor.py

if errorlevel 1 (
    echo.
    echo  ================================================================
    echo  [ERROR] NeuralCensor exited with an error.
    echo          Check the messages above for details.
    echo  ================================================================
    echo.
    echo  Type "exit" to close this window.
    goto :EOF
)

echo.
echo  [OK] NeuralCensor closed normally. Goodbye!
echo.

:: SUCCESS: close the window automatically
endlocal
exit