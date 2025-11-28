@echo off
REM Script to run Speaker Diarization Evaluation on Windows

cd /d "%~dp0"

echo ============================================================
echo SPEAKER DIARIZATION EVALUATION
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Parse arguments or use defaults
set DATA_DIR=%1
set MODEL=%2
set MAX_GENUINE=%3
set MAX_IMPOSTOR=%4

if "%DATA_DIR%"=="" (
    set DATA_DIR=test_audio
    echo Using default data_dir: test_audio
)

if "%MODEL%"=="" (
    set MODEL=speechbrain
    echo Using default model: speechbrain
)

if "%MAX_GENUINE%"=="" (
    set MAX_GENUINE=50
)

if "%MAX_IMPOSTOR%"=="" (
    set MAX_IMPOSTOR=100
)

echo.
echo Configuration:
echo   Data directory: %DATA_DIR%
echo   Model: %MODEL%
echo   Max genuine pairs: %MAX_GENUINE%
echo   Max impostor pairs: %MAX_IMPOSTOR%
echo.

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo ERROR: Data directory not found: %DATA_DIR%
    echo.
    echo Please create the directory and add speaker audio files
    pause
    exit /b 1
)

echo Starting evaluation...
echo.

python eval_diarization.py ^
    --data_dir "%DATA_DIR%" ^
    --model %MODEL% ^
    --max_genuine %MAX_GENUINE% ^
    --max_impostor %MAX_IMPOSTOR% ^
    --use_cache

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo EVALUATION COMPLETED SUCCESSFULLY!
echo ============================================================
echo Results saved in: eval_results/
echo.

pause
