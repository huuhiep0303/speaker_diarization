@echo off
REM Script to run ASR Evaluation on Windows for JVS Dataset

cd /d "%~dp0"

echo ============================================================
echo ASR EVALUATION - JVS DATASET
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
set DATASET=%1
set MODEL=%2
set DEVICE=%3
set WHISPER_SIZE=%4

if "%DATASET%"=="" (
    set DATASET=dataset_400_testcases.csv
    echo Using default dataset: dataset_400_testcases.csv
)

if "%MODEL%"=="" (
    set MODEL=whisper
    echo Using default model: whisper
)

if "%DEVICE%"=="" (
    set DEVICE=cpu
    echo Using default device: cpu
)

if "%WHISPER_SIZE%"=="" (
    set WHISPER_SIZE=small
)

echo.
echo Configuration:
echo   Dataset:      %DATASET%
echo   Model:        %MODEL%
echo   Device:       %DEVICE%
if "%MODEL%"=="whisper" (
    echo   Whisper Size: %WHISPER_SIZE%
)
echo.

REM Check if dataset exists
if not exist "%DATASET%" (
    echo ERROR: Dataset file not found: %DATASET%
    echo.
    echo Please create dataset first:
    echo   python create_dataset.py --jvs_root ..\dataset\jvs_ver1
    pause
    exit /b 1
)

echo Starting evaluation...
echo Progress will be auto-saved to checkpoint file.
echo You can stop and resume anytime.
echo.

if "%MODEL%"=="whisper" (
    python eval_asr.py ^
        --dataset "%DATASET%" ^
        --model whisper ^
        --whisper_size %WHISPER_SIZE% ^
        --device %DEVICE% ^
        --compute_type int8 ^
        --resume
) else if "%MODEL%"=="sensevoice" (
    python eval_asr.py ^
        --dataset "%DATASET%" ^
        --model sensevoice ^
        --device %DEVICE% ^
        --resume
) else (
    echo ERROR: Unknown model: %MODEL%
    echo Supported models: whisper, sensevoice
    pause
    exit /b 1
)

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
echo Results saved in: eval_results\
echo   - Checkpoint CSV with detailed results
echo   - Summary JSON with statistics
echo.
echo To resume: Just run this script again with same parameters
echo.

pause
