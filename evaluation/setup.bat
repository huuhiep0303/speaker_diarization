@echo off
REM Quick setup script for evaluation

cd /d "%~dp0"

echo ============================================================
echo SETUP EVALUATION ENVIRONMENT
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo Python found!
echo.

REM Install dependencies
echo Installing dependencies...
echo.

echo Installing core packages...
pip install numpy scipy scikit-learn matplotlib tqdm

echo.
echo Installing ASR evaluation packages...
pip install jiwer regex librosa soundfile

echo.
echo Installing speaker diarization packages...
pip install torch torchaudio speechbrain

echo.
echo Installing ASR models (optional)...
pip install faster-whisper funasr

echo.
echo ============================================================
echo CREATING SAMPLE DATASET
echo ============================================================
echo.

python create_dataset.py --mode sample

echo.
echo ============================================================
echo SETUP COMPLETED!
echo ============================================================
echo.
echo Next steps:
echo   1. Place your audio files in test_audio/ folder
echo   2. Run: eval_diarization.bat test_audio speechbrain
echo   3. Run: eval_asr.bat sample_dataset.csv whisper cpu
echo.

pause
