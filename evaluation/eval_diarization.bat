@echo off
echo ========================================
echo SPEAKER DIARIZATION EVALUATION
echo ========================================

cd /d "%~dp0"

echo Activating virtual environment...
if exist "..\..\venv\Scripts\activate.bat" (
    call "..\..\venv\Scripts\activate.bat"
) else if exist "..\cpuvenv\Scripts\activate.bat" (
    call "..\cpuvenv\Scripts\activate.bat"
) else if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
) else (
    echo Warning: No virtual environment found, using system Python
)

echo.
echo Installing required packages...
pip install numpy tqdm

echo.
echo Choose evaluation mode:
echo [1] Full dataset evaluation (all files - ~15000 files)
echo [2] Quick test (100 files for testing)
echo [3] Medium test (1000 files)
echo [4] Custom number of files
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Running full dataset evaluation...
    python eval_diarization.py
) else if "%choice%"=="2" (
    echo Running quick test with 100 files...
    python eval_diarization.py --max_files 100
) else if "%choice%"=="3" (
    echo Running medium test with 1000 files...
    python eval_diarization.py --max_files 1000
) else if "%choice%"=="4" (
    set /p max_files="Enter number of files to evaluate: "
    python eval_diarization.py --max_files %max_files%
) else (
    echo Invalid choice, running quick test...
    python eval_diarization.py --max_files 100
)

echo.
echo ========================================
echo EVALUATION COMPLETED
echo ========================================
echo.
echo Results are saved in: eval_results/
echo.
echo To compare results, run:
echo python compare_diarization.py
echo.

pause
