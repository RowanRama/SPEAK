@echo off
echo Setting up SPEAK environment on Windows...

:: Check if Conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda is not installed. Please install Miniconda or Anaconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Create and activate conda environment
echo Creating conda environment...
call conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo Failed to create conda environment.
    pause
    exit /b 1
)

:: Activate the environment
echo Activating conda environment...
call conda activate speak
if %errorlevel% neq 0 (
    echo Failed to activate conda environment.
    pause
    exit /b 1
)

:: Create necessary directories
echo Creating project directories...
mkdir data 2>nul
mkdir checkpoints 2>nul
mkdir logs 2>nul
mkdir results 2>nul

:: Verify installation
echo Verifying installation...
python -c "import torch; import librosa; import numpy; print('PyTorch version:', torch.__version__); print('Librosa version:', librosa.__version__); print('NumPy version:', numpy.__version__)"
if %errorlevel% neq 0 (
    echo Failed to verify installation.
    pause
    exit /b 1
)

echo Setup completed successfully!
echo To activate the environment in the future, run: conda activate speak
pause 