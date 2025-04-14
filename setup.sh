#!/bin/bash

# Print colorful messages
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

print_yellow() {
    echo -e "\033[33m$1\033[0m"
}

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Linux"
fi

print_green "Setting up SPEAK environment on $OS..."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    print_red "Conda is not installed. Please install Miniconda or Anaconda first."
    print_yellow "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create and activate conda environment
print_green "Creating conda environment..."
conda env create -f environment.yml
if [ $? -ne 0 ]; then
    print_red "Failed to create conda environment."
    exit 1
fi

# Activate the environment
print_green "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate speak
if [ $? -ne 0 ]; then
    print_red "Failed to activate conda environment."
    exit 1
fi

# Create necessary directories
print_green "Creating project directories..."
mkdir -p data checkpoints logs results

# Verify installation
print_green "Verifying installation..."
python -c "import torch; import librosa; import numpy; print('PyTorch version:', torch.__version__); print('Librosa version:', librosa.__version__); print('NumPy version:', numpy.__version__)"
if [ $? -ne 0 ]; then
    print_red "Failed to verify installation."
    exit 1
fi

print_green "Setup completed successfully!"
print_yellow "To activate the environment in the future, run: conda activate speak" 