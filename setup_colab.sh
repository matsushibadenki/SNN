#!/bin/bash
# setup_colab.sh
# SNN Research Project Environment Setup Script for Google Colab

set -e  # Exit immediately if a command exits with a non-zero status.

echo "🚀 Starting SNN Project Setup..."

# 1. System Dependencies (if any specific ones are needed)
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y python3-opengl

# 2. Python Dependencies
echo "🐍 Installing Python dependencies from pyproject.toml..."
# Upgrade pip first
pip install --upgrade pip

# Install the project in editable mode, which installs dependencies listed in pyproject.toml
pip install -e .

# 3. Validation
echo "🔍 Verifying installation..."
python -c "import snn_research; print(f'✅ SNN Research Library v{snn_research.__version__} installed successfully.')"
python -c "import torch; print(f'✅ PyTorch v{torch.__version__} available.')"

echo "📂 Creating workspace directories..."
mkdir -p workspace/logs workspace/models workspace/data workspace/results

echo "✅ Setup Complete! You can now run the SNN CLI."
echo "   Example: snn-cli --help"
