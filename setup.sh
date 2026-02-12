#!/bin/bash
# Setup script for K-Number Predicate Device Extractor

set -e

echo "=========================================="
echo "K-Number Extractor Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip upgraded"

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Check .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  .env file not found!"
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✓ Created .env file (edit with your credentials)"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env with your Snowflake and Z.ai credentials"
    echo "2. Run: python k_number_extractor_batch.py --limit 5"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "=========================================="
echo "Setup complete! ✓"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Edit .env with your credentials"
echo "  2. source venv/bin/activate"
echo "  3. python k_number_extractor_batch.py --limit 5"
echo ""
