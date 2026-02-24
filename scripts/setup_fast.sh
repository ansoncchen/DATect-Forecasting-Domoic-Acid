#!/bin/bash
# DATect Fast Setup Script
# Uses UV (Rust-based pip) for 10-100x faster package installation

set -e

echo "========================================"
echo "DATect Fast Setup with UV Package Manager"
echo "========================================"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "UV installed successfully!"
else
    echo "UV is already installed: $(uv --version)"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with UV (much faster than pip)
echo ""
echo "Installing dependencies with UV (this is ~10x faster than pip)..."
uv pip install -r requirements.txt

# Install optional Redis support
read -p "Install Redis caching support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv pip install redis hiredis
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the application:"
echo "  python run_datect.py"
echo ""
echo "To use the fast Granian server:"
echo "  granian --interface asgi backend.api:app --port 8000"
echo "========================================"
