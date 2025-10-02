#!/bin/bash

# PromptBN Installation Script
# This script sets up the PromptBN environment

set -e  # Exit on any error

echo "🚀 PromptBN Installation Script"
echo "==============================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10+ is required, but found Python $PYTHON_VERSION"
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip is required but not installed."
    echo "Please install pip and try again."
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "✅ pip found: $($PIP_CMD --version)"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    echo "Please check your internet connection and try again"
    exit 1
fi

# Setup environment
echo ""
echo "🔧 Setting up environment..."
python3 setup_environment.py

if [ $? -eq 0 ]; then
    echo "✅ Environment setup completed"
else
    echo "❌ Environment setup failed"
    exit 1
fi

# Check for R installation
echo ""
echo "🔍 Checking R installation..."
if command -v R &> /dev/null; then
    echo "✅ R found: $(R --version | head -1)"
    echo ""
    echo "📦 To install required R packages, run the following in R console:"
    echo "   if (!requireNamespace(\"BiocManager\", quietly=TRUE)) install.packages(\"BiocManager\")"
    echo "   BiocManager::install(c(\"graph\",\"RBGL\",\"pcalg\"))"
    echo ""
    echo "🧪 Testing R packages..."
    if R -e "library(graph); library(RBGL); library(pcalg); cat('✅ R packages installed successfully\n')" 2>/dev/null; then
        echo "✅ R packages are working correctly"
    else
        echo "⚠️  R packages not installed. Please install them manually:"
        echo "   R -e \"if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager'); BiocManager::install(c('graph','RBGL','pcalg'))\""
    fi
else
    echo "❌ R not found. R is required for SHD calculation on CPDAGs."
    echo "Please install R and the required packages:"
    echo "   1. Install R from https://www.r-project.org/"
    echo "   2. Install packages: graph, RBGL, pcalg"
    echo ""
    echo "Installation commands:"
    echo "   macOS: brew install r"
    echo "   Ubuntu: sudo apt-get install r-base"
    echo "   Windows: Download from r-project.org"
    exit 1
fi

# Check for API key
echo ""
echo "🔑 Checking API key configuration..."
if [ -f ".env" ]; then
    if grep -q "OPENAI_API_KEY" .env && ! grep -q "your-key-here" .env; then
        echo "✅ API key configured in .env file"
    else
        echo "⚠️  Please edit .env file with your actual OpenAI API key"
        echo "   echo 'OPENAI_API_KEY=your-actual-key-here' > .env"
    fi
else
    echo "⚠️  No .env file found. Please create one with your API key:"
    echo "   echo 'OPENAI_API_KEY=your-actual-key-here' > .env"
fi

# Test basic functionality
echo ""
echo "🧪 Testing basic functionality..."
python3 -c "
import sys
sys.path.insert(0, 'promptbn')
try:
    from utils.path_utils import resolve_path
    print('✅ Path utilities working')
    
    from utils.data_utils import parse_var_csv_to_string
    print('✅ Data utilities working')
    
    print('✅ Basic functionality test passed')
except Exception as e:
    print(f'❌ Basic functionality test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Basic functionality test passed"
else
    echo "❌ Basic functionality test failed"
    echo "Some dependencies may be missing. Check the troubleshooting guide."
fi

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Set up your OpenAI API key in .env file"
echo "2. Install R packages (if you haven't already)"
echo "3. Add dataset files to data/BNlearnR/ or data/BnRep/"
echo "4. Run experiments: python3 run_promptbn.py --load_config --config configs/promptbn_asia_baseline.yaml"
echo ""
echo "📚 For more information:"
echo "   - Full documentation: cat README.md"
echo ""
echo "Happy experimenting! 🚀"
