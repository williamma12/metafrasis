#!/bin/bash
# Metafrasis Setup Script
# Installs system dependencies and Python packages

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
INSTALL_DEV=false
INSTALL_OCR_FULL=false
DOWNLOAD_MODELS=false

for arg in "$@"; do
    case $arg in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --ocr-full)
            INSTALL_OCR_FULL=true
            shift
            ;;
        --download-models)
            DOWNLOAD_MODELS=true
            shift
            ;;
        --help|-h)
            echo "Metafrasis Setup Script"
            echo ""
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev              Install development dependencies (pytest, ruff, etc.)"
            echo "  --ocr-full         Install all OCR component dependencies (CRAFT, DB, Kraken)"
            echo "  --download-models  Download pretrained OCR model weights"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./setup.sh                           # Install base dependencies only"
            echo "  ./setup.sh --dev                     # Install base + dev dependencies"
            echo "  ./setup.sh --ocr-full                # Install base + all OCR components"
            echo "  ./setup.sh --download-models         # Install dependencies + download models"
            echo "  ./setup.sh --dev --ocr-full --download-models  # Full installation"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Metafrasis Setup${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    echo "This script supports macOS and Linux only."
    echo "For Windows, please install dependencies manually."
    exit 1
fi

echo -e "${GREEN}Detected OS: $OS${NC}"
echo ""

# Install Tesseract
echo -e "${BLUE}Installing Tesseract OCR...${NC}"

if [[ "$OS" == "macos" ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew not found!${NC}"
        echo "Please install Homebrew first: https://brew.sh"
        exit 1
    fi

    # Install Tesseract and language data
    echo "Installing via Homebrew..."
    brew install tesseract
    brew install tesseract-lang

elif [[ "$OS" == "linux" ]]; then
    # Check if we have sudo
    if ! command -v sudo &> /dev/null; then
        echo -e "${RED}sudo not found!${NC}"
        exit 1
    fi

    # Detect Linux distro
    if command -v apt-get &> /dev/null; then
        echo "Installing via apt..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-grc poppler-utils
    elif command -v yum &> /dev/null; then
        echo "Installing via yum..."
        sudo yum install -y tesseract tesseract-langpack-grc poppler-utils
    else
        echo -e "${RED}Unsupported package manager${NC}"
        echo "Please install tesseract-ocr manually"
        exit 1
    fi
fi

# Verify Tesseract installation
if command -v tesseract &> /dev/null; then
    TESSERACT_VERSION=$(tesseract --version | head -n1)
    echo -e "${GREEN}✓ Tesseract installed: $TESSERACT_VERSION${NC}"
else
    echo -e "${RED}✗ Tesseract installation failed${NC}"
    exit 1
fi

echo ""

# Install poppler (for PDF support) on macOS
if [[ "$OS" == "macos" ]]; then
    echo -e "${BLUE}Installing poppler (for PDF support)...${NC}"
    brew install poppler
    echo -e "${GREEN}✓ poppler installed${NC}"
    echo ""
fi

# Check for uv
echo -e "${BLUE}Checking for uv package manager...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv not found!${NC}"
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo -e "${RED}uv installation failed${NC}"
        echo "Please install uv manually: https://github.com/astral-sh/uv"
        exit 1
    fi
fi

UV_VERSION=$(uv --version)
echo -e "${GREEN}✓ uv installed: $UV_VERSION${NC}"
echo ""

# Check for Node.js and npm
echo -e "${BLUE}Checking for Node.js and npm...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js not found!${NC}"
    echo "Installing Node.js..."

    if [[ "$OS" == "macos" ]]; then
        brew install node
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif command -v yum &> /dev/null; then
            curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
            sudo yum install -y nodejs
        fi
    fi

    if ! command -v node &> /dev/null; then
        echo -e "${RED}Node.js installation failed${NC}"
        echo "Please install Node.js manually: https://nodejs.org"
        exit 1
    fi
fi

NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)
echo -e "${GREEN}✓ Node.js installed: $NODE_VERSION${NC}"
echo -e "${GREEN}✓ npm installed: $NPM_VERSION${NC}"
echo ""

# Install frontend component dependencies
echo -e "${BLUE}Installing frontend component dependencies...${NC}"
if [ -d "frontend/ocr_viewer" ]; then
    cd frontend/ocr_viewer
    echo "Running npm install in frontend/ocr_viewer..."
    npm install
    cd ../..
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${RED}Warning: frontend/ocr_viewer directory not found${NC}"
    echo "Skipping frontend dependencies..."
fi
echo ""

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"

# Build extras string
EXTRAS=""
if [[ "$INSTALL_DEV" == true ]]; then
    EXTRAS="$EXTRAS --extra dev"
fi

if [[ "$INSTALL_OCR_FULL" == true ]]; then
    EXTRAS="$EXTRAS --extra ocr-full"
fi

# Run uv sync
if [[ -n "$EXTRAS" ]]; then
    echo "Installing with extras:$EXTRAS"
    uv sync $EXTRAS
else
    echo "Installing base dependencies only"
    uv sync
fi

echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# Download OCR models (optional)
if [[ "$DOWNLOAD_MODELS" == true ]]; then
    echo -e "${BLUE}Downloading pretrained OCR model weights...${NC}"
    echo ""

    # Check if download script exists
    if [ ! -f "models/download_models.py" ]; then
        echo -e "${RED}✗ models/download_models.py not found${NC}"
        echo "Skipping model downloads..."
    else
        # Download CRAFT and CRNN models
        echo "Downloading CRAFT detector weights (MLT dataset)..."
        uv run python models/download_models.py --craft base
        echo ""

        echo "Downloading CRNN recognizer weights..."
        uv run python models/download_models.py --crnn base
        echo ""

        echo -e "${GREEN}✓ OCR model weights downloaded${NC}"
    fi
    echo ""
else
    echo -e "${BLUE}Skipping model downloads${NC}"
    echo "To download OCR models later, run:"
    echo "  python models/download_models.py --all"
    echo ""
fi

# Summary
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Installed components:"
echo "  ✓ Tesseract OCR with Ancient Greek language data"
echo "  ✓ poppler-utils (for PDF support)"
echo "  ✓ Node.js and npm"
echo "  ✓ Frontend component dependencies"
echo "  ✓ Python dependencies via uv"

if [[ "$INSTALL_DEV" == true ]]; then
    echo "  ✓ Development dependencies (pytest, ruff, etc.)"
fi

if [[ "$INSTALL_OCR_FULL" == true ]]; then
    echo "  ✓ All OCR component dependencies (scipy, pyclipper, kraken)"
fi

if [[ "$DOWNLOAD_MODELS" == true ]]; then
    echo "  ✓ Pretrained OCR model weights (CRAFT + CRNN)"
fi

echo ""
echo "To run the application:"
echo "  uv run streamlit run app.py"
echo ""
echo "To develop the frontend component:"
echo "  cd frontend/ocr_viewer && npm run dev"
echo ""
echo "To build the frontend component:"
echo "  cd frontend/ocr_viewer && npm run build"
echo ""
echo "To run tests (if --dev was installed):"
echo "  uv run pytest"
echo ""

if [[ "$INSTALL_OCR_FULL" == false ]]; then
    echo "To install specific OCR components later:"
    echo "  uv sync --extra craft     # CRAFT detector"
    echo "  uv sync --extra db        # DB detector"
    echo "  uv sync --extra kraken    # Kraken recognizer"
    echo "  uv sync --extra ocr-full  # All components"
    echo ""
fi

if [[ "$DOWNLOAD_MODELS" == false ]]; then
    echo "To download OCR model weights later:"
    echo "  python models/download_models.py --all"
    echo ""
fi
