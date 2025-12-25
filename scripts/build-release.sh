#!/bin/bash
# Metafrasis Release Build Script
# Builds frontend component, runs tests, and creates git tag

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Metafrasis Release Build${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# ============================================================================
# 1. Environment Checks
# ============================================================================
echo -e "${BLUE}Checking environment...${NC}"

# Check if we're in project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    echo "Please install Node.js: https://nodejs.org"
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)
UV_VERSION=$(uv --version)

echo -e "${GREEN}✓ Node.js: $NODE_VERSION${NC}"
echo -e "${GREEN}✓ npm: $NPM_VERSION${NC}"
echo -e "${GREEN}✓ uv: $UV_VERSION${NC}"
echo ""

# ============================================================================
# 2. Git Status Check
# ============================================================================
echo -e "${BLUE}Checking git status...${NC}"

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: You have uncommitted changes${NC}"
    echo ""
    echo "Uncommitted files:"
    git status --short
    echo ""
    echo "Please commit or stash your changes before building a release."
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
LATEST_COMMIT=$(git log -1 --pretty=format:"%h - %s")

echo -e "${GREEN}✓ No uncommitted changes${NC}"
echo -e "  Branch: $CURRENT_BRANCH"
echo -e "  Latest commit: $LATEST_COMMIT"
echo ""

# ============================================================================
# 3. Run Tests
# ============================================================================
echo -e "${BLUE}Running tests...${NC}"

if [ -d "tests" ]; then
    uv run pytest
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${YELLOW}⚠ No tests directory found, skipping tests${NC}"
fi
echo ""

# ============================================================================
# 4. Build Frontend Component
# ============================================================================
echo -e "${BLUE}Building frontend component...${NC}"

cd frontend/ocr_viewer

# Run npm build
npm run build

# Verify build directory exists and has files
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build/ directory not found${NC}"
    exit 1
fi

if [ -z "$(ls -A build)" ]; then
    echo -e "${RED}Error: build/ directory is empty${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Frontend build successful${NC}"
echo -e "  Build directory: frontend/ocr_viewer/build/"

cd ../..
echo ""

# ============================================================================
# 5. Create Git Tag
# ============================================================================
echo -e "${BLUE}Creating git tag...${NC}"

# Generate timestamp-based version
VERSION_DATE=$(date +%Y.%m.%d)
VERSION="v${VERSION_DATE}"

# Check if tag already exists, append counter if needed
COUNTER=1
while git tag | grep -q "^${VERSION}$"; do
    COUNTER=$((COUNTER + 1))
    VERSION="v${VERSION_DATE}.${COUNTER}"
done

# Create annotated tag
git tag -a "$VERSION" -m "Release $VERSION"

echo -e "${GREEN}✓ Created git tag: $VERSION${NC}"
echo ""

# ============================================================================
# 6. Display Success Message
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}  Release Build Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Build Summary:"
echo "  Version: $VERSION"
echo "  Frontend: Built successfully"
echo "  Tests: Passed"
echo "  Git tag: Created"
echo ""
echo -e "${YELLOW}To run in production mode:${NC}"
echo -e "  ${GREEN}VIEWER_RELEASE=true uv run streamlit run app.py${NC}"
echo ""
echo -e "${YELLOW}To push the tag to remote:${NC}"
echo -e "  ${GREEN}git push origin $VERSION${NC}"
echo ""

# Ask if user wants to push tag
read -p "Do you want to push the tag to remote now? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin "$VERSION"
    echo -e "${GREEN}✓ Tag pushed to remote${NC}"
else
    echo "Tag not pushed. You can push it later with:"
    echo "  git push origin $VERSION"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
