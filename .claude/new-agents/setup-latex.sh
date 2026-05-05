#!/bin/bash
# LaTeX setup script for new agents
# Run from project root: bash .claude/new-agents/setup-latex.sh

set -e

echo "=== MonU Data Science Project - LaTeX Setup ==="
echo ""

# Ensure on correct branch (NEVER master)
echo "Checking git branch..."
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" = "master" ] || [ "$CURRENT_BRANCH" = "main" ]; then
    echo "ERROR: You are on $CURRENT_BRANCH branch!"
    echo "NEVER interact with master/main branch."
    echo "Switching to 'new' branch now..."
    git checkout new
    echo "✓ Switched to 'new' branch. NEVER checkout master again."
elif [ "$CURRENT_BRANCH" != "new" ]; then
    echo "WARNING: You are on '$CURRENT_BRANCH' branch, not 'new'."
    echo "Switching to 'new' branch..."
    git checkout new
else
    echo "✓ Correctly on 'new' branch"
fi
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
else
    OS="unknown"
fi

echo "Detected OS: $OS"
echo ""

# Install LaTeX based on OS
if [ "$OS" = "linux" ]; then
    echo "Installing LaTeX packages for Linux..."
    sudo apt-get update
    sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra biber
elif [ "$OS" = "mac" ]; then
    echo "Installing MacTeX..."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Install from https://brew.sh"
        exit 1
    fi
    brew install --cask mactex
else
    echo "Unsupported OS. Install LaTeX manually from:"
    echo "  Windows: https://miktex.org/ or https://www.tug.org/texlive/"
    echo "  Linux: sudo apt-get install texlive-latex-extra biber"
    echo "  macOS: brew install --cask mactex"
fi

echo ""
echo "Setting up VSCode config..."

# Create .vscode directory
mkdir -p .vscode

# Copy settings file
cp .claude/new-agents/vscode-settings.json .vscode/settings.json
echo "✓ VSCode settings copied to .vscode/settings.json"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Install VSCode extension: James-Yu.latex-workshop"
echo "2. Open Weekly Report/CS703.tex in VSCode"
echo "3. Click 'Build LaTeX project' in the bottom toolbar"
echo ""
echo "For detailed info, see: .claude/new-agents/LATEX_SETUP.md"
