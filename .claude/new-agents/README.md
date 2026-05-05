# New Agents Onboarding

Welcome! This directory contains setup guides and configs for new agents working on this project.

## LaTeX Setup

### Quick Start
```bash
bash .claude/new-agents/setup-latex.sh
```

This script:
- Installs LaTeX distribution (auto-detects OS)
- Copies VSCode config to `.vscode/settings.json`
- Ready to compile on next build

### Files

| File | Purpose |
|------|---------|
| `LATEX_SETUP.md` | Complete LaTeX setup guide (read this first) |
| `vscode-settings.json` | VSCode config for LaTeX (auto-copied by setup script) |
| `setup-latex.sh` | Automated setup script |
| `README.md` | This file |

## What You Need to Know

### Document Structure
```
Weekly Report/
  ├── preamble.tex     (all packages & config)
  ├── CS703.tex        (main document)
  └── Assignment/
      └── weekX/main.tex  (week content)
```

### Key Constraints
- All reports output to `Weekly Report/` folder only
- Replace em dashes (`—`) with `,`, `:`, or `()` in .tex files
- Each week self-contained in its directory

### Compilation
Main document: `Weekly Report/CS703.tex`

Use VSCode LaTeX Workshop extension (auto-builds on save).

## Manual Setup

If you prefer manual setup instead of the script:

1. **Install LaTeX**
   - Linux: `sudo apt-get install texlive-latex-extra biber`
   - macOS: `brew install --cask mactex`
   - Windows: Install MiKTeX or TeX Live

2. **Install VSCode Extension**
   - Open VSCode → Extensions
   - Search: "LaTeX Workshop" (by James Yu)
   - Install `James-Yu.latex-workshop`

3. **Copy VSCode Settings**
   ```bash
   mkdir -p .vscode
   cp .claude/new-agents/vscode-settings.json .vscode/settings.json
   ```

4. **Build Document**
   - Open `Weekly Report/CS703.tex`
   - Click "Build LaTeX project" button (bottom toolbar)

## Questions?

See `LATEX_SETUP.md` for detailed info, troubleshooting, and build commands.
