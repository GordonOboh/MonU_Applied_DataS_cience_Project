# LaTeX Setup Guide for New Agents

## Required LaTeX Packages

This project uses APA 7th edition formatting with the following packages. Install via your LaTeX distribution:

### Core Packages (in preamble.tex)
- `apa7` — APA 7th edition document class
- `comment` — Multi-line comment support
- `babel` — Language support (american)
- `amsmath` — Advanced math environments
- `csquotes` — Context-sensitive quotation marks
- `biblatex` — Bibliography management
- `fontenc` — Font encoding (T1)
- `mathptmx` — Times New Roman font
- `graphicx` — Graphics inclusion
- `pdfpages` — PDF page inclusion
- `enumitem` — Enhanced list environments
- `pdflscape` — Landscape page support
- `setspace` — Line spacing control
- `hyperref` — Hyperlinks and PDF metadata
- `url` — URL formatting

## Installation

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra biber
```

### macOS (Homebrew)
```bash
brew install --cask mactex
```

### Windows
- Install MiKTeX or TeX Live from official sources

## Project LaTeX Configuration

### File Structure
```
Weekly Report/
  ├── preamble.tex        (packages & document setup)
  ├── CS703.tex           (main document)
  └── Assignment/
      └── weekX/
          └── main.tex    (week-specific content)
```

### Key Settings (in preamble.tex)
```latex
\documentclass[stu,12pt,floatsintext,article]{apa7}
\hyphenpenalty=10000           % Prevent hyphenation
\exhyphenpenalty=10000         % Prevent ex-hyphenation
\graphicspath{ {./images/} }   % Image search path
\newcommand{\comma}{,}         % Custom comma command (em dash replacement)
```

### Report Output Location
All compiled PDFs save to `Weekly Report/` folder only.

## VSCode Configuration

### Extensions Required
1. **LaTeX Workshop** (James Yu)
   - ID: `james-yu.latex-workshop`
   - Provides LaTeX compilation, preview, and snippets

### VSCode Settings for LaTeX

Add to `.vscode/settings.json`:

```json
{
  "[latex]": {
    "editor.defaultFormatter": "James-Yu.latex-workshop"
  },
  "latex-workshop.latex.tools": [
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "%DOC%"
      ]
    },
    {
      "name": "biber",
      "command": "biber",
      "args": ["%DOCFILE%"]
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "pdflatex → biber → pdflatex × 2",
      "tools": ["pdflatex", "biber", "pdflatex", "pdflatex"]
    }
  ],
  "latex-workshop.view.pdf.viewer": "browser",
  "latex-workshop.synctex.afterBuild.enabled": true,
  "editor.formatOnSave": false
}
```

### Copy VSCode Config to Project
```bash
mkdir -p /home/cc/MonU_Applied_DataS_cience_Project/.vscode
cp /home/cc/.vscode/settings.json /home/cc/MonU_Applied_DataS_cience_Project/.vscode/
```

## Build Commands

### Compile Main Document
```bash
cd "Weekly Report"
pdflatex -synctex=1 -interaction=nonstopmode CS703.tex
biber CS703
pdflatex -synctex=1 -interaction=nonstopmode CS703.tex
pdflatex -synctex=1 -interaction=nonstopmode CS703.tex
```

### Clean Build Artifacts
```bash
rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz
```

## Important Rules

### Em Dash Replacement
Replace em dashes (`—`) in .tex files with commas, colons, or parentheses instead:
- ❌ `text — more text`
- ✅ `text, more text` or `text: more text` or `text (more text)`

### File Structure
- `Main.tex` in root calls `weekX/main.tex`
- Each week is self-contained in its `weekX/` directory
- All reports output to `Weekly Report/` folder

### Bibliography (if needed)
Uncomment in `preamble.tex`:
```latex
\addbibresource{bibliography.bib}
```
Then place `bibliography.bib` in `Weekly Report/` folder.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Package not found | Run `sudo apt-get install texlive-latex-extra` (Linux) |
| Biber fails | Ensure biber is installed: `sudo apt-get install biber` |
| VSCode doesn't recognize LaTeX | Install "LaTeX Workshop" extension |
| PDF won't compile | Check for em dashes (`—`) in .tex files, replace with `,`, `:`, or `()` |
| Bibliography not showing | Uncomment `\addbibresource{}` in preamble.tex |

## ⚠️ CRITICAL: Master Branch Rule

**NEVER interact with `master` branch.** Not checkout, not merge, not push. Ignore it completely.

Only work on `new` branch.

---

## Quick Start for New Agent

0. **Checkout `new` branch:** `git checkout new` (NEVER touch `master`)
1. Install LaTeX distribution
2. Install LaTeX Workshop extension in VSCode
3. Copy VSCode config: `cp /home/cc/.vscode/settings.json .vscode/`
4. Open `Weekly Report/CS703.tex` in VSCode
5. Click "Build LaTeX project" in bottom toolbar
6. View PDF in browser preview
