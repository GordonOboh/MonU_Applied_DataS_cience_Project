---
name: LaTeX file structure requirement
description: All LaTeX files must follow specific hierarchy — main.tex calls weekX/main.tex, each weekX folder self-contained
type: feedback
originSessionId: cabb42c3-e9d1-49d7-a091-b0898b8b2d5e
---
**Mandatory structure:**
```
Weekly Report/
├── CS703.tex (calls Week 1, Week 2, etc. via \input)
├── week1/
│   ├── main.tex
│   ├── Phase_1.tex (or other phase files)
│   └── images/ (all images for week1)
├── week2/
│   ├── main.tex
│   ├── Phase_2.tex
│   └── images/
```

**Why:** Self-contained week folders enable independent compilation, clean organization, reusable structure across projects.

**How to apply:** Every .tex file created goes in weekX/ folder (not root). Every image goes in weekX/images/. Root main.tex only calls weekX/main.tex files via \input. No exceptions.
