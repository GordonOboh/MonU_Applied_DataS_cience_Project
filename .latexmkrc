# LaTeX Workshop configuration for this project

# PDF generator
$pdf_mode = 1;  # Use pdflatex
$postscript_mode = 0;
$dvi_mode = 0;

# Specify root file location
$root_filename = 'Weekly Report/CS703.tex';

# Output directory
$out_dir = 'Weekly Report';
$aux_dir = 'Weekly Report';

# Compilation settings
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Clean up auxiliary files but keep PDF
@generated_exts = ('aux', 'auxlock', 'lof', 'lot', 'maf', 'mtc', 'mtc*', 'toc', 'fmt', 'fls', 'fdb_latexmk', 'dpth', 'md5', 'auxlock');

# Continue on errors
$silent = 0;
