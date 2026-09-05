# Paper

Source for *Scoring conventions reorder adaptation policies in streaming anomaly
detection*.

Published on Zenodo: [10.5281/zenodo.22333127](https://doi.org/10.5281/zenodo.22333127) (https://zenodo.org/records/22333127).

The deposited PDF is the committed one. Zenodo files are immutable once
published, so any correction means a new version there, not an edit.

## What is here

| File | What it is |
| --- | --- |
| `main.tex` | The manuscript. Includes everything below. |
| `sections/results.tex` | Results section. |
| `sections/discussion.tex` | Discussion, limitations, conclusion. |
| `tables/*.tex` | **Generated.** Every table, written from `results/*.csv`. |
| `tables/numbers.tex` | **Generated.** Every number the prose quotes, as macros. |
| `references.bib` | Bibliography. |
| `paper.html` | **Generated.** Standalone render, for printing to PDF locally. |
| `Drift-Aware Anomaly Detection Scoring Conventions Research Paper.pdf` | **Generated.** The deposit copy, rendered from `paper.html`. |

Nothing in `tables/` is edited by hand. `main.tex` quotes `\FlagAll` rather than
`0.512`, so a sentence cannot end up disagreeing with the table beside it, and an
undefined macro breaks the build rather than leaving a stale number in place.

## Rebuilding it

Regenerate the numbers and figures from the committed results:

```bash
python scripts/make_paper_tables.py     # tables/*.tex and tables/numbers.tex
python scripts/make_paper_figures.py    # results/figures/paper/*.pdf and .png
python scripts/build_paper_html.py      # paper/paper.html
```

To regenerate the results themselves first, see the reproduction commands in the
discussion section, or the repository README.

## Getting a PDF

There is no LaTeX toolchain in this repository's environment, so there are two
paths and they produce the same content.

**Overleaf** is the one for the submission PDF. Upload `paper/` and
`results/figures/paper/` together, keeping the directory layout, and compile
`main.tex` with pdfLaTeX. The document uses only `booktabs`, `graphicx`,
`natbib`, `microtype`, `hyperref` and `caption`, all of which Overleaf has.

**`paper.html`** is the local one. Open it in a browser and print to PDF. The
figures are embedded as data URIs, so the file works on its own with no other
assets. It is generated from `main.tex` by `scripts/build_paper_html.py`, which
handles the specific LaTeX this paper uses and raises on anything it does not
recognise rather than dropping it silently.
