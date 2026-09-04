"""Render the manuscript to a standalone HTML file.

There is no LaTeX toolchain on the machine this was written on, so `main.tex`
compiles on Overleaf and this script produces the local path to a PDF: open the
output in a browser and print to PDF. Both come from the same source, so the two
cannot say different things.

This is a converter for the LaTeX *this paper* uses, not a general one. It
handles the sectioning, lists, emphasis, citations, cross-references, generated
tables and figures that appear in `main.tex` and the files it includes, and it
raises on a construct it does not know rather than dropping it silently.

    python scripts/build_paper_html.py

Writes `paper/paper.html`, with figures embedded as data URIs so the file can be
mailed or uploaded on its own.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "paper.html"


# --------------------------------------------------------------------------
# Source assembly
# --------------------------------------------------------------------------

def read_tex(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def expand_inputs(text: str, base: Path, depth: int = 0) -> str:
    """Splice in \\input{...} files, so the converter sees one document."""
    if depth > 5:
        raise RuntimeError("\\input nesting too deep")

    def replace(match: re.Match) -> str:
        name = match.group(1)
        candidate = base / name
        if candidate.suffix != ".tex":
            candidate = candidate.with_suffix(".tex")
        if not candidate.exists():
            raise FileNotFoundError(f"\\input{{{name}}} -> {candidate} not found")
        return expand_inputs(read_tex(candidate), base, depth + 1)

    return re.sub(r"\\input\{([^}]+)\}", replace, text)


def collect_macros(text: str) -> dict[str, str]:
    """Pick up \\newcommand definitions that take no arguments.

    These are the generated numbers. Definitions with arguments (\\ci) are
    handled structurally further down.
    """
    macros: dict[str, str] = {}
    # One level of nested braces is allowed in the value, so a number formatted
    # as 17{,}160 still resolves instead of quietly failing to match and leaving
    # the macro name sitting in the rendered text.
    pattern = r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}"
    for match in re.finditer(pattern, text):
        macros[match.group(1)] = match.group(2).replace("{", "").replace("}", "")
    return macros


# --------------------------------------------------------------------------
# Inline conversion
# --------------------------------------------------------------------------

# Placeholder for a LaTeX double-hyphen guard while the rest of the inline
# substitutions run. Chosen so it cannot occur in the manuscript.
DASH_GUARD = "\x00dash\x00"

# Filled in by main() before any conversion runs. LaTeX resolves these itself;
# without them the HTML would show raw keys like "fig:ranking" and
# "kim2022rigorous" where the PDF shows "Figure 1" and "Kim et al. (2022)".
LABEL_NUMBERS: dict[str, str] = {}
CITATIONS: dict[str, str] = {}


def build_label_numbers(text: str) -> dict[str, str]:
    """Number figures and tables in the order they are defined.

    Sections are left unnumbered and render as a plain link, because the paper
    refers to them by name in the surrounding sentence anyway.
    """
    counters = {"fig": 0, "tab": 0}
    numbers: dict[str, str] = {}
    for match in re.finditer(r"\\label\{([^}]*)\}", text):
        label = match.group(1)
        prefix = label.split(":")[0]
        if prefix in counters:
            counters[prefix] += 1
            numbers[label] = str(counters[prefix])
    return numbers


def build_citations(path: Path) -> dict[str, str]:
    """Author-year strings for each bib key, e.g. "Kim et al. (2022)"."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for match in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, flags=re.S):
        key = match.group(1).strip()
        fields = {
            m.group(1).lower(): m.group(2)
            for m in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}(?:,|\s*$)", match.group(2), flags=re.S)
        }
        authors = [a.strip() for a in fields.get("author", "").split(" and ") if a.strip()]
        surnames = []
        for author in authors:
            name = author.split(",")[0] if "," in author else author.split()[-1]
            for tex, char in ACCENTS.items():
                name = name.replace(tex, char)
            surnames.append(name.replace("{", "").replace("}", "").replace("\\", ""))
        if not surnames:
            label = key
        elif len(surnames) == 1:
            label = surnames[0]
        elif len(surnames) == 2:
            label = f"{surnames[0]} and {surnames[1]}"
        else:
            label = f"{surnames[0]} et al."
        out[key] = f"{label} ({fields.get('year', 'n.d.').strip()})"
    return out

ACCENTS = {
    r"\~{a}": "ã", r"\v{Z}": "Ž", r"\.{e}": "ė", r"\`{a}": "à",
    r"\"{o}": "ö", r"\"{a}": "ä", r"\'{e}": "é", r"\'{E}": "É",
    r"\v{s}": "š", r"\"{e}": "ë",
}


def inline(text: str, macros: dict[str, str]) -> str:
    """Convert the inline LaTeX this paper uses into HTML."""
    # Escape HTML first, then reintroduce the markup we generate ourselves.
    text = html.escape(text, quote=False)

    for tex, char in ACCENTS.items():
        text = text.replace(html.escape(tex, quote=False), char)

    # `-{}-` is LaTeX's guard against a double hyphen becoming an en-dash, which
    # command-line flags need. Park it so neither the brace handling nor the
    # en-dash substitution below touches it, and restore it at the end.
    text = text.replace("-{}-", DASH_GUARD)

    # Confidence intervals: \ci{lo}{hi}
    text = re.sub(
        r"\\ci\{([^}]*)\}\{([^}]*)\}",
        r'<span class="ci">[\1,&nbsp;\2]</span>',
        text,
    )

    text = re.sub(r"\\textbf\{([^{}]*)\}", r"<strong>\1</strong>", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", text)
    text = re.sub(r"\\textit\{([^{}]*)\}", r"<em>\1</em>", text)
    # One level of nesting, so \texttt{... -{}- ...} survives intact.
    text = re.sub(r"\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}", r"<code>\1</code>", text)
    text = re.sub(r"\\footnotesize\s*", "", text)

    # Citations become plain author-year-ish markers; the bibliography is
    # rendered in full at the end, so the reader can still resolve them.
    def cite(match: re.Match) -> str:
        keys = [k.strip() for k in match.group(2).split(",")]
        parenthetical = match.group(1) == "citep"
        parts = []
        for key in keys:
            shown = CITATIONS.get(key, key)
            if parenthetical:
                # Inside parentheses the year loses its own brackets.
                shown = shown.replace(" (", ", ").rstrip(")")
            parts.append(f'<a href="#bib-{key}">{shown}</a>')
        rendered = "; ".join(parts)
        opener, closer = ("(", ")") if parenthetical else ("", "")
        return f'<span class="cite">{opener}{rendered}{closer}</span>'

    text = re.sub(r"\\(citep|citet)\{([^}]*)\}", cite, text)

    def reference(match: re.Match) -> str:
        label = match.group(1)
        shown = LABEL_NUMBERS.get(label)
        if shown is None:
            return f'<a href="#{label}">this section</a>'
        return f'<a href="#{label}">{shown}</a>'

    text = re.sub(r"\\ref\{([^}]*)\}", reference, text)
    text = re.sub(r"\\label\{([^}]*)\}", "", text)
    # A tie is a non-breaking space, and "Figure~1" should not render a gap.
    text = text.replace("~", "&nbsp;")

    # Simple inline maths: only the handful of forms this paper uses.
    def maths(match: re.Match) -> str:
        body = match.group(1)
        body = body.replace(r"\%", "%")
        body = re.sub(r"([A-Za-z])\^\{?([0-9n])\}?", r"\1<sup>\2</sup>", body)
        body = re.sub(r"([A-Za-z])_\{?([0-9a-z])\}?", r"\1<sub>\2</sub>", body)
        body = body.replace(r"\alpha", "α").replace(r"\emph", "")
        body = body.replace("<", "&lt;").replace(">", "&gt;")
        return f'<span class="math">{body}</span>'

    text = re.sub(r"\$([^$]*)\$", maths, text)

    # Remaining escapes and no-argument macros. Longest name first: LaTeX reads
    # a control sequence to its end, but plain string replacement does not, so
    # substituting \MoverPoint before \MoverPointRank would turn the latter into
    # "0.477Rank". Sorting by length is what makes the two agree.
    for name in sorted(macros, key=len, reverse=True):
        value = macros[name]
        text = text.replace(f"\\{name}{{}}", value).replace(f"\\{name}", value)

    text = text.replace(r"\%", "%").replace(r"\&", "&amp;").replace(r"\_", "_")
    text = text.replace(r"\\", "")
    text = text.replace("``", "&ldquo;").replace("''", "&rdquo;")
    text = text.replace("--", "&ndash;")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(DASH_GUARD, "--")


# --------------------------------------------------------------------------
# Block conversion
# --------------------------------------------------------------------------

def convert_tabular(body: str, macros: dict[str, str]) -> str:
    """Turn a booktabs tabular into an HTML table."""
    rows_html: list[str] = []
    header_done = False
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(r"\begin{tabular}") or line.startswith(r"\end{tabular}"):
            continue
        if line.startswith((r"\toprule", r"\bottomrule")):
            continue
        if line.startswith(r"\midrule"):
            header_done = True
            rows_html.append('<tr class="rule"><td colspan="99"></td></tr>')
            continue
        if line.startswith(r"\cmidrule"):
            continue
        line = re.sub(r"\\\\\s*$", "", line)
        if line.startswith(r"\multicolumn"):
            continue
        cells = [c.strip() for c in line.split("&")]
        cells = [re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", c) for c in cells]
        tag = "td" if header_done else "th"
        rendered = "".join(f"<{tag}>{inline(c, macros)}</{tag}>" for c in cells)
        rows_html.append(f"<tr>{rendered}</tr>")
    return '<table class="data">' + "".join(rows_html) + "</table>"


def embed_figure(name: str) -> str | None:
    for candidate in (
        PAPER / name,
        ROOT / "results" / "figures" / "paper" / f"{Path(name).stem}.png",
        ROOT / "results" / "figures" / f"{Path(name).stem}.png",
    ):
        if candidate.exists() and candidate.suffix.lower() == ".png":
            data = base64.b64encode(candidate.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{data}"
    return None


def convert_body(text: str, macros: dict[str, str]) -> str:
    """Walk the document body and emit HTML."""
    out: list[str] = []
    i = 0
    lines = text.splitlines()
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            joined = " ".join(paragraph).strip()
            if joined:
                out.append(f"<p>{inline(joined, macros)}</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("%") or not stripped:
            flush()
            i += 1
            continue

        heading = re.match(r"\\(section|subsection|paragraph)\*?\{(.+?)\}(.*)$", stripped)
        if heading:
            flush()
            level = {"section": "h2", "subsection": "h3", "paragraph": "h4"}[heading.group(1)]
            label = re.search(r"\\label\{([^}]*)\}", stripped)
            anchor = f' id="{label.group(1)}"' if label else ""
            out.append(f"<{level}{anchor}>{inline(heading.group(2), macros)}</{level}>")
            rest = heading.group(3).strip()
            if rest:
                paragraph.append(rest)
            i += 1
            continue

        env = re.match(r"\\begin\{(\w+\*?)\}", stripped)
        if env:
            flush()
            name = env.group(1)
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(f"\\end{{{name}}}"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(convert_environment(name, "\n".join(block), macros))
            continue

        if stripped.startswith(r"\input{"):
            flush()
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush()
    return "\n".join(out)


def convert_environment(name: str, block: str, macros: dict[str, str]) -> str:
    if name == "abstract":
        return f'<div class="abstract"><h2>Abstract</h2><p>{inline(block, macros)}</p></div>'

    if name in ("itemize", "enumerate"):
        tag = "ul" if name == "itemize" else "ol"
        items = [x.strip() for x in re.split(r"\\item\b", block) if x.strip()]
        rendered = "".join(f"<li>{inline(x, macros)}</li>" for x in items)
        return f"<{tag}>{rendered}</{tag}>"

    if name == "description":
        items = [x.strip() for x in re.split(r"\\item\b", block) if x.strip()]
        rendered = []
        for item in items:
            match = re.match(r"\[([^\]]*)\]\s*(.*)$", item, flags=re.S)
            if match:
                rendered.append(
                    f"<dt>{inline(match.group(1), macros)}</dt>"
                    f"<dd>{inline(match.group(2), macros)}</dd>"
                )
            else:
                rendered.append(f"<dd>{inline(item, macros)}</dd>")
        return "<dl>" + "".join(rendered) + "</dl>"

    if name in ("table", "table*", "figure", "figure*"):
        label = re.search(r"\\label\{([^}]*)\}", block)
        anchor = f' id="{label.group(1)}"' if label else ""
        caption = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|$)", block, flags=re.S)
        number = LABEL_NUMBERS.get(label.group(1)) if label else None
        kind = "Table" if (label and label.group(1).startswith("tab")) else "Figure"
        prefix = f"<strong>{kind}&nbsp;{number}.</strong> " if number else ""
        caption_html = (
            f'<figcaption>{prefix}{inline(caption.group(1), macros)}</figcaption>'
            if caption else ""
        )
        inner = ""
        include = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", block)
        if include:
            src = embed_figure(include.group(1))
            inner = f'<img src="{src}" alt="">' if src else "<p><em>figure missing</em></p>"

        # The tabular usually arrives already spliced in, because expand_inputs
        # runs over the whole document first. Handle both that and a surviving
        # \input, so the converter does not depend on the order of the two.
        tabular = re.search(
            r"\\begin\{tabular\}.*?\\end\{tabular\}", block, flags=re.S
        )
        if tabular:
            inner = convert_tabular(tabular.group(0), macros)
        else:
            table_input = re.search(r"\\input\{([^}]*)\}", block)
            if table_input:
                path = PAPER / (table_input.group(1) + ".tex")
                inner = (
                    convert_tabular(read_tex(path), macros)
                    if path.exists()
                    else "<p><em>table missing</em></p>"
                )

        wrapper = "table-wrap" if inner.startswith("<table") else ""
        return f'<figure{anchor} class="{wrapper}">{inner}{caption_html}</figure>'

    if name == "tabular":
        return convert_tabular(block, macros)

    if name == "quote":
        # Used for the block of reproduction commands, which are separated by
        # LaTeX line breaks rather than paragraphs.
        lines = [line.strip() for line in block.split(r"\\") if line.strip()]
        rendered = "<br>".join(inline(line, macros) for line in lines)
        return f'<blockquote>{rendered}</blockquote>'

    if name in ("document", "center"):
        return convert_body(block, macros)

    raise ValueError(f"unhandled LaTeX environment: {name}")


def render_bibliography() -> str:
    """Render references.bib in the order the entries appear."""
    path = PAPER / "references.bib"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    entries = []
    for match in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, flags=re.S):
        key = match.group(1).strip()
        fields = dict(
            (m.group(1).lower(), m.group(2).strip())
            for m in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}(?:,|\s*$)", match.group(2), flags=re.S)
        )
        author = fields.get("author", "").replace(" and ", ", ")
        for tex, char in ACCENTS.items():
            author = author.replace(tex, char)
        author = author.replace("{", "").replace("}", "").replace("\\", "")
        title = fields.get("title", "").replace("{", "").replace("}", "")
        venue = fields.get("journal") or fields.get("booktitle") or fields.get("note", "")
        venue = venue.replace("{", "").replace("}", "").replace("--", "&ndash;")
        year = fields.get("year", "")
        entries.append(
            f'<li id="bib-{key}"><span class="bibkey">{key}</span> '
            f"{html.escape(author)}. {html.escape(title)}. "
            f"<em>{venue}</em>, {year}.</li>"
        )
    return "<h2>References</h2><ol class=\"bib\">" + "".join(entries) + "</ol>"


STYLE = """
:root { color-scheme: light; }
body { margin: 0; background: #f6f6f4; color: #14140f;
       font: 16px/1.62 Georgia, 'Iowan Old Style', 'Times New Roman', serif; }
main { max-width: 46rem; margin: 0 auto; padding: 3.5rem 1.5rem 6rem;
       background: #fdfdfc; }
h1 { font-size: 1.85rem; line-height: 1.25; margin: 0 0 .4rem; letter-spacing: -.01em; }
h2 { font-size: 1.22rem; margin: 2.4rem 0 .7rem; border-bottom: 1px solid #e2e1db;
     padding-bottom: .25rem; }
h3 { font-size: 1.04rem; margin: 1.7rem 0 .5rem; }
h4 { font-size: .97rem; margin: 1.2rem 0 .3rem; font-style: italic; font-weight: 600; }
p { margin: 0 0 .85rem; }
.byline { color: #52514e; font-size: .95rem; margin-bottom: 1.6rem; }
.byline code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; }
.abstract { background: #f2f1ec; border-left: 3px solid #2a78d6;
            padding: 1rem 1.2rem; margin: 1.6rem 0 2rem; }
.abstract h2 { margin: 0 0 .5rem; border: 0; font-size: 1rem;
               text-transform: uppercase; letter-spacing: .06em; }
.abstract p { margin: 0; font-size: .95rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86em;
       background: #eeede8; padding: .07em .3em; border-radius: 3px; }
figure { margin: 1.6rem 0; }
figure img { width: 100%; height: auto; display: block; }
figure.table-wrap { overflow-x: auto; }
figcaption { font-size: .85rem; color: #52514e; margin-top: .5rem; }
table.data { border-collapse: collapse; width: 100%; font-size: .82rem;
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
table.data th { text-align: left; font-weight: 600; padding: .35rem .5rem;
                border-bottom: 1.5px solid #14140f; }
table.data td { padding: .3rem .5rem; border-bottom: 1px solid #eeede8; }
table.data tr.rule td { padding: 0; border-bottom: 1px solid #14140f; }
.ci { color: #52514e; font-size: .82em; }
.cite a { color: #2a5fa8; text-decoration: none; }
.cite a:hover { text-decoration: underline; }
blockquote { margin: 1rem 0; padding: .7rem 1rem; background: #f2f1ec;
             border-left: 3px solid #d8d7d2; font-family: ui-monospace,
             SFMono-Regular, Menlo, monospace; font-size: .82rem; }
dl { margin: .6rem 0 1rem; }
dt { font-weight: 600; margin-top: .5rem; }
dd { margin: .1rem 0 .1rem 1.4rem; }
ol.bib { font-size: .88rem; padding-left: 1.4rem; }
ol.bib li { margin-bottom: .45rem; }
.bibkey { display: none; }
.math { font-family: 'Times New Roman', serif; font-style: italic; }
@media print {
  body { background: #fff; }
  main { max-width: none; padding: 0; }
  h2 { break-after: avoid; }
  figure, table { break-inside: avoid; }
}
"""


def main() -> int:
    source = read_tex(PAPER / "main.tex")
    source = expand_inputs(source, PAPER)
    macros = collect_macros(source)

    LABEL_NUMBERS.update(build_label_numbers(source))
    CITATIONS.update(build_citations(PAPER / "references.bib"))

    title = re.search(r"\\title\{(.*?)\}\s*\n", source, flags=re.S)
    title_text = inline(title.group(1).replace(r"\\", " "), macros) if title else "Paper"

    body_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", source, flags=re.S)
    if not body_match:
        raise ValueError("no document body found")
    body = body_match.group(1)
    body = body.replace(r"\maketitle", "")
    body = re.sub(r"\\bibliography\{[^}]*\}", "", body)
    body = re.sub(r"\\newcommand.*", "", body)

    html_body = convert_body(body, macros)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_text}</title>
<style>{STYLE}</style>
</head><body><main>
<h1>{title_text}</h1>
<p class="byline">Dhruvil Shah &middot; <code>contactshahdhruvil@gmail.com</code> &middot; September 2026</p>
{html_body}
{render_bibliography()}
</main></body></html>
"""
    OUT.write_text(page, encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"Wrote paper/paper.html ({size:.0f} KB)")
    print("Open it in a browser and print to PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
