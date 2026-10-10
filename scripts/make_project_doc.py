"""Build docs/PROJECT_DOCUMENTATION.{md,html,pdf} from the template and results/.

    python scripts/make_project_doc.py [--gpu rtx3050_sm86] [--no-pdf]

Every number comes from ``archtune.facts.collect`` (results/ only). The PDF is printed from the
styled HTML with a headless Chromium browser (Edge or Chrome), so tables, diagrams and figures
render exactly as in the HTML.
"""

from __future__ import annotations

import argparse
import shutil
import string
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from archtune import facts  # noqa: E402

TEMPLATE = ROOT / "docs" / "templates" / "project_documentation.md.tmpl"
OUT = ROOT / "docs" / "PROJECT_DOCUMENTATION"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
:root { --ink:#16181d; --ink2:#4a4f59; --accent:#76b900; --accent2:#2a78d6; --rule:#e3e5e8; --code:#f5f6f8; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; color: var(--ink);
       font-size: 10.2pt; line-height: 1.45; margin: 0; background: #fff; }
.cover { border-left: 6px solid var(--accent); padding: 6px 0 6px 16px; margin: 0 0 14px 0; }
.cover h1 { font-size: 30pt; margin: 0; letter-spacing: -0.5px; }
.cover h2 { font-size: 15pt; margin: 2px 0 8px 0; color: var(--ink2); border: none; padding: 0; font-weight: 600; }
.cover p { margin: 4px 0; color: var(--ink2); }
.cards table { width: 100%; border-collapse: separate; border-spacing: 6px; margin: 4px 0 10px 0; table-layout: fixed; }
.cards th { background: #f1f8e6; color: #3d6b00; font-size: 19pt; padding: 8px 4px 2px 4px; border-radius: 8px 8px 0 0; border: none; }
.cards td { background: #f1f8e6; font-size: 8.4pt; color: var(--ink2); padding: 2px 8px 10px 8px;
            border-radius: 0 0 8px 8px; vertical-align: top; text-align: center; border: none; }
.cards tr:nth-child(even) td { background: #f1f8e6; }
h2 { font-size: 15pt; margin: 20px 0 8px 0; padding-bottom: 4px; border-bottom: 2px solid var(--accent); break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14px 0 6px 0; color: var(--accent2); break-after: avoid; }
p, li { margin: 4px 0; }
blockquote { margin: 8px 0; padding: 8px 12px; background: #f7f9fc; border-left: 4px solid var(--accent2); color: var(--ink); }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.8pt; break-inside: avoid; }
th { background: #eef1f4; text-align: left; font-weight: 600; }
th, td { border: 1px solid var(--rule); padding: 4px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }
code { font-family: "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: var(--code); padding: 0 3px; border-radius: 3px; }
pre { background: var(--code); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 10px;
      font-size: 7.9pt; line-height: 1.32; overflow: hidden; white-space: pre; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
.fig { margin: 10px 0 14px 0; break-inside: avoid; text-align: center; }
.fig img { max-width: 100%; max-height: 92mm; border: 1px solid var(--rule); border-radius: 4px; }
.fig p { font-size: 8.6pt; color: var(--ink2); text-align: left; }
hr { border: none; border-top: 1px solid var(--rule); margin: 12px 0; }
strong { color: #000; }
"""


def render_markdown(gpu: str) -> str:
    """Fill the template with facts computed from results/."""
    values = facts.collect(ROOT / "results", gpu)
    return string.Template(TEMPLATE.read_text(encoding="utf-8")).substitute(values)


def to_html(md_text: str) -> str:
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "md_in_html", "sane_lists"])
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>ArchTune — project documentation</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        raise RuntimeError("no Edge/Chrome found to print the PDF; open the .html and print to PDF manually")
    cmd = [exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
           f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="rtx3050_sm86")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    md_text = render_markdown(args.gpu)
    OUT.with_suffix(".md").write_text(
        "<!-- Generated by scripts/make_project_doc.py from results/; edit docs/templates/ instead. -->\n" + md_text,
        encoding="utf-8")
    html_path = OUT.with_suffix(".html")
    html_path.write_text(to_html(md_text), encoding="utf-8")
    print("wrote", OUT.with_suffix(".md"), "and", html_path)
    if not args.no_pdf:
        print_pdf(html_path, OUT.with_suffix(".pdf"))
        print("wrote", OUT.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
