#!/usr/bin/env python3
"""Export Official Documentation chapters to GitHub Wiki–ready Markdown.

GitHub Wikis use a separate git repository (<owner>/<repo>.wiki.git). Pages are
flat ``*.md`` files at the wiki root; ``Home.md`` is the landing page.

**Push workflow**

1. Enable the wiki on the GitHub repository (Settings or create first page on GitHub).
2. Clone the wiki repo::

       git clone https://github.com/<owner>/<repo>.wiki.git
       cd <repo>.wiki

3. Copy all files from ``Official Documentation/github_wiki/`` into this clone
   (replace ``Home.md`` and chapter files).
4. Commit and push::

       git add -A
       git commit -m "Sync wiki from Official Documentation export"
       git push

**Internal links**

Wiki page links must omit the ``.md`` extension or GitHub may serve raw Markdown.
This script rewrites ``](NN_Chapter_Name.md)`` to ``](NN_Chapter_Name)``.

**Repository links**

``../../MAPseq_wizard/README.md`` in the source is rewritten to a ``blob`` URL
under the main repo. Pass ``--repo-url`` (e.g. ``https://github.com/org/mapseq_processing_Jacobs``)
and optional ``--branch`` (default: ``main``). If omitted, a placeholder URL is used.

Regenerate the export::

    python3 Official Documentation/scripts/export_github_wiki.py
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# Wiki page stem (filename without .md) -> one-line blurb for Home.md
CHAPTER_BLURBS: list[tuple[str, str, str]] = [
    ("01_Introduction", "Chapter 1: Introduction", "Overview, key concepts, pipeline architecture, and research questions"),
    ("02_Installation_Setup", "Chapter 2: Installation and Setup", "System requirements, installation methods, and environment setup"),
    ("03_Data_Preparation", "Chapter 3: Data Preparation", "Input data format, preprocessing workflow, and quality control"),
    ("04_Main_Processing_Pipeline", "Chapter 4: Main Processing Pipeline", "Command-line interface, processing steps, and output structure"),
    ("05_Statistical_Methods", "Chapter 5: Statistical Methods", "N₀ estimation, binomial testing, multiple testing correction, and effect sizes"),
    ("06_Probability_Models", "Chapter 6: Probability Models", "Uniform, region-specific, correlated, and additional probability models"),
    ("07_Helper_Scripts", "Chapter 7: Helper Scripts", "Cross-age analysis scripts, execution order, and dependencies"),
    ("08_Output_Files_Interpretation", "Chapter 8: Output Files and Structure", "Output paths, file formats, and column reference"),
    ("09_Code_Review", "Chapter 9: Code Review", "Architecture, key functions, data flow, and implementation details"),
    ("10_Mathematical_Functions", "Chapter 10: Mathematical Functions Reference", "Formulas with code references and interpretations"),
    ("11_Stability_Analysis", "Chapter 11: Stability Analysis", "Generic stability metrics framework (not study-specific results)"),
    ("12_Troubleshooting_Best_Practices", "Chapter 12: Troubleshooting and Best Practices", "Common errors, parameter selection, quality control, and best practices"),
    ("13_References_Appendices", "Chapter 13: References and Appendices", "Code references, notation glossary, statistical test reference, and quick reference"),
    ("14_Experimental_Features", "Chapter 14: Experimental Features", "GUI wizards, maintainer batch scripts; use bash + command file for production"),
    ("15_Trajectory_Results_Interpretation", "Chapter 15: Trajectory Results", "Helper 15 outputs and methods (file reference; no bundled results)"),
    ("16_Cross_Anchor_Comparative_Analysis", "Chapter 16: Cross-Anchor Analysis", "Conceptual workflow for comparing anchor configurations"),
]

# Same-directory chapter cross-links: (NN_Name.md) or (NN_Name.md#anchor)
_CHAPTER_LINK = re.compile(
    r"\((?P<base>\d{2}_[A-Za-z0-9_]+)\.md(?P<frag>#[^)\s]*)?\)"
)

_PLACEHOLDER_REPO = "https://github.com/OWNER/REPO"


def _blob_url(repo_url: str, branch: str, path_in_repo: str) -> str:
    base = repo_url.rstrip("/")
    path = path_in_repo.lstrip("/")
    return f"{base}/blob/{branch}/{path}"


def _rewrite_segment(segment: str, mapseq_readme_url: str) -> str:
    s = _CHAPTER_LINK.sub(r"(\g<base>\g<frag>)", segment)
    s = s.replace("../../MAPseq_wizard/README.md", mapseq_readme_url)
    return s


def transform_markdown(body: str, mapseq_readme_url: str) -> str:
    """Rewrite links only outside fenced code blocks (``` ... ```)."""
    parts = body.split("```")
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(_rewrite_segment(part, mapseq_readme_url))
        else:
            out.append("```" + part + "```")
    return "".join(out)


def render_home_md(repo_url: str, branch: str) -> str:
    lines = [
        "# MAPseq Processing Pipeline Documentation",
        "",
        "Comprehensive documentation for the MAPseq (Multiplexed Analysis of Projections by Sequencing) processing pipeline, including installation, usage, statistical methods, code review, and mathematical functions.",
        "",
        "This wiki is generated from the main repository under `Official Documentation/chapters/`. To refresh, run `Official Documentation/scripts/export_github_wiki.py` and push the `github_wiki/` contents to this wiki repository.",
        "",
        "## Documentation chapters",
        "",
    ]
    for n, (stem, title, blurb) in enumerate(CHAPTER_BLURBS, start=1):
        lines.append(f"{n}. **[{title}]({stem})** — {blurb}")
    lines.extend(
        [
            "",
            "## Quick start",
            "",
            "### For new users",
            "",
            "1. Start with [Chapter 1: Introduction](01_Introduction) for overview",
            "2. Follow [Chapter 2: Installation and Setup](02_Installation_Setup) for installation",
            "3. Review [Chapter 3: Data Preparation](03_Data_Preparation) for data format",
            "4. **Run the pipeline**: Edit `all_commands.txt` (or `all_commands_all-parameters.txt`) to match your paths and samples, then from the repository root run `./run_commands.sh`. See [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline) for details.",
            "",
            "### For understanding methods",
            "",
            "1. Read [Chapter 5: Statistical Methods](05_Statistical_Methods) for the statistical framework",
            "2. Review [Chapter 6: Probability Models](06_Probability_Models) for model details",
            "3. Consult [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions) for formulas",
            "",
            "### For outputs and file layout",
            "",
            "1. See [Chapter 8: Output Files and Structure](08_Output_Files_Interpretation) for paths and columns",
            "2. Check [Chapter 7: Helper Scripts](07_Helper_Scripts) for helper outputs and run order",
            "3. Review [Chapter 11: Stability Analysis](11_Stability_Analysis) for a generic metrics framework",
            "4. Advanced: [Chapter 15](15_Trajectory_Results_Interpretation), [Chapter 16](16_Cross_Anchor_Comparative_Analysis)",
            "",
            "### For developers",
            "",
            "1. Review [Chapter 9: Code Review](09_Code_Review) for architecture",
            "2. Consult [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions) for implementations",
            "3. Check [Chapter 13: References and Appendices](13_References_Appendices) for code references",
            "",
            "## Main repository",
            "",
            f"Pipeline source and HTML documentation: `{repo_url}` (branch `{branch}`).",
            "",
            "### MAPseq_wizard (experimental)",
            "",
            f"See the [MAPseq_wizard README]({_blob_url(repo_url, branch, 'MAPseq_wizard/README.md')}) in the main repository.",
            "",
            "---",
            "",
            "*Documentation version: April 2026 (wiki export).*",
            "",
        ]
    )
    return "\n".join(lines)


def export_wiki(
    chapters_dir: Path,
    output_dir: Path,
    repo_url: str,
    branch: str,
) -> None:
    mapseq_readme_url = _blob_url(repo_url, branch, "MAPseq_wizard/README.md")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    stems_on_disk = {p.stem for p in chapters_dir.glob("*.md")}
    expected = {t[0] for t in CHAPTER_BLURBS}
    missing = expected - stems_on_disk
    if missing:
        raise SystemExit(f"Missing chapter files for stems: {sorted(missing)}")
    extra = stems_on_disk - expected
    if extra:
        raise SystemExit(
            f"Chapter .md files not listed in CHAPTER_BLURBS (update the script): {sorted(extra)}"
        )

    for path in sorted(chapters_dir.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        transformed = transform_markdown(body, mapseq_readme_url)
        (output_dir / path.name).write_text(transformed, encoding="utf-8")

    (output_dir / "Home.md").write_text(render_home_md(repo_url, branch), encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_chapters = script_dir.parent / "chapters"
    default_out = script_dir.parent / "github_wiki"

    parser = argparse.ArgumentParser(
        description="Export Official Documentation chapters to GitHub Wiki–ready Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=default_chapters,
        help=f"Directory of chapter .md files (default: {default_chapters})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Wiki staging directory (default: {default_out})",
    )
    parser.add_argument(
        "--repo-url",
        default=_PLACEHOLDER_REPO,
        help=f"Main GitHub repository URL without trailing path (default: {_PLACEHOLDER_REPO})",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch name for blob links into the main repo (default: main)",
    )
    args = parser.parse_args()
    chapters_dir = args.chapters_dir.resolve()
    if not chapters_dir.is_dir():
        raise SystemExit(f"Chapters directory not found: {chapters_dir}")

    export_wiki(chapters_dir, args.output_dir.resolve(), args.repo_url.rstrip("/"), args.branch)
    print(f"Wrote wiki export to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()