"""Arrange an ACM Digital Library bulk download into the layout `lakdb ingest` expects.

ACM's bulk "Download PDFs" gives ZIP files of PDFs (named after the DOI for older
proceedings, after a truncated title for newer ones), and "Export Citations" gives BibTeX.
This script takes one folder holding any number of both, keeps only the 506 papers listed in
`docs/corpus.csv`, and writes

    <out>/<year>/<long|short>/acm.bib
    <out>/<year>/<long|short>/<original PDF name>.pdf

PDFs are matched the same way `lakdb ingest` matches them: DOI-derived filename first, then
an exact normalised title or a unique title prefix of at least 32 characters. BibTeX entries
are written in `paper_id` order, so a fresh `lakdb ingest` assigns the same paper ids as the
study (Stage 4 names its papers by id). Anything not in the corpus list, such as posters or
front matter, is skipped and reported. Nothing in the download folder is modified.

Usage:
    uv run python scripts/organize_acm_download.py --downloads ~/Downloads/acm --out ../lak_dataset
    # a small test set: all metadata, PDFs for three papers only
    uv run python scripts/organize_acm_download.py --downloads ~/Downloads/acm --out ../lak_small \
        --paper-id 1 --paper-id 2 --paper-id 69
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import filename_doi_key, normalize_text  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "docs" / "corpus.csv"
ENTRY_START = re.compile(r"^@", re.MULTILINE)
DOI_FIELD = re.compile(r"^\s*doi\s*=\s*[{\"]\s*([^}\"]+?)\s*[}\"]", re.IGNORECASE | re.MULTILINE)
KEY_FIELD = re.compile(r"^@\s*\w+\s*[{(]\s*([^,\s]+)\s*,")


@dataclass(frozen=True)
class PdfSource:
    name: str
    path: Path
    member: str | None = None  # set when the PDF sits inside a ZIP

    @property
    def key(self) -> str:
        return normalize_text(Path(self.name).stem)

    def write_to(self, target: Path) -> None:
        if self.member is None:
            shutil.copy2(self.path, target)
            return
        with zipfile.ZipFile(self.path) as archive, archive.open(self.member) as src:
            with target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def load_corpus() -> list[dict[str, str]]:
    with CORPUS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_bib_entries(root: Path) -> dict[str, str]:
    """DOI (lower case) -> raw BibTeX entry, from every .bib file under root."""
    entries: dict[str, str] = {}
    for bib in sorted(root.rglob("*.bib")):
        text = bib.read_text(encoding="utf-8")
        starts = [m.start() for m in ENTRY_START.finditer(text)] + [len(text)]
        for begin, end in zip(starts, starts[1:]):
            chunk = text[begin:end].strip()
            match = DOI_FIELD.search(chunk) or KEY_FIELD.search(chunk)
            if match:
                entries.setdefault(match.group(1).strip().lower(), chunk)
    return entries


def collect_pdfs(root: Path) -> list[PdfSource]:
    pdfs: list[PdfSource] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            pdfs.append(PdfSource(path.name, path))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    name = Path(info.filename).name
                    if name.lower().endswith(".pdf"):
                        pdfs.append(PdfSource(name, path, info.filename))
    return pdfs


def match_pdf(row: dict[str, str], pdfs: list[PdfSource]) -> list[PdfSource]:
    # Bulk ZIPs name files after the full DOI; a single-article download is often named
    # after the DOI suffix alone (3448139.3448140.pdf). Both identify the paper exactly.
    doi_keys = {filename_doi_key(row["doi"]), filename_doi_key(row["doi"].split("/", 1)[-1])}
    by_doi = [p for p in pdfs if p.key in doi_keys]
    if by_doi:
        return by_doi[:1]
    title = normalize_text(row["title"])
    hits = {
        p.key: p
        for p in pdfs
        if p.key and (p.key == title or (len(p.key) >= 32 and title.startswith(p.key)))
    }
    return list(hits.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--downloads", type=Path, required=True, help="folder with ACM ZIPs and .bib exports")
    parser.add_argument("--out", type=Path, required=True, help="dataset root to create, e.g. ../lak_dataset")
    parser.add_argument(
        "--paper-id",
        type=int,
        action="append",
        help="copy the PDF of only this paper (repeatable). Every BibTeX entry is still written, "
        "so paper ids stay identical to the study; the other papers are ingested without a PDF.",
    )
    args = parser.parse_args()

    corpus = load_corpus()
    only = set(args.paper_id or [])
    bibs = collect_bib_entries(args.downloads)
    pdfs = collect_pdfs(args.downloads)

    problems: list[str] = []
    grouped: dict[tuple[str, str], list[str]] = {}
    placed = 0
    for row in corpus:
        folder = args.out / row["year"] / row["kind"]
        folder.mkdir(parents=True, exist_ok=True)
        label = f"paper {row['paper_id']} {row['year']}/{row['kind']} {row['doi']}"
        entry = bibs.get(row["doi"].lower())
        if entry is None:
            problems.append(f"missing BibTeX: {label}")
            continue
        grouped.setdefault((row["year"], row["kind"]), []).append(entry)
        if only and int(row["paper_id"]) not in only:
            continue
        hits = match_pdf(row, pdfs)
        if len(hits) != 1:
            problems.append(f"{'missing' if not hits else 'ambiguous'} PDF: {label}")
            continue
        hits[0].write_to(folder / hits[0].name)
        placed += 1

    for (year, kind), entries in grouped.items():
        (args.out / year / kind / "acm.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")

    wanted = {row["doi"].lower() for row in corpus}
    extra = sum(1 for doi in bibs if doi not in wanted)
    print(f"corpus papers:            {len(corpus)}")
    print(f"placed with BibTeX + PDF: {placed}")
    print(f"BibTeX entries skipped (not in corpus, e.g. posters): {extra}")
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
