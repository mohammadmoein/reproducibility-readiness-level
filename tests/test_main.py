from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import main


class NormalizationTests(unittest.TestCase):
    def test_normalizes_doi_urls_and_filename_form(self) -> None:
        self.assertEqual(main.normalize_doi("https://doi.org/10.1145/3448139.3448140."), "10.1145/3448139.3448140")
        self.assertEqual(main.filename_doi_key("10.1145/3448139.3448140"), main.normalize_text("10_1145_3448139_3448140"))

    def test_preserves_raw_bibtex_entry(self) -> None:
        source = "@inproceedings{sample,\n title = {A {Title}},\n note = {keep spacing}\n}\n"
        entries = main.extract_raw_bib_entries(source)
        self.assertEqual(entries["sample"], source.rstrip("\n"))

    def test_matches_unambiguous_truncated_title_filename(self) -> None:
        folder = Path(tempfile.gettempdir())
        partition = main.Partition(2024, "long", folder, folder / "acm.bib", ())
        title = "A Very Long Paper Title Used To Verify a Truncated Filename Match Without Guessing"
        entry = main.BibEntry(partition, "citation", "inproceedings", {"title": title}, "")
        pdf = folder / "A Very Long Paper Title Used To Verify a Truncated Filename.pdf"
        self.assertEqual(main.entry_candidates(pdf, [entry]), [entry])


if __name__ == "__main__":
    unittest.main()
