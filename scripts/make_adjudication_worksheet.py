"""Generate the Stage 1 URL adjudication worksheet.

Finds every case where a URL printed in a paper resolves today but the benchmark coded the
corresponding field as 'No', and writes a worksheet with one questionnaire per case.

Usage:
    uv run python scripts/make_adjudication_worksheet.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import sqlite3

SURVEY = pathlib.Path("baseline/haim-lak/artifacts/qualtrics-survey-data.csv")
OUT = pathlib.Path("artifacts/analysis/url-adjudication-worksheet.md")
RESOLVED = {"resolved", "fetched"}


def load_survey() -> dict[str, dict[str, str]]:
    rows = list(csv.reader(SURVEY.open(encoding="utf-8-sig")))
    header = rows[1]
    idx = {c: i for i, c in enumerate(header)}
    out: dict[str, dict[str, str]] = {}
    for row in rows[3:]:
        if row and row[0].startswith("http"):
            doi = row[0].strip().lower().replace("https://doi.org/", "")
            out[doi] = {
                "open_data": row[idx["Open Data"]],
                "open_materials": row[idx["Open Materials"]],
                "data_documentation": row[idx["Data Documentation"]],
                "materials_documentation": row[idx["Materials Documentation"]],
                "readme": row[idx["README"]],
                "license": row[idx["License"]],
            }
    return out


def main() -> None:
    survey = load_survey()
    con = sqlite3.connect("data/lak.sqlite")
    con.row_factory = sqlite3.Row

    cases = []
    for r in con.execute(
        """SELECT u.id url_id, u.original_url, u.final_url, u.host_class, u.artifact_kind,
                  u.retrieval_state, u.size_bytes, u.field_name,
                  p.id paper_id, p.canonical_doi, p.title, p.collection_year
           FROM artifact_urls u JOIN papers p ON p.id = u.paper_id
           ORDER BY p.collection_year, p.canonical_doi"""
    ):
        if (r["retrieval_state"] or "") not in RESOLVED:
            continue
        doi = (r["canonical_doi"] or "").strip().lower()
        haim = survey.get(doi)
        if not haim:
            continue
        kind = (r["artifact_kind"] or "").lower()
        targets = []
        if kind in ("data", "both") and haim["open_data"] == "No":
            targets.append("open_data")
        if kind in ("code", "both") and haim["open_materials"] == "No":
            targets.append("open_materials")
        for field in targets:
            ours = con.execute(
                """SELECT verdict, quote, section FROM artifact_field_results
                   WHERE paper_id=? AND field_name=? ORDER BY experiment_id DESC LIMIT 1""",
                (r["paper_id"], field),
            ).fetchone()
            cases.append(
                {
                    "doi": r["canonical_doi"],
                    "title": r["title"],
                    "year": r["collection_year"],
                    "field": field,
                    "url": r["original_url"],
                    "final_url": r["final_url"],
                    "host": r["host_class"],
                    "kind": r["artifact_kind"],
                    "size": r["size_bytes"],
                    "haim": haim[field],
                    "ours": ours["verdict"] if ours else "(none)",
                    "quote": (ours["quote"] if ours else "") or "",
                    "section": (ours["section"] if ours else "") or "",
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    A = lines.append

    A("# Stage 1 URL adjudication worksheet")
    A("")
    A("**What this is.** Every case where a URL printed in the paper resolves today, but the")
    A("reference audit coded the corresponding field as `No`. The reference protocol allowed")
    A("15 minutes per paper and explicitly treated links that took longer to locate as absent,")
    A("so some of these `No` codes may mean *not found in time* rather than *not shared*.")
    A("")
    A("**What you are deciding.** For each case, whether the resolving artifact is the paper's")
    A("own contribution (in which case the benchmark under-detected it) or a third party's")
    A("(in which case the benchmark is correct and so is its `No`).")
    A("")
    A("**Why it matters.** If a non-trivial share are under-detections, the paper can report")
    A("that the manual audit's time budget — not the field's sharing practice — produced part")
    A("of its result. That is a quantified bias in the reference study, not merely a")
    A("disagreement with it. If most are third-party, the claim is dropped. Either outcome is")
    A("reportable; the claim must not be made without this adjudication.")
    A("")
    A("**The distinction that keeps it honest.** A link *printed in the paper* that the coders")
    A("missed is benchmark under-detection. An artifact that exists somewhere but is not")
    A("referenced in the paper is a different construct — both protocols code what the paper")
    A("claims — and is not an error. Every case below is of the first kind by construction:")
    A("the URL was extracted from the paper's own text.")
    A("")
    A(f"**Cases: {len(cases)}.** Budget about three minutes each.")
    A("")
    A("---")
    A("")
    A("## How to answer")
    A("")
    A("For each case, open the DOI and the artifact URL, then fill the five answers. Replace")
    A("the `[ ]` with `[x]`. Add a sentence of evidence — a quoted phrase from the paper is")
    A("ideal, since it is what makes the judgement checkable by someone else later.")
    A("")
    A("`Q1` is usually decisive: if the repository belongs to another group and the paper")
    A("merely uses or cites it, the benchmark's `No` is right and the case ends there.")
    A("")
    A("---")
    A("")

    for n, c in enumerate(cases, 1):
        size = f"{c['size']:,} bytes" if c["size"] else "size unknown"
        A(f"## Case {n} of {len(cases)} — `{c['field']}`")
        A("")
        A(f"**Paper.** {c['title']}")
        A(f"**DOI.** https://doi.org/{c['doi']}  ({c['year']})")
        A(f"**Artifact URL.** {c['url']}")
        if c["final_url"] and c["final_url"] != c["url"]:
            A(f"**Resolves to.** {c['final_url']}")
        A(f"**Host / kind / size.** {c['host']} / {c['kind']} / {size}")
        A("")
        A(f"**Benchmark says.** `{c['field']} = {c['haim']}`")
        A(f"**Our audit says.** `{c['ours']}`")
        if c["quote"]:
            A("")
            A(f"**Supporting quote our audit extracted** (section: {c['section'] or 'unknown'}):")
            A("")
            A(f"> {c['quote'][:400]}")
        A("")
        A("### Questions")
        A("")
        A("**Q1. Whose artifact is it?** Read the sentence in the paper that introduces the link.")
        A("First person — \"we release\", \"our code\", \"we maintain\" — means the authors'.")
        A("")
        A("- [ ] The authors' own")
        A("- [ ] A third party's (another group's repository, a public benchmark, a library)")
        A("- [ ] Cannot tell from the paper")
        A("")
        A("**Q2. Does the artifact actually contain what the field claims?** Open it.")
        A("")
        A("- [ ] Yes — it holds the data / code the field refers to")
        A("- [ ] Partly — something is there but not the analysis or dataset of the paper")
        A("- [ ] No — it is a project page, a paper copy, a tool advert, or empty")
        A("")
        A("**Q3. Could a reader have found this within a 15-minute budget?** Consider whether")
        A("the link is in a data-availability statement or buried in a footnote or reference.")
        A("")
        A("- [ ] Easily — stated plainly in an availability statement or the body")
        A("- [ ] With effort — a footnote, a reference entry, an appendix")
        A("- [ ] Only by chance — obscured, indirect, or behind a shortener")
        A("")
        A("**Q4. Was the benchmark's `No` defensible?**")
        A("")
        A("- [ ] No — the artifact is the authors' own, present, and findable. Under-detection.")
        A("- [ ] Yes — third-party, or absent, or not what the field means")
        A("- [ ] Ambiguous — defensible either way")
        A("")
        A("**Q5. Final classification.**")
        A("")
        A("- [ ] **UNDER-DETECTION** — counts toward the time-budget claim")
        A("- [ ] **BENCHMARK CORRECT** — does not count")
        A("- [ ] **AMBIGUOUS** — reported separately, counts toward neither")
        A("")
        A("**Evidence.** One sentence, ideally quoting the paper:")
        A("")
        A("> ")
        A("")
        A("---")
        A("")

    A("## Tally — fill in when finished")
    A("")
    A("| Classification | Count |")
    A("|---|---|")
    A("| Under-detection | |")
    A("| Benchmark correct | |")
    A("| Ambiguous | |")
    A(f"| **Total** | **{len(cases)}** |")
    A("")
    A("**How to read the result.** With ten cases the interval is wide whatever the split —")
    A("report the count and an exact binomial interval, not a bare percentage. Six or more")
    A("under-detections would make the time-budget claim worth stating in the paper; two or")
    A("fewer would mean dropping it. Anything between is reported as inconclusive, which is")
    A("itself honest and cheap to say.")
    A("")
    A("**Do not** generalise from these ten to the whole corpus. They are the cases where a")
    A("link resolved *and* the benchmark said `No`, not a random sample of benchmark codings.")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} with {len(cases)} cases")
    print("\nDOIs to open:")
    for c in cases:
        print(f"  https://doi.org/{c['doi']:26s} {c['field']:16s} {c['url'][:54]}")


if __name__ == "__main__":
    main()
