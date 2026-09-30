"""Phase 4: merge field sources, align with Haim, OddPub baseline, report."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agreement import _clopper_pearson, _f1, _safe_div
from artifact import (
    CLAIM_FIELDS,
    ensure_schema,
    get_experiment,
    load_haim_fields,
    utc_now,
)
from screening import normalize_doi

# Keyword lists follow OddPub (Riedel, Kip, & Bobrov, 2020; quest-bih/oddpub).
# Python reimplementation of the published regex detectors; not a line-for-line port.
ODDPUB_OPEN_DATA = (
    r"\bdata (?:are|is) available\b",
    r"\bdata availability\b",
    r"\bsupplementary data\b",
    r"\bdeposited (?:in|at|on)\b",
    r"\bfigshare\b",
    r"\bzenodo\b",
    r"\bdryad\b",
    r"\bosf\.io\b",
    r"\bopen science framework\b",
    r"\bdataverse\b",
    r"\baccession number\b",
    r"\bgeo series\b",
    r"\barrayexpress\b",
)
ODDPUB_OPEN_CODE = (
    r"github\.com",
    r"gitlab\.com",
    r"bitbucket\.org",
    r"\bsource code\b",
    r"\bcode (?:is|are) available\b",
    r"\banalysis (?:code|scripts?)\b",
    r"\brepository\b",
)
MACHINE_READABLE_STATES = {"resolved", "fetched"}
DATA_KINDS = {"data", "both"}


def oddpub_detect(text: str) -> dict[str, bool]:
    return {
        "open_data": any(re.search(pattern, text, re.I) for pattern in ODDPUB_OPEN_DATA),
        "open_code": any(re.search(pattern, text, re.I) for pattern in ODDPUB_OPEN_CODE),
    }


async def merge_fields(catalog: Any, claim_experiment: Any) -> int:
    cursor = await catalog.db.execute(
        "SELECT DISTINCT paper_id FROM artifact_field_results WHERE experiment_id=?",
        (claim_experiment["id"],),
    )
    papers = [row["paper_id"] for row in await cursor.fetchall()]
    docs_by_paper: dict[tuple[int, str], dict[str, Any]] = {}
    cursor = await catalog.db.execute(
        "SELECT r.paper_id,r.field_name,r.verdict,r.quote,r.section,r.urls_json "
        "FROM artifact_field_results r JOIN artifact_experiments e ON e.id=r.experiment_id "
        "WHERE e.stage='docs' AND e.claim_experiment_id=?",
        (claim_experiment["id"],),
    )
    for row in await cursor.fetchall():
        docs_by_paper[(row["paper_id"], row["field_name"])] = dict(row)
    snapshot_flags: dict[int, dict[str, bool]] = {}
    cursor = await catalog.db.execute(
        "SELECT s.paper_id,s.manifest_json FROM artifact_snapshots s"
    )
    for row in await cursor.fetchall():
        manifest = json.loads(row["manifest_json"] or "{}")
        inventory = manifest.get("inventory") or {}
        flags = snapshot_flags.setdefault(row["paper_id"], {"readme": False, "license": False})
        flags["readme"] = flags["readme"] or bool(inventory.get("readme"))
        flags["license"] = flags["license"] or bool(inventory.get("license"))
    degradation: dict[int, list[str]] = {}
    cursor = await catalog.db.execute(
        "SELECT paper_id,host_class,retrieval_state,artifact_kind FROM artifact_urls"
    )
    for row in await cursor.fetchall():
        if row["retrieval_state"] in {"unreachable", "error", "robots_disallowed"}:
            degradation.setdefault(row["paper_id"], []).append(
                row["artifact_kind"] or row["host_class"] or "unknown"
            )
    count = 0
    now = utc_now()
    for paper_id in papers:
        cursor = await catalog.db.execute(
            "SELECT field_name,verdict,quote,section,urls_json FROM artifact_field_results "
            "WHERE experiment_id=? AND paper_id=?",
            (claim_experiment["id"], paper_id),
        )
        claims = {row["field_name"]: dict(row) for row in await cursor.fetchall()}
        retrieved = paper_id in snapshot_flags
        for field in CLAIM_FIELDS:
            row = claims.get(field)
            if row is None:
                continue
            source = "claim"
            verdict = row["verdict"]
            quote = row["quote"]
            section = row["section"]
            urls = row["urls_json"]
            if field in {"data_documentation", "materials_documentation"} and (
                paper_id,
                field,
            ) in docs_by_paper:
                docs = docs_by_paper[(paper_id, field)]
                verdict = docs["verdict"]
                quote = docs["quote"]
                section = docs["section"]
                urls = docs["urls_json"]
                source = "docs"
            if field in {"readme", "license"} and retrieved:
                verdict = "Yes" if snapshot_flags[paper_id][field] else "No"
                source = "resolver"
                quote = quote or f"resolver inventory: {field}={verdict}"
            await catalog.db.execute(
                "INSERT INTO artifact_merged_fields(claim_experiment_id,paper_id,field_name,verdict,quote,section,source,urls_json,merged_at) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(claim_experiment_id,paper_id,field_name) DO UPDATE SET "
                "verdict=excluded.verdict,quote=excluded.quote,section=excluded.section,source=excluded.source,"
                "urls_json=excluded.urls_json,merged_at=excluded.merged_at",
                (
                    claim_experiment["id"],
                    paper_id,
                    field,
                    verdict,
                    quote,
                    section,
                    source,
                    urls,
                    now,
                ),
            )
            count += 1
        deg = sorted(set(degradation.get(paper_id, [])))
        await catalog.db.execute(
            "INSERT INTO artifact_merged_fields(claim_experiment_id,paper_id,field_name,verdict,quote,section,source,urls_json,merged_at) "
            "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(claim_experiment_id,paper_id,field_name) DO UPDATE SET "
            "verdict=excluded.verdict,quote=excluded.quote,source=excluded.source,merged_at=excluded.merged_at",
            (
                claim_experiment["id"],
                paper_id,
                "reference_degradation",
                ",".join(deg) if deg else "none",
                "resolver states for this paper",
                "resolver",
                "resolver",
                "[]",
                now,
            ),
        )
        count += 1
    await catalog.db.commit()
    return count


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    labels = sorted({expected for expected, _ in pairs} | {predicted for _, predicted in pairs})
    matrix = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1
    per_class: dict[str, Any] = {}
    for label in labels:
        tp = matrix[label].get(label, 0)
        fp = sum(matrix[other].get(label, 0) for other in labels if other != label)
        fn = sum(matrix[label].get(other, 0) for other in labels if other != label)
        support = tp + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, support)
        per_class[label] = {
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "recall_95pct_exact": _clopper_pearson(tp, support),
            "f1": _f1(precision, recall),
        }
    total = len(pairs)
    correct = sum(1 for expected, predicted in pairs if expected == predicted)
    expected_counts = Counter(expected for expected, _ in pairs)
    predicted_counts = Counter(predicted for _, predicted in pairs)
    expected_agreement = (
        sum(expected_counts[label] * predicted_counts[label] for label in labels) / (total * total)
        if total
        else None
    )
    observed = _safe_div(correct, total)
    kappa = None
    if observed is not None and expected_agreement is not None and expected_agreement < 1:
        kappa = (observed - expected_agreement) / (1 - expected_agreement)
    f1s = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    return {
        "evaluated": total,
        "correct": correct,
        "accuracy": observed,
        "cohen_kappa": kappa,
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else None,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


async def _repeated_run_agreement(catalog: Any, experiment_id: int) -> float | None:
    cursor = await catalog.db.execute(
        "SELECT paper_id,parsed_result_json FROM artifact_runs "
        "WHERE experiment_id=? AND status='succeeded' ORDER BY paper_id,repetition_index",
        (experiment_id,),
    )
    grouped: dict[int, list[str]] = {}
    for row in await cursor.fetchall():
        if not row["parsed_result_json"]:
            continue
        payload = json.loads(row["parsed_result_json"])
        key = json.dumps({field: payload[field]["verdict"] for field in payload}, sort_keys=True)
        grouped.setdefault(row["paper_id"], []).append(key)
    if not grouped:
        return None
    unanimous = sum(1 for values in grouped.values() if len(set(values)) == 1)
    return unanimous / len(grouped)


HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<body", b"<script")
DATA_FILE_EXT = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet", ".sav",
                 ".dta", ".rds", ".rdata", ".arff", ".db", ".sqlite", ".npy", ".pkl",
                 ".mat", ".zip", ".gz", ".tar", ".xml", ".sql", ".txt"}
CODE_FILE_EXT = {".py", ".ipynb", ".r", ".rmd", ".js", ".java", ".sh", ".jl", ".m", ".cpp"}
# A paper's own claim, per Stage 1. Anything else is somebody else's artifact.
DATA_CLAIMED = {"Yes", "Open Data"}
CODE_CLAIMED = {"Full", "Partial"}


def _snapshot_holds(store_path: str | None, exts: set[str], min_bytes: int = 0) -> bool:
    """Whether a snapshot contains a real file of the wanted sort.

    The census used to count any snapshot with `size_bytes > 0`. Bytes were never inspected,
    and most "retrieved datasets" turned out to be saved HTML landing pages. A transfer is
    not a retrieval.
    """
    if not store_path:
        return False
    content = Path(store_path) / "content"
    if not content.exists():
        return False
    for item in content.rglob("*"):
        if not item.is_file():
            continue
        if item.suffix.lower() in exts:
            if min_bytes and item.stat().st_size < min_bytes:
                continue
            return True
        if min_bytes:
            continue  # a size floor means only the listed extensions count
        try:
            head = item.open("rb").read(2048).lower()
        except OSError:
            continue
        if not any(marker in head for marker in HTML_MARKERS):
            return True
    return False


async def executable_candidates(catalog: Any) -> dict[str, Any]:
    cursor = await catalog.db.execute(
        "SELECT p.collection_year year,u.paper_id,u.retrieval_state,u.artifact_kind,"
        "(SELECT s.store_path FROM artifact_snapshots s WHERE s.url_id=u.id "
        " ORDER BY s.size_bytes DESC LIMIT 1) store_path "
        "FROM artifact_urls u JOIN papers p ON p.id=u.paper_id"
    )
    rows = await cursor.fetchall()
    # Verdict gating: a URL counts only where the paper itself claims the artifact. Paper 98
    # names a third party's repository in its methods while its own verdict is No.
    cursor = await catalog.db.execute(
        "SELECT paper_id, field_name, verdict FROM artifact_merged_fields "
        "WHERE field_name IN ('open_data','open_materials')"
    )
    claims: dict[int, dict[str, str]] = {}
    for row in await cursor.fetchall():
        claims.setdefault(row["paper_id"], {})[row["field_name"]] = (row["verdict"] or "").strip()
    by_paper: dict[int, dict[str, Any]] = {}
    for row in rows:
        item = by_paper.setdefault(
            row["paper_id"], {"year": row["year"], "data": False, "code": False}
        )
        owned = claims.get(row["paper_id"], {})
        kind = row["artifact_kind"]
        if kind in DATA_KINDS and owned.get("open_data") in DATA_CLAIMED:
            if _snapshot_holds(row["store_path"], DATA_FILE_EXT):
                item["data"] = True
        if kind in {"code", "both"} and owned.get("open_materials") in CODE_CLAIMED:
            if _snapshot_holds(row["store_path"], CODE_FILE_EXT | DATA_FILE_EXT):
                item["code"] = True
            # Data shipped inside the repository is still shared data, and for most of our
            # candidates it is the only copy that survives retrieval. Require the paper to
            # claim open data as well, so a repo of pure code does not qualify.
            if owned.get("open_data") in DATA_CLAIMED and _snapshot_holds(
                row["store_path"], DATA_FILE_EXT - {".txt", ".xml", ".gz", ".zip", ".tar"},
                min_bytes=2000,
            ):
                item["data"] = True
                item["data_in_repo"] = True
    data_only = [paper_id for paper_id, item in by_paper.items() if item["data"]]
    data_and_code = [
        paper_id for paper_id, item in by_paper.items() if item["data"] and item["code"]
    ]
    by_year: dict[int, dict[str, int]] = {}
    for item in by_paper.values():
        year = int(item["year"] or 0)
        bucket = by_year.setdefault(year, {"data_reachable": 0, "data_and_code": 0})
        if item["data"]:
            bucket["data_reachable"] += 1
        if item["data"] and item["code"]:
            bucket["data_and_code"] += 1
    return {
        "criteria": {
            "executable_candidate": (
                "the paper's own Stage 1 verdict claims open data, AND a snapshot for a "
                "data-kind URL holds at least one non-HTML file. Revised 2026-09-21: the "
                "previous criterion (retrieval_state in {resolved, fetched} and kind in "
                "{data, both}) counted saved landing pages and third-party artifacts, and "
                "produced the superseded figure of 58 papers"
            ),
            "data_and_code": "executable_candidate plus a code artifact the paper claims as its own",
        },
        "executable_candidate_count": len(data_only),
        "data_and_code_count": len(data_and_code),
        "by_year": by_year,
        "executable_candidate_paper_ids": sorted(data_only),
        "data_and_code_paper_ids": sorted(data_and_code),
    }


async def command_artifact_align(args: Any, logger: Any, Catalog: Any) -> int:
    survey = load_haim_fields(Path(args.survey))
    console = Console()
    async with Catalog(Path(args.database)) as catalog:
        await ensure_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        if experiment["stage"] != "claim":
            raise ValueError("artifact align scores the claim experiment against Haim")
        await merge_fields(catalog, experiment)
        cursor = await catalog.db.execute(
            "SELECT p.canonical_doi doi,m.paper_id,m.field_name,m.verdict,m.quote,m.source "
            "FROM artifact_merged_fields m JOIN papers p ON p.id=m.paper_id "
            "WHERE m.claim_experiment_id=?",
            (experiment["id"],),
        )
        merged: dict[str, dict[str, dict[str, Any]]] = {}
        for row in await cursor.fetchall():
            if not row["doi"]:
                continue
            merged.setdefault(normalize_doi(row["doi"]), {})[row["field_name"]] = dict(row)
        support: dict[str, dict[str, int]] = {}
        field_metrics: dict[str, Any] = {}
        disagreements: list[dict[str, Any]] = []
        for field in CLAIM_FIELDS:
            pairs: list[tuple[str, str]] = []
            haim_counts: Counter[str] = Counter()
            for doi, haim in survey.items():
                expected = haim[field]
                haim_counts[expected] += 1
                ours = (merged.get(doi) or {}).get(field)
                if ours is None:
                    continue
                pairs.append((expected, ours["verdict"]))
                if expected != ours["verdict"]:
                    disagreements.append(
                        {
                            "doi": doi,
                            "field": field,
                            "ours": ours["verdict"],
                            "haim": expected,
                            "quote": ours.get("quote"),
                            "source": ours.get("source"),
                            "cause_code": None,
                            "cause_codes": [
                                "ours_right",
                                "benchmark_right",
                                "ambiguous",
                                "link_changed_since_2023",
                            ],
                        }
                    )
            field_metrics[field] = _confusion(pairs)
            support[field] = dict(haim_counts)
        cursor = await catalog.db.execute(
            "SELECT p.canonical_doi doi,cv.markdown FROM papers p "
            "JOIN file_assets fa ON fa.paper_id=p.id "
            "JOIN content_versions cv ON cv.file_asset_id=fa.id "
            "WHERE cv.is_current=1 AND p.canonical_doi IS NOT NULL"
        )
        oddpub_pairs_data: list[tuple[str, str]] = []
        oddpub_pairs_code: list[tuple[str, str]] = []
        for row in await cursor.fetchall():
            doi = normalize_doi(row["doi"])
            if doi not in survey:
                continue
            detected = oddpub_detect(row["markdown"])
            haim_data_yes = survey[doi]["open_data"] == "Yes"
            haim_materials = survey[doi]["open_materials"] in {"Full", "Partial"}
            oddpub_pairs_data.append(
                ("Yes" if haim_data_yes else "No", "Yes" if detected["open_data"] else "No")
            )
            oddpub_pairs_code.append(
                ("Yes" if haim_materials else "No", "Yes" if detected["open_code"] else "No")
            )
        oddpub = {
            "open_data_vs_haim_yes": _confusion(oddpub_pairs_data),
            "open_code_vs_haim_materials_non_no": _confusion(oddpub_pairs_code),
            "note": "OddPub is boolean regex; Haim is multi-class. Comparison is collapsed to Yes/No.",
        }
        census = await executable_candidates(catalog)
        agreement = await _repeated_run_agreement(catalog, experiment["id"])
        report = {
            "experiment": args.experiment,
            "coverage": {"benchmark": len(survey), "joined": sum(1 for doi in survey if doi in merged)},
            "support": support,
            "per_field": field_metrics,
            "repeated_run_agreement": agreement,
            "gated": False,
            "gate_note": "Per-field acceptance thresholds are not set here. Review support, then record the gate decision.",
            "oddpub": oddpub,
            "disagreements": disagreements,
            "executable_candidates": census,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        table = Table(title="Stage 1 per-field support (Haim LAK, n=133). Not gated.")
        table.add_column("field")
        table.add_column("Haim counts")
        table.add_column("evaluated")
        table.add_column("accuracy")
        table.add_column("kappa")
        for field in CLAIM_FIELDS:
            counts = ", ".join(f"{key}={value}" for key, value in sorted(support[field].items()))
            metrics = field_metrics[field]
            table.add_row(
                field,
                counts,
                str(metrics["evaluated"]),
                "—" if metrics["accuracy"] is None else f"{metrics['accuracy']:.3f}",
                "—" if metrics["cohen_kappa"] is None else f"{metrics['cohen_kappa']:.3f}",
            )
        console.print(table)
        console.print(f"Wrote {output}")
        console.print("Do not set acceptance thresholds from this table in code.")
    logger.info("artifact_align_wrote", path=str(args.output), disagreements=len(disagreements))
    return 0


async def command_artifact_report(args: Any, logger: Any, Catalog: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        await ensure_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        claim = experiment if experiment["stage"] == "claim" else await get_experiment(
            catalog,
            # docs experiments store the parent id; look up by id via name path is not needed
            args.experiment,
        )
        if experiment["stage"] == "docs":
            cursor = await catalog.db.execute(
                "SELECT name FROM artifact_experiments WHERE id=?",
                (experiment["claim_experiment_id"],),
            )
            parent = await cursor.fetchone()
            if parent:
                claim = await get_experiment(catalog, parent["name"])
        await merge_fields(catalog, claim)
        census = await executable_candidates(catalog)
        console = Console()
        table = Table(title="Executable-candidate census")
        table.add_column("year")
        table.add_column("data reachable")
        table.add_column("data and code")
        for year, counts in sorted(census["by_year"].items()):
            table.add_row(str(year), str(counts["data_reachable"]), str(counts["data_and_code"]))
        console.print(table)
        console.print(f"executable_candidate = {census['executable_candidate_count']}")
        console.print(f"data_and_code = {census['data_and_code_count']}")
        console.print(census["criteria"]["executable_candidate"])
        cursor = await catalog.db.execute(
            "SELECT retrieval_state, COUNT(*) n FROM artifact_urls GROUP BY retrieval_state"
        )
        for row in await cursor.fetchall():
            console.print(f"  {row['retrieval_state']}: {row['n']}")
    logger.info(
        "artifact_report_complete",
        executable_candidate_count=census["executable_candidate_count"],
        data_and_code_count=census["data_and_code_count"],
    )
    return 0
