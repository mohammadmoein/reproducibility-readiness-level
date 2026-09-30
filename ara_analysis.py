"""Comparative statistics and dependency-free publication artifacts for ARA runs."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from agreement import load_haim_survey

SCORE_FIELDS = (
    "source_score",
    "methods_score",
    "experiments_score",
    "sink_score",
    "content_score",
    "structural_score",
    "reproducibility_index",
)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _spearman(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "rho": None}
    left, right = map(list, zip(*pairs, strict=True))
    return {"n": len(pairs), "rho": _pearson(_rank(left), _rank(right))}


def aggregate_paper_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "succeeded":
            grouped[int(row["paper_id"])].append(row)
    papers: list[dict[str, Any]] = []
    for paper_id, runs in sorted(grouped.items()):
        first = runs[0]
        record: dict[str, Any] = {
            "paper_id": paper_id,
            "doi": first["canonical_doi"],
            "title": first["title"],
            "year": first["collection_year"],
            "paper_kind": first["paper_kind"],
            "successful_runs": len(runs),
        }
        for field in SCORE_FIELDS:
            values = [float(run[field]) for run in runs if run[field] is not None]
            record[field] = statistics.fmean(values) if values else None
            record[f"{field}_sd"] = (
                statistics.stdev(values)
                if len(values) > 1
                else (0.0 if values else None)
            )
        papers.append(record)
    return papers


def compare_with_haim(
    papers: list[dict[str, Any]],
    survey_path: Path,
    run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    survey = load_haim_survey(survey_path)
    data_levels = {"No": 0.0, "Data Available on Request": 1.0, "Yes": 2.0}
    materials_levels = {"No": 0.0, "On Request": 1.0, "Partial": 2.0, "Full": 3.0}
    matched: list[dict[str, Any]] = []
    data_pairs: list[tuple[float, float]] = []
    materials_pairs: list[tuple[float, float]] = []
    papers_by_doi = {paper["doi"]: paper for paper in papers if paper["doi"]}
    runs_by_doi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row["canonical_doi"]:
            runs_by_doi[row["canonical_doi"]].append(row)
    for paper in papers:
        target = survey.get(paper["doi"] or "")
        if not target:
            continue
        record = {
            **paper,
            "haim_reproducible": target["haim_reproducible"],
            "haim_open_data": target["haim_open_data"],
            "haim_open_materials": target["haim_open_materials"],
        }
        matched.append(record)
        if (
            target["haim_open_data"] in data_levels
            and paper["source_score"] is not None
        ):
            data_pairs.append(
                (data_levels[target["haim_open_data"]], paper["source_score"])
            )
        if (
            target["haim_open_materials"] in materials_levels
            and paper["methods_score"] is not None
        ):
            materials_pairs.append(
                (
                    materials_levels[target["haim_open_materials"]],
                    paper["methods_score"],
                )
            )
    completed_scores = [
        paper["reproducibility_index"]
        for paper in papers
        if paper["reproducibility_index"] is not None
    ]
    likely_reproducible_candidates: list[dict[str, Any]] = []
    for doi, target in survey.items():
        if not (
            target["haim_open_data"] == "Yes"
            and target["haim_open_materials"] == "Full"
        ):
            continue
        paper = papers_by_doi.get(doi)
        candidate_runs = runs_by_doi.get(doi, [])
        likely_reproducible_candidates.append(
            {
                "doi": doi,
                "title": (
                    paper["title"]
                    if paper
                    else (candidate_runs[0]["title"] if candidate_runs else None)
                ),
                "haim_reproducible_15_minutes": target["haim_reproducible"],
                "haim_basis": "Open Data=Yes and Open Materials=Full; described by Haim et al. as likely reproducible with additional configuration",
                "ara_statuses": sorted({row["status"] for row in candidate_runs}),
                "ara_rank_by_r": (
                    1
                    + sum(
                        score > paper["reproducibility_index"]
                        for score in completed_scores
                    )
                    if paper and paper["reproducibility_index"] is not None
                    else None
                ),
                "ara_completed_papers": len(completed_scores),
                **{field: paper[field] if paper else None for field in SCORE_FIELDS},
            }
        )
    likely_reproducible_candidates.sort(key=lambda row: row["doi"])
    reproducibility_labels = sorted({record["haim_reproducible"] for record in matched})
    return {
        "survey_path": str(survey_path),
        "matched_papers": len(matched),
        "haim_reproducibility_labels_present": reproducibility_labels,
        "binary_threshold_identifiable": len(reproducibility_labels) > 1,
        "exploratory_construct_associations": {
            "haim_open_data_ordinal_vs_ara_source_score": _spearman(data_pairs),
            "haim_open_materials_ordinal_vs_ara_methods_score": _spearman(
                materials_pairs
            ),
            "warning": "These are convergent-construct associations, not accuracy or reproduction validation.",
        },
        "haim_likely_reproducible_candidates": likely_reproducible_candidates,
        "papers": matched,
    }


def _save_figure(figure: Figure, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.png", output / f"{stem}.svg"]
    for path in paths:
        figure.savefig(path, dpi=300, bbox_inches="tight")
    return paths


def _write_scatter(
    papers: list[dict[str, Any]],
    output: Path,
    candidate_dois: set[str],
) -> list[Path]:
    points = [
        paper
        for paper in papers
        if paper["content_score"] is not None and paper["structural_score"] is not None
    ]
    if not points:
        return []
    point_data = {
        field: [paper[field] for paper in points]
        for field in (
            "content_score",
            "structural_score",
            "reproducibility_index",
            "paper_kind",
        )
    }

    sns.set_theme(style="whitegrid", context="paper")
    figure = Figure(figsize=(7.5, 6.5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    sns.scatterplot(
        data=point_data,
        x="content_score",
        y="structural_score",
        hue="reproducibility_index",
        hue_norm=(0, 1),
        style="paper_kind",
        palette="viridis",
        s=75,
        alpha=0.8,
        ax=axis,
    )
    candidates = [paper for paper in points if paper["doi"] in candidate_dois]
    candidate_data = {
        field: [paper[field] for paper in candidates]
        for field in ("content_score", "structural_score")
    }
    if candidates:
        sns.scatterplot(
            data=candidate_data,
            x="content_score",
            y="structural_score",
            marker="*",
            color="#c0392b",
            edgecolor="white",
            linewidth=0.7,
            s=240,
            label="Haim likely-reproducible candidate",
            ax=axis,
        )
        for paper in candidates:
            axis.annotate(
                f"paper {paper['paper_id']}",
                (paper["content_score"], paper["structural_score"]),
                xytext=(7, -12),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel=r"Content reconstructability $R_c$",
        ylabel=r"Structural reconstructability $R_s$",
        title=f"ARA content and structural scores (completed papers, n={len(points)})",
    )
    axis.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )
    return _save_figure(figure, output, "content-vs-structure")


def _write_boxplot(
    papers: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    labels = {
        "source_score": "Sources",
        "methods_score": "Methods",
        "experiments_score": "Experiments",
        "sink_score": "Outputs",
        "content_score": r"Content $R_c$",
        "structural_score": r"Structure $R_s$",
        "reproducibility_index": r"Composite $R$",
    }
    observation_pairs = [
        (labels[field], paper[field])
        for field in SCORE_FIELDS
        for paper in papers
        if paper[field] is not None
    ]
    observations = {
        "metric": [pair[0] for pair in observation_pairs],
        "score": [pair[1] for pair in observation_pairs],
    }
    if not observation_pairs:
        return []

    sns.set_theme(style="whitegrid", context="paper")
    figure = Figure(figsize=(9.5, 6.0))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    sns.boxplot(
        data=observations,
        x="score",
        y="metric",
        color="#85c1e9",
        fliersize=0,
        ax=axis,
    )
    sns.stripplot(
        data=observations,
        x="score",
        y="metric",
        color="#1b4f72",
        alpha=0.55,
        jitter=0.18,
        size=3.5,
        ax=axis,
    )
    axis.set(
        xlim=(-0.02, 1.02),
        xlabel="Score",
        ylabel="",
        title=f"ARA score distributions (completed papers, n={len(papers)})",
    )
    return _save_figure(figure, output, "score-distributions")


def write_analysis_artifacts(report: dict[str, Any], root: Path) -> list[str]:
    output = root / report["experiment"]["name"]
    output.mkdir(parents=True, exist_ok=True)
    papers = report["paper_summaries"]
    paths: list[str] = []
    csv_path = output / "paper-scores.csv"
    fields = ["paper_id", "doi", "title", "year", "paper_kind", "successful_runs"] + [
        field for score in SCORE_FIELDS for field in (score, f"{score}_sd")
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: paper.get(field) for field in fields} for paper in papers
        )
    paths.append(str(csv_path))
    candidates = report["haim_comparison"]["haim_likely_reproducible_candidates"]
    candidate_dois = {candidate["doi"] for candidate in candidates}
    paths.extend(str(path) for path in _write_scatter(papers, output, candidate_dois))
    paths.extend(str(path) for path in _write_boxplot(papers, output))
    return paths
