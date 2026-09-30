"""Deterministic alignment analysis against Haim et al.'s released LAK survey."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
from rich.console import Console
from rich.table import Table
from scipy.stats import beta as beta_dist

LABELS = ("applicable", "not_applicable", "unclear")
PRIMARY_LABELS = LABELS[:2]
ALIGNMENT_VERSION = "2.0.0"
# Acceptance criteria, v2.0.0 (see docs/protocol.md section 10.1).
#
# WHY NOT KAPPA. The Haim LAK benchmark holds 133 papers of which only 6-7 are
# not_applicable. Kappa and macro-F1 are dominated by that class, and at n=7 an exact
# interval on not-applicable recall spans 0.41 in width even when the classifier is
# PERFECT (7/7 -> [0.590, 1.000]). A bar of 0.80 therefore cannot be certified by any
# classifier, and one paper moves kappa by about 0.10. Such a bar tests sampling noise.
# By contrast the two quantities gated below are estimated to +-0.03 at perfect
# performance, because their support is 126 and 130 rather than 7.
#
# WHAT IS GATED, AND WHY THESE TWO. The two errors are not symmetric.
#   * Excluding a computational paper is silent and unrecoverable: the paper leaves the
#     study and biases every downstream count, with nothing downstream able to reveal it.
#   * Admitting a non-computational paper only dilutes the corpus, and is visible later -
#     it acquires no artifacts and can never reach L4+.
# So gate recall hard, and gate the admitted-error rate loosely but explicitly.
#
# HOW THE LEVELS WERE SET. Each bar is a 95%-confidence statement about the TRUE rate,
# chosen so the bias it allows in the headline reproducibility rate stays near or below
# 10% relative - comfortably inside that rate's own uncertainty, which under the Tier 1
# analysis of protocol section 7 is far wider. Contamination c biases the reported rate
# by approximately c in relative terms, hence the 0.10 ceiling; the recall floor of 0.95
# bounds silent exclusion at the same order.
#
# ONE BAR PER QUANTITY. Point-estimate bars were removed after a power check showed them
# dominated: at 124/126 a 0.98 point bar passed while the interval bar failed, and the
# contamination point and interval bars flipped at the identical count. Two bars where one
# is dominated obscure which criterion is actually deciding the gate.
THRESHOLDS = {
    # 95% confident the true recall on computational papers is at least 0.95.
    # At n=126 this admits at most one observed error.
    "applicable_recall_ci_low": 0.95,
    # 95% confident that at most 10% of the working corpus does not belong in it.
    # At n=130 this admits at most six observed false positives.
    "contamination_ci_high_max": 0.10,
    "unclear_rate_max": 0.10,
    "repeated_run_agreement": 0.95,
}

REPORTED_NOT_GATED = ("macro_f1", "cohen_kappa", "balanced_accuracy", "not_applicable_recall")


def _normalize_doi(value: str) -> str:
    doi = value.strip().lower()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip(".,;:)}]>")


def load_haim_explanations(path: Path) -> dict[str, str]:
    """Extract the authors' DOI-delimited rationales from their released PDF."""
    with pymupdf.open(path) as document:
        text = "\n".join(page.get_text() for page in document)
    matches = list(
        re.finditer(r"https?://doi\.org/(10\.1145/\d+\.\d+)", text, flags=re.IGNORECASE)
    )
    explanations: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        explanation = re.sub(r"\s+", " ", text[match.end() : end]).strip()
        explanations[_normalize_doi(match.group(1))] = explanation
    return explanations


def load_haim_survey(path: Path) -> dict[str, dict[str, Any]]:
    """Read the authors' raw Qualtrics export and derive the applicability target.

    Haim et al. did not publish an applicability variable. Their final
    ``Reproducible`` value is mapped as specified in the study protocol:
    ``N/A`` means outside their framework; ``No`` remains applicable.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 4:
        raise ValueError(f"Haim survey has no data rows: {path}")
    headers = rows[1]
    required = {
        "Paper DOI Link",
        "Conference Proceedings",
        "Type",
        "Open Data",
        "Open Materials",
        "Reproducible",
    }
    if not required.issubset(headers):
        raise ValueError(
            f"Haim survey is missing columns: {sorted(required - set(headers))}"
        )

    result: dict[str, dict[str, Any]] = {}
    for values in rows[3:]:
        if not values or not values[0].strip():
            continue
        row = dict(zip(headers, values, strict=False))
        reproducible = row["Reproducible"].strip()
        if reproducible == "N/A":
            expected = "not_applicable"
        elif reproducible == "No":
            expected = "applicable"
        else:
            raise ValueError(
                f"Unsupported Haim Reproducible value {reproducible!r} for {row['Paper DOI Link']}"
            )
        doi = _normalize_doi(row["Paper DOI Link"])
        if doi in result:
            raise ValueError(f"Duplicate DOI in Haim survey: {doi}")
        conference = row["Conference Proceedings"].strip()
        result[doi] = {
            "doi": doi,
            "year": 2010 + int(conference),
            "paper_type": row["Type"].strip(),
            "haim_reproducible": reproducible,
            "haim_open_data": row["Open Data"].strip(),
            "haim_open_materials": row["Open Materials"].strip(),
            "expected": expected,
        }
    return result


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision is not None and recall is not None else None
    return 2 * precision * recall / (precision + recall)


def _wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> list[float] | None:
    """Exact (Clopper-Pearson) binomial confidence interval.

    Preferred over Wilson at the support this benchmark provides: the not_applicable class
    holds fewer than ten papers, where Wilson's normal approximation is unreliable.
    """
    if total <= 0 or successes < 0 or successes > total:
        return None
    lower = (
        0.0
        if successes == 0
        else float(beta_dist.ppf(alpha / 2, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta_dist.ppf(1 - alpha / 2, successes + 1, total - successes))
    )
    return [lower, upper]


def _covered_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [case for case in cases if case["predicted"] is not None]
    confusion = {
        expected: {predicted: 0 for predicted in LABELS} for expected in PRIMARY_LABELS
    }
    for case in covered:
        confusion[case["expected"]][case["predicted"]] += 1

    per_class: dict[str, Any] = {}
    for label in PRIMARY_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in PRIMARY_LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "recall_95pct_wilson": _wilson(tp, tp + fn),
            "recall_95pct_exact": _clopper_pearson(tp, tp + fn),
            "f1": _f1(precision, recall),
        }

    f1_values = [per_class[label]["f1"] for label in PRIMARY_LABELS]
    total = len(covered)
    correct = sum(confusion[label][label] for label in PRIMARY_LABELS)
    predicted_counts = Counter(case["predicted"] for case in covered)
    expected_counts = Counter(case["expected"] for case in covered)
    expected_agreement = (
        sum(expected_counts[label] * predicted_counts[label] for label in LABELS)
        / (total * total)
        if total
        else None
    )
    observed_agreement = _safe_div(correct, total)
    kappa = None
    if (
        observed_agreement is not None
        and expected_agreement is not None
        and expected_agreement < 1
    ):
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return {
        "evaluated": total,
        "correct": correct,
        "accuracy": observed_agreement,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": (sum(value or 0.0 for value in f1_values) / len(f1_values))
        if total
        else None,
        "cohen_kappa": kappa,
        # Mean of the two class recalls. Unlike kappa its trivial-classifier baseline is
        # exactly 0.5 regardless of prevalence, so it stays interpretable at this support.
        "balanced_accuracy": (
            sum(per_class[label]["recall"] or 0.0 for label in PRIMARY_LABELS)
            / len(PRIMARY_LABELS)
        )
        if total
        else None,
        # Share of the working corpus that should not be in it: papers predicted applicable
        # that the benchmark calls not_applicable. This is the error the downstream study
        # actually pays for, and at n_predicted ~ 130 it is estimable.
        "contamination": _safe_div(
            per_class["applicable"]["false_positive"],
            per_class["applicable"]["true_positive"]
            + per_class["applicable"]["false_positive"],
        ),
        "contamination_95pct_exact": _clopper_pearson(
            per_class["applicable"]["false_positive"],
            per_class["applicable"]["true_positive"]
            + per_class["applicable"]["false_positive"],
        ),
        "negative_support": per_class["not_applicable"]["support"],
        "unclear_count": predicted_counts["unclear"],
        "unclear_rate": _safe_div(predicted_counts["unclear"], total),
    }


def _threshold_result(
    metrics: dict[str, Any], repeated_agreement: float | None
) -> dict[str, Any]:
    applicable_ci = metrics["per_class"]["applicable"].get("recall_95pct_exact")
    contamination_ci = metrics.get("contamination_95pct_exact")
    values = {
        "applicable_recall": metrics["per_class"]["applicable"]["recall"],
        "applicable_recall_ci": applicable_ci,
        "contamination": metrics.get("contamination"),
        "contamination_ci": contamination_ci,
        "unclear_rate": metrics["unclear_rate"],
        "repeated_run_agreement": repeated_agreement,
    }
    reported_only = {
        "macro_f1": metrics["macro_f1"],
        "cohen_kappa": metrics["cohen_kappa"],
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "not_applicable_recall": metrics["per_class"]["not_applicable"]["recall"],
        "not_applicable_recall_ci": metrics["per_class"]["not_applicable"].get(
            "recall_95pct_exact"
        ),
        "negative_support": metrics.get("negative_support"),
    }
    checks = {
        "applicable_recall_ci_low": applicable_ci is not None
        and applicable_ci[0] >= THRESHOLDS["applicable_recall_ci_low"],
        "contamination_ci_high": contamination_ci is not None
        and contamination_ci[1] <= THRESHOLDS["contamination_ci_high_max"],
        "unclear_rate": values["unclear_rate"] is not None
        and values["unclear_rate"] <= THRESHOLDS["unclear_rate_max"],
        "repeated_run_agreement": repeated_agreement is not None
        and repeated_agreement >= THRESHOLDS["repeated_run_agreement"],
    }
    negative_support = metrics.get("negative_support") or 0
    return {
        "values": values,
        "reported_not_gated": reported_only,
        "thresholds": THRESHOLDS,
        "checks": checks,
        "all_pass": all(checks.values()),
        "power_note": (
            f"Negative-class support is {negative_support}. Metrics dominated by that class "
            "(kappa, macro-F1, not-applicable recall) are reported with exact intervals but "
            "are not acceptance criteria; at this support their intervals span most of the "
            "range and cannot separate a good classifier from a mediocre one. Validating "
            "negative-class performance requires new labels - see docs/protocol.md section 10."
        ),
    }


async def compute_haim_alignment(
    catalog: Any,
    experiment_names: list[str],
    survey_path: Path,
    explanations_path: Path,
) -> dict[str, Any]:
    """Pool disjoint, configuration-identical experiments against Haim labels by DOI."""
    if len(set(experiment_names)) != len(experiment_names):
        raise ValueError("Specify each --experiment at most once")

    benchmark = load_haim_survey(survey_path)
    explanations = load_haim_explanations(explanations_path)
    placeholders = ",".join("?" for _ in experiment_names)
    cursor = await catalog.db.execute(
        "SELECT id,name,status,provider,model,reasoning_effort,temperature,seed,"
        "prompt_sha256,schema_sha256,repeat_count,configuration_json "
        f"FROM study_experiments WHERE name IN ({placeholders})",
        experiment_names,
    )
    experiments_by_name = {row["name"]: dict(row) for row in await cursor.fetchall()}
    missing = [name for name in experiment_names if name not in experiments_by_name]
    if missing:
        raise ValueError(f"Unknown applicability experiment(s): {', '.join(missing)}")
    experiments = [experiments_by_name[name] for name in experiment_names]
    configuration_fields = (
        "provider", "model", "reasoning_effort", "temperature", "seed",
        "prompt_sha256", "schema_sha256", "repeat_count", "configuration_json",
    )
    reference = tuple(experiments[0][field] for field in configuration_fields)
    incompatible = [
        experiment["name"] for experiment in experiments[1:]
        if tuple(experiment[field] for field in configuration_fields) != reference
    ]
    if incompatible:
        raise ValueError(
            "Cannot pool experiments with different frozen configurations; "
            f"{experiments[0]['name']} differs from {', '.join(incompatible)}"
        )

    experiment_ids = [experiment["id"] for experiment in experiments]
    cursor = await catalog.db.execute(
        "SELECT p.id paper_id,p.canonical_doi,p.title,p.collection_year,"
        "d.final_label,d.agreement_type,d.completed_run_count,d.vote_counts_json,"
        "e.name experiment_name "
        "FROM applicability_decisions d JOIN papers p ON p.id=d.paper_id "
        "JOIN study_experiments e ON e.id=d.experiment_id "
        f"WHERE d.experiment_id IN ({placeholders}) "
        "ORDER BY p.collection_year,p.canonical_doi,e.name",
        experiment_ids,
    )
    predictions: dict[str, Any] = {}
    unmatched_predictions: list[dict[str, Any]] = []
    for row in await cursor.fetchall():
        doi = _normalize_doi(row["canonical_doi"] or "")
        record = dict(row)
        if doi in predictions:
            raise ValueError(
                f"Selected experiments overlap on DOI {doi}: "
                f"{predictions[doi]['experiment_name']} and {record['experiment_name']}"
            )
        if doi in benchmark:
            predictions[doi] = record
        else:
            unmatched_predictions.append(record)

    cursor = await catalog.db.execute(
        "SELECT paper_id,repetition_index,label,confidence,contribution_type,parsed_result_json "
        f"FROM applicability_runs WHERE experiment_id IN ({placeholders}) AND status='succeeded' "
        "ORDER BY paper_id,repetition_index",
        experiment_ids,
    )
    runs_by_paper: dict[int, list[dict[str, Any]]] = {}
    for row in await cursor.fetchall():
        runs_by_paper.setdefault(int(row["paper_id"]), []).append(
            {
                "repetition": row["repetition_index"],
                "label": row["label"],
                "confidence": row["confidence"],
                "contribution_type": row["contribution_type"],
                "result": json.loads(row["parsed_result_json"]),
            }
        )

    cases: list[dict[str, Any]] = []
    for doi, target in sorted(
        benchmark.items(), key=lambda item: (item[1]["year"], item[0])
    ):
        prediction = predictions.get(doi)
        predicted = prediction["final_label"] if prediction else None
        consistent = predicted == target["expected"] if predicted is not None else None
        detailed = target["expected"] == "not_applicable" or consistent is False
        cases.append(
            {
                **target,
                "paper_id": prediction["paper_id"] if prediction else None,
                "title": prediction["title"] if prediction else None,
                "predicted": predicted,
                "consistent": consistent,
                "agreement_type": prediction["agreement_type"] if prediction else None,
                "completed_run_count": prediction["completed_run_count"]
                if prediction
                else 0,
                "vote_counts": json.loads(prediction["vote_counts_json"])
                if prediction
                else {},
                "haim_explanation": explanations.get(doi) if detailed else None,
                "model_runs": runs_by_paper.get(int(prediction["paper_id"]), [])
                if prediction and detailed
                else [],
            }
        )

    metrics = _covered_metrics(cases)
    covered = [case for case in cases if case["predicted"] is not None]
    unanimous = sum(case["agreement_type"] == "unanimous" for case in covered)
    repeated_agreement = _safe_div(unanimous, len(covered))
    years = {
        str(year): {
            "benchmark": len(year_cases),
            "covered": sum(case["predicted"] is not None for case in year_cases),
            "metrics": _covered_metrics(year_cases),
        }
        for year in sorted({case["year"] for case in cases})
        if (year_cases := [case for case in cases if case["year"] == year])
    }
    thresholds = _threshold_result(metrics, repeated_agreement)
    complete_coverage = len(covered) == len(cases)
    return {
        "alignment_version": ALIGNMENT_VERSION,
        "benchmark": {
            "source_path": str(survey_path),
            "explanations_path": str(explanations_path),
            "mapping": {"No": "applicable", "N/A": "not_applicable"},
            "total": len(cases),
            "applicable": sum(case["expected"] == "applicable" for case in cases),
            "not_applicable": sum(
                case["expected"] == "not_applicable" for case in cases
            ),
        },
        # Keep the old singular field for existing one-experiment consumers; the
        # plural field is the complete provenance for pooled analyses.
        "experiment": experiments[0] if len(experiments) == 1 else None,
        "experiments": experiments,
        "experiment_names": experiment_names,
        "coverage": {
            "matched": len(covered),
            "benchmark_total": len(cases),
            "rate": _safe_div(len(covered), len(cases)),
            "complete": complete_coverage,
            "unmatched_experiment_predictions": len(unmatched_predictions),
        },
        "metrics_on_covered_records": metrics,
        "repeated_run_agreement": repeated_agreement,
        "threshold_evaluation": {
            **thresholds,
            "accepted": complete_coverage and thresholds["all_pass"],
            "coverage_required": 1.0,
        },
        "by_year": years,
        "inconsistencies": [case for case in cases if case["consistent"] is False],
        "missing_predictions": [case for case in cases if case["predicted"] is None],
        "historical_not_applicable_cases": [
            case for case in cases if case["expected"] == "not_applicable"
        ],
        "unmatched_experiment_predictions": unmatched_predictions,
        "cases": cases,
    }


def render_haim_alignment(report: dict[str, Any]) -> None:
    console = Console()
    coverage = report["coverage"]
    metrics = report["metrics_on_covered_records"]
    console.print(
        f"[bold]Haim alignment: {', '.join(report['experiment_names'])}[/bold]  "
        f"coverage={coverage['matched']}/{coverage['benchmark_total']} "
        f"({coverage['rate']:.1%})  accepted={report['threshold_evaluation']['accepted']}"
    )
    table = Table(title="Covered-record metrics")
    table.add_column("Measure")
    table.add_column("Value", justify="right")
    table.add_column("Protocol", justify="right")
    table.add_column("Pass", justify="center")
    checks = report["threshold_evaluation"]["checks"]
    rows = (
        (
            "Applicable precision",
            metrics["per_class"]["applicable"]["precision"],
            None,
            None,
        ),
        (
            "Applicable recall (gated on CI low)",
            metrics["per_class"]["applicable"]["recall"],
            None,
            checks["applicable_recall_ci_low"],
        ),
        (
            "Not-applicable precision",
            metrics["per_class"]["not_applicable"]["precision"],
            None,
            None,
        ),
        (
            "Not-applicable recall",
            metrics["per_class"]["not_applicable"]["recall"],
            None,
            None,
        ),
        (
            "Contamination (gated on CI high)",
            metrics.get("contamination"),
            None,
            checks["contamination_ci_high"],
        ),
        ("Macro-F1 (reported, not gated)", metrics["macro_f1"], None, None),
        ("Cohen's kappa (reported, not gated)", metrics["cohen_kappa"], None, None),
        (
            "Balanced accuracy (reported, not gated)",
            metrics.get("balanced_accuracy"),
            None,
            None,
        ),
        ("Unclear rate", metrics["unclear_rate"], THRESHOLDS["unclear_rate_max"], checks["unclear_rate"]),
        (
            "Repeated-run agreement",
            report["repeated_run_agreement"],
            0.95,
            checks["repeated_run_agreement"],
        ),
    )
    for name, value, threshold, passed in rows:
        table.add_row(
            name,
            "n/a" if value is None else f"{value:.3f}",
            "—" if threshold is None else f"{threshold:.2f}",
            "—" if passed is None else ("yes" if passed else "no"),
        )
    console.print(table)

    evaluation = report["threshold_evaluation"]
    thresholds = evaluation.get("thresholds", {})
    applicable_ci = metrics["per_class"]["applicable"].get("recall_95pct_exact")
    negative_ci = metrics["per_class"]["not_applicable"].get("recall_95pct_exact")
    contamination_ci = metrics.get("contamination_95pct_exact")
    intervals = Table(title="Exact 95% intervals")
    intervals.add_column("Quantity")
    intervals.add_column("Estimate", justify="right")
    intervals.add_column("95% CI", justify="right")
    intervals.add_column("Bound", justify="right")
    for name, value, interval, bound in (
        (
            "Applicable recall",
            metrics["per_class"]["applicable"]["recall"],
            applicable_ci,
            f"low >= {thresholds.get('applicable_recall_ci_low', 0):.2f}",
        ),
        (
            "Contamination",
            metrics.get("contamination"),
            contamination_ci,
            f"high <= {thresholds.get('contamination_ci_high_max', 0):.2f}",
        ),
        (
            f"Not-applicable recall (n={metrics.get('negative_support')})",
            metrics["per_class"]["not_applicable"]["recall"],
            negative_ci,
            "not gated",
        ),
    ):
        intervals.add_row(
            name,
            "n/a" if value is None else f"{value:.3f}",
            "n/a" if not interval else f"[{interval[0]:.3f}, {interval[1]:.3f}]",
            bound,
        )
    console.print(intervals)
    if evaluation.get("power_note"):
        console.print(f"[dim]{evaluation['power_note']}[/dim]")

    confusion = metrics["confusion_matrix"]
    matrix = Table(title="Confusion matrix (covered records)")
    matrix.add_column("Haim \\ model")
    for label in LABELS:
        matrix.add_column(label, justify="right")
    for expected in PRIMARY_LABELS:
        matrix.add_row(expected, *(str(confusion[expected][label]) for label in LABELS))
    console.print(matrix)

    negatives = Table(title="Every historical not-applicable case")
    for column in ("Year", "DOI", "Prediction", "Agreement", "Result"):
        negatives.add_column(column)
    for case in report["historical_not_applicable_cases"]:
        negatives.add_row(
            str(case["year"]),
            case["doi"],
            case["predicted"] or "missing",
            case["agreement_type"] or "—",
            "aligned"
            if case["consistent"]
            else ("missing" if case["consistent"] is None else "inconsistent"),
        )
    console.print(negatives)


async def command_haim_alignment(args: Any, logger: Any, Catalog: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        from screening import ensure_schema

        await ensure_schema(catalog)
        report = await compute_haim_alignment(
            catalog,
            args.experiments,
            Path(args.survey),
            Path(args.explanations),
        )
    render_haim_alignment(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    logger.info(
        "haim_alignment_complete",
        experiments=args.experiments,
        coverage=report["coverage"],
        accepted=report["threshold_evaluation"]["accepted"],
        inconsistency_count=len(report["inconsistencies"]),
        output=str(args.output) if args.output else None,
    )
    return 0
