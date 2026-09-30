"""Stage 2 R analytics: distribution, trend, instrument behaviour, node-level deficits.

Read-only over the stored ARA evidence. The database is opened with SQLite's `mode=ro`
URI so the driver itself refuses a write; nothing here touches `ara_runs`,
`ara_experiments` or `artifacts/ara/**`. Output goes to `artifacts/analysis/` only.

Observed value domains, probed over all 1,413 successful runs of
`s2-ara-2021_2022-gpt56t-r3` and `s2-ara-2023_2026-gpt56t-r3` on 2026-09-18, and
cross-checked against the upstream schemas in
`.external/ara/src/ara_pipeline/online_llm_pipelines.py`:

* `wiring_metrics_json`: `lwcc_fraction`, `resolved_input_ratio`, `sinks_produced_ratio`,
  `source_to_sink_reachability`, `sources_consumed_ratio` — all floats in [0, 1], all
  present on every run.
* `nodes_source[].availability`: `open` | `upon_request` | `not_mentioned` (all three
  observed; upstream `DataAvailability`).
* `nodes_source[].license`: free string or null; null on 4,020 of 4,030 nodes.
* `nodes_source[].reproducibility_score`: integer percentage in [0, 100]; 10 distinct
  values observed, 74% at 0.
* `nodes_process[].process_type`: `method` | `experiment` (upstream `ProcessType`).
* `nodes_process[].algorithm_clarity`: integer in {1, 2, 3, 4}; all four observed.
* `nodes_process[].{parameters,tools}_{mentioned,required}`: lists of strings, possibly
  empty. `parameters_required` is empty on 115 of 16,142 process nodes.
* `nodes_process[].reproducibility_score`: integer percentage in [0, 100].
* `nodes_sink[].type`: `figure` | `table` (upstream `SinkType`).
* `nodes_sink[].statement_clarity`: integer in {1, 2, 3, 4}; all four observed.
* `nodes_sink[].statement_validity`: `supported` | `partially_supported` observed;
  upstream `StatementValidity` also permits `unsupported` and `not_assessed`.
* `nodes_sink[].reproducibility_score`: integer percentage in [0, 100].

Any value outside these domains raises rather than being coerced, so a prompt or model
change that shifts the schema surfaces as a failure instead of a quietly wrong number.

No threshold on R is selected here; A4 reports a sensitivity curve. Gate decisions are
the human's.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import statistics as st
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import stats as sps

BENCHMARK_EXPERIMENT = "s2-ara-2021_2022-gpt56t-r3"
LATER_EXPERIMENT = "s2-ara-2023_2026-gpt56t-r3"
DEFAULT_EXPERIMENTS = (BENCHMARK_EXPERIMENT, LATER_EXPERIMENT)
DRIFT_EXPERIMENT = "ara-lak2021-gpt56terra-medium-v1"
LUNA_EXPERIMENT = "s2-ara-luna-probe-r3"

# Frozen Stage 1 experiments whose resolver output is the verified-retrieval evidence.
# The reliability replicate is deliberately excluded: it re-audits papers already covered.
STAGE1_CLAIM_EXPERIMENTS = (
    "s1-claim-2021_2022-gpt56t-p1.1.0",
    "s1-claim-2023_2026-gpt56t-p1.1.0",
)
STAGE1_URL_EXPERIMENTS = STAGE1_CLAIM_EXPERIMENTS + (
    "s1-docs-2021_2022-gpt56t-p1.0.0",
    "s1-docs-2023_2026-gpt56t-p1.0.0",
)

DEFAULT_SEED = 20260918
DEFAULT_BOOTSTRAP = 10000
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)

# docs/protocol.md section 6. T5 (non-quantitative) is excluded at Stage 0 and never reaches ARA.
CLAIM_TIERS = {
    "inferential_effect": "T1 inferential effect",
    "predictive_performance": "T2 predictive performance",
    "comparative_ranking": "T3 comparative / ranking",
    "descriptive_structural": "T4 descriptive / structural",
}

# Kept verbatim from ara._ara_candidates so the claim-type projection cannot drift from
# the selection rule. _load_dataset asserts the resulting set equals _ara_candidates'.
MAJORITY_CLAIM_TYPE_SQL = (
    "SELECT paper_id,claim_type FROM ("
    "SELECT paper_id,claim_type,ROW_NUMBER() OVER ("
    "PARTITION BY paper_id ORDER BY COUNT(*) DESC,claim_type) rn "
    "FROM applicability_runs WHERE experiment_id=? AND status='succeeded' "
    "AND claim_type IS NOT NULL GROUP BY paper_id,claim_type"
    ") WHERE rn=1"
)

AVAILABILITY_DOMAIN = ("open", "upon_request", "not_mentioned")
AVAILABILITY_ORDER = {"not_mentioned": 0, "upon_request": 1, "open": 2}
VALIDITY_DOMAIN = ("supported", "partially_supported", "unsupported", "not_assessed")
PROCESS_TYPE_DOMAIN = ("method", "experiment")
SINK_TYPE_DOMAIN = ("figure", "table")
CLARITY_DOMAIN = (1, 2, 3, 4)
WIRING_KEYS = (
    "lwcc_fraction",
    "resolved_input_ratio",
    "sinks_produced_ratio",
    "source_to_sink_reachability",
    "sources_consumed_ratio",
)
STAGE_SCORES = ("source_score", "methods_score", "experiments_score", "sink_score")

# Data-bearing counts recorded by the authors' run before the retrieval census was
# corrected. Not a result. A mismatch is reported in the JSON rather than silently accepted.
EXPECTED_SHARING_CORPUS = {2021: 1, 2022: 2, 2023: 6, 2024: 11, 2025: 21, 2026: 17}


# --------------------------------------------------------------------------------------
# statistics primitives
# --------------------------------------------------------------------------------------


def _rng(seed: int, analysis_id: str) -> np.random.Generator:
    """Per-analysis generator so `--only` reproduces the same numbers as a full run."""
    digest = hashlib.blake2b(analysis_id.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng([seed, int.from_bytes(digest, "big")])


def _finite(values: Iterable[Any]) -> list[float]:
    return [float(v) for v in values if v is not None and not _isnan(v)]


def _isnan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _describe(values: Iterable[Any]) -> dict[str, Any]:
    data = _finite(values)
    if not data:
        return {"n": 0}
    array = np.asarray(data, dtype=float)
    q1, median, q3 = (float(v) for v in np.quantile(array, (0.25, 0.5, 0.75)))
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(array.max()),
        "skew": float(sps.skew(array)) if array.size > 2 else None,
    }


def _boot_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    resamples: int,
    statistic: Callable[[np.ndarray], float] = lambda a: float(a.mean()),
) -> dict[str, Any]:
    data = np.asarray(_finite(values), dtype=float)
    if data.size == 0:
        return {"n": 0, "estimate": None, "ci_low": None, "ci_high": None}
    estimate = statistic(data)
    if data.size == 1:
        return {"n": 1, "estimate": estimate, "ci_low": None, "ci_high": None, "resamples": 0}
    draws = rng.integers(0, data.size, size=(resamples, data.size))
    replicates = np.asarray([statistic(data[row]) for row in draws], dtype=float)
    low, high = (float(v) for v in np.quantile(replicates, (0.025, 0.975)))
    return {
        "n": int(data.size),
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "resamples": resamples,
    }


def _boot_statistic(
    rng: np.random.Generator,
    resamples: int,
    size: int,
    statistic: Callable[[np.ndarray], float | None],
) -> dict[str, Any]:
    """Bootstrap an arbitrary statistic by resampling row indices."""
    if size < 2:
        return {"n": size, "ci_low": None, "ci_high": None}
    replicates: list[float] = []
    with warnings.catch_warnings():
        # A resample of a near-constant column is undefined rather than wrong; those
        # replicates are dropped below and counted, so the warning is noise.
        warnings.simplefilter("ignore", sps.ConstantInputWarning)
        for _ in range(resamples):
            value = statistic(rng.integers(0, size, size=size))
            if value is not None and not _isnan(value):
                replicates.append(float(value))
    if len(replicates) < resamples // 2:
        return {"n": size, "ci_low": None, "ci_high": None, "degenerate_resamples": True}
    low, high = (float(v) for v in np.quantile(np.asarray(replicates), (0.025, 0.975)))
    return {"n": size, "ci_low": low, "ci_high": high, "resamples": resamples}


def _spearman(
    x: Sequence[float],
    y: Sequence[float],
    rng: np.random.Generator,
    resamples: int,
    label: str = "",
) -> dict[str, Any]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y, strict=True)
        if a is not None and b is not None and not _isnan(a) and not _isnan(b)
    ]
    n = len(pairs)
    if n < 3:
        return {"label": label, "n": n, "rho": None, "underpowered": True}
    left = np.asarray([p[0] for p in pairs])
    right = np.asarray([p[1] for p in pairs])
    result = sps.spearmanr(left, right)
    band = _boot_statistic(
        rng,
        resamples,
        n,
        lambda idx: sps.spearmanr(left[idx], right[idx]).statistic,
    )
    return {
        "label": label,
        "n": n,
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_low": band["ci_low"],
        "ci_high": band["ci_high"],
        # n travels with rho so no reader can quote one without the other.
        "small_n_warning": (
            f"n={n} < 10; this correlation is not interpretable on its own" if n < 10 else None
        ),
    }


def _mannwhitney(
    group_a: Sequence[float],
    group_b: Sequence[float],
    rng: np.random.Generator,
    resamples: int,
    labels: tuple[str, str] = ("a", "b"),
) -> dict[str, Any]:
    a = np.asarray(_finite(group_a), dtype=float)
    b = np.asarray(_finite(group_b), dtype=float)
    out: dict[str, Any] = {
        "groups": list(labels),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "median_a": float(np.median(a)) if a.size else None,
        "median_b": float(np.median(b)) if b.size else None,
    }
    if a.size < 2 or b.size < 2:
        out["underpowered"] = True
        return out
    test = sps.mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 2.0 * float(test.statistic) / (a.size * b.size) - 1.0
    # Each group is resampled within itself, so the bootstrap respects the group sizes
    # rather than reshuffling membership, which would test a different null.
    replicates: list[float] = []
    for _ in range(resamples):
        sa = a[rng.integers(0, a.size, size=a.size)]
        sb = b[rng.integers(0, b.size, size=b.size)]
        if np.ptp(np.concatenate([sa, sb])) == 0:
            continue
        u = float(sps.mannwhitneyu(sa, sb, alternative="two-sided").statistic)
        replicates.append(2.0 * u / (sa.size * sb.size) - 1.0)
    low = high = None
    if len(replicates) >= resamples // 2:
        low, high = (float(v) for v in np.quantile(np.asarray(replicates), (0.025, 0.975)))
    out.update(
        u_statistic=float(test.statistic),
        p_value=float(test.pvalue),
        rank_biserial_r=rank_biserial,
        ci_low=low,
        ci_high=high,
        resamples=len(replicates),
    )
    return out


def _kruskal(
    groups: Mapping[str, Sequence[float]],
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    named = {k: _finite(v) for k, v in groups.items()}
    usable = {k: v for k, v in sorted(named.items()) if len(v) >= 2}
    per_group: dict[str, Any] = {}
    # Groups are visited in sorted order and each gets a spawned child generator, so the
    # per-group intervals do not depend on how many groups happen to be present.
    children = iter(rng.spawn(len(named)))
    for key, values in sorted(named.items()):
        band = _boot_ci(values, next(children), resamples, lambda a: float(np.median(a)))
        per_group[key] = {
            "n": len(values),
            "median": band["estimate"],
            "median_ci_low": band["ci_low"],
            "median_ci_high": band["ci_high"],
        }
    if len(usable) < 2:
        return {"per_group": per_group, "underpowered": True}
    test = sps.kruskal(*usable.values())
    total = sum(len(v) for v in usable.values())
    k = len(usable)
    eta_squared = (float(test.statistic) - k + 1) / (total - k) if total > k else None
    return {
        "per_group": per_group,
        "h_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "degrees_of_freedom": k - 1,
        "n": total,
        "eta_squared": eta_squared,
        "eta_squared_note": "(H - k + 1) / (n - k); negative values mean no separation",
    }


def _ols(y: np.ndarray, design: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """Classical OLS with t-based intervals. statsmodels is not a dependency."""
    n, k = design.shape
    if n <= k:
        return {"n": n, "underpowered": True}
    if not np.isfinite(y).all() or not np.isfinite(design).all():
        # A missing cell reaching the design would surface as an opaque SVD failure.
        raise ValueError(
            f"non-finite values in the regression of {names}; drop the incomplete rows "
            "explicitly and report how many, rather than letting them through"
        )
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = n - k
    sigma2 = float(residual @ residual) / dof
    covariance = np.linalg.pinv(design.T @ design) * sigma2
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    critical = float(sps.t.ppf(0.975, dof))
    total = float(((y - y.mean()) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(se > 0, beta / se, np.nan)
    return {
        "n": n,
        "degrees_of_freedom": dof,
        "r_squared": 1.0 - float(residual @ residual) / total if total else None,
        "residual_sd": math.sqrt(sigma2),
        "terms": {
            name: {
                "coefficient": float(beta[i]),
                "std_error": float(se[i]),
                "t": None if _isnan(t_values[i]) else float(t_values[i]),
                "p_value": (
                    None
                    if _isnan(t_values[i])
                    else float(2 * sps.t.sf(abs(float(t_values[i])), dof))
                ),
                "ci_low": float(beta[i] - critical * se[i]),
                "ci_high": float(beta[i] + critical * se[i]),
            }
            for i, name in enumerate(names)
        },
    }


def _cochran_armitage(successes: Sequence[int], totals: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    s = np.asarray(scores, dtype=float)
    grand_n = float(n.sum())
    grand_x = float(x.sum())
    if grand_n == 0 or grand_x in (0.0, grand_n):
        return {"underpowered": True, "n": int(grand_n)}
    p_bar = grand_x / grand_n
    t = float((s * (x - n * p_bar)).sum())
    variance = p_bar * (1 - p_bar) * float((n * s**2).sum() - (n * s).sum() ** 2 / grand_n)
    if variance <= 0:
        return {"underpowered": True, "n": int(grand_n)}
    z = t / math.sqrt(variance)
    return {
        "n": int(grand_n),
        "z": z,
        "p_value": float(2 * sps.norm.sf(abs(z))),
        "groups": [
            {"score": float(si), "successes": int(xi), "total": int(ni), "proportion": xi / ni}
            for si, xi, ni in zip(s, x, n, strict=True)
            if ni
        ],
    }


def _partial_spearman(
    y: Sequence[float],
    x: Sequence[float],
    controls: Sequence[Sequence[float]],
    rng: np.random.Generator,
    resamples: int,
    label: str = "",
) -> dict[str, Any]:
    """Spearman on residuals after linearly removing the rank-transformed controls."""
    rows = [
        (float(a), float(b), [float(c) for c in cs])
        for a, b, cs in zip(y, x, zip(*controls, strict=True), strict=True)
        if not any(v is None or _isnan(v) for v in (a, b, *cs))
    ]
    if len(rows) < 5:
        return {"label": label, "n": len(rows), "underpowered": True}
    ry = sps.rankdata([r[0] for r in rows])
    rx = sps.rankdata([r[1] for r in rows])
    rc = np.column_stack([sps.rankdata(col) for col in zip(*[r[2] for r in rows], strict=True)])
    design = np.column_stack([np.ones(len(rows)), rc])

    def residualise(values: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        d = design if idx is None else design[idx]
        v = values if idx is None else values[idx]
        beta, *_ = np.linalg.lstsq(d, v, rcond=None)
        return v - d @ beta

    rho = float(sps.pearsonr(residualise(ry), residualise(rx)).statistic)
    band = _boot_statistic(
        rng,
        resamples,
        len(rows),
        lambda idx: sps.pearsonr(residualise(ry, idx), residualise(rx, idx)).statistic,
    )
    return {
        "label": label,
        "n": len(rows),
        "partial_rho": rho,
        "ci_low": band["ci_low"],
        "ci_high": band["ci_high"],
        "controls": ["control_%d" % i for i in range(len(controls))],
    }


def _cohen_kappa(
    pairs: Sequence[tuple[Any, Any]], raters: tuple[str, str] = ("rater_a", "rater_b")
) -> dict[str, Any]:
    """Both raters must already use the same label set, or the diagonal is meaningless."""
    if not pairs:
        return {"n": 0, "kappa": None}
    categories = sorted({v for pair in pairs for v in pair}, key=str)
    index = {c: i for i, c in enumerate(categories)}
    table = np.zeros((len(categories), len(categories)), dtype=float)
    for a, b in pairs:
        table[index[a], index[b]] += 1
    total = table.sum()
    observed = float(np.trace(table)) / total
    expected = float((table.sum(axis=0) * table.sum(axis=1)).sum()) / total**2
    return {
        "n": int(total),
        "raters": list(raters),
        "categories": [str(c) for c in categories],
        "table": table.astype(int).tolist(),
        "table_note": f"rows are {raters[0]}, columns are {raters[1]}",
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
    }


def stability_stats(r_by_paper: Mapping[int, Sequence[float]]) -> dict[str, Any]:
    """Run-to-run stability of R (A3.1). Imported by `scripts/make_stage2_report.py`."""
    if not r_by_paper:
        return {"n": 0}
    sds = [st.pstdev([float(v) for v in values]) for values in r_by_paper.values()]
    means = [st.fmean([float(v) for v in values]) for values in r_by_paper.values()]
    between = st.pstdev(means)
    return {
        "n": len(sds),
        "sd_mean": st.fmean(sds),
        "sd_median": st.median(sds),
        "sd_max": max(sds),
        "le09": sum(s <= 0.09 for s in sds),
        "gt09": sum(s > 0.09 for s in sds),
        "lt02": sum(s < 0.02 for s in sds),
        "r_mean": st.fmean(means),
        "r_between": between,
        "r_min": min(means),
        "r_max": max(means),
        "ratio": st.fmean(sds) / between if between else None,
        "means": means,
        "sds": sds,
    }


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    papers: list[dict[str, Any]]
    provenance: dict[str, Any]
    self_checks: dict[str, Any]
    sharing_corpus: dict[int, dict[str, int]]
    drift_pairs: list[dict[str, Any]]
    luna_pairs: list[dict[str, Any]]
    node_totals: dict[str, Any] = field(default_factory=dict)

    def column(self, key: str) -> list[Any]:
        return [paper.get(key) for paper in self.papers]

    def where(self, predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        return [paper for paper in self.papers if predicate(paper)]


def connect_readonly(database: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _check_domain(name: str, value: Any, domain: Sequence[Any]) -> Any:
    if value not in domain:
        raise ValueError(
            f"{name}={value!r} is outside the recorded domain {tuple(domain)}; the ARA "
            "schema or prompt changed, so refusing to coerce it (see module docstring)"
        )
    return value


def _profile_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Per-run node aggregates, with every enum validated against the recorded domain."""
    sources = profile.get("nodes_source") or []
    processes = profile.get("nodes_process") or []
    sinks = profile.get("nodes_sink") or []
    availability = Counter()
    for node in sources:
        availability[_check_domain("availability", node.get("availability"), AVAILABILITY_DOMAIN)] += 1
    clarity = Counter()
    params_mentioned = params_required = params_capped = 0
    tools_mentioned = tools_required = tools_capped = 0
    no_required_params = over_specified = 0
    process_types = Counter()
    for node in processes:
        clarity[_check_domain("algorithm_clarity", node.get("algorithm_clarity"), CLARITY_DOMAIN)] += 1
        process_types[_check_domain("process_type", node.get("process_type"), PROCESS_TYPE_DOMAIN)] += 1
        mentioned = len(node.get("parameters_mentioned") or [])
        required = len(node.get("parameters_required") or [])
        params_mentioned += mentioned
        params_required += required
        params_capped += min(mentioned, required)
        if required == 0:
            no_required_params += 1
        elif mentioned > required:
            over_specified += 1
        t_mentioned = len(node.get("tools_mentioned") or [])
        t_required = len(node.get("tools_required") or [])
        tools_mentioned += t_mentioned
        tools_required += t_required
        tools_capped += min(t_mentioned, t_required)
    statement_clarity = Counter()
    validity = Counter()
    sink_types = Counter()
    for node in sinks:
        statement_clarity[
            _check_domain("statement_clarity", node.get("statement_clarity"), CLARITY_DOMAIN)
        ] += 1
        validity[_check_domain("statement_validity", node.get("statement_validity"), VALIDITY_DOMAIN)] += 1
        sink_types[_check_domain("sink type", node.get("type"), SINK_TYPE_DOMAIN)] += 1

    def mean_score(nodes: Sequence[Mapping[str, Any]]) -> float | None:
        values = _finite(node.get("reproducibility_score") for node in nodes)
        return st.fmean(values) if values else None

    return {
        "source_node_score": mean_score(sources),
        "process_node_score": mean_score(processes),
        "sink_node_score": mean_score(sinks),
        "availability_counts": dict(availability),
        "any_source_open": availability.get("open", 0) > 0,
        "best_availability": max(
            (a for a in availability if availability[a]),
            key=lambda a: AVAILABILITY_ORDER[a],
            default="not_mentioned",
        ),
        "any_source_licensed": any(node.get("license") for node in sources),
        "any_source_url": any(node.get("url") for node in sources),
        "algorithm_clarity_counts": dict(clarity),
        "algorithm_clarity_mean": (
            sum(k * v for k, v in clarity.items()) / sum(clarity.values()) if clarity else None
        ),
        "process_type_counts": dict(process_types),
        "parameters_mentioned": params_mentioned,
        "parameters_required": params_required,
        "parameters_capped": params_capped,
        "parameter_completion": params_capped / params_required if params_required else None,
        "process_nodes_without_required_parameters": no_required_params,
        "process_nodes_over_specified": over_specified,
        "tools_mentioned": tools_mentioned,
        "tools_required": tools_required,
        "tools_capped": tools_capped,
        "tool_completion": tools_capped / tools_required if tools_required else None,
        "statement_clarity_counts": dict(statement_clarity),
        "statement_clarity_mean": (
            sum(k * v for k, v in statement_clarity.items()) / sum(statement_clarity.values())
            if statement_clarity
            else None
        ),
        "statement_validity_counts": dict(validity),
        "sink_type_counts": dict(sink_types),
        "node_names": {
            "source": sorted({str(node.get("node_name") or node.get("node_id")) for node in sources}),
            "process": sorted({str(node.get("node_name") or node.get("node_id")) for node in processes}),
            "sink": sorted({str(node.get("node_name") or node.get("node_id")) for node in sinks}),
        },
    }


def _mean_or_none(values: Iterable[Any]) -> float | None:
    data = _finite(values)
    return st.fmean(data) if data else None


def _mean_jaccard(sets: Sequence[set[str]]) -> float | None:
    pairs = list(combinations(range(len(sets)), 2))
    if not pairs:
        return None
    scores = []
    for i, j in pairs:
        union = sets[i] | sets[j]
        scores.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return st.fmean(scores)


def _experiment_row(db: sqlite3.Connection, name: str) -> sqlite3.Row:
    row = db.execute(
        "SELECT id,name,model,repeat_count,upstream_commit,configuration_json,"
        "applicability_experiment_id FROM ara_experiments WHERE name=?",
        (name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown ARA experiment: {name}")
    return row


def _sharing_sets(db: sqlite3.Connection) -> dict[str, set[int]]:
    """Verified retrieval, distinguishing code from data through `url_id`.

    Counting `artifact_snapshots` without joining `artifact_urls` reports repositories as
    datasets; that error was made and caught on 2026-09-18. `artifact_kind='both'` is a
    single URL carrying code and data, so it counts towards each.
    """
    placeholders = ",".join("?" * len(STAGE1_URL_EXPERIMENTS))
    rows = db.execute(
        "SELECT DISTINCT u.artifact_kind kind, s.paper_id FROM artifact_snapshots s "
        "JOIN artifact_urls u ON u.id=s.url_id "
        "JOIN artifact_experiments e ON e.id=u.experiment_id "
        f"WHERE s.size_bytes>0 AND e.name IN ({placeholders})",
        STAGE1_URL_EXPERIMENTS,
    ).fetchall()
    data: set[int] = set()
    code: set[int] = set()
    for row in rows:
        if row["kind"] in ("data", "both"):
            data.add(int(row["paper_id"]))
        if row["kind"] in ("code", "both"):
            code.add(int(row["paper_id"]))
    return {"data": data, "code": code}


def _stage1_fields(db: sqlite3.Connection) -> dict[int, dict[str, str]]:
    placeholders = ",".join("?" * len(STAGE1_CLAIM_EXPERIMENTS))
    rows = db.execute(
        "SELECT m.paper_id,m.field_name,m.verdict FROM artifact_merged_fields m "
        "JOIN artifact_experiments e ON e.id=m.claim_experiment_id "
        f"WHERE e.name IN ({placeholders})",
        STAGE1_CLAIM_EXPERIMENTS,
    ).fetchall()
    merged: dict[int, dict[str, str]] = defaultdict(dict)
    for row in rows:
        merged[int(row["paper_id"])][row["field_name"]] = row["verdict"]
    return dict(merged)


def load_dataset(
    database: Path,
    experiments: Sequence[str],
    candidate_ids: Mapping[str, set[int]] | None = None,
) -> Dataset:
    """Assemble one record per paper per experiment binding, plus provenance and checks."""
    db = connect_readonly(database)
    try:
        return _load_dataset(db, experiments, candidate_ids or {})
    finally:
        db.close()


def _load_dataset(
    db: sqlite3.Connection,
    experiments: Sequence[str],
    candidate_ids: Mapping[str, set[int]],
) -> Dataset:
    sharing = _sharing_sets(db)
    stage1 = _stage1_fields(db)
    environment = {
        int(row["paper_id"]): float(row["e_score"])
        for row in db.execute(
            "SELECT paper_id,MAX(e_score) e_score FROM artifact_environment GROUP BY paper_id"
        )
    }
    papers: list[dict[str, Any]] = []
    provenance_experiments: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    node_totals = Counter()

    for name in experiments:
        experiment = _experiment_row(db, name)
        claim_types = {
            int(row["paper_id"]): row["claim_type"]
            for row in db.execute(
                MAJORITY_CLAIM_TYPE_SQL, (experiment["applicability_experiment_id"],)
            )
        }
        runs = db.execute(
            "SELECT r.*,p.collection_year year,p.canonical_doi doi,p.title,"
            "fa.paper_kind kind,LENGTH(cv.markdown) markdown_chars "
            "FROM ara_runs r JOIN papers p ON p.id=r.paper_id "
            "JOIN file_assets fa ON fa.id=r.file_asset_id "
            "LEFT JOIN content_versions cv ON cv.file_asset_id=fa.id AND cv.is_current=1 "
            "WHERE r.experiment_id=? AND r.status='succeeded' "
            "ORDER BY r.paper_id,r.repetition_index",
            (experiment["id"],),
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for run in runs:
            grouped[int(run["paper_id"])].append(run)
        incomplete = sorted(
            pid for pid, rows in grouped.items() if len(rows) != experiment["repeat_count"]
        )
        for paper_id, rows in sorted(grouped.items()):
            if len(rows) != experiment["repeat_count"]:
                continue
            profiles = [_profile_summary(json.loads(row["profile_json"] or "{}")) for row in rows]
            wiring = [json.loads(row["wiring_metrics_json"] or "{}") for row in rows]
            for block in wiring:
                missing = [key for key in WIRING_KEYS if key not in block]
                if missing:
                    raise ValueError(f"paper {paper_id}: wiring metrics missing {missing}")
            claim_type = claim_types.get(paper_id)
            fields = stage1.get(paper_id, {})
            r_values = [float(row["reproducibility_index"]) for row in rows]
            record: dict[str, Any] = {
                "paper_id": paper_id,
                "experiment": name,
                "doi": rows[0]["doi"],
                "title": rows[0]["title"],
                "year": int(rows[0]["year"]),
                "kind": rows[0]["kind"],
                "claim_type": claim_type,
                "claim_tier": CLAIM_TIERS.get(claim_type or "", claim_type),
                "repetitions": len(rows),
                "markdown_chars": (
                    int(rows[0]["markdown_chars"]) if rows[0]["markdown_chars"] else None
                ),
                "r": st.fmean(r_values),
                "r_sd": st.pstdev(r_values),
                "r_min": min(r_values),
                "r_max": max(r_values),
                "r_values": r_values,
                "content_score": _mean_or_none(row["content_score"] for row in rows),
                "structural_score": _mean_or_none(row["structural_score"] for row in rows),
                "input_tokens": _mean_or_none(row["input_tokens"] for row in rows),
                "output_tokens": _mean_or_none(row["output_tokens"] for row in rows),
                "estimated_cost_usd": _mean_or_none(row["estimated_cost_usd"] for row in rows),
                "duration_ms": _mean_or_none(row["duration_ms"] for row in rows),
                "open_data": fields.get("open_data"),
                "open_materials": fields.get("open_materials"),
                "license_stated": fields.get("license"),
                "readme": fields.get("readme"),
                "data_documentation": fields.get("data_documentation"),
                "materials_documentation": fields.get("materials_documentation"),
                "verified_data": paper_id in sharing["data"],
                "verified_code": paper_id in sharing["code"],
                "e_score": environment.get(paper_id),
            }
            for score in STAGE_SCORES:
                record[score] = _mean_or_none(row[score] for row in rows)
            for count in (
                "source_node_count",
                "method_node_count",
                "experiment_node_count",
                "sink_node_count",
            ):
                record[count] = _mean_or_none(row[count] for row in rows)
            record["total_node_count"] = sum(
                record[c] or 0.0
                for c in (
                    "source_node_count",
                    "method_node_count",
                    "experiment_node_count",
                    "sink_node_count",
                )
            )
            for key in WIRING_KEYS:
                record[key] = _mean_or_none(block.get(key) for block in wiring)
            for key in (
                "source_node_score",
                "process_node_score",
                "sink_node_score",
                "algorithm_clarity_mean",
                "statement_clarity_mean",
                "parameter_completion",
                "tool_completion",
            ):
                record[key] = _mean_or_none(profile[key] for profile in profiles)
            for key in (
                "parameters_mentioned",
                "parameters_required",
                "parameters_capped",
                "tools_mentioned",
                "tools_required",
                "tools_capped",
                "process_nodes_without_required_parameters",
                "process_nodes_over_specified",
            ):
                record[key] = st.fmean(float(profile[key]) for profile in profiles)
                node_totals[key] += sum(int(profile[key]) for profile in profiles)
            for key in (
                "algorithm_clarity_counts",
                "statement_clarity_counts",
                "statement_validity_counts",
                "availability_counts",
                "process_type_counts",
                "sink_type_counts",
            ):
                pooled = Counter()
                for profile in profiles:
                    pooled.update(profile[key])
                record[key] = {str(k): int(v) for k, v in sorted(pooled.items(), key=lambda kv: str(kv[0]))}
                for k, v in pooled.items():
                    node_totals[f"{key}:{k}"] += int(v)
            record["any_source_open"] = sum(p["any_source_open"] for p in profiles) > len(profiles) / 2
            record["any_source_licensed"] = (
                sum(p["any_source_licensed"] for p in profiles) > len(profiles) / 2
            )
            record["ara_best_availability"] = Counter(
                p["best_availability"] for p in profiles
            ).most_common(1)[0][0]
            record["node_jaccard"] = {
                cls: _mean_jaccard([set(p["node_names"][cls]) for p in profiles])
                for cls in ("source", "process", "sink")
            }
            papers.append(record)

        cfg = json.loads(experiment["configuration_json"] or "{}")
        provenance_experiments.append(
            {
                "name": name,
                "model": experiment["model"],
                "repetitions": int(experiment["repeat_count"]),
                "upstream_commit": experiment["upstream_commit"],
                "reasoning_effort": cfg.get("reasoning_effort"),
                "selection": cfg.get("selection"),
                "selection_version": cfg.get("selection_version"),
                "papers_complete": sum(1 for p in papers if p["experiment"] == name),
                "papers_incomplete": incomplete,
            }
        )
        expected = candidate_ids.get(name)
        if expected is not None:
            present = {p["paper_id"] for p in papers if p["experiment"] == name}
            checks[f"selection_matches_ara_candidates[{name}]"] = {
                "ok": present == expected,
                "candidates": len(expected),
                "analysed": len(present),
                "missing_from_analysis": sorted(expected - present),
                "unexpected_in_analysis": sorted(present - expected),
            }

    checks["paper_count"] = {
        "ok": len(papers) == 471,
        "analysed": len(papers),
        "expected": 471,
        "note": "120 benchmark + 351 held out; a different n must be explained, not dropped",
    }
    by_year = Counter(p["year"] for p in papers)
    checks["papers_by_year"] = {str(y): int(by_year[y]) for y in sorted(by_year)}

    sharing_corpus = _sharing_by_year(db, sharing["data"], sharing["code"])
    observed = {y: sharing_corpus[y]["data"] for y in sharing_corpus}
    checks["verified_data_bearing_corpus"] = {
        "ok": observed == EXPECTED_SHARING_CORPUS,
        "observed": {str(k): v for k, v in sorted(observed.items())},
        "expected": {str(k): v for k, v in sorted(EXPECTED_SHARING_CORPUS.items())},
        "total": sum(observed.values()),
    }

    drift_pairs, drift_check = _drift_pairs(db, papers)
    checks["drift_identical_source_pdf"] = drift_check
    luna_pairs = _luna_pairs(db, papers)

    return Dataset(
        papers=papers,
        provenance={
            "generated_on": date.today().isoformat(),
            "experiments": provenance_experiments,
            "papers": len(papers),
            "assessments": sum(p["repetitions"] for p in papers),
            "claim_tiers": CLAIM_TIERS,
        },
        self_checks=checks,
        sharing_corpus=sharing_corpus,
        drift_pairs=drift_pairs,
        luna_pairs=luna_pairs,
        node_totals=dict(sorted(node_totals.items())),
    )


def _sharing_by_year(
    db: sqlite3.Connection, data: set[int], code: set[int]
) -> dict[int, dict[str, int]]:
    rows = db.execute("SELECT id,collection_year FROM papers").fetchall()
    out: dict[int, dict[str, int]] = {
        year: {"papers": 0, "data": 0, "code": 0, "data_and_code": 0} for year in YEARS
    }
    for row in rows:
        year = int(row["collection_year"])
        bucket = out.setdefault(year, {"papers": 0, "data": 0, "code": 0, "data_and_code": 0})
        bucket["papers"] += 1
        paper_id = int(row["id"])
        if paper_id in data:
            bucket["data"] += 1
        if paper_id in code:
            bucket["code"] += 1
        if paper_id in data and paper_id in code:
            bucket["data_and_code"] += 1
    return dict(sorted(out.items()))


def _drift_pairs(
    db: sqlite3.Connection, papers: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paired old single-draw R against the new 3-repetition mean, identical PDFs only."""
    try:
        old_experiment = _experiment_row(db, DRIFT_EXPERIMENT)
    except ValueError:
        return [], {"ok": False, "note": f"{DRIFT_EXPERIMENT} absent"}
    old = {
        int(row["paper_id"]): row
        for row in db.execute(
            "SELECT paper_id,reproducibility_index r,content_score rc,source_pdf_sha256 "
            "FROM ara_runs WHERE experiment_id=? AND status='succeeded'",
            (old_experiment["id"],),
        )
    }
    new_sha = {
        int(row["paper_id"]): row["source_pdf_sha256"]
        for row in db.execute(
            "SELECT DISTINCT paper_id,source_pdf_sha256 FROM ara_runs r "
            "JOIN ara_experiments e ON e.id=r.experiment_id "
            "WHERE e.name=? AND r.status='succeeded'",
            (BENCHMARK_EXPERIMENT,),
        )
    }
    by_id = {p["paper_id"]: p for p in papers}
    pairs: list[dict[str, Any]] = []
    mismatched: list[int] = []
    for paper_id, row in sorted(old.items()):
        paper = by_id.get(paper_id)
        if paper is None or paper_id not in new_sha:
            continue
        if row["source_pdf_sha256"] != new_sha[paper_id]:
            mismatched.append(paper_id)
            continue
        pairs.append(
            {
                "paper_id": paper_id,
                "year": paper["year"],
                "old_r": float(row["r"]),
                "new_r": paper["r"],
                "new_r_sd": paper["r_sd"],
                "difference": paper["r"] - float(row["r"]),
            }
        )
    return pairs, {
        "ok": len(pairs) == 64 and not mismatched,
        "pairs": len(pairs),
        "expected_pairs": 64,
        "source_pdf_mismatches": mismatched,
        "note": "64 papers qualified on 2026-09-18; a different count must be reported",
    }


def _luna_pairs(
    db: sqlite3.Connection, papers: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    try:
        probe = _experiment_row(db, LUNA_EXPERIMENT)
    except ValueError:
        return []
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in db.execute(
        "SELECT paper_id,reproducibility_index r FROM ara_runs "
        "WHERE experiment_id=? AND status='succeeded'",
        (probe["id"],),
    ):
        grouped[int(row["paper_id"])].append(float(row["r"]))
    by_id = {p["paper_id"]: p for p in papers}
    pairs = []
    for paper_id, values in sorted(grouped.items()):
        paper = by_id.get(paper_id)
        if paper is None:
            continue
        pairs.append(
            {
                "paper_id": paper_id,
                "terra_r": paper["r"],
                "luna_r": st.fmean(values),
                "difference": st.fmean(values) - paper["r"],
            }
        )
    return pairs


# --------------------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------------------


@dataclass
class Context:
    data: Dataset
    seed: int
    resamples: int

    def rng(self, analysis_id: str) -> np.random.Generator:
        return _rng(self.seed, analysis_id)


def _log_length(paper: Mapping[str, Any]) -> float | None:
    chars = paper.get("markdown_chars")
    return math.log10(float(chars)) if chars else None


def a1_1(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    rng = ctx.rng("A1.1")
    return {
        "pooled": _describe(p["r"] for p in papers),
        "pooled_mean_ci": _boot_ci([p["r"] for p in papers], rng, ctx.resamples),
        "by_year": {
            str(year): {
                **_describe(p["r"] for p in papers if p["year"] == year),
                "mean_ci": _boot_ci(
                    [p["r"] for p in papers if p["year"] == year],
                    ctx.rng(f"A1.1:{year}"),
                    ctx.resamples,
                ),
            }
            for year in sorted({p["year"] for p in papers})
        },
    }


def a1_2(ctx: Context) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for paper in ctx.data.papers:
        groups[str(paper["claim_tier"])].append(paper["r"])
    result = _kruskal(groups, ctx.rng("A1.2"), ctx.resamples)
    result["order_by_median"] = [
        key
        for key, _ in sorted(
            ((k, v["median"]) for k, v in result["per_group"].items() if v["median"] is not None),
            key=lambda kv: kv[1],
        )
    ]
    return result


def a1_3(ctx: Context) -> dict[str, Any]:
    long = [p["r"] for p in ctx.data.papers if p["kind"] == "long"]
    short = [p["r"] for p in ctx.data.papers if p["kind"] == "short"]
    return _mannwhitney(long, short, ctx.rng("A1.3"), ctx.resamples, ("long", "short"))


def a1_4(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    rc = np.asarray([p["content_score"] for p in papers], dtype=float)
    rs = np.asarray([p["structural_score"] for p in papers], dtype=float)
    r = np.asarray([p["r"] for p in papers], dtype=float)
    ceiling = int((rs >= 0.999).sum())
    on_rc = _ols(r, np.column_stack([np.ones(r.size), rc]), ("intercept", "content_score"))
    on_rs = _ols(r, np.column_stack([np.ones(r.size), rs]), ("intercept", "structural_score"))
    on_both = _ols(
        r,
        np.column_stack([np.ones(r.size), rc, rs]),
        ("intercept", "content_score", "structural_score"),
    )
    variance_from_rc = on_rc["r_squared"]
    return {
        "content_score": _describe(rc),
        "structural_score": _describe(rs),
        "structural_ceiling": {
            "at_1_000": ceiling,
            "of": int(rs.size),
            "fraction": ceiling / rs.size,
            "note": "R_s saturates; R_c never does, so R_s cannot carry a ranking",
        },
        "r_vs_content": _spearman(r, rc, ctx.rng("A1.4:rc"), ctx.resamples, "R ~ R_c"),
        "r_vs_structural": _spearman(r, rs, ctx.rng("A1.4:rs"), ctx.resamples, "R ~ R_s"),
        "content_vs_structural": _spearman(
            rc, rs, ctx.rng("A1.4:rcrs"), ctx.resamples, "R_c ~ R_s"
        ),
        "variance_explained": {
            "content_only_r_squared": variance_from_rc,
            "structural_only_r_squared": on_rs["r_squared"],
            "both_r_squared": on_both["r_squared"],
            "content_share_of_explained": (
                variance_from_rc / on_both["r_squared"] if on_both["r_squared"] else None
            ),
        },
        "regression_on_both": on_both,
        "interpretation": (
            "R_c alone explains {:.1%} of the between-paper variance in R; reporting R and "
            "R_c as separate findings would report one finding twice"
        ).format(variance_from_rc or 0.0),
    }


def a1_5(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    r = np.asarray([p["r"] for p in papers], dtype=float)
    columns = {score: np.asarray([p[score] for p in papers], dtype=float) for score in STAGE_SCORES}
    per_score = {
        score: {
            **_describe(col),
            "missing": int(np.isnan(col).sum()),
            "variance": float(np.nanvar(col, ddof=1)),
            "spearman_with_r": _spearman(
                r.tolist(),
                [None if _isnan(v) else float(v) for v in col],
                ctx.rng(f"A1.5:{score}"),
                ctx.resamples,
                f"R ~ {score}",
            ),
        }
        for score, col in columns.items()
    }
    # A stage score is null when the paper yielded no node of that class in a repetition,
    # which is missing rather than zero, so the joint regression is on complete cases.
    complete = np.all(
        np.isfinite(np.column_stack([columns[score] for score in STAGE_SCORES])), axis=1
    )

    def z(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / (values.std(ddof=1) or 1.0)

    standardised = _ols(
        z(r[complete]),
        np.column_stack(
            [np.ones(int(complete.sum()))]
            + [z(columns[score][complete]) for score in STAGE_SCORES]
        ),
        ("intercept", *STAGE_SCORES),
    )
    carrier = max(
        per_score,
        key=lambda s: abs(per_score[s]["spearman_with_r"]["rho"] or 0.0),
    )
    return {
        "per_score": per_score,
        "standardised_regression": standardised,
        "complete_cases": int(complete.sum()),
        "dropped_incomplete": int((~complete).sum()),
        "missingness_note": (
            "a null stage score means the paper produced no node of that class, not a "
            "score of zero; those papers are dropped from the joint regression only"
        ),
        "variance_carrier": carrier,
        "note": "standardised betas are on z-scored R and z-scored stage scores",
    }


def _year_slope(
    values: Sequence[float], years: Sequence[int]
) -> tuple[dict[str, Any], float]:
    y = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones(y.size), np.asarray(years, dtype=float) - 2021.0])
    fit = _ols(y, design, ("intercept", "year"))
    return fit, float(y.std(ddof=1))


def a2_1(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    years = [p["year"] for p in papers]
    r = [p["r"] for p in papers]
    share = [1.0 if p["verified_data"] else 0.0 for p in papers]
    r_fit, r_sd = _year_slope(r, years)
    share_fit, share_sd = _year_slope(share, years)
    r_slope = r_fit["terms"]["year"]["coefficient"]
    share_slope = share_fit["terms"]["year"]["coefficient"]

    r_array = np.asarray(r)
    share_array = np.asarray(share)
    year_array = np.asarray(years, dtype=float) - 2021.0

    def standardised(idx: np.ndarray) -> tuple[float | None, float | None]:
        design = np.column_stack([np.ones(idx.size), year_array[idx]])
        rv, sv = r_array[idx], share_array[idx]
        if rv.std(ddof=1) == 0 or sv.std(ddof=1) == 0:
            return None, None
        br = np.linalg.lstsq(design, rv, rcond=None)[0][1] / rv.std(ddof=1)
        bs = np.linalg.lstsq(design, sv, rcond=None)[0][1] / sv.std(ddof=1)
        return float(br), float(bs)

    rng = ctx.rng("A2.1:contrast")
    r_std_reps: list[float] = []
    share_std_reps: list[float] = []
    gap_reps: list[float] = []
    for _ in range(ctx.resamples):
        idx = rng.integers(0, len(papers), size=len(papers))
        br, bs = standardised(idx)
        if br is None or bs is None:
            continue
        r_std_reps.append(br)
        share_std_reps.append(bs)
        gap_reps.append(bs - br)

    def band(reps: Sequence[float]) -> dict[str, Any]:
        if len(reps) < 2:
            return {"ci_low": None, "ci_high": None}
        low, high = (float(v) for v in np.quantile(np.asarray(reps), (0.025, 0.975)))
        return {"ci_low": low, "ci_high": high, "resamples": len(reps)}

    r_std = r_slope / r_sd
    share_std = share_slope / share_sd
    r_band = band(r_std_reps)
    share_band = band(share_std_reps)
    gap_band = band(gap_reps)

    corpus = ctx.data.sharing_corpus
    trend = _cochran_armitage(
        [corpus[y]["data"] for y in sorted(corpus)],
        [corpus[y]["papers"] for y in sorted(corpus)],
        [float(y) - 2021.0 for y in sorted(corpus)],
    )
    yearly_mean_r = {
        year: st.fmean([p["r"] for p in papers if p["year"] == year])
        for year in sorted({p["year"] for p in papers})
    }
    return {
        "hypothesis": (
            "R is flat across 2021-2026 while verified data sharing rises; the contrast "
            "between the two slopes is the result, not either slope alone"
        ),
        "r_regression_on_year": r_fit,
        "r_between_paper_sd": r_sd,
        "r_slope_sd_per_year": {
            "estimate": r_std,
            **r_band,
            "raw_slope_per_year": r_slope,
        },
        "sharing_regression_on_year": share_fit,
        "sharing_slope_proportion_per_year": {
            "estimate": share_slope,
            "ci_low": share_fit["terms"]["year"]["ci_low"],
            "ci_high": share_fit["terms"]["year"]["ci_high"],
        },
        "sharing_slope_sd_per_year": {"estimate": share_std, **share_band},
        "contrast": {
            "gap_sd_per_year": share_std - r_std,
            **gap_band,
            "r_ci_excludes_sharing_slope": (
                None
                if r_band["ci_low"] is None
                else not (r_band["ci_low"] <= share_std <= r_band["ci_high"])
            ),
            "gap_ci_excludes_zero": (
                None
                if gap_band["ci_low"] is None
                else not (gap_band["ci_low"] <= 0.0 <= gap_band["ci_high"])
            ),
            "units": (
                "both slopes divided by the between-paper SD of their own outcome, so they "
                "are comparable; raw units are reported alongside"
            ),
            "framing": (
                "if R rises slightly the claim is not 'no improvement' but that describing "
                "improved far less than sharing did"
            ),
        },
        "spearman_year_vs_paper_r": _spearman(
            [float(y) for y in years], r, ctx.rng("A2.1:paper"), ctx.resamples, "year ~ R"
        ),
        "spearman_year_vs_yearly_mean_r": _spearman(
            [float(y) for y in sorted(yearly_mean_r)],
            [yearly_mean_r[y] for y in sorted(yearly_mean_r)],
            ctx.rng("A2.1:yearly"),
            ctx.resamples,
            "year ~ mean R (six points)",
        ),
        "kruskal_across_years": _kruskal(
            {str(y): [p["r"] for p in papers if p["year"] == y] for y in sorted(set(years))},
            ctx.rng("A2.1:kw"),
            ctx.resamples,
        ),
        "cochran_armitage_sharing_corpus": trend,
        "sharing_by_year": {
            str(year): {
                **corpus[year],
                "proportion": corpus[year]["data"] / corpus[year]["papers"]
                if corpus[year]["papers"]
                else None,
                "analysis_set_papers": sum(1 for p in papers if p["year"] == year),
                "analysis_set_data": sum(
                    1 for p in papers if p["year"] == year and p["verified_data"]
                ),
            }
            for year in sorted(corpus)
        },
        "mean_r_by_year": {str(y): v for y, v in yearly_mean_r.items()},
        "denominators": (
            "R slope and the standardised sharing slope use the 471 analysed papers so the "
            "contrast is within one sample; the Cochran-Armitage test uses all 506 corpus "
            "papers, which is where the 1/2/6/11/21/17 counts come from"
        ),
    }


def a2_2(ctx: Context) -> dict[str, Any]:
    papers = [p for p in ctx.data.papers if _log_length(p) is not None]
    tiers = sorted({str(p["claim_tier"]) for p in papers})[1:]
    names = ["intercept", "year", "log10_markdown_chars", "kind_short", "total_node_count"]
    names += [f"claim_{t.split()[0]}" for t in tiers]
    rows = []
    for paper in papers:
        row = [
            1.0,
            float(paper["year"]) - 2021.0,
            _log_length(paper) or 0.0,
            1.0 if paper["kind"] == "short" else 0.0,
            float(paper["total_node_count"]),
        ]
        row += [1.0 if str(paper["claim_tier"]) == tier else 0.0 for tier in tiers]
        rows.append(row)
    y = np.asarray([p["r"] for p in papers], dtype=float)
    design = np.asarray(rows, dtype=float)
    uncontrolled = _ols(y, design[:, :2], ("intercept", "year"))
    controlled = _ols(y, design, names)
    year_only = uncontrolled["terms"]["year"]
    year_adjusted = controlled["terms"]["year"]
    length_fit = _ols(
        y, design[:, [0, 2]], ("intercept", "log10_markdown_chars")
    )
    return {
        "uncontrolled": uncontrolled,
        "controlled": controlled,
        "year_coefficient": {
            "before": year_only,
            "after": year_adjusted,
            "attenuation": (
                1.0 - year_adjusted["coefficient"] / year_only["coefficient"]
                if year_only["coefficient"]
                else None
            ),
            "sign_flipped": year_only["coefficient"] * year_adjusted["coefficient"] < 0,
        },
        "length_alone": length_fit,
        "length_vs_r": _spearman(
            y,
            [_log_length(p) for p in papers],
            ctx.rng("A2.2:length"),
            ctx.resamples,
            "R ~ log10 markdown chars",
        ),
        "reference_claim_tier": sorted({str(p["claim_tier"]) for p in papers})[0],
        "interpretation_rule": (
            "if the year coefficient loses most of its magnitude once length enters, the "
            "trend is an artefact of papers getting longer and must be reported as such"
        ),
    }


def _split_by_flag(
    papers: Sequence[Mapping[str, Any]], key: str, positive: Callable[[Any], bool]
) -> tuple[list[float], list[float]]:
    yes = [p["r"] for p in papers if p.get(key) is not None and positive(p[key])]
    no = [p["r"] for p in papers if p.get(key) is not None and not positive(p[key])]
    return yes, no


def a2_3(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    signals: dict[str, tuple[list[float], list[float], tuple[str, str]]] = {}
    for name, key, predicate, labels in (
        ("open_data_stated", "open_data", lambda v: v == "Yes", ("Yes", "not Yes")),
        ("open_materials_full", "open_materials", lambda v: v == "Full", ("Full", "not Full")),
        ("verified_retrievable_data", "verified_data", bool, ("verified", "none")),
        ("fetched_code", "verified_code", bool, ("fetched", "none")),
        ("readme_present", "readme", lambda v: v == "Yes", ("Yes", "not Yes")),
        ("license_stated", "license_stated", lambda v: v == "Yes", ("Yes", "not Yes")),
        (
            "data_documentation",
            "data_documentation",
            lambda v: v in ("Yes", "Partial"),
            ("Yes/Partial", "No"),
        ),
    ):
        yes, no = _split_by_flag(papers, key, predicate)
        signals[name] = (yes, no, labels)
    tests = {
        name: _mannwhitney(yes, no, ctx.rng(f"A2.3:{name}"), ctx.resamples, labels)
        for name, (yes, no, labels) in signals.items()
    }
    with_e = [p for p in papers if p["e_score"] is not None]
    environment = {
        "papers_with_e": len(with_e),
        "e_score": _describe(p["e_score"] for p in with_e),
        "distinct_e_values": len({round(p["e_score"], 6) for p in with_e}),
        "spearman_r_vs_e": _spearman(
            [p["r"] for p in with_e],
            [p["e_score"] for p in with_e],
            ctx.rng("A2.3:e"),
            ctx.resamples,
            "R ~ E",
        ),
        "d1_prediction": (
            "The prediction is that E is too concentrated to compete as a signal; the distribution "
            "above is the test, not the correlation alone"
        ),
    }
    sized = [p for p in papers if _log_length(p) is not None]
    partial = _partial_spearman(
        [p["r"] for p in sized],
        [1.0 if p["verified_data"] else 0.0 for p in sized],
        [
            [float(p["year"]) for p in sized],
            [_log_length(p) or 0.0 for p in sized],
        ],
        ctx.rng("A2.3:partial"),
        ctx.resamples,
        "R ~ verified data | year, length",
    )
    biggest = max(
        (name for name in tests if tests[name].get("rank_biserial_r") is not None),
        key=lambda name: abs(tests[name]["rank_biserial_r"]),
        default=None,
    )
    return {
        "tests": tests,
        "environment": environment,
        "partial_correlation": partial,
        "largest_effect": biggest,
        "interpretation_rule": (
            "if R and sharing are close to independent, the L1/L2 rung of the ladder is "
            "doing real work that artifact presence does not already capture"
        ),
    }


def a3_1(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    pooled = stability_stats({p["paper_id"]: p["r_values"] for p in papers})
    by_year = {
        str(year): stability_stats(
            {p["paper_id"]: p["r_values"] for p in papers if p["year"] == year}
        )
        for year in sorted({p["year"] for p in papers})
    }
    by_experiment = {
        name: stability_stats(
            {p["paper_id"]: p["r_values"] for p in papers if p["experiment"] == name}
        )
        for name in sorted({p["experiment"] for p in papers})
    }

    def trim(block: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in block.items() if k not in ("means", "sds")}

    return {
        "pooled": trim(pooled),
        "pooled_sd_mean_ci": _boot_ci(pooled["sds"], ctx.rng("A3.1"), ctx.resamples),
        "by_year": {k: trim(v) for k, v in by_year.items()},
        "by_experiment": {k: trim(v) for k, v in by_experiment.items()},
        "ara_reference_band": [0.02, 0.09],
        "note": (
            "`scripts/make_stage2_report.py` imports `stability_stats` from this module, so "
            "the two documents cannot disagree"
        ),
    }


def a3_2(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    r = np.asarray([p["r"] for p in papers], dtype=float)
    sd = np.asarray([p["r_sd"] for p in papers], dtype=float)
    centred = r - r.mean()
    quadratic = _ols(
        sd,
        np.column_stack([np.ones(r.size), centred, centred**2]),
        ("intercept", "r_centred", "r_centred_squared"),
    )
    quintiles = np.quantile(r, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins: dict[str, Any] = {}
    for i in range(5):
        low, high = quintiles[i], quintiles[i + 1]
        mask = (r >= low) & (r <= high) if i == 4 else (r >= low) & (r < high)
        bins[f"q{i + 1}"] = {
            "r_range": [float(low), float(high)],
            "n": int(mask.sum()),
            "mean_sd": float(sd[mask].mean()) if mask.any() else None,
        }
    return {
        "spearman_sd_vs_r": _spearman(sd, r, ctx.rng("A3.2"), ctx.resamples, "SD ~ R"),
        "quadratic_regression": quadratic,
        "mid_range_least_stable": (
            quadratic["terms"]["r_centred_squared"]["coefficient"] < 0
            and quadratic["terms"]["r_centred_squared"]["p_value"] is not None
            and quadratic["terms"]["r_centred_squared"]["p_value"] < 0.05
        ),
        "sd_by_r_quintile": bins,
    }


def a3_3(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    sd = [p["r_sd"] for p in papers]
    return {
        "spearman_sd_vs_length": _spearman(
            sd,
            [_log_length(p) for p in papers],
            ctx.rng("A3.3:length"),
            ctx.resamples,
            "SD ~ log10 markdown chars",
        ),
        "spearman_sd_vs_node_count": _spearman(
            sd,
            [p["total_node_count"] for p in papers],
            ctx.rng("A3.3:nodes"),
            ctx.resamples,
            "SD ~ total node count",
        ),
    }


def a3_4(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    per_class = {
        cls: [p["node_jaccard"][cls] for p in papers if p["node_jaccard"][cls] is not None]
        for cls in ("source", "process", "sink")
    }
    return {
        "mean_pairwise_jaccard": {
            cls: {
                **_describe(values),
                "mean_ci": _boot_ci(values, ctx.rng(f"A3.4:{cls}"), ctx.resamples),
            }
            for cls, values in per_class.items()
        },
        "identical_node_sets": {
            cls: sum(1 for v in values if v >= 0.999) for cls, values in per_class.items()
        },
        "method": (
            "Jaccard on node names within a paper, averaged over the three repetition pairs. "
            "Graph edit distance is not computed: node names are free text, so an edit "
            "distance would measure paraphrase rather than structure"
        ),
    }


def a3_5(ctx: Context) -> dict[str, Any]:
    pairs = ctx.data.drift_pairs
    if len(pairs) < 3:
        return {"n": len(pairs), "status": "insufficient overlap"}
    old = np.asarray([p["old_r"] for p in pairs], dtype=float)
    new = np.asarray([p["new_r"] for p in pairs], dtype=float)
    differences = new - old
    wilcoxon = sps.wilcoxon(new, old)
    mean_difference = float(differences.mean())
    sd_difference = float(differences.std(ddof=1))
    return {
        "n": len(pairs),
        "experiments": [DRIFT_EXPERIMENT, BENCHMARK_EXPERIMENT],
        "days_apart": 7,
        "wilcoxon": {
            "statistic": float(wilcoxon.statistic),
            "p_value": float(wilcoxon.pvalue),
            "alternative": "two-sided",
        },
        "mean_difference": mean_difference,
        "mean_difference_ci": _boot_ci(differences.tolist(), ctx.rng("A3.5"), ctx.resamples),
        "bland_altman": {
            "bias": mean_difference,
            "sd_of_differences": sd_difference,
            "limits_of_agreement": [
                mean_difference - 1.96 * sd_difference,
                mean_difference + 1.96 * sd_difference,
            ],
            "outside_limits": int(
                (
                    (differences < mean_difference - 1.96 * sd_difference)
                    | (differences > mean_difference + 1.96 * sd_difference)
                ).sum()
            ),
        },
        "spearman_ranking": _spearman(
            old.tolist(), new.tolist(), ctx.rng("A3.5:rank"), ctx.resamples, "old R ~ new mean R"
        ),
        "median_within_paper_sd_new": float(np.median([p["new_r_sd"] for p in pairs])),
        "caveat": (
            "the 2026-09-11 runs are single draws, so disagreement on the order of the "
            "within-paper SD is expected noise rather than drift; this is a cross-time "
            "replication of a hosted model, not a model comparison"
        ),
    }


def _tau_curve(
    papers: Sequence[Mapping[str, Any]], key: str, taus: Sequence[float]
) -> list[dict[str, Any]]:
    values = np.asarray([p[key] for p in papers], dtype=float)
    years = np.asarray([p["year"] for p in papers])
    curve = []
    for tau in taus:
        at_l2 = values >= tau
        curve.append(
            {
                "tau": round(float(tau), 2),
                "l2": int(at_l2.sum()),
                "l1": int((~at_l2).sum()),
                "l2_fraction": float(at_l2.mean()),
                "by_year": {
                    str(int(year)): int(at_l2[years == year].sum())
                    for year in sorted(set(years.tolist()))
                },
            }
        )
    return curve


def a4_1(ctx: Context) -> dict[str, Any]:
    taus = [0.30 + 0.01 * i for i in range(41)]
    curve = _tau_curve(ctx.data.papers, "r", taus)
    counts = np.asarray([point["l2"] for point in curve], dtype=float)
    slope = np.abs(np.gradient(counts))
    inflection = int(np.argmax(slope))
    return {
        "tau_range": [0.30, 0.70],
        "step": 0.01,
        "statistic": "mean R across three repetitions",
        "curve": curve,
        "steepest_point": {
            "tau": curve[inflection]["tau"],
            "papers_per_0_01_tau": float(slope[inflection]),
            "l2": curve[inflection]["l2"],
        },
        "span": {
            "l2_at_0_30": curve[0]["l2"],
            "l2_at_0_70": curve[-1]["l2"],
            "papers_moved": curve[0]["l2"] - curve[-1]["l2"],
        },
        "no_threshold_selected": True,
        "note": (
            "there is no calibrated tau_R; this curve converts a "
            "missing calibration into a reported sensitivity and selects nothing"
        ),
    }


def a4_2(ctx: Context) -> dict[str, Any]:
    taus = [0.30 + 0.01 * i for i in range(41)]
    variants = {
        statistic: _tau_curve(ctx.data.papers, key, taus)
        for statistic, key in (("mean", "r"), ("minimum", "r_min"), ("maximum", "r_max"))
    }
    band = [
        {
            "tau": variants["mean"][i]["tau"],
            "l2_mean": variants["mean"][i]["l2"],
            "l2_min_draw": variants["minimum"][i]["l2"],
            "l2_max_draw": variants["maximum"][i]["l2"],
            "noise_width": variants["maximum"][i]["l2"] - variants["minimum"][i]["l2"],
        }
        for i in range(len(taus))
    ]
    widest = max(band, key=lambda point: point["noise_width"])
    return {
        "variants": variants,
        "noise_band": band,
        "widest_noise_band": widest,
        "mean_noise_width": st.fmean(point["noise_width"] for point in band),
        "interpretation": (
            "the band is how many papers change rung purely because a different repetition "
            "was drawn; compare it against the number moved by the threshold itself (A4.1)"
        ),
    }


def a5_1(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    classes = {
        "source": [p["source_node_score"] for p in papers],
        "process": [p["process_node_score"] for p in papers],
        "sink": [p["sink_node_score"] for p in papers],
    }
    summary = {
        cls: {
            **_describe(values),
            "mean_ci": _boot_ci(values, ctx.rng(f"A5.1:{cls}"), ctx.resamples),
        }
        for cls, values in classes.items()
    }
    ranked = sorted(
        (cls for cls in summary if summary[cls].get("mean") is not None),
        key=lambda cls: summary[cls]["mean"],
    )
    return {
        "scale": "node reproducibility_score, integer percentage in [0, 100]",
        "pooling": "repetitions pooled to a paper-level mean, then bootstrapped over papers",
        "by_node_class": summary,
        "bottleneck": ranked[0] if ranked else None,
        "ranking_worst_first": ranked,
    }


def _gap_block(
    rows: Sequence[Mapping[str, Any]], mentioned: str, required: str, capped: str
) -> dict[str, Any]:
    total_required = sum(float(r[required]) for r in rows)
    total_capped = sum(float(r[capped]) for r in rows)
    total_mentioned = sum(float(r[mentioned]) for r in rows)
    return {
        "n_papers": len(rows),
        "required": total_required,
        "mentioned": total_mentioned,
        "counted_towards_completion": total_capped,
        "completion": total_capped / total_required if total_required else None,
    }


def a5_2(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    parameters = _gap_block(papers, "parameters_mentioned", "parameters_required", "parameters_capped")
    tools = _gap_block(papers, "tools_mentioned", "tools_required", "tools_capped")
    per_paper_parameters = [p["parameter_completion"] for p in papers]
    per_paper_tools = [p["tool_completion"] for p in papers]
    return {
        "definition": (
            "completion = sum(min(mentioned, required)) / sum(required) over nodes_process, "
            "pooled across repetitions; capping at the required count keeps a node that "
            "names more parameters than it needs from offsetting one that names too few"
        ),
        "parameters": {
            **parameters,
            "per_paper": _describe(per_paper_parameters),
            "per_paper_mean_ci": _boot_ci(
                per_paper_parameters, ctx.rng("A5.2:params"), ctx.resamples
            ),
        },
        "tools": {
            **tools,
            "per_paper": _describe(per_paper_tools),
            "per_paper_mean_ci": _boot_ci(per_paper_tools, ctx.rng("A5.2:tools"), ctx.resamples),
        },
        "excluded": {
            "process_nodes_without_required_parameters": ctx.data.node_totals.get(
                "process_nodes_without_required_parameters", 0
            ),
            "process_nodes_naming_more_than_required": ctx.data.node_totals.get(
                "process_nodes_over_specified", 0
            ),
        },
        "by_year": {
            str(year): {
                "parameters": _gap_block(
                    [p for p in papers if p["year"] == year],
                    "parameters_mentioned",
                    "parameters_required",
                    "parameters_capped",
                ),
                "tools": _gap_block(
                    [p for p in papers if p["year"] == year],
                    "tools_mentioned",
                    "tools_required",
                    "tools_capped",
                ),
            }
            for year in sorted({p["year"] for p in papers})
        },
        "by_claim_tier": {
            str(tier): {
                "parameters": _gap_block(
                    [p for p in papers if str(p["claim_tier"]) == tier],
                    "parameters_mentioned",
                    "parameters_required",
                    "parameters_capped",
                ),
                "tools": _gap_block(
                    [p for p in papers if str(p["claim_tier"]) == tier],
                    "tools_mentioned",
                    "tools_required",
                    "tools_capped",
                ),
            }
            for tier in sorted({str(p["claim_tier"]) for p in papers})
        },
        "headline": (
            "LAK papers name {:.1%} of the parameters their own described methods require"
        ).format(parameters["completion"] or 0.0),
    }


def a5_3(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    pooled = Counter()
    for paper in papers:
        pooled.update({int(k): v for k, v in paper["algorithm_clarity_counts"].items()})
    total = sum(pooled.values())
    return {
        "scale": "algorithm_clarity, integer in {1, 2, 3, 4}; 1=missing, 4=sufficient",
        "node_level_counts": {str(level): int(pooled.get(level, 0)) for level in CLARITY_DOMAIN},
        "node_level_fractions": {
            str(level): pooled.get(level, 0) / total if total else None for level in CLARITY_DOMAIN
        },
        "nodes_counted": total,
        "repetitions_pooled": True,
        "paper_level_mean": _describe(p["algorithm_clarity_mean"] for p in papers),
        "spearman_with_content_score": _spearman(
            [p["algorithm_clarity_mean"] for p in papers],
            [p["content_score"] for p in papers],
            ctx.rng("A5.3"),
            ctx.resamples,
            "algorithm clarity ~ R_c",
        ),
    }


def a5_4(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    metrics = {
        key: {
            **_describe(p[key] for p in papers),
            "mean_ci": _boot_ci([p[key] for p in papers], ctx.rng(f"A5.4:{key}"), ctx.resamples),
            "at_1_000": sum(1 for p in papers if (p[key] or 0.0) >= 0.999),
            "below_1_000": sum(1 for p in papers if (p[key] or 0.0) < 0.999),
        }
        for key in WIRING_KEYS
    }
    broken = [p for p in papers if (p["source_to_sink_reachability"] or 0.0) < 0.999]
    intact = [p for p in papers if (p["source_to_sink_reachability"] or 0.0) >= 0.999]
    return {
        "metrics": metrics,
        "broken_evidence_chain": {
            "papers": len(broken),
            "of": len(papers),
            "fraction": len(broken) / len(papers) if papers else None,
            "meaning": (
                "reachability below 1.0 means the paper has, as described, at least one "
                "reported result that cannot be traced back to a named data source"
            ),
        },
        "spearman_reachability_vs_r": _spearman(
            [p["source_to_sink_reachability"] for p in papers],
            [p["r"] for p in papers],
            ctx.rng("A5.4:r"),
            ctx.resamples,
            "reachability ~ R",
        ),
        "reachability_by_sharing": _mannwhitney(
            [p["source_to_sink_reachability"] for p in papers if p["verified_data"]],
            [p["source_to_sink_reachability"] for p in papers if not p["verified_data"]],
            ctx.rng("A5.4:share"),
            ctx.resamples,
            ("verified data", "no verified data"),
        ),
        "r_by_chain_integrity": _mannwhitney(
            [p["r"] for p in intact],
            [p["r"] for p in broken],
            ctx.rng("A5.4:split"),
            ctx.resamples,
            ("reachability 1.0", "reachability < 1.0"),
        ),
    }


def a5_5(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    clarity = Counter()
    validity = Counter()
    sink_types = Counter()
    for paper in papers:
        clarity.update({int(k): v for k, v in paper["statement_clarity_counts"].items()})
        validity.update(paper["statement_validity_counts"])
        sink_types.update(paper["sink_type_counts"])
    clarity_total = sum(clarity.values())
    validity_total = sum(validity.values())
    unclear = [p for p in papers if (p["statement_clarity_mean"] or 4.0) < 2.5]
    return {
        "statement_clarity": {
            "counts": {str(level): int(clarity.get(level, 0)) for level in CLARITY_DOMAIN},
            "fractions": {
                str(level): clarity.get(level, 0) / clarity_total if clarity_total else None
                for level in CLARITY_DOMAIN
            },
            "paper_level_mean": _describe(p["statement_clarity_mean"] for p in papers),
        },
        "statement_validity": {
            "counts": {key: int(validity.get(key, 0)) for key in VALIDITY_DOMAIN},
            "fractions": {
                key: validity.get(key, 0) / validity_total if validity_total else None
                for key in VALIDITY_DOMAIN
            },
        },
        "sink_types": {key: int(sink_types.get(key, 0)) for key in SINK_TYPE_DOMAIN},
        "nodes_counted": clarity_total,
        "stage3_preregistered_comparison": {
            "papers_ara_marks_unclear": len(unclear),
            "threshold": "paper-level mean statement_clarity < 2.5",
            "paper_ids": sorted(p["paper_id"] for p in unclear),
            "prediction": (
                "Stage 3 claim extraction, which reads the papers independently with a "
                "different prompt, should fail or return low-confidence claims "
                "disproportionately on these papers. Recorded before Stage 3 runs, so the "
                "comparison is confirmatory for both instruments rather than post hoc"
            ),
        },
    }


def a5_6(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    sized = [p for p in papers if _log_length(p) is not None]
    per_class = {
        count: _spearman(
            [p[count] for p in papers],
            [p["r"] for p in papers],
            ctx.rng(f"A5.6:{count}"),
            ctx.resamples,
            f"{count} ~ R",
        )
        for count in (
            "source_node_count",
            "method_node_count",
            "experiment_node_count",
            "sink_node_count",
            "total_node_count",
        )
    }
    return {
        "node_counts": {
            count: _describe(p[count] for p in papers)
            for count in (
                "source_node_count",
                "method_node_count",
                "experiment_node_count",
                "sink_node_count",
                "total_node_count",
            )
        },
        "spearman_with_r": per_class,
        "spearman_nodes_vs_length": _spearman(
            [p["total_node_count"] for p in sized],
            [_log_length(p) for p in sized],
            ctx.rng("A5.6:length"),
            ctx.resamples,
            "total node count ~ log10 markdown chars",
        ),
        "partial_nodes_vs_r_given_length": _partial_spearman(
            [p["r"] for p in sized],
            [p["total_node_count"] for p in sized],
            [[_log_length(p) or 0.0 for p in sized]],
            ctx.rng("A5.6:partial"),
            ctx.resamples,
            "R ~ node count | length",
        ),
        "interpretation_rule": (
            "a negative correlation is itself a finding: ambitious pipelines would be the "
            "worst described. A positive one supports the objection that R tracks length"
        ),
    }


def a5_7(ctx: Context) -> dict[str, Any]:
    papers = ctx.data.papers
    stage1_known = [p for p in papers if p["open_data"] is not None]
    license_known = [p for p in papers if p["license_stated"] is not None]

    def binary(flag: bool) -> str:
        return "yes" if flag else "no"

    # Both sides are reduced to the same yes/no label so the agreement diagonal lines up.
    availability_pairs = [
        (binary(p["any_source_open"]), binary(p["open_data"] == "Yes")) for p in stage1_known
    ]
    license_pairs = [
        (binary(p["any_source_licensed"]), binary(p["license_stated"] == "Yes"))
        for p in license_known
    ]
    retrieval_pairs = [
        (binary(p["any_source_open"]), binary(p["verified_data"])) for p in papers
    ]
    crosstab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for paper in stage1_known:
        crosstab[str(paper["ara_best_availability"])][str(paper["open_data"])] += 1
    return {
        "instruments": (
            "ARA reads data availability from nodes_source[].availability, .license and "
            ".url; Stage 1's open_data and license come from a different prompt over the "
            "same paper. Neither is ground truth"
        ),
        "availability_vs_open_data": _cohen_kappa(
            availability_pairs, ("ara_source_open", "stage1_open_data_yes")
        ),
        "availability_by_open_data_full": {
            key: dict(sorted(value.items())) for key, value in sorted(crosstab.items())
        },
        "license_vs_license": _cohen_kappa(
            license_pairs, ("ara_source_licensed", "stage1_license_yes")
        ),
        "availability_vs_verified_retrieval": _cohen_kappa(
            retrieval_pairs, ("ara_source_open", "verified_retrievable_data")
        ),
        "ara_says_open_only": sum(
            1 for p in stage1_known if p["any_source_open"] and p["open_data"] != "Yes"
        ),
        "stage1_says_open_only": sum(
            1 for p in stage1_known if not p["any_source_open"] and p["open_data"] == "Yes"
        ),
        "both_say_open": sum(
            1 for p in stage1_known if p["any_source_open"] and p["open_data"] == "Yes"
        ),
        "ara_open_but_nothing_retrieved": sum(
            1 for p in papers if p["any_source_open"] and not p["verified_data"]
        ),
        "reading": (
            "agreement is convergent validity for both instruments; disagreement localises "
            "which one is wrong on which paper, and the write-up must not imply one is right"
        ),
    }


def a6_1(ctx: Context) -> dict[str, Any]:
    pairs = ctx.data.luna_pairs
    if len(pairs) < 2:
        return {"n": len(pairs), "status": "probe absent"}
    terra = [p["terra_r"] for p in pairs]
    luna = [p["luna_r"] for p in pairs]
    differences = [p["difference"] for p in pairs]
    return {
        "status": "sensitivity note, not cross-model validation",
        "n": len(pairs),
        "experiment": LUNA_EXPERIMENT,
        "mean_shift": st.fmean(differences),
        "luna_higher_on_all_papers": all(d > 0 for d in differences),
        "papers": pairs,
        "spearman_ranking": _spearman(
            terra, luna, ctx.rng("A6.1"), ctx.resamples, "terra R ~ luna R"
        ),
        "decision_reference": (
            "luna was rejected as a second model because a systematic level shift "
            "makes any tau_R calibrated on one model meaningless on the other. With n=5 the "
            "rank correlation cannot be called at all"
        ),
    }


def a7_1(ctx: Context) -> dict[str, Any]:
    executable = [p for p in ctx.data.papers if p["verified_data"] and p["verified_code"]]
    return {
        "status": "pending",
        "reason": (
            "Stage 4 execution does not exist yet; no outcome table to read. No placeholder "
            "correlation is reported"
        ),
        "executable_candidates": len(executable),
        "paper_ids": sorted(p["paper_id"] for p in executable),
        "blocked_by": "data fetch retrieves nothing for six deposits",
    }


@dataclass(frozen=True)
class Analysis:
    id: str
    question: str
    exploratory: bool
    run: Callable[[Context], dict[str, Any]]


ANALYSES: tuple[Analysis, ...] = (
    Analysis("A1.1", "What does the distribution of R look like, pooled and per year?", True, a1_1),
    Analysis("A1.2", "Does R differ by claim type T1-T4?", True, a1_2),
    Analysis("A1.3", "Does R differ between long and short papers?", True, a1_3),
    Analysis("A1.4", "How do R_c and R_s relate, and how much of R is R_c alone?", True, a1_4),
    Analysis("A1.5", "Which stage score carries the variance in R?", True, a1_5),
    Analysis(
        "A2.1",
        "PRIMARY: is R flat across years while verified data sharing rises?",
        False,
        a2_1,
    ),
    Analysis("A2.2", "Does the year effect survive length, kind, claim mix and node count?", True, a2_2),
    Analysis("A2.3", "Is describing the same virtue as sharing?", True, a2_3),
    Analysis("A3.1", "How stable is R across repetitions, pooled and by year?", True, a3_1),
    Analysis("A3.2", "Does run-to-run SD depend on where R sits?", True, a3_2),
    Analysis("A3.3", "Does run-to-run SD depend on paper length or node count?", True, a3_3),
    Analysis("A3.4", "How stable is the extracted node set across repetitions?", True, a3_4),
    Analysis("A3.5", "Has the hosted model drifted over seven days on identical PDFs?", True, a3_5),
    Analysis("A4.1", "How sensitive is the L1/L2 split to an uncalibrated tau_R?", True, a4_1),
    Analysis("A4.2", "How much of the L1/L2 split is decided by run-to-run noise?", True, a4_2),
    Analysis("A5.1", "Where is the bottleneck: data, methods or results?", True, a5_1),
    Analysis("A5.2", "What fraction of required parameters and tools do papers name?", True, a5_2),
    Analysis("A5.3", "How is algorithm clarity distributed, and does it track R_c?", True, a5_3),
    Analysis("A5.4", "How many papers describe a claim that cannot be traced back to data?", True, a5_4),
    Analysis("A5.5", "How clearly are results stated? (pre-registered for Stage 3)", True, a5_5),
    Analysis("A5.6", "Is R a proxy for workflow complexity or paper length?", True, a5_6),
    Analysis("A5.7", "Do ARA and Stage 1 agree on data availability?", True, a5_7),
    Analysis("A6.1", "How far does R move if the model changes?", True, a6_1),
    Analysis("A7.1", "Does R predict execution outcome? (reserved for Stage 4)", True, a7_1),
)
ANALYSES_BY_ID = {analysis.id: analysis for analysis in ANALYSES}
NODE_ANALYSIS_PREFIX = "A5"


def select_analyses(only: str | None) -> list[Analysis]:
    if not only:
        return list(ANALYSES)
    wanted = [token.strip() for token in only.split(",") if token.strip()]
    chosen = [a for a in ANALYSES if any(a.id.startswith(token) for token in wanted)]
    unmatched = [
        token for token in wanted if not any(a.id.startswith(token) for a in ANALYSES)
    ]
    if unmatched:
        raise ValueError(f"No analysis matches {unmatched}; try `lakdb rstats list`")
    return chosen


def run_analyses(ctx: Context, analyses: Sequence[Analysis]) -> dict[str, Any]:
    return {
        analysis.id: {
            "id": analysis.id,
            "question": analysis.question,
            "exploratory": analysis.exploratory,
            **analysis.run(ctx),
        }
        for analysis in analyses
    }


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def _theme() -> Any:
    import logging

    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    # Year and tier labels are categorical on purpose; matplotlib logs an info line per
    # axis to say it could read them as numbers, which buries the structured events.
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
    return sns


SINGLE_COLUMN = 3.4
DOUBLE_COLUMN = 7.0


def _new_figure(width: float, height: float) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def _papers_from(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("series", {}).get("papers", []))


def _f1(payload: Mapping[str, Any], sns: Any) -> Any:
    papers = _papers_from(payload)
    if not papers:
        return None
    years = sorted({p["year"] for p in papers})
    figure = _new_figure(DOUBLE_COLUMN, 3.0)
    axis = figure.subplots()
    frame = {"year": [str(p["year"]) for p in papers], "r": [p["r"] for p in papers]}
    order = [str(y) for y in years]
    sns.violinplot(
        data=frame, x="year", y="r", order=order, inner="quartile", cut=0,
        palette="colorblind", hue="year", legend=False, ax=axis,
    )
    sns.stripplot(
        data=frame, x="year", y="r", order=order, color="black", alpha=0.25, size=2.0,
        jitter=0.22, ax=axis,
    )
    for index, year in enumerate(years):
        values = [p["r"] for p in papers if p["year"] == year]
        axis.scatter([index], [st.median(values)], marker="D", s=18, color="white",
                     edgecolor="black", zorder=5, linewidth=0.6)
        axis.annotate(f"n={len(values)}", (index, 1.02), ha="center", fontsize=7,
                      annotation_clip=False)
    axis.set(xlabel="LAK proceedings year", ylabel="$R$ (mean of three repetitions)",
             ylim=(0.0, 1.05))
    return figure


def _f2(payload: Mapping[str, Any], sns: Any) -> Any:
    papers = _papers_from(payload)
    result = payload.get("analyses", {}).get("A1.2")
    if not papers or not result:
        return None
    order = result.get("order_by_median") or sorted({str(p["claim_tier"]) for p in papers})
    figure = _new_figure(SINGLE_COLUMN, 3.2)
    axis = figure.subplots()
    frame = {"tier": [str(p["claim_tier"]) for p in papers], "r": [p["r"] for p in papers]}
    sns.violinplot(
        data=frame, x="r", y="tier", order=order, inner="quartile", cut=0,
        palette="colorblind", hue="tier", legend=False, ax=axis,
    )
    axis.set(xlabel="$R$", ylabel="", xlim=(0.0, 1.0))
    axis.set_yticks(range(len(order)), labels=[label.split()[0] for label in order])
    for index, tier in enumerate(order):
        n = sum(1 for p in papers if str(p["claim_tier"]) == tier)
        axis.annotate(f"n={n}", (0.02, index), fontsize=7, va="center")
    return figure


def _f3(payload: Mapping[str, Any], sns: Any) -> Any:
    papers = _papers_from(payload)
    if not papers:
        return None
    figure = _new_figure(SINGLE_COLUMN, SINGLE_COLUMN)
    grid = figure.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
    )
    main = figure.add_subplot(grid[1, 0])
    top = figure.add_subplot(grid[0, 0], sharex=main)
    right = figure.add_subplot(grid[1, 1], sharey=main)
    rc = [p["content_score"] for p in papers]
    rs = [p["structural_score"] for p in papers]
    main.scatter(rc, rs, s=9, alpha=0.5, color="#0072B2", edgecolor="none")
    sns.kdeplot(x=rc, ax=top, color="#0072B2", fill=True, linewidth=0.8)
    sns.kdeplot(y=rs, ax=right, color="#0072B2", fill=True, linewidth=0.8)
    ceiling = sum(1 for v in rs if v >= 0.999)
    main.axhline(1.0, color="#D55E00", linestyle="--", linewidth=0.9)
    main.annotate(
        f"$R_s=1.000$ for {ceiling}/{len(papers)}",
        (0.03, 0.985), fontsize=7, color="#D55E00", va="top",
    )
    main.set(xlabel="content reconstructability $R_c$", ylabel="structural $R_s$",
             xlim=(0.0, 1.0), ylim=(0.0, 1.03))
    for axis in (top, right):
        axis.set_axis_off()
    return figure


def _f3b(payload: Mapping[str, Any], sns: Any) -> Any:
    papers = _papers_from(payload)
    if not papers:
        return None
    labels = {
        "source_score": "sources",
        "methods_score": "methods",
        "experiments_score": "experiments",
        "sink_score": "outputs",
    }
    figure = _new_figure(DOUBLE_COLUMN, 2.4)
    axes = figure.subplots(1, 4, sharey=True)
    palette = sns.color_palette("colorblind", 4)
    for axis, (score, label), colour in zip(axes, labels.items(), palette, strict=True):
        values = [p[score] for p in papers if p.get(score) is not None]
        sns.violinplot(y=values, inner="quartile", cut=0, color=colour, ax=axis)
        axis.set(xlabel=label, ylim=(0.0, 1.05))
        axis.set_ylabel("stage score" if axis is axes[0] else "")
    return figure


def _f4(payload: Mapping[str, Any], sns: Any) -> Any:
    analyses = payload.get("analyses", {})
    primary = analyses.get("A2.1")
    distribution = analyses.get("A1.1")
    if not primary or not distribution:
        return None
    years = sorted(int(y) for y in primary["sharing_by_year"])
    means = [distribution["by_year"][str(y)]["mean_ci"]["estimate"] for y in years]
    low = [distribution["by_year"][str(y)]["mean_ci"]["ci_low"] for y in years]
    high = [distribution["by_year"][str(y)]["mean_ci"]["ci_high"] for y in years]
    share = [primary["sharing_by_year"][str(y)]["proportion"] for y in years]
    figure = _new_figure(SINGLE_COLUMN, 2.7)
    axis = figure.subplots()
    axis.fill_between(years, low, high, color="#0072B2", alpha=0.20, linewidth=0)
    axis.plot(years, means, marker="o", markersize=4, color="#0072B2", label="mean $R$")
    axis.set(xlabel="LAK proceedings year", ylabel="mean $R$ (95% CI)", ylim=(0.40, 0.62))
    axis.tick_params(axis="y", labelcolor="#0072B2")
    twin = axis.twinx()
    twin.plot(years, share, marker="s", markersize=4, linestyle="--", color="#D55E00",
              label="verified data")
    twin.set_ylabel("papers with verified retrievable data", color="#D55E00")
    twin.tick_params(axis="y", labelcolor="#D55E00")
    twin.set_ylim(0.0, max(share) * 1.35 if any(share) else 1.0)
    twin.grid(False)
    handles = axis.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    axis.legend(handles=handles, loc="upper left", fontsize=7, frameon=True)
    return figure


def _f5(payload: Mapping[str, Any], sns: Any) -> Any:
    papers = _papers_from(payload)
    if not papers:
        return None

    def bucket(paper: Mapping[str, Any]) -> str:
        if paper["verified_data"] and paper["verified_code"]:
            return "data+code"
        if paper["verified_data"] or paper["verified_code"]:
            return "one only"
        return "neither"

    order = ["neither", "one only", "data+code"]
    frame = {"bucket": [bucket(p) for p in papers], "r": [p["r"] for p in papers]}
    figure = _new_figure(SINGLE_COLUMN, 2.8)
    axis = figure.subplots()
    sns.violinplot(
        data=frame, x="bucket", y="r", order=order, inner="quartile", cut=0,
        palette="colorblind", hue="bucket", legend=False, ax=axis,
    )
    sns.stripplot(data=frame, x="bucket", y="r", order=order, color="black", alpha=0.3,
                  size=2.0, jitter=0.2, ax=axis)
    for index, name in enumerate(order):
        axis.annotate(f"n={frame['bucket'].count(name)}", (index, 1.02), ha="center",
                      fontsize=7, annotation_clip=False)
    axis.set(xlabel="verified retrievable artifacts", ylabel="$R$", ylim=(0.0, 1.05))
    return figure


def _f6(payload: Mapping[str, Any], sns: Any) -> Any:
    result = payload.get("analyses", {}).get("A4.2")
    if not result:
        return None
    band = result["noise_band"]
    taus = [point["tau"] for point in band]
    figure = _new_figure(SINGLE_COLUMN, 2.7)
    axis = figure.subplots()
    axis.fill_between(
        taus, [p["l2_min_draw"] for p in band], [p["l2_max_draw"] for p in band],
        color="#999999", alpha=0.35, linewidth=0, label="run-to-run noise",
    )
    axis.plot(taus, [p["l2_mean"] for p in band], color="#0072B2", linewidth=1.4,
              label="papers at L2 (mean $R$)")
    axis.set(xlabel=r"candidate threshold $\tau_R$", ylabel="papers at L2")
    axis.legend(fontsize=7, loc="upper right", frameon=True)
    axis.annotate("no threshold is selected", (0.31, axis.get_ylim()[0] + 12), fontsize=7)
    return figure


def _f7(payload: Mapping[str, Any], sns: Any) -> Any:
    result = payload.get("analyses", {}).get("A3.5")
    pairs = payload.get("series", {}).get("drift_pairs", [])
    if not result or not pairs or "bland_altman" not in result:
        return None
    means = [(p["old_r"] + p["new_r"]) / 2 for p in pairs]
    differences = [p["new_r"] - p["old_r"] for p in pairs]
    bias = result["bland_altman"]["bias"]
    lower, upper = result["bland_altman"]["limits_of_agreement"]
    figure = _new_figure(SINGLE_COLUMN, 2.8)
    axis = figure.subplots()
    axis.scatter(means, differences, s=12, alpha=0.7, color="#0072B2", edgecolor="none")
    axis.axhline(bias, color="#D55E00", linewidth=1.1, label=f"bias {bias:+.3f}")
    for limit in (lower, upper):
        axis.axhline(limit, color="#333333", linestyle=":", linewidth=0.9)
    axis.axhline(0.0, color="#999999", linewidth=0.6)
    axis.set(xlabel="mean of the two assessments", ylabel="new mean $R$ - old single-draw $R$")
    axis.annotate(f"n={len(pairs)}, 95% LoA [{lower:+.3f}, {upper:+.3f}]",
                  (0.02, 0.02), xycoords="axes fraction", fontsize=7)
    axis.legend(fontsize=7, loc="upper right", frameon=True)
    return figure


def _f8(payload: Mapping[str, Any], sns: Any) -> Any:
    papers = _papers_from(payload)
    if not papers:
        return None
    sds = [p["r_sd"] for p in papers]
    figure = _new_figure(SINGLE_COLUMN, 2.5)
    axis = figure.subplots()
    axis.axvspan(0.02, 0.09, color="#999999", alpha=0.30, linewidth=0,
                 label="ARA's reported 0.02-0.09")
    sns.histplot(x=sds, bins=30, color="#0072B2", ax=axis)
    axis.axvline(st.fmean(sds), color="#D55E00", linewidth=1.1,
                 label=f"mean {st.fmean(sds):.4f}")
    axis.set(xlabel="within-paper SD of $R$ across three repetitions", ylabel="papers")
    axis.legend(fontsize=7, frameon=True)
    return figure


def _f9(payload: Mapping[str, Any], sns: Any) -> Any:
    result = payload.get("analyses", {}).get("A5.1")
    if not result:
        return None
    classes = [c for c in ("source", "process", "sink") if c in result["by_node_class"]]
    means = [result["by_node_class"][c]["mean_ci"]["estimate"] for c in classes]
    low = [result["by_node_class"][c]["mean_ci"]["ci_low"] for c in classes]
    high = [result["by_node_class"][c]["mean_ci"]["ci_high"] for c in classes]
    figure = _new_figure(SINGLE_COLUMN, 2.4)
    axis = figure.subplots()
    palette = sns.color_palette("colorblind", len(classes))
    positions = range(len(classes))
    axis.bar(positions, means, width=0.55, color=palette, edgecolor="black", linewidth=0.5)
    axis.errorbar(
        list(positions), means,
        yerr=[[m - lo for m, lo in zip(means, low, strict=True)],
              [hi - m for m, hi in zip(means, high, strict=True)]],
        fmt="none", ecolor="black", capsize=3, linewidth=0.9,
    )
    for position, mean in zip(positions, means, strict=True):
        axis.annotate(f"{mean:.1f}", (position, mean), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=7)
    axis.set_xticks(list(positions))
    axis.set_xticklabels([f"{c}\nnodes" for c in classes])
    axis.set(ylabel="mean node reproducibility score", ylim=(0, 100))
    return figure


def _f10(payload: Mapping[str, Any], sns: Any) -> Any:
    result = payload.get("analyses", {}).get("A5.2")
    if not result:
        return None
    years = sorted(int(y) for y in result["by_year"])
    required = [result["by_year"][str(y)]["parameters"]["required"] for y in years]
    mentioned = [result["by_year"][str(y)]["parameters"]["counted_towards_completion"] for y in years]
    completion = [result["by_year"][str(y)]["parameters"]["completion"] for y in years]
    figure = _new_figure(DOUBLE_COLUMN, 2.8)
    axis = figure.subplots()
    offsets = np.arange(len(years))
    axis.bar(offsets - 0.20, required, width=0.38, color="#CCCCCC", edgecolor="black",
             linewidth=0.5, label="parameters required")
    axis.bar(offsets + 0.20, mentioned, width=0.38, color="#0072B2", edgecolor="black",
             linewidth=0.5, hatch="//", label="parameters named")
    axis.set_xticks(offsets)
    axis.set_xticklabels([str(y) for y in years])
    axis.set(xlabel="LAK proceedings year", ylabel="parameter mentions (all repetitions)")
    twin = axis.twinx()
    twin.plot(offsets, completion, marker="o", markersize=4, color="#D55E00",
              linewidth=1.3, label="completion ratio")
    twin.set_ylabel("named / required", color="#D55E00")
    twin.tick_params(axis="y", labelcolor="#D55E00")
    twin.set_ylim(0.0, 1.0)
    twin.grid(False)
    handles = axis.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    axis.legend(handles=handles, loc="upper left", fontsize=7, frameon=True)
    overall = result["parameters"]["completion"]
    axis.annotate(
        f"corpus-wide: {overall:.1%} of required parameters named",
        (0.985, 0.94), xycoords="axes fraction", fontsize=7, ha="right", va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    return figure


FIGURES: tuple[tuple[str, str, Callable[[Mapping[str, Any], Any], Any]], ...] = (
    ("F1", "R by year", _f1),
    ("F2", "R by claim type", _f2),
    ("F3", "R_c against R_s with the structural ceiling", _f3),
    ("F3b", "Stage score distributions", _f3b),
    ("F4", "Trend contrast: mean R against verified data sharing", _f4),
    ("F5", "R by verified artifact availability", _f5),
    ("F6", "tau_R sensitivity with the run-to-run noise band", _f6),
    ("F7", "Bland-Altman drift check", _f7),
    ("F8", "Within-paper SD of R", _f8),
    ("F9", "Node deficit by node class", _f9),
    ("F10", "Parameter gap by year", _f10),
)


def write_figures(payload: Mapping[str, Any], charts_dir: Path) -> list[str]:
    from ara_analysis import _save_figure

    sns = _theme()
    charts_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for stem, _description, builder in FIGURES:
        figure = builder(payload, sns)
        if figure is None:
            continue
        written.extend(str(path) for path in _save_figure(figure, charts_dir, stem))
    return written


# --------------------------------------------------------------------------------------
# payload assembly
# --------------------------------------------------------------------------------------

SERIES_FIELDS = (
    "paper_id",
    "experiment",
    "doi",
    "year",
    "kind",
    "claim_type",
    "claim_tier",
    "markdown_chars",
    "r",
    "r_sd",
    "r_min",
    "r_max",
    "r_values",
    "content_score",
    "structural_score",
    "source_score",
    "methods_score",
    "experiments_score",
    "sink_score",
    "source_node_count",
    "method_node_count",
    "experiment_node_count",
    "sink_node_count",
    "total_node_count",
    "source_node_score",
    "process_node_score",
    "sink_node_score",
    "algorithm_clarity_mean",
    "statement_clarity_mean",
    "parameter_completion",
    "tool_completion",
    "open_data",
    "open_materials",
    "license_stated",
    "readme",
    "verified_data",
    "verified_code",
    "e_score",
    "any_source_open",
    "ara_best_availability",
    *WIRING_KEYS,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if _isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_payload(
    data: Dataset,
    analyses: Sequence[Analysis],
    seed: int,
    resamples: int,
    database: Path,
) -> dict[str, Any]:
    ctx = Context(data=data, seed=seed, resamples=resamples)
    results = run_analyses(ctx, analyses)
    return _jsonable(
        {
            "provenance": {
                **data.provenance,
                "database": str(database),
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
                "analyses_run": [analysis.id for analysis in analyses],
                "primary_hypothesis": "A2.1",
                "read_only": "database opened with SQLite mode=ro; no table is written",
                "threshold_policy": "no tau_R is selected; A4 reports a sensitivity curve",
            },
            "self_checks": data.self_checks,
            "analyses": results,
            "series": {
                "papers": [
                    {key: paper[key] for key in SERIES_FIELDS if key in paper}
                    for paper in data.papers
                ],
                "drift_pairs": data.drift_pairs,
                "luna_pairs": data.luna_pairs,
                "node_totals": data.node_totals,
            },
        }
    )


def node_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provenance": payload["provenance"],
        "self_checks": payload["self_checks"],
        "analyses": {
            key: value
            for key, value in payload["analyses"].items()
            if key.startswith(NODE_ANALYSIS_PREFIX)
        },
        "series": {"node_totals": payload["series"]["node_totals"]},
    }


def dump_json(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------
# CLI handlers
# --------------------------------------------------------------------------------------


async def _candidate_sets(catalog: Any, experiments: Sequence[str]) -> dict[str, set[int]]:
    """Selection truth from `ara._ara_candidates`, so the analysis set cannot drift."""
    from ara import _ara_candidates, _get_ara_experiment, ensure_ara_schema

    await ensure_ara_schema(catalog)
    selection_args = SimpleNamespace(
        paper_id=None, year=None, kind=None, limit=None, include_non_quantitative=False
    )
    sets: dict[str, set[int]] = {}
    for name in experiments:
        experiment = await _get_ara_experiment(catalog, name)
        rows = await _ara_candidates(catalog, experiment, selection_args)
        sets[name] = {int(row["paper_id"]) for row in rows}
    return sets


def _experiment_names(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_EXPERIMENTS)
    names = [token.strip() for token in raw.split(",") if token.strip()]
    if not names:
        raise ValueError("--experiments was given but lists no experiment")
    return names


async def command_rstats_run(args: Any, logger: Any, Catalog: Any) -> int:
    database = Path(args.database)
    experiments = _experiment_names(getattr(args, "experiments", None))
    async with Catalog(database) as catalog:
        candidates = await _candidate_sets(catalog, experiments)
    analyses = select_analyses(getattr(args, "only", None))
    data = await asyncio.to_thread(load_dataset, database, experiments, candidates)
    payload = await asyncio.to_thread(
        build_payload, data, analyses, args.seed, args.bootstrap, database
    )
    output = Path(args.output)
    dump_json(payload, output)
    nodes_output = output.with_name(output.stem + "-nodes" + output.suffix)
    if any(a.id.startswith(NODE_ANALYSIS_PREFIX) for a in analyses):
        dump_json(node_payload(payload), nodes_output)
    figures: list[str] = []
    if not getattr(args, "no_figures", False):
        figures = await asyncio.to_thread(write_figures, payload, Path(args.charts_dir))
    failed = sorted(
        key for key, check in payload["self_checks"].items()
        if isinstance(check, dict) and check.get("ok") is False
    )
    for key in failed:
        logger.warning("rstats_self_check_failed", check=key, detail=payload["self_checks"][key])
    logger.info(
        "rstats_run_complete",
        papers=payload["provenance"]["papers"],
        analyses=len(analyses),
        output=str(output),
        figures=len(figures),
        bootstrap=args.bootstrap,
        seed=args.seed,
        self_checks_failed=failed,
    )
    _print_summary(payload, failed)
    return 1 if failed else 0


async def command_rstats_nodes(args: Any, logger: Any, Catalog: Any) -> int:
    database = Path(args.database)
    experiments = _experiment_names(getattr(args, "experiments", None))
    async with Catalog(database) as catalog:
        candidates = await _candidate_sets(catalog, experiments)
    analyses = select_analyses(NODE_ANALYSIS_PREFIX)
    data = await asyncio.to_thread(load_dataset, database, experiments, candidates)
    payload = await asyncio.to_thread(
        build_payload, data, analyses, args.seed, args.bootstrap, database
    )
    output = Path(args.output)
    dump_json(node_payload(payload), output)
    _print_node_summary(payload)
    logger.info(
        "rstats_nodes_complete",
        papers=payload["provenance"]["papers"],
        output=str(output),
        seed=args.seed,
    )
    return 0


async def command_rstats_figures(args: Any, logger: Any, Catalog: Any) -> int:
    """Regenerate every figure from the saved JSON alone; no database access."""
    source = Path(args.source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    figures = await asyncio.to_thread(write_figures, payload, Path(args.charts_dir))
    logger.info(
        "rstats_figures_complete", source=str(source), figures=len(figures), paths=figures
    )
    from rich.console import Console

    Console().print(f"wrote {len(figures)} files from [bold]{source}[/bold] with no database read")
    return 0


async def command_rstats_list(args: Any, logger: Any, Catalog: Any) -> int:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="R analyses", header_style="bold cyan")
    table.add_column("ID")
    table.add_column("Role")
    table.add_column("Question")
    for analysis in ANALYSES:
        table.add_row(
            analysis.id,
            "exploratory" if analysis.exploratory else "PRIMARY",
            analysis.question,
        )
    console = Console()
    console.print(table)
    console.print("Figures: " + ", ".join(f"{stem} ({desc})" for stem, desc, _ in FIGURES))
    logger.info("rstats_listed", analyses=len(ANALYSES), figures=len(FIGURES))
    return 0


def _print_summary(payload: Mapping[str, Any], failed: Sequence[str]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    provenance = payload["provenance"]
    console.print(
        f"[bold]{provenance['papers']}[/bold] papers, "
        f"{provenance['assessments']} assessments, "
        f"{len(provenance['analyses_run'])} analyses, "
        f"bootstrap {provenance['bootstrap_resamples']} at seed {provenance['bootstrap_seed']}"
    )
    primary = payload["analyses"].get("A2.1")
    if primary:
        table = Table(title="A2.1 (primary): describing against sharing", header_style="bold cyan")
        for column in ("quantity", "slope per year", "95% CI"):
            table.add_column(column)
        r_slope = primary["r_slope_sd_per_year"]
        share_slope = primary["sharing_slope_sd_per_year"]
        gap = primary["contrast"]
        for label, block in (
            ("R (SD/year)", r_slope),
            ("verified data sharing (SD/year)", share_slope),
        ):
            table.add_row(
                label,
                f"{block['estimate']:+.4f}",
                f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}]"
                if block.get("ci_low") is not None
                else "-",
            )
        table.add_row(
            "gap (sharing - R)",
            f"{gap['gap_sd_per_year']:+.4f}",
            f"[{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}]"
            if gap.get("ci_low") is not None
            else "-",
        )
        console.print(table)
        console.print(
            "R slope CI excludes the sharing slope: "
            f"[bold]{gap['r_ci_excludes_sharing_slope']}[/bold]"
        )
    gap_result = payload["analyses"].get("A5.2")
    if gap_result:
        console.print(gap_result["headline"])
    if failed:
        console.print(f"[bold red]self-checks failed:[/bold red] {', '.join(failed)}")
    else:
        console.print("[green]all self-checks passed[/green]")
    console.print("[yellow]No tau_R is selected and no gate is decided here.[/yellow]")


def _print_node_summary(payload: Mapping[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    deficit = payload["analyses"].get("A5.1")
    if deficit:
        table = Table(title="A5.1 node deficit", header_style="bold cyan")
        for column in ("node class", "mean score", "95% CI", "n"):
            table.add_column(column)
        for cls, block in deficit["by_node_class"].items():
            ci = block["mean_ci"]
            table.add_row(
                cls,
                f"{ci['estimate']:.2f}",
                f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]" if ci.get("ci_low") else "-",
                str(block.get("n")),
            )
        console.print(table)
        console.print(f"bottleneck: [bold]{deficit['bottleneck']}[/bold] nodes")
    gap = payload["analyses"].get("A5.2")
    if gap:
        console.print(gap["headline"])
        console.print(
            f"tools named: {gap['tools']['completion']:.1%} of required"
            if gap["tools"]["completion"] is not None
            else "tools: no required tools recorded"
        )
    chain = payload["analyses"].get("A5.4")
    if chain:
        broken = chain["broken_evidence_chain"]
        console.print(
            f"{broken['papers']}/{broken['of']} papers ({broken['fraction']:.1%}) describe a "
            "result that cannot be traced back to a named data source"
        )
