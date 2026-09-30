"""Click command line interface with lazy subsystem imports."""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import click

DEFAULT_DATABASE = Path("data/lak.sqlite")
DEFAULT_LOG = Path("logs/lakdb.jsonl")
PAPER_KINDS = click.Choice(("long", "short"))
REASONING = click.Choice(("none", "minimal", "low", "medium", "high", "xhigh"))
LABELS = click.Choice(("applicable", "not_applicable", "unclear"))


def _candidates(kind: str, ctx: click.Context, incomplete: str) -> list[click.shell_completion.CompletionItem]:
    """Complete database-backed values without importing OpenAI/PyMuPDF/Textual."""
    from completions import completion_candidates

    database = Path(str(ctx.find_root().params.get("database", DEFAULT_DATABASE)))
    values = completion_candidates(database, kind)
    return [
        click.shell_completion.CompletionItem(row["value"], help=row["description"])
        for row in values
        if row["value"].startswith(incomplete)
    ]


def complete_experiment(ctx: click.Context, param: click.Parameter, incomplete: str):
    return _candidates("experiments", ctx, incomplete)


def complete_batch(ctx: click.Context, param: click.Parameter, incomplete: str):
    return _candidates("batches", ctx, incomplete)


def complete_paper(ctx: click.Context, param: click.Parameter, incomplete: str):
    return _candidates("papers", ctx, incomplete)


def complete_artifact_experiment(ctx: click.Context, param: click.Parameter, incomplete: str):
    return _candidates("artifact-experiments", ctx, incomplete)


def complete_artifact_batch(ctx: click.Context, param: click.Parameter, incomplete: str):
    return _candidates("artifact-batches", ctx, incomplete)


def _args(ctx: click.Context, **values: Any) -> SimpleNamespace:
    """Adapt Click's parsed values to the existing handler attribute interface."""
    return SimpleNamespace(
        database=ctx.obj["database"],
        log_file=ctx.obj["log_file"],
        verbose=ctx.obj["verbose"],
        **values,
    )


def _run(ctx: click.Context, module: str, function: str, **values: Any) -> None:
    """Import command implementation only after Click parsed the requested command."""
    import main

    main.configure_logging(ctx.obj["log_file"], ctx.obj["verbose"])
    logger = main.structlog.get_logger(main.APP_NAME).bind(command=ctx.command_path)
    implementation = getattr(__import__(module, fromlist=[function]), function)
    result = asyncio.run(implementation(_args(ctx, **values), logger, main.Catalog))
    if result:
        raise click.exceptions.Exit(result)


def _run_core(ctx: click.Context, function: str, **values: Any) -> None:
    import main

    main.configure_logging(ctx.obj["log_file"], ctx.obj["verbose"])
    logger = main.structlog.get_logger(main.APP_NAME).bind(command=ctx.command_path)
    result = asyncio.run(getattr(main, function)(_args(ctx, **values), logger))
    if result:
        raise click.exceptions.Exit(result)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--database", type=click.Path(path_type=Path), default=DEFAULT_DATABASE, show_default=True)
@click.option("--log-file", type=click.Path(path_type=Path), default=DEFAULT_LOG, show_default=True)
@click.option("--verbose", is_flag=True, help="Show debug structured events.")
@click.pass_context
def cli(ctx: click.Context, database: Path, log_file: Path, verbose: bool) -> None:
    """Reproducible LAK corpus catalog, screening, and evidence pipeline."""
    ctx.ensure_object(dict)
    ctx.obj.update(database=database, log_file=log_file, verbose=verbose)


@cli.command()
@click.argument("dataset_root", type=click.Path(path_type=Path))
@click.pass_context
def inventory(ctx: click.Context, dataset_root: Path) -> None:
    _run_core(ctx, "command_inventory", dataset_root=dataset_root)


@cli.command()
@click.argument("dataset_root", type=click.Path(path_type=Path))
@click.option("--no-progress", is_flag=True)
@click.pass_context
def ingest(ctx: click.Context, dataset_root: Path, no_progress: bool) -> None:
    _run_core(ctx, "command_ingest", dataset_root=dataset_root, no_progress=no_progress)


@cli.command()
@click.option("--workers", type=click.IntRange(1), default=1, show_default=True)
@click.option("--ocr-workers", type=click.IntRange(1), default=1, show_default=True)
@click.option("--limit", type=click.IntRange(1))
@click.option("--force", is_flag=True)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def parse(ctx: click.Context, workers: int, ocr_workers: int, limit: int | None, force: bool, no_progress: bool) -> None:
    _run_core(ctx, "command_parse", workers=workers, ocr_workers=ocr_workers, limit=limit, force=force, no_progress=no_progress)


@cli.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    _run_core(ctx, "command_validate")


@cli.command()
@click.argument("query")
@click.option("--limit", type=click.IntRange(1), default=10, show_default=True)
@click.pass_context
def search(ctx: click.Context, query: str, limit: int) -> None:
    _run_core(ctx, "command_search", query=query, limit=limit)


@cli.command()
@click.argument("dataset_root", type=click.Path(path_type=Path))
@click.option("--yes", is_flag=True)
@click.option("--parse", "parse_after", is_flag=True)
@click.option("--workers", type=click.IntRange(1), default=1)
@click.option("--ocr-workers", type=click.IntRange(1), default=1)
@click.option("--limit", type=click.IntRange(1))
@click.option("--force", is_flag=True)
@click.option("--no-progress", is_flag=True)
@click.option("--delete-study-results", is_flag=True)
@click.pass_context
def rebuild(ctx: click.Context, dataset_root: Path, yes: bool, parse_after: bool, workers: int, ocr_workers: int, limit: int | None, force: bool, no_progress: bool, delete_study_results: bool) -> None:
    _run_core(ctx, "command_rebuild", dataset_root=dataset_root, yes=yes, parse=parse_after, workers=workers, ocr_workers=ocr_workers, limit=limit, force=force, no_progress=no_progress, delete_study_results=delete_study_results)


def _screen_options(command: Callable[..., Any]) -> Callable[..., Any]:
    command = click.option("--experiment", required=True, shell_complete=complete_experiment)(command)
    command = click.option("--paper-id", type=int, shell_complete=complete_paper)(command)
    command = click.option("--year", type=int)(command)
    command = click.option("--kind", type=PAPER_KINDS)(command)
    command = click.option("--limit", type=click.IntRange(1))(command)
    command = click.option("--no-progress", is_flag=True)(command)
    return command


def _artifact_options(command: Callable[..., Any]) -> Callable[..., Any]:
    command = click.option("--experiment", required=True, shell_complete=complete_artifact_experiment)(command)
    command = click.option("--paper-id", type=int, shell_complete=complete_paper)(command)
    command = click.option("--year", type=int)(command)
    command = click.option("--kind", type=PAPER_KINDS)(command)
    command = click.option("--limit", type=click.IntRange(1), help="Number of papers, not API calls.")(command)
    command = click.option("--no-progress", is_flag=True)(command)
    return command


@cli.group()
def experiment() -> None:
    """Create, freeze, list, or remove applicability experiments."""


@experiment.command("create")
@click.argument("name")
@click.option("--model", required=True)
@click.option("--reasoning-effort", type=REASONING, default="medium", show_default=True)
@click.option("--repeats", type=click.IntRange(1), default=3, show_default=True)
@click.option("--freeze", "freeze", is_flag=True)
@click.pass_context
def experiment_create(ctx: click.Context, name: str, model: str, reasoning_effort: str, repeats: int, freeze: bool) -> None:
    _run(ctx, "screening", "command_screen_create", name=name, model=model, reasoning_effort=reasoning_effort, repeats=repeats, freeze=freeze)


@experiment.command("freeze")
@click.argument("experiment", shell_complete=complete_experiment)
@click.pass_context
def experiment_freeze(ctx: click.Context, experiment: str) -> None:
    _run(ctx, "screening", "command_screen_freeze", experiment=experiment)


@experiment.command("list")
@click.pass_context
def experiment_list(ctx: click.Context) -> None:
    """List experiments using standard-library SQLite only (fast and offline)."""
    if not ctx.obj["database"].is_file():
        rows = []
    else:
        try:
            with sqlite3.connect(ctx.obj["database"]) as db:
                rows = db.execute("SELECT id,name,status,model,reasoning_effort,repeat_count FROM study_experiments ORDER BY id").fetchall()
        except sqlite3.Error:
            rows = []
    if not rows:
        click.echo("No applicability experiments.")
        return
    click.echo("ID  NAME  STATUS  MODEL  EFFORT  REPEATS")
    for row in rows:
        click.echo("  ".join(str(value or "—") for value in row))


@experiment.command("delete")
@click.option("--experiment", "experiments", multiple=True, shell_complete=complete_experiment)
@click.option("--all", "all_experiments", is_flag=True)
@click.option("--yes", is_flag=True)
@click.option("--force-active", is_flag=True)
@click.option("--delete-highlight-files", is_flag=True)
@click.pass_context
def experiment_delete(ctx: click.Context, experiments: tuple[str, ...], all_experiments: bool, yes: bool, force_active: bool, delete_highlight_files: bool) -> None:
    _run(ctx, "screening", "command_experiment_delete", experiment=list(experiments) or None, all=all_experiments, yes=yes, force_active=force_active, delete_highlight_files=delete_highlight_files)


@cli.group()
def screen() -> None:
    """Run or finalize Stage-1 applicability screening."""


@screen.command("run")
@_screen_options
@click.option("--retries", type=click.IntRange(0), default=2, show_default=True)
@click.pass_context
def screen_run(ctx: click.Context, retries: int, **kwargs: Any) -> None:
    _run(ctx, "screening", "command_screen", retries=retries, **kwargs)


@screen.command("finalize")
@click.option("--experiment", required=True, shell_complete=complete_experiment)
@click.pass_context
def screen_finalize(ctx: click.Context, experiment: str) -> None:
    _run(ctx, "screening", "command_finalize", experiment=experiment)


@cli.group()
def batch() -> None:
    """Prepare, submit, inspect, and collect OpenAI batches."""


@batch.command("prepare")
@_screen_options
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.pass_context
def batch_prepare(ctx: click.Context, output: Path, **kwargs: Any) -> None:
    _run(ctx, "screening", "command_batch_prepare", output=output, **kwargs)


@batch.command("submit")
@click.option("--input", "input_path", type=click.Path(path_type=Path), required=True)
@click.pass_context
def batch_submit(ctx: click.Context, input_path: Path) -> None:
    _run(ctx, "screening", "command_batch_submit", input=input_path)


@batch.command("status")
@click.option("--batch-id", type=int, required=True, shell_complete=complete_batch)
@click.option("--wait", is_flag=True)
@click.option("--poll-seconds", type=click.FloatRange(1), default=30.0, show_default=True)
@click.pass_context
def batch_status(ctx: click.Context, batch_id: int, wait: bool, poll_seconds: float) -> None:
    _run(ctx, "screening", "command_batch_status", batch_id=batch_id, wait=wait, poll_seconds=poll_seconds)


@batch.command("collect")
@click.option("--batch-id", type=int, required=True, shell_complete=complete_batch)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def batch_collect(ctx: click.Context, batch_id: int, no_progress: bool) -> None:
    _run(ctx, "screening", "command_batch_collect", batch_id=batch_id, no_progress=no_progress)


@cli.group()
def highlight() -> None:
    """Create PDF evidence annotations."""


@highlight.command("run")
@click.option("--experiment", required=True, shell_complete=complete_experiment)
@click.option("--paper-id", type=int, shell_complete=complete_paper)
@click.option("--doi", "dois", multiple=True, help="Canonical DOI, DOI URL, or doi: prefixed DOI; repeat to select multiple papers.")
@click.option("--all", "all_papers", is_flag=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("artifacts/highlighted"), show_default=True)
@click.option("--final-label", type=LABELS)
@click.option("--workers", type=click.IntRange(1), default=4, show_default=True)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def highlight_run(ctx: click.Context, experiment: str, paper_id: int | None, dois: tuple[str, ...], all_papers: bool, output_dir: Path, final_label: str | None, workers: int, no_progress: bool) -> None:
    selections = sum((paper_id is not None, bool(dois), all_papers))
    if selections != 1:
        raise click.UsageError("Use exactly one of --paper-id, one or more --doi, or --all.")
    _run(ctx, "screening", "command_highlight", experiment=experiment, paper_id=paper_id, dois=list(dois), all=all_papers, output_dir=output_dir, final_label=final_label, workers=workers, no_progress=no_progress)


@cli.command()
@click.option("--experiment", shell_complete=complete_experiment)
@click.pass_context
def usage(ctx: click.Context, experiment: str | None) -> None:
    _run(ctx, "screening", "command_usage", experiment=experiment)


@cli.command()
@click.option("--experiment", shell_complete=complete_experiment)
@click.option("--year", type=int)
@click.option("--status", type=click.Choice(("pending", "running", "succeeded", "failed", "cancelled")))
@click.pass_context
def dashboard(ctx: click.Context, experiment: str | None, year: int | None, status: str | None) -> None:
    # Dashboard and its Textual dependency remain lazy.
    import main
    from dashboard import run_dashboard
    from screening import ensure_schema

    async def launch() -> None:
        async with main.Catalog(ctx.obj["database"]) as catalog:
            await ensure_schema(catalog)
        await run_dashboard(ctx.obj["database"], experiment, year, status)
    asyncio.run(launch())


@cli.command("haim-align")
@click.option("--experiment", "experiments", required=True, multiple=True, shell_complete=complete_experiment, help="Frozen experiment to include; repeat to pool configuration-identical experiments.")
@click.option("--survey", type=click.Path(path_type=Path), default=Path("baseline/haim-lak/artifacts/qualtrics-survey-data.csv"), show_default=True)
@click.option("--explanations", type=click.Path(path_type=Path), default=Path("baseline/haim-lak/artifacts/explanations.pdf"), show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("artifacts/analysis/haim-alignment.json"), show_default=True)
@click.pass_context
def haim_align(ctx: click.Context, experiments: tuple[str, ...], survey: Path, explanations: Path, output: Path) -> None:
    _run(ctx, "agreement", "command_haim_alignment", experiments=list(experiments), survey=survey, explanations=explanations, output=output)


@cli.group()
def ara() -> None:
    """Run and report pinned ARA assessments."""


@ara.command("list")
@click.pass_context
def ara_list(ctx: click.Context) -> None:
    _run(ctx, "ara", "command_ara_experiment_list")


@ara.command("create")
@click.argument("name")
@click.option("--applicability-experiment", required=True, shell_complete=complete_experiment)
@click.option("--model", required=True)
@click.option("--reasoning-effort", type=REASONING)
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--repeats", type=click.IntRange(1), default=1, show_default=True)
@click.option("--checkout", type=click.Path(path_type=Path), default=Path(".external/ara"), show_default=True)
@click.option("--upstream-commit", default="e015be37d82c57e6da826ad50fe8023e9f1255f5", show_default=True)
@click.pass_context
def ara_create(ctx: click.Context, name: str, applicability_experiment: str, model: str, reasoning_effort: str | None, temperature: float, repeats: int, checkout: Path, upstream_commit: str) -> None:
    _run(ctx, "ara", "command_ara_create", name=name, applicability_experiment=applicability_experiment, model=model, reasoning_effort=reasoning_effort, temperature=temperature, repeats=repeats, checkout=checkout, upstream_commit=upstream_commit)


@ara.command("run")
@click.option("--experiment", required=True)
@click.option("--checkout", type=click.Path(path_type=Path), default=Path(".external/ara"), show_default=True)
@click.option("--paper-id", type=int, shell_complete=complete_paper)
@click.option("--year", type=int)
@click.option("--kind", type=PAPER_KINDS)
@click.option("--limit", type=click.IntRange(1))
@click.option("--workers", type=click.IntRange(1), default=2, show_default=True)
@click.option("--retry-failed", is_flag=True)
@click.option("--include-non-quantitative", is_flag=True, help="Assess L0 papers too; off by default per protocol.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("artifacts/ara"), show_default=True)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def ara_run(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "ara", "command_ara_run", **kwargs)


@ara.command("report")
@click.option("--experiment", required=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("artifacts/analysis/ara-report.json"), show_default=True)
@click.option("--survey", type=click.Path(path_type=Path), default=Path("baseline/haim-lak/artifacts/qualtrics-survey-data.csv"), show_default=True)
@click.option("--charts-dir", type=click.Path(path_type=Path), default=Path("artifacts/analysis"), show_default=True)
@click.pass_context
def ara_report(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "ara", "command_ara_report", **kwargs)


@cli.group()
def rstats() -> None:
    """Stage 2 R analytics: distribution, trend, instrument, node deficits."""


@rstats.command("run")
@click.option("--experiments", default="s2-ara-2021_2022-gpt56t-r3,s2-ara-2023_2026-gpt56t-r3", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("artifacts/analysis/r-analysis.json"), show_default=True)
@click.option("--charts-dir", type=click.Path(path_type=Path), default=Path("artifacts/analysis/figures"), show_default=True)
@click.option("--bootstrap", type=click.IntRange(100), default=10000, show_default=True)
@click.option("--seed", type=int, default=20260918, show_default=True)
@click.option("--only", help="Analysis ids or group prefixes, e.g. A1,A2.1.")
@click.option("--no-figures", is_flag=True)
@click.pass_context
def rstats_run(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "r_analysis", "command_rstats_run", **kwargs)


@rstats.command("nodes")
@click.option("--experiments", default="s2-ara-2021_2022-gpt56t-r3,s2-ara-2023_2026-gpt56t-r3", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("artifacts/analysis/r-analysis-nodes.json"), show_default=True)
@click.option("--bootstrap", type=click.IntRange(100), default=10000, show_default=True)
@click.option("--seed", type=int, default=20260918, show_default=True)
@click.pass_context
def rstats_nodes(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "r_analysis", "command_rstats_nodes", **kwargs)


@rstats.command("figures")
@click.option("--from", "source", type=click.Path(path_type=Path), default=Path("artifacts/analysis/r-analysis.json"), show_default=True)
@click.option("--charts-dir", type=click.Path(path_type=Path), default=Path("artifacts/analysis/figures"), show_default=True)
@click.pass_context
def rstats_figures(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "r_analysis", "command_rstats_figures", **kwargs)


@rstats.command("list")
@click.pass_context
def rstats_list(ctx: click.Context) -> None:
    _run(ctx, "r_analysis", "command_rstats_list")


@cli.group()
def artifact() -> None:
    """Stage 1 artifact audit: claims, docs, alignment."""


@artifact.command("create")
@click.argument("name")
@click.option("--model", required=True)
@click.option("--reasoning-effort", type=REASONING, default="medium", show_default=True)
@click.option("--repeats", type=click.IntRange(1), default=3, show_default=True)
@click.option("--freeze", "freeze", is_flag=True)
@click.option("--stage", type=click.Choice(("claim", "docs")), default="claim", show_default=True)
@click.option("--claim-experiment", shell_complete=complete_artifact_experiment)
@click.option("--docs-char-budget", type=click.IntRange(1), default=8000, show_default=True)
@click.option("--docs-tree-entry-cap", type=click.IntRange(1), default=200, show_default=True)
@click.pass_context
def artifact_create(ctx: click.Context, name: str, model: str, reasoning_effort: str, repeats: int, freeze: bool, stage: str, claim_experiment: str | None, docs_char_budget: int, docs_tree_entry_cap: int) -> None:
    _run(ctx, "artifact", "command_artifact_create", name=name, model=model, reasoning_effort=reasoning_effort, repeats=repeats, freeze=freeze, stage=stage, claim_experiment=claim_experiment, docs_char_budget=docs_char_budget, docs_tree_entry_cap=docs_tree_entry_cap)


@artifact.command("freeze")
@click.argument("experiment", shell_complete=complete_artifact_experiment)
@click.pass_context
def artifact_freeze(ctx: click.Context, experiment: str) -> None:
    _run(ctx, "artifact", "command_artifact_freeze", experiment=experiment)


@artifact.command("list")
@click.pass_context
def artifact_list(ctx: click.Context) -> None:
    """List artifact experiments using standard-library SQLite only."""
    if not ctx.obj["database"].is_file():
        rows = []
    else:
        try:
            with sqlite3.connect(ctx.obj["database"]) as db:
                rows = db.execute(
                    "SELECT id,name,stage,status,model,reasoning_effort,repeat_count FROM artifact_experiments ORDER BY id"
                ).fetchall()
        except sqlite3.Error:
            rows = []
    if not rows:
        click.echo("No artifact experiments.")
        return
    click.echo("ID  NAME  STAGE  STATUS  MODEL  EFFORT  REPEATS")
    for row in rows:
        click.echo("  ".join(str(value or "—") for value in row))


@artifact.command("run")
@_artifact_options
@click.option("--retries", type=click.IntRange(0), default=2, show_default=True)
@click.pass_context
def artifact_run(ctx: click.Context, retries: int, **kwargs: Any) -> None:
    _run(ctx, "artifact", "command_artifact_run", retries=retries, **kwargs)


@artifact.command("finalize")
@click.option("--experiment", required=True, shell_complete=complete_artifact_experiment)
@click.pass_context
def artifact_finalize(ctx: click.Context, experiment: str) -> None:
    _run(ctx, "artifact", "command_artifact_finalize", experiment=experiment)


@artifact.command("diagnostic")
@click.option("--experiment", required=True, shell_complete=complete_artifact_experiment)
@click.option("--survey", type=click.Path(path_type=Path), default=Path("baseline/haim-lak/artifacts/qualtrics-survey-data.csv"), show_default=True)
@click.option("--run", "run_models", is_flag=True, help="Run 1 repetition on the diagnostic subset, then print the table (requires OPENAI_API_KEY).")
@click.option("--output", "output", type=Path, default=None, help="Also write the diagnostic table to this file as CSV.")
@click.option("--retries", type=click.IntRange(0), default=2, show_default=True)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def artifact_diagnostic(ctx: click.Context, experiment: str, survey: Path, run_models: bool, output: Path, retries: int, no_progress: bool) -> None:
    _run(ctx, "artifact", "command_artifact_diagnostic", experiment=experiment, survey=survey, run=run_models, output=output, retries=retries, no_progress=no_progress)


@artifact.group("batch")
def artifact_batch() -> None:
    """Prepare, submit, inspect, and collect Stage 1 OpenAI batches."""


@artifact_batch.command("prepare")
@_artifact_options
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.pass_context
def artifact_batch_prepare(ctx: click.Context, output: Path, **kwargs: Any) -> None:
    _run(ctx, "artifact", "command_batch_prepare", output=output, **kwargs)


@artifact_batch.command("submit")
@click.option("--input", "input_path", type=click.Path(path_type=Path), required=True)
@click.pass_context
def artifact_batch_submit(ctx: click.Context, input_path: Path) -> None:
    _run(ctx, "artifact", "command_batch_submit", input=input_path)


@artifact_batch.command("status")
@click.option("--batch-id", type=int, required=True, shell_complete=complete_artifact_batch)
@click.option("--wait", is_flag=True)
@click.option("--poll-seconds", type=click.FloatRange(1), default=30.0, show_default=True)
@click.pass_context
def artifact_batch_status(ctx: click.Context, batch_id: int, wait: bool, poll_seconds: float) -> None:
    _run(ctx, "artifact", "command_batch_status", batch_id=batch_id, wait=wait, poll_seconds=poll_seconds)


@artifact_batch.command("collect")
@click.option("--batch-id", type=int, required=True, shell_complete=complete_artifact_batch)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def artifact_batch_collect(ctx: click.Context, batch_id: int, no_progress: bool) -> None:
    _run(ctx, "artifact", "command_batch_collect", batch_id=batch_id, no_progress=no_progress)


@artifact.command("docs")
@_artifact_options
@click.option("--retries", type=click.IntRange(0), default=2, show_default=True)
@click.pass_context
def artifact_docs(ctx: click.Context, retries: int, **kwargs: Any) -> None:
    """Realtime Phase 3 run. The experiment must be --stage docs."""
    _run(ctx, "artifact", "command_artifact_run", retries=retries, **kwargs)


@artifact.command("align")
@click.option("--experiment", required=True, shell_complete=complete_artifact_experiment)
@click.option("--survey", type=click.Path(path_type=Path), default=Path("baseline/haim-lak/artifacts/qualtrics-survey-data.csv"), show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("artifacts/analysis/stage1-alignment.json"), show_default=True)
@click.pass_context
def artifact_align(ctx: click.Context, experiment: str, survey: Path, output: Path) -> None:
    _run(ctx, "artifact_report", "command_artifact_align", experiment=experiment, survey=survey, output=output)


@artifact.command("report")
@click.option("--experiment", required=True, shell_complete=complete_artifact_experiment)
@click.pass_context
def artifact_report(ctx: click.Context, experiment: str) -> None:
    _run(ctx, "artifact_report", "command_artifact_report", experiment=experiment)


@cli.group()
def resolve() -> None:
    """Phase 2a: resolve extracted URLs without downloading."""


@resolve.command("run")
@click.option("--experiment", shell_complete=complete_artifact_experiment)
@click.option("--year", type=int)
@click.option("--all", "all_years", is_flag=True, help="Resolve every extracted URL, not only 2021–2022.")
@click.option("--paper-id", type=int, shell_complete=complete_paper)
@click.option("--workers", type=click.IntRange(1), default=8, show_default=True)
@click.option("--urls-file", type=click.Path(path_type=Path), help="Fixture list of URLs, one per line. Does not touch the corpus.")
@click.option("--force", is_flag=True)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def resolve_run(ctx: click.Context, experiment: str | None, year: int | None, all_years: bool, paper_id: int | None, workers: int, urls_file: Path | None, force: bool, no_progress: bool) -> None:
    _run(ctx, "artifact_resolver", "command_resolve_run", experiment=experiment, year=year, all=all_years, paper_id=paper_id, workers=workers, urls_file=urls_file, force=force, no_progress=no_progress)


@resolve.command("report")
@click.pass_context
def resolve_report(ctx: click.Context) -> None:
    _run(ctx, "artifact_resolver", "command_resolve_report")


@cli.group()
def fetch() -> None:
    """Phase 2b/2c: fetch code repositories and budgeted data."""


@fetch.command("code")
@click.option("--experiment", shell_complete=complete_artifact_experiment)
@click.option("--workers", type=click.IntRange(1), default=2, show_default=True)
@click.option("--max-repo-gb", type=float, default=5.0, show_default=True)
@click.option("--store", type=click.Path(path_type=Path), default=Path("artifacts/stage1"), show_default=True)
@click.option("--force", is_flag=True)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def fetch_code(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "artifact_resolver", "command_fetch_code", **kwargs)


@fetch.command("data")
@click.option("--budget-gb", type=float, required=True)
@click.option("--require-code", is_flag=True, default=False, help="Only fetch data for papers whose code repository was already retrieved. Off by default: the candidate census needs data-only papers too.")
@click.option("--max-repo-gb", type=float, default=5.0, show_default=True)
@click.option("--store", type=click.Path(path_type=Path), default=Path("artifacts/stage1"), show_default=True)
@click.option("--force", is_flag=True)
@click.option("--no-progress", is_flag=True)
@click.pass_context
def fetch_data(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "artifact_resolver", "command_fetch_data", **kwargs)


@cli.command("completion-candidates", hidden=True)
@click.argument("candidate_type", type=click.Choice(("experiments", "ara-experiments", "artifact-experiments", "batches", "artifact-batches", "papers")))
@click.option("--experiment")
@click.option("--year", type=int)
@click.option("--status")
@click.option("--kind-filter", type=PAPER_KINDS)
@click.pass_context
def completion_candidates_command(ctx: click.Context, candidate_type: str, experiment: str | None, year: int | None, status: str | None, kind_filter: str | None) -> None:
    from completions import candidates_json, completion_candidates
    click.echo(candidates_json(completion_candidates(ctx.obj["database"], candidate_type, experiment=experiment, year=year, status=status, paper_kind=kind_filter)))


@cli.command()
@click.argument("shell", type=click.Choice(("nushell", "bash", "zsh", "fish", "powershell")))
def completions(shell: str) -> None:
    """Print a shell completion script; source or install it in the selected shell."""
    if shell == "nushell":
        from completions import nushell_module
        click.echo(nushell_module(), nl=False)
        return
    from click.shell_completion import get_completion_class
    completion = get_completion_class(shell)
    if completion is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    click.echo(completion(cli, {}, "lakdb", "_LAKDB_COMPLETE").source(), nl=False)
@cli.group()
def claim() -> None:
    """Stage 3: focal claim extraction, seeded by ARA sink nodes."""


@claim.command("create")
@click.argument("name")
@click.option("--model", default="gpt-5.6-terra", show_default=True)
@click.option("--reasoning-effort", type=REASONING, default="medium", show_default=True)
@click.option("--repeats", type=click.IntRange(1), default=3, show_default=True)
@click.option(
    "--ara-experiment",
    multiple=True,
    help="ARA experiment name to seed sinks from; repeatable. Defaults to every frozen "
    "multi-repetition experiment on the same model.",
)
@click.pass_context
def claim_create(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_create", **kwargs)


@claim.command("run")
@click.option("--experiment", required=True)
@click.option("--paper-id", type=int, shell_complete=complete_paper)
@click.option(
    "--all-applicable",
    is_flag=True,
    help="Extract for every in-scope paper instead of the executable set. Expensive; the "
    "executable set is what Stage 4 can act on.",
)
@click.option("--force", is_flag=True, help="Re-run papers that already succeeded.")
@click.option("--no-progress", is_flag=True)
@click.pass_context
def claim_run(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_run", **kwargs)


@claim.command("aggregate")
@click.option("--experiment", required=True)
@click.pass_context
def claim_aggregate(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_aggregate", **kwargs)


@claim.command("report")
@click.option("--experiment", required=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/analysis/claims.json"),
    show_default=True,
)
@click.pass_context
def claim_report(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_report", **kwargs)
@claim.group("batch")
def claim_batch() -> None:
    """Batch mode: half price, 24-hour completion window."""


@claim_batch.command("prepare")
@click.option("--experiment", required=True)
@click.option("--paper-id", type=int, shell_complete=complete_paper)
@click.option("--all-applicable", is_flag=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/batch/claim-input.jsonl"),
    show_default=True,
)
@click.pass_context
def claim_batch_prepare(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_batch_prepare", **kwargs)


@claim_batch.command("submit")
@click.option("--batch-id", type=int, required=True)
@click.pass_context
def claim_batch_submit(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_batch_submit", **kwargs)


@claim_batch.command("status")
@click.option("--batch-id", type=int, required=True)
@click.option("--wait", is_flag=True, help="Poll until the batch reaches a terminal state.")
@click.option("--poll-seconds", type=click.IntRange(5), default=60, show_default=True)
@click.pass_context
def claim_batch_status(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_batch_status", **kwargs)


@claim_batch.command("collect")
@click.option("--batch-id", type=int, required=True)
@click.pass_context
def claim_batch_collect(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "claim", "command_claim_batch_collect", **kwargs)


@cli.group("arm-a")
def arm_a() -> None:
    """Stage 4 Arm A: run the authors' own code and see whether it reproduces."""


@arm_a.command("create")
@click.argument("name")
@click.option("--model", default="gpt-5.6-terra", show_default=True)
@click.option("--reasoning-effort", type=REASONING, default="medium", show_default=True)
@click.option("--repeats", type=click.IntRange(1), default=3, show_default=True)
@click.option("--claim-experiment", help="Stage 3 experiment supplying the focal metric name. Latest frozen one by default.")
@click.option("--message-limit", type=click.IntRange(1), default=100, show_default=True)
@click.option("--cost-limit", type=float, default=2.0, show_default=True, help="USD per run. Pass 0 to disable and rely on --message-limit.")
@click.option("--time-limit", type=click.IntRange(60), default=1800, show_default=True, help="Wall clock per run, seconds; reported with the result.")
@click.option("--command-timeout", type=click.IntRange(10), default=300, show_default=True)
@click.option("--no-network", is_flag=True, help="Cut the sandbox off the network entirely. Dependency installation (rung A1) then cannot happen.")
@click.pass_context
def arm_a_create(ctx: click.Context, cost_limit: float, **kwargs: Any) -> None:
    _run(ctx, "arm_a_runner", "command_arm_a_create", cost_limit=cost_limit or None, **kwargs)


@arm_a.command("run")
@click.option("--experiment", required=True)
@click.option("--paper-id", type=int, shell_complete=complete_paper)
@click.option("--repeats", type=click.IntRange(1), help="Override the experiment's repetition count; 1 for a pilot.")
@click.option("--log-dir", type=click.Path(path_type=Path), default=Path("artifacts/arm_a/logs"), show_default=True)
@click.option("--force", is_flag=True, help="Re-run repetitions that already succeeded.")
@click.option("--skip-leak-check", is_flag=True, help="Proceed when the reported value's digits appear in the prompt by coincidence. Recorded on the run.")
@click.option("--no-progress", is_flag=True)
@click.pass_context
def arm_a_run(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_a_runner", "command_arm_a_run", **kwargs)


@arm_a.command("collect")
@click.option("--experiment", required=True)
@click.pass_context
def arm_a_collect(ctx: click.Context, **kwargs: Any) -> None:
    """Parse the Inspect logs into arm_a_runs. Safe to repeat."""
    _run(ctx, "arm_a_runner", "command_arm_a_collect", **kwargs)


@arm_a.command("verify")
@click.option("--experiment", required=True)
@click.pass_context
def arm_a_verify(ctx: click.Context, **kwargs: Any) -> None:
    """Confirm each produced value is at the location the agent named."""
    _run(ctx, "arm_a_runner", "command_arm_a_verify", **kwargs)


@arm_a.command("compare")
@click.option("--experiment", required=True)
@click.pass_context
def arm_a_compare(ctx: click.Context, **kwargs: Any) -> None:
    """Join the withheld reported value and apply the docs/protocol.md §6 tolerances."""
    _run(ctx, "arm_a_runner", "command_arm_a_compare", **kwargs)


@arm_a.command("report")
@click.option("--experiment", required=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/analysis/arm-a.json"),
    show_default=True,
)
@click.pass_context
def arm_a_report(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_a_runner", "command_arm_a_report", **kwargs)



@cli.group("arm-b")
def arm_b() -> None:
    """Stage 4 Arm B: independently reanalyse from paper and data, without author code."""


@arm_b.command("create")
@click.argument("name")
@click.option("--model", default="gpt-5.6-terra", show_default=True)
@click.option("--reasoning-effort", type=REASONING, default="medium", show_default=True)
@click.option("--repeats", type=click.IntRange(1), default=3, show_default=True)
@click.option("--claim-experiment", help="Frozen Stage 3 experiment supplying focal claims.")
@click.option("--message-limit", type=click.IntRange(1), default=100, show_default=True)
@click.option("--cost-limit", type=float, default=2.0, show_default=True, help="USD per run; pass 0 to disable.")
@click.option("--time-limit", type=click.IntRange(60), default=1800, show_default=True)
@click.option("--command-timeout", type=click.IntRange(10), default=300, show_default=True)
@click.option("--no-network", is_flag=True, help="Disallow outbound network access in the sandbox.")
@click.pass_context
def arm_b_create(ctx: click.Context, cost_limit: float, **kwargs: Any) -> None:
    _run(ctx, "arm_b_runner", "command_arm_b_create", cost_limit=cost_limit or None, **kwargs)


@arm_b.command("run")
@click.option("--experiment", required=True)
@click.option("--paper-id", type=int, shell_complete=complete_paper)
@click.option("--repeats", type=click.IntRange(1), help="Override the experiment's repetition count.")
@click.option("--log-dir", type=click.Path(path_type=Path), default=Path("artifacts/arm_b/logs"), show_default=True)
@click.option("--force", is_flag=True, help="Re-run repetitions already marked succeeded.")
@click.option("--skip-leak-check", is_flag=True, help="Record and acknowledge a coincidental prompt digit match.")
@click.option("--no-progress", is_flag=True)
@click.pass_context
def arm_b_run(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_b_runner", "command_arm_b_run", **kwargs)


@arm_b.command("collect")
@click.option("--experiment", required=True)
@click.pass_context
def arm_b_collect(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_b_runner", "command_arm_b_collect", **kwargs)


@arm_b.command("verify")
@click.option("--experiment", required=True)
@click.pass_context
def arm_b_verify(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_b_runner", "command_arm_b_verify", **kwargs)


@arm_b.command("compare")
@click.option("--experiment", required=True)
@click.pass_context
def arm_b_compare(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_b_runner", "command_arm_b_compare", **kwargs)


@arm_b.command("report")
@click.option("--experiment", required=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("artifacts/analysis/arm-b.json"), show_default=True)
@click.pass_context
def arm_b_report(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_b_runner", "command_arm_b_report", **kwargs)


@arm_b.command("contrast")
@click.option("--arm-a", "arm_a", required=True, help="Completed Arm A experiment name.")
@click.option("--arm-b", "arm_b", required=True, help="Completed Arm B experiment name.")
@click.pass_context
def arm_b_contrast(ctx: click.Context, **kwargs: Any) -> None:
    _run(ctx, "arm_b_runner", "command_arm_b_contrast", **kwargs)


def main() -> None:
    try:
        cli(prog_name="lakdb")
    except (FileNotFoundError, sqlite3.Error, ValueError, RuntimeError, OSError) as exc:
        click.echo(f"Error: {type(exc).__name__}: {exc}", err=True)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        click.echo("Interrupted.", err=True)
        raise SystemExit(130)
