"""Read-only Textual dashboard for screening history, evidence, and usage."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, ClassVar

from rich.text import Text
from ara_usage import ara_experiment_names, assessment_rows, report_windows
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)

EVIDENCE_LIMIT = 2_000
MATCHED_STATUSES = {"exact", "normalized", "word_sequence"}
UNRESOLVED_STATUSES = {"ambiguous", "not_found", "pending"}


class CostLineChart(Static):
    """Small terminal line chart without a plotting backend or image files."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.data: list[float] = []

    def set_data(self, values: list[float]) -> None:
        self.data = values
        self.refresh()

    def render(self) -> Text:
        if not self.data:
            return Text("No cost data", style="dim")
        width = max(12, self.size.width - 2)
        height = 3
        values = [self.data[round(i * (len(self.data) - 1) / (width - 1))]
                  for i in range(min(width, len(self.data)))]
        low, high = min(values), max(values)
        scale = high - low or 1.0
        rows = [height - 1 - round((value - low) / scale * (height - 1)) for value in values]
        canvas = [[" " for _ in values] for _ in range(height)]
        for column, row in enumerate(rows):
            canvas[row][column] = "●"
            if column and rows[column - 1] != row:
                canvas[(rows[column - 1] + row) // 2][column - 1] = "╲" if row > rows[column - 1] else "╱"
        return Text("\n".join("".join(line) for line in canvas), style="#58a6ff")


def short_title(value: str | None) -> str:
    """Return the required compact table title; full titles remain in details."""
    title = " ".join((value or "[untitled]").split())
    return title if len(title) <= 20 else title[:20] + "…"


def preview_text(value: str | None, width: int = 72) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= width else text[:width] + "…"


def human_tokens(value: int | None) -> str:
    number = value or 0
    if number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:,}"


def human_time(value: int | None) -> str:
    milliseconds = value or 0
    if milliseconds >= 3_600_000:
        return f"{milliseconds / 3_600_000:.1f}h"
    if milliseconds >= 60_000:
        return f"{milliseconds / 60_000:.1f}m"
    return f"{milliseconds / 1000:.2f}s"


def parse_result(value: str | None) -> dict[str, Any]:
    try:
        result = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def launch_file(path: Path) -> None:
    """Open a local artifact without invoking a shell."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class MetricCard(Static):
    def set_metric(self, title: str, value: str, detail: str = "") -> None:
        self.update(
            f"[bold #8b949e]{title}[/]\n[bold #58a6ff]{value}[/]\n[dim]{detail}[/]"
        )


HELP_TEXT = """[bold #58a6ff]LAK dashboard navigation[/]

[bold]Global[/]
1–7       switch tabs                 r          refresh, preserving position
/         focus evidence search       Esc        clear search / focus tree
?         open this help              q          quit

[bold]Evidence tree[/]
↑ ↓       move and preview             ← →        collapse / expand
Space     toggle branch                Enter      select / toggle branch
f / a     final evidence / all runs    m / u      matched / unresolved
d         disagreement papers          Alt+1–3    one repetition; Alt+0 all
n / N     next / previous unresolved

[bold]Files and clipboard[/]
o         open highlighted PDF         Shift+O    open original PDF
c         copy selected quotation      Shift+C    copy highlighted/source path

The Evidence workspace uses Paper → Run → Evidence. Final evidence deduplicates
matching quotations and retains all contributing run numbers. Color is paired
with ✓ matched, ! ambiguous, and × not found text for accessibility.
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Static(HELP_TEXT, id="help-dialog")

    def action_dismiss(self) -> None:
        self.dismiss(None)


class ScreeningDashboard(App[None]):
    TITLE = "LAK Applicability Dashboard"
    CSS = """
    Screen { background: #0d1117; color: #f0f6fc; }
    Header { background: #161b22; color: #f0f6fc; }
    Footer { background: #161b22; color: #8b949e; }
    #subtitle { height: 2; padding: 0 2; color: #8b949e; background: #161b22; }
    #metrics { height: 7; padding: 1 1; }
    MetricCard { width: 1fr; height: 5; margin: 0 1; padding: 0 1; border: round #30363d; background: #161b22; text-align: center; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1; }
    DataTable { height: 1fr; background: #0d1117; border: round #30363d; }
    DataTable:focus { border: round #58a6ff; }
    DataTable > .datatable--header { background: #21262d; color: #58a6ff; text-style: bold; }
    DataTable > .datatable--cursor { background: #1f6feb; color: #ffffff; }
    Tree { height: 1fr; width: 2fr; background: #0d1117; border: round #30363d; padding: 0 1; }
    Tree:focus { border: round #58a6ff; }
    Tree > .tree--cursor { background: #1f6feb; color: #ffffff; }
    .detail { height: 14; border: round #30363d; background: #161b22; padding: 1 2; scrollbar-color: #58a6ff; }
    .detail:focus { border: round #58a6ff; }
    #papers-toolbar, #evidence-toolbar, #ara-cost-toolbar { height: 3; layout: horizontal; }
    #paper-label-filter { width: 28; }
    #evidence-search { width: 1fr; margin-right: 1; }
    #evidence-mode { width: 27; margin-right: 1; }
    #evidence-run { width: 18; }
    #evidence-status, #ara-cost-status { height: 2; padding: 0 1; color: #8b949e; }
    #ara-cost-experiment { width: 48; }
    #ara-cost-graph-label { height: 1; padding: 0 1; color: #8b949e; }
    #ara-cost-graph { height: 3; margin: 0 0 1 0; }
    /* Keep the per-paper list usable; reveal detail only after selecting a row. */
    #ara-cost-detail { display: none; height: 10; }
    #evidence-workspace { height: 1fr; layout: horizontal; }
    #evidence-detail { height: 1fr; width: 3fr; margin-left: 1; }
    Screen.narrow #metrics { display: none; }
    Screen.narrow #evidence-workspace { layout: vertical; }
    Screen.narrow #evidence-tree { width: 1fr; height: 1fr; }
    Screen.narrow #evidence-detail { width: 1fr; height: 1fr; margin-left: 0; margin-top: 1; }
    Screen.narrow #evidence-mode { width: 22; }
    Screen.narrow #evidence-run { display: none; }
    HelpScreen { align: center middle; background: rgba(0, 0, 0, 0.72); }
    #help-dialog { width: 82; max-width: 95%; height: 28; padding: 1 2; border: round #58a6ff; background: #161b22; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_data", "Refresh"),
        Binding("slash", "focus_search", "Search"),
        Binding("escape", "clear_search", "Back", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("1", "show_tab('overview')", "Overview", show=False),
        Binding("2", "show_tab('tasks')", "Tasks", show=False),
        Binding("3", "show_tab('papers')", "Papers", show=False),
        Binding("4", "show_tab('batches')", "Batches", show=False),
        Binding("5", "show_tab('evidence')", "Evidence"),
        Binding("6", "show_tab('errors')", "Errors", show=False),
        Binding("7", "show_tab('ara-cost')", "ARA cost", show=False),
        Binding("f", "evidence_mode('final')", "Final", show=False),
        Binding("a", "evidence_mode('all')", "All", show=False),
        Binding("m", "evidence_mode('matched')", "Matched", show=False),
        Binding("u", "evidence_mode('unresolved')", "Unresolved", show=False),
        Binding("d", "evidence_mode('disagreements')", "Disagreements", show=False),
        Binding("l", "focus_paper_label", "Paper label", show=False),
        Binding("alt+0", "run_filter(0)", "All runs", show=False),
        Binding("alt+1", "run_filter(1)", "Run 1", show=False),
        Binding("alt+2", "run_filter(2)", "Run 2", show=False),
        Binding("alt+3", "run_filter(3)", "Run 3", show=False),
        Binding("n", "next_unresolved(1)", "Next issue", show=False),
        Binding("shift+n", "next_unresolved(-1)", "Previous issue", show=False),
        Binding("o", "open_highlight", "Open highlight", show=False),
        Binding("shift+o", "open_source", "Open source", show=False),
        Binding("c", "copy_quote", "Copy quote", show=False),
        Binding("shift+c", "copy_path", "Copy path", show=False),
    ]

    def __init__(
        self,
        database: Path,
        experiment: str | None = None,
        year: int | None = None,
        status: str | None = None,
        refresh_seconds: float = 30.0,
    ):
        super().__init__()
        self.database = database.expanduser().resolve()
        self.experiment = experiment
        self.year = year
        self.status_filter = status
        # Kept as an internal constructor argument for test callers; the UI is
        # deliberately manual-refresh only (press r).
        self.last_signature: tuple[Any, ...] | None = None
        self.last_evidence_signature: tuple[Any, ...] | None = None
        # Artifact JSON is comparatively expensive to parse; cache the ARA view
        # until a manual refresh or the selected experiment's runs change.
        self.last_ara_cost_signature: tuple[Any, ...] | None = None
        self.evidence_mode = "final"
        self.evidence_run = 0
        self.paper_label_filter = "all"
        self.evidence_search = ""
        self.evidence_nodes: dict[str, Any] = {}
        self.unresolved_nodes: list[Any] = []
        self.selected_tree_key: str | None = None
        self.selected_quote: str | None = None
        self.selected_source_path: str | None = None
        self.selected_highlight_path: str | None = None
        self.ara_cost_experiment: str | None = None
        self.ara_cost_rows: dict[str, dict[str, Any]] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(
            "Read-only • SQLite WAL • 1–6 tabs • / search • ? help",
            id="subtitle",
        )
        with Horizontal(id="metrics"):
            yield MetricCard(id="metric-total")
            yield MetricCard(id="metric-input")
            yield MetricCard(id="metric-output")
            yield MetricCard(id="metric-progress")
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                yield DataTable(id="overview-table", zebra_stripes=True, cursor_type="row")
            with TabPane("Tasks", id="tasks"), Vertical():
                yield DataTable(id="tasks-table", zebra_stripes=True, cursor_type="row")
                yield RichLog(id="task-detail", classes="detail", wrap=True, markup=False)
            with TabPane("Papers", id="papers"), Vertical():
                with Horizontal(id="papers-toolbar"):
                    yield Select(
                        [("All final labels", "all"), ("Applicable", "applicable"), ("Not applicable", "not_applicable"), ("Unclear", "unclear"), ("No final decision", "none")],
                        value="all", allow_blank=False, id="paper-label-filter",
                    )
                    yield Label("Loading papers…", id="papers-status")
                yield DataTable(id="papers-table", zebra_stripes=True, cursor_type="row")
                yield RichLog(id="paper-detail", classes="detail", wrap=True, markup=False)
            with TabPane("Batches", id="batches"):
                yield DataTable(id="batches-table", zebra_stripes=True, cursor_type="row")
            with TabPane("Evidence", id="evidence"), Vertical():
                with Horizontal(id="evidence-toolbar"):
                    yield Input(placeholder="Search paper, quote, explanation, type, or PDF text  [/ ]", id="evidence-search")
                    yield Select(
                        [
                            ("Final evidence", "final"),
                            ("All runs", "all"),
                            ("Matched", "matched"),
                            ("Unresolved", "unresolved"),
                            ("Disagreements", "disagreements"),
                        ],
                        value="final",
                        allow_blank=False,
                        id="evidence-mode",
                    )
                    yield Select(
                        [("All runs", 0), ("Run 1", 1), ("Run 2", 2), ("Run 3", 3)],
                        value=0,
                        allow_blank=False,
                        id="evidence-run",
                    )
                yield Label("Loading evidence…", id="evidence-status")
                with Horizontal(id="evidence-workspace"):
                    yield Tree("Papers", id="evidence-tree")
                    yield RichLog(id="evidence-detail", classes="detail", wrap=True, markup=False)
            with TabPane("Errors", id="errors"):
                yield DataTable(id="errors-table", zebra_stripes=True, cursor_type="row")
            with TabPane("ARA Cost", id="ara-cost"), Vertical():
                with Horizontal(id="ara-cost-toolbar"):
                    yield Select([], prompt="Select ARA experiment", allow_blank=True, id="ara-cost-experiment")
                    yield Label("Artifact-backed prompt-cache pricing", id="ara-cost-status")
                yield DataTable(id="ara-cost-summary-table", zebra_stripes=True, cursor_type="row")
                yield Label("Cost per assessment over time", id="ara-cost-graph-label")
                yield CostLineChart(id="ara-cost-graph")
                yield DataTable(id="ara-cost-runs-table", zebra_stripes=True, cursor_type="row")
                yield RichLog(id="ara-cost-detail", classes="detail", wrap=True, markup=False)
        yield Footer()

    def _connect(self) -> sqlite3.Connection:
        if not self.database.exists():
            raise FileNotFoundError(f"Database does not exist: {self.database}")
        connection = sqlite3.connect(
            f"file:{self.database.as_posix()}?mode=ro", uri=True, timeout=5
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def on_mount(self) -> None:
        self.screen.set_class(self.size.width < 110, "narrow")
        self._setup_tables()
        self._refresh_data(force_evidence=True)
        # Do not poll artifacts automatically: refresh explicitly with r.

    def on_resize(self, event: events.Resize) -> None:
        self.screen.set_class(event.size.width < 110, "narrow")

    def _setup_tables(self) -> None:
        columns: dict[str, tuple[str, ...]] = {
            "overview-table": ("Year", "Tasks", "Papers", "Input", "Cached", "Output", "Reasoning", "Total", "Time"),
            "tasks-table": ("ID", "Year", "Paper", "Run", "Mode", "Input", "Output", "Total", "Time", "Label", "Status", "Model"),
            "papers-table": ("Year", "Paper", "Experiment", "Type", "Runs", "Input", "Output", "Total", "Time", "Label", "Agreement", "Evidence"),
            "batches-table": ("ID", "Experiment", "Status", "Requests", "Completed", "Failed", "Submitted", "Turnaround"),
            "errors-table": ("Task", "Year", "Paper", "Type", "Error", "When"),
            "ara-cost-summary-table": ("Window", "Runs", "Cached", "Write", "Fresh", "Output", "Cost", "$/run", "Health"),
            "ara-cost-runs-table": ("Paper", "DOI", "Runs", "Cached", "Write", "Fresh", "Output", "Total cost", "Avg/run"),
        }
        for table_id, names in columns.items():
            table = self.query_one(f"#{table_id}", DataTable)
            for name in names:
                table.add_column(name, key=name.lower())

    def _where(self, alias: str = "e") -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.experiment:
            clauses.append(f"{alias}.name=?")
            params.append(self.experiment)
        if self.year:
            clauses.append("p.collection_year=?")
            params.append(self.year)
        return (" AND " + " AND ".join(clauses) if clauses else "", params)

    def _refresh_if_changed(self) -> None:
        try:
            with closing(self._connect()) as db:
                signature = self._database_signature(db)
            if signature != self.last_signature:
                self._refresh_data(force_evidence=False)
        except (sqlite3.Error, FileNotFoundError):
            return

    def action_refresh_data(self) -> None:
        self._refresh_data(force_evidence=True)

    def _refresh_data(self, force_evidence: bool) -> None:
        try:
            with closing(self._connect()) as db:
                self._load_metrics(db)
                self._load_overview(db)
                self._load_tasks(db)
                self._load_papers(db)
                self._load_batches(db)
                ara_signature = self._ara_cost_signature(db)
                if force_evidence or ara_signature != self.last_ara_cost_signature:
                    self._load_ara_cost(db)
                    self.last_ara_cost_signature = ara_signature
                evidence_signature = self._evidence_signature(db)
                if force_evidence or evidence_signature != self.last_evidence_signature:
                    self._load_evidence(db)
                    self.last_evidence_signature = evidence_signature
                self._load_errors(db)
                self.last_signature = self._database_signature(db)
        except (sqlite3.Error, FileNotFoundError) as exc:
            self.notify(str(exc), severity="error", timeout=8)

    @staticmethod
    def _database_signature(db: sqlite3.Connection) -> tuple[Any, ...]:
        return tuple(
            db.execute(
                "SELECT (SELECT COALESCE(MAX(id),0) FROM task_executions),"
                "(SELECT COUNT(*) FROM task_executions),"
                "(SELECT COALESCE(MAX(last_polled_at),'') FROM openai_batch_jobs),"
                "(SELECT COALESCE(MAX(aggregated_at),'') FROM applicability_decisions),"
                "(SELECT COALESCE(MAX(created_at),'') FROM highlight_artifacts),"
                "(SELECT COALESCE(MAX(finished_at),'') FROM ara_runs),"
                "(SELECT COUNT(*) FROM ara_runs)"
            ).fetchone()
        )

    def _ara_cost_signature(self, db: sqlite3.Connection) -> tuple[Any, ...]:
        """A cheap SQLite-only invalidation key; do not read artifacts here."""
        return (self.ara_cost_experiment, *tuple(db.execute(
            "SELECT COUNT(r.id), COALESCE(MAX(r.finished_at),''), COALESCE(MAX(r.id),0) "
            "FROM ara_runs r JOIN ara_experiments e ON e.id=r.experiment_id "
            "WHERE e.name=? AND r.status='succeeded'",
            (self.ara_cost_experiment or "",),
        ).fetchone()))

    def _evidence_signature(self, db: sqlite3.Connection) -> tuple[Any, ...]:
        where, params = self._where()
        return tuple(
            db.execute(
                "SELECT COUNT(ev.id),COALESCE(MAX(ev.id),0),"
                "COALESCE(SUM(CASE WHEN ev.match_status IN ('exact','normalized','word_sequence') THEN 1 ELSE 0 END),0),"
                "COALESCE(SUM(CASE WHEN ev.match_status IN ('ambiguous','not_found','pending') THEN 1 ELSE 0 END),0),"
                "COALESCE(MAX(h.created_at),'') "
                "FROM applicability_evidence ev "
                "JOIN applicability_runs r ON r.id=ev.applicability_run_id "
                "JOIN papers p ON p.id=r.paper_id "
                "JOIN study_experiments e ON e.id=r.experiment_id "
                "LEFT JOIN highlight_artifacts h ON h.paper_id=p.id AND h.experiment_id=e.id "
                "WHERE 1=1" + where,
                params,
            ).fetchone()
        )

    def _load_metrics(self, db: sqlite3.Connection) -> None:
        where, params = self._where()
        row = db.execute(
            "SELECT COUNT(*) tasks,COUNT(DISTINCT te.paper_id) papers,"
            "COALESCE(SUM(te.input_tokens),0) input_tokens,COALESCE(SUM(te.cached_input_tokens),0) cached,"
            "COALESCE(SUM(te.output_tokens),0) output_tokens,COALESCE(SUM(te.reasoning_output_tokens),0) reasoning,"
            "COALESCE(SUM(te.total_tokens),0) total_tokens FROM task_executions te "
            "LEFT JOIN papers p ON p.id=te.paper_id LEFT JOIN study_experiments e ON e.id=te.experiment_id "
            "WHERE te.status='succeeded'" + where,
            params,
        ).fetchone()
        progress_params: list[Any] = []
        decision_join = ""
        if self.experiment:
            decision_join = " AND d.experiment_id=(SELECT id FROM study_experiments WHERE name=?)"
            progress_params.append(self.experiment)
        progress_where = ""
        if self.year:
            progress_where = " WHERE p.collection_year=?"
            progress_params.append(self.year)
        progress = db.execute(
            "SELECT COUNT(DISTINCT p.id) total,COUNT(DISTINCT CASE WHEN d.id IS NOT NULL THEN p.id END) complete "
            "FROM papers p LEFT JOIN applicability_decisions d ON d.paper_id=p.id" + decision_join + progress_where,
            progress_params,
        ).fetchone()
        self.query_one("#metric-total", MetricCard).set_metric("TOTAL TOKENS", human_tokens(row["total_tokens"]), f"{row['tasks']:,} tasks")
        self.query_one("#metric-input", MetricCard).set_metric("INPUT", human_tokens(row["input_tokens"]), f"{human_tokens(row['cached'])} cached")
        self.query_one("#metric-output", MetricCard).set_metric("OUTPUT", human_tokens(row["output_tokens"]), f"{human_tokens(row['reasoning'])} reasoning")
        self.query_one("#metric-progress", MetricCard).set_metric("PAPERS", f"{progress['complete'] or 0}/{progress['total'] or 0}", "final decisions")

    def _selected_table_key(self, table: DataTable[Any]) -> str | None:
        try:
            return str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        except Exception:
            return None

    def _fill(self, table_id: str, rows: list[tuple[Any, ...]], keys: list[str] | None = None) -> None:
        table = self.query_one(f"#{table_id}", DataTable)
        selected = self._selected_table_key(table)
        table.clear()
        for index, row in enumerate(rows):
            table.add_row(*row, key=(keys[index] if keys else None))
        if selected and keys and selected in keys:
            table.move_cursor(row=table.get_row_index(selected), animate=False)

    def _load_overview(self, db: sqlite3.Connection) -> None:
        where, params = self._where()
        rows = db.execute(
            "SELECT COALESCE(p.collection_year,0) year,COUNT(*) tasks,COUNT(DISTINCT te.paper_id) papers,"
            "COALESCE(SUM(te.input_tokens),0) input_tokens,COALESCE(SUM(te.cached_input_tokens),0) cached,"
            "COALESCE(SUM(te.output_tokens),0) output_tokens,COALESCE(SUM(te.reasoning_output_tokens),0) reasoning,"
            "COALESCE(SUM(te.total_tokens),0) total_tokens,COALESCE(SUM(te.total_duration_ms),0) duration "
            "FROM task_executions te LEFT JOIN papers p ON p.id=te.paper_id "
            "LEFT JOIN study_experiments e ON e.id=te.experiment_id WHERE te.status='succeeded'" + where +
            " GROUP BY p.collection_year ORDER BY p.collection_year",
            params,
        ).fetchall()
        self._fill("overview-table", [(
            str(r["year"] or "Local"), f"{r['tasks']:,}", f"{r['papers']:,}", human_tokens(r["input_tokens"]),
            human_tokens(r["cached"]), human_tokens(r["output_tokens"]), human_tokens(r["reasoning"]),
            human_tokens(r["total_tokens"]), human_time(r["duration"]),
        ) for r in rows])

    @staticmethod
    def _status(value: str) -> Text:
        styles = {"succeeded": "bold #3fb950", "completed": "bold #3fb950", "failed": "bold #f85149", "expired": "bold #f85149", "cancelled": "bold #f85149", "running": "bold #58a6ff", "submitted": "bold #a371f7", "prepared": "bold #d29922", "pending": "#8b949e"}
        symbols = {"succeeded": "✓", "completed": "✓", "failed": "×", "expired": "×", "cancelled": "×", "running": "●", "submitted": "◆", "prepared": "◌", "pending": "○"}
        return Text(f"{symbols.get(value, '•')} {value.upper()}", style=styles.get(value, "#8b949e"))

    def _load_tasks(self, db: sqlite3.Connection) -> None:
        where, params = self._where()
        if self.status_filter:
            where += " AND te.status=?"
            params.append(self.status_filter)
        rows = db.execute(
            "SELECT te.id,p.collection_year,p.title,r.repetition_index,te.execution_mode,te.input_tokens,te.output_tokens,"
            "te.total_tokens,te.total_duration_ms,r.label,te.status,te.model FROM task_executions te "
            "LEFT JOIN papers p ON p.id=te.paper_id LEFT JOIN applicability_runs r ON r.id=te.applicability_run_id "
            "LEFT JOIN study_experiments e ON e.id=te.experiment_id WHERE 1=1" + where + " ORDER BY te.id DESC LIMIT 500",
            params,
        ).fetchall()
        values: list[tuple[Any, ...]] = []
        for row in rows:
            label_style = {"applicable": "#3fb950", "not_applicable": "#f0883e", "unclear": "#d29922"}.get(row["label"], "#8b949e")
            values.append((str(row["id"]), str(row["collection_year"] or "—"), short_title(row["title"]), str(row["repetition_index"] or "—"), row["execution_mode"], human_tokens(row["input_tokens"]), human_tokens(row["output_tokens"]), human_tokens(row["total_tokens"]), human_time(row["total_duration_ms"]), Text(row["label"] or "—", style=label_style), self._status(row["status"]), row["model"] or "—"))
        self._fill("tasks-table", values, [str(row["id"]) for row in rows])

    def _load_papers(self, db: sqlite3.Connection) -> None:
        where, params = self._where()
        label_clause = ""
        if self.paper_label_filter == "none":
            label_clause = " AND d.final_label IS NULL"
        elif self.paper_label_filter != "all":
            label_clause = " AND d.final_label=?"
            params.append(self.paper_label_filter)
        rows = db.execute(
            "SELECT p.id paper_id,e.id experiment_id,p.collection_year,p.title,e.name experiment,fa.paper_kind,"
            "COUNT(DISTINCT r.id) runs,COALESCE(SUM(te.input_tokens),0) input_tokens,COALESCE(SUM(te.output_tokens),0) output_tokens,"
            "COALESCE(SUM(te.total_tokens),0) total_tokens,COALESCE(SUM(te.total_duration_ms),0) duration,d.final_label,d.agreement_type,d.evidence_status "
            "FROM papers p JOIN file_assets fa ON fa.paper_id=p.id LEFT JOIN applicability_runs r ON r.paper_id=p.id "
            "LEFT JOIN task_executions te ON te.applicability_run_id=r.id LEFT JOIN applicability_decisions d ON d.paper_id=p.id "
            "AND d.content_version_id=r.content_version_id AND d.experiment_id=r.experiment_id "
            "LEFT JOIN study_experiments e ON e.id=r.experiment_id WHERE 1=1" + where + label_clause +
            " GROUP BY p.id,e.id,d.id ORDER BY p.collection_year,p.title,e.name",
            params,
        ).fetchall()
        values = [(str(r["collection_year"]), short_title(r["title"]), r["experiment"] or "—", r["paper_kind"], str(r["runs"]), human_tokens(r["input_tokens"]), human_tokens(r["output_tokens"]), human_tokens(r["total_tokens"]), human_time(r["duration"]), r["final_label"] or "—", r["agreement_type"] or "—", r["evidence_status"] or "—") for r in rows]
        self._fill("papers-table", values, [f"{r['paper_id']}:{r['experiment_id'] or 0}" for r in rows])
        filter_text = "all final labels" if self.paper_label_filter == "all" else self.paper_label_filter.replace("_", " ")
        self.query_one("#papers-status", Label).update(f"{len(rows)} papers shown · filter: {filter_text}")

    def _load_batches(self, db: sqlite3.Connection) -> None:
        clauses: list[str] = []
        params: list[Any] = []
        if self.experiment:
            clauses.append("e.name=?")
            params.append(self.experiment)
        rows = db.execute(
            "SELECT b.id,e.name,b.status,b.request_count,b.request_counts_json,b.submitted_at,b.completed_at "
            "FROM openai_batch_jobs b JOIN study_experiments e ON e.id=b.experiment_id" +
            (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY b.id DESC",
            params,
        ).fetchall()
        values = []
        for row in rows:
            counts = parse_result(row["request_counts_json"])
            turnaround = "—"
            if row["submitted_at"] and row["completed_at"]:
                from datetime import datetime
                turnaround = human_time(round((datetime.fromisoformat(row["completed_at"]) - datetime.fromisoformat(row["submitted_at"])).total_seconds() * 1000))
            values.append((str(row["id"]), row["name"], self._status(row["status"]), str(row["request_count"]), str(counts.get("completed", 0)), str(counts.get("failed", 0)), row["submitted_at"] or "—", turnaround))
        self._fill("batches-table", values, [str(row["id"]) for row in rows])

    def _load_ara_cost(self, db: sqlite3.Connection) -> None:
        names = ara_experiment_names(db)
        selector = self.query_one("#ara-cost-experiment", Select)
        if self.ara_cost_experiment not in names:
            self.ara_cost_experiment = names[0] if names else None
        selector.set_options([(name, name) for name in names])
        if self.ara_cost_experiment:
            selector.value = self.ara_cost_experiment
        summary_table = self.query_one("#ara-cost-summary-table", DataTable)
        runs_table = self.query_one("#ara-cost-runs-table", DataTable)
        graph = self.query_one("#ara-cost-graph", CostLineChart)
        detail = self.query_one("#ara-cost-detail", RichLog)
        self.ara_cost_rows = {}
        if not self.ara_cost_experiment:
            self._fill("ara-cost-summary-table", [])
            self._fill("ara-cost-runs-table", [])
            graph.set_data([])
            detail.clear()
            detail.styles.display = "none"
            self.query_one("#ara-cost-graph-label", Label).update("No cost data")
            self.query_one("#ara-cost-status", Label).update("No ARA experiments in this database")
            return
        try:
            model, rows = assessment_rows(db, self.ara_cost_experiment)
        except (OSError, ValueError, sqlite3.Error) as exc:
            self._fill("ara-cost-summary-table", [])
            self._fill("ara-cost-runs-table", [])
            graph.set_data([])
            detail.clear()
            detail.styles.display = "none"
            self.query_one("#ara-cost-graph-label", Label).update("No cost data")
            self.query_one("#ara-cost-status", Label).update(str(exc))
            return
        summaries = report_windows(rows)
        self._fill("ara-cost-summary-table", [
            (label, str(item["assessments"]),
             "—" if item["cached_ratio"] is None else f"{item['cached_ratio']:.1%}",
             "—" if item["written_ratio"] is None else f"{item['written_ratio']:.1%}",
             human_tokens(item["fresh"]), human_tokens(item["output"]),
             f"${item['total_cost']:.2f}",
             "—" if item["mean_cost"] is None else f"${item['mean_cost']:.4f}",
             item["cache_health"])
            for label, item in summaries
        ])
        # ARA repeats are assessments, not independent papers. Aggregate them
        # here so the graph and lower table answer the useful paper-level cost
        # question while retaining the number of completed repetitions.
        per_paper: dict[str, dict[str, Any]] = {}
        for row in rows:
            paper_key = row["doi"] or row["title"] or str(row["run_id"])
            aggregate = per_paper.setdefault(paper_key, {
                "title": row["title"], "doi": row["doi"], "members": [],
                "cached": 0, "written": 0, "fresh": 0, "output": 0, "cost": 0.0,
            })
            aggregate["members"].append(row)
            for field in ("cached", "written", "fresh", "output"):
                aggregate[field] += row[field]
            aggregate["cost"] += row["cost"]
        paper_rows = list(per_paper.values())
        run_values = []
        run_keys = []
        for index, row in enumerate(paper_rows):
            key = str(index)
            self.ara_cost_rows[key] = row
            count = len(row["members"])
            run_values.append((short_title(row["title"]), row["doi"] or "—", str(count),
                               human_tokens(row["cached"]), human_tokens(row["written"]),
                               human_tokens(row["fresh"]), human_tokens(row["output"]),
                               f"${row['cost']:.4f}", f"${row['cost'] / count:.4f}"))
            run_keys.append(key)
        self._fill("ara-cost-runs-table", run_values, run_keys)
        costs = [row["cost"] / len(row["members"]) for row in paper_rows]
        graph.set_data(costs)
        detail.clear()
        detail.styles.display = "none"
        if costs:
            self.query_one("#ara-cost-graph-label", Label).update(
                f"Mean cost per paper over completed repetitions · {len(paper_rows)} papers · overall mean ${sum(costs) / len(costs):.4f} · range ${min(costs):.4f}–${max(costs):.4f}"
            )
        else:
            self.query_one("#ara-cost-graph-label", Label).update("No readable per-request usage in saved ARA artifacts")
        self.query_one("#ara-cost-status", Label).update(
            f"{self.ara_cost_experiment} · {model} · {len(rows)} artifact-backed assessments · line graph: mean cost per paper"
        )

    def _tree_state(self, tree: Tree[Any]) -> tuple[set[str], str | None]:
        expanded: set[str] = set()
        stack = list(tree.root.children)
        while stack:
            node = stack.pop()
            data = node.data or {}
            if node.is_expanded and data.get("key"):
                expanded.add(str(data["key"]))
            stack.extend(node.children)
        cursor_data = tree.cursor_node.data if tree.cursor_node and tree.cursor_node.data else {}
        return expanded, cursor_data.get("key")

    def _evidence_query(self, db: sqlite3.Connection) -> tuple[list[sqlite3.Row], bool]:
        where, params = self._where()
        clauses: list[str] = []
        if self.evidence_run:
            clauses.append("r.repetition_index=?")
            params.append(self.evidence_run)
        if self.evidence_mode == "matched":
            clauses.append("ev.match_status IN ('exact','normalized','word_sequence')")
        elif self.evidence_mode == "unresolved":
            clauses.append("ev.match_status IN ('ambiguous','not_found','pending')")
        elif self.evidence_mode == "disagreements":
            clauses.append("d.agreement_type IS NOT NULL AND d.agreement_type!='unanimous'")
        elif self.evidence_mode == "final":
            clauses.append("(d.final_label='unclear' OR r.label=d.final_label)")
        if self.evidence_search:
            pattern = f"%{self.evidence_search}%"
            clauses.append("(p.title LIKE ? OR ev.quote LIKE ? OR ev.explanation LIKE ? OR ev.supports LIKE ? OR ev.matched_pdf_text LIKE ? OR r.parsed_result_json LIKE ?)")
            params.extend([pattern] * 6)
        extra = "".join(f" AND {clause}" for clause in clauses)
        rows = db.execute(
            "SELECT ev.*,p.id paper_id,p.collection_year,p.title,r.id run_id,r.experiment_id,e.name experiment,r.repetition_index,r.label run_label,r.confidence,"
            "r.parsed_result_json,d.final_label,d.agreement_type,d.vote_counts_json,fa.relative_path,sr.dataset_path,"
            "(SELECT output_path FROM highlight_artifacts h WHERE h.applicability_run_id=r.id ORDER BY h.id DESC LIMIT 1) highlight_path,"
            "(SELECT COUNT(*) FROM applicability_evidence ev2 JOIN applicability_runs r2 ON r2.id=ev2.applicability_run_id WHERE r2.paper_id=p.id AND r2.experiment_id=r.experiment_id) evidence_total,"
            "(SELECT COUNT(*) FROM applicability_evidence ev2 JOIN applicability_runs r2 ON r2.id=ev2.applicability_run_id WHERE r2.paper_id=p.id AND r2.experiment_id=r.experiment_id AND ev2.match_status IN ('exact','normalized','word_sequence')) matched_total "
            "FROM applicability_evidence ev JOIN applicability_runs r ON r.id=ev.applicability_run_id "
            "JOIN papers p ON p.id=r.paper_id JOIN study_experiments e ON e.id=r.experiment_id "
            "LEFT JOIN applicability_decisions d ON d.experiment_id=r.experiment_id AND d.paper_id=r.paper_id AND d.content_version_id=r.content_version_id "
            "JOIN content_versions cv ON cv.id=r.content_version_id JOIN file_assets fa ON fa.id=cv.file_asset_id JOIN source_roots sr ON sr.id=fa.source_root_id "
            "WHERE 1=1" + where + extra + " ORDER BY p.collection_year,p.title,r.repetition_index,ev.ordinal LIMIT ?",
            [*params, EVIDENCE_LIMIT + 1],
        ).fetchall()
        return list(rows[:EVIDENCE_LIMIT]), len(rows) > EVIDENCE_LIMIT

    def _load_evidence(self, db: sqlite3.Connection) -> None:
        rows, truncated = self._evidence_query(db)
        tree = self.query_one("#evidence-tree", Tree)
        expanded, cursor_key = self._tree_state(tree)
        selected_key = cursor_key or self.selected_tree_key
        tree.root.remove_children()
        self.evidence_nodes = {}
        self.unresolved_nodes = []
        papers: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for raw in rows:
            row = dict(raw)
            row["source_path"] = str(Path(row["dataset_path"]) / row["relative_path"])
            row["result"] = parse_result(row["parsed_result_json"])
            row["run_numbers"] = [int(row["repetition_index"])]
            row["evidence_ids"] = [int(row["id"])]
            papers[(int(row["paper_id"]), int(row["experiment_id"]))].append(row)

        shown_nodes = matched = unresolved = 0
        for (paper_id, experiment_id), paper_rows in papers.items():
            first = paper_rows[0]
            paper_key = f"paper:{paper_id}:{experiment_id}"
            paper_label = Text()
            paper_label.append(f"{first['collection_year']}  ", style="bold #58a6ff")
            paper_label.append(first["title"] or "[untitled]", style="bold #f0f6fc")
            paper_label.append(f"  · {first['experiment']} · {first['final_label'] or 'incomplete'} · {first['agreement_type'] or 'pending'} · {first['matched_total']}/{first['evidence_total']} matched", style="#8b949e")
            paper_node = tree.root.add(paper_label, data={**first, "type": "paper", "key": paper_key}, expand=paper_key in expanded)
            self.evidence_nodes[paper_key] = paper_node

            run_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            if self.evidence_mode == "final":
                deduplicated: dict[tuple[str, str, Any], dict[str, Any]] = {}
                for row in paper_rows:
                    key = (" ".join((row["quote"] or "").lower().split()), row["supports"])
                    if key not in deduplicated:
                        deduplicated[key] = row
                    else:
                        existing = deduplicated[key]
                        existing["run_numbers"] = sorted(set(existing["run_numbers"] + row["run_numbers"]))
                        existing["evidence_ids"].extend(row["evidence_ids"])
                        if existing["match_status"] not in MATCHED_STATUSES and row["match_status"] in MATCHED_STATUSES:
                            preserved_runs = existing["run_numbers"]
                            preserved_ids = existing["evidence_ids"]
                            existing.update(row)
                            existing["run_numbers"] = preserved_runs
                            existing["evidence_ids"] = preserved_ids
                run_groups["final"] = list(deduplicated.values())
            else:
                for row in paper_rows:
                    run_groups[str(row["run_id"])].append(row)

            for run_key_value, evidence_rows in run_groups.items():
                run_first = evidence_rows[0]
                run_key = f"run:{paper_id}:{experiment_id}:{run_key_value}"
                if run_key_value == "final":
                    runs = sorted({number for item in evidence_rows for number in item["run_numbers"]})
                    run_label_text = f"Final evidence · Runs {', '.join(map(str, runs))} · deduplicated"
                else:
                    run_label_text = f"Run {run_first['repetition_index']} · {run_first['run_label']} · {run_first['confidence']} confidence"
                run_node = paper_node.add(Text(run_label_text, style="bold #a371f7"), data={**run_first, "type": "run", "key": run_key}, expand=run_key in expanded)
                self.evidence_nodes[run_key] = run_node
                for item in evidence_rows:
                    status = item["match_status"] or "pending"
                    symbol = "✓" if status in MATCHED_STATUSES else ("!" if status == "ambiguous" else "×")
                    style = "bold #3fb950" if status in MATCHED_STATUSES else ("bold #d29922" if status == "ambiguous" else "bold #f85149")
                    evidence_key = "evidence:" + ",".join(map(str, sorted(item["evidence_ids"])))
                    label = Text()
                    label.append(f"{symbol} {status}  ", style=style)
                    label.append(f"{item['supports']} · ", style="#58a6ff")
                    if len(item["run_numbers"]) > 1:
                        label.append(f"Runs {','.join(map(str, item['run_numbers']))} · ", style="#a371f7")
                    label.append(preview_text(item["quote"]))
                    node = run_node.add_leaf(label, data={**item, "type": "evidence", "key": evidence_key})
                    self.evidence_nodes[evidence_key] = node
                    shown_nodes += 1
                    if status in MATCHED_STATUSES:
                        matched += 1
                    else:
                        unresolved += 1
                        self.unresolved_nodes.append(node)

        tree.root.expand()
        target = self.evidence_nodes.get(selected_key or "")
        if target:
            parent = target.parent
            while parent and parent is not tree.root:
                parent.expand()
                parent = parent.parent
            tree.call_after_refresh(tree.move_cursor, target, animate=False)
            tree.call_after_refresh(tree.scroll_to_node, target, animate=False)
        elif tree.root.children:
            tree.move_cursor(tree.root.children[0], animate=False)
        suffix = " · first 2,000 shown" if truncated else ""
        search = f" · search “{self.evidence_search}”" if self.evidence_search else ""
        self.query_one("#evidence-status", Label).update(
            f"{len(papers)} papers · {shown_nodes} evidence nodes · {matched} matched · {unresolved} unresolved · {self.evidence_mode} view{search}{suffix}"
        )

    def _load_errors(self, db: sqlite3.Connection) -> None:
        where, params = self._where()
        rows = db.execute(
            "SELECT te.id,te.task_type,p.collection_year,p.title,te.error_type,te.error_text,te.completed_at "
            "FROM task_executions te LEFT JOIN papers p ON p.id=te.paper_id LEFT JOIN study_experiments e ON e.id=te.experiment_id "
            "WHERE te.status='failed'" + where + " ORDER BY te.id DESC LIMIT 500",
            params,
        ).fetchall()
        self._fill("errors-table", [(r["task_type"], str(r["collection_year"] or "—"), short_title(r["title"]), r["error_type"] or "Error", (r["error_text"] or "")[:80], r["completed_at"] or "—") for r in rows], [str(r["id"]) for r in rows])

    def _write_sections(self, widget_id: str, sections: list[tuple[str, list[tuple[str, Any]]]]) -> None:
        detail = self.query_one(f"#{widget_id}", RichLog)
        detail.clear()
        for section, fields in sections:
            heading = Text(section, style="bold #a371f7")
            detail.write(heading)
            for label, value in fields:
                line = Text()
                line.append(f"{label}: ", style="bold #58a6ff")
                line.append("—" if value is None or value == "" else str(value))
                detail.write(line)
            detail.write(Text(""))
        detail.scroll_home(animate=False)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._preview_table_row(event.data_table.id, str(event.row_key.value))

    def action_focus_paper_label(self) -> None:
        self.query_one(TabbedContent).active = "papers"
        self.query_one("#paper-label-filter", Select).focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._preview_table_row(event.data_table.id, str(event.row_key.value))

    def _preview_table_row(self, table_id: str | None, row_key: str) -> None:
        try:
            with closing(self._connect()) as db:
                if table_id == "ara-cost-runs-table":
                    self._show_ara_cost_run(row_key)
                elif table_id == "tasks-table":
                    self._show_task(db, row_key)
                elif table_id == "papers-table":
                    paper_id, experiment_id = map(int, row_key.split(":"))
                    self._show_paper(db, paper_id, experiment_id)
        except (sqlite3.Error, ValueError, json.JSONDecodeError) as exc:
            self.notify(str(exc), severity="error")

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        if event.control.id == "evidence-tree":
            self._preview_tree_node(event.node)

    def on_tree_node_selected(self, event: Tree.NodeSelected[Any]) -> None:
        if event.control.id == "evidence-tree":
            self._preview_tree_node(event.node)

    def _preview_tree_node(self, node: Any) -> None:
        data = node.data or {}
        if data.get("key"):
            self.selected_tree_key = str(data["key"])
        node_type = data.get("type")
        if node_type == "paper":
            self._show_evidence_paper(data)
        elif node_type == "run":
            self._show_evidence_run(data)
        elif node_type == "evidence":
            self._show_evidence(data)

    def _select_paths(self, source: str | None, highlight: str | None, quote: str | None = None) -> None:
        self.selected_source_path = source
        self.selected_highlight_path = highlight
        self.selected_quote = quote

    def _show_task(self, db: sqlite3.Connection, task_id: str) -> None:
        row = db.execute(
            "SELECT te.*,p.title,p.collection_year,r.repetition_index,r.label,r.confidence,r.parsed_result_json,"
            "CASE WHEN sr.dataset_path IS NULL THEN NULL ELSE sr.dataset_path || '/' || fa.relative_path END source_path,"
            "(SELECT output_path FROM highlight_artifacts h WHERE h.applicability_run_id=r.id ORDER BY h.id DESC LIMIT 1) highlight_path "
            "FROM task_executions te LEFT JOIN papers p ON p.id=te.paper_id LEFT JOIN applicability_runs r ON r.id=te.applicability_run_id "
            "LEFT JOIN content_versions cv ON cv.id=r.content_version_id LEFT JOIN file_assets fa ON fa.id=cv.file_asset_id "
            "LEFT JOIN source_roots sr ON sr.id=fa.source_root_id WHERE te.id=?",
            (task_id,),
        ).fetchone()
        if not row:
            return
        result = parse_result(row["parsed_result_json"])
        self._select_paths(row["source_path"], row["highlight_path"])
        self._write_sections("task-detail", [
            ("CLASSIFICATION", [("Paper", f"{row['title'] or '[no paper]'} ({row['collection_year'] or '—'})"), ("Run", row["repetition_index"]), ("Label", row["label"]), ("Confidence", row["confidence"]), ("Reason", result.get("reason"))]),
            ("STUDY", [("Study", result.get("study_description")), ("Data", result.get("data_description")), ("Analysis", result.get("analysis_description"))]),
            ("USAGE", [("Tokens", f"input {row['input_tokens'] or 0:,} | cached {row['cached_input_tokens'] or 0:,} | output {row['output_tokens'] or 0:,} | reasoning {row['reasoning_output_tokens'] or 0:,} | total {row['total_tokens'] or 0:,}"), ("Duration", human_time(row["total_duration_ms"])), ("Request", row["request_id"])]),
            ("FILES", [("Original PDF", row["source_path"]), ("Highlighted PDF", row["highlight_path"] or "Not generated yet")]),
        ])

    def _show_paper(self, db: sqlite3.Connection, paper_id: int, experiment_id: int) -> None:
        row = db.execute(
            "SELECT p.title,p.collection_year,d.final_label,d.agreement_type,d.vote_counts_json,d.evidence_status,"
            "fa.relative_path,sr.dataset_path,(SELECT output_path FROM highlight_artifacts h WHERE h.paper_id=p.id AND h.experiment_id=? ORDER BY h.id DESC LIMIT 1) highlight_path "
            "FROM papers p JOIN file_assets fa ON fa.paper_id=p.id JOIN source_roots sr ON sr.id=fa.source_root_id "
            "LEFT JOIN applicability_decisions d ON d.paper_id=p.id AND d.experiment_id=? WHERE p.id=? ORDER BY d.id DESC LIMIT 1",
            (experiment_id, experiment_id, paper_id),
        ).fetchone()
        if not row:
            return
        source = str(Path(row["dataset_path"]) / row["relative_path"])
        runs = db.execute("SELECT repetition_index,label,confidence,parsed_result_json FROM applicability_runs WHERE paper_id=? AND experiment_id=? AND status='succeeded' ORDER BY repetition_index", (paper_id, experiment_id)).fetchall()
        self._select_paths(source, row["highlight_path"])
        run_fields: list[tuple[str, Any]] = []
        for run in runs:
            result = parse_result(run["parsed_result_json"])
            run_fields.extend([(f"Run {run['repetition_index']}", f"{run['label']} · {run['confidence']}"), (f"Run {run['repetition_index']} reason", result.get("reason"))])
        self._write_sections("paper-detail", [
            ("FINAL DECISION", [("Paper", f"{row['title']} ({row['collection_year']})"), ("Final label", row["final_label"]), ("Agreement", row["agreement_type"]), ("Votes", row["vote_counts_json"]), ("Evidence", row["evidence_status"])]),
            ("RUNS", run_fields),
            ("FILES", [("Original PDF", source), ("Highlighted PDF", row["highlight_path"] or "Not generated yet")]),
        ])

    def _show_ara_cost_run(self, row_key: str) -> None:
        row = self.ara_cost_rows.get(row_key)
        if not row:
            return
        count = len(row["members"])
        detail = self.query_one("#ara-cost-detail", RichLog)
        detail.styles.display = "block"
        self._write_sections("ara-cost-detail", [
            ("ARA PAPER COST", [
                ("Paper", row["title"]), ("DOI", row["doi"]),
                ("Completed repetitions", count),
                ("Total estimated cost", f"${row['cost']:.6f}"),
                ("Mean cost per repetition", f"${row['cost'] / count:.6f}"),
            ]),
            ("TOKEN BILLING", [
                ("Cached input", human_tokens(row["cached"])),
                ("Cache-write input", human_tokens(row["written"])),
                ("Fresh input", human_tokens(row["fresh"])),
                ("Output", human_tokens(row["output"])),
            ]),
            ("ARTIFACTS", [("Example raw ARA artifact", row["members"][-1]["artifact_path"])]),
        ])

    def _show_evidence_paper(self, data: dict[str, Any]) -> None:
        self._select_paths(data.get("source_path"), data.get("highlight_path"))
        self._write_sections("evidence-detail", [
            ("PAPER", [("Breadcrumb", f"{data.get('collection_year')} › {data.get('title')}"), ("Final label", data.get("final_label")), ("Agreement", data.get("agreement_type")), ("Votes", data.get("vote_counts_json")), ("Evidence", f"{data.get('matched_total')}/{data.get('evidence_total')} matched")]),
            ("FILES", [("Original PDF", data.get("source_path")), ("Highlighted PDF", data.get("highlight_path") or "Not generated yet")]),
            ("NAVIGATION", [("Next", "Expand a run, then move through evidence with ↑/↓. Press ? for shortcuts.")]),
        ])

    def _show_evidence_run(self, data: dict[str, Any]) -> None:
        result = data.get("result") or {}
        self._select_paths(data.get("source_path"), data.get("highlight_path"))
        self._write_sections("evidence-detail", [
            ("CLASSIFICATION", [("Breadcrumb", f"{data.get('collection_year')} › {data.get('title')} › Run {data.get('repetition_index')}"), ("Run label", data.get("run_label")), ("Confidence", data.get("confidence")), ("Reason", result.get("reason"))]),
            ("STUDY", [("Study", result.get("study_description")), ("Data", result.get("data_description")), ("Analysis", result.get("analysis_description"))]),
            ("FILES", [("Original PDF", data.get("source_path")), ("Highlighted PDF", data.get("highlight_path") or "Not generated yet")]),
        ])

    def _show_evidence(self, data: dict[str, Any]) -> None:
        status = data.get("match_status") or "pending"
        if status in MATCHED_STATUSES:
            match_summary = f"✓ {status.upper()} — page {data.get('pdf_page_number')}"
        elif status == "ambiguous":
            match_summary = "! AMBIGUOUS — multiple possible PDF locations"
        elif status == "not_found":
            match_summary = "× NOT FOUND — no safe PDF text location"
        else:
            match_summary = "○ PENDING — run `highlight run` to resolve PDF locations"
        runs = ", ".join(map(str, data.get("run_numbers") or [data.get("repetition_index")]))
        result = data.get("result") or {}
        self._select_paths(data.get("source_path"), data.get("highlight_path"), data.get("quote"))
        self._write_sections("evidence-detail", [
            ("CLASSIFICATION", [("Breadcrumb", f"{data.get('collection_year')} › {data.get('title')} › Run(s) {runs} › {data.get('supports')}"), ("Run label", data.get("run_label")), ("Confidence", data.get("confidence")), ("Overall run reason", result.get("reason"))]),
            ("EVIDENCE", [("Type", data.get("supports")), ("Full quotation", data.get("quote")), ("Model explanation", data.get("explanation")), ("Contributing runs", runs)]),
            ("PDF MATCH", [("Status", match_summary), ("Score", data.get("match_score")), ("Matched PDF text", data.get("matched_pdf_text")), ("Match error", data.get("match_error"))]),
            ("FILES", [("Original PDF", data.get("source_path")), ("Highlighted PDF", data.get("highlight_path") or "Not generated yet"), ("Actions", "o highlighted · Shift+O source · c quote · Shift+C path")]),
        ])

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "evidence-search":
            return
        self.evidence_search = event.value.strip()
        self._reload_evidence()

    def on_select_changed(self, event: Select.Changed) -> None:
        if not self.is_mounted:
            return
        if event.select.id == "ara-cost-experiment" and isinstance(event.value, str):
            # Selecting an experiment is intentionally non-destructive and
            # non-refreshing. The user commits the change with `r`, avoiding
            # an artifact read or table redraw while navigating the dashboard.
            self.ara_cost_experiment = event.value
            self.last_ara_cost_signature = None
            self.query_one("#ara-cost-status", Label).update(
                f"Selected {event.value}; press r to load its cost data"
            )
        elif event.select.id == "paper-label-filter" and isinstance(event.value, str):
            self.paper_label_filter = event.value
            try:
                with closing(self._connect()) as db:
                    self._load_papers(db)
            except (sqlite3.Error, FileNotFoundError) as exc:
                self.notify(str(exc), severity="error")
        elif event.select.id == "evidence-mode" and isinstance(event.value, str):
            self.evidence_mode = event.value
            self._reload_evidence()
        elif event.select.id == "evidence-run" and isinstance(event.value, int):
            self.evidence_run = event.value
            self._reload_evidence()

    def _reload_evidence(self) -> None:
        if not self.is_mounted:
            return
        try:
            with closing(self._connect()) as db:
                self._load_evidence(db)
        except (sqlite3.Error, FileNotFoundError) as exc:
            self.notify(str(exc), severity="error")

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id
        if tab_id == "evidence":
            self.query_one("#evidence-tree", Tree).focus()

    def action_focus_search(self) -> None:
        self.query_one(TabbedContent).active = "evidence"
        self.query_one("#evidence-search", Input).focus()

    def action_clear_search(self) -> None:
        search = self.query_one("#evidence-search", Input)
        if search.value:
            search.value = ""
        else:
            self.query_one("#evidence-tree", Tree).focus()

    def action_evidence_mode(self, mode: str) -> None:
        self.query_one(TabbedContent).active = "evidence"
        self.query_one("#evidence-mode", Select).value = mode

    def action_run_filter(self, repetition: int) -> None:
        self.query_one(TabbedContent).active = "evidence"
        self.query_one("#evidence-run", Select).value = repetition

    def action_next_unresolved(self, direction: int) -> None:
        self.query_one(TabbedContent).active = "evidence"
        if not self.unresolved_nodes:
            self.notify("No unresolved evidence in the current view", severity="information")
            return
        tree = self.query_one("#evidence-tree", Tree)
        current = tree.cursor_node
        try:
            index = self.unresolved_nodes.index(current)
            target = self.unresolved_nodes[(index + direction) % len(self.unresolved_nodes)]
        except ValueError:
            target = self.unresolved_nodes[0 if direction > 0 else -1]
        parent = target.parent
        while parent and parent is not tree.root:
            parent.expand()
            parent = parent.parent
        tree.move_cursor(target, animate=False)
        tree.scroll_to_node(target, animate=False)
        tree.focus()
        self._preview_tree_node(target)

    def _open_path(self, value: str | None, description: str) -> None:
        if not value:
            self.notify(f"{description} is not available", severity="warning")
            return
        path = Path(value)
        if not path.is_file():
            self.notify(f"{description} does not exist: {path}", severity="error", timeout=8)
            return
        try:
            launch_file(path)
            self.notify(f"Opened {description}: {path.name}")
        except OSError as exc:
            self.notify(f"Could not open {description}: {exc}", severity="error", timeout=8)

    def action_open_highlight(self) -> None:
        self._open_path(self.selected_highlight_path, "Highlighted PDF")

    def action_open_source(self) -> None:
        self._open_path(self.selected_source_path, "Original PDF")

    def _copy(self, value: str | None, description: str) -> None:
        if not value:
            self.notify(f"No {description} is selected", severity="warning")
            return
        try:
            self.copy_to_clipboard(value)
            self.notify(f"Copied {description}")
        except Exception as exc:
            self.notify(f"Clipboard unavailable; value remains visible in details ({exc})", severity="warning", timeout=8)

    def action_copy_quote(self) -> None:
        self._copy(self.selected_quote, "evidence quotation")

    def action_copy_path(self) -> None:
        self._copy(self.selected_highlight_path or self.selected_source_path, "PDF path")

    def action_help(self) -> None:
        self.push_screen(HelpScreen())


async def run_dashboard(
    database: Path,
    experiment: str | None,
    year: int | None,
    status: str | None,
) -> None:
    await ScreeningDashboard(database, experiment, year, status).run_async()
