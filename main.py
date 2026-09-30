"""Build and query a reproducible SQLite catalog for the LAK PDF corpus."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from io import StringIO
from html import unescape
import os
import re
import sqlite3
import unicodedata
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from urllib.parse import unquote

import aiosqlite
import bibtexparser
import structlog
from rich.console import Console
from rich.highlighter import JSONHighlighter
from liteparse import LiteParse
from tqdm import tqdm

APP_NAME = "lakdb"
DEFAULT_DATABASE = Path("data/lak.sqlite")
DEFAULT_LOG = Path("logs/lakdb.jsonl")
PAPER_KINDS = ("long", "short")
CONSOLE_JSON_RENDERER = structlog.processors.JSONRenderer(
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
)
CONSOLE_JSON_HIGHLIGHTER = JSONHighlighter()


SCHEMA = """
CREATE TABLE IF NOT EXISTS source_roots (
    id INTEGER PRIMARY KEY,
    dataset_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    source_identity TEXT NOT NULL UNIQUE,
    canonical_doi TEXT,
    title TEXT,
    abstract TEXT,
    publication_year INTEGER,
    collection_year INTEGER,
    venue TEXT,
    publisher TEXT,
    bibtex_type TEXT,
    canonical_metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS papers_canonical_doi_unique
    ON papers(canonical_doi) WHERE canonical_doi IS NOT NULL;

CREATE TABLE IF NOT EXISTS bibtex_records (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE RESTRICT,
    source_root_id INTEGER NOT NULL REFERENCES source_roots(id) ON DELETE CASCADE,
    relative_bib_path TEXT NOT NULL,
    citation_key TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    raw_bibtex TEXT NOT NULL,
    fields_json TEXT NOT NULL,
    entry_sha256 TEXT NOT NULL,
    parse_status TEXT NOT NULL CHECK(parse_status IN ('parsed', 'error')),
    parse_error TEXT,
    imported_at TEXT NOT NULL,
    UNIQUE(source_root_id, relative_bib_path, citation_key)
) STRICT;

CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE
) STRICT;

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES authors(id) ON DELETE RESTRICT,
    author_position INTEGER NOT NULL CHECK(author_position >= 0),
    role TEXT NOT NULL DEFAULT 'author',
    PRIMARY KEY(paper_id, author_id, role)
) STRICT;

CREATE TABLE IF NOT EXISTS file_assets (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE SET NULL,
    source_root_id INTEGER NOT NULL REFERENCES source_roots(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    dataset_year INTEGER NOT NULL,
    paper_kind TEXT NOT NULL CHECK(paper_kind IN ('long', 'short')),
    mime_type TEXT NOT NULL DEFAULT 'application/pdf',
    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
    sha256 TEXT NOT NULL,
    page_count INTEGER,
    is_primary_pdf INTEGER NOT NULL DEFAULT 1 CHECK(is_primary_pdf IN (0, 1)),
    discovered_at TEXT NOT NULL,
    last_verified_at TEXT NOT NULL,
    UNIQUE(source_root_id, relative_path)
) STRICT;
CREATE INDEX IF NOT EXISTS file_assets_year_kind ON file_assets(dataset_year, paper_kind);
CREATE INDEX IF NOT EXISTS file_assets_paper ON file_assets(paper_id);
CREATE INDEX IF NOT EXISTS file_assets_sha256 ON file_assets(sha256);

CREATE TABLE IF NOT EXISTS parse_runs (
    id INTEGER PRIMARY KEY,
    file_asset_id INTEGER NOT NULL REFERENCES file_assets(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'succeeded', 'failed')),
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    source_pdf_sha256 TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    error_text TEXT
) STRICT;
CREATE INDEX IF NOT EXISTS parse_runs_file_status ON parse_runs(file_asset_id, status);

CREATE TABLE IF NOT EXISTS content_versions (
    id INTEGER PRIMARY KEY,
    file_asset_id INTEGER NOT NULL REFERENCES file_assets(id) ON DELETE CASCADE,
    parse_run_id INTEGER NOT NULL REFERENCES parse_runs(id) ON DELETE RESTRICT,
    markdown TEXT NOT NULL,
    markdown_sha256 TEXT NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    quality_status TEXT NOT NULL CHECK(quality_status IN ('accepted', 'review_required', 'rejected')),
    quality_metrics_json TEXT NOT NULL,
    is_current INTEGER NOT NULL CHECK(is_current IN (0, 1)),
    created_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS content_versions_current ON content_versions(file_asset_id, is_current);

CREATE TABLE IF NOT EXISTS ingest_issues (
    id INTEGER PRIMARY KEY,
    source_root_id INTEGER NOT NULL REFERENCES source_roots(id) ON DELETE CASCADE,
    severity TEXT NOT NULL CHECK(severity IN ('warning', 'error')),
    issue_type TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    details_json TEXT NOT NULL,
    resolved_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_root_id, issue_type, relative_path)
) STRICT;

CREATE VIRTUAL TABLE IF NOT EXISTS paper_search USING fts5(
    paper_id UNINDEXED,
    title,
    abstract,
    authors,
    markdown,
    tokenize='unicode61'
);
"""


@dataclass(frozen=True)
class Partition:
    year: int
    paper_kind: str
    path: Path
    bib_path: Path
    pdf_paths: tuple[Path, ...]


@dataclass(frozen=True)
class BibEntry:
    partition: Partition
    citation_key: str
    entry_type: str
    fields: dict[str, str]
    raw_bibtex: str


@dataclass(frozen=True)
class ParseJob:
    file_asset_id: int
    paper_id: int
    path: Path
    source_sha256: str
    title: str | None


@dataclass(frozen=True)
class ParseOutcome:
    job: ParseJob
    started_at: str
    finished_at: str
    success: bool
    markdown: str
    page_count: int | None
    warnings: list[str]
    error: str | None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = unquote(value).strip().lower()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    doi = doi.rstrip(".,;:)}]>")
    return doi or None


def normalize_text(value: str | None) -> str:
    """Return a conservative filename/title comparison key."""
    if not value:
        return ""
    value = unescape(value.replace(r"\&", "&"))
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "")
    value = (
        unicodedata.normalize("NFKD", unquote(value)).encode("ascii", "ignore").decode()
    )
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def filename_doi_key(value: str | None) -> str:
    return normalize_text(normalize_doi(value))


def split_authors(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        name.strip().strip("{}")
        for name in re.split(r"\s+and\s+", value)
        if name.strip()
    ]


def raw_entry(entry_type: str, key: str, fields: dict[str, str]) -> str:
    """Fallback only: raw source is normally retained by extract_raw_bib_entries."""
    lines = [f"@{entry_type}{{{key},"]
    lines.extend(f"  {name} = {{{value}}}," for name, value in fields.items())
    lines.append("}")
    return "\n".join(lines)


def extract_raw_bib_entries(source: str) -> dict[str, str]:
    """Extract complete source entries without normalising whitespace or field values."""
    entries: dict[str, str] = {}
    for match in re.finditer(r"@\s*[A-Za-z]+\s*([\{(])\s*([^,\s]+)\s*[,]", source):
        opening = match.group(1)
        closing = "}" if opening == "{" else ")"
        depth = 0
        quoted = False
        escaped = False
        end = None
        for index in range(match.start(), len(source)):
            character = source[index]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"':
                quoted = not quoted
                continue
            if quoted:
                continue
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is not None:
            entries[match.group(2)] = source[match.start() : end]
    return entries


def discover_dataset(root: Path) -> list[Partition]:
    root = root.expanduser().resolve()
    partitions: list[Partition] = []
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    for year_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if not year_dir.is_dir() or not re.fullmatch(r"\d{4}", year_dir.name):
            continue
        for paper_kind in PAPER_KINDS:
            folder = year_dir / paper_kind
            if not folder.is_dir():
                continue
            pdfs = tuple(
                sorted(
                    (
                        item
                        for item in folder.rglob("*")
                        if item.is_file() and item.suffix.lower() == ".pdf"
                    ),
                    key=lambda item: item.as_posix(),
                )
            )
            partitions.append(
                Partition(
                    int(year_dir.name), paper_kind, folder, folder / "acm.bib", pdfs
                )
            )
    return partitions


def parse_bibtex_file(partition: Partition) -> tuple[list[BibEntry], str | None]:
    if not partition.bib_path.is_file():
        return [], f"Missing {partition.bib_path.name}"
    try:
        source = partition.bib_path.read_text(encoding="utf-8")
        raw_entries = extract_raw_bib_entries(source)
        database = bibtexparser.loads(source)
    except Exception as exc:  # bibtexparser errors vary by release
        return [], f"{type(exc).__name__}: {exc}"

    entries: list[BibEntry] = []
    for parsed in database.entries:
        key = str(parsed.get("ID", "")).strip()
        entry_type = str(parsed.get("ENTRYTYPE", "misc")).strip().lower()
        if not key:
            continue
        fields = {
            str(name).lower(): str(value)
            for name, value in parsed.items()
            if name not in {"ID", "ENTRYTYPE"}
        }
        entries.append(
            BibEntry(
                partition,
                key,
                entry_type,
                fields,
                raw_entries.get(key, raw_entry(entry_type, key, fields)),
            )
        )
    return entries, None


def entry_candidates(pdf_path: Path, entries: Iterable[BibEntry]) -> list[BibEntry]:
    """Find exact, explainable candidates; do not use fuzzy matching."""
    stem = pdf_path.stem
    normalized_stem = normalize_text(stem)
    candidates: list[BibEntry] = []
    for entry in entries:
        declared_file = entry.fields.get("file", "")
        if (
            declared_file
            and normalize_text(Path(declared_file).stem) == normalized_stem
        ):
            candidates.append(entry)
            continue
        doi_key = filename_doi_key(entry.fields.get("doi") or entry.citation_key)
        if doi_key and doi_key == normalized_stem:
            candidates.append(entry)
            continue
        title_key = normalize_text(entry.fields.get("title"))
        # Some LAK folders truncate a title-based filename. A long exact prefix is
        # still deterministic; a collision remains an explicit ambiguous match.
        if title_key and (
            title_key == normalized_stem
            or (len(normalized_stem) >= 32 and title_key.startswith(normalized_stem))
        ):
            candidates.append(entry)
    return candidates


def liteparse_version() -> str:
    try:
        from importlib.metadata import version

        return version("liteparse")
    except Exception:
        return "unknown"


def parser_config(ocr_workers: int) -> dict[str, Any]:
    """One quality profile for every PDF; no classifier or routing pass."""
    return {
        "output_format": "markdown",
        "ocr_enabled": True,
        "ocr_language": "eng",
        "dpi": 300,
        "num_workers": ocr_workers,
        "continue_on_page_error": True,
        "preserve_very_small_text": True,
        "extract_links": True,
        "keep_headers_footers": False,
    }


def parse_pdf_sync(job: ParseJob, config: dict[str, Any]) -> ParseOutcome:
    started_at = utc_now()
    try:
        parser = LiteParse(quiet=True, **config)
        result = parser.parse(job.path)
        markdown = result.text or ""
        warnings: list[str] = []
        for error in result.page_errors or []:
            warnings.append(str(error))
        return ParseOutcome(
            job=job,
            started_at=started_at,
            finished_at=utc_now(),
            success=True,
            markdown=markdown,
            page_count=getattr(result, "total_pages", None),
            warnings=warnings,
            error=None,
        )
    except Exception as exc:
        return ParseOutcome(
            job,
            started_at,
            utc_now(),
            False,
            "",
            None,
            [],
            f"{type(exc).__name__}: {exc}",
        )


def render_console_json(_: Any, __: str, event_dict: dict[str, Any]) -> str:
    """Render structured events as indented, syntax-highlighted terminal JSON."""
    output = StringIO()
    Console(file=output, force_terminal=True, color_system="truecolor").print(
        CONSOLE_JSON_HIGHLIGHTER(CONSOLE_JSON_RENDERER(_, __, event_dict)),
        end="",
        soft_wrap=True,
    )
    return output.getvalue()


def configure_logging(log_file: Path, verbose: bool) -> None:
    """Send coloured, indented JSON events to the terminal and JSONL to a rotating file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=render_console_json,
            foreign_pre_chain=pre_chain,
        )
    )
    json_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    json_handler.setLevel(level)
    json_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=pre_chain,
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(json_handler)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *pre_chain,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class Catalog:
    def __init__(self, database_path: Path):
        self.database_path = database_path.expanduser()
        self.connection: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "Catalog":
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = await aiosqlite.connect(self.database_path)
        self.connection.row_factory = aiosqlite.Row
        await self.connection.execute("PRAGMA foreign_keys = ON")
        await self.connection.execute("PRAGMA journal_mode = WAL")
        await self.connection.execute("PRAGMA busy_timeout = 5000")
        await self.connection.executescript(SCHEMA)
        await self.connection.commit()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.connection:
            await self.connection.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self.connection is None:
            raise RuntimeError("Catalog is not open")
        return self.connection

    async def source_root_id(self, root: Path) -> int:
        now = utc_now()
        resolved = str(root.expanduser().resolve())
        await self.db.execute(
            "INSERT INTO source_roots(dataset_path, created_at, last_seen_at) VALUES (?, ?, ?) "
            "ON CONFLICT(dataset_path) DO UPDATE SET last_seen_at = excluded.last_seen_at",
            (resolved, now, now),
        )
        cursor = await self.db.execute(
            "SELECT id FROM source_roots WHERE dataset_path = ?", (resolved,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return int(row["id"])

    async def record_issue(
        self,
        source_root_id: int,
        severity: str,
        issue_type: str,
        relative_path: str,
        details: dict[str, Any],
    ) -> None:
        now = utc_now()
        await self.db.execute(
            "INSERT INTO ingest_issues(source_root_id, severity, issue_type, relative_path, details_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_root_id, issue_type, relative_path) DO UPDATE SET "
            "severity=excluded.severity, details_json=excluded.details_json, updated_at=excluded.updated_at, resolved_at=NULL",
            (
                source_root_id,
                severity,
                issue_type,
                relative_path,
                json_text(details),
                now,
                now,
            ),
        )

    async def upsert_bib_entry(
        self, source_root_id: int, entry: BibEntry, root: Path
    ) -> int:
        now = utc_now()
        bib_relative = entry.partition.bib_path.relative_to(root).as_posix()
        source_identity = f"{bib_relative}#{entry.citation_key}"
        doi = normalize_doi(entry.fields.get("doi"))
        row = None
        if doi:
            cursor = await self.db.execute(
                "SELECT id FROM papers WHERE canonical_doi = ?", (doi,)
            )
            row = await cursor.fetchone()
        if row is None:
            cursor = await self.db.execute(
                "SELECT id FROM papers WHERE source_identity = ?", (source_identity,)
            )
            row = await cursor.fetchone()
        title = entry.fields.get("title")
        abstract = entry.fields.get("abstract")
        publication_year = parse_year(entry.fields.get("year"))
        canonical = json_text(entry.fields)
        if row is None:
            cursor = await self.db.execute(
                "INSERT INTO papers(source_identity, canonical_doi, title, abstract, publication_year, collection_year, venue, publisher, bibtex_type, canonical_metadata_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_identity,
                    doi,
                    title,
                    abstract,
                    publication_year,
                    entry.partition.year,
                    entry.fields.get("booktitle") or entry.fields.get("journal"),
                    entry.fields.get("publisher"),
                    entry.entry_type,
                    canonical,
                    now,
                    now,
                ),
            )
            paper_id = int(cursor.lastrowid)
        else:
            paper_id = int(row["id"])
            await self.db.execute(
                "UPDATE papers SET canonical_doi=COALESCE(?, canonical_doi), title=?, abstract=?, publication_year=?, "
                "collection_year=?, venue=?, publisher=?, bibtex_type=?, canonical_metadata_json=?, updated_at=? WHERE id=?",
                (
                    doi,
                    title,
                    abstract,
                    publication_year,
                    entry.partition.year,
                    entry.fields.get("booktitle") or entry.fields.get("journal"),
                    entry.fields.get("publisher"),
                    entry.entry_type,
                    canonical,
                    now,
                    paper_id,
                ),
            )
        await self.db.execute(
            "INSERT INTO bibtex_records(paper_id, source_root_id, relative_bib_path, citation_key, entry_type, raw_bibtex, fields_json, entry_sha256, parse_status, parse_error, imported_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'parsed', NULL, ?) "
            "ON CONFLICT(source_root_id, relative_bib_path, citation_key) DO UPDATE SET "
            "paper_id=excluded.paper_id, entry_type=excluded.entry_type, raw_bibtex=excluded.raw_bibtex, fields_json=excluded.fields_json, entry_sha256=excluded.entry_sha256, parse_status='parsed', parse_error=NULL, imported_at=excluded.imported_at",
            (
                paper_id,
                source_root_id,
                bib_relative,
                entry.citation_key,
                entry.entry_type,
                entry.raw_bibtex,
                canonical,
                sha256_text(entry.raw_bibtex),
                now,
            ),
        )
        await self.db.execute(
            "DELETE FROM paper_authors WHERE paper_id = ? AND role = 'author'",
            (paper_id,),
        )
        for position, name in enumerate(split_authors(entry.fields.get("author"))):
            normalized_name = normalize_text(name)
            if not normalized_name:
                continue
            await self.db.execute(
                "INSERT INTO authors(display_name, normalized_name) VALUES (?, ?) "
                "ON CONFLICT(normalized_name) DO UPDATE SET display_name=excluded.display_name",
                (name, normalized_name),
            )
            cursor = await self.db.execute(
                "SELECT id FROM authors WHERE normalized_name = ?", (normalized_name,)
            )
            author = await cursor.fetchone()
            assert author is not None
            await self.db.execute(
                "INSERT OR REPLACE INTO paper_authors(paper_id, author_id, author_position, role) VALUES (?, ?, ?, 'author')",
                (paper_id, int(author["id"]), position),
            )
        return paper_id

    async def upsert_pdf(
        self,
        source_root_id: int,
        root: Path,
        partition: Partition,
        pdf_path: Path,
        paper_id: int | None,
    ) -> None:
        now = utc_now()
        relative = pdf_path.relative_to(root).as_posix()
        digest = await asyncio.to_thread(sha256_file, pdf_path)
        size = pdf_path.stat().st_size
        await self.db.execute(
            "INSERT INTO file_assets(paper_id, source_root_id, relative_path, dataset_year, paper_kind, byte_size, sha256, discovered_at, last_verified_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_root_id, relative_path) DO UPDATE SET paper_id=excluded.paper_id, dataset_year=excluded.dataset_year, "
            "paper_kind=excluded.paper_kind, byte_size=excluded.byte_size, sha256=excluded.sha256, last_verified_at=excluded.last_verified_at",
            (
                paper_id,
                source_root_id,
                relative,
                partition.year,
                partition.paper_kind,
                size,
                digest,
                now,
                now,
            ),
        )

    async def pending_parse_jobs(
        self, limit: int | None, force: bool
    ) -> list[ParseJob]:
        where = "fa.paper_id IS NOT NULL"
        if not force:
            where += " AND NOT EXISTS (SELECT 1 FROM content_versions cv JOIN parse_runs pr ON pr.id=cv.parse_run_id WHERE cv.file_asset_id=fa.id AND cv.is_current=1 AND cv.quality_status='accepted' AND pr.source_pdf_sha256=fa.sha256)"
        sql = (
            "SELECT fa.id AS file_asset_id, fa.paper_id, fa.relative_path, fa.sha256, sr.dataset_path, p.title "
            "FROM file_assets fa JOIN source_roots sr ON sr.id=fa.source_root_id JOIN papers p ON p.id=fa.paper_id "
            f"WHERE {where} ORDER BY fa.dataset_year, fa.paper_kind, fa.relative_path"
        )
        if limit is not None:
            sql += " LIMIT ?"
            cursor = await self.db.execute(sql, (limit,))
        else:
            cursor = await self.db.execute(sql)
        rows = await cursor.fetchall()
        return [
            ParseJob(
                int(row["file_asset_id"]),
                int(row["paper_id"]),
                Path(row["dataset_path"]) / row["relative_path"],
                row["sha256"],
                row["title"],
            )
            for row in rows
        ]

    async def record_parse(self, outcome: ParseOutcome, config: dict[str, Any]) -> None:
        actual_hash = (
            await asyncio.to_thread(sha256_file, outcome.job.path)
            if outcome.job.path.is_file()
            else outcome.job.source_sha256
        )
        parser_name = "liteparse"
        version = liteparse_version()
        await self.db.execute("BEGIN")
        try:
            cursor = await self.db.execute(
                "INSERT INTO parse_runs(file_asset_id, status, parser_name, parser_version, config_json, source_pdf_sha256, started_at, finished_at, warnings_json, error_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    outcome.job.file_asset_id,
                    "succeeded" if outcome.success else "failed",
                    parser_name,
                    version,
                    json_text(config),
                    actual_hash,
                    outcome.started_at,
                    outcome.finished_at,
                    json_text(outcome.warnings),
                    outcome.error,
                ),
            )
            parse_run_id = int(cursor.lastrowid)
            if outcome.success:
                non_empty = len(outcome.markdown.strip())
                quality_status = (
                    "accepted"
                    if non_empty >= 20 and (outcome.page_count or 0) > 0
                    else "review_required"
                )
                metrics = {
                    "characters": non_empty,
                    "page_count": outcome.page_count,
                    "warning_count": len(outcome.warnings),
                }
                await self.db.execute(
                    "UPDATE content_versions SET is_current = 0 WHERE file_asset_id = ?",
                    (outcome.job.file_asset_id,),
                )
                await self.db.execute(
                    "INSERT INTO content_versions(file_asset_id, parse_run_id, markdown, markdown_sha256, parser_name, parser_version, config_json, quality_status, quality_metrics_json, is_current, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                    (
                        outcome.job.file_asset_id,
                        parse_run_id,
                        outcome.markdown,
                        sha256_text(outcome.markdown),
                        parser_name,
                        version,
                        json_text(config),
                        quality_status,
                        json_text(metrics),
                        utc_now(),
                    ),
                )
                await self.db.execute(
                    "UPDATE file_assets SET sha256=?, page_count=?, last_verified_at=? WHERE id=?",
                    (
                        actual_hash,
                        outcome.page_count,
                        utc_now(),
                        outcome.job.file_asset_id,
                    ),
                )
                await self.refresh_search(outcome.job.paper_id)
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            raise

    async def refresh_search(self, paper_id: int) -> None:
        cursor = await self.db.execute(
            "SELECT p.title, p.abstract, COALESCE(group_concat(a.display_name, '; '), '') AS authors "
            "FROM papers p LEFT JOIN paper_authors pa ON pa.paper_id=p.id LEFT JOIN authors a ON a.id=pa.author_id "
            "WHERE p.id=? GROUP BY p.id",
            (paper_id,),
        )
        paper = await cursor.fetchone()
        cursor = await self.db.execute(
            "SELECT cv.markdown FROM content_versions cv JOIN file_assets fa ON fa.id=cv.file_asset_id "
            "WHERE fa.paper_id=? AND cv.is_current=1 AND cv.quality_status='accepted' ORDER BY cv.id DESC LIMIT 1",
            (paper_id,),
        )
        content = await cursor.fetchone()
        await self.db.execute(
            "DELETE FROM paper_search WHERE paper_id = ?", (str(paper_id),)
        )
        if paper and content:
            await self.db.execute(
                "INSERT INTO paper_search(paper_id, title, abstract, authors, markdown) VALUES (?, ?, ?, ?, ?)",
                (
                    str(paper_id),
                    paper["title"] or "",
                    paper["abstract"] or "",
                    paper["authors"] or "",
                    content["markdown"],
                ),
            )


def parse_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


async def command_inventory(args: SimpleNamespace, logger: Any) -> int:
    partitions = await asyncio.to_thread(discover_dataset, Path(args.dataset_root))
    report: list[dict[str, Any]] = []
    for partition in partitions:
        entries, error = await asyncio.to_thread(parse_bibtex_file, partition)
        report.append(
            {
                "year": partition.year,
                "type": partition.paper_kind,
                "pdfs": len(partition.pdf_paths),
                "bib_entries": len(entries),
                "bib_error": error,
            }
        )
    logger.info(
        "inventory_complete",
        dataset_root=str(Path(args.dataset_root).resolve()),
        partitions=report,
        pdf_count=sum(item["pdfs"] for item in report),
        bib_entry_count=sum(item["bib_entries"] for item in report),
    )
    return 0


async def command_ingest(args: SimpleNamespace, logger: Any) -> int:
    root = Path(args.dataset_root).expanduser().resolve()
    partitions = await asyncio.to_thread(discover_dataset, root)
    async with Catalog(Path(args.database)) as catalog:
        source_root_id = await catalog.source_root_id(root)
        await catalog.db.execute(
            "DELETE FROM ingest_issues WHERE source_root_id=? AND issue_type IN ('missing_bibtex', 'invalid_bibtex', 'unmatched_pdf', 'ambiguous_pdf_match')",
            (source_root_id,),
        )
        entry_total = pdf_total = matched_total = issue_total = 0
        with tqdm(
            total=sum(len(partition.pdf_paths) + 1 for partition in partitions),
            desc="Ingesting",
            unit="source",
            disable=args.no_progress,
        ) as progress:
            for partition in partitions:
                entries, error = await asyncio.to_thread(parse_bibtex_file, partition)
                progress.update(1)
                if error:
                    await catalog.record_issue(
                        source_root_id,
                        "error",
                        "missing_bibtex"
                        if not partition.bib_path.exists()
                        else "invalid_bibtex",
                        partition.bib_path.relative_to(root).as_posix(),
                        {"error": error},
                    )
                    issue_total += 1
                entry_paper_ids: dict[str, int] = {}
                for entry in entries:
                    entry_paper_ids[
                        entry.citation_key
                    ] = await catalog.upsert_bib_entry(source_root_id, entry, root)
                entry_total += len(entries)
                for pdf_path in partition.pdf_paths:
                    candidates = entry_candidates(pdf_path, entries)
                    paper_id = (
                        entry_paper_ids[candidates[0].citation_key]
                        if len(candidates) == 1
                        else None
                    )
                    await catalog.upsert_pdf(
                        source_root_id, root, partition, pdf_path, paper_id
                    )
                    relative = pdf_path.relative_to(root).as_posix()
                    if len(candidates) == 1:
                        matched_total += 1
                    elif not candidates:
                        await catalog.record_issue(
                            source_root_id,
                            "warning",
                            "unmatched_pdf",
                            relative,
                            {"reason": "No exact DOI, title, or declared-file match"},
                        )
                        issue_total += 1
                    else:
                        await catalog.record_issue(
                            source_root_id,
                            "warning",
                            "ambiguous_pdf_match",
                            relative,
                            {
                                "citation_keys": [
                                    entry.citation_key for entry in candidates
                                ]
                            },
                        )
                        issue_total += 1
                    pdf_total += 1
                    progress.update(1)
                await catalog.db.commit()
        await catalog.db.commit()
    logger.info(
        "ingest_complete",
        dataset_root=str(root),
        bib_entries=entry_total,
        pdfs=pdf_total,
        matched_pdfs=matched_total,
        issues=issue_total,
        database=str(Path(args.database)),
    )
    return 0


async def command_parse(args: SimpleNamespace, logger: Any) -> int:
    workers = max(1, args.workers)
    config = parser_config(max(1, args.ocr_workers))
    async with Catalog(Path(args.database)) as catalog:
        jobs = await catalog.pending_parse_jobs(args.limit, args.force)
        if not jobs:
            logger.info("parse_nothing_to_do", database=str(Path(args.database)))
            return 0
        logger.info("parse_started", jobs=len(jobs), workers=workers, config=config)
        iterator = iter(jobs)
        active: set[asyncio.Task[ParseOutcome]] = set()
        for _ in range(min(workers, len(jobs))):
            active.add(
                asyncio.create_task(
                    asyncio.to_thread(parse_pdf_sync, next(iterator), config)
                )
            )
        succeeded = failed = 0
        with tqdm(
            total=len(jobs), desc="Parsing PDFs", unit="pdf", disable=args.no_progress
        ) as progress:
            while active:
                done, active = await asyncio.wait(
                    active, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    outcome = task.result()
                    await catalog.record_parse(outcome, config)
                    if outcome.success:
                        succeeded += 1
                    else:
                        failed += 1
                        logger.error(
                            "parse_failed",
                            path=str(outcome.job.path),
                            error=outcome.error,
                        )
                    progress.update(1)
                    if (succeeded + failed) % 10 == 0 or succeeded + failed == len(
                        jobs
                    ):
                        logger.info(
                            "parse_progress",
                            completed=succeeded + failed,
                            total=len(jobs),
                            succeeded=succeeded,
                            failed=failed,
                        )
                    try:
                        next_job = next(iterator)
                    except StopIteration:
                        continue
                    active.add(
                        asyncio.create_task(
                            asyncio.to_thread(parse_pdf_sync, next_job, config)
                        )
                    )
    logger.info("parse_complete", total=len(jobs), succeeded=succeeded, failed=failed)
    return 0 if failed == 0 else 2


async def command_validate(args: SimpleNamespace, logger: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        # Study tables are additive and are also checked when screening has not
        # yet been run, so validation describes one consistent database shape.
        from ara import ensure_ara_schema

        await ensure_ara_schema(catalog)
        cursor = await catalog.db.execute("PRAGMA integrity_check")
        integrity = [row[0] for row in await cursor.fetchall()]
        cursor = await catalog.db.execute("PRAGMA foreign_key_check")
        foreign_keys = await cursor.fetchall()
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM ingest_issues WHERE resolved_at IS NULL"
        )
        issues = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM file_assets WHERE paper_id IS NULL"
        )
        unmatched = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM content_versions WHERE is_current=1 AND quality_status='accepted'"
        )
        accepted_markdown = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute("SELECT COUNT(*) AS count FROM file_assets")
        pdfs = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM study_experiments"
        )
        experiments = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM applicability_runs WHERE status IN ('pending','prepared','submitted','running')"
        )
        incomplete_screening_runs = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM applicability_evidence WHERE match_status IN ('pending','ambiguous','not_found')"
        )
        unresolved_evidence = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM ara_experiments"
        )
        ara_experiments = int((await cursor.fetchone())["count"])
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) AS count FROM ara_runs WHERE status IN ('pending','running')"
        )
        incomplete_ara_runs = int((await cursor.fetchone())["count"])
    valid = integrity == ["ok"] and not foreign_keys
    logger.info(
        "validation_complete",
        valid=valid,
        integrity=integrity,
        foreign_key_violations=len(foreign_keys),
        pdfs=pdfs,
        accepted_markdown=accepted_markdown,
        unmatched_pdfs=unmatched,
        open_issues=issues,
        study_experiments=experiments,
        incomplete_screening_runs=incomplete_screening_runs,
        unresolved_evidence=unresolved_evidence,
        ara_experiments=ara_experiments,
        incomplete_ara_runs=incomplete_ara_runs,
    )
    return 0 if valid else 2


async def command_search(args: SimpleNamespace, logger: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        cursor = await catalog.db.execute(
            "SELECT s.paper_id, p.title, p.canonical_doi, p.publication_year, snippet(paper_search, 4, '[', ']', '…', 16) AS snippet "
            "FROM paper_search s JOIN papers p ON p.id=CAST(s.paper_id AS INTEGER) WHERE paper_search MATCH ? ORDER BY rank LIMIT ?",
            (args.query, args.limit),
        )
        results = [dict(row) for row in await cursor.fetchall()]
    logger.info(
        "search_complete", query=args.query, result_count=len(results), results=results
    )
    return 0


def database_has_study_results(database: Path) -> bool:
    if not database.exists():
        return False
    with closing(sqlite3.connect(database)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='study_experiments'"
        ).fetchone()
        return bool(
            table
            and connection.execute("SELECT 1 FROM study_experiments LIMIT 1").fetchone()
        )


def backup_database(database: Path) -> Path:
    """Create and integrity-check an online SQLite backup before rebuilding."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = database.with_name(f"{database.stem}.backup-{stamp}{database.suffix}")
    suffix = 1
    while backup.exists():
        backup = database.with_name(
            f"{database.stem}.backup-{stamp}-{suffix}{database.suffix}"
        )
        suffix += 1
    with (
        closing(sqlite3.connect(database)) as source,
        closing(sqlite3.connect(backup)) as target,
    ):
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()
    if not integrity or integrity[0] != "ok":
        backup.unlink(missing_ok=True)
        raise sqlite3.DatabaseError("Rebuild backup failed integrity_check")
    return backup


async def command_rebuild(args: SimpleNamespace, logger: Any) -> int:
    if not args.yes:
        logger.error(
            "rebuild_requires_confirmation",
            hint="Run again with --yes. This removes the SQLite database and its parse history.",
        )
        return 2
    database = Path(args.database)
    if database_has_study_results(database) and not getattr(
        args, "delete_study_results", False
    ):
        logger.error(
            "rebuild_preserves_study_results",
            hint="This database contains screening experiments. Export/use the existing database, or explicitly pass --delete-study-results to create a backup and remove them.",
        )
        return 2
    backup = backup_database(database) if database.exists() else None
    if backup:
        logger.warning(
            "database_backup_created",
            database=str(database),
            backup=str(backup),
            backup_sha256=await asyncio.to_thread(sha256_file, backup),
        )
    for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        if candidate.exists():
            candidate.unlink()
    logger.warning(
        "database_removed",
        database=str(database),
        backup=str(backup) if backup else None,
    )
    ingest_args = SimpleNamespace(**vars(args))
    result = await command_ingest(ingest_args, logger)
    if result != 0 or not args.parse:
        return result
    return await command_parse(SimpleNamespace(**vars(args)), logger)


if __name__ == "__main__":
    raise SystemExit("main.py is an implementation module; use `uv run lakdb ...`.")
