"""Offline Stage 1 checks: schema, CLI, E score, resolver fixtures."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from pydantic import ValidationError

import artifact
import artifact_docs
import artifact_report
import environment
import main
from lakdb.cli import cli


class NullLog:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


CLAIM = {
    "open_data": {
        "verdict": "No",
        "quote": "We collected interview transcripts from 12 teachers.",
        "section": "Methods",
        "urls": ["https://example.org/data"],
    },
    "data_documentation": {
        "verdict": "No",
        "quote": "No codebook is provided.",
        "section": "Methods",
        "urls": [],
    },
    "open_materials": {
        "verdict": "Partial",
        "quote": "Scripts are at https://github.com/example/repo",
        "section": "Availability",
        "urls": ["https://github.com/example/repo"],
    },
    "materials_documentation": {
        "verdict": "No",
        "quote": "The repository is mentioned without usage notes.",
        "section": "Availability",
        "urls": [],
    },
    "readme": {
        "verdict": "No",
        "quote": "No README is mentioned.",
        "section": "Availability",
        "urls": [],
    },
    "license": {
        "verdict": "No",
        "quote": "No licence is stated.",
        "section": "Availability",
        "urls": [],
    },
    "preregistration": {
        "verdict": "No",
        "quote": "The study was not preregistered.",
        "section": "Methods",
        "urls": [],
    },
}


class SchemaTests(unittest.TestCase):
    def test_claim_schema_requires_every_field_and_quote(self):
        schema = artifact.claim_schema()
        self.assertEqual(schema.get("additionalProperties"), False)
        required = set(schema["required"])
        self.assertEqual(required, set(artifact.CLAIM_FIELDS))
        result = artifact.ArtifactClaimResult.model_validate(CLAIM)
        self.assertEqual(result.open_data.verdict, "No")
        with self.assertRaises(ValidationError):
            artifact.ArtifactClaimResult.model_validate({**CLAIM, "open_data": {**CLAIM["open_data"], "quote": ""}})
        with self.assertRaises(ValidationError):
            artifact.ArtifactClaimResult.model_validate({**result.model_dump(), "unexpected": True})

    def test_request_body_is_strict_responses_schema(self):
        body = artifact.request_body("paper", "gpt-test", "medium", "claim")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse(body["store"])
        self.assertEqual(body["reasoning"], {"effort": "medium"})

    def test_docs_schema_uses_source_vocabularies(self):
        schema = artifact.docs_schema()
        self.assertEqual(set(schema["required"]), {"data_documentation", "materials_documentation"})

    def test_diagnostic_subset_from_haim_csv(self):
        survey = artifact.load_haim_fields(Path("baseline/haim-lak/artifacts/qualtrics-survey-data.csv"))
        dois = artifact.diagnostic_dois(survey)
        self.assertGreaterEqual(len(dois), 20)
        self.assertEqual(sum(1 for doi in dois if survey[doi]["open_data"] == "Yes"), 7)
        self.assertEqual(sum(1 for doi in dois if survey[doi]["open_data"] == "Data Available on Request"), 2)
        self.assertEqual(sum(1 for doi in dois if survey[doi]["open_data"] == "N/A"), 6)
        self.assertGreaterEqual(sum(1 for doi in dois if survey[doi]["open_materials"] == "Full"), 6)
        self.assertEqual(artifact.diagnostic_dois(survey), dois)

    def test_oddpub_detects_zenodo_phrase(self):
        detected = artifact_report.oddpub_detect("Data are available at zenodo.org under DOI 10.5281/zenodo.1")
        self.assertTrue(detected["open_data"])


class EnvironmentTests(unittest.TestCase):
    def test_e_components_on_synthetic_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("pandas==2.2.0\nnumpy==1.26.0\n", encoding="utf-8")
            (root / "uv.lock").write_text("placeholder", encoding="utf-8")
            (root / ".python-version").write_text("3.13\n", encoding="utf-8")
            (root / "analysis.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "random.seed(0)\nPath('out').mkdir(exist_ok=True)\nopen('out/x.txt','w')\n",
                encoding="utf-8",
            )
            scores = environment.score_repository(root)
            self.assertTrue(scores["manifest_present"])
            self.assertTrue(scores["versions_pinned"])
            self.assertTrue(scores["lockfile_or_container"])
            self.assertTrue(scores["seeds_set"])
            self.assertTrue(scores["relative_paths"])
            self.assertTrue(scores["creates_output_dirs"])
            self.assertTrue(scores["runtime_declared"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("pandas>=2\n", encoding="utf-8")
            (root / "bad.py").write_text("open('/Users/someone/data.csv')\n", encoding="utf-8")
            scores = environment.score_repository(root)
            self.assertTrue(scores["manifest_present"])
            self.assertFalse(scores["versions_pinned"])
            self.assertFalse(scores["relative_paths"])
            self.assertFalse(scores["lockfile_or_container"])


class ClickCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_artifact_help_is_click_not_argparse(self) -> None:
        for args in (
            ["artifact", "--help"],
            ["artifact", "create", "--help"],
            ["artifact", "batch", "prepare", "--help"],
            ["artifact", "diagnostic", "--help"],
            ["resolve", "run", "--help"],
            ["fetch", "data", "--help"],
        ):
            result = self.runner.invoke(cli, args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertNotIn("unrecognized arguments", result.output)

    def test_limit_help_counts_papers(self) -> None:
        result = self.runner.invoke(cli, ["artifact", "run", "--help"])
        self.assertIn("Number of papers, not API calls", result.output)

    def test_artifact_list_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.runner.invoke(
                cli, ["--database", str(Path(directory) / "missing.sqlite"), "artifact", "list"]
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No artifact experiments", result.output)


class FixtureHTTP(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:  # noqa: N802
        self._handle()

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        if self.path == "/go":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
            return
        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "2")
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class DatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "test.sqlite"
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()
        await artifact.ensure_schema(self.catalog)
        now = artifact.utc_now()
        cursor = await self.catalog.db.execute(
            "INSERT INTO source_roots(dataset_path,created_at,last_seen_at) VALUES (?,?,?)",
            (str(self.root), now, now),
        )
        source_id = cursor.lastrowid
        cursor = await self.catalog.db.execute(
            "INSERT INTO papers(source_identity,canonical_doi,title,collection_year,canonical_metadata_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            ("paper-1", "10.1145/3448139.3448143", "A Study", 2021, "{}", now, now),
        )
        self.paper_id = cursor.lastrowid
        pdf = self.root / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.1 fixture")
        cursor = await self.catalog.db.execute(
            "INSERT INTO file_assets(paper_id,source_root_id,relative_path,dataset_year,paper_kind,byte_size,sha256,page_count,discovered_at,last_verified_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (self.paper_id, source_id, "paper.pdf", 2021, "long", 16, "a" * 64, 1, now, now),
        )
        self.file_id = cursor.lastrowid
        cursor = await self.catalog.db.execute(
            "INSERT INTO parse_runs(file_asset_id,status,parser_name,parser_version,config_json,source_pdf_sha256,started_at,finished_at,warnings_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (self.file_id, "succeeded", "fixture", "1", "{}", "a" * 64, now, now, "[]"),
        )
        parse_id = cursor.lastrowid
        markdown = "# A Study\n\nWe collected interview transcripts from 12 teachers. " * 40
        cursor = await self.catalog.db.execute(
            "INSERT INTO content_versions(file_asset_id,parse_run_id,markdown,markdown_sha256,parser_name,parser_version,config_json,quality_status,quality_metrics_json,is_current,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.file_id,
                parse_id,
                markdown,
                artifact.sha256_text(markdown),
                "fixture",
                "1",
                "{}",
                "accepted",
                "{}",
                1,
                now,
            ),
        )
        self.content_id = cursor.lastrowid
        self.markdown = markdown
        self.experiment_id = await artifact.create_experiment(
            self.catalog,
            name="art-test",
            model="gpt-test",
            reasoning_effort="medium",
            repeats=3,
            freeze=True,
        )
        await self.catalog.db.commit()

    async def asyncTearDown(self) -> None:
        await self.catalog.__aexit__(None, None, None)
        self.temp.cleanup()

    async def test_schema_idempotent_and_tables_exist(self) -> None:
        await artifact.ensure_schema(self.catalog)
        cursor = await self.catalog.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'artifact_%'"
        )
        names = {row[0] for row in await cursor.fetchall()}
        self.assertTrue(
            {
                "artifact_experiments",
                "artifact_runs",
                "artifact_field_results",
                "artifact_urls",
                "artifact_snapshots",
                "artifact_environment",
            }.issubset(names)
        )

    async def test_round_trip_parsed_result(self) -> None:
        experiment = await artifact.get_experiment(self.catalog, "art-test")
        row = {
            "paper_id": self.paper_id,
            "content_version_id": self.content_id,
            "file_asset_id": self.file_id,
            "markdown_sha256": artifact.sha256_text(self.markdown),
            "source_pdf_sha256": "a" * 64,
        }
        body = artifact.request_body(self.markdown, "gpt-test", "medium", "claim")
        run = await artifact.ensure_run(self.catalog, experiment, row, 1, "realtime", body)
        result = artifact.ArtifactClaimResult.model_validate(CLAIM)
        now = artifact.utc_now()
        await artifact.save_result(
            self.catalog,
            run["id"],
            result,
            raw={"fixture": True},
            response_id="resp-1",
            usage={"input": 1, "output": 1, "total": 2},
            duration=1,
            started=now,
            finished=now,
        )
        await artifact.finalize(self.catalog, experiment)
        stored = await (
            await self.catalog.db.execute("SELECT parsed_result_json FROM artifact_runs WHERE id=?", (run["id"],))
        ).fetchone()
        again = artifact.ArtifactClaimResult.model_validate_json(stored["parsed_result_json"])
        self.assertEqual(again.open_data.quote, CLAIM["open_data"]["quote"])
        field = await (
            await self.catalog.db.execute(
                "SELECT verdict FROM artifact_field_results WHERE field_name='open_data'"
            )
        ).fetchone()
        self.assertEqual(field["verdict"], "No")

    async def test_batch_prepare_limit_is_papers_and_resumable(self) -> None:
        args = SimpleNamespace(
            database=self.db_path,
            experiment="art-test",
            output=self.root / "batch.jsonl",
            paper_id=None,
            year=None,
            kind=None,
            limit=1,
            no_progress=True,
            dois=None,
        )
        await self.catalog.__aexit__(None, None, None)
        await artifact.command_batch_prepare(args, NullLog(), main.Catalog)
        first = args.output.read_text(encoding="utf-8")
        lines = [json.loads(line) for line in first.splitlines()]
        self.assertEqual(len(lines), 3)
        await artifact.command_batch_prepare(args, NullLog(), main.Catalog)
        second = args.output.read_text(encoding="utf-8")
        self.assertEqual(len([line for line in second.splitlines() if line.strip()]), 3)
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()
        await self.catalog.db.execute("UPDATE artifact_runs SET status='succeeded'")
        await self.catalog.db.commit()
        await self.catalog.__aexit__(None, None, None)
        args.output = self.root / "batch-again.jsonl"
        await artifact.command_batch_prepare(args, NullLog(), main.Catalog)
        leftover = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        self.assertEqual(leftover.strip(), "")
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()

    async def test_docs_payload_excludes_source_bodies(self) -> None:
        store = self.root / "snap"
        content = store / "content"
        content.mkdir(parents=True)
        (content / "README.md").write_text("How to run the analysis.", encoding="utf-8")
        (content / "requirements.txt").write_text("pandas==2.2.0\n", encoding="utf-8")
        (content / "secret.py").write_text("print('do not send')\n", encoding="utf-8")
        (content / "data.csv").write_text("id,score\n1,2\n3,4\n", encoding="utf-8")
        built = artifact_docs.build_docs_payload(store, char_budget=8000, tree_cap=50)
        self.assertIn("How to run the analysis.", built["payload"])
        self.assertIn("pandas==2.2.0", built["payload"])
        self.assertIn("secret.py", built["payload"])
        self.assertNotIn("do not send", built["payload"])
        self.assertIn("id,score", built["payload"])
        self.assertNotIn("\n1,2", built["payload"])


class ResolverFixtureTests(unittest.IsolatedAsyncioTestCase):
    def test_request_url_normalizes_bare_hosts_and_rejects_relative_paths(self) -> None:
        import artifact_resolver

        self.assertEqual(
            artifact_resolver.request_url("github.com/duolingo/halflife-regression"),
            "https://github.com/duolingo/halflife-regression",
        )
        self.assertIsNone(artifact_resolver.request_url("/robots.txt"))
        self.assertIsNone(artifact_resolver.request_url("mailto:researcher@example.org"))

    async def test_resolve_ok_404_and_redirect(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHTTP)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            urls = self._write_urls(port)
            args = SimpleNamespace(
                database=Path(tempfile.gettempdir()) / "unused.sqlite",
                urls_file=urls,
                no_progress=True,
                experiment=None,
                year=None,
                all=False,
                paper_id=None,
                force=False,
                workers=1,
            )
            logs = []
            logger = NullLog()
            with tempfile.TemporaryDirectory() as directory:
                args.database = Path(directory) / "db.sqlite"
                await artifact_resolver_run(args, logger)
        finally:
            server.shutdown()
            server.server_close()

    def _write_urls(self, port: int) -> Path:
        handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt")
        base = f"http://127.0.0.1:{port}"
        handle.write(
            f"{base}/ok\n{base}/missing\n{base}/go\n"
            f"127.0.0.1:{port}/ok\n/robots.txt\n"
        )
        handle.close()
        return Path(handle.name)


async def artifact_resolver_run(args: SimpleNamespace, logger: SimpleNamespace) -> None:
    import artifact_resolver

    await artifact_resolver.command_resolve_run(args, logger, main.Catalog)


if __name__ == "__main__":
    unittest.main()
