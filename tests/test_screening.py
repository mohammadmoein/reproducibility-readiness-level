from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from pydantic import ValidationError

import completions
import dashboard
import main
import screening
from dashboard import ScreeningDashboard, human_time, human_tokens, short_title


class FakeUsage:
    def model_dump(self, mode="json"):
        return {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 25},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 120,
        }


class FakeResponse:
    id = "resp-fixture"
    usage = FakeUsage()

    def __init__(self):
        self.output_parsed = screening.ApplicabilityResult(
            reproducibility_applicability="applicable",
            contribution_type="empirical_study",
            claim_type="inferential_effect",
            claim_description=None,
            study_description="A fixture study",
            data_description="Student data",
            analysis_description="Regression",
            evidence=[],
            confidence="high",
            reason="Reports empirical analysis.",
        )

    def model_dump(self, mode="json"):
        return {"id": self.id, "usage": self.usage.model_dump(), "output": []}


class FakeResponses:
    calls = 0

    async def parse(self, **kwargs):
        self.__class__.calls += 1
        self.last_kwargs = kwargs
        return FakeResponse()


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.responses = FakeResponses()


class FakeRequestCounts:
    def model_dump(self):
        return {"completed": 3, "failed": 0, "total": 3}


class FakeBatch:
    id = "batch-fixture"
    status = "completed"
    output_file_id = "output-fixture"
    error_file_id = None
    request_counts = FakeRequestCounts()

    def model_dump(self, mode="json"):
        return {
            "id": self.id,
            "status": self.status,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
            "request_counts": self.request_counts.model_dump(),
        }


class FakeBatches:
    async def retrieve(self, _batch_id):
        return FakeBatch()


class FakeFileContent:
    def __init__(self, content: bytes):
        self.content = content


class FakeFiles:
    output = b""

    async def content(self, file_id):
        return FakeFileContent(self.output if file_id == "output-fixture" else b"")


class FakeBatchOpenAI:
    def __init__(self, **kwargs):
        self.batches = FakeBatches()
        self.files = FakeFiles()


class NullLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def make_pdf(path: Path, lines: list[str]) -> None:
    document = pymupdf.open()
    for line in lines:
        page = document.new_page()
        page.insert_text((72, 72), line, fontsize=11)
    document.save(path)
    document.close()


class ModelTests(unittest.TestCase):
    def test_nushell_completion_generation_is_deterministic_and_canonical(self):
        generated = completions.nushell_module()
        self.assertEqual(generated, completions.nushell_module())
        for command, _ in completions._CANONICAL_SIGNATURES:
            name = f"lakdb {command}".strip()
            uv_name = f"uv run lakdb {command}".strip()
            self.assertIn(f'export extern "{name}"', generated)
            self.assertIn(f'export extern "{uv_name}"', generated)
        for command in (
            "screen-create", "screen-freeze", "screen-experiments", "screen-delete",
            "screen", "screen-batch-prepare", "screen-batch-submit", "screen-batch-status",
            "screen-batch-collect", "screen-finalize", "screen-highlight",
            "ara-experiments", "ara-create", "ara-run", "ara-report",
        ):
            self.assertNotIn(f'export extern "lakdb {command}" [', generated)
            self.assertNotIn(f'export extern "uv run lakdb {command}" [', generated)

    def test_strict_structured_output(self):
        result = screening.ApplicabilityResult.model_validate(
            {
                "reproducibility_applicability": "applicable",
                "contribution_type": "empirical_study",
                "claim_type": "inferential_effect",
                "claim_description": None,
                "study_description": "Experiment",
                "data_description": "Logs",
                "analysis_description": "Regression",
                "evidence": [
                    {
                        "quote": "We analyzed student log data.",
                        "section": "Methods",
                        "supports": "data_use",
                        "explanation": "Identifies analyzed data.",
                    }
                ],
                "confidence": "high",
                "reason": "Reports an empirical analysis.",
            }
        )
        self.assertEqual(result.reproducibility_applicability.value, "applicable")
        with self.assertRaises(ValidationError):
            screening.ApplicabilityResult.model_validate(
                {**result.model_dump(), "unexpected": True}
            )

    def test_request_uses_strict_responses_schema_and_reasoning(self):
        body = screening.request_body("paper", "gpt-test", "low")
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(
            body["text"]["format"]["schema"]["additionalProperties"], False
        )
        self.assertFalse(body["store"])

    def test_batch_response_text_and_usage(self):
        payload = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "{}"}]}
            ]
        }
        self.assertEqual(screening.extract_response_text(payload), "{}")
        usage = screening.usage_values(
            {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 120,
            }
        )
        self.assertEqual(
            (
                usage["input"],
                usage["cached"],
                usage["output"],
                usage["reasoning"],
                usage["total"],
            ),
            (100, 40, 20, 8, 120),
        )

    def test_dashboard_formatters(self):
        self.assertEqual(short_title("a" * 21), "a" * 20 + "…")
        self.assertEqual(human_tokens(21_300), "21.3K")
        self.assertEqual(human_time(12_310), "12.31s")

    def test_rebuild_backup_is_integrity_checked_and_closed(self):
        import sqlite3
        from contextlib import closing

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "catalog.sqlite"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE fixture(value TEXT)")
                connection.execute("INSERT INTO fixture VALUES ('preserved')")
                connection.commit()
            backup = main.backup_database(database)
            with closing(sqlite3.connect(backup)) as connection:
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("SELECT value FROM fixture").fetchone()[0], "preserved")
            backup.unlink()


class EvidenceTests(unittest.TestCase):
    def test_unique_exact_quote_and_highlight_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            output = root / "output.pdf"
            quote = "We analyzed student log data using regression."
            make_pdf(source, [quote])
            before = screening.sha256_file(source)
            match = screening.locate_quote(source, quote)
            self.assertEqual(match["status"], "exact")
            rows = [
                {
                    "match_status": match["status"],
                    "pdf_page_number": match["page"],
                    "quads_json": json.dumps(match["quads"]),
                    "supports": "data_use",
                    "quote": quote,
                    "applicability_run_id": 1,
                }
            ]
            count = screening.add_highlights(source, output, rows, "applicable")
            self.assertGreaterEqual(count, 1)
            self.assertEqual(screening.sha256_file(source), before)
            with pymupdf.open(output) as document:
                self.assertIsNotNone(document[0].first_annot)

    def test_duplicate_quote_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            quote = "Repeated evidence sentence appears here."
            make_pdf(source, [quote + " " + quote])
            self.assertEqual(
                screening.locate_quote(source, quote)["status"], "ambiguous"
            )

    def test_quality_failure_for_corrupt_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            make_pdf(source, ["Paper title"])
            markdown = "Paper title\n" + ("�" * 20) + ("x" * 1100)
            status, metrics, warnings = screening.quality_metrics(
                markdown, "Paper title", source, 1
            )
            self.assertEqual(status, "failed")
            self.assertGreater(metrics["replacement_rate"], 0.01)
            self.assertIn("replacement_characters", warnings)


class DatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "test.sqlite"
        self.pdf_path = self.root / "paper.pdf"
        make_pdf(self.pdf_path, ["A Study of Learning " + "empirical evidence " * 10])
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()
        await screening.ensure_schema(self.catalog)
        now = screening.utc_now()
        cursor = await self.catalog.db.execute(
            "INSERT INTO source_roots(dataset_path,created_at,last_seen_at) VALUES (?,?,?)",
            (str(self.root), now, now),
        )
        source_id = cursor.lastrowid
        cursor = await self.catalog.db.execute(
            "INSERT INTO papers(source_identity,title,collection_year,canonical_metadata_json,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            ("paper-1", "A Study of Learning", 2023, "{}", now, now),
        )
        self.paper_id = cursor.lastrowid
        digest = screening.sha256_file(self.pdf_path)
        cursor = await self.catalog.db.execute(
            "INSERT INTO file_assets(paper_id,source_root_id,relative_path,dataset_year,paper_kind,byte_size,sha256,page_count,discovered_at,last_verified_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self.paper_id,
                source_id,
                "paper.pdf",
                2023,
                "long",
                self.pdf_path.stat().st_size,
                digest,
                1,
                now,
                now,
            ),
        )
        self.file_id = cursor.lastrowid
        cursor = await self.catalog.db.execute(
            "INSERT INTO parse_runs(file_asset_id,status,parser_name,parser_version,config_json,source_pdf_sha256,started_at,finished_at,warnings_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (self.file_id, "succeeded", "fixture", "1", "{}", digest, now, now, "[]"),
        )
        parse_id = cursor.lastrowid
        self.markdown = "# A Study of Learning\n\n" + (
            "This empirical study analyzed student data with regression. " * 30
        )
        cursor = await self.catalog.db.execute(
            "INSERT INTO content_versions(file_asset_id,parse_run_id,markdown,markdown_sha256,parser_name,parser_version,config_json,quality_status,quality_metrics_json,is_current,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.file_id,
                parse_id,
                self.markdown,
                screening.sha256_text(self.markdown),
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
        self.experiment_id = await screening.create_experiment(
            self.catalog,
            name="test-v1",
            model="gpt-test",
            reasoning_effort="low",
            repeats=3,
            freeze=True,
        )
        await self.catalog.db.commit()

    async def asyncTearDown(self):
        await self.catalog.__aexit__(None, None, None)
        self.temp.cleanup()

    async def add_dashboard_evidence(self, labels=None):
        labels = labels or ["applicable", "applicable", "applicable"]
        experiment = await screening.get_experiment(self.catalog, "test-v1")
        candidate = {
            "paper_id": self.paper_id,
            "content_version_id": self.content_id,
            "file_asset_id": self.file_id,
            "markdown": self.markdown,
            "markdown_sha256": screening.sha256_text(self.markdown),
            "source_pdf_sha256": screening.sha256_file(self.pdf_path),
        }
        body = screening.request_body(self.markdown, "gpt-test", "low")
        quote = "This empirical study analyzed student data with regression."
        for repetition, label in enumerate(labels, 1):
            run = await screening.ensure_run(
                self.catalog, experiment, candidate, repetition, "realtime", body
            )
            result = screening.ApplicabilityResult(
                reproducibility_applicability=label,
                contribution_type="empirical_study" if label == "applicable" else "unclear",
                claim_type="inferential_effect",
                claim_description=None,
                study_description="Fixture study",
                data_description="Student data",
                analysis_description="Regression",
                evidence=[
                    {
                        "quote": quote,
                        "section": "Results",
                        "supports": "reported_results",
                        "explanation": "Reports an empirical result.",
                    }
                ],
                confidence="high",
                reason=f"Run {repetition} reports empirical analysis.",
            )
            now = screening.utc_now()
            await screening.save_result(
                self.catalog,
                run["id"],
                result,
                raw={},
                response_id=f"fixture-{repetition}",
                usage={"input": 10, "output": 2, "total": 12},
                duration=5,
                started=now,
                finished=now,
            )
        evidence = await (
            await self.catalog.db.execute(
                "SELECT id FROM applicability_evidence ORDER BY id"
            )
        ).fetchall()
        for index, item in enumerate(evidence):
            status = "not_found" if index == len(evidence) - 1 else "exact"
            await self.catalog.db.execute(
                "UPDATE applicability_evidence SET match_status=?,pdf_page_number=?,match_score=?,matched_pdf_text=? WHERE id=?",
                (status, None if status == "not_found" else 1, 0.0 if status == "not_found" else 1.0, quote if status == "exact" else None, item["id"]),
            )
        await screening.finalize(self.catalog, experiment)
        await self.catalog.db.commit()

    async def test_batch_prepare_is_idempotent_and_has_three_unique_ids(self):
        args = SimpleNamespace(
            database=self.db_path,
            experiment="test-v1",
            output=self.root / "batch.jsonl",
            paper_id=None,
            year=None,
            kind=None,
            limit=1,
            no_progress=True,
        )
        await self.catalog.__aexit__(None, None, None)
        await screening.command_batch_prepare(args, NullLogger(), main.Catalog)
        first = args.output.read_text(encoding="utf-8")
        await screening.command_batch_prepare(args, NullLogger(), main.Catalog)
        second = args.output.read_text(encoding="utf-8")
        self.assertEqual(first, second)
        lines = [json.loads(line) for line in first.splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({line["custom_id"] for line in lines}), 3)
        self.assertTrue(all(line["url"] == "/v1/responses" for line in lines))
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()
        stored_input = await (
            await self.catalog.db.execute(
                "SELECT content,sha256 FROM openai_batch_files WHERE file_kind='input'"
            )
        ).fetchone()
        self.assertEqual(stored_input["content"], first)
        self.assertEqual(stored_input["sha256"], screening.sha256_file(args.output))
        count = (
            await (
                await self.catalog.db.execute("SELECT COUNT(*) FROM applicability_runs")
            ).fetchone()
        )[0]
        self.assertEqual(count, 3)
        await self.catalog.db.execute(
            "UPDATE applicability_runs SET status='succeeded',label='unclear'"
        )
        await self.catalog.db.commit()
        await self.catalog.__aexit__(None, None, None)
        await screening.command_batch_prepare(args, NullLogger(), main.Catalog)
        self.assertEqual(args.output.read_text(encoding="utf-8"), first)
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()

    async def test_batch_collect_roundtrip_validates_and_saves_raw_files(self):
        args = SimpleNamespace(
            database=self.db_path,
            experiment="test-v1",
            output=self.root / "batch.jsonl",
            paper_id=None,
            year=None,
            kind=None,
            limit=1,
            no_progress=True,
        )
        await self.catalog.__aexit__(None, None, None)
        await screening.command_batch_prepare(args, NullLogger(), main.Catalog)
        requests = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()]
        result = screening.ApplicabilityResult(
            reproducibility_applicability="applicable",
            contribution_type="empirical_study",
            claim_type="inferential_effect",
            claim_description=None,
            study_description="Fixture study",
            data_description="Student logs",
            analysis_description="Regression",
            evidence=[],
            confidence="high",
            reason="Reports empirical results.",
        )
        output_rows = []
        for request in requests:
            output_rows.append(
                {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "request_id": "request-fixture",
                        "body": {
                            "id": "response-fixture",
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "total_tokens": 12,
                            },
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": result.model_dump_json(),
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                    "error": None,
                }
            )
        FakeFiles.output = ("\n".join(json.dumps(row) for row in output_rows) + "\n").encode()
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()
        await self.catalog.db.execute(
            "UPDATE openai_batch_jobs SET openai_batch_id='batch-fixture',status='submitted',submitted_at=?",
            (screening.utc_now(),),
        )
        await self.catalog.db.execute(
            "UPDATE applicability_runs SET status='submitted',started_at=?",
            (screening.utc_now(),),
        )
        await self.catalog.db.commit()
        await self.catalog.__aexit__(None, None, None)
        collect_args = SimpleNamespace(database=self.db_path, batch_id=1, no_progress=True)
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-key"}),
            patch("screening.AsyncOpenAI", FakeBatchOpenAI),
        ):
            self.assertEqual(
                await screening.command_batch_collect(collect_args, NullLogger(), main.Catalog),
                0,
            )
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()
        self.assertEqual(
            (await (await self.catalog.db.execute("SELECT COUNT(*) FROM applicability_runs WHERE status='succeeded'")).fetchone())[0],
            3,
        )
        self.assertEqual(
            (await (await self.catalog.db.execute("SELECT SUM(total_tokens) FROM task_executions WHERE task_type='applicability_batch_item'")).fetchone())[0],
            36,
        )
        stored_output = await (
            await self.catalog.db.execute(
                "SELECT content FROM openai_batch_files WHERE file_kind='output'"
            )
        ).fetchone()
        self.assertEqual(stored_output["content"].encode(), FakeFiles.output)

    async def test_realtime_screen_is_mocked_repeated_and_records_usage(self):
        args = SimpleNamespace(
            database=self.db_path,
            experiment="test-v1",
            paper_id=None,
            year=None,
            kind=None,
            limit=1,
            retries=0,
            no_progress=True,
        )
        await self.catalog.__aexit__(None, None, None)
        FakeResponses.calls = 0
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-key"}),
            patch("screening.AsyncOpenAI", FakeOpenAI),
        ):
            result = await screening.command_screen(args, NullLogger(), main.Catalog)
        self.assertEqual(result, 0)
        self.assertEqual(FakeResponses.calls, 3)
        self.catalog = main.Catalog(self.db_path)
        await self.catalog.__aenter__()
        usage = await (
            await self.catalog.db.execute(
                "SELECT SUM(input_tokens),SUM(output_tokens),SUM(total_tokens),SUM(cached_input_tokens),SUM(reasoning_output_tokens) FROM task_executions WHERE task_type='applicability_realtime'"
            )
        ).fetchone()
        self.assertEqual(tuple(usage), (300, 60, 360, 75, 15))
        decision = await (
            await self.catalog.db.execute(
                "SELECT final_label,agreement_type FROM applicability_decisions"
            )
        ).fetchone()
        self.assertEqual(tuple(decision), ("applicable", "unanimous"))
        raw = await (
            await self.catalog.db.execute(
                "SELECT request_json,raw_response_json FROM applicability_runs LIMIT 1"
            )
        ).fetchone()
        experiment = await screening.get_experiment(self.catalog, "test-v1")
        self.assertEqual(await screening.candidates(self.catalog, args, experiment), [])
        self.assertIn("A Study of Learning", raw["request_json"])
        self.assertIn("resp-fixture", raw["raw_response_json"])

    async def test_protocol_aggregation(self):
        experiment = await screening.get_experiment(self.catalog, "test-v1")
        row = {
            "paper_id": self.paper_id,
            "content_version_id": self.content_id,
            "file_asset_id": self.file_id,
            "markdown": self.markdown,
            "markdown_sha256": screening.sha256_text(self.markdown),
            "source_pdf_sha256": screening.sha256_file(self.pdf_path),
        }
        body = screening.request_body(self.markdown, "gpt-test", "low")
        labels = ["applicable", "applicable", "unclear"]
        for repetition, label in enumerate(labels, 1):
            run = await screening.ensure_run(
                self.catalog, experiment, row, repetition, "batch", body
            )
            result = screening.ApplicabilityResult(
                reproducibility_applicability=label,
                contribution_type="empirical_study"
                if label == "applicable"
                else "unclear",
                claim_type="inferential_effect",
                claim_description=None,
                study_description=None,
                data_description=None,
                analysis_description=None,
                evidence=[],
                confidence="high",
                reason="fixture result",
            )
            now = screening.utc_now()
            await screening.save_result(
                self.catalog,
                run["id"],
                result,
                raw={},
                response_id="fixture",
                usage={"input": 1, "output": 1, "total": 2},
                duration=5,
                started=now,
                finished=now,
            )
        await screening.finalize(self.catalog, experiment)
        decision = await (
            await self.catalog.db.execute("SELECT * FROM applicability_decisions")
        ).fetchone()
        self.assertEqual(decision["final_label"], "applicable")
        self.assertEqual(decision["agreement_type"], "majority")

    async def test_rebuild_refuses_to_silently_delete_study_results(self):
        args = SimpleNamespace(
            database=self.db_path, yes=True, delete_study_results=False
        )
        result = await main.command_rebuild(args, NullLogger())
        self.assertEqual(result, 2)
        self.assertTrue(self.db_path.exists())
        experiment = await screening.get_experiment(self.catalog, "test-v1")
        self.assertEqual(experiment["name"], "test-v1")

    async def test_textual_dashboard_renders_headlessly(self):
        await self.add_dashboard_evidence()
        app = ScreeningDashboard(self.db_path, experiment="test-v1", refresh_seconds=60)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            tasks = app.query_one("#tasks-table")
            self.assertEqual(str(tasks.columns["year"].label), "Year")
            self.assertEqual(str(tasks.columns["paper"].label), "Paper")
            tree = app.query_one("#evidence-tree")
            self.assertEqual(len(tree.root.children), 1)
            paper = tree.root.children[0]
            self.assertEqual(paper.data["type"], "paper")
            self.assertEqual(len(paper.children), 1)
            final_run = paper.children[0]
            self.assertEqual(final_run.data["type"], "run")
            self.assertEqual(len(final_run.children), 1, "final view deduplicates evidence")
            evidence = final_run.children[0]
            app._preview_tree_node(evidence)
            self.assertIn("student data", app.selected_quote)
            self.assertEqual(app.selected_source_path, str(self.pdf_path))

            paper.expand()
            final_run.expand()
            tree.move_cursor(evidence, animate=False)
            selected_key = evidence.data["key"]
            app._reload_evidence()
            await pilot.pause()
            self.assertEqual(tree.cursor_node.data["key"], selected_key)
            self.assertTrue(tree.cursor_node.parent.is_expanded)

            app.evidence_mode = "unresolved"
            app._reload_evidence()
            self.assertEqual(len(tree.root.children[0].children[0].children), 1)
            self.assertEqual(
                tree.root.children[0].children[0].children[0].data["match_status"],
                "not_found",
            )

            app.evidence_search = "does not exist"
            app._reload_evidence()
            self.assertEqual(len(tree.root.children), 0)

    async def test_dashboard_file_actions_are_safe_and_responsive(self):
        await self.add_dashboard_evidence()
        app = ScreeningDashboard(self.db_path, experiment="test-v1", refresh_seconds=60)
        async with app.run_test(size=(90, 35)) as pilot:
            await pilot.pause()
            self.assertTrue(app.screen.has_class("narrow"))
            app.selected_source_path = str(self.pdf_path)
            with patch("dashboard.launch_file") as opener:
                app.action_open_source()
                opener.assert_called_once_with(self.pdf_path)
            app.selected_quote = "fixture quotation"
            with patch.object(app, "copy_to_clipboard") as clipboard:
                app.action_copy_quote()
                clipboard.assert_called_once_with("fixture quotation")

    async def test_dashboard_disagreement_and_run_filters(self):
        await self.add_dashboard_evidence(["applicable", "applicable", "unclear"])
        app = ScreeningDashboard(self.db_path, experiment="test-v1", refresh_seconds=60)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            tree = app.query_one("#evidence-tree")
            app.evidence_mode = "disagreements"
            app.evidence_run = 3
            app._reload_evidence()
            self.assertEqual(len(tree.root.children), 1)
            run = tree.root.children[0].children[0]
            self.assertEqual(run.data["repetition_index"], 3)
            self.assertEqual(run.data["run_label"], "unclear")

    async def test_experiment_listing_and_cascade_delete(self):
        await self.add_dashboard_evidence()
        rows = await screening.experiment_rows(self.catalog)
        self.assertEqual([row["name"] for row in rows], ["test-v1"])
        self.assertEqual(rows[0]["runs"], 3)
        artifact_paths = await screening.delete_experiments(
            self.catalog, [self.experiment_id]
        )
        self.assertEqual(artifact_paths, [])
        self.assertEqual(len(await screening.experiment_rows(self.catalog)), 0)
        self.assertEqual(
            (await (await self.catalog.db.execute("SELECT COUNT(*) FROM applicability_runs")).fetchone())[0],
            0,
        )
        self.assertEqual(
            (await (await self.catalog.db.execute("SELECT COUNT(*) FROM applicability_decisions")).fetchone())[0],
            0,
        )

    async def test_completion_candidates_are_machine_readable_and_filterable(self):
        experiments = completions.completion_candidates(self.db_path, "experiments")
        self.assertEqual(experiments[0]["value"], "test-v1")
        self.assertIn("frozen", experiments[0]["description"])
        papers = completions.completion_candidates(
            self.db_path, "papers", year=2023, paper_kind="long"
        )
        self.assertEqual(papers[0]["value"], str(self.paper_id))
        self.assertIn("A Study of Learning", papers[0]["description"])
        self.assertEqual(json.loads(completions.candidates_json(papers)), papers)

    async def test_repeat_count_is_protocol_fixed(self):
        with self.assertRaises(ValueError):
            await screening.create_experiment(
                self.catalog, name="bad", model="x", reasoning_effort="low", repeats=2
            )


if __name__ == "__main__":
    unittest.main()
