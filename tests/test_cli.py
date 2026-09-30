"""Click CLI contract and lazy-import checks."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
from lakdb.cli import cli


class ClickCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_canonical_grouped_command_help(self) -> None:
        commands = (
            ("experiment", "create"), ("experiment", "freeze"),
            ("experiment", "list"), ("experiment", "delete"),
            ("screen", "run"), ("screen", "finalize"),
            ("batch", "prepare"), ("batch", "submit"),
            ("batch", "status"), ("batch", "collect"),
            ("highlight", "run"), ("ara", "create"), ("ara", "list"),
            ("ara", "run"), ("ara", "report"), ("artifact", "create"),
            ("artifact", "batch", "prepare"), ("resolve", "run"),
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(self.runner.invoke(cli, [*command, "--help"]).exit_code, 0)

    def test_legacy_commands_and_screen_shorthand_are_removed(self) -> None:
        for command in (
            "screen-create", "screen-freeze", "screen-experiments", "screen-delete",
            "screen-batch-prepare", "screen-batch-submit", "screen-batch-status",
            "screen-batch-collect", "screen-finalize", "screen-highlight",
            "ara-experiments", "ara-create", "ara-run", "ara-report",
        ):
            with self.subTest(command=command):
                result = self.runner.invoke(cli, [command, "--help"])
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("No such command", result.output)
        self.assertNotEqual(
            self.runner.invoke(cli, ["screen", "--experiment", "lak-2021"]).exit_code,
            0,
        )

    def test_lightweight_help_does_not_import_heavy_modules(self) -> None:
        code = "from lakdb.cli import cli; import sys; cli.main(args=['--help'], prog_name='lakdb', standalone_mode=False); print('HEAVY=' + str(any(name in sys.modules for name in ('openai','pymupdf','textual'))))"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertIn("HEAVY=False", result.stdout)

    def test_experiment_list_handles_missing_database_without_heavy_imports(self) -> None:
        with TemporaryDirectory() as directory:
            result = self.runner.invoke(cli, ["--database", str(Path(directory) / "missing.sqlite"), "experiment", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No applicability experiments", result.output)

    def test_generated_completion_scripts(self) -> None:
        for shell in ("nushell", "bash", "zsh", "fish", "powershell"):
            result = self.runner.invoke(cli, ["completions", shell])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(result.output.strip())


if __name__ == "__main__":
    unittest.main()
