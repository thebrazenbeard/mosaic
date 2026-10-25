"""Source-only regression guard for concealed editor-startup execution.

This does not prove how an editor behaved on any installed workstation.
"""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUSPECT_ASSETS = (
    "fa-solid.lkf",
    "fa-solid-500.woff2",
)


def read_jsonc(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    return json.loads(re.sub(r",\s*(?=[}\]])", "", raw))


class EditorStartupGuardTests(unittest.TestCase):
    def test_disguised_executable_assets_are_absent(self):
        for name in SUSPECT_ASSETS:
            with self.subTest(name=name):
                self.assertFalse((ROOT / "public" / "fonts" / name).exists())

    def test_no_implicit_editor_task_or_font_loader(self):
        config = read_jsonc(ROOT / ".vscode" / "tasks.json")
        for task in config.get("tasks", ()):
            with self.subTest(task=task.get("label")):
                self.assertNotEqual(
                    task.get("runOptions", {}).get("runOn"), "folderOpen"
                )
                command = str(task.get("command", ""))
                for name in SUSPECT_ASSETS:
                    self.assertNotIn(name, command)

    def test_workspace_does_not_claim_automatic_task_consent(self):
        settings = read_jsonc(ROOT / ".vscode" / "settings.json")
        self.assertNotIn("task.allowAutomaticTasks", settings)
        self.assertNotIn("tasks", settings)


if __name__ == "__main__":
    unittest.main()
