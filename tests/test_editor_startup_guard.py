"""Source-only guard against workspace-open task execution and disguised font payloads.

This checks committed source; VS Code's application-scoped automatic-task
preference is NOT a repository-enforced runtime security boundary.
"""
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# SHA-256 of font assets from the verified pre-injection parent; SVG line
# endings are canonicalized to LF to tolerate Windows checkout conversion.
# A legitimate font upgrade requires reviewing the new bytes and updating this list.
KNOWN_FONTS = {
    "fa-brands-400.eot": "811e1577184066108ceca86d5a624a9012303df5206c5cb0b27d5e3fedcba5a3",
    "fa-brands-400.svg": "02340ba1789d09bcc507563c89c4a847ba537bb8132082777a50519cf74bcb5b",
    "fa-brands-400.ttf": "a0c145382acfe59767d1487d6505f349665b685c2885cd91b621d141a292b0f8",
    "fa-brands-400.woff": "e836c5d5dfc13e18170c5bddefb4fdac7c65ac0ad43a4b431acdc4a98865a260",
    "fa-brands-400.woff2": "5d9190292acdd48ba0fc35080f7e7448f3cdf0d79199a4d23f0f49b5341fdf29",
    "fa-regular-400.eot": "01f4416f5db59e2dd6b6fbd9dc32336d99db18f7eb623a49f584d04afd279473",
    "fa-regular-400.svg": "865fd23de13b1920280e182359579e09eff7a09eb24676eceec7552838559f55",
    "fa-regular-400.ttf": "f32b94a29060dfec0c797a1dfe111357eb1d682f1aca4ac4b95cf0f4c3e2bfa9",
    "fa-regular-400.woff": "8e2d96198711b63752e5f2f5987b8ffc536df6156a065bdbe28cb7231cad793d",
    "fa-regular-400.woff2": "6a8c8e9e1e7f692c21af1956de163f3d026778e6449fe93a09a671847ca1ae65",
    "fa-solid-900.eot": "32501727bb23fc77615b1ec76b5f298ec22198c0f3d6a3e7d6fe4ac3cf315db9",
    "fa-solid-900.svg": "e768539808757f7eec20f09c9ee9b36b0e6c5636e90c9ee697995cbbc6ee0555",
    "fa-solid-900.ttf": "231d68cefbc6846fb78cafca8467401a4f371e226418662e8900e3d52adc00ab",
    "fa-solid-900.woff": "352f8ead280419dd953d2bc7843386bb9430e8152c2dbd2c2d2494f6b9567ee7",
    "fa-solid-900.woff2": "7f4d3fd0a705dbf8403298aad91d5de6972e6b5d536068eba8b24954a5a0a8c7",
}
TRAILING_COMMA = re.compile(r'("(?:\\.|[^"\\])*")|,(?=\s*[}\]])')


def parse_jsonc(source: str):
    """Parse JSON with BOM, comments and trailing commas, without touching strings."""
    result = []
    i = 0
    in_string = False
    escaped = False
    while i < len(source):
        char = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
        elif char == '"':
            in_string = True
            result.append(char)
            i += 1
        elif char == "/" and nxt == "/":
            j = source.find("\n", i + 2)
            if j == -1:
                break
            result.append("\n")
            i = j + 1
        elif char == "/" and nxt == "*":
            j = source.find("*/", i + 2)
            if j == -1:
                raise ValueError("unterminated JSONC block comment")
            result.append("\n" * source[i:j + 2].count("\n"))
            i = j + 2
        else:
            result.append(char)
            i += 1
    clean = "".join(result).lstrip("\ufeff")
    clean = TRAILING_COMMA.sub(lambda match: match.group(1) or "", clean)
    return json.loads(clean)


def _read_object(path: Path):
    value = parse_jsonc(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _has_folder_open(value):
    """Find runOn in tasks and their OS overrides, including inherited commands."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "runOn" and isinstance(child, str) and child.casefold() == "folderopen":
                return True
            if _has_folder_open(child):
                return True
    elif isinstance(value, list):
        return any(_has_folder_open(child) for child in value)
    return False


def source_violations(root: Path):
    errors = []
    editor = root / ".vscode"
    configs = [editor / "tasks.json", editor / "settings.json"]
    configs.extend(
        path for path in root.rglob("*.code-workspace")
        if not ({".git", ".venv", "venv", "node_modules"} & set(path.parts))
    )
    for path in configs:
        if not path.is_file():
            continue
        document = _read_object(path)
        location = str(path.relative_to(root))
        tasks = document.get("tasks")
        if path.name == "settings.json" or path.suffix == ".code-workspace":
            settings = document if path.name == "settings.json" else document.get("settings", {})
            if not isinstance(settings, dict):
                errors.append(f"{location}: invalid settings object")
            else:
                permission = settings.get("task.allowAutomaticTasks")
                if permission is not None and permission != "off":
                    errors.append(f"{location}: automatic-task preference enabled/invalid")
                if "tasks" in settings:
                    errors.append(f"{location}: unexpected tasks in settings")
        task_config = document if path.name == "tasks.json" else tasks
        if task_config is not None and _has_folder_open(task_config):
            errors.append(f"{location}: folderOpen task (including OS overrides)")
    fonts = root / "public" / "fonts"
    if fonts.is_dir():
        for path in fonts.iterdir():
            if not path.is_file() or path.name == "README.md":
                continue
            expected = KNOWN_FONTS.get(path.name)
            data = path.read_bytes()
            if path.suffix.lower() == ".svg":
                data = data.replace(b"\r\n", b"\n")
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                errors.append(f"public/fonts/{path.name}: unreviewed/modified font asset")
    return errors


class EditorStartupGuardTests(unittest.TestCase):
    def test_repository_has_no_implicit_task_or_untrusted_font(self):
        self.assertEqual([], source_violations(ROOT))

    def test_bom_jsonc_and_os_override_cannot_hide_auto_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ".vscode" / "tasks.json"
            path.parent.mkdir()
            path.write_text(
                '\ufeff{"version":"2.0.0", // benign comment\n'
                '"tasks":[{"label":"check","command":"node",'
                '"windows":{"runOptions":{"runOn":"folderOpen",},},},],}',
                encoding="utf-8",
            )
            self.assertTrue(any("folderOpen" in issue for issue in source_violations(root)))

    def test_workspace_tasks_and_automatic_setting_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested.code-workspace").write_text(
                '{"tasks":{"tasks":[{"linux":{"runOptions":{"runOn":"folderOpen"}}}]},'
                '"settings":{"task.allowAutomaticTasks":"on"}}',
                encoding="utf-8",
            )
            issues = source_violations(root)
            self.assertEqual(2, len(issues))

    def test_top_level_os_specific_tasks_cannot_hide_auto_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ".vscode" / "tasks.json"
            path.parent.mkdir()
            path.write_text(
                '{"version":"2.0.0","tasks":[],'
                '"windows":{"tasks":[{"label":"lint","type":"shell",'
                '"command":"echo safe","runOptions":{"runOn":"folderOpen"}}]}}',
                encoding="utf-8",
            )
            self.assertTrue(any("folderOpen" in issue for issue in source_violations(root)))

    def test_comments_strings_and_trailing_commas_are_not_false_alarms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ".vscode" / "tasks.json"
            path.parent.mkdir()
            path.write_text(
                '{"tasks":[{"command":"echo // folderOpen",},],'
                '/* "runOptions": {"runOn":"folderOpen"} */}',
                encoding="utf-8",
            )
            self.assertEqual([], source_violations(root))

    def test_font_magic_prefix_cannot_bypass_hash_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fonts = root / "public" / "fonts"
            fonts.mkdir(parents=True)
            (fonts / "fa-brands-400.ttf").write_bytes(bytes((0, 1, 0, 0)) + b"var x = 1;")
            (fonts / "fa-solid-600.ttf").write_bytes(b"woff" + b"var x = 2;")
            self.assertEqual(2, len(source_violations(root)))

    def test_svg_line_endings_are_normalized_without_weakening_hash_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fonts = root / "public" / "fonts"
            fonts.mkdir(parents=True)
            original = (ROOT / "public" / "fonts" / "fa-brands-400.svg").read_bytes()
            lf = original.replace(b"\r\n", b"\n")
            (fonts / "fa-brands-400.svg").write_bytes(lf.replace(b"\n", b"\r\n"))
            self.assertIn(bytes((13, 10)), (fonts / "fa-brands-400.svg").read_bytes())
            self.assertEqual([], source_violations(root))
            font = fonts / "fa-brands-400.svg"
            font.write_bytes(font.read_bytes() + b"inert-append")
            self.assertEqual(1, len(source_violations(root)))


if __name__ == "__main__":
    unittest.main()
