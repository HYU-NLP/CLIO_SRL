"""Run the real splitter with synthetic records; no model dependencies."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get("CLIO_SPLITTER", ROOT / "data_split.py"))


class SplitTests(unittest.TestCase):
    def run_split(self, n):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        data_dir = folder / "data"
        data_dir.mkdir()
        records = [{"id": i, "line": f"한국어 문장 {i}", "predicate": {"lemma": "검증"},
                    "roles": [i, None, {"text": "내용"}]} for i in range(n)]
        input_file = data_dir / "CLIO SRL dataset_ver2.json"
        input_file.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        original_bytes = input_file.read_bytes()
        proc = subprocess.run([sys.executable, "-B", str(SCRIPT)], cwd=folder, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(input_file.read_bytes(), original_bytes)
        if os.environ.get("CLIO_EVIDENCE_DIR"):
            dest = Path(os.environ["CLIO_EVIDENCE_DIR"]) / self._testMethodName / str(n)
            shutil.copytree(data_dir, dest / "data", dirs_exist_ok=True)
            (dest / "execution.json").write_text(json.dumps({"command": proc.args,
                "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}, indent=2), encoding="utf-8")
        return data_dir, records

    def splits(self, folder):
        # Permit the old filename only in invariant tests, not contract tests.
        val = folder / "clio_val.json"
        if not val.exists():
            val = folder / "clio_valid.json"
        return [json.loads(p.read_text(encoding="utf-8")) for p in
                (folder / "clio_train.json", val, folder / "clio_test.json")]

    def test_expected_filenames(self):
        folder, _ = self.run_split(20)
        self.assertEqual({p.name for p in folder.glob("clio_*.json")},
                         {"clio_train.json", "clio_val.json", "clio_test.json"})

    def test_representative_counts(self):
        folder, _ = self.run_split(20)
        self.assertEqual([len(x) for x in self.splits(folder)], [16, 2, 2])

    def test_disjoint_membership_and_contents(self):
        folder, records = self.run_split(20)
        splits = self.splits(folder)
        ids = [{r["id"] for r in part} for part in splits]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        combined = [r for part in splits for r in part]
        self.assertEqual(len(combined), len({r["id"] for r in combined}))
        self.assertEqual(sorted(combined, key=lambda r: r["id"]), records)

    def test_small_dataset_rounding(self):
        for n in range(11):
            with self.subTest(n=n):
                folder, _ = self.run_split(n)
                train, val = int(n * 0.8), int(n * 0.1)
                self.assertEqual([len(x) for x in self.splits(folder)], [train, val, n - train - val])

    def test_training_default_and_readme(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--val_file_path")
        default = next(ast.literal_eval(k.value) for k in call.keywords if k.arg == "default")
        self.assertIn(Path(default).name, (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertNotIn("--val_file_path", (ROOT / "train.sh").read_text())
        folder, _ = self.run_split(20)
        # Perform the same file lookup as train.py, without importing training.
        with (folder.parent / default).open(encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), 2)

    def test_json_encoding_and_format(self):
        folder, _ = self.run_split(20)
        for path in folder.glob("clio_*.json"):
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text, json.dumps(json.loads(text), ensure_ascii=False, indent=2))
            self.assertIn("한국어", text)


if __name__ == "__main__":
    unittest.main()
