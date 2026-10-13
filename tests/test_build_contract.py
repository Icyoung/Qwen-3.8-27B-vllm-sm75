"""CPU-only native source layout and speculative overlay installation."""

import ast, runpy, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BuildContract(unittest.TestCase):
    def test_no_container_deployment_inputs(self):
        self.assertFalse((ROOT / "docker").exists())
        self.assertTrue((ROOT / "bootstrap.sh").exists())
        self.assertTrue((ROOT / "runtime/native-lock.json").exists())

    def test_native_helper_syntax(self):
        for p in (ROOT / "runtime").rglob("*.py"):
            ast.parse(p.read_text(), filename=str(p))

    def test_speculative_install_has_nine_reviewed_files(self):
        install = runpy.run_path(str(ROOT / "runtime/helpers/install_speculative.py"))[
            "install"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "site/vllm"
            evidence = Path(tmp) / "evidence.json"
            self.assertEqual(
                len(install(ROOT / "runtime/speculative", package, evidence)), 9
            )
            self.assertTrue(evidence.exists())


if __name__ == "__main__":
    unittest.main()
