from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


REMOTE_ARTIFACT_PATTERNS = {
    "data/splits/",
    "data/teacher/",
    "data/training/",
    "data/harness/",
    "data/fewshot/",
    "data/baseline/e3-dev*/",
}


class SyncProtectionTest(unittest.TestCase):
    def test_sync_excludes_remote_generated_artifacts(self):
        sync_path = ROOT / "sync.sh"
        if not sync_path.exists():
            self.skipTest("sync.sh is a local deployment file")
        sync_text = sync_path.read_text(encoding="utf-8")

        for pattern in REMOTE_ARTIFACT_PATTERNS:
            self.assertIn(f"'{pattern}'", sync_text)

    def test_gitignore_ignores_remote_generated_artifacts(self):
        ignored = {
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertTrue(REMOTE_ARTIFACT_PATTERNS <= ignored)


if __name__ == "__main__":
    unittest.main()
