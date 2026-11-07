"""run.sh の自己テスト。標準ライブラリだけで動く。

python3 -m unittest discover -s .github/audit -p 'test_*.py'
"""

import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_SH = os.path.join(HERE, "run.sh")

MODEL = "opencode-go/deepseek-v4.1-flash"
CAPACITY = (
    '429: {"type":"server_error","message":"Upstream request failed: '
    '[prefill_cap_refused] the cell is at capacity; retry shortly (after 5 s)"}'
)

FAKE_PI = """#!/bin/sh
model=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--model" ]; then
    model="$arg"
  fi
  prev="$arg"
done
echo "$model" >>"$PI_LOG"
if [ -n "${PI_ERROR:-}" ]; then
  printf '%s\\n' "$PI_ERROR" >&2
  exit "${PI_STATUS:-1}"
fi
echo "report from $model"
"""


class RunTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        bin_dir = os.path.join(self.dir, "bin")
        os.makedirs(bin_dir)
        pi = os.path.join(bin_dir, "pi")
        with open(pi, "w", encoding="utf-8") as f:
            f.write(FAKE_PI)
        os.chmod(pi, 0o755)
        self.log = os.path.join(self.dir, "pi.log")
        self.report = os.path.join(self.dir, "report.md")
        self.model_file = os.path.join(self.dir, "model.txt")
        prompt = os.path.join(self.dir, "prompt.md")
        existing = os.path.join(self.dir, "existing.json")
        for path in (prompt, existing):
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
        self.env = dict(os.environ)
        self.env.update(
            {
                "PATH": bin_dir + os.pathsep + os.environ["PATH"],
                "PI_LOG": self.log,
                "AUDIT_MODEL": MODEL,
                "AUDIT_PROMPT": prompt,
                "AUDIT_EXISTING": existing,
                "AUDIT_REPORT": self.report,
                "AUDIT_MODEL_FILE": self.model_file,
            }
        )
        self.env.pop("PI_ERROR", None)
        self.env.pop("PI_STATUS", None)

    def run_audit(self):
        return subprocess.run(
            ["bash", RUN_SH], env=self.env, capture_output=True, text=True
        )

    def calls(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log, encoding="utf-8") as f:
            return f.read().split()

    def read_model(self):
        with open(self.model_file, encoding="utf-8") as f:
            return f.read()

    def test_success(self):
        proc = self.run_audit()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.calls(), [MODEL])
        self.assertEqual(self.read_model(), MODEL + "\n")

    def test_capacity_failure(self):
        self.env["PI_ERROR"] = CAPACITY
        proc = self.run_audit()
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(self.calls(), [MODEL])
        self.assertFalse(os.path.exists(self.model_file))
        self.assertIn("429", proc.stderr)

    def test_unrelated_failure(self):
        self.env["PI_ERROR"] = "No API key found"
        proc = self.run_audit()
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(self.calls(), [MODEL])
        self.assertFalse(os.path.exists(self.model_file))
        self.assertIn("No API key found", proc.stderr)

    def test_pi_status_is_not_passed_through(self):
        # pi が 2・3 を返しても、予約コード（入力不足・制限）と混同しない。
        self.env["PI_ERROR"] = "No API key found"
        for status in (2, 3):
            with self.subTest(status=status):
                self.env["PI_STATUS"] = str(status)
                proc = self.run_audit()
                self.assertEqual(proc.returncode, 1, proc.stderr)
                self.assertEqual(self.calls(), [MODEL])
                self.assertFalse(os.path.exists(self.model_file))
                self.assertNotIn("::warning::", proc.stdout)
                os.remove(self.log)

    def test_missing_input(self):
        del self.env["AUDIT_PROMPT"]
        proc = self.run_audit()
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual(self.calls(), [])
        self.assertIn("AUDIT_PROMPT is required", proc.stderr)


if __name__ == "__main__":
    unittest.main()
