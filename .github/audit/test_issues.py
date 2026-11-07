"""issues.py の自己テスト。標準ライブラリだけで動く。

python3 -m unittest discover -s .github/audit -p 'test_*.py'
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ISSUES_PY = os.path.join(HERE, "issues.py")

FAKE_GH = """#!/bin/sh
echo "=== gh" >>"$GH_LOG"
printf '%s\\n' "$@" >>"$GH_LOG"
if [ "$1" = "api" ]; then
  if [ -n "${GH_FAIL_API:-}" ]; then
    echo "api failed" >&2
    exit 1
  fi
  cat "$GH_EXISTING"
  exit 0
fi
n=$(cat "$GH_COUNT" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" >"$GH_COUNT"
prev=""
for arg in "$@"; do
  if [ "$prev" = "--body-file" ]; then
    cp "$arg" "$GH_BODIES/$n.md"
  fi
  prev="$arg"
done
echo "https://example.invalid/issues/$n"
"""


def candidate(**overrides):
    item = {
        "title": "共通化する",
        "severity": "中",
        "principle": "DRY",
        "locations": ["crates/a/src/lib.rs:10-20"],
        "problem": "同じ処理がある",
        "fix": "共通化する",
        "acceptance": ["共通関数を使う"],
    }
    item.update(overrides)
    return item


def report(*candidates):
    return "# 監査レポート\n\n本文\n\n```json\n" + json.dumps(
        list(candidates), ensure_ascii=False
    ) + "\n```\n"


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def run_extract(self, text):
        report_path = os.path.join(self.dir, "report.md")
        out_path = os.path.join(self.dir, "out.json")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
        proc = subprocess.run(
            [sys.executable, ISSUES_PY, "extract", report_path, out_path],
            capture_output=True,
            text=True,
        )
        return proc, out_path

    def test_valid_report(self):
        proc, out_path = self.run_extract(report() + "\n" + report(candidate()))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["title"], "共通化する")

    def test_empty_array_is_valid(self):
        proc, out_path = self.run_extract(report())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [])

    def test_five_candidates_is_valid(self):
        items = [candidate(title=f"直す {i}") for i in range(5)]
        proc, _ = self.run_extract(report(*items))
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_code_fence_inside_json_string(self):
        proc, out_path = self.run_extract(
            report(candidate(problem="```rust\nfn a() {}\n```"))
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["problem"], "```rust\nfn a() {}\n```")

    def test_last_block_wins(self):
        first = report(candidate(title="前のブロック"))
        second = report(candidate(title="後のブロック"))
        proc, out_path = self.run_extract(first + "\n途中の説明\n\n" + second)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["title"], "後のブロック")

    def test_missing_block(self):
        proc, _ = self.run_extract("# レポート\n\n```json\n")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("```json", proc.stderr)

    def test_broken_json(self):
        proc, _ = self.run_extract("```json\n[{\"title\": ]\n```\n")
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(proc.stderr)

    def test_too_many_candidates(self):
        items = [candidate(title=f"直す {i}") for i in range(6)]
        proc, _ = self.run_extract(report(*items))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("最大 5 件", proc.stderr)

    def test_type_violations(self):
        cases = {
            "title が空": candidate(title=""),
            "title が長すぎる": candidate(title="あ" * 121),
            "title に改行": candidate(title="改行\nあり"),
            "severity が不正": candidate(severity="critical"),
            "principle が空": candidate(principle=""),
            "problem が空": candidate(problem=""),
            "fix が空": candidate(fix=""),
            "locations が空": candidate(locations=[]),
            "locations が文字列でない": candidate(locations=[123]),
            "locations の書式が違う": candidate(locations=["crates/a.rs"]),
            "locations に空白": candidate(locations=["crates/a b.rs:1-2"]),
            "locations の末尾に改行": candidate(locations=["crates/a.rs:1-2\n"]),
            "locations の開始行が 0": candidate(locations=["crates/a.rs:0"]),
            "locations の終了が開始より前": candidate(locations=["crates/a.rs:20-10"]),
            "acceptance が空": candidate(acceptance=[]),
            "acceptance に空文字": candidate(acceptance=["ok", ""]),
            "related_closed が数値でない": candidate(related_closed=["3"]),
        }
        for name, item in cases.items():
            with self.subTest(name):
                proc, _ = self.run_extract(report(item))
                self.assertEqual(proc.returncode, 1, proc.stdout)
                self.assertTrue(proc.stderr)


class FileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.bin = os.path.join(self.dir, "bin")
        self.bodies = os.path.join(self.dir, "bodies")
        self.candidates = os.path.join(self.dir, "candidates")
        for path in (self.bin, self.bodies, self.candidates):
            os.makedirs(path)
        gh = os.path.join(self.bin, "gh")
        with open(gh, "w", encoding="utf-8") as f:
            f.write(FAKE_GH)
        os.chmod(gh, 0o755)
        self.log = os.path.join(self.dir, "gh.log")
        self.count = os.path.join(self.dir, "gh.count")
        self.summary = os.path.join(self.dir, "summary.md")
        self.existing = os.path.join(self.dir, "existing.json")
        self.write_existing([])
        self.env = dict(os.environ)
        self.env.update(
            {
                "PATH": self.bin + os.pathsep + os.environ["PATH"],
                "GH_LOG": self.log,
                "GH_COUNT": self.count,
                "GH_BODIES": self.bodies,
                "GH_EXISTING": self.existing,
                "GH_TOKEN": "fake",
                "GH_REPO": "owner/repo",
                "AUDIT_RANGE": "array",
                "AUDIT_FOCUS": "DRY",
                "AUDIT_REF": "dev",
                "AUDIT_SHA": "abc1234",
                "AUDIT_MODEL": "test/model",
                "GITHUB_STEP_SUMMARY": self.summary,
            }
        )

    def write_existing(self, items):
        with open(self.existing, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    def write(self, kind, items):
        path = os.path.join(self.candidates, f"candidates-{kind}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    def write_model(self, kind, text):
        path = os.path.join(self.candidates, f"model-{kind}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def run_file(self):
        return subprocess.run(
            [sys.executable, ISSUES_PY, "file", self.candidates],
            env=self.env,
            capture_output=True,
            text=True,
        )

    def bodies_written(self):
        names = sorted(os.listdir(self.bodies), key=lambda n: int(n.split(".")[0]))
        return [pathlib.Path(self.bodies, n).read_text(encoding="utf-8") for n in names]

    def invocations(self):
        with open(self.log, encoding="utf-8") as f:
            return [call.strip().splitlines() for call in f.read().split("=== gh\n") if call.strip()]

    def create_invocations(self):
        return [args for args in self.invocations() if "create" in args]

    def titles(self):
        return [args[args.index("--title") + 1] for args in self.create_invocations()]

    def test_cross_kind_duplicate_is_dropped(self):
        self.write(
            "design",
            [
                candidate(title="design 1", locations=["crates/a.rs:10-20"]),
            ],
        )
        self.write(
            "ponytail",
            [
                candidate(title="重複", locations=["crates/a.rs:15-25"]),
                candidate(title="別ファイル", locations=["crates/b.rs:1-3"]),
            ],
        )
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("重複", proc.stdout)
        self.assertEqual(len(self.bodies_written()), 2)
        self.assertNotIn("重複", "\n".join(self.bodies_written()))

    def test_cap_and_body(self):
        self.write(
            "design",
            [candidate(title=f"設計 {i}", locations=[f"crates/d{i}.rs:1-2"]) for i in range(5)],
        )
        self.write(
            "ponytail",
            [candidate(title=f"無駄 {i}", locations=[f"crates/p{i}.rs:1-2"]) for i in range(4)],
        )
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        bodies = self.bodies_written()
        self.assertEqual(len(bodies), 6)
        titles = self.titles()
        self.assertEqual(
            sorted(titles),
            sorted(["設計 0", "設計 1", "設計 2", "無駄 0", "無駄 1", "無駄 2"]),
        )

        body = bodies[0]
        self.assertIn("## 問題\n\n同じ処理がある", body)
        self.assertIn("## 修正方針\n\n共通化する", body)
        self.assertIn("## 受け入れ条件\n\n- [ ] 共通関数を使う", body)
        self.assertIn("## 該当箇所\n\n- `crates/d0.rs:1-2`", body)
        self.assertIn(
            "定期監査（design、範囲 array、重点 DRY、DRY、重大度 中）で見つかった項目です。"
            "test/model による静的な確認で、ビルドや実行による確認はしていません。"
            "行番号は dev abc1234 のものです。",
            body,
        )

        for args in self.create_invocations():
            self.assertIn("--label", args)
            self.assertEqual(args[args.index("--label") + 1], "audit")
            self.assertIn("issue", args)
            self.assertIn("create", args)

        with open(self.summary, encoding="utf-8") as f:
            summary = f.read()
        self.assertIn("https://example.invalid/issues/1", summary)
        self.assertIn("上限", summary)

    def test_same_kind_overlap_is_not_deduplicated(self):
        self.write(
            "design",
            [
                candidate(title="同じ種類 1", locations=["crates/a.rs:10-20"]),
                candidate(title="同じ種類 2", locations=["crates/a.rs:15-25"]),
            ],
        )
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sorted(self.titles()), ["同じ種類 1", "同じ種類 2"])

    def test_existing_titles_count_toward_cap_and_block_overlap(self):
        """再実行: 既存タイトルで見送った design も起票済みと同じに扱う。"""
        with open(self.existing, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"number": 1, "title": "設計 0", "state": "open"},
                    {"number": 2, "title": "設計 1", "state": "open"},
                    {"number": 3, "title": "設計 2", "state": "open"},
                ],
                f,
                ensure_ascii=False,
            )
        self.write(
            "design",
            [candidate(title=f"設計 {i}", locations=[f"crates/d{i}.rs:1-2"]) for i in range(5)],
        )
        self.write(
            "ponytail",
            [
                candidate(title=f"前回の設計 {i} と重なる", locations=[f"crates/d{i}.rs:1-2"])
                for i in range(3)
            ],
        )
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.titles(), [])
        with open(self.summary, encoding="utf-8") as f:
            summary = f.read()
        self.assertIn("設計 3 — 1 種類につき 3 件までの上限", summary)
        self.assertIn("設計 4 — 1 種類につき 3 件までの上限", summary)
        self.assertIn("前回の設計 0 と重なる — 起票済みの他の種類の候補と同じ箇所", summary)

    def test_same_title_in_one_run_is_filed_once(self):
        self.write("design", [candidate(title="同じ題", locations=["crates/a.rs:1-2"])])
        self.write("ponytail", [candidate(title="同じ題", locations=["crates/b.rs:1-2"])])
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.titles(), ["同じ題"])
        self.assertEqual(len(self.bodies_written()), 1)

    def test_unfiled_design_does_not_block_ponytail(self):
        self.write(
            "design",
            [candidate(title=f"設計 {i}", locations=[f"crates/d{i}.rs:1-2"]) for i in range(4)],
        )
        self.write(
            "ponytail",
            [candidate(title="4 件目と重なる", locations=["crates/d3.rs:1-2"])],
        )
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("4 件目と重なる", self.titles())

    def test_existing_title_is_not_filed(self):
        with open(self.existing, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"number": 12, "title": "共通化する", "state": "closed", "body": ""},
                    {"number": 13, "title": "PR の題", "pull_request": {}},
                ],
                f,
                ensure_ascii=False,
            )
        self.write(
            "ponytail",
            [candidate(title="共通化する"), candidate(title="PR の題")],
        )
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.titles(), ["PR の題"])
        with open(self.summary, encoding="utf-8") as f:
            self.assertIn("同じタイトルの Issue が既にある（#12）", f.read())

    def test_existing_issues_failure_stops_filing(self):
        self.env["GH_FAIL_API"] = "1"
        self.write("design", [candidate(title="起票しない")])
        proc = self.run_file()
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(self.bodies_written(), [])

    def test_related_closed_line(self):
        self.write_existing(
            [
                {
                    "number": 42,
                    "title": "前回の再発",
                    "state": "closed",
                    "state_reason": "completed",
                }
            ]
        )
        self.write(
            "ponytail",
            [candidate(title="再発", related_closed=[42])],
        )
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(
            "#42 は修正済みとして閉じられていますが、問題が残っています。",
            self.bodies_written()[0],
        )

    def test_related_closed_only_lists_closed_completed(self):
        """open の Issue を「修正済みとして閉じられている」と書かない。"""
        self.write_existing(
            [
                {"number": 63, "title": "まだ開いている", "state": "open"},
                {
                    "number": 42,
                    "title": "閉じた",
                    "state": "closed",
                    "state_reason": "completed",
                },
            ]
        )
        self.write("ponytail", [candidate(title="再発", related_closed=[63, 42])])
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = self.bodies_written()[0]
        self.assertIn("#42 は修正済みとして", body)
        self.assertNotIn("#63", body)
        with open(self.summary, encoding="utf-8") as f:
            summary = f.read()
        self.assertIn("related_closed から除外: #63（state open", summary)

    def test_related_closed_unknown_number_is_not_listed(self):
        self.write("ponytail", [candidate(title="再発", related_closed=[999])])
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("#999", self.bodies_written()[0])
        with open(self.summary, encoding="utf-8") as f:
            self.assertIn("related_closed から除外: #999（監査 Issue の一覧にない）", f.read())

    def test_single_kind(self):
        self.write("ponytail", [candidate(title="一つだけ")])
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        bodies = self.bodies_written()
        self.assertEqual(len(bodies), 1)
        self.assertIn("（ponytail、", bodies[0])

    def test_model_file_overrides_footer(self):
        self.write("design", [candidate(title="モデルを書く")])
        self.write_model("design", "commandcode/deepseek/deepseek-v4.1-flash\n")
        proc = self.run_file()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = self.bodies_written()[0]
        self.assertIn(
            "commandcode/deepseek/deepseek-v4.1-flash による静的な確認", body
        )
        self.assertNotIn("test/model", body)

    def test_invalid_model_file_stops_filing(self):
        self.write("design", [candidate(title="起票しない")])
        self.write_model("design", "commandcode model\n")
        proc = self.run_file()
        self.assertEqual(proc.returncode, 1)
        self.assertIn("モデル名が不正", proc.stderr)
        self.assertEqual(self.bodies_written(), [])

    def test_missing_env(self):
        del self.env["AUDIT_MODEL"]
        proc = self.run_file()
        self.assertEqual(proc.returncode, 1)
        self.assertIn("AUDIT_MODEL", proc.stderr)


if __name__ == "__main__":
    unittest.main()
