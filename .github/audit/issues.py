#!/usr/bin/env python3
"""定期監査の候補抽出と起票。標準ライブラリだけを使う。

extract: 監査レポートから候補の JSON ブロックを取り出して検証する。
file:    候補を既存 Issue との照合・重複除去・上限適用のうえ GitHub Issue として起票する。
          GitHub への書き込みはこのスクリプトだけが行う（LLM には権限を渡さない）。
"""

import json
import os
import re
import subprocess
import sys
import tempfile

START_FENCE = "```json"
END_FENCE = "```"
LOCATION_RE = re.compile(r"[^\s:]+:\d+(?:-\d+)?")
SEVERITIES = ("高", "中", "低")
MAX_CANDIDATES = 5
MAX_ISSUES_PER_KIND = 3
KINDS = ("design", "ponytail")


def fail(message):
    raise SystemExit(message)


def _text(obj, key, *, max_len=None, single_line=False):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{key}: 空でない文字列が必要")
    if max_len is not None and len(value) > max_len:
        fail(f"{key}: {max_len} 文字以内にする")
    if single_line and ("\n" in value or "\r" in value):
        fail(f"{key}: 改行を含めない")
    return value


def split_blocks(text):
    """```json の行から ``` の行までを 1 ブロックとして取り出す。

    行の途中にある三連バッククォート（JSON 文字列の中身など）は無視する。
    閉じられていないブロックは捨てる。
    """
    blocks = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if current is None:
            if stripped == START_FENCE:
                current = []
            continue
        if stripped == END_FENCE:
            blocks.append("\n".join(current))
            current = None
        else:
            current.append(line)
    return blocks


def check_location(location):
    if not isinstance(location, str) or not LOCATION_RE.fullmatch(location):
        fail(f"locations: 'パス:開始行-終了行' の形ではない: {location!r}")
    _, start, end = parse_location(location)
    if start < 1 or end < start:
        fail(f"locations: 行番号が不正: {location!r}")


def validate_candidate(candidate):
    if not isinstance(candidate, dict):
        fail("候補はオブジェクトである必要がある")
    title = _text(candidate, "title", max_len=120, single_line=True)
    severity = candidate.get("severity")
    if severity not in SEVERITIES:
        fail('severity: "高"・"中"・"低" のいずれか')
    principle = _text(candidate, "principle")
    locations = candidate.get("locations")
    if not isinstance(locations, list) or not locations:
        fail("locations: 1 件以上の配列が必要")
    for location in locations:
        check_location(location)
    problem = _text(candidate, "problem")
    fix = _text(candidate, "fix")
    acceptance = candidate.get("acceptance")
    if not isinstance(acceptance, list) or not acceptance:
        fail("acceptance: 1 件以上の配列が必要")
    if not all(isinstance(a, str) and a.strip() for a in acceptance):
        fail("acceptance: 空でない文字列の配列が必要")
    related = candidate.get("related_closed", [])
    if not isinstance(related, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in related
    ):
        fail("related_closed: 正の整数の配列")
    return {
        "title": title,
        "severity": severity,
        "principle": principle,
        "locations": list(locations),
        "problem": problem,
        "fix": fix,
        "acceptance": list(acceptance),
        "related_closed": list(related),
    }


def extract(report_path, out_path):
    with open(report_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    blocks = split_blocks(text)
    if not blocks:
        fail("レポートに ```json ブロックがない")
    try:
        data = json.loads(blocks[-1])
    except json.JSONDecodeError as e:
        fail(f"候補の JSON が読めない: {e}")
    if not isinstance(data, list):
        fail("候補は配列である必要がある")
    if len(data) > MAX_CANDIDATES:
        fail(f"候補は最大 {MAX_CANDIDATES} 件、{len(data)} 件あった")
    candidates = [validate_candidate(c) for c in data]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=1)
    print(f"候補 {len(candidates)} 件を {out_path} に書いた")


def parse_location(location):
    path, _, lines = location.rpartition(":")
    start, _, end = lines.partition("-")
    return path, int(start), int(end or start)


def overlaps(a, b):
    path_a, start_a, end_a = parse_location(a)
    path_b, start_b, end_b = parse_location(b)
    return path_a == path_b and start_a <= end_b and start_b <= end_a


def split_related_closed(numbers, states):
    """本文に書く番号と、除外した (番号, 理由) に分ける。

    誤った一文を書かないよう、closed かつ completed のものだけを本文に書く。
    """
    keep, dropped = [], []
    for number in numbers:
        state, reason = states.get(number, (None, None))
        if state == "closed" and reason == "completed":
            keep.append(number)
        elif state is None:
            dropped.append((number, "監査 Issue の一覧にない"))
        else:
            dropped.append((number, f"state {state}、state_reason {reason or 'なし'}"))
    return keep, dropped


def build_body(candidate, kind, env, related):
    lines = ["## 問題", "", candidate["problem"], "", "## 修正方針", "", candidate["fix"], ""]
    lines += ["## 受け入れ条件", ""]
    lines += [f"- [ ] {item}" for item in candidate["acceptance"]]
    lines += ["", "## 該当箇所", ""]
    lines += [f"- `{location}`" for location in candidate["locations"]]
    if related:
        lines.append("")
        lines += [
            f"#{number} は修正済みとして閉じられていますが、問題が残っています。"
            for number in related
        ]
    lines += [
        "",
        "---",
        f"定期監査（{kind}、範囲 {env['AUDIT_RANGE']}、重点 {env['AUDIT_FOCUS']}、"
        f"{candidate['principle']}、重大度 {candidate['severity']}）で見つかった項目です。"
        f"{env['AUDIT_MODEL']} による静的な確認で、ビルドや実行による確認はしていません。"
        f"行番号は {env['AUDIT_REF']} {env['AUDIT_SHA']} のものです。",
    ]
    return "\n".join(lines) + "\n"


def create_issue(title, body):
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(body)
        body_file = f.name
    try:
        proc = subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--label",
                "audit",
                "--title",
                title,
                "--body-file",
                body_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(body_file)
    url = proc.stdout.strip().splitlines()
    return url[-1] if url else "(URL 不明)"


def existing_issues(repo):
    """既存の監査 Issue を `title -> number` と `number -> (state, state_reason)` で返す。

    PR は除く。取得できなければ起票せずに終了する。
    """
    try:
        proc = subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                f"repos/{repo}/issues?labels=audit&state=all&per_page=100",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        fail(f"既存の監査 Issue を取得できない: {e}")
    decoder = json.JSONDecoder()
    text = proc.stdout
    index = 0
    titles = {}
    states = {}
    # `gh api --paginate` はページごとに 1 つの JSON 配列を並べて出す。
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            page, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as e:
            fail(f"既存 Issue の JSON が読めない: {e}")
        if not isinstance(page, list):
            fail("既存 Issue の JSON は配列である必要がある")
        for item in page:
            if not isinstance(item, dict) or "pull_request" in item:
                continue
            title, number = item.get("title"), item.get("number")
            if isinstance(title, str) and isinstance(number, int):
                titles.setdefault(title, number)
                states.setdefault(
                    number, (item.get("state"), item.get("state_reason"))
                )
    return titles, states


def write_summary(filed, skipped, excluded):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    lines = ["## 起票結果", ""]
    lines += [f"- 起票（{kind}）: {url} — {title}" for kind, title, url in filed]
    lines += [f"- 見送り（{kind}）: {title} — {reason}" for kind, title, reason in skipped]
    lines += [
        f"- related_closed から除外: #{number}（{reason}） — {title}"
        for title, number, reason in excluded
    ]
    with open(summary, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def file_issues(directory):
    env = {
        key: os.environ.get(key, "")
        for key in (
            "GH_REPO",
            "GH_TOKEN",
            "AUDIT_RANGE",
            "AUDIT_FOCUS",
            "AUDIT_REF",
            "AUDIT_SHA",
            "AUDIT_MODEL",
        )
    }
    missing = [key for key, value in env.items() if not value]
    if missing:
        fail(f"環境変数が足りない: {', '.join(missing)}")

    filed = []
    skipped = []
    excluded = []
    try:
        existing, existing_states = existing_issues(env["GH_REPO"])
        # 起票済みの候補の locations（1 件ずつ）を種類ごとに持つ。
        filed_locations = {kind: [] for kind in KINDS}
        for kind in KINDS:
            path = os.path.join(directory, f"candidates-{kind}.json")
            if not os.path.exists(path):
                continue
            kind_env = dict(env)
            model_path = os.path.join(directory, f"model-{kind}.txt")
            if os.path.exists(model_path):
                with open(model_path, encoding="utf-8") as f:
                    model = f.read().strip()
                if not re.fullmatch(r"[A-Za-z0-9._/:-]{1,100}", model):
                    fail(f"{model_path}: モデル名が不正: {model!r}")
                kind_env["AUDIT_MODEL"] = model
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                fail(f"{path} は候補の配列ではない")
            count = 0
            for item in raw:
                try:
                    candidate = validate_candidate(item)
                except SystemExit as e:
                    skipped.append((kind, "(タイトル不明)", str(e)))
                    continue
                number = existing.get(candidate["title"])
                if number is not None:
                    # 既にあるものは起票済みと同じに扱う。再実行で上限と重複除去が崩れない
                    # ように、件数に数え、locations も起票済みとして覚える。
                    count += 1
                    filed_locations[kind].extend(candidate["locations"])
                    ref = number if isinstance(number, str) else f"#{number}"
                    skipped.append(
                        (kind, candidate["title"], f"同じタイトルの Issue が既にある（{ref}）")
                    )
                    continue
                # 同じ種類の中では重複除去しない。他種類は起票済みのものとだけ照合する。
                filed_elsewhere = [
                    other
                    for other_kind, locations in filed_locations.items()
                    if other_kind != kind
                    for other in locations
                ]
                duplicate = next(
                    (
                        other
                        for other in filed_elsewhere
                        if any(overlaps(a, other) for a in candidate["locations"])
                    ),
                    None,
                )
                if duplicate is not None:
                    skipped.append(
                        (
                            kind,
                            candidate["title"],
                            f"起票済みの他の種類の候補と同じ箇所（{duplicate}）",
                        )
                    )
                    continue
                if count >= MAX_ISSUES_PER_KIND:
                    skipped.append(
                        (
                            kind,
                            candidate["title"],
                            f"1 種類につき {MAX_ISSUES_PER_KIND} 件までの上限",
                        )
                    )
                    continue
                count += 1
                related, dropped = split_related_closed(
                    candidate["related_closed"], existing_states
                )
                excluded.extend(
                    (candidate["title"], number, reason) for number, reason in dropped
                )
                url = create_issue(
                    candidate["title"], build_body(candidate, kind, kind_env, related)
                )
                print(f"起票した: {url} — {candidate['title']}")
                filed.append((kind, candidate["title"], url))
                filed_locations[kind].extend(candidate["locations"])
                # 同じ実行の中で同じタイトルを二重に起票しないよう照合表を更新する。
                existing[candidate["title"]] = url
    finally:
        for kind, title, reason in skipped:
            print(f"見送り（{kind}）: {title} — {reason}")
        print(f"起票 {len(filed)} 件、見送り {len(skipped)} 件")
        for title, number, reason in excluded:
            print(f"related_closed から除外: #{number}（{reason}） — {title}")
        write_summary(filed, skipped, excluded)


USAGE = "usage: issues.py extract <report.md> <out.json> | issues.py file <dir>"


def main(argv):
    if len(argv) == 4 and argv[1] == "extract":
        extract(argv[2], argv[3])
        return 0
    if len(argv) == 3 and argv[1] == "file":
        file_issues(argv[2])
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
