# ponytail-audit の適用範囲

この後に続く ponytail-audit の手順書に従って監査する。手順書との差分だけをここに書く。

- 静的監査のみ。ビルド・テスト・対象コードを実行しない。
- 監査するのは実行条件に書かれた scope だけである（手順書の「Audit what the user names」に当たる）。
- 報告は日本語で書く。手順書の「Very simple English」は「平易な日本語」に読み替える。
- 報告の構成は手順書に従う（What this repo does、Must fix / Should fix / Nice to have、各件の 4 項目、Verdict、Not checked）。
- 起票候補にするのは Must fix と Should fix だけ。Nice to have は候補にしない。起票のしかたは、この後に続く output.md に従う。
- 手順書の「changes nothing」に従い、リポジトリのファイルは変更しない。
