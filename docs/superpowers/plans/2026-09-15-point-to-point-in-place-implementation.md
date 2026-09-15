# PointToPoint分散転置 in-place 実装計画（実装済み）

- 日付: 2026-09-15
- 基点: `198638e`
- 状態: `execute_in_place`の実装・検証を完了。Alltoallv/P2Pのout/in-placeを実装済みで、後続未実装はFFTのみ。
- 範囲: `execute_in_place`、最小限の共通化、仕様・公開docs・既存MPI suiteの更新。
- 関連: [P2P out-of-place計画](2026-09-15-point-to-point-transpose-implementation.md)、仕様18.2/18.3。
- 本計画は設計・実装・検証の記録であり、commit、push、PRは行わない。

## 境界と公開API

追加する公開APIは次の一つだけとし、workspace/errorは共有型を使う。

```rust
pub fn execute_in_place<T>(
    &self,
    array: &mut ManyPencilArray<T, N, M>,
    workspace: &mut TransposeWorkspace<T>,
) -> Result<(), TransposeError>
where
    T: mpi::datatype::Equivalence + Copy;
```

既存のplan、workspace、errorの公開名とvariantは変えない。P2P専用workspace/error、
trait、公開storage API、unsafe、FFT、WaitAny、requestの永続保持、追加本体Vecは作らない。
active layoutはplanのsource、destinationは登録済みであることを要求する。通常の
header/descriptor/preflight errorではarrayのstate・内容とworkspaceを保持する。
MPI故障、任意panic、プロセス喪失後のglobal recoveryは保証しない。

## header・descriptor・preflight

`transpose.rs`の5語header
`[schema, combined_operation, N, M, descriptor_len]`を維持し、operation 1..=5は
変更せず、6だけをP2P in-placeに割り当てる。

```text
1 Alltoallv new       2 Alltoallv views       3 Alltoallv in-place
4 P2P new             5 P2P views             6 P2P in-place
```

`agree_execute_descriptor`をoperation 6で呼び、planの方向・変更軸・内部topology/grid、
global shape、ordered decomposition、permutation、exact extra shape、`T`の
`type_name`/size/alignment/`Equivalence::Out`型名を既存のexact比較へ渡す。header不一致
ならdescriptor、axis subcommunicator、payloadへ進まない。従ってnew、P2P views、
Alltoallv views/in-place、P2P in-placeのrank間取り違えを全rankで拒否できる。

descriptorはPoisoned arrayでも`active_view`/`active_index`前に、`array.extra_shape()`と
plan metadataから作る。descriptor後のrank-local errorは即returnせず、source Cartesian
communicatorの`collective_valid`を完了してから全rankがreturnする。active state、source
layout、destination登録、source/destinationのchecked必要prefix、workspaceの初期化済み
`len`、count/displacement、総量、region offsetを検査する。sourceとdestinationの
local_lenは一致すると仮定しない。

## 共通処理とP2P transport

`alltoallv_transpose.rs`のin-place preflightと、post-transferのguard/unpack/commitを、
二consumerだけが使う小さいcrate-private `transpose.rs` helperへ抽出する。既存の
`ManyPencilArray` hooks、`PreparedExchange`、`prepare_common`、`pack_source`、
`unpack_destination`を使い、新しい公開抽象は導入しない。

- `prepare_in_place`相当はactive sourceを検査し、destination indexとsource/destination
  別のchecked prefix lengthを求め、既存のexchange preparationへ渡す。
- `finish_in_place`相当は転送完了後だけ`begin_in_place_write`し、destination必要prefix
  全体へunpackしてから`commit(destination_index)`する。prefix外のtailは触らない。
- 失敗したpreflightではguardを作らない。post-write callbackはなく、`T: Copy`と既存
  `LayoutWriteGuard`のpanic時Poisoned保証を使う。

P2Pの既存`scope`部分を、views/in-placeから呼べる
`execute_point_to_point_exchange`相当の最小private helperへ切り出す。helperはsource
`Pencil`とsource sliceからworkspaceへpackし、request完了までを担当する。

1. `Vec<Request>`のsend/receive slotを`try_reserve`し、pack・最初のpostより前に
   reserve成否を全Cartesian rankで合意する。失敗時はworkspaceもarrayも触らない。
2. workspace backing Vecをresize/reallocせず、既存prefixを使う。
3. 固定`POINT_TO_POINT_RESERVED_TAG`（`0x5054`）と内部変更軸subcommunicatorを使い、
   同じcontextで未完了転置を重ねない。world rankやdynamic tagは使わない。
4. peer順でnonzeroの全Irecvをpostし、次に全Isendをpostする。self peerもcountが
   非zeroなら通常通り扱い、片方向empty、zero-extraではcountが0のpeerだけrequestを省略する。
5. receive/sendの全requestを`wait_without_status`で完了させ、segmentとworkspaceを保持
   したままscopeを終了する。未完了requestをforgetしない。

in-placeではpack・Irecv・Isend・wait中にarrayを書き換えない。全requestとborrowが
scope終了で解放された後だけ、共通finish helperでPoisoned、全destination unpack、commit
の順に進む。

## 実装順序・文書

1. `transpose.rs`へoperation 6と二つの小さいprivate in-place helperを追加する。
2. `point_to_point_transpose.rs`へAPIとoperation 6を追加し、transport/finish helperを使う。
3. `alltoallv_transpose.rs`を共通helperへ最小接続し、既存operation 1..=5の挙動を保つ。
4. `crates/pencil-array/tests/alltoallv_transpose.rs`へ、既存fixture・success helper・
   一度の`mpi::initialize()`を共用してテストを追加する。
5. `README.md`、`crates/pencil-array/src/lib.rs`、P2P公開docs、既存one-rank doctest、
   CI suite説明を、両方式のout-of-place/in-place対応へ最小更新する。
6. FFTだけを後続に残し、new test binary・fixture大量複製・監査一括修正はしない。

## テスト計画

既存の`alltoallv_transpose.rs`の2D/3D値生成、独立physical expected、workspace helper、
`ManyPencilArray` fixtureを再利用する。既存3D physical fixtureは必要ならgeneric化して、
u64/f64の両方を複製なしで通す。

- 1/4/6 rank、6 rankの非正方`[2, 3]`、world rankを反転したcommunicatorを使う。
- 2D/3Dの非自明permutation、uneven partition、self peerをu64/f64で検証する。
- P2P in-place、Alltoallv in-place、P2P viewsを完全一致させ、独立raw oracle、往復、
  source期待値も確認する。
- source/destination local_len相違、余剰tail、source empty→destination nonempty、
  逆方向、zero-extra、片方向empty、全zero payloadを含める。
- 同一workspaceでpreflight失敗後に成功し、同一plan/workspaceを繰返し再利用する。
  通常Errではarray state/内容/workspaceを保持し、成功後のtailも保持する。
- rank-local source layout不一致、destination未登録、Poisoned、workspace不足を一rankだけ
  発生させ、payload/guard前の全rank Errとtimeout内完了を確認する。
- rank-local extra shape、u64/f64 type、source/destination方向差をdescriptorで拒否する。
- P2P in-place対Alltoallv new/views/in-placeおよびP2P new/viewsを混在させ、operation 6の
  headerでpayload前に全rankが拒否する。constructor対executeも既存混在テストへ足す。
- test専用panic hookやunsafe状態注入は作らず、panic後のPoisoned契約は既存guardに委ねる。

実装時の確認コマンドは次とする。

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked
for n in 1 4 6; do
  timeout --foreground 120s mpiexec --oversubscribe -n "$n" \
    cargo test -p pencil-array --test alltoallv_transpose --locked -- \
    --nocapture --test-threads=1 || exit 1
done
cargo doc --workspace --no-deps --locked
cargo test --workspace --doc --locked -- --show-output
```

この計画の実装段階でもcommit/pushは行わない。