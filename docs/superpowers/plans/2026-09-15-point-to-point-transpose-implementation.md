# Point-toPoint方式のout-of-place分散転置 実装計画（実装済み）

- 日付: 2026-09-15
- 基点: `159a16d`
- 状態: `PointToPointTransposePlan` のcheckedなout-of-place実装と検証を完了。
  P2P in-place、FFT、`WaitAny`による重畳、性能最適化は未実装。
- 関連仕様: [Rust移植設計仕様](../specs/2026-09-11-pencil-arrays-rust-port-design.md)
- 実装: [`point_to_point_transpose.rs`](../../../crates/pencil-array/src/point_to_point_transpose.rs)
- 範囲: この計画は上記実装の設計・検証記録であり、commit・push・PRは行わない。

## APIと共有境界

公開するのは次だけである。`execute` wrapperや通信方式を選ぶ公開enumは追加しない。

```rust
pub struct PointToPointTransposePlan<const N: usize, const M: usize>;

impl<const N: usize, const M: usize> PointToPointTransposePlan<N, M> {
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, TransposeError>;

    pub fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<TransposeWorkspaceRequirements, TransposeError>;

    pub fn execute_views<T>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: mpi::datatype::Equivalence + Copy;
}
```

`TransposeError`、`TransposeWorkspace<T>`、`TransposeWorkspaceRequirements`を
通信方式に依存しない実体として`crates/pencil-array/src/transpose.rs`へ移す。
現在の`AllToAllvTransposeError`、`AllToAllvTransposeWorkspace<T>`、
`AllToAllvTransposeWorkspaceRequirements`は、同じ型を`lib.rs`から次の名前で
re-exportする。

```rust
pub use transpose::TransposeError;
pub use transpose::TransposeError as AllToAllvTransposeError;
pub use transpose::TransposeWorkspace;
pub use transpose::TransposeWorkspace as AllToAllvTransposeWorkspace;
pub use transpose::TransposeWorkspaceRequirements;
pub use transpose::TransposeWorkspaceRequirements
    as AllToAllvTransposeWorkspaceRequirements;
```

既存Alltoallvのシグネチャ、エラーvariant、`len`によるworkspace検査、
`Poisoned`を含むin-placeの状態遷移は変えない。既存の
`crates/pencil-array/src/alltoallv_transpose.rs`は`TransposePlanCore`を使う
Alltoallv transportとして残し、`crates/pencil-array/src/point_to_point_transpose.rs`
を追加する。`transpose.rs`のprivate coreは、source/destination、変更軸、peerごとの
交差region、spatial長と累積displacement、canonical descriptorを一つだけ保持する。
`PeerMetadata`、checkedなcount/displacement/総量/offset準備、descriptor/header合意、
`pack_source`、`unpack_destination`、region mappingをここで共有する。
`PreparedExchange`も通信方式非依存のprivate準備値として共有し、Alltoallvだけが
`Partition`用displacementを使い、P2Pは同じcountとsegment位置を使う。trait、公開
strategy framework、P2P専用workspace、metadata/packの複製は作らない。

## collective protocol

privateな`CommunicationMode`（`AllToAllv`/`PointToPoint`）と組み合わせ済みの操作コードだけを
追加する。両方式の`new`と実行は、source topologyの全Cartesian communicatorで
次の5語の固定headerを最初に比較する。

```text
[schema, combined_operation, N, M, descriptor_len]
```

既存Alltoallvの`new`、`execute_views`、`execute_in_place`はそれぞれ1、2、3を維持し、
P2Pの`new`、`execute_views`は4、5を使う。方式と操作は`combined_operation`一語で
区別するため余分なheader wordは追加しない。header不一致なら、可変descriptor比較や
変更軸subcommunicatorへ進まず全rankがreturnするため、方式、方向、`new`対実行、
views対Alltoallv in-placeの取り違えをpeer通信前に拒否できる。canonical descriptorには
同じcombined operation（必要な方式固有の固定tagを含む）を保持する。

header後は既存方式と同じく、fallibleなdescriptor準備成功を全rankで合意し、
`u64` native wordのmin/max reduction二回でexact descriptorを比較する。descriptor
には方式、操作、source-to-destination方向、変更軸、grid、global shape、ordered
decomposition、permutationを含め、実行時にはexactなextra shape（rankとextents）、
`type_name::<T>()`、size、alignment、`Equivalence::Out`の型名を連結する。
型記述は誤用検出であり、同じ`T`・正しい`Equivalence`・同じcommunicatorと
collective順序を使う契約を置き換えない。

P2P実行のlocal preflightはdescriptor合意後、source topology全体のvalidity
all-reduce前に完了させる。view layout、extra shape、初期化済みworkspaceの
`len`、source/destination必要長、peer count、displacement、count+displacement、
総量、checked offsetを既存Alltoallvと同じ検査にかける。失敗rankがあっても
その場でreturnせず、全rankがvalidity all-reduceを完了する。さらにP2P request数と
`Vec<Request>`のcapacityをcheckedに準備し、追加のsegment-holder Vecは作らず、どの通常の
`Err`もpack前に返せる順序にする。workspaceのbacking `Vec`はresize/reallocしない。

## P2P交換

データ通信には`self.source.topology().subcommunicator(changed_topology_axis)`を
使う。world rankやCartesian座標を混同せず、既存peer metadataの
`peer_rank`をそのsubcommunicatorのrankとして`process_at_rank`へ渡す。タグは
内部context専用の固定値 `POINT_TO_POINT_RESERVED_TAG: mpi::Tag = 0x5054` とする。
`MpiTopology`が所有する内部Cartesian/subcommunicator contextが利用者通信と隔離
する。同じ内部contextで未完了の転置を重ねないことを契約とし、公開実行は同期的に
完了してreturnするため、固定tagにplan idや動的tagを足さない。

pack/unpackはAlltoallvと同じく、peer rank順、extra row-major順、logical spatial
row-major順で、同じsend/receive region mappingを使う。各peerのsend/receive
segmentはworkspaceの既存prefix内に置く。pack後にsend segmentをimmutable sliceとして
借用し、receive segmentはcheckedな連続prefixを`split_at_mut`で直接postして、追加の
holder Vec、重複mutable borrow、`unsafe`を作らない。zero-count peerはsendとreceiveを個別にpostしなくて
よい。全rankのdescriptor、preflight、準備合意は必ず行う。self peerも特別扱いせず、
分離されたsend/receive bufferへ同じsubcommunicator rankとtagでIrecv/Isendする。

実行本体は次の一経路だけにする。

1. 全ての通常検査、request Vecのcapacity確保、送受信領域の準備を完了する。
2. `mpi::request::scope`へ入り、`Vec<Request>`の`try_reserve`を最初のpost前に
   行う。reserve成功をsource topology全体のvalidity all-reduceで合意し、失敗なら
   どのrankもpack/postせず`PreparationFailed`（他rankはcollective failure）を返す。
3. sourceを共有`pack_source`でsend bufferへpackする。
4. 非zero receive segmentの全`Irecv`を先にpostする。
5. 非zero send segmentの全`Isend`を次にpostする。
6. vector内の全requestを`wait_without_status`で完了させてからscopeを抜ける。
   wait前にbufferを変更・dropせず、未完了requestを`forget`してreturnしない。
7. requestと借用segmentを解放してから、共有`unpack_destination`でdestinationへ
   書き、returnする。

rsmpi 0.8.2の`Request`はbufferをborrowし、dropだけではpanicするため、requestを
plan/workspaceへ保存しない。scope外でworkspaceを使うのは全requestとsegmentの
lifetime終了後だけにする。`WaitAny`、`RequestCollection`、unpack重畳、raw FFI、
`unsafe`、`MaybeUninit`、実行中の`T: Default`/`Clone` callbackは使わない。
MPI障害、任意panic、プロセス喪失後のglobal recoveryは既存Alltoallvと同じく保証
しない。Alltoallv側の`Partition`、一回の`all_to_all_varcount_into`、全zero時の
既存dummy fallback、in-placeのguard処理は変更しない。

## 実装と検証の対象

1. `docs/superpowers/specs/2026-09-11-pencil-arrays-rust-port-design.md`へ
   section 18.3を追加し、専用API、方式付き固定header、exact descriptor、内部
   subcommunicator/tag、非重畳契約、preflight、request lifetimeを規定する。
2. `transpose.rs`へ共有型・core・helperを移し、`alltoallv_transpose.rs`の既存
   transportを最小差分で接続する。`lib.rs`へcanonical名、旧Alltoallv alias、
   `PointToPointTransposePlan`をre-exportする。新しい依存は追加しない。
3. 既存の一つのMPI integration binary
   `crates/pencil-array/tests/alltoallv_transpose.rs`へP2P成功・失敗ケースを追加し、
   既存の`fill_2d`、`fill_3d_physical`、独立physical expected、topology fixtureを
   共用する。新しいtest binaryや二重のMPI initializeは作らない。既存Alltoallv
   in-place/Poisonedテストはそのまま回帰させる。

成功ケースは1/4/6 rank、2D/3D、6 rankの非正方grid `[2, 3]`と逆順rank communicator、
任意の非自明permutation、extra dimensions、uneven/empty partition、self peer、
zero payloadを含める。P2P結果を同じ入力から得たAlltoallv結果と完全一致させ、独立
raw physical oracle、逆方向round-trip、out-of-place source保持も確認する。

失敗ケースはsize>1でrank 0だけworkspace/layout/extra shape/plan方向/`T`型/方式を
変える。`new`対`execute_views`、P2P views対Alltoallv views/in-placeも混在させ、
全rankがpeer通信前に同じ失敗経路へ到達することをtimeout付きで確認する。通常失敗後のplan/workspace
再利用、異なる転置を固定tagで連続実行した際のtag残留なし、1 rankのself/zero
segmentを確認する。大payloadは現状の実測で非zero peerごとに `128 * 512` 個の
`u64`、すなわち `65,536 * 8 = 524,288` bytes（512 KiB）である。これは全Irecv先行の
経路がblocking sendや部分postに退行しないことを検出する実行ケースであり、MPI実装が
eagerかrendezvousか、またはどの閾値を使うかは規定しない（全実装でrendezvousになるとは
主張しない）。post順そのものはrsmpiにhookがないため、コード上の固定順序もレビュー
対象とする。

## 採用事項（短縮報告）

- **API**: `PointToPointTransposePlan::{new, workspace_requirements, execute_views}`、`T: Equivalence + Copy`、共有`TransposeWorkspace`。P2P in-place/FFT/WaitAnyは延期。
- **private共有境界**: `transpose.rs`の`TransposePlanCore`、peer region、checked preparation、header/descriptor、pack/unpack。旧Alltoallv 3型はcanonical型のre-export alias。
- **rsmpi API**: 内部`CartesianCommunicator::process_at_rank`、`Destination::immediate_send_with_tag`、`Source::immediate_receive_into_with_tag`、`mpi::request::scope`、`Request::wait_without_status`。
- **lifetime/失敗**: segmentとworkspaceをwait完了まで保持し、requestはlocal scope限定。全reserveをpost/pack前に全Cartesian rankで合意し、失敗時はrequestなしでreturnする。
- **テストコマンド**:
  `cargo fmt --all -- --check && cargo clippy --workspace --all-targets --locked -- -D warnings && cargo test --workspace --lib --locked`
  および `for n in 1 4 6; do timeout --foreground 120s mpiexec --oversubscribe -n "$n" cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1 || exit 1; done`、`cargo test --workspace --doc --locked -- --show-output`。
