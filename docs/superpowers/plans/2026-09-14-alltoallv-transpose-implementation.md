# Alltoallv out-of-place distributed transpose implementation plan

- Date: 2026-09-14
- Base: `e192d53`
- Status: Alltoallv out-of-placeと後続の分散in-place実装・検証を完了。P2P、FFTは次段階。
- Scope: Milestone 5 の最初の小さなPR。`MPI_Alltoallv`を使うout-of-place分散転置だけを実装する。

## 境界と採用方針

既存の`Pencil`、`PencilArray`/view、`ManyPencilArray`、および完成済みの
`LocalTransposePlan`をそのまま利用する。新しい公開の総合`TransposePlan`や
`TransposeMethod`は作らない。

sourceとdestinationは次を満たす場合だけ受け付ける。

- 同じ`MpiTopology`（各rankで`Arc::ptr_eq`、かつ同じMPI communicator context）
- 同じglobal shape
- decompositionの差がちょうど一つのtopology位置だけ
- permutationは任意の有効な置換

decompositionが同じ場合は分散転置ではないため、
`AllToAllvTransposePlan::new`は全rankでpreflightした後に
`UnsupportedDecompositionChange`を返す。呼出し側は既存の
`LocalTransposePlan`を使う。二つ以上の位置が異なる場合も受け付けない。
`LocalTransposePlan`は意図的にnoncollectiveなので、あるrankだけがlocal APIを
呼び、他rankが本APIを呼ぶ使い方は契約違反とする（呼ばないrankを本API側から
検出することはできない）。全rankが本APIへ入る場合はmode、変更軸、
source/destinationの順序をpreflightで一致させ、異なる軸やlocal相当（同一
decomposition）をsubcommunicatorへ進めない。

in-place、point-to-point、FFT、unchecked fast path、新依存、未使用の通信方式
enum、将来の全機能を表すdummy APIはこのPRに含めない。

## 公開API

`crates/pencil-array/src/alltoallv_transpose.rs`を追加し、次だけを
`lib.rs`からre-exportする。

```rust
#[derive(Debug)]
pub struct AllToAllvTransposePlan<const N: usize, const M: usize> { /* private */ }

#[derive(Debug)]
pub struct AllToAllvTransposeWorkspace<T> { /* private Vec<T> buffers */ }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllToAllvTransposeWorkspaceRequirements {
    pub send_len: usize,
    pub receive_len: usize,
}

impl<T> AllToAllvTransposeWorkspace<T> {
    pub fn from_vecs(send: Vec<T>, receive: Vec<T>) -> Self;
}

impl<const N: usize, const M: usize> AllToAllvTransposePlan<N, M> {
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, AllToAllvTransposeError>;

    pub fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<AllToAllvTransposeWorkspaceRequirements, AllToAllvTransposeError>;

    pub fn execute_views<T>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut AllToAllvTransposeWorkspace<T>,
    ) -> Result<(), AllToAllvTransposeError>
    where
        T: mpi::datatype::Equivalence + Copy;
}
```

owner用`execute` wrapperは作らず、既存の`PencilArray::view`/`view_mut`を使う。
`workspace_requirements`と`from_vecs`はMPIを呼ばない。要求量はcalling rankの
peer patternに対する量なのでrankごとに異なってよい。

エラー型は同じmoduleに置き、少なくとも次を持つ。

```rust
pub enum AllToAllvTransposeError {
    IncompatibleTopology,
    IncompatibleGlobalShape,
    UnsupportedDecompositionChange,
    SourceLayoutMismatch,
    DestinationLayoutMismatch,
    ExtraShapeMismatch,
    WorkspaceTooSmall {
        send_required: usize,
        send_len: usize,
        receive_required: usize,
        receive_len: usize,
    },
    CountOverflow,
    PreparationFailed,
    CollectiveDescriptorMismatch,
    CollectivePreconditionFailed,
    Array(#[from] ArrayError),
}
```

`CountOverflow`は`mpi::Count`に収まらないcountまたはdisplacement、
`PreparationFailed`はcheckedなpattern/descriptor準備の失敗に使う。rsmpi
0.8.2の`CommunicatorCollectives::all_to_all_varcount_into`は`()`を返すため、
このPRでは独自FFIやMPIエラー変換を追加しない。

## plan構築

1. 入力のlocal検査結果を保存し、失敗してもこの時点ではreturnしない。差分位置を
   求め、ちょうど一つなら`changed_topology_axis`を得る。失敗時はdescriptorに
   invalid sentinelを入れてpreflightへ進む。
2. source topologyの**全Cartesian communicator**
   (`source.topology().cartesian()`)で、N、M、descriptor長をscalarのmin/max
   all-reduceで合意する。header不一致ならここで全rankがreturnする。可変長の
   descriptor reductionやaxis subcommunicatorのcollectiveはまだ呼ばない。
3. scalar合意が成立し、descriptor長が`mpi::Count`/`usize`に収まる場合だけ、
   descriptorと同じ長さの`u64` min/max受信配列を`try_reserve`でfallibleに確保する。
   二つの受信配列を準備できたことをscalar flagで全rank合意した後、descriptorを
   nativeな要素別min/max `all_reduce`へ渡す。minとmaxがword単位で完全一致した
   場合だけdescriptor一致とする。全rank分の巨大配列や独自hashは作らない。
   descriptorにはschema/version、`AllToAllv` distributed mode、ordered
   source-to-destination direction、変更topology軸、process grid、sourceと
   destinationそれぞれのglobal shape/decomposition/permutationを含める。
4. descriptor一致後、descriptorに対するlocal basic validationを全Cartesian
   communicatorでvalidity all-reduceする。失敗rankがあれば、どのrankもaxis
   subcommunicatorへ進まず、MPI payload前に全rankがreturnする。失敗rankは具体的な
   local error、成功rankは`CollectivePreconditionFailed`を返してよい。
5. 全rankのbasic validation成功後だけ、変更軸のstored one-dimensional
   subcommunicatorからpeer patternを作る。subcommunicatorのrank `p`ごとに
   `rank_to_coordinates_into(p, ...)`でpeer coordinateを取得し、`peer_rank`と
   `peer_coordinate`を別々に保存する。`peer_rank == coordinate`とは仮定しない。
   現rankのpattern生成結果をもう一度全Cartesian communicatorでvalidity
   all-reduceする。pattern準備に失敗したrankがあればpayloadへ進まずreturnする。
6. すべて成功した後だけ、source/destination、変更軸、peer patternを保持する
   planを返す。constructorは同一decompositionをlocalへ暗黙変換しない。

### peer pattern

現rankのsource local rangeを`S_i`、destination local rangeを`D_i`、peer
coordinate `q`のrangeを`S_q`/`D_q`とする。

- peer `p`へ送るregion: `intersection(S_i, D_q)`
- peer `p`から受けるregion: `intersection(S_q, D_i)`

intersectionは全N個のlogical spatial axisについて計算する。regionのglobal
範囲とspatial長だけをpeerごとに保持し、要素ごとのsend/receive offset列や全要素の
index表は保持しない。pack/unpack時にregion内をlogical spatial row-major順に
走査し、既存のrow-major mappingとpermutationからsource/destination offsetを
checked計算する。constructorは各ローカル要素を列挙せず、巨大shapeでも
O(peers*N)のmetadataだけを作る。checked offset事前検査が必要な場合は、データを
書かない追加走査として実行する。

各peerのspatial長、送受信count、peer順の累積displacementを`usize`で保持し、
count/displacement/合計はchecked検査する。小さいmetadataの確保は必要なら
`try_reserve`で行い、失敗を全Cartesian communicatorで合意する。extra shapeの
element countはplanに埋め込まない。

## executeのcollective preflightと順序

executeは毎回、次の順序を固定する。全rankが同じsource communicator context
で同じ順序に呼ぶことをAPI契約とする。constructorとexecuteは共通の固定サイズ
header（schema/version、operation、N、M、canonical descriptor word数）を最初に
合意する。headerが一致しない場合も全rankが同じheader reductionsを完了して
一斉にreturnし、異なるdescriptor長・collective回数の後続処理へ進まない。

1. plan descriptorにruntime情報を連結する。source/destination viewのexactな
   extra shape（rankと各extent）、Tの識別・表現
   (`type_name::<T>()`、`size_of::<T>()`、`align_of::<T>()`、および
   `Equivalence::Out`の型識別）を含める。local coordinate、local length、
   workspace lengthはrank固有なのでdescriptorには含めない。
2. 全Cartesian communicatorで、固定長headerの各scalarを固定順に合意し、可変長
   descriptorの長さと準備成功も同じscalar段階で合意する。長さが一致して準備が
   全rankで成功した後だけ、descriptor wordをexact比較する。extra/typeの長さが
   descriptor内部に明示される場合、冗長な長さ専用collectiveは行わない。
3. descriptor不一致または準備失敗なら、全rankが可変payload collectiveや
   axis subcommunicatorへ進まずreturnする。どの失敗経路でも別rankを後続collective
   へ置き去りにしない。
4. 各rankで以下をMPIなしに検査する。
   - source viewのpencilがplan source layoutと一致
   - destination viewのpencilがplan destination layoutと一致
   - source/destination extra shapeがexactに一致
   - `workspace.send_buffer.len()`/`receive_buffer.len()`が要求量以上
   - peerごとのcount/displacement、合計、region内offsetがchecked範囲内
   - `mpi::Count`への変換が成功し、count + displacementがbuffer prefix内
5. 上のlocal結果を全Cartesian communicatorでmin all-reduceする。失敗が一つでも
   あれば、pack・Alltoallv・destination writeの前に全rankがErrを返す。local
   preparationはworkspaceを変更しない。これにより一rankだけのlayout、extra、
   workspace、count、displacement、offset失敗でpeerがpayload待ちにならない。
6. all rank validの後にだけworkspaceへpackし、変更軸のsubcommunicatorで
   `MPI_Alltoallv`を一回呼び、受信workspaceからdestinationへunpackする。pack/unpack
   はregion内をlogical順に走査し、要素数依存のoffset表は使わない。post-validation
   のpack/unpackはchecked preflight済みで、destinationへの最初のwriteはAlltoallv
   完了後に行う。

### rsmpi 0.8.2の呼出し

`mpi::Count`は`i32`であり、`Partition::new`/`PartitionMut::new`はcountと
displacementの長さや`count + displacement <= buf.count()`を完全には検証せず、
内部assertを持つ。したがって実装側で全条件を先に検査し、次の形で**有効長の
slice**だけを渡す。

```rust
let send_partition = mpi::datatype::Partition::new(
    &send_buffer[..send_len],
    &send_counts[..],
    &send_displacements[..],
);
let mut receive_partition = mpi::datatype::PartitionMut::new(
    &mut receive_buffer[..receive_len],
    &receive_counts[..],
    &receive_displacements[..],
);
subcommunicator.all_to_all_varcount_into(
    &send_partition,
    &mut receive_partition,
);
```

count/displacement配列の順序と長さはsubcommunicatorのrank順と一致させる。
`send_counts[p]`はpeer `p`へ送る量、`receive_counts[p]`はpeer `p`から受ける量
である。

## pack/unpackとworkspace契約

peer rank昇順、extra batchのrow-major linear順、logical spatial regionの
row-major順を唯一の順序とする。sendとreceiveでこの3段の順序を一致させる。
extra軸はpermutationしない。

`AllToAllvTransposeWorkspace::from_vecs`へ渡すVecは初期化済みでなければなら
ない。必要量の検証はcapacityではなく`len`で行う。executeは必要prefixを
`&mut [T]`として使うだけで、capacityをlenへ変えず、`resize`やbacking storageの
拡張を行わない。capacityだけに書く処理、unsafe、`MaybeUninit`、独自の要素trait、
新依存は使わない。`T: Equivalence + Copy`とし、pack/unpackは代入コピーだけで
`Clone`/`Default` callbackを呼ばない。

`workspace_requirements`は各peer spatial長に`extra_shape.element_count()`を
checked乗算してsend/receive総量を返す。MPI count/displacementまたは`usize`
に収まらない場合はErrとする。executeも同じ検査をcollective preflightで再実施
する。Tには独自`MpiElement` traitを導入せず、`Equivalence + Copy`だけを要求
する。実行時の`Default`/`Clone` callbackは使わない。

成功時もsource viewは不変である。descriptor、layout、extra、workspace、
count、preparationのいずれかのpreflight error時はdestinationも不変である。
Alltoallv開始後のMPI障害やpanic時のdestination内容は未規定だが、sourceは実行中
も保持する。

## 変更ファイルと実装順序

1. この計画に対応する仕様追補を`docs/superpowers/specs/2026-09-11-pencil-arrays-rust-port-design.md`へ追加し、metadata-only pattern、固定header、exact descriptor、workspace、collective契約、型契約、障害境界を明記する。

2. `crates/pencil-array/src/alltoallv_transpose.rs`を追加し、error、canonical
   descriptor、checked peer metadata、workspace、preflight、pack、Alltoallv、
   unpackを実装する。要素ごとのoffset表は作らない。
3. `crates/pencil-array/src/lib.rs`でmoduleを登録し、公開型をre-exportする。
   既存`LocalTransposePlan`は変更しない。
4. `crates/pencil-array/tests/alltoallv_transpose.rs`を追加する。MPI binary内の
   `mpi::initialize()`は一つのintegration test関数で一度だけ呼ぶ。

MPI実行時障害、任意のpanic、プロセス喪失については、既存rsmpi bindingと同様に
安全な`Result`回収を保証しない。

## テスト計画

同一global添字から値を直接生成し、destinationのlocal logical indexをglobal
rangeへ戻して期待値を照合する。少なくとも一つの3D caseではsource物理bufferを
[z, x, y]、独立expected Vecをdestination物理順[y, z, x]の明示的nested loopで
構築し、`destination.as_slice()`と比較する。各成功caseでsource保持、直接期待値、
逆方向planによるround-tripを確認する。

- 1 rank: 2D、3D、scalar extra、複数extra、permutation、subcomm size 1。
- 4 ranks: uneven decomposition、empty local partition、2D/3D、unsorted
  decomposition、source/destination permutationの組合せ。
- 6 ranksが使用可能ならprocess grid `[2, 3]`の非正方3D case。変更軸をaxis 0
  とaxis 1の両方で試す。可能ならworld rank順を反転したcommunicatorからtopologyを
  作り、Cartesian `rank_to_coordinates_into`を使う実装がworld rankと混同しないことを
  結果で確認する。ただしpeer rankとcoordinateが必ず異なるとは仮定しない。
- extra shape `[2, 3]`とzero-extraの`ExtraShape::scalar()`。可能なら`[2, 0]`
  のzero payloadも実行し、全rankがAlltoallvへ参加することを確認する。
- 同じdecompositionはAllToAllv planがcollectiveに
  `UnsupportedDecompositionChange`を返し、実データは既存
  `LocalTransposePlan`で処理することを確認する。
- 一rankだけsource/destination layout、extra shape、workspace長を壊す。全rankが
  timeout内にErrとなり、destinationのsentinelとsourceが全rankで変化しないことを
  確認する。
- forward/reverseなどの有効planを全rankで先に作り、実行時だけ一rankが別planを
  渡す。さらにplan construction時に一rankだけ変更軸または同一decompositionを
  選ぶ。いずれもmode、方向、変更軸をexact descriptor/preflightで検出し、
  subcommunicatorへ入る前に全rankがErrとなることを確認する。
- 一rankだけconst `N`を変えてplan constructionを呼び、N/M/descriptor長の
  scalar preflightが不一致countのdescriptor reductionやpayloadへ進まないことを確認
  する。別caseでは一rankだけextra rankまたは同じrankでextent内容を変え、同じ
  propertyをexecuteで確認する。
- `u32`/`u64`などTの識別またはsize/alignが異なるrankを混ぜ、T descriptorで
  payload前にErrとなることを確認する。

成功caseと失敗caseを同じMPI binaryの一つのtest関数から順に実行し、全topology・
arrayをMPI finalize前にdropする。

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --lib --locked

timeout --foreground 120s mpiexec --oversubscribe -n 1 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec --oversubscribe -n 4 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
timeout --foreground 120s mpiexec --oversubscribe -n 6 cargo test -p pencil-array --test alltoallv_transpose --locked -- --nocapture --test-threads=1
cargo test --workspace --doc --locked -- --show-output
```

6 rankをサポートしない実行環境ではそのcaseだけをskipし、1/4 rankを必須とする。
本stageでは上記Rust実装とintegration testsを行う。commit、push、PRは行わない。
