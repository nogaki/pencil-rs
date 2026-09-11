# PencilArrays / PencilFFTs Rust移植 設計仕様

日付: 2026-09-11  
状態: レビュー用設計仕様  
対象: CPU + MPIによる任意次元分散配列基盤と分散FFT基盤

## 1. 参照実装

本設計は、以下のJulia実装を観測可能な仕様とアルゴリズム上の参照として用いる。

- PencilArrays.jl: `jipolanco/PencilArrays.jl`, commit `12229b99b827e07880517982c3365a18d1f9b8dc`
- PencilFFTs.jl: `jipolanco/PencilFFTs.jl`, commit `1d98a3ff790c40445987ad64b99eb3b946a11034`

両参照実装はMIT Licenseである。Rust移植で参照実装のコードまたは実質的な部分を翻案する場合、元の著作権表示と許諾文を保持する。

## 2. 目的

最終目的は、Rust上で次を提供する数値計算基盤を構築することである。

1. MPIプロセス間に分散した任意次元配列の記述と操作
2. 複数の分散配置間の再分配
3. 同一データバッファ上でのin-place再分配
4. 全空間軸に対する分散C2C FFT
5. 全空間軸に対するout-of-place R2C/C2R FFT
6. Julia実装と比較可能な正当性・性能評価基盤

分散配列基盤とFFT層は別crateとし、一方向の依存関係にする。

```text
pencil-fft
    depends on
pencil-array

pencil-array
    does not depend on FFT libraries
```

## 3. 初期スコープ

### 3.1 含める機能

- 空間次元数 `N` とMPI topology次元数 `M` をconst genericsで表現
- `1 <= M <= N` の分散配列
- row-major連続ストレージ
- 任意個のextra dimensions
- `Pencil`
- `PencilArray`
- `ManyPencilArray`
- out-of-place転置
- in-place転置
- `MPI_Alltoallv`方式
- nonblocking point-to-point方式
- C2C分散FFTのout-of-place実行
- C2C分散FFTのin-place実行
- R2C/C2R分散FFTのout-of-place実行
- `f32`および`f64`
- CPUメモリ上の`Vec<T>`
- RustFFTおよびRealFFTを用いたローカルFFT

### 3.2 初期スコープ外

- in-place R2C/C2R
- GPUストレージ
- FFTWバックエンド
- DCT、DST、Chebyshev変換
- 任意のspatial軸部分集合だけを変換する部分FFT
- runtime可変次元の`DynPencil`
- MPI-IOおよびParallel HDF5
- Julia版のbroadcast、reduction、global view、local grid、ODE連携の全面移植
- 複数の未完了転置を同一communicator上で同時実行するAPI
- `Executor`またはそれに相当する実行器ラッパー

これらは中核設計を破壊しない形で後続フェーズへ追加する。

## 4. 基本用語

### 4.1 Spatial dimensions

MPI分割、空間軸置換、FFTの対象となる軸である。個数はコンパイル時定数`N`。

### 4.2 Extra dimensions

MPI分割も空間軸置換もFFTも行わない外側のバッチ軸である。個数と形状は実行時に保持する。

例:

```text
A[orbital, spin, x, y, z]

extra dimensions   = [orbital, spin]
spatial dimensions = [x, y, z]
```

### 4.3 Logical order

利用者から見た軸順序である。Rust版では常に次とする。

```text
[extra..., spatial...]
```

### 4.4 Memory order

実際のrow-majorストレージ上の軸順序である。

```text
[extra..., permuted spatial...]
```

row-majorなので、最後の物理軸が最も連続である。

### 4.5 Decomposition

MPI topologyの各軸を、どのspatial軸へ対応させるかを示す順序付き写像である。

```text
process_grid = [2, 4]
decomposition = [0, 2]

MPI topology axis 0 -> spatial axis 0を2分割
MPI topology axis 1 -> spatial axis 2を4分割
```

`[0, 2]`と`[2, 0]`は異なる配置である。

## 5. 設計原則

1. 配置情報と可変作業領域を分離する。
2. 数学的変換計画とデータ所有を分離する。
3. FFT固有の状態をArray基盤へ持ち込まない。
4. Julia APIの概念と操作の意味を参考にするが、Rustの所有権を優先する。
5. 不変情報は`Arc`で共有し、可変バッファは単一所有か排他的借用にする。
6. 同じ情報を重複保持して食い違いを作らない。
7. 公開APIから不正なaliasを作れないようにする。
8. MPI collective操作では、rankごとのローカル失敗によるデッドロックを避ける。
9. 初期実装のストレージ抽象化は`Vec<T>`に限定し、未検証のGPU汎用化を行わない。
10. 正しさの基準実装と性能最適化を分離する。

## 6. Cargo workspace構成

```text
workspace/
├── crates/
│   ├── pencil-array/
│   │   ├── topology
│   │   ├── geometry
│   │   ├── array
│   │   └── transpose
│   └── pencil-fft/
│       ├── backend
│       ├── plan
│       ├── workspace
│       └── operations
├── tests/
│   ├── cross-language
│   └── mpi
├── benchmarks/
└── docs/
```

`pencil-array`はMPIに依存するが、RustFFT、RealFFT、FFTWには依存しない。

`pencil-fft`は`pencil-array`、RustFFT、RealFFTに依存する。

## 7. MPI資源とtopology

### 7.1 `MpiTopology<M>`

```rust
pub struct MpiTopology<const M: usize> {
    // private
}
```

責務:

- Cartesian communicatorの所有
- 各topology軸のsubcommunicatorの所有
- process grid形状
- local rankのCartesian座標
- Cartesian座標とrankの対応
- communicator資源のRAII管理

公開APIの骨格:

```rust
impl<const M: usize> MpiTopology<M> {
    pub fn new(
        comm: &impl Communicator,
        process_grid: [usize; M],
    ) -> Result<Arc<Self>, TopologyError>;

    pub fn auto(
        comm: &impl Communicator,
    ) -> Result<Arc<Self>, TopologyError>;

    pub fn from_cartesian(
        comm: &impl CartesianCommunicator,
    ) -> Result<Arc<Self>, TopologyError>;

    pub fn process_grid(&self) -> &[usize; M];
    pub fn local_coords(&self) -> &[usize; M];
    pub fn rank(&self) -> i32;
    pub fn size(&self) -> usize;
    pub fn rank_at(&self, coords: [usize; M]) -> Result<i32, TopologyError>;
}
```

`MpiTopology::new`、`auto`、`from_cartesian`はcollectiveである。

`MpiTopology`は入力communicatorとは別の内部communicator contextを所有する。`new`ではCartesian communicatorを新規作成し、`from_cartesian`では受け取ったcommunicatorを複製して所有する。これにより、ライブラリ内部のpoint-to-point tagを利用者側の通信から隔離する。`MpiTopology`はMPI finalize後まで生存してはならない。実際の寿命表現は採用するMPI bindingに従い、`Send`または`Sync`を手動で無条件実装しない。

## 8. 軸型と置換

### 8.1 `SpatialAxis`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpatialAxis(usize);
```

`0..N`の範囲を検証したspatial軸番号である。

### 8.2 `AxisPermutation<N>`

```rust
pub struct AxisPermutation<const N: usize> {
    axes_in_memory_order: [SpatialAxis; N],
}
```

意味は常に「物理メモリ順に並べたspatial軸列」とする。

例:

```text
AxisPermutation::<3>::new([0, 2, 1])

physical spatial axes = [x, z, y]
yがrow-majorの連続軸
```

公開API:

```rust
impl<const N: usize> AxisPermutation<N> {
    pub fn new(axes: [usize; N]) -> Result<Self, AxisError>;
    pub fn identity() -> Self;
    pub fn axes(&self) -> &[SpatialAxis; N];
    pub fn inverse_position(&self, axis: SpatialAxis) -> usize;
}
```

構築時に全軸が一度ずつ現れることを検証する。

## 9. `Pencil<N, M>`

```rust
pub struct Pencil<const N: usize, const M: usize> {
    topology: Arc<MpiTopology<M>>,
    global_shape: [usize; N],
    decomposition: [SpatialAxis; M],
    permutation: AxisPermutation<N>,
    // all-rank/local ranges and cached lengths
}
```

`Pencil`は分散配置だけを表し、以下を所有しない。

- 数値データ
- send/receive buffer
- FFT計画
- FFT状態

### 9.1 構築

```rust
impl<const N: usize, const M: usize> Pencil<N, M> {
    pub fn new(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        decomposition: [usize; M],
    ) -> Result<Arc<Self>, PencilError>;

    pub fn new_default(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
    ) -> Result<Arc<Self>, PencilError>;

    pub fn new_permuted(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        decomposition: [usize; M],
        permutation: AxisPermutation<N>,
    ) -> Result<Arc<Self>, PencilError>;
}
```

row-majorのデフォルト分割軸は先頭`M`軸とする。

```text
N=3, M=2 -> [0, 1]
N=5, M=2 -> [0, 1]
N=M      -> [0, 1, ..., N-1]
```

### 9.2 派生

```rust
impl<const N: usize, const M: usize> Pencil<N, M> {
    pub fn with_decomposition(
        self: &Arc<Self>,
        decomposition: [usize; M],
    ) -> Result<Arc<Self>, PencilError>;

    pub fn with_permutation(
        self: &Arc<Self>,
        permutation: AxisPermutation<N>,
    ) -> Result<Arc<Self>, PencilError>;

    pub fn with_global_shape(
        self: &Arc<Self>,
        global_shape: [usize; N],
    ) -> Result<Arc<Self>, PencilError>;

    pub fn reconfigured(
        self: &Arc<Self>,
        config: PencilConfig<N, M>,
    ) -> Result<Arc<Self>, PencilError>;
}
```

`with_decomposition`は元の`Pencil`を変更せず、同じtopology、global shape、permutationを共有・継承して担当範囲を再計算する。数値データの移動は行わない。

### 9.3 領域分割

長さ`L`の軸を`P`分割したとき、0-based process coordinate `p`の担当範囲は次とする。

```text
floor(L*p/P) .. floor(L*(p+1)/P)
```

Julia版の1-based閉区間を、Rustの0-based半開区間へ変換した規則である。

`P > L`もArray基盤では許可する。ライブラリは暗黙のログ警告を出さず、空ローカル領域を正しく表現する。利用者は`local_len()`およびローカル範囲から空担当を検出できる。転置・FFT実装は要素数0のrankを正常なno-op参加者として扱う。

### 9.4 検証

構築時に以下を検査する。

- `1 <= M <= N`
- `global_shape[d] > 0`
- decomposition各要素が`0..N`
- decompositionに重複がない
- permutationが有効
- process grid積とcommunicator sizeが一致
- 形状積とoffset計算で`usize` overflowがない

`M=N`は許容する。ただし現在のpencil FFT方式では`M<N`を要求する。

### 9.5 参照API

```rust
impl<const N: usize, const M: usize> Pencil<N, M> {
    pub fn topology(&self) -> &Arc<MpiTopology<M>>;
    pub fn global_shape(&self) -> &[usize; N];
    pub fn decomposition(&self) -> &[SpatialAxis; M];
    pub fn permutation(&self) -> &AxisPermutation<N>;
    pub fn local_ranges(&self) -> &[std::ops::Range<usize>; N];
    pub fn local_shape_logical(&self) -> [usize; N];
    pub fn local_shape_memory(&self) -> [usize; N];
    pub fn local_len(&self) -> usize;
    pub fn global_len(&self) -> usize;
    pub fn ranges_at(
        &self,
        process_coords: [usize; M],
    ) -> Result<[std::ops::Range<usize>; N], PencilError>;

    pub fn same_topology(&self, other: &Self) -> bool;
    pub fn same_distribution(&self, other: &Self) -> bool;
    pub fn same_layout(&self, other: &Self) -> bool;
}
```

比較規則:

```text
same_topology:
    同一のArc<MpiTopology>

same_distribution:
    same_topology
    + global_shape
    + decomposition

same_layout:
    same_distribution
    + permutation
```

`Arc::ptr_eq`を`Pencil`全体の等価性判定には使わない。

## 10. `ExtraShape`

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtraShape {
    dimensions: Box<[usize]>,
    element_count: usize,
}
```

公開API:

```rust
impl ExtraShape {
    pub fn new(
        dimensions: impl Into<Box<[usize]>>,
    ) -> Result<Self, ShapeError>;

    pub fn scalar() -> Self;
    pub fn dimensions(&self) -> &[usize];
    pub fn element_count(&self) -> usize;
}
```

`scalar()`は空のextra shapeで、`element_count == 1`。

extra shapeの積はoverflow検査済みである。

## 11. `PencilArray<T, N, M>`

```rust
pub struct PencilArray<T, const N: usize, const M: usize> {
    pencil: Arc<Pencil<N, M>>,
    extra_shape: ExtraShape,
    storage: Vec<T>,
}
```

必要なローカル長は次である。

```text
pencil.local_len() * extra_shape.element_count()
```

### 11.1 構築

```rust
impl<T, const N: usize, const M: usize> PencilArray<T, N, M> {
    pub fn from_vec(
        pencil: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        storage: Vec<T>,
    ) -> Result<Self, ArrayError>;

    pub fn from_elem(
        pencil: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        value: T,
    ) -> Result<Self, ArrayError>
    where
        T: Clone;

    pub fn from_fn(
        pencil: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        f: impl FnMut() -> T,
    ) -> Result<Self, ArrayError>;
}
```

`from_vec`は`storage.len()`の完全一致を要求する。

通常の公開型に未初期化`T`を格納しない。未初期化最適化は、必要性が実測された場合に別型で追加する。

### 11.2 形状

```rust
impl<T, const N: usize, const M: usize> PencilArray<T, N, M> {
    pub fn pencil(&self) -> &Arc<Pencil<N, M>>;
    pub fn extra_shape(&self) -> &ExtraShape;
    pub fn local_spatial_shape(&self) -> [usize; N];
    pub fn local_spatial_memory_shape(&self) -> [usize; N];
    pub fn logical_shape(&self) -> Vec<usize>;
    pub fn memory_shape(&self) -> Vec<usize>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

```text
logical_shape = [extra..., local spatial logical...]
memory_shape  = [extra..., local spatial memory...]
```

### 11.3 データアクセス

```rust
impl<T, const N: usize, const M: usize> PencilArray<T, N, M> {
    pub fn as_slice(&self) -> &[T];
    pub fn as_mut_slice(&mut self) -> &mut [T];

    pub fn get_local(
        &self,
        extra_indices: &[usize],
        spatial_indices: [usize; N],
    ) -> Option<&T>;

    pub fn get_local_mut(
        &mut self,
        extra_indices: &[usize],
        spatial_indices: [usize; N],
    ) -> Option<&mut T>;
}
```

`as_slice`の並びは`memory_shape()`に対するrow-major順である。

## 12. 借用view

```rust
pub struct PencilArrayView<'a, T, const N: usize, const M: usize> {
    pencil: &'a Pencil<N, M>,
    extra_shape: &'a ExtraShape,
    storage: &'a [T],
}

pub struct PencilArrayViewMut<'a, T, const N: usize, const M: usize> {
    pencil: &'a Pencil<N, M>,
    extra_shape: &'a ExtraShape,
    storage: &'a mut [T],
}
```

`PencilArray`は`view()`と`view_mut()`を提供する。viewは親より長生きしない。

同一storageへの複数の可変viewは作れない。

## 13. `ManyPencilArray<T, N, M>`

```rust
pub struct ManyPencilArray<T, const N: usize, const M: usize> {
    pencils: Box<[Arc<Pencil<N, M>>]>,
    extra_shape: ExtraShape,
    storage: Vec<T>,
    state: LayoutState,
}

enum LayoutState {
    Valid(usize),
    Poisoned,
}
```

責務:

- 一つの本体storageを所有する
- 同じグローバルデータに対する複数の分散配置を登録する
- 現在どの配置としてstorageが有効かを記録する
- 現在有効な配置のviewだけを公開する

通信バッファやFFT状態は所有しない。

### 13.1 不変条件

- pencilsが空でない
- 全Pencilが同じ`MpiTopology`を共有
- 全Pencilが同じglobal shapeを持つ
- 全Pencilの`N`と`M`が同じ
- 同一layoutの重複がない
- extra shapeは全配置で共通
- storage長は全配置の最大ローカル容量と完全一致

必要storage長:

```text
max_i(pencils[i].local_len()) * extra_shape.element_count()
```

登録された全配置が互いに直接転置可能である必要はない。

### 13.2 構築

```rust
impl<T, const N: usize, const M: usize> ManyPencilArray<T, N, M> {
    pub fn from_vec(
        pencils: impl Into<Box<[Arc<Pencil<N, M>>]>>,
        active: usize,
        extra_shape: ExtraShape,
        storage: Vec<T>,
    ) -> Result<Self, ArrayError>;

    pub fn from_elem(
        pencils: impl Into<Box<[Arc<Pencil<N, M>>]>>,
        active: usize,
        extra_shape: ExtraShape,
        value: T,
    ) -> Result<Self, ArrayError>
    where
        T: Clone;
}
```

### 13.3 公開アクセス

```rust
impl<T, const N: usize, const M: usize> ManyPencilArray<T, N, M> {
    pub fn pencils(&self) -> &[Arc<Pencil<N, M>>];
    pub fn active_pencil(&self) -> Result<&Pencil<N, M>, ArrayError>;
    pub fn extra_shape(&self) -> &ExtraShape;

    pub fn active_view(
        &self,
    ) -> Result<PencilArrayView<'_, T, N, M>, ArrayError>;

    pub fn active_view_mut(
        &mut self,
    ) -> Result<PencilArrayViewMut<'_, T, N, M>, ArrayError>;
}
```

以下は公開しない。

```text
任意indexのデータview
active indexの直接変更
storageを変更せず配置だけを有効化する操作
```

### 13.4 `overwrite_with`

```rust
pub fn overwrite_with<F, E>(
    &mut self,
    target: &Pencil<N, M>,
    write: F,
) -> Result<(), OverwriteError<E>>
where
    F: FnOnce(PencilArrayViewMut<'_, T, N, M>) -> Result<(), E>;
```

意味:

1. target layoutが登録済みか検証
2. stateを`Poisoned`へ変更
3. target layout用の全viewをclosureへ渡す
4. closureが成功した場合だけ`Valid(target)`へ変更
5. closureが`Err`またはpanicで終了した場合は`Poisoned`のまま

closureはview全体へ意味のある値を書き込む契約を持つ。無条件のactive-layout変更は許さない。

`overwrite_with`は、out-of-place FFTの最初の局所変換結果を書き込む用途と、既知の正しい全データによるpoison状態の復旧に利用できる。

## 14. 転置の意味

転置は、同じグローバルデータを異なるMPI分散配置および物理軸順序へ移す操作である。

転置は数値変換を行わず、要素の位置と所有rankだけを変更する。

### 14.1 互換条件

`source`と`destination`は次を満たす。

- 同じ`MpiTopology`
- 同じglobal shape
- decomposition配列が高々一位置だけ異なる
- permutationは任意の有効置換

例:

```text
[0, 1] -> [0, 2]  可能
[0, 1] -> [1, 2]  直接には不可
```

後者は中間配置を通す。

同じdecompositionでpermutationだけが異なる場合、MPI通信を行わないローカル転置とする。

`M=N`の場合、非自明なdecomposition変更は通常この条件を満たせないため、実質的には同一分割上のローカル軸置換だけが可能である。

## 15. `TransposePlan<N, M>`

```rust
pub struct TransposePlan<const N: usize, const M: usize> {
    source: Arc<Pencil<N, M>>,
    destination: Arc<Pencil<N, M>>,
    kind: TransposePlanKind<N>,
}

enum TransposePlanKind<const N: usize> {
    Local(LocalTransposePlan<N>),
    Distributed {
        exchange: ExchangePattern<N>,
        communication: CommunicationPlan,
    },
}

enum CommunicationPlan {
    AllToAllV(AllToAllVPlan),
    PointToPoint(PointToPointPlan),
}
```

### 15.1 用語

`ExchangePattern`:

- どのpeer rankと
- どの空間領域を
- 何要素交換するか

を表す。MPI APIの選択を知らない。

`CommunicationPlan`:

- `MPI_Alltoallv`
- nonblocking `MPI_Irecv` / `MPI_Isend`

のどちらでexchangeを実現するか、および方式固有の事前計算情報を表す。

通信方式を別の`method`フィールドとして重複保存しない。採用方式は`CommunicationPlan`のvariant自体で表す。

### 15.2 構築

```rust
pub enum TransposeMethod {
    AllToAllV,
    PointToPoint,
}

impl<const N: usize, const M: usize> TransposePlan<N, M> {
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
        method: TransposeMethod,
    ) -> Result<Self, TransposeError>;
}
```

分割が同じ場合は`Local`を構築し、method指定は実行に影響しない。

分割が一位置異なる場合、その位置に対応する1次元subcommunicatorを通信範囲として用いる。

### 15.3 `ExchangePattern`

```rust
struct ExchangePattern<const N: usize> {
    peers: Box<[PeerExchange<N>]>,
    send_spatial_len: usize,
    receive_spatial_len: usize,
}

struct PeerExchange<const N: usize> {
    peer_rank: i32,
    send_region: [std::ops::Range<usize>; N],
    receive_region: [std::ops::Range<usize>; N],
    send_offset_spatial: usize,
    receive_offset_spatial: usize,
    send_len_spatial: usize,
    receive_len_spatial: usize,
}
```

countとoffsetはspatial要素単位で保持する。extra dimensionsは実行時に乗算する。

同一`TransposePlan`を異なるextra shapeへ再利用できる。ただしsourceとdestinationのexactなextra shape一致を実行時に要求する。

### 15.4 AllToAllV

`AllToAllVPlan`はpeer順のsend/receive countsおよびdisplacementsを保持する。実行時にextra element countを反映し、MPI count型への変換時に範囲検査する。

処理:

```text
pack all peers
MPI_Alltoallv
unpack all peers
```

### 15.5 PointToPoint

`PointToPointPlan`はpeerごとのsend/receive slice、rank、tag規約、開始順を保持する。

初期実装の処理:

```text
pack all peers
post all Irecv
post all Isend
wait for all receives
unpack
wait for all sends
return
```

公開APIは同期的であり、戻る時点で全requestが完了している。

受信完了順の`WaitAny`とunpack重畳は後続最適化とする。公開APIは変更しない。

Point-to-point通信は`MpiTopology`が所有する内部subcommunicatorを使い、ライブラリ予約の固定tagを用いる。同一subcommunicator上で複数の未完了転置を並行実行しない。内部communicator contextによる利用者通信との隔離と、この逐次実行制約によってtag衝突を防ぐ。

## 16. `TransposeWorkspace<T>`

```rust
pub struct TransposeWorkspace<T> {
    send_buffer: Vec<T>,
    receive_buffer: Vec<T>,
}
```

必要ならpoint-to-point用のrequest記述領域を内部に持てるが、bufferを借用する未完了requestをworkspace内へ恒久保存しない。

```rust
pub struct TransposeWorkspaceRequirements {
    pub send_len: usize,
    pub receive_len: usize,
}
```

```rust
impl<const N: usize, const M: usize> TransposePlan<N, M> {
    pub fn workspace_requirements(
        &self,
        extra_shape: &ExtraShape,
    ) -> Result<TransposeWorkspaceRequirements, TransposeError>;
}
```

複数plan用workspaceは、各必要量の最大値で作る。

```rust
TransposeWorkspace::for_plans(
    [&plan_xy, &plan_yz],
    &extra_shape,
)?
```

実行中に自動拡張しない。容量不足は通信開始前のエラーとする。

## 17. 転置実行API

### 17.1 Out-of-place

```rust
impl<const N: usize, const M: usize> TransposePlan<N, M> {
    pub fn execute<T>(
        &self,
        source: &PencilArray<T, N, M>,
        destination: &mut PencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: MpiElement;

    pub fn execute_views<T>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        destination: PencilArrayViewMut<'_, T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: MpiElement;
}
```

事前条件:

- source layoutがplan sourceと一致
- destination layoutがplan destinationと一致
- exactなextra shapeが一致
- workspace容量が十分
- `T`がMPI送信可能

失敗保証:

- sourceは常に不変
- 通信開始前の失敗ではdestination不変
- 通信開始後の失敗ではdestination内容は未規定
- planは再利用可能
- workspaceの容量・型は有効だが内容は未規定

### 17.2 In-place

```rust
impl<const N: usize, const M: usize> TransposePlan<N, M> {
    pub fn execute_in_place<T>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: MpiElement;
}
```

実行順:

1. active layoutがsourceと一致するか検証
2. destination layoutが登録済みか検証
3. extra shapeとworkspace容量を検証
4. rank間の事前条件をcollectiveに合意
5. source storageをsend bufferへpack
6. MPI通信を完了
7. array stateを`Poisoned`へ変更
8. receive bufferから同じstorageへunpack
9. 成功時だけdestinationを`Valid`へ設定

packと通信は本体storageを変更しないため、poisoningは本体への書き込み直前に行う。ローカルin-place permutationでは、最初の本体書き換え直前にpoisoningする。

panicによる巻き戻しでも`Poisoned`が維持されるよう内部guardを使う。

## 18. Collective契約

以下はcollective operationである。

- `TransposePlan::new`のうちdistributed planを構築する場合。local plan構築はローカル操作
- `TransposePlan::execute`
- `TransposePlan::execute_in_place`
- 分散FFT plan構築のMPI整合性確認
- 分散FFT実行

通信開始前に、各rankのローカル事前検査結果をcollectiveに集約する。一つでも失敗すれば、全rankが実通信前にエラーを返す。

distributed plan構築時には、source/destination global shape、decomposition、permutation、変更されるtopology軸、通信方式からなる固定形式の`CollectiveDescriptor`を全rankで厳密比較する。planはそのdescriptorから安定に計算した128-bitの`PlanFingerprint`を保持する。実行時の事前検査では、このfingerprintとforward/backward等の操作種別をローカル成功フラグと一緒に小さなcollective検査へ含め、rankごとに異なるplanまたは方向を呼ぶ典型的な誤りを検出する。hash衝突の理論的可能性は残るため、全rankが同じcollectiveを同じ順序で呼ぶことはAPI契約でもある。テスト・debug buildではdescriptorの厳密比較を再実行できる。

初期版の公開実行APIはchecked実行のみとする。crate-privateなunchecked fast pathは初期版には実装せず、性能測定で必要性が確認された後に別途設計する。

## 19. 便利関数

Julia版の`transpose!`に対応する一回限りのAPIを提供する。

```rust
pub fn transpose<T, const N: usize, const M: usize>(
    source: &PencilArray<T, N, M>,
    destination: &mut PencilArray<T, N, M>,
    method: TransposeMethod,
) -> Result<(), TransposeError>
where
    T: MpiElement;
```

内部でplanとworkspaceを作る。反復計算では明示的plan/workspaceを使用する。

## 20. FFT crateのバックエンド

初期実装:

- C2C: RustFFT
- R2C/C2R: RealFFT

公開APIへRustFFT固有型を露出させない。

```rust
pub trait FftReal: private::Sealed + Copy + Send + Sync + 'static {}
impl FftReal for f32 {}
impl FftReal for f64 {}
```

複素型は`num_complex::Complex<R>`を使用する。

初期版で公開`FftBackend` traitは定義しない。二つ目の実バックエンドを実装した時点で共通境界を抽出する。

## 21. 共通FFT計画モデル

R2Cだけ例外的なstage構造にしない。C2CとR2Cは共通の段階列で表す。

```rust
struct TransformPlanCore<R, const N: usize, const M: usize> {
    stages: Box<[TransformStage<R, N, M>]>,
    transitions: Box<[StageTransition<N, M>]>,
    extra_shape: ExtraShape,
    normalization: R,
}
```

不変条件:

```text
stages.len() == N
transitions.len() + 1 == stages.len()
```

### 21.1 Endpoint

```rust
struct StageEndpoint<const N: usize, const M: usize> {
    pencil: Arc<Pencil<N, M>>,
    value_kind: ValueKind,
}

enum ValueKind {
    Real,
    Complex,
}
```

### 21.2 Stage

```rust
struct TransformStage<R, const N: usize, const M: usize> {
    axis: SpatialAxis,
    input: StageEndpoint<N, M>,
    output: StageEndpoint<N, M>,
    local_transform: LocalTransformPlan<R>,
}
```

```rust
enum LocalTransformPlan<R> {
    ComplexFft {
        forward: /* RustFFT plan */,
        backward: /* RustFFT plan */,
        len: usize,
    },
    RealComplexFft {
        forward: /* RealFFT R2C plan */,
        backward: /* RealFFT C2R plan */,
        real_len: usize,
        complex_len: usize,
    },
}
```

`RealComplexFft`は順方向にR2C、逆方向にC2Rを行う。同じ一般stage型を用いる。

### 21.3 Transition

```rust
struct StageTransition<const N: usize, const M: usize> {
    forward: TransposePlan<N, M>,
    backward: TransposePlan<N, M>,
}
```

整合条件:

```text
stage[i].output.pencil == transition[i].forward.source
transition[i].forward.destination == stage[i+1].input.pencil
stage[i].output.value_kind == stage[i+1].input.value_kind
```

逆方向も同様に整合する。

## 22. 公開FFT計画型

内部coreは共通化し、公開APIはmarker typeで型安全に分ける。

```rust
pub struct FftPlan<R, Kind, const N: usize, const M: usize> {
    core: Arc<TransformPlanCore<R, N, M>>,
    // workspace specs
    marker: std::marker::PhantomData<Kind>,
}

pub enum ComplexToComplex {}
pub enum RealToComplex {}

pub type C2cPlan<R, const N: usize, const M: usize> =
    FftPlan<R, ComplexToComplex, N, M>;

pub type R2cPlan<R, const N: usize, const M: usize> =
    FftPlan<R, RealToComplex, N, M>;
```

この分離により、R2C計画のin-place APIや、誤った入出力型を公開API上で表現できない。

## 23. FFT対象軸と配置列

初期版は全spatial軸を変換する。extra dimensionsは全てバッチ軸である。

FFT axis順序はrow-majorに合わせ、後ろから前へ処理する。

```text
[N-1, N-2, ..., 0]
```

初期入力制約:

- `M < N`
- input permutationはidentity
- input decompositionは`[0, 1, ..., M-1]`

### 23.1 物理軸置換規則

次にFFTする軸を現在の物理軸列から取り出し、末尾へ移す。他軸の相対順序は維持する。

3D例:

```text
z stage: [x, y, z]
y stage: [x, z, y]
x stage: [z, y, x]
```

### 23.2 分割軸更新規則

次のFFT軸`a`が現在分割されている場合、そのdecomposition位置で`a`を直前にFFT済みの`a+1`へ置換する。

3D、M=2:

```text
[0, 1] -> [0, 2] -> [1, 2]
```

一回の更新でdecompositionの一位置だけが変わるため、直接転置条件を満たす。

次のFFT軸が分割されていなければMPI再分配は不要で、必要な物理軸置換だけをlocal transitionとして行う。

概ね:

```text
distributed transitions = M
local transitions       = N - M - 1
all transitions         = N - 1
```

## 24. C2C計画

### 24.1 構築

```rust
impl<R, const N: usize, const M: usize> C2cPlan<R, N, M>
where
    R: FftReal,
{
    pub fn from_array(
        input: &PencilArray<Complex<R>, N, M>,
        method: TransposeMethod,
    ) -> Result<Self, FftError>;

    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, FftError>;

    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, FftError>;
}
```

planは配列を借用し続けない。

### 24.2 配列確保

```rust
pub fn allocate_input(&self)
    -> Result<PencilArray<Complex<R>, N, M>, FftError>;

pub fn allocate_output(&self)
    -> Result<PencilArray<Complex<R>, N, M>, FftError>;

pub fn allocate_in_place(&self)
    -> Result<C2cInPlaceArray<R, N, M>, FftError>;
```

### 24.3 Out-of-place実行

```rust
pub fn forward(
    &self,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
) -> Result<(), FftError>;

pub fn inverse(
    &self,
    source: &PencilArray<Complex<R>, N, M>,
    destination: &mut PencilArray<Complex<R>, N, M>,
    workspace: &mut C2cOutOfPlaceWorkspace<R, N, M>,
) -> Result<(), FftError>;
```

実行戦略:

```text
first local FFT:
    source -> one intermediate ManyPencilArray

middle stages:
    in-place transpose on intermediate
    in-place local FFT on intermediate

last local FFT:
    intermediate -> destination
```

最後の全配列コピーは行わない。

`N=1`では中間storageを確保せず、sourceからdestinationへ直接変換する。

### 24.4 In-place実行

```rust
pub fn forward_in_place(
    &self,
    array: &mut C2cInPlaceArray<R, N, M>,
    workspace: &mut C2cInPlaceWorkspace<R, N, M>,
) -> Result<(), FftError>;

pub fn inverse_in_place(
    &self,
    array: &mut C2cInPlaceArray<R, N, M>,
    workspace: &mut C2cInPlaceWorkspace<R, N, M>,
) -> Result<(), FftError>;
```

論理stage列とtransition列はout-of-placeと共有する。異なるのはbuffer割り当てと実行ループだけである。

## 25. `C2cInPlaceArray`

Array基盤の`ManyPencilArray`は物理配置状態だけを管理する。FFT完了状態はFFT crate側で管理する。

```rust
pub struct C2cInPlaceArray<R, const N: usize, const M: usize> {
    spec: Arc<C2cInPlaceSpec<N, M>>,
    array: ManyPencilArray<Complex<R>, N, M>,
    state: C2cState,
}

pub enum C2cState {
    Input,
    Output,
    Poisoned,
}
```

意味:

```text
Input:
    順変換開始可能な完全な入力

Output:
    逆変換開始可能な完全な周波数空間データ

Poisoned:
    FFT途中で失敗し、InputでもOutputでもない
```

`active_layout`だけでは「一部の軸だけFFT済み」という中間状態を表せないため、このラッパーが必要である。

順変換:

```text
Input -> Poisoned -> Output
```

逆変換:

```text
Output -> Poisoned -> Input
```

事前検査で失敗した場合は元状態を維持する。データ変更開始後の失敗ではPoisonedを維持する。

公開viewはInputまたはOutput時だけ取得可能。

## 26. R2C/C2R計画

R2Cは一般的な`TransformStage`の`RealComplexFft` variantとして表す。専用の`R2cHeadStage`は作らない。

row-majorの最後のspatial軸を最初にR2Cする。

実数global shape:

```text
[N0, ..., N(N-1)]
```

複素global shape:

```text
[N0, ..., floor(N(N-1)/2)+1]
```

最初のstage:

```text
input  endpoint: Real,    original shape
output endpoint: Complex, reduced shape
local transform: RealComplexFft
```

以後のstageはreduced complex shape上のC2Cである。

`TransposePlan`は同じglobal shape・同じ要素型の配置間だけを結ぶ。したがって、R2Cの形状変更自体は`TransposePlan`ではない。

### 26.1 公開API

```rust
impl<R, const N: usize, const M: usize> R2cPlan<R, N, M>
where
    R: FftReal,
{
    pub fn from_array(
        input: &PencilArray<R, N, M>,
        method: TransposeMethod,
    ) -> Result<Self, FftError>;

    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, FftError>;

    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        global_shape: [usize; N],
        extra_shape: ExtraShape,
        method: TransposeMethod,
    ) -> Result<Self, FftError>;

    pub fn allocate_input(&self)
        -> Result<PencilArray<R, N, M>, FftError>;

    pub fn allocate_output(&self)
        -> Result<PencilArray<Complex<R>, N, M>, FftError>;

    pub fn forward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftError>;

    pub fn inverse(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftError>;
}
```

in-place APIは初期版に存在しない。

元の実数長をplanへ必ず保存する。複素長だけでは偶数・奇数の元長を一意に復元できない。

RealFFTが入力bufferをscratchとして変更するため、out-of-place APIの入力保持には1本のline bufferを使う。

## 27. 正規化

順変換は無正規化。

逆変換`inverse`は、全spatial軸の元の長さの積で除算する。

```text
normalization = product(global spatial input shape)
```

extra dimensionsは含めない。

R2C/C2Rでもreduced complex shapeではなく、元の実数shapeの積を使う。

無正規化逆変換`backward_unscaled`は初期版には含めない。主要APIは`forward`と正規化済み`inverse`とする。

## 28. FFT workspace

### 28.1 意味論

`TransposeWorkspace`は容量契約だけで複数の転置planへ再利用可能。

FFT workspaceは次へ依存する。

- global/local shape
- exactなextra shape
- stage列
- local FFT scratch要求
- transpose最大必要量
- in-place/out-of-place方式
- FFT backend

したがってFFT planから生成し、非公開`WorkspaceSpec`をplanとworkspaceが`Arc`で共有する。

### 28.2 C2C out-of-place

```rust
pub struct C2cOutOfPlaceWorkspace<R, const N: usize, const M: usize> {
    spec: Arc<C2cOutOfPlaceWorkspaceSpec<N, M>>,
    intermediate: Option<ManyPencilArray<Complex<R>, N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
}
```

`intermediate`は`N=1`で`None`。

主要追加メモリ:

```text
one intermediate storage ~ Lmax
send buffer              ~ Smax
receive buffer           ~ Rmax
local FFT scratch
```

Julia版のout-of-place実装が`ibuf`と`obuf`の二つのFFT中間bufferをping-pong利用するのに対し、Rust版は中間本体一つとin-place中間処理で全配列buffer一つ分を削減する。

### 28.3 C2C in-place

```rust
pub struct C2cInPlaceWorkspace<R, const N: usize, const M: usize> {
    spec: Arc<C2cInPlaceWorkspaceSpec<N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    fft_scratch: Vec<Complex<R>>,
}
```

本体データは`C2cInPlaceArray`が所有するため、全配列中間storageを持たない。

### 28.4 R2C

```rust
pub struct R2cWorkspace<R, const N: usize, const M: usize> {
    spec: Arc<R2cWorkspaceSpec<N, M>>,
    intermediate: Option<ManyPencilArray<Complex<R>, N, M>>,
    transpose: TransposeWorkspace<Complex<R>>,
    c2c_scratch: Vec<Complex<R>>,
    r2c_scratch: Vec<Complex<R>>,
    real_line: Vec<R>,
    complex_line: Vec<Complex<R>>,
}
```

R2C順変換:

```text
real source
-> first R2C directly into complex intermediate
-> in-place complex transitions/stages
-> final C2C directly into complex destination
```

C2R逆変換は逆順で、最後のC2Rをreal destinationへ直接書く。

### 28.5 Workspace生成API

```rust
plan.allocate_out_of_place_workspace()?;
plan.allocate_in_place_workspace()?;
r2c_plan.allocate_workspace()?;
```

同じplanから複数workspaceを生成できる。plan自身は可変workspaceを所有しない。

## 29. FFT planと配列の互換性

実行前に次を検査する。

- source layoutがplan input layoutと一致
- destination layoutがplan output layoutと一致
- exactなextra shapeがplanと一致
- workspace specがplanのspecと一致
- communicator内の全rankが同じ操作方向とplanを使用

C2C in-placeでは追加で:

- 必要な全Pencilが内部`ManyPencilArray`へ登録済み
- C2c stateが方向に対応
- active Pencilが方向に対応
- storage容量が全stageを収容

## 30. エラー型

層ごとに分ける。

```rust
pub enum TopologyError {
    InvalidProcessGrid,
    CommunicatorSizeMismatch,
    InvalidCartesianCommunicator,
    Mpi(/* source */),
}

pub enum PencilError {
    InvalidGlobalShape,
    InvalidDecomposition,
    InvalidPermutation,
    SizeOverflow,
}

pub enum ArrayError {
    StorageLengthMismatch,
    IncompatiblePencils,
    InvalidActiveLayout,
    Poisoned,
    SizeOverflow,
}

pub enum TransposeError {
    IncompatibleTopology,
    IncompatibleGlobalShape,
    UnsupportedDecompositionChange,
    SourceLayoutMismatch,
    DestinationLayoutMismatch,
    ExtraShapeMismatch,
    WorkspaceTooSmall,
    CollectivePreconditionFailed,
    ArrayPoisoned,
    CountOverflow,
    Mpi(/* source */),
}

pub enum FftError {
    InvalidInputLayout,
    InvalidOutputLayout,
    InvalidActiveLayout,
    WorkspaceMismatch,
    UnsupportedInPlaceTransform,
    CollectivePreconditionFailed,
    Transpose(TransposeError),
    Array(ArrayError),
    Backend(/* private */),
}
```

利用者入力・環境エラーは`Result`で返し、panicしない。

内部不変条件の確認には`debug_assert!`を使用できるが、公開入力で到達可能な失敗には使わない。

## 31. テスト戦略

### 31.1 MPIなしの幾何学テスト

- 任意`N,M`の分割範囲
- 不均等分割
- 空担当領域
- decomposition順序
- permutationと逆置換
- logical indexとrow-major offsetの対応
- extra dimensionsがspatial permutationの影響を受けないこと
- overflow拒否

property-based testingを用いて、各グローバル要素がちょうど一rankに属することを検証する。

### 31.2 `PencilArray`

- storage長の完全一致
- logical/memory shape
- `get_local`とflat storageの対応
- mutable borrow排他性

### 31.3 `ManyPencilArray`

- 最大ローカル容量
- active layout
- inactive layoutのview非公開
- successful `overwrite_with`
- failed/panicking `overwrite_with`のpoisoning
- poison状態から全上書きによる復旧
- 重複layout拒否
- topology/global shape不一致拒否

### 31.4 転置

同じテストをAllToAllVとPointToPointへ適用する。

- out-of-place
- in-place
- local copy
- local permutation
- distributed transpose
- 均等・不均等分割
- extra dimensions
- unsorted decomposition
- forward/backward
- multi-stage route
- `N=2,M=1`
- `N=3,M=1`
- `N=3,M=2`
- `N=4,M=2`
- `M=N`のlocal permutation

入力値はグローバル添字から一意に決まる整数値とし、転置後の各rankで期待値を直接照合する。

AllToAllV結果とPointToPoint結果は完全一致を要求する。

### 31.5 Collective失敗

一rankだけで以下を故意に壊す。

- active layout
- extra shape
- workspace容量
- plan signatureまたは方向

全rankが実通信開始前にエラーを返し、デッドロックしないことをタイムアウト付きで検証する。

### 31.6 C2C

- single-rank direct DFTとの比較
- small shapes: 1D/2D/3D/4D
- plane waveの単一ピーク
- random inputをrootへgatherして参照FFT比較
- extra dimensionごとの独立バッチ
- out-of-placeとin-placeの一致
- Input -> Output -> Input状態遷移
- 誤方向呼び出しの事前拒否
- inverse normalization

### 31.7 R2C/C2R

- 偶数長と奇数長
- reduced length `floor(N/2)+1`
- 元実数長の保存
- Hermitian symmetryとの整合
- round trip
- 入力保持
- extra dimensionsを正規化へ含めない

### 31.8 Juliaとの交差検証

同じglobal shape、process grid、decomposition、extra shape、決定論的入力を使用する。

比較対象:

- rankごとのglobal ranges
- local logical shape
- 転置後のglobal logical array
- C2C結果
- R2C結果
- inverse結果

Juliaはcolumn-majorかつextra軸の表面順序が異なるため、物理storageではなく共通のlogical representationへ正規化して比較する。

## 32. 性能とメモリ計測

個別に計測する。

- Pencil構築
- TransposePlan構築
- pack
- MPI通信
- unpack
- local FFT
- distributed FFT全体

通信方式は同一条件で比較する。

rankごとの最大時間を主要指標とする。

メモリは以下を記録する。

- input capacity
- output capacity
- intermediate capacity
- send capacity
- receive capacity
- FFT scratch capacity
- total additional capacity

初期版では性能を理由に正当性検査を外さない。内部fast pathの導入は、測定により必要性が確認され、公開checked APIと同じテストを通過した場合に限る。

## 33. 実装順序

### Milestone 1: 幾何学中核

- axis types
- permutation
- partition ranges
- static `N,M`
- pure geometry property tests

### Milestone 2: MPI topologyとPencil

- Cartesian topology
- subcommunicators
- `Pencil` construction/derivation
- Julia rangesとの比較

### Milestone 3: Array型

- `ExtraShape`
- `PencilArray`
- views
- `ManyPencilArray`
- poison state
- `overwrite_with`

### Milestone 4: Local transpose

- out-of-place local copy/permutation
- in-place local permutation through workspace

### Milestone 5: Distributed transpose

- common `ExchangePattern`
- AllToAllV
- PointToPoint
- checked collective preconditions
- in-place/out-of-place parity

### Milestone 6: Local FFT backend

- RustFFT C2C
- RealFFT R2C/C2R
- row-major batched lines
- input-preserving out-of-place wrappers

### Milestone 7: Distributed C2C

- common stage/path generation
- one-intermediate out-of-place execution
- `C2cInPlaceArray`
- in-place execution
- normalization

### Milestone 8: Distributed R2C/C2R

- generic RealComplex stage
- reduced shape path
- one complex intermediate
- even/odd lengths

### Milestone 9: 交差検証と性能評価

- Julia reference driver
- MPI test matrix
- benchmarks
- memory accounting
- API documentation

## 34. 完了条件

### Array基盤

- 任意のconst `N,M`で`1 <= M <= N`
- row-major logical/memory mappingが検証済み
- extra dimensions対応
- out-of-place/in-place転置
- AllToAllV/PointToPoint一致
- collective事前条件違反でデッドロックしない

### FFT層

- 全spatial軸のC2C out-of-place
- 全spatial軸のC2C in-place
- 全spatial軸のR2C/C2R out-of-place
- `f32`,`f64`
- single-rankおよびmulti-rank参照結果と一致
- Julia logical resultと一致
- 入力保持契約と正規化規約が検証済み

### 状態安全性

- inactive layoutをデータviewとして取得不能
- in-place転置失敗時にpoisoning
- in-place FFT途中失敗時にC2C state poisoning
- 誤方向FFT実行を変更前に拒否

## 35. 将来拡張時に維持する境界

- GPU追加時も`Pencil`へbufferを持たせない
- FFTW追加時もArray crateを変更しない
- in-place R2C追加時も共通`TransformStage`を維持し、異種型共有storageだけを専用化する
- DCT/DST追加時も局所transform variantとして追加し、配置列を再利用する
- 部分FFT追加時はtransform axis pathを公開入力にし、既存の全軸pathを既定値とする
- I/O、reduction、global viewはArray crate上の独立モジュールとして追加する

## 36. 設計上の要点

本設計の中心は次の分離である。

```text
Pencil:
    不変の分散配置

PencilArray:
    一配置の本体データ

ManyPencilArray:
    一つの本体データ + 複数配置 + 現在の物理配置状態

TransposePlan:
    配置間の不変な移動計画

TransposeWorkspace:
    再利用可能な可変通信buffer

TransformPlanCore:
    FFTの軸・配置・転置の論理列

C2cInPlaceArray:
    FFT固有の数学的状態をArray層から分離

FFT workspace:
    一つの中間storageと共通通信workspace
```

この境界により、Julia実装の機能的な構造を保ちつつ、Rustの所有権、alias規則、寿命、状態安全性をAPIへ明示する。
