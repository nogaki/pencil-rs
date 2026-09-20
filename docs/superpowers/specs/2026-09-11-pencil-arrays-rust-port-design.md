# PencilArrays / PencilFFTs Rust移植 設計仕様

日付: 2026-09-11  
状態: Array基盤、LocalTranspose、Alltoallv out/in-place、P2P out/in-place、local C2C、local R2C/C2R out/in-place（raw backward含む）、local R2R、AxisSelection付きAlltoallv/P2P分散C2C FFT out/in-place、AxisSelection付き分散R2C/C2R out/in-place（raw backward含む）、分散R2R/DHT out/in-place、Julia format-6交差検証fixture（68 fixture、110 layout/policy、両方で220）は実装済み。
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
4. 全空間軸または選択軸集合に対する分散C2C FFT
5. 全空間軸または選択軸集合に対するout-of-place R2C/C2R FFT
6. DCT/DST-I-IV分散R2R変換
7. Julia実装と比較可能な正当性・性能評価基盤

分散配列基盤とFFT層は別crateとし、一方向の依存関係にする。`pencil-array`はMPIに依存する
一方でFFTライブラリには依存しない。`pencil-fft`のlocal pathはRustFFT/RealFFTに依存する
がMPIと`pencil-array`には依存せず、`distributed` featureのときだけ分散FFTのために
`pencil-array`とMPIを有効にする。

```text
local FFT (default):
    pencil-array -- MPI
    pencil-fft   -- RustFFT + RealFFT

optional distributed C2C:
    pencil-fft
        depends on
    pencil-array + MPI

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
- DCT/DST-I-IV分散R2Rのout-of-placeおよびin-place実行
- `f32`および`f64`ならびに対応する複素R2Rスカラー
- CPUメモリ上の`Vec<T>`
- RustFFTおよびRealFFTを用いたローカルFFT

### 3.2 初期スコープ外

- in-place R2C/C2R
- GPUストレージ
- FFTWバックエンド
- Chebyshev変換
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
`pencil-fft`のlocal pathはRustFFT/RealFFTに依存するが、MPIと`pencil-array`には依存しない。
`distributed` featureを有効にした`pencil-fft`だけが分散FFTのために`pencil-array`とMPIへ
依存する。

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

    pub fn get_global(
        &self,
        extra_indices: &[usize],
        global_spatial_indices: [usize; N],
    ) -> Option<&T>;

    pub fn get_global_mut(
        &mut self,
        extra_indices: &[usize],
        global_spatial_indices: [usize; N],
    ) -> Option<&mut T>;

    pub fn local_grid<'a, C>(
        &self,
        coordinates: [&'a [C]; N],
    ) -> Result<LocalGrid<'a, C, N>, LocalGridError>;
}
```

`as_slice`の並びは`memory_shape()`に対するrow-major順である。
`get_global`はlocal rangesを検査するだけの非collective accessであり、他rankが
所有する座標、global shape外、empty local rangeは`None`を返す。`permutation`を
通じて物理offsetへ変換するが、MPI通信や要素コピーは行わない。read-only/mutable
borrowed viewも同じglobal accessors（mutable viewは`get_global_mut`を含む）を提供する。

`LocalGrid<'a, C, N>`は各spatial軸のglobal coordinate sliceを借用する。各sliceの
長さは対応するglobal extentと一致しなければならず、grid生成時に検証される。gridは
local rangeのcoordinate slices、local shape、permutationだけを保持し、MPI resourceは
保持しない。`axis`、local logical indexによる`get_local`、`iter`を提供し、`iter`の
tupleはlogical axis順だが順序はarrayのphysical spatial memory orderである。extra axis
はgrid対象外なのでextra batchごとに同じgridを繰り返す。unowned rangeとempty rankは
要素を生成しない。

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

### 17.3 Milestone 4のlocal path

Milestone 4では、将来のcollectiveな`TransposePlan`とは別に、プロセスローカルな`LocalTransposePlan`を先に実装する。15節の`TransposePlanKind::Local`は将来の構成上の接続点であり、この段階で`TransposePlan`へ暗黙に統合しない。

```rust
pub struct LocalTransposePlan<const N: usize, const M: usize> { /* private */ }

impl<const N: usize, const M: usize> LocalTransposePlan<N, M> {
    pub fn new(
        source: Arc<Pencil<N, M>>,
        destination: Arc<Pencil<N, M>>,
    ) -> Result<Self, LocalTransposeError>;

    pub fn execute_views<T: Clone>(
        &self,
        source: PencilArrayView<'_, T, N, M>,
        destination: PencilArrayViewMut<'_, T, N, M>,
    ) -> Result<(), LocalTransposeError>;

    pub fn execute_in_place<T: Clone>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        scratch: &mut Vec<T>,
    ) -> Result<(), LocalTransposeError>;
}
```

`new`は同じ`MpiTopology`、global shape、decompositionの配置だけを受け付け、permutationは同一を含む任意の有効置換とする。local pathはMPI通信、`MpiElement`、通信方式variantを持たない。`execute_views`はexactなextra shapeとlayoutを本体書込み前に検査し、sourceを論理indexごとにcloneしてdestinationのmemory orderへ書く。out-of-placeのclone panicではsourceは不変だが、destinationは部分書込みになり得る。

in-placeはactive sourceと登録済みdestination、checkedな必要要素数、`scratch.capacity()`を先に検査する。scratchへsourceの物理storage順でclone退避が終わるまでarrayを変更せず、退避後の本体書込み直前に`LayoutWriteGuard`で`Poisoned`にする。既存のlogical-index mappingでscratchのsource物理offsetを読み、destination順に本体へ書く。完全な書込み後だけdestinationをcommitし、退避中のclone panicではsource stateを維持し、Poisoned後のclone panicでは`Poisoned`を維持する。

`LocalTransposePlan`の構築・実行はnoncollectiveであり、他rankの呼出しを要求しない。topologyの構築自体がcollectiveである既存契約は変わらない。

## 18. Collective契約

以下はdistributedな`TransposePlan`および分散FFTのcollective operationである。

- `TransposePlan::new`のうちdistributed planを構築する場合
- `TransposePlan::execute`
- `TransposePlan::execute_in_place`
- 分散FFT plan構築のMPI整合性確認
- 分散FFT実行

17.3の`LocalTransposePlan`の構築・`execute_views`・`execute_in_place`はこの一覧に含まれない。これらは通信を行わないprocess-localな操作なので、呼出しrankを揃えずに実行でき、section 18のcollective事前合意も適用しない。

通信開始前に、各rankのローカル事前検査結果をcollectiveに集約する。一つでも失敗すれば、全rankが実通信前にエラーを返す。

将来の汎用distributed planでは専用の`CollectiveDescriptor`を設計できるが、Milestone 7の分散C2C第一PRは後述の最小descriptorを使う。第一PRでは128-bitの`PlanFingerprint`やhashを導入しない。全rankが同じcommunicatorで同じcollectiveを同じ順序に呼ぶことは、いずれの方式でもAPI契約である。

初期版の公開実行APIはchecked実行のみとする。crate-privateなunchecked fast pathは初期版には実装せず、性能測定で必要性が確認された後に別途設計する。

### 18.1 Milestone 5最初のPRに対する追補（2026-09-14、Alltoallv out-of-place実装済み）

この追補は、sections 14--18にある将来の総合`TransposePlan`案を変更せず、
最初の小さな分散転置PRへ適用する範囲だけを定める。PRで公開する型は次の
`AllToAllv`専用APIだけとする。

```rust
pub struct AllToAllvTransposePlan<const N: usize, const M: usize>;
pub struct AllToAllvTransposeWorkspace<T>;
pub struct AllToAllvTransposeWorkspaceRequirements {
    pub send_len: usize,
    pub receive_len: usize,
}
pub enum AllToAllvTransposeError { /* checked and collective errors */ }
```

planの`new(source, destination)`と`execute_views(source, destination,
workspace)`を提供し、workspaceは初期化済み`Vec<T>`を受け取る
`from_vecs(send, receive)`で構築する。`T`の実行制約は独自traitではなく、
`mpi::datatype::Equivalence + Copy`とする。`execute` wrapper、in-place、
point-to-point、`TransposeMethod`、総合`TransposePlan`、FFTはこのPRでは作らない。

分散planは同じ`MpiTopology`、global shape、ちょうど一つだけ異なる
ordered decompositionを要求し、permutationは任意とする。decompositionが
同じ場合は既存`LocalTransposePlan`を使い、AllToAllv planは全rankのpreflight後に
拒否する。全rankが同じsource topologyのCartesian communicator contextで、
plan構築・実行を同じcollective順序で呼ぶ。local APIとAllToAllv APIをrank間で
混在させる呼出しは許可しない。

plan構築・実行とも、payloadやaxis subcommunicatorの通信より先にsource
topology全体でscalar合意を行う。constructorとexecuteの各固定headerでN、M、
canonical descriptor長をmin/maxで確認する。extra rank、extent、T descriptorの
長さはdescriptor内部に明示し、別の可変長collectiveを呼ばない。その後だけcanonical
`u64` descriptorと同じ長さのmin/max受信配列をfallibleに
確保し、準備成功をscalarで全rank合意する。続いてnativeな要素別min/max
`all_reduce`を二回行い、min == maxならword毎のexact一致とする。全rank分の
巨大配列や独自hashは作らない。global shape、decomposition、permutation、変更軸、
AllToAllv mode、ordered source-to-destination direction、extra shapeのrankと
内容、Tのtype identity/size/alignmentを比較する。rank固有のlocal lengthや
workspace lengthはdescriptorに含めない。既存section 18の独自128-bit
`PlanFingerprint`はこのPRでは採用せず、nativeなexact descriptor比較を使う。
必要性は性能測定後に別途判断する。

変更されたtopology軸の1次元subcommunicatorについて、peer rank `p`から
`rank_to_coordinates_into(p, ...)`で座標を得る。peer rankと座標を同じ値と仮定しない。
送信regionはsource local boxとpeer destination boxの交差、受信regionはpeer
source boxとlocal destination boxの交差とする。pack/unpackはpeer rank順、extra
batch row-major順、logical spatial row-major順を一致させる。

rsmpi 0.8.2では`mpi::Count`は`i32`であり、
`datatype::Partition::new`/`PartitionMut::new`を有効長sliceへ適用して
`CommunicatorCollectives::all_to_all_varcount_into`を一回呼ぶ。count、
displacement、count + displacement、総buffer長を先にchecked検査し、内部assert
へ不正値を渡さない。workspaceは`len`以内の初期化済み領域だけを使い、capacity
だけへの書込み、`MaybeUninit`、実行時のbacking storage拡張は行わない。

layout、extra、workspace、count/displacement、offsetなどの一rankのlocal errorは
全rankでvalidityを合意してから返す。out-of-place `execute_views`のpreflight errorではdestinationを書かず、
sourceは常に保持する。preflight成功後だけpack、Alltoallv、unpackを行う。

このPRのplan内部はpeerごとの交差region、spatial count、displacementなど
O(peers*N)のmetadataだけを保持する。全要素のsend/receive offset配列は保持しない。
pack/unpackはregion内をlogical row-major順に走査し、既存のmappingとpermutationを
使ってoffsetをchecked計算する。newは各ローカル要素を列挙せず、巨大shapeでも
metadataだけを扱う。offsetの事前検査に必要な追加走査はデータを書かない。

constructorとexecuteはschema/version、operation、N、M、descriptor word数を含む
共通の固定サイズheaderを先に合意する。headerが不一致なら同じ結果を全rankが見て
その時点でreturnする。header一致後にdescriptor準備成功をscalarで合意し、二つの
native min/max reductionでexact比較する。extra/typeの長さがdescriptorに明示される
場合は長さ専用collectiveを重ねない。どの通常Err経路でもrankを後続collectiveへ
置き去りにしない。

`type_name`、size、alignおよびEquivalenceの記述比較は、誤ったunsafe
Equivalence実装や同名異型を証明するものではない。全rankが同じ`T`とMPI表現、
正しいEquivalence、同一communicator、同一collective順序を使う契約を別途守る。
型記述比較はu32/u64などのよくある誤用検出であり、MPI datatype handle値をrank間で
比較しない。

workspaceは初期化済みVecのlenで検証し、execute中にresize/reallocしない。`T`は
`Equivalence + Copy`、pack/unpackは代入コピーで、Clone/Default callbackを呼ばない。
小さいmetadata割当は必要なら`try_reserve`で失敗をcollectiveに合意する。MPI実行時
障害、任意のpanic、プロセス喪失では、既存binding同様にResult回収を保証しない。

### 18.2 Alltoallv in-place追補（2026-09-14、基点 `f87bb23`）

18.1の専用Alltoallv APIに、既存workspaceを使う次の後続操作を追加する。

```rust
impl<const N: usize, const M: usize> AllToAllvTransposePlan<N, M> {
    pub fn execute_in_place<T>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut AllToAllvTransposeWorkspace<T>,
    ) -> Result<(), AllToAllvTransposeError>
    where
        T: mpi::datatype::Equivalence + Copy;
}
```

実行の最初に、source topology全体のCartesian communicatorで、schema、
専用のin-place operation code、`N`、`M`、descriptor長を含む固定headerを
既存のmin/maxで比較する。`new`、`execute_views`、`execute_in_place`の混在は
header段階で全rankが拒否し、後続の可変長descriptorやaxis subcommunicatorへ
進まない。header後も、planの方向・変更軸・topology・shape・decomposition・
permutation、exactなextra shapeのrank/extent、`T`のtype name/size/alignment、
`Equivalence::Out`のtype nameを既存のnative word-by-word min/maxで比較する。

その後のlocal preflightはsource topology全体でvalidityを合意してから結果を返す。
activeがPoisoned、source layout不一致、destination未登録、workspace不足、
checked length/count/displacement/offset不備など、一rankの失敗でも他rankを後続
collectiveへ置き去りにしない。Poisonedなarrayからも`extra_shape`とplan metadataは
header/descriptor用に取得する。sourceとdestinationの必要prefixは別々にchecked計算し、
`ManyPencilArray`が持つ「最大registered local layout分のstorage」という既存不変条件を
利用するため、local lengthの一致やstorage拡張を仮定しない。通常のpreflight errorでは
arrayのstateと内容を変更しない。

全rankのpreflight成功後だけ、Validなactive source prefixをpackし、既存の一回の
`MPI_Alltoallv`を実行する。zero/片方向empty、`mpi::Count`、Partition境界、初期化済み
workspaceの`len`だけを使う契約は18.1のまま維持し、resize、再allocation、別workspace、
別trait、dummy通信方式は追加しない。MPI完了後、本体書込み直前に既存
`begin_in_place_write`でPoisonedにし、destination必要prefixへunpackを完了してから
commitする。成功したin-place実行ではactive sourceはdestination内容に置き換わる。
unpackはdestination `Pencil`、mutable slice、receive buffer、extra countを受け取れる
共通helperとし、guard配下storageにも使える。local lengthが異なる場合もsource/destination
それぞれのprefixを使い、prefix外の余剰tailは書き換えない。

`T: Copy`なのでpack/unpackにClone/Drop callbackはなく、panic時のPoisoned維持は既存
`LayoutWriteGuard`の構造と既存テストに委ねる。公開panic hookやunsafeな不正状態注入は
追加しない。MPI障害、任意panic、プロセス欠落後のglobal recoveryは保証しない。

### 18.3 PointToPoint out-of-place追補（2026-09-15、基点 `159a16d`、実装済み）

実装計画: [Point-to-point transpose plan](../plans/2026-09-15-point-to-point-transpose-implementation.md)。
15.5の将来の汎用`PointToPointPlan`案をこの段階で公開せず、専用のchecked APIだけを
追加する。P2P in-place、R2C/C2R、分散FFT、`WaitAny`によるunpack重畳、性能用fast pathは
含めない。
`TransposeError`、`TransposeWorkspace<T>`、`TransposeWorkspaceRequirements`は
Alltoallvと共有する通信方式非依存の実体とし、既存の
`AllToAllvTransposeError`、`AllToAllvTransposeWorkspace<T>`、
`AllToAllvTransposeWorkspaceRequirements`はそれぞれ同じ型のre-export aliasとして
維持する。P2P専用workspaceや同型のerror/requirementsは作らない。

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

`new`と`execute_views`はsource topology全体のCartesian communicatorで、方式と操作を
組み合わせた5語の固定header
`[schema, combined_operation, N, M, descriptor_len]`を最初にmin/max比較する。既存
Alltoallvの`new`、`execute_views`、`execute_in_place`は1、2、3を維持し、P2Pの
`new`、`execute_views`は4、5を使う。方式と操作は一つのwordで区別するため余分な
header wordは追加しない。`new`、P2P views、Alltoallv views、Alltoallv in-placeを混在
させたrankは、このheaderで可変descriptorまたは変更軸subcommunicatorへ進まず拒否する。
header後は既存と同じfallibleな準備成功の合意とnative `u64` wordごとのmin/max二回で
exact descriptorを比較する。canonical descriptorにもcombined operation（必要なら
方式固有の固定tag）を含め、方向、変更軸、topology/grid、
global shape、ordered decomposition、permutationを持ち、実行時にはexactなextra
shape（rankとextents）、`T`のtype name/size/alignment、`Equivalence::Out`の型名も
持つ。local layout、workspace、count、offsetなどの失敗は全Cartesian rankでvalidityを
合意してから返し、成功rankがpeer通信へ先行しない。

データ通信は変更軸の`MpiTopology`内部subcommunicatorだけで行い、固定予約tag
`POINT_TO_POINT_RESERVED_TAG = 0x5054`を使う。内部communicator contextは利用者通信
と隔離する。同一contextで未完了の転置を重ねず、公開呼出しは全request完了後に戻る。
zero-count peerはsend/receiveを個別にpostしなくてよいが、全体のpreflightは全rankで
行う。self peerは通常の同じsubcommunicator rankとして処理する。検証の大payloadは
非zero peerごとに `128 * 512` 個の `u64`、`524,288` bytes（512 KiB）であり、
大きなメッセージ経路を実行するが、MPI実装ごとのeager/rendezvous選択や閾値は保証
しない。

Alltoallvと同じ初期化済みworkspaceの`len`だけを使い、backing `Vec`をresize/realloc
しない。全てのcount、displacement、総量、offsetとrequest capacityの準備をpack前に
checkedに完了する。初期化済みworkspaceの連続prefixを`split_at_mut`で直接postするため、
追加のsegment-holder Vecは作らない。15.5の一般案にある受信waitと送信waitの間のunpackはこの段階では
採用せず、両方のrequestを先に完了させる。`mpi::request::scope`内の`Vec<Request>`は最初のpost前に
`try_reserve`し、その成功を全Cartesian rankで合意する。成功後の順序は
「全Irecv、全Isend、全requestのwait、scope終了、unpack」で固定する。requestが借用する
send/receive segmentとworkspaceはwait完了まで生存させ、requestをplan/workspaceへ
保存せず、未完了のまま`forget`してreturnしない。利用するのはrsmpi 0.8.2の
`immediate_receive_into_with_tag`、`immediate_send_with_tag`、`Request::wait_without_status`
であり、raw FFI、`unsafe`、`MaybeUninit`、実行時の`Default`/`Clone` callbackは追加
しない。MPI障害、任意panic、プロセス喪失後のglobal recoveryは既存契約と同じく保証
しない。

### 18.4 PointToPoint in-place追補（2026-09-15、基点 `198638e`、実装済み）

実装計画: [P2P in-place計画](../plans/2026-09-15-point-to-point-in-place-implementation.md)。
18.2のAlltoallv in-place状態契約と18.3のP2P transport契約を、既存の
共有workspace/error型に対して組み合わせた。追加した公開APIは次だけである。

```rust
impl<const N: usize, const M: usize> PointToPointTransposePlan<N, M> {
    pub fn execute_in_place<T>(
        &self,
        array: &mut ManyPencilArray<T, N, M>,
        workspace: &mut TransposeWorkspace<T>,
    ) -> Result<(), TransposeError>
    where
        T: mpi::datatype::Equivalence + Copy;
}
```

固定headerは18.3の5語
`[schema, combined_operation, N, M, descriptor_len]`を維持する。既存の
operation 1（Alltoallv new）、2（Alltoallv views）、3（Alltoallv in-place）、
4（P2P new）、5（P2P views）は変更せず、6をP2P in-placeに割り当てる。これに
より5語headerの段階で`new`、views、Alltoallv in-place、P2P in-placeの混在を
全rankが拒否し、descriptorまたは変更軸subcommunicatorへ進まない。

実行descriptorはarrayの`active_view`を要求せず、`array.extra_shape()`とplan
metadataから作る。arrayがPoisonedでもextra shape、方向、変更軸、topology、
global shape、ordered decomposition、permutation、`T`の型記述を全rankで比較
する。descriptor後のactive state、source layout、destination登録、source/destination
のchecked必要prefix、workspace length、count/displacement、region offsetの
local結果は、Cartesian communicatorで合意してから返す。一rankの失敗でもpack、
request、guardへ進まない。sourceとdestinationのlocal lengthは一致すると仮定せず、
最大registered storageのdestination prefix外のtailは保持する。

Alltoallv in-placeとP2P in-placeは、上記preflightとpost-transferのguard、
`unpack_destination`、`commit`を小さいcrate-private処理として共有してよい。
P2Pは18.3の固定`POINT_TO_POINT_RESERVED_TAG`（`0x5054`）とtopology所有の
変更軸subcommunicatorを使い、同じ内部contextで未完了転置を重ねない。source slice
からworkspaceへpackするtransport helperは、request `Vec`のfallible reserveと
その成否の全rank合意をpack・最初のpostより前に完了する。成功後の順序は、全Irecv、
全Isend、全requestのwait、scope終了、destination prefixへのunpack、commitとする。
wait中およびscope中にarrayを書き換えず、requestが借用するworkspace segmentを
wait完了まで保持する。zero/片方向emptyとzero-extraではcountが0のpeerだけ
requestを省略し、self peerもcountが非zeroなら他のpeerと同じIrecv/Isendを行う。

全request完了後だけ`begin_in_place_write`でPoisonedに遷移し、destinationの必要
prefixを全てunpackしてからdestinationをcommitする。通常のpreflight errorでは
arrayのstate・内容とworkspaceを保持する。MPI故障、任意panic、プロセス喪失後の
global recoveryは保証しない。新workspace/error/trait、unsafe、WaitAny、R2C/C2R、分散FFT、
test専用panic hookはこの追補に含めず、panic時のPoisoned保証は既存
`LayoutWriteGuard`に委ねる。

検証は既存`alltoallv_transpose.rs`の一つのMPI initialize、fixture、success helperを
共用し、1/4/6 rank、u64/f64、2D/3D、非正方・逆順communicator、両方式との完全一致、
独立oracle、往復、長さ変化、tail、empty/zero payload、失敗後のworkspace再利用、
rank-local preflightと全operation混在をtimeout付きで確認した。公開README、lib.rs、
既存one-rank doctest、CI suite説明もP2P in-placeを反映した。local C2Cは実装済みで、
R2C/C2Rと分散FFTを後続に残す。

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

`AxisSelection<N>`で選択したspatial軸を変換し、未選択軸はidentity stageとして配置routeに
残す。既存constructorは全軸選択を既定値とする。extra dimensionsは全てバッチ軸である。

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

非空の`AxisSelection<N>`から最大Rust軸`r`を選び、その軸を最初にR2Cする。
未選択軸はidentity stageとしてrouteには残し、選択された`r`だけを
`n/2+1`へreductionする。

実数global shape:

```text
[N0, ..., Nr, ..., N(N-1)]
```

複素global shape（`r`のみreduced）:

```text
[N0, ..., floor(Nr/2)+1, ..., N(N-1)]
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

    pub fn backward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut R2cWorkspace<R, N, M>,
    ) -> Result<(), FftError>;
}
```

`inverse`は正規化済み、`backward`は正符号・無正規化である。in-place APIは
初期版に存在しない。

元の実数長をplanへ必ず保存する。複素長だけでは偶数・奇数の元長を一意に復元できない。

RealFFTが入力bufferをscratchとして変更するため、out-of-place APIの入力保持には1本のline bufferを使う。

## 27. 正規化

順変換は無正規化。

逆変換`inverse`は、選択されたspatial軸の元の長さの積で除算する。

```text
normalization = product(selected spatial input extents)
```

extra dimensions、identity軸、reduced complex extentは含めない。C2Cでも同じ規則を
使い、空の選択はtransposeだけのidentityとなる。

`forward`は無正規化、`inverse`は正規化済み、`backward`は正符号の無正規化である。
R2C/C2Rの`backward`は選択されたreal boundary長とcomplex tail長の積だけをforward結果に
掛け、extra次元、identity軸、reduced complex長は正規化係数に含めない。

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

- RustFFT C2C（実装済み）
- RealFFT R2C/C2R（実装済み）
- row-major batched lines
- input-preserving out-of-place wrappers

#### Milestone 6最初のPR追補（2026-09-15、基点 `28b1358`）

上記Milestone 6の最初のPRは、後続のRealFFTおよび分散FFTとは分け、local C2C
だけを実装対象とした。この追補はsections 20、27、28、31にある全体設計のうち、
現在実装済みのlocal C2C範囲を限定する。

- 新crate `pencil-fft`は追加済みだが、`pencil-array`のmanifest、公開API、ソースは
 変更しない。local slice処理は`pencil-array`やMPIを使わず、両方を
 `pencil-fft`の依存にも入れない。分散FFT統合時に必要な依存だけを後続で追加する。
- MSRV 1.61のRustFFT 6.4系をC2C backendとして使う（workspace MSRV 1.85に適合）。
 RealFFT/R2C/C2R、分散FFT、FFTW、GPU、公開`FftBackend` trait、stage framework、
 未使用feature flagはこのPRに含めない。
- 公開境界はsealedな`FftReal`（実装は`f32`/`f64`のみ）と、1D line lengthを
 runtimeに持つ`LocalC2cPlan<R>`とする。公開signatureにはRustFFTのplan、direction、
 `FftNum`などを要求しない。複素値はbackend-neutralな`num_complex::Complex<R>`を
 使い、RustFFTのre-exportを使う場合も型自体をbackend型にしない。
- `forward`/`inverse`は同じlengthの連続batch sliceを受けるout-of-place API、
 `forward_in_place`/`inverse_in_place`は最小のin-place APIとする。forwardは無正規化、
 inverseは各lineの`n`だけで除算し、batch数は含めない。
- このMilestone 6時点では、local C2Cへ正符号・無正規化の
  `backward`/`backward_in_place`を追加した。これは当時のlocal-only境界を記録した
  記述であり、後続のMilestone 7第四PR追補がdistributed C2Cのraw backward APIを
  追加してこの範囲を更新する。R2C/C2RのAPIは変更しない。
- planはimmutableなforward/inverse計画とlength/scratch metadataだけを持ち、
 scratchはcaller-ownedの初期化済みslice/Vecとする。RustFFTの
 `get_immutable_scratch_len()`および`get_inplace_scratch_len()`を正逆計画について
 問い合わせ、公開する一つのscratch requirementはその最大値とする。out-of-placeは
 入力を保持する`process_immutable_with_scratch`を使い、入力を変更し得る
 `process_outofplace_with_scratch`で代用しない。in-placeは`process_with_scratch`を使う。
 `n`とscratch長は同じとは仮定しない。
- `n=0`、checked計算で表現できないlength、非整数batch、src/dst長不一致、
 scratch不足はbackend呼出しおよび出力更新前に`Result`で返す。zero batchは長さと
 scratchを検査した後にno-opとし、backendへ空sliceを渡さない。`new(0)`は有効な
 `new(1)`のzero batchとは別に扱う。plannerの資源枯渇や任意backend panicを
 `Result`へ回収する保証はなく、unsafeは使わない。
- このPRのlocal C2CテストはMPIなしで実行し、小さい独立直接DFTとのforward/inverse
 直接比較を必須とする。符号、単一周波数、正規化、f32/f64、single/multiple batch、
 prime/composite/length1、zero batch、invalid length/destination/scratch、未変更保証、
 scratch再利用を確認する。単なる往復だけをoracleにしない。

#### Milestone 6第二PR追補（2026-09-17、基点 `f5f4658`）

local `pencil-fft`へRealFFT 3.5系を追加し、既存のC2C APIを変更せず、1次元の
out-of-place R2C/C2Rだけを実装した。`pencil-array`、MPI、分散FFT、in-place real FFT、
共通workspace abstraction、公開backend traitはこのPRにも含めない。

- 公開APIはsealed `FftReal`（`f32`/`f64`のみ）、`Complex<R>`、
  `LocalR2cPlan<R>::new`、`real_len`、`complex_len`、`scratch_len`、`forward`、`inverse`。
  forwardのsource/destinationはreal line長`n`とreduced complex line長`n/2+1`、inverseの
  source/destinationはreduced complex line長`n/2+1`とreal line長`n`の連続batchで、batch数は
  各sliceを方向に応じて除算して一致を確認する。in-place real APIはない。
- planは`realfft`のimmutable forward/inverse plan、元の正の`n`、reduced length、
  両native `get_scratch_len()`の最大値だけを所有する。forwardにはcaller-owned real line、
  inverseにはcaller-owned complex line、両方にcaller-owned initialized complex scratchを
  渡す。oversized tailは使わず、実行中のallocation/resizeは行わない。
- forwardは無正規化、inverseはRealFFT出力を各lineの元の`n`だけで除算する。両方向とも
  sourceは保持し、RealFFTが入力lineを変更するため一度に一lineだけcaller bufferへcopyする。
- inverseはbackend実行前に全batchのDC、偶数`n`のNyquistのimaginary componentを検査する。
  `+0.0`/`-0.0`だけがzeroで、NaNは不正。odd`n`が1より大きい場合のfinal binは制約せず、
  interior binはいずれの長さでも制約しない。`n=1`のfinal binはDCなので制約する。通常のvalidationおよびこのendpoint errorでは
  source、destination、line/scratch workspaceを
  保持する。backend/resource panicや検証済み不変条件後の予期しないbackend errorをResultへ
  変換せず、output atomicityも保証しない。
- `n=0`、address-spaceを超えるreal/complex/odd staging長、非整数batch、batch数不一致、
  短いline/scratchはbackend前に拒否する。valid empty batchもline/scratch検査後にworkspaceを
  変更しないno-opとする。テストは独立DFT、実数入力のC2C結果とのhalf-spectrum prefix比較、
  Hermitian reconstruction、f32/f64、odd/evenのprime/composite、n=1/2、複数batch、DC/Nyquist、
  任意half-spectrum、dirty/oversized workspace、later-batch endpointのatomicityを確認する。

#### Milestone 7最初のPR追補（2026-09-17、Alltoallv C2C）

Milestone 7の第一PRでは、将来のgeneric `TransformPlanCore`を先取りせず、
`pencil-fft`のfeature-gatedなC2C専用実装だけを追加する。18.1--18.4の既存
transpose transport固有descriptor metadataはこのC2C APIには引き継がない。デフォルト
featureは空のままなので、`cargo test -p pencil-fft --no-default-features --locked`はMPIと
`pencil-array`を有効化しない。

- `distributed` featureの公開APIは`C2cPlan`、
  `C2cOutOfPlaceWorkspace`、`FftError`である。対象は`N >= 2`、
  `1 <= M < N`、identity permutationと`[0, ..., M-1]` decompositionの入力で、
  `from_pencil`、`from_array`、`from_shape`を提供する。forwardは無正規化、
  inverseは各spatial軸のlocal inverseを一度ずつ使う正規化済みであり、extra
  batchは正規化に含めない。両方向とも入力を保持し、出力を書き込む。workspaceは
  再利用可能なmutable scratchであり、実行中に更新される。
- routeはaxis `N-1`から`0`へ進み、memory tailに移したaxisが未分割になるように
  stageを作る。same-decomposition edgeはlocal transpose、decompositionが一箇所だけ
  変わるedgeはchecked Alltoallvとし、唯一の中間`ManyPencilArray`を使う。公開in-place
  FFT、P2P FFT、R2C/C2R、generic marker hierarchyはこのPRに含めない。
- `LocalTransposePlan`には`TransposeWorkspace`の初期化済みsend prefixへ直接stageする
  narrow adapterを追加する。adapterはsend Vecのlength、capacity、tailを変更せず、
  既存のVec APIはcapacity契約のまま残す。`MpiTopology::communicator()`は所有権を移さない
  借用native Cartesian communicatorであり、`self`のdropまたはMPI finalizationまで有効で、
  同じcontext/order/countのcollective契約を持つ。追加のraw accessorは作らない。
- 分散C2Cの計画生成、forward、inverseは、固定5語header
  `[schema, operation, N, M, descriptor_len]`（`schema = 1`、operationは計画生成`7`、
  forward`8`、inverse`9`）を全Cartesian communicatorでmin/max比較する。header後の
  最小descriptor payloadは`global_shape`、`process_grid`、exactなextra shapeのrankと
  dimensions、sealed scalarのbyte widthだけであり、全stage/transition metadata、
  type name/alignment、`Equivalence::Out`、fingerprintは含めない。canonical routeは
  `N`、`M`、global shape、process gridから決定的に再生成する。実行時はactual source/
  destination layoutを各rankでplanと照合してからfull Cartesianでvalidityを合意し、合意
  前にbackend、FFT、workspace、destinationを変更しない。zero global extent、型/N/M/
  shape/extra/layout/workspaceの不一致は初期preflightで全rankが`Err`を返す。
  zero extra batchとempty local rankは有効なまま残す。初期preflightのErrはsource、
  destination、workspaceを保持するが、実行開始後のresource failureまたは後段Alltoallv
  metadata failureはworkspaceを変更し得る。全実行をallocation-freeまたはrollback可能
  とは約束しない。
- MPI統合テストは一つのbinaryと一つのtop-level testで、1/4/6 rank、f32/f64、
  `N=2,3,4`、`M=1,2`、direct DFT oracle、逆変換、入力保持、zero batch、空local領域、
  workspace再利用、negative collective phasesを検証する。README、CI、feature付きdoc/
  clippy/checkコマンドはこの境界を明記する。

### Milestone 7: Distributed C2C

- common stage/path generation
- one-intermediate out-of-place execution (第一PRで実装)
- normalized distributed C2C inverse (第一PRで実装)
- unnormalized positive-sign distributed C2C backward (第四PRで実装)
- `C2cInPlaceArray` and in-place execution (第二PRで実装済み)

#### Milestone 7第二PR追補（2026-09-18、基点 `cbd1c17`）

実装計画: [分散C2C in-place計画](../plans/2026-09-18-distributed-c2c-in-place-implementation.md)。

分散C2Cの第二PRとして、既存のAlltoallv route/core、local FFT、transitionを
共有するsingle-buffer in-place APIを追加した。pencil-arrayのソース変更や新しい
transport/backend abstractionは行わない。

- `distributed` featureの公開APIに`C2cInPlaceArray`、`C2cInPlaceWorkspace`、
  `C2cState`を追加した。arrayはprivateな`Arc`、一つの`ManyPencilArray`、stateだけを
  所有し、workspaceはprivateなplan `Arc`、`TransposeWorkspace`、native scratchだけを
  所有する。公開されるのは`state`とInput/Output時だけの`view`/`view_mut`であり、
  raw storage、Many access、state setter、recovery APIはない。
- `Input -> Poisoned -> Output`、`Output -> Poisoned -> Input`を実行契約とした。
  固定5語headerのoperation 10/11を予約し、descriptorは第一PRと同じものを借用する。
  header、descriptor、全Cartesian preflight（state、active layout、exact extra shape、
  private allocationのArc provenance、initialized workspace長）の成功後、最初のlocal
  FFT前にPoisonedへ変更し、全stage成功後だけtarget stateへcommitする。Errまたはpanicでは
  Poisonedを維持し、rollbackや本番catch_unwindは行わない。
- arrayの一つのMany bufferへ全local FFTをin-placeで適用し、既存のforward/reverse
  transitionをそのまま実行する。workspaceへ中間arrayを複製せず、inverseは既存local
  inverseの各空間軸正規化だけを使う。empty local rankとzero extra batchも全transitionへ
  参加する。
- 一つのMPI integration binary/top-level testへf32/f64、N=2/3/4、M=1/2、非均等・空領域・
  zero extra・逆順communicator、OOP/DFT比較、任意spectrum inverse、再利用、状態/endpoint、
  rank-local preflightとAPI混在を追加した。private transaction helperの実際のErr/panic
  経路はfeature付きunit testでcatch_unwindを外側から使ってPoisonedとview gateを確認する。
  README、public doctest、privacy compile-fail、CI suite名も反映した。

#### Milestone 7第三PR追補（2026-09-18、分散C2C PointToPoint）

実装計画: [分散C2C PointToPoint計画](../plans/2026-09-18-distributed-c2c-point-to-point-implementation.md)。

第一・第二PRのlocal C2C、route、in-place状態契約、共有workspaceを維持し、分散edgeのtransportだけを選択可能にした。新しいpublic backend trait、factory、workspace型、P2P request処理の複製は追加しない。

- `distributed` featureに`TransposeMethod::{AllToAllv, PointToPoint}`を追加した。既存の`from_pencil`、`from_array`、`from_shape`はAlltoallvへ委譲し、`*_with_method`の3 constructorが明示選択を受ける。入力保全、inverseの各spatial軸正規化、`N >= 2`、`1 <= M < N`、extra shape、array/workspaceのplan identityは不変である。
- 固定5語headerとoperation 7--11を維持する。schema=2のpayloadは`global_shape[N]`、`process_grid[M]`、extra rank/dimensions、canonical axis mask[N]、value kind、scalar width、method word（Alltoallv=0、PointToPoint=1）を含み、checked長を`2*N + M + 4 + extra_rank`とする。constructorと全実行のdescriptorがselection、value kind、precision、methodをnative FFT、destination/workspace書込み、in-place poisonより前に合意するため、rank-localな不一致は全rankで`CollectiveDescriptorMismatch`となる。
- 分散edgeは選択methodの既存`AllToAllvTransposePlan`または`PointToPointTransposePlan`のforward/backward pairを保持する。各edgeのforward作成後とbackward作成後に、既存のcollective requirement agreementを行い、最大send/receive長と既存`TransposeWorkspace`を共有する。local transition、FFT scratch、OOP/IP loopは変更しない。
- P2Pのchanged-axis topology context、固定`0x5054` tag、receive-before-send、wait-all、request metadata reservation、MPI failure/panic/process-lossの回復不能契約はarray crateから継承する。同一context上の未完了transposeを重ねず、成功時はnative requestを完了して返る。allocation-free executionやglobal rollbackは約束しない。
- 既存の一つのMPI integration binary/top-level testを両methodで実行し、f32/f64、N=2/3/4、M=1/2、非均等・empty local、extra/zero extra、逆順communicator、direct DFT、OOP/IP parity、入力保全、pointer stability、constructor default、transport parityを確認する。constructorとOOP/IP forward/inverseのrank-local method mismatchは全workspace・state・dataをsnapshotして再利用まで検証し、6-rank multi-axisのchanged-axis subgroup外rankも含める。private one-rank poison testは両methodのErr/panicとpost-start native errorを確認する。

#### Milestone 7第四PR追補（distributed C2C raw backward）

既存route、Alltoallv/PointToPoint transition pair、workspace、in-place transactionを共有し、
`C2cPlan::backward`と`backward_in_place`を追加する。両APIはdefaultではreversed output layoutを
source（明示layoutではそのpolicy）、canonical input layoutをdestinationとし、正符号の無正規化local backwardを全stageで
呼ぶ。`inverse`は従来どおり各stageで正規化し、raw forward/backward roundtripだけが
selected spatial extentの積（identity軸とextra dimensionsを除く）を掛ける。

固定5語headerのschema=2、selection/value-kind/precisionを含むdescriptor、operation 1--14は維持する。C2C raw
out-of-place/in-placeにはoperation 15/16を割り当て、raw/normalized、raw/forward、
OOP/in-placeの取り違えをfull Cartesian headerでnative FFT、payload、workspace、
state変更より先に拒否する。逆routeのOutput -> Poisoned -> Inputはinverseとbackwardで
共通であり、初期失敗では全resourceを保持し、開始後のin-place Err/panicではPoisonedを
維持する。共有complex tailのR2C inverse callerは既存の正規化指定を明示するだけで、
R2C API、endpoint policy、projection、数理は変えない。

独立positive-sign DFT、raw scaling、両transport、f32/f64、N/M、extra/zero batch、empty
local、reordered communicator、OOP/IP、descriptor rejection、post-start poisoningを既存
MPI suiteとunit transaction testで確認する。Julia/FFTWのformat 6はcanonical
`selected_axes`、`element_kind`、`axis_kinds` metadataを持つ。C2C、R2C、R2Rすべてに
`backward_expected`を持ち、R2Rはpaired kindのraw resultとする。


### Milestone 8: Distributed R2C/C2R

- generic `RealComplex` stage（実装済み）
- reduced shape path（実装済み）
- one complex intermediate（実装済み）
- even/odd lengths（実装済み）
- out-of-place forward/inverse/backward over Alltoallv and PointToPoint（実装済み）

#### Milestone 8追補（2026-09-18、分散R2C/C2R out-of-place実装済み）

実装計画: [分散R2C/C2R計画](../plans/2026-09-18-distributed-r2c-implementation.md)。この追補は
現在の実装境界を記録し、以前の追補にある将来計画・履歴の記述は変更しない。

The distributed feature exposes `R2cPlan<R, N, M>`, `R2cWorkspace`, and
`R2cError` only when `distributed` is enabled. `N >= 2`, `1 <= M < N`, and
the input pencil is canonical identity with decomposition `[0..M)`. A non-empty
`AxisSelection<N>` chooses its largest Rust axis as the real boundary; only
that selected extent `n` becomes `m = n/2+1`. Constructors mirror `C2cPlan`,
including both checked transports, and allocation is noncollective. At this
2026-09-18 out-of-place snapshot there was no real in-place API; the current
single-allocation addition is recorded in the dated section below. Empty R2C
selections are collectively rejected.

The selected boundary stage is `LocalR2cPlan::forward`, `inverse`, or
`backward`; selected axes below it are complex FFTs and unselected stages are
identities. The reduced canonical pencil is never passed through a full C2C
plan at the boundary, so the reduced axis is not transformed again. A
workspace owns one reduced-complex `ManyPencilArray`, shared transpose storage,
the maximum native complex scratch, and one real line plus one complex line.
Inverse scaling is exactly once per selected original spatial axis, including
`n`; extra batch dimensions, identity axes, and `m` are never factors.

Inverse boundary acceptance is post-tail and per extra batch/per constrained
plane. DC is constrained always; Nyquist is constrained only for even `n`,
while an odd final bin is unconstrained. For each plane `z(x)`, all real and
imaginary values must be finite and the plane is accepted when
`max_x abs(Im z) <= 128*min_subnormal_R*D` or
`||Im z||_2 <= 128*epsilon_R*D*||Re z||_2`, with
`D = 1 + sum(ceil(log2(n_a)))` for selected axes below the real boundary.
This is an explicit normwise-relative/componentwise-absolute policy, not a
formal RustFFT error bound. Fixed four-word max and four-word sum arrays are reduced
across the full Cartesian communicator for each batch; nonfinite values are
excluded from MAX/SUM but set a validity flag. Every rank completes all
reductions, including empty ranks, and one final MIN-valid reduction occurs
before any real destination write. Invalid boundaries return
`R2cError::InvalidSpectrum`; the source and destination stay unchanged, while
workspace mutation is permitted after execution has started. Accepted endpoint imaginary values are zeroed only in the private
intermediate before the strict local C2R operation. `backward` uses the same
relative criterion
and finite endpoint requirement, but its absolute threshold is the inverse
threshold multiplied by `T = product(selected complex-tail n_a)`; the real axis,
identity axes, and extra dimensions are excluded. `T` and the resulting
finite-positive raw threshold are collectively validated during plan
construction before native planning.
Normalized inverse threshold arithmetic is unchanged.

#### Milestone 8 current capability: distributed real in-place (2026-09-20)

The distributed R2C/C2R plan now also exposes
`allocate_in_place`, `allocate_in_place_workspace`, and the typed
`forward_in_place`, `inverse_in_place`, and `backward_in_place` operations.
`R2cInPlaceArray` owns one allocation whose bytes are safely recast between a
real-prefix `ManyPencilArray<R>` registry and a reduced-complex-suffix
`ManyPencilArray<Complex<R>>` registry. It supports selected and non-last real
boundaries, both checked transports, extra batches, empty local partitions,
and length-one/odd/even reduced axes. The public views are state checked and
return only the matching `PencilArrayView` type.

The data allocation is sized once for the checked maximum byte requirement of
both registries; representation handoff uses bytemuck's fallible ownership
casts, never unsafe/raw parts or a full-array temporary. Real boundary rows are
packed and processed back-to-front for forward and front-to-back for C2R.
Complex suffix transitions and native stages run in the same allocation. The
outer state changes `RealInput -> Poisoned -> ComplexOutput` for forward and
`ComplexOutput -> Poisoned -> RealInput` for reverse operations. Initial
collective preflight errors preserve state/data/workspace; after start, errors,
invalid spectra, and panics leave the array poisoned. The normalized/raw
boundary policy is the existing out-of-place policy; invalid spectra are
reported after complex-tail work without a rollback promise.

R2C in-place collective operation words are 25, 26, and 27 for forward,
normalized inverse, and raw backward. Words 18 through 24 remain reserved for
the parallel distributed R2R API.

Collective operation words 12, 13, 14, and 17 are R2C construction, forward,
inverse, and raw backward. The five-word header and operation words remain fixed. The exact schema-2
R2C descriptor records the original real shape, process grid, extra rank and
extents, canonical axis mask, value kind, scalar width, and transport method,
distinguishing selection and even/odd shapes that share `m`. No public
tolerance knobs, offender-ID reductions, or full-spectrum gather are part of
this API.

### Milestone 9: 交差検証と性能評価

Julia/FFTW cross-validation tooling is implemented as an opt-in local check;
its canonical command, locked Julia environment, exactly 68 temporary
format-6 fixtures and 110 case/layout combinations per transpose method and
memory-layout policy (220 with both policies) are documented in
[`tools/fftw-reference/README.md`](../../../tools/fftw-reference/README.md).
It validates distributed C2C, R2C/C2R, R2R, and DHT forward/inverse/raw
backward through both out-of-place and single-allocation real in-place paths,
with both layout policies, without changing production tolerances, CI, or
checked-in numeric data.

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

- 全spatial軸または選択軸集合のC2C out-of-place
- 全spatial軸または選択軸集合のC2C in-place
- 全spatial軸または選択軸集合のR2C/C2R out-of-place
- 全spatial軸のDCT/DST-I-IV R2R out-of-place/in-place
- 分散R2C/C2Rのsingle-allocation in-place
- `f32`,`f64`および対応する複素R2Rスカラー
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
- R2C in-placeは共通`TransformStage`を維持し、異種型共有storageだけを専用化する
- DCT/DST追加時も局所transform variantとして追加し、配置列を再利用する
- 部分FFTは公開`AxisSelection<N>`を入力にし、既存の全軸選択を既定値とする。実行routeは常に軸`N-1`から`0`までの`N`段・`N-1`遷移を保ち、未選択stageはnative FFTとscaleを持たないidentityとする。したがって空のC2Cは値を変えずにcanonical full routeの転置だけを行う。R2Cは選択集合の最大Rust軸をreal boundaryにし、その軸だけを`n/2+1`へreductionする。
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
