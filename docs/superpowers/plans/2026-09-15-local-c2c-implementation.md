# Milestone 6 最初のPR: local C2C 実装

- 日付: 2026-09-15
- 基点: `28b1358`
- 状態: local C2C実装済み。workspace memberと`pencil-fft` manifestは反映済み。
  `pencil-array`のソースとmanifestは変更せず、R2C/C2Rと分散FFTは後続とする。
- 検証: local/workspace/MPIの検証は完了。MSRV 1.85は既存GitHub jobで確認する。
- 参照: [Rust移植設計仕様](../specs/2026-09-11-pencil-arrays-rust-port-design.md)

## 目的と境界

`crates/pencil-fft`を追加し、MPIや配列配置を知らない1Dのローカル複素FFTを
最初の動作単位として実装した。Milestone 6全体ではなく、その最初のPRの範囲である。

- `pencil-array`のmanifest、公開API、ソース、既存テストは変更しない。
- `pencil-fft`のlocal pathはflatなsliceだけを受け取る。`pencil-array`と`mpi`には
  依存せず、分散FFTのstage/transpose統合時に必要な依存だけを後続で追加する。
- RustFFT 6.4系（`rustfft = "~6.4.1"`、lockは6.4.1、MSRV 1.61）を使う。
  workspace MSRV 1.85に適合する。`RealFFT`、R2C/C2R、分散FFT、FFTW、GPU、公開
  `FftBackend` trait、stage framework、将来用のfeature flagはこの範囲に含めない。
- `pencil-array`へFFT依存を逆向きに入れない。workspace memberと新crateのmanifestは
  反映済みであり、`pencil-array`は引き続きFFT非依存とする。

## 実装ファイル

local C2C本体で追加した主要ファイルは次の通りである。

```text
Cargo.toml                         workspace memberへの1行の追加
crates/pencil-fft/Cargo.toml       crate metadata、rustfft、既存thiserror
crates/pencil-fft/src/lib.rs       公開API、実装、#[cfg(test)]の小さいunit tests
```

別のbackend/module、テストframework、MPI integration binaryは追加しない。`Cargo.lock`
はmanifest追加に伴うRustFFT 6.4.1とその推移依存だけを反映し、無関係なupgradeを
混ぜない。

現在のcrate manifestの依存境界は次の通りである。

```toml
[dependencies]
rustfft = "~6.4.1"
thiserror.workspace = true
```

`pencil-array`、`mpi`、`realfft`、`fftw`、GPU依存、追加のdev-dependency、crateの
`[features]` tableは置かない。複素型はRustFFTが再exportする実体の
`num_complex::Complex`（`pub use rustfft::num_complex::Complex`）を使い、RustFFTの
plan/direction型を公開型にしない。

## API

仕様20のsealed `FftReal`方針を、そのままlocal crateの公開境界にする。sealed module
の内部実装だけがRustFFTに必要な数値boundを持ち、実装は`f32`と`f64`だけに限定する。
公開signatureに`rustfft::FftNum`、`FftDirection`、`Fft`、`Arc<dyn Fft<_>>`を出さない。

```rust
pub trait FftReal: private::Sealed + Copy + Send + Sync + 'static {}

pub struct LocalC2cPlan<R> { /* private immutable plans and metadata */ }

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LocalC2cError {
    #[error("FFT line length must be non-zero")]
    InvalidLength,
    #[error("FFT length or derived length cannot be represented")]
    LengthOverflow,
    #[error("buffer length is not an integral number of FFT lines")]
    NonIntegralBatch,
    #[error("source and destination lengths differ")]
    BufferLengthMismatch,
    #[error("FFT scratch is too small")]
    ScratchTooSmall { required: usize, actual: usize },
}

impl<R: FftReal> LocalC2cPlan<R> {
    pub fn new(n: usize) -> Result<Self, LocalC2cError>;
    pub fn line_len(&self) -> usize;
    pub fn scratch_len(&self) -> usize;

    pub fn forward(
        &self,
        src: &[Complex<R>],
        dst: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError>;

    pub fn inverse(
        &self,
        src: &[Complex<R>],
        dst: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError>;

    pub fn forward_in_place(
        &self,
        data: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError>;

    pub fn inverse_in_place(
        &self,
        data: &mut [Complex<R>],
        scratch: &mut [Complex<R>],
    ) -> Result<(), LocalC2cError>;
}
```

`src`/`dst`または`data`は、line length `n`の連続row-major batchである。batch数を
別引数にはせず、buffer lengthから`len / n`で決め、RustFFTのbatch処理へ全sliceを一度
渡す。`LocalC2cPlan`は`n`、privateなforward/inverseのimmutable RustFFT plan、
`n`とは独立に計算したscratch requirementだけを持つ。scratchや入力・出力bufferは
所有しない。全公開操作は`&self`で呼べる。

## RustFFTとの対応とscratch

構築時に`FftPlanner::<R>::new()`から`plan_fft_forward(n)`と
`plan_fft_inverse(n)`を作る。`n`は0を検査してからnative APIへ渡す。

- `scratch_len()`は、forward/inverseそれぞれの
  `get_immutable_scratch_len()`と`get_inplace_scratch_len()`の最大値を返す。
  これを一つの最小公開requirementとし、OOPとin-place、正逆で同じcaller-owned
  scratchを再利用できるようにする。
- callerは初期化済みの`&mut [Complex<R>]`（通常はその長さ以上の`Vec`のslice）を
  渡す。検査対象は`Vec::capacity()`ではなくsliceの`len()`であり、planはresizeや
  reallocをしない。native queryが0、`n`未満、または`n`超になるいずれも許容する。
- OOPの`forward`/`inverse`は必ず
  `process_immutable_with_scratch(src, dst, scratch)`を使う。このnative APIは入力を
  保持するため、wrapperで全入力をcopyしない。
- `process_outofplace_with_scratch`は入力を`&mut`で受け取り変更し得るので、この
  input-preserving APIには使わない。
- in-placeの正逆は`process_with_scratch(data, scratch)`を使う。scratch内容は呼出し
  後にbackendが破壊してよいが、初期化済みsliceとして次の呼出しへそのまま再利用する。

## 数学・正規化契約

RustFFTの符号規約に合わせ、各lineについて

```text
forward: X_k = Σ_j x_j * exp(-2π i j k / n)   （無正規化）
inverse: x_j = (1/n) * Σ_k X_k * exp(+2π i j k / n)
```

とする。RustFFTのinverse実行後に、出力の全要素をそのlineの`n`だけで除算する。
複数batchの本数は分母に含めない。forwardは追加のscaleを行わず、inverseもbatch間の
全体copyや全batchをまとめた別の正規化は行わない。

## 検証・失敗保証

backend呼出しおよび`dst`/`data`の最初の更新より前に、次を同じ順序でchecked検査
する。

1. `new(0)`は`InvalidLength`で返し、RustFFT plannerを呼ばない。`new(1)`に空sliceを
   渡すzero batchとは別の状態である。
2. `usize`のnative lengthを狭い型へcastしない。必要な導出値・変換はcheckedに行い、
   表現できない場合は`LengthOverflow`を返す。
3. OOPは`src.len() == dst.len()`、かつその長さが`n`の整数倍であることを検査する。
   in-placeも`data.len() % n == 0`を検査する。n>0を先に保証するため除算は安全である。
4. `scratch.len() >= plan.scratch_len()`を検査する。

検査を通った後、src/dst（またはdata）の長さが0ならzero batchとして`Ok(())`を返す。
RustFFTへ空sliceを渡さない。src/dst長不一致、非整数batch長、scratch不足を含む通常の
validation `Err`では、source、destination、in-place data、scratchを変更しない。
成功したOOP操作はsourceを保持し、成功したin-place操作だけdataを更新する。

RustFFT 6.4.1のplannerや任意backendの資源枯渇・panicをcatchして`Result`へ変換する
ことは約束しない。実装に`unsafe`、`catch_unwind`、未初期化bufferは持ち込まない。

## 実装順序

1. workspace memberと`crates/pencil-fft/Cargo.toml`を追加し、RustFFT 6.4.1をlockした。
   `pencil-array`側は変更していない。
2. `lib.rs`にsealed `FftReal`、backend planを隠した`LocalC2cPlan`、error、公開docsを
   実装した。plannerは構築時だけmutable、公開planとscratchは分離している。
3. 共通のvalidation helperを先に実行し、OOPはimmutable native path、in-placeはnative
   in-place path、inverseだけline length scaleという二つの短い実行経路にした。
4. `#[cfg(test)]`内に独立DFT oracleと手書きの許容誤差比較を置いた。新しいテスト
   framework、MPI、`pencil-array` fixtureは作っていない。

## テスト計画

往復だけを正しさの根拠にせず、次の独立oracleを必須にする。forward oracleは上記の
負符号DFT、inverse oracleは正符号DFTを各lineの`1/n`で正規化したものとする。

- `f32`/`f64`、single/multiple batch、prime/composite、`n=1`を含む小さい長さを
  forwardのoracleと直接比較する。
- inverseも独立oracleと直接比較し、forward後のinverse round tripを別に確認する。
- 符号、DC、単一周波数のピーク、無正規化forward、lineごとのinverse正規化（複数batch
  を`n * batch_count`で割っていないこと）を確認する。
- OOPとin-placeの結果を比較し、同じ初期化済みscratchを複数回、forward/inverse間で
  再利用する。scratchは呼出し後にdirtyでもよいことを確認する。
- `new(0)`、非整数batch、src/dst長不一致、scratch不足を検査し、backend呼出し前の
  source/destination/data/scratch未変更を確認する。zero batchは長さ検査後のno-opで、
  空sliceをbackendへ渡さない。`new(1)`との区別もテストする。
- `n=1`、prime length、composite length、zero batchを含め、短い直接DFTで全binを
  比較する。RustFFTのscratch requirementが`n`と異なるケースはquery値で容量を
  作り、`n`を仮定した実装になっていないことを確認する。

## 検証コマンドと回帰確認

local crate単独のテストがMPIなしで通ることを確認した。

```bash
cargo test -p pencil-fft --locked
cargo tree -p pencil-fft --edges normal
cargo tree -p pencil-array --edges normal
```

treeでは`pencil-fft`のdirect/normal依存に`mpi`と`pencil-array`がなく、
`pencil-array`側に`rustfft`、`realfft`、`pencil-fft`が入っていないことを確認した。
`cargo test -p pencil-fft`を`mpiexec`で包む必要はない。

既存CI相当を回帰確認した。

```bash
cargo fmt --all -- --check
cargo test --workspace --lib --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo doc --workspace --no-deps --locked
cargo test --workspace --doc --locked -- --show-output
# Existing GitHub msrv job; no local toolchain install or rustup configuration change.
cargo +1.85.0 check --workspace --all-targets --locked
```

既存の`mpiexec` topology/pencil/array/many/local-transpose/
`alltoallv_transpose`（1/4/6 rank、timeout付き）もCIのコマンドのまま回帰確認した。
新crate追加によってMPI test binaryや既存のworkspace/docs/clippy/MSRV 1.85の結果を
変えないことを確認する。MSRV 1.85は既存GitHub jobで確認し、ローカルではrustupの
恒久設定変更やtoolchainのインストールを行わない。

## 採用事項（短縮報告）

- **API**: sealed `FftReal`（`f32`/`f64`のみ）と、`LocalC2cPlan::{new, line_len, scratch_len, forward, inverse, forward_in_place, inverse_in_place}`。public APIは`Complex<R>`とsliceだけで、RustFFT plan/direction/FftNumを出さない。
- **scratch/normalization/error**: caller-owned初期化済みsliceの`len`を、4 native scratch queryの最大値と比較してから実行する。OOPは`process_immutable_with_scratch`、in-placeは`process_with_scratch`、forward無正規化・inverseは各lineの`n`だけで除算。入力長・batch・scratchのvalidation `Err`は実行前かつ未変更、planner/backendの資源枯渇・任意panicは回収しない。
- **依存境界**: 今回の`pencil-fft`は`rustfft`と既存`thiserror`だけ。`pencil-array`/MPIはlocal slice段階に入れず、分散FFT統合時へ延期する。