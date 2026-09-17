# Local R2C/C2R 実装計画

- 日付: 2026-09-17
- 基点: `f5f46582c5dd3df7dc7fa995601e7d55e27ad917`
- 範囲: `pencil-fft` の process-local な、out-of-place batched R2C/C2R のみ。
  分散FFT、Array統合、in-place real FFT、backend trait、共有workspace抽象は追加しない。
- 状態: local R2C/C2R実装済み。

## 方針

`realfft` 3.5.0 の `RealFftPlanner` から immutable な forward/inverse plan を一度ずつ
作り、`LocalR2cPlan<R>` が元の実数長 `n`、reduced complex 長 `n / 2 + 1`、両native
`get_scratch_len()` の最大値だけを保持する。`FftReal` は既存のsealed `f32`/`f64` 境界を
使い、公開APIにはRealFFTの型やerrorを露出しない。

```rust
pub struct LocalR2cPlan<R: FftReal> { /* private native plans and lengths */ }

impl<R: FftReal> LocalR2cPlan<R> {
    pub fn new(real_len: usize) -> Result<Self, LocalR2cError>;
    pub fn real_len(&self) -> usize;
    pub fn complex_len(&self) -> usize;
    pub fn scratch_len(&self) -> usize;
    pub fn forward(&self, src: &[R], dst: &mut [Complex<R>],
                   real_line: &mut [R], scratch: &mut [Complex<R>])
        -> Result<(), LocalR2cError>;
    pub fn inverse(&self, src: &[Complex<R>], dst: &mut [R],
                   complex_line: &mut [Complex<R>], scratch: &mut [Complex<R>])
        -> Result<(), LocalR2cError>;
}
```

各操作は方向に応じて、forwardではsourceをreal line、destinationをreduced complex line、
inverseではsourceをreduced complex line、destinationをreal lineとして各sliceを割って
batch数を検査する。乗算で総batch長を再計算しない。caller-owned の初期化済み
line bufferの必要prefixと共通scratchの必要prefixだけを使い、oversizedなtailには触れない。
forwardは各real lineをコピーしてからRealFFTを実行し、inverseは各complex lineをコピー
してから実行する。従って両方ともsourceを保持し、実行時にallocationやresizeを行わない。

inverseは全batchのDC、偶数長ならNyquistのimaginary componentを、line/scratch/outputの
最初の書込みより前に走査する。`+0.0`/`-0.0`だけをzeroとし、NaNは不正とする。odd長が
1より大きい場合のfinal binは検査しない。interior binはいずれの長さでも検査しない。n=1のfinal binはDCなので検査する。
RealFFTは検査後のbackend errorを`expect`で扱い、
通常のvalidation errorと入力endpoint制約だけを公開errorにする。成功したnative inverseの
出力を元の`n`で割る。

## 実装・検証

- `src/r2c.rs` を追加し、rootから `LocalR2cPlan`/`LocalR2cError` を明示的にre-exportする。
- `realfft = "~3.5.0"` を追加する。既存のRustFFT 6.4.1 lock entryは更新しない。
- zero/impossible length（real/complex slice address spaceとodd native staging）、非整数
  batch、batch数不一致、short line/scratchをbackend前に検査する。empty valid batchも
  line/scratch検査後のno-opとする。
- direct DFT、実数入力をimag=0としたC2C結果のhalf-spectrum prefix比較、Hermitian reconstructionを用い、
  f32/f64、n=1/2、odd/even prime/composite、複数batch、DC/Nyquist、sign/normalization、任意half
  spectrum、source保持、dirty scratch、oversized tail、later-batch endpoint error atomicityを小さい
  data-driven testsで確認する。
- README、root doctest、specのcurrent local API/statusだけを更新し、既存のC2C歴史記述とCI jobは変更しない。
- 検証: `cargo fmt --all -- --check`、`cargo test -p pencil-fft --locked`（20 unit tests、3 doctests）、
  `cargo test --workspace --lib --locked`（56 unit tests）、`cargo test --workspace --doc --locked -- --show-output`
  （17 doctests）、`cargo clippy --workspace --all-targets --locked -- -D warnings`、`cargo doc --workspace
  --no-deps --locked`、`git diff --check`を完了。
- 既存MPI suiteは1/4 rank、distributed transposeはさらに6 rankの計13実行を確認した。
  依存treeはFFT/Array分離を維持し、lockfileの追加はRealFFT 3.5.0だけである。
  MSRV 1.85は既存GitHub CI jobで確認する。
