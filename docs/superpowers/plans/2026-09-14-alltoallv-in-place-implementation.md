# Alltoallv 分散転置 in-place 実装計画

- 日付: 2026-09-14
- 基点: `f87bb23`（PR #3 の out-of-place 完成後）
- 範囲: `AllToAllvTransposePlan::execute_in_place` の実装、仕様追補、既存テスト更新。
  Rust実装と検証は完了。commit、pushは行わない。

## 境界とAPI

既存の `AllToAllvTransposePlan`、`AllToAllvTransposeWorkspace`、エラー型、
`MPI_Alltoallv`、および `ManyPencilArray` のcrate-privateな
`active_index`、`find_layout`、`begin_in_place_write`、`LayoutWriteGuard`を使う。
新しいworkspace、trait、通信方式、P2P、FFT、公開storage API、unsafeな不正状態
注入は追加しない。`many.rs`の既存の最大storage不変条件をそのまま利用する。

追加するAPIは次だけである。

```rust
pub fn execute_in_place<T>(
    &self,
    array: &mut ManyPencilArray<T, N, M>,
    workspace: &mut AllToAllvTransposeWorkspace<T>,
) -> Result<(), AllToAllvTransposeError>
where
    T: mpi::datatype::Equivalence + Copy;
```

active layoutはplanのsource、destinationは登録済みでなければならない。通常の
descriptorまたはpreflight errorではarrayのstateとstorageを変更しない。MPI障害、
任意panic、プロセス欠落後のglobal recoveryは保証しない。

## 共通化と固定検証順序

1. `alltoallv_transpose.rs`にin-place専用の
   `OPERATION_EXECUTE_IN_PLACE`（既存コードと異なる値）を追加する。in-placeも最初に
   source topology全体のCartesian communicatorで
   `[schema, operation, N, M, descriptor_len]`をmin/max比較する。`new`、
   `execute_views`、`execute_in_place`の混在は、このheader不一致として全rankで
   拒否し、どのrankも後続descriptorまたはaxis subcommunicatorへ進まない。

2. header一致後のin-place descriptorは、既存planのcanonical metadata（方向、変更軸、
   topology、global shape、decomposition、permutation）に、arrayのexactな
   `extra_shape`（rankと各extent）、`T`のtype name/size/alignment、
   `Equivalence::Out`のtype nameを連結する。既存のdescriptor長検査、fallibleなmin/max配列、nativeな
   word-by-word min/maxによるexact比較を再利用する。descriptor作成はactive viewを
   要求せず、`array.extra_shape()`とplan metadataから行うので、arrayがPoisonedでも
   header/descriptorのcollectiveを完了できる。

3. descriptor合意後、各rankでin-place用local preflightを作るが、ここでは失敗を直ちに
   returnしない。`active_index()`のstate確認（Poisonedを含む）、active source layout
   の一致、`find_layout()`によるdestination登録を検査する。sourceとdestinationの
   必要prefixはそれぞれ
   `checked_product([source.local_len(), batch])`と
   `checked_product([destination.local_len(), batch])`で計算し、同じlocal lengthとは
   仮定しない。登録layoutが最大storage以下であることは
   `ManyPencilArray` constructorsの既存不変条件に任せる。

   view依存のlayout検査だけを呼出し側に残し、workspace length、extra、region offset、
   spatial length、count、displacement、`Count`境界、count + displacement、総量を
   検査する既存 `prepare_execution` の共通部分を、pencilとstorage lengthを受け取る
   小さな共通preflightへ最小分割する。out-of-placeとin-placeは同じ
   `workspace_requirements`、`prepare_exchange_counts`、checked offset検査を使う。

4. 上のlocal結果をsource topology全体で既存のvalidity all-reduceにかける。一rankでも
   activeがPoisoned、source不一致、destination未登録、workspace不足、shape/count/offset
   不備なら、全rankがpack・payload・本体書込み前にErrを返す。成功rankもこの合意前に
   `?`で抜けない。Poisoned rankからもextra shapeとplan metadataだけは読み取る。

5. 全rank成功後、arrayのactive viewをsource prefixとして読み、既存pack順
   （peer rank、extra row-major、logical spatial row-major）でworkspace send prefixへ
   packする。pack helperはview専用にせず、`&Pencil`と`&[T]`を受ける形へ最小変更して
   out-of-place/in-placeで共有する。pack中もarrayのstate/storageは変更しない。

6. 既存の `Partition`/`PartitionMut` と `all_to_all_varcount_into` の一回の経路を
   helper化して共有する。count/displacementのchecked検査、有効prefix、zero payload、
   片方向emptyの既存fallbackを維持し、新しいdummy通信方式は作らない。通信完了までは
   arrayをValid(source)のままにする。

7. MPI完了後の本体書込み直前に既存 `begin_in_place_write()` でPoisonedにする。
   collective後のactive view/guard取得はpreflight済みの不変条件を使い、rank-localな
   `?` returnで他rankを分岐させない。guardのstorageから、checked済みのdestination必要prefixだけを取り出し、unpackを
   次の形に一般化して呼ぶ。

   ```text
   unpack_destination(peers, destination_pencil, destination_storage,
                      receive_buffer, extra_count)
   ```

   つまりdestination `Pencil`、`&mut [T]`、receive buffer、extra countだけを受け取り、
   view traitやcallbackを導入しない。out-of-placeはviewのpencil/sliceを渡し、in-placeは
   guard配下のdestination prefixを渡す。全destination要素を埋め終わってからだけ
   `guard.commit(destination_index)`する。最大storageの余剰tailはprefix外なので変更しない。
   guardの既存構造によりunpack中のpanic後もPoisonedを維持し、T: Copyのunpackは
   Clone/Drop callbackを呼ばない。

## テストと文書更新

`crates/pencil-array/tests/alltoallv_transpose.rs`の既存一つのintegration testとfixtureを
共用し、`mpi::initialize()`はprocessごとに一度だけ呼ぶ。1/4/6 rank（6 rankは非正方
`[2, 3]` grid）で次を検証する。

- 既存out-of-place結果との一致、2D/3Dの直接physical buffer期待値、forward/reverse
  round-trip、out-of-place source保持。
- source/destinationのlocal_len相違、source local emptyからdestination nonempty、
  その逆、scalar/zero-extra、片方向emptyとzero payload。
- 最大registered storageの余剰tailが不変で、destination必要prefixだけが更新されること。
- rank 0だけのsource不一致、destination未登録、Poisoned、workspace不足、extra shape
  違い。Poisonedは既存 `overwrite_with` の失敗経路で作り、全rankのErr、no-deadlock、
  失敗前後のarray state/data不変を確認する。
- `new`対`execute_in_place`、`execute_views`対`execute_in_place`の混在をheaderで拒否する。
- 通常のpreflight失敗後に同じplan/有効なarray/workspaceを再利用できることを確認する。
  panic保証は既存 `LayoutWriteGuard`/既存testsの契約を利用し、新しい公開panic hookや
  unsafeな状態注入は追加しない。

変更対象は `alltoallv_transpose.rs`、同integration test、仕様追補、README/lib.rsの
API説明と既存one-rank doctest、CIの既存1/4/6-rank Alltoallv suiteの説明だけとした。
CIのsuiteとsingle-initialize/timeout契約は維持し、P2P・FFT・一括監査修正は後続に残す。
