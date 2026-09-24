# Fixed-commit audit remediation

User requested fixes after the non-passing audit of `2df5f2dd33364d8aa2acca003f370e7c9fc2714b`. The historical audit and its frozen criteria/evidence remain unchanged. This is a new source revision, not a retroactive pass of that snapshot.

## Scope

- [x] D1: describe `ldd`/runtime versions as native linkage provenance, not general ABI proof; retain the compatible-native-interface requirement and require real parallel I/O checks in the recorded environment.
- [x] D2: correct C2C route documentation to N stages and N-1 transitions.
- [x] D3: document that `IoError::Native` also covers resource/transfer operations and may be returned after mutation. No enum or error-mapping changes.
- [x] U1: state Mixed R2C's exact suffix-depth, embedding allowance, logical normalization, per-plane/per-batch finite/absolute/relative policy and failure boundaries. Tests cover real-prefix/identity exclusion, f32/f64, odd/even/unit boundaries, independent planes/batches, OOP preservation and in-place poison/preflight behavior. No numerical-policy or tolerance changes.
- [x] U3: trace completed real FFTW time-limit API calls immediately after success, actual wisdom-only failure and injected unwind, before any subsequent constructor can overwrite the observation. Both precisions must yield the requested budget then -1. This is actual-call/argument evidence, not an invented FFTW internal-state getter.
- [x] U5: explicitly execute both ordinary FFTW descriptor/op-word tests without `--ignored`, add the existing native/local wisdom-only and MPI callback/reverse-overlap cases, and add the new native time-limit regression. Keep billing, permissions, concurrency and protection unchanged.
- [x] O1/type coverage: describe intermediate checked-integer overflow and pin topology `!Send`/`!Sync` through compile-fail documentation.
- [x] U2 tooling: verify all nine manifest-pinned Julia package trees before instantiation can execute artifact-selector hooks, and again before any FFTW import/generation. Distinguish absent roots from malformed existing roots, reject manifest overrides and symlinks, and disable reused compiled modules. Guard regressions include a malicious artifact selector with a positive unguarded control. This checks pinned source integrity, not hostile-artifact authentication or concurrent-depot-writer safety.
- [x] Parent final combined-tree checks and independent integration review.

## Trusted Julia input recovery

The two differing user cache trees were NOT modified. Their mismatch cause was not diagnosed and is not described as compromise. The private remediation depot was populated with the seven already hash-matching trees plus fresh public upstream source archives:

- `https://codeload.github.com/JuliaMath/AbstractFFTs.jl/tar.gz/refs/tags/v1.5.0`
  - archive SHA256 `e134135a6ad08b79416a1481231d62c9c60bd236b49c7c67cc7a1f7779639a4e`
  - verified Git-tree SHA1 `d92ad398961a3ed262d8bf04a1a2b8340f915fef`
- `https://codeload.github.com/JuliaMath/FFTW.jl/tar.gz/refs/tags/v1.10.0`
  - archive SHA256 `4f1440b62e5ae240b116801ba1c792333ab32e09f96a0de47747c866073fdb9f`
  - verified Git-tree SHA1 `97f08406df914023af55ade2f843c39e99c5d969`

All nine source trees matched the unchanged checked-in Manifest. The active FFTW artifact `fddee2a92d37e18a0b5265dce2aab7af84fd9242` and installed IntelOpenMP artifact `0acf350efdf0dfc8dc7ab9d79ed22e90e0c2e807` were independently verified in the private depot; unused lazy MKL inputs were not required. Oracle metadata reports Julia1.12.6 / FFTW.jl1.10.0 / native FFTW3.3.11. Rust's actual native consumer remains separately reported.

Initial clean-environment package-server DNS attempts and a package-endpoint HTTP403 failed. The public upstream archive fallback succeeded; hashes were checked before use. No credential/proxy inspection, user cache repair, pin updates, system installation or global configuration changes were performed. Build/test commands use an explicit allowlisted environment, isolated HOME/CARGO_HOME and a separate target per source worktree/compiler. Never dump environment variables.

## Evidence and limits

Astra implemented the scoped changes; Sol independently reviewed their correctness and guard ordering. Mixed endpoint allowance removal and omitted time-limit reset mutants failed; only the permanent Rust regression tests are committed, not the temporary source-mutating mutation driver. Worker stable/exact1.85 MPI1/4/6 and six complete guarded reference runs passed after source recovery.

Parent independently verified the combined source using fresh worktree-private targets and the explicit environment wrapper:
- Stable Rust1.98.1: 23 static/build/native jobs; exact Rust1.85.0: 19 jobs. Default/all-feature/all-target checks, ordinary/library/doctests, old exhaustive error callers, native adapter (10 ignored tests) and local native suites passed. Stable fmt, strict Clippy and warning-denying rustdoc passed.
- Main MPI matrix: 119 successful positive launches per compiler (238 total), including the newly selected wisdom-only, callback and mixed-reverse cases, plus the existing expected asymmetric abort86 children.
- Independent oracle: six complete guarded runs (RustFFT threads1 and FFTW threads2 in both builder orders, on both compilers). Each required 87 fixtures / 312 configurations at ranks1/4/6 and all30 intended corruption rejections. Combined with the main matrices: 274 positive MPI launches and180 reference corruption rejections, excluding expected abort children.
- Integrity guard: 38 assertions, including a hash-only depot fixture with a malicious artifact selector that the precheck blocks; its unguarded control executes the selector.
- Separate direct compile probes confirm the new topology Send/Sync examples fail specifically with E0277 on both compilers, not missing-symbol errors.

Evidence is under `/tmp/pencil-rs-audit-remediation.9gn1aB/logs/pencil-rs-audit-fix-integration.aOASGw/{stable,msrv}/parent-*.log`, with exact commands in `commands.jsonl`; parent summaries are in that remediation directory. Original audit artifacts remain unchanged.

One integration-review claim was retracted after checking the actual accumulation: both R2C variants accumulate real squares and imaginary squares separately. The common maximum-component scale does not make the denominator a complex norm. The accepted relative comparison remains imaginary L2 versus real-part L2. An initial new Mixed rustdoc typo was corrected to `real_plane`; the original README/R2C/spec policy and production arithmetic were not changed. Sol and an independent Astra check confirmed this.

U4 is a verification limit, not a concrete implementation defect: this work does not prove all API semantics, all execution paths, native internals, every resource failure, MPICH or other architectures. The independent audit rejected the proposed universal `WriteIncomplete` requirement and the claim that every legacy HDF native-property allocation must precede file creation. Those implementations remain unchanged. The hypothetical MPI_Comm_dup error/non-null output issue is not represented as an observed conforming-MPI bug and is not altered without a stronger contract/reproducer.

GPU draft and Julia PencilIO format compatibility remain excluded. Local evidence is not actual GitHub Actions success; the existing September CI-success waiver remains, with no account or protection changes.
