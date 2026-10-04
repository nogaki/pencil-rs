//! Independent canonical-input complex-to-real transforms.

use std::sync::Arc;

use mpi::datatype::Equivalence;
use pencil_array::{ExtraShape, MpiTopology, Pencil, PencilArray};

use super::{
    AxisSelection, BackendChoice, DistributedLayout, FftError, R2cPlan, R2cWorkspace, StageGeometry,
};
#[cfg(feature = "fftw")]
use crate::PlanOptions;
use crate::{BackendInitError, Complex, FftReal};

// Separate family and operation words, including backward versus inverse.
pub(super) const VALUE_KIND_C2R: u64 = 5;
pub(super) const OPERATION_C2R_PLAN: u64 = 133;
#[cfg(feature = "fftw")]
pub(super) const OPERATION_NATIVE_C2R_PLAN: u64 = 134;
const OPERATION_C2R_FORWARD: u64 = 135;
const OPERATION_C2R_INVERSE: u64 = 136;
const OPERATION_C2R_BACKWARD: u64 = 137;

/// C2R shares the checked real-transform errors, including `InvalidSpectrum`.
pub type C2rError = super::R2cError;

/// An independent distributed half-complex-to-real plan (Julia BRFFT convention).
///
/// See the [crate-level compatibility and failure contract](crate#distributed-compatibility-and-failure-contract)
/// for endpoint identity, workspace ownership, and error boundaries.
///
/// `forward` is raw positive-sign C2R; `backward` is raw negative-sign R2C;
/// `inverse` is negative-sign R2C divided by the product of selected real
/// extents. Thus `forward` followed by `inverse` recovers the complex input.
/// Only selected spatial axes contribute to normalization, not extra batches.
///
/// The input is a canonical **complex** pencil: identity memory permutation,
/// decomposition `[0..M)`. It is not an existing R2C plan's transposed output.
/// A nonempty selection reduces its lowest Rust axis, visited last among the
/// selected axes of the descending canonical route. The caller must supply
/// that axis's original real length `n`: the input extent must be `n / 2 + 1`,
/// which cannot distinguish odd from even `n`. Output uses the original real
/// extent and decomposition `[1..=M]`, with reversed spatial memory order by
/// default. `permute_dims = false` keeps identity memory order at every stage.
/// Unselected axes still participate in the full route as identity stages.
/// Requires `N >= 2` and `1 <= M < N`.
///
/// Constructors, backend rebuilds, and execution are collective on the same
/// Cartesian communicator context. All ranks must agree on API/call order,
/// precision, complex shape, real length, selection, batches, layout, and
/// backend. Defaults remain RustFFT and Alltoallv even with `fftw` enabled.
/// Allocation is noncollective: coordinate local failures before execution.
/// Workspaces belong to the exact plan which allocated them.
///
/// Every operation preserves its source. Descriptor/preflight rejection also
/// preserves the destination and workspace. Forward validates DC and, for
/// even `n`, Nyquist after the transverse transforms, using the same finite,
/// normwise-relative/componentwise-absolute policy as [`R2cPlan::backward`].
/// Its depth and raw absolute scaling include all selected non-reduction
/// axes. Accepted imaginary endpoint noise is projected only in private
/// workspace. `InvalidSpectrum` preserves source and destination, but the
/// workspace may already have changed. Interior bins (including the final
/// odd bin when `n > 1`) have no blanket finite-or-real requirement.
/// Other execution failures after preflight are not transactional. Checked
/// transports inherit the existing MPI failure contract; unfinished P2P
/// transposes must not overlap on the topology's fixed context/tag.
///
/// This API is out-of-place only: no in-place, timing, collections, or overlap
/// variants are provided.
#[derive(Debug)]
pub struct C2rPlan<R: FftReal, const N: usize, const M: usize> {
    inner: R2cPlan<R, N, M>,
}

/// Reusable noncollective storage bound to one exact [`C2rPlan`].
#[derive(Debug)]
pub struct C2rWorkspace<R: FftReal, const N: usize, const M: usize> {
    inner: R2cWorkspace<R, N, M>,
}

impl<R: FftReal, const N: usize, const M: usize> C2rPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    /// Collectively builds a plan from a canonical complex pencil and real length.
    pub fn from_pencil(
        input: Arc<Pencil<N, M>>,
        real_len: usize,
        extra_shape: ExtraShape,
    ) -> Result<Self, C2rError> {
        Self::from_pencil_with_selection_and_layout(
            input,
            real_len,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan with a selection, transport, and memory layout.
    pub fn from_pencil_with_selection_and_layout(
        input: Arc<Pencil<N, M>>,
        real_len: usize,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, C2rError> {
        Self::construct(
            Arc::clone(input.topology()),
            *input.global_shape(),
            real_len,
            extra_shape,
            Ok(input),
            selection,
            layout,
        )
    }

    /// Collectively builds a plan from a canonical complex array and real length.
    pub fn from_array(
        input: &PencilArray<Complex<R>, N, M>,
        real_len: usize,
    ) -> Result<Self, C2rError> {
        Self::from_array_with_selection_and_layout(
            input,
            real_len,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from a complex array with selection and layout.
    pub fn from_array_with_selection_and_layout(
        input: &PencilArray<Complex<R>, N, M>,
        real_len: usize,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, C2rError> {
        Self::from_pencil_with_selection_and_layout(
            Arc::clone(input.pencil()),
            real_len,
            input.extra_shape().clone(),
            selection,
            layout,
        )
    }

    /// Collectively builds a plan from the reduced complex shape and real length.
    pub fn from_shape(
        topology: Arc<MpiTopology<M>>,
        complex_shape: [usize; N],
        real_len: usize,
        extra_shape: ExtraShape,
    ) -> Result<Self, C2rError> {
        Self::from_shape_with_selection_and_layout(
            topology,
            complex_shape,
            real_len,
            extra_shape,
            AxisSelection::all(),
            DistributedLayout::default(),
        )
    }

    /// Collectively builds a plan from a reduced complex shape, selection, and layout.
    pub fn from_shape_with_selection_and_layout(
        topology: Arc<MpiTopology<M>>,
        complex_shape: [usize; N],
        real_len: usize,
        extra_shape: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, C2rError> {
        let input = Pencil::new(
            Arc::clone(&topology),
            complex_shape,
            std::array::from_fn(|a| a),
        )
        .map_err(FftError::Pencil);
        Self::construct(
            topology,
            complex_shape,
            real_len,
            extra_shape,
            input,
            selection,
            layout,
        )
    }

    fn construct(
        topology: Arc<MpiTopology<M>>,
        complex_shape: [usize; N],
        real_len: usize,
        extra_shape: ExtraShape,
        input: Result<Arc<Pencil<N, M>>, FftError>,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
    ) -> Result<Self, C2rError> {
        R2cPlan::construct_oriented(
            topology,
            complex_shape,
            extra_shape,
            input,
            selection,
            layout,
            BackendChoice::RustFft,
            Some(real_len),
        )
        .map(|inner| Self { inner })
        .map_err(|error| match error {
            BackendInitError::Local(error) => error,
            #[cfg(feature = "fftw")]
            BackendInitError::Native(_) => C2rError::Fft(FftError::PreparationFailed),
            BackendInitError::PeerPreflight => C2rError::Fft(FftError::PreparationFailed),
        })
    }

    /// Returns the canonical reduced-complex input pencil.
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        self.inner.output_pencil()
    }

    /// Returns the real output pencil in the chosen spatial memory order.
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        self.inner.input_pencil()
    }

    /// Returns the selected spatial axes.
    pub fn selection(&self) -> AxisSelection<N> {
        self.inner.selection()
    }

    /// Returns the reduction axis (the lowest selected Rust axis).
    pub fn reduction_axis(&self) -> usize {
        (0..N)
            .find(|&axis| self.selection().contains(axis))
            .expect("nonempty selection")
    }

    /// Returns the explicit original real length of the reduction axis.
    pub fn real_len(&self) -> usize {
        self.output_pencil().global_shape()[self.reduction_axis()]
    }

    /// Returns the exact extra batch shape required by the plan.
    pub fn extra_shape(&self) -> &ExtraShape {
        self.inner.extra_shape()
    }

    /// Returns the selected transport and memory-layout policy.
    pub fn layout(&self) -> DistributedLayout {
        self.inner.layout()
    }

    /// Returns checked geometry in C2R forward (descending axis) order.
    pub fn stage_geometry(&self) -> Box<[StageGeometry<N, M>]> {
        let mut stages = self.inner.stage_geometry();
        stages.reverse();
        for stage in &mut stages {
            std::mem::swap(&mut stage.source, &mut stage.output);
        }
        stages
    }

    /// Returns the local FFT backend in use.
    pub fn backend_kind(&self) -> crate::BackendKind {
        self.inner.backend_kind()
    }

    /// Returns native planning options, or `None` for RustFFT.
    #[cfg(feature = "fftw")]
    pub fn options(&self) -> Option<PlanOptions> {
        self.inner.options()
    }

    /// Collectively rebuilds with FFTW, preserving the C2R direction and geometry.
    ///
    /// Use the returned plan's endpoint-pencil `Arc`s and allocate new workspaces.
    /// See the [crate-level contract](crate#distributed-compatibility-and-failure-contract).
    #[cfg(feature = "fftw")]
    pub fn with_fftw(&self, options: PlanOptions) -> Result<Self, BackendInitError<C2rError>> {
        self.inner.with_fftw(options).map(|inner| Self { inner })
    }

    /// Allocates a zero-initialized canonical complex input array (noncollective).
    pub fn allocate_input(&self) -> Result<PencilArray<Complex<R>, N, M>, C2rError> {
        self.inner.allocate_output()
    }

    /// Allocates a zero-initialized real output array (noncollective).
    pub fn allocate_output(&self) -> Result<PencilArray<R, N, M>, C2rError> {
        self.inner.allocate_input()
    }

    /// Allocates noncollective reusable storage bound to this exact plan.
    pub fn allocate_workspace(&self) -> Result<C2rWorkspace<R, N, M>, C2rError> {
        self.inner
            .allocate_workspace()
            .map(|inner| C2rWorkspace { inner })
    }

    /// Collectively computes raw positive-sign C2R, validating endpoint planes.
    /// See [`Self`] for preservation and `InvalidSpectrum` guarantees.
    pub fn forward(
        &self,
        source: &PencilArray<Complex<R>, N, M>,
        destination: &mut PencilArray<R, N, M>,
        workspace: &mut C2rWorkspace<R, N, M>,
    ) -> Result<(), C2rError> {
        self.inner.execute_reverse(
            source,
            destination,
            &mut workspace.inner,
            false,
            OPERATION_C2R_FORWARD,
            None,
        )
    }

    /// Collectively computes raw negative-sign R2C from the real output layout.
    pub fn backward(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2rWorkspace<R, N, M>,
    ) -> Result<(), C2rError> {
        self.inner.execute_forward(
            source,
            destination,
            &mut workspace.inner,
            OPERATION_C2R_BACKWARD,
            None,
        )
    }

    /// Collectively computes negative-sign R2C normalized by selected real extents.
    /// The collective call is distinct from [`Self::backward`] before any write.
    pub fn inverse(
        &self,
        source: &PencilArray<R, N, M>,
        destination: &mut PencilArray<Complex<R>, N, M>,
        workspace: &mut C2rWorkspace<R, N, M>,
    ) -> Result<(), C2rError> {
        self.inner.execute_forward(
            source,
            destination,
            &mut workspace.inner,
            OPERATION_C2R_INVERSE,
            None,
        )?;
        // Scale per extent instead of multiplying dimensions in usize.
        for (axis, &length) in self.output_pencil().global_shape().iter().enumerate() {
            if self.selection().contains(axis) {
                R::normalize_inverse(destination.as_mut_slice(), length);
            }
        }
        Ok(())
    }
}
