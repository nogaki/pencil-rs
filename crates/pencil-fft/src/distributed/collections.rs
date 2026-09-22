//! Sequential array collections with one reusable workspace.
//!
//! All ranks first agree two five-word MIN/MAX reductions, then the complete
//! existing plan descriptor (including backend and Fourier directions). The
//! descriptor plus member count specifies the expected metadata of every member;
//! each member is checked against that plan on every rank before any execution.
//! Rank-local extents are validated locally, not compared across uneven ranks.
//! Empty collections are collectively rejected. Preflight errors change neither
//! arrays nor workspace. Execution errors carry a member index: earlier outputs
//! may already be changed and there is no collection-wide rollback. Out-of-place
//! sources remain unchanged. Existing ownership and poison contracts apply:
//! poisoned opaque arrays/workspaces must be reallocated, not retried.
//! An arbitrary member-execution panic is fail-stop: after Rust unwinds owned
//! guards, the collection aborts its MPI communicator. No recovery collective is
//! safe at an unknown single-member MPI phase; global panic recovery is not promised.

use super::mixed::{
    MixedC2cInPlaceArray, MixedC2cInPlaceWorkspace, MixedC2cPlan, MixedC2cWorkspace, MixedError,
    MixedR2cInPlaceArray, MixedR2cInPlaceWorkspace, MixedR2cPlan, MixedR2cWorkspace,
};
use super::*;
use crate::R2rScalar;
use mpi::topology::Communicator;

/// Collection agreement, validation, or indexed execution failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CollectionError<E> {
    /// Ranks disagree on operation, family, counts, or complete plan descriptor.
    #[error("collection headers or plan descriptors differ across ranks")]
    HeaderMismatch,
    /// The shared plan descriptor could not be prepared or agreed.
    #[error("collection descriptor agreement failed: {error}")]
    Descriptor {
        /// The existing descriptor agreement error.
        #[source]
        error: FftError,
    },
    /// Empty collections are not executable.
    #[error("empty FFT collection")]
    Empty,
    /// Source and destination counts differ.
    #[error("collection counts differ: {sources} sources, {destinations} destinations")]
    CountMismatch {
        /// Number of sources.
        sources: usize,
        /// Number of destinations.
        destinations: usize,
    },
    /// This rank rejected or failed the indexed member.
    #[error("collection member {index}: {error}")]
    Member {
        /// Zero-based member index.
        index: usize,
        /// Existing single-array error, unchanged.
        #[source]
        error: E,
    },
    /// Another rank rejected the indexed member before execution began.
    #[error("collection member {index} failed preflight on a peer")]
    PeerPreflight {
        /// Zero-based member index.
        index: usize,
    },
    /// Another rank failed while executing this member.
    #[error("collection member {index} failed execution on a peer")]
    PeerExecution {
        /// Zero-based member index.
        index: usize,
    },
}

// Reserved collection operation words: 121..=126 (forward/inverse/backward,
// then the corresponding in-place operations). Family words: 1..=6 in the
// order C2C, R2C, R2R, DHT, mixed C2C, mixed R2C.
fn begin<E, const N: usize, const M: usize>(
    comm: &mpi::topology::CartesianCommunicator,
    operation: u64,
    family: u64,
    sources: usize,
    destinations: usize,
    descriptor: &[u64],
) -> Result<(), CollectionError<E>> {
    if !agree_header(
        comm,
        [
            DESCRIPTOR_SCHEMA,
            operation,
            u64::try_from(sources).unwrap_or(INVALID_WORD),
            u64::try_from(destinations).unwrap_or(INVALID_WORD),
            family,
        ],
    ) {
        return Err(CollectionError::HeaderMismatch);
    }
    if sources != destinations {
        return Err(CollectionError::CountMismatch {
            sources,
            destinations,
        });
    }
    if sources == 0 {
        return Err(CollectionError::Empty);
    }
    agree_execution_descriptor_ref::<N, M>(comm, operation, descriptor).map_err(|error| {
        if matches!(error, FftError::CollectiveDescriptorMismatch) {
            CollectionError::HeaderMismatch
        } else {
            CollectionError::Descriptor { error }
        }
    })
}

fn member_result<E>(
    comm: &mpi::topology::CartesianCommunicator,
    index: usize,
    result: Result<(), E>,
    preflight: bool,
) -> Result<(), CollectionError<E>> {
    if collective_valid(comm, result.is_ok()) {
        return Ok(());
    }
    Err(match result {
        Err(error) => CollectionError::Member { index, error },
        Ok(()) if preflight => CollectionError::PeerPreflight { index },
        Ok(()) => CollectionError::PeerExecution { index },
    })
}

// Single execution can return local errors (including transaction failures), so
// retain result agreement. A panic cannot safely join that agreement: peers may
// still be inside the single operation. Unwind its guards, then use MPI fail-stop.
fn execute_member<E>(
    comm: &mpi::topology::CartesianCommunicator,
    index: usize,
    execute: impl FnOnce() -> Result<(), E>,
) -> Result<(), CollectionError<E>> {
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(execute)) {
        Ok(result) => result,
        Err(_) => {
            use std::io::Write;
            // A closed stderr must not prevent communicator termination.
            let _ = writeln!(
                std::io::stderr(),
                "COLLECTION_MEMBER_PANIC_ABORT index={index}"
            );
            comm.abort(86)
        }
    };
    member_result(comm, index, result, false)
}

macro_rules! oop {
    ($name:ident, $single:ident, $src:ty, $dst:ty, $ws:ty, $err:ty, $family:expr, $op:expr, $validate:expr, $descriptor:expr) => {
        /// Transform all members sequentially with one workspace.
        ///
        /// Collective: counts, operation and complete plan must agree. All members
        /// and workspace are validated before writes. Empty collections are rejected.
        /// A post-start failure may leave earlier outputs modified; its index is
        /// reported without rollback. Sources are preserved. Arbitrary member
        /// panics unwind owned guards then abort the MPI communicator, not recover.
        pub fn $name(
            &self,
            sources: &[PencilArray<$src, N, M>],
            destinations: &mut [PencilArray<$dst, N, M>],
            workspace: &mut $ws,
        ) -> Result<(), CollectionError<$err>> {
            let comm = self.input_pencil().topology().communicator();
            begin::<$err, N, M>(
                comm,
                $op,
                $family,
                sources.len(),
                destinations.len(),
                ($descriptor)(self),
            )?;
            for (index, (source, destination)) in
                sources.iter().zip(destinations.iter()).enumerate()
            {
                member_result(
                    comm,
                    index,
                    ($validate)(self, source, destination, &*workspace),
                    true,
                )?;
            }
            for (index, (source, destination)) in
                sources.iter().zip(destinations.iter_mut()).enumerate()
            {
                execute_member(comm, index, || self.$single(source, destination, workspace))?;
            }
            Ok(())
        }
    };
}
macro_rules! ip {
    ($name:ident, $single:ident, $array:ty, $ws:ty, $err:ty, $family:expr, $op:expr, $direction:expr, $validate:ident, $descriptor:expr) => {
        /// Transform all opaque arrays sequentially using one workspace.
        ///
        /// All states and storage are checked before execution. Empty collections
        /// are rejected. There is no rollback after execution starts; poisoned
        /// arrays or workspaces require reallocation, not a retry. Arbitrary member
        /// panics unwind owned guards then abort the MPI communicator, not recover.
        pub fn $name(
            &self,
            arrays: &mut [$array],
            workspace: &mut $ws,
        ) -> Result<(), CollectionError<$err>> {
            let comm = self.input_pencil().topology().communicator();
            begin::<$err, N, M>(
                comm,
                $op,
                $family,
                arrays.len(),
                arrays.len(),
                ($descriptor)(self),
            )?;
            for (index, array) in arrays.iter().enumerate() {
                member_result(
                    comm,
                    index,
                    self.$validate($direction, array, workspace),
                    true,
                )?;
            }
            for (index, array) in arrays.iter_mut().enumerate() {
                execute_member(comm, index, || self.$single(array, workspace))?;
            }
            Ok(())
        }
    };
}
macro_rules! methods {
    ($src:ty, $dst:ty, $ws:ty, $array:ty, $ipws:ty, $err:ty, $family:expr, $forward:expr, $reverse:expr, $ipv:ident, $descriptor:expr) => {
        oop!(
            forward_many,
            forward,
            $src,
            $dst,
            $ws,
            $err,
            $family,
            121,
            $forward,
            $descriptor
        );
        oop!(
            inverse_many,
            inverse,
            $dst,
            $src,
            $ws,
            $err,
            $family,
            122,
            $reverse,
            $descriptor
        );
        oop!(
            backward_many,
            backward,
            $dst,
            $src,
            $ws,
            $err,
            $family,
            123,
            $reverse,
            $descriptor
        );
        ip!(
            forward_many_in_place,
            forward_in_place,
            $array,
            $ipws,
            $err,
            $family,
            124,
            Direction::Forward,
            $ipv,
            $descriptor
        );
        ip!(
            inverse_many_in_place,
            inverse_in_place,
            $array,
            $ipws,
            $err,
            $family,
            125,
            Direction::Inverse,
            $ipv,
            $descriptor
        );
        ip!(
            backward_many_in_place,
            backward_in_place,
            $array,
            $ipws,
            $err,
            $family,
            126,
            Direction::Backward,
            $ipv,
            $descriptor
        );
    };
}

impl<R: FftReal, const N: usize, const M: usize> C2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    fn collection_descriptor(&self) -> &[u64] {
        &self.core.descriptor
    }
    methods!(Complex<R>, Complex<R>, C2cOutOfPlaceWorkspace<R,N,M>, C2cInPlaceArray<R,N,M>, C2cInPlaceWorkspace<R,N,M>, FftError, 1,
        |p: &Self, s, d, w| p.preflight(Direction::Forward, s, d, w),
        |p: &Self, s, d, w| p.preflight(Direction::Inverse, s, d, w), preflight_in_place, Self::collection_descriptor);
}
impl<R: FftReal + Equivalence, const N: usize, const M: usize> R2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    methods!(R, Complex<R>, R2cWorkspace<R,N,M>, R2cInPlaceArray<R,N,M>, R2cInPlaceWorkspace<R,N,M>, R2cError, 2,
        Self::preflight_forward, Self::preflight_inverse, preflight_in_place, Self::collection_descriptor);
}
impl<T: R2rScalar + Equivalence, const N: usize, const M: usize> R2rPlan<T, N, M> {
    methods!(T, T, R2rWorkspace<T,N,M>, R2rInPlaceArray<T,N,M>, R2rInPlaceWorkspace<T,N,M>, R2rError, 3,
        |p: &Self, s, d, w| p.collection_preflight(Direction::Forward, s, d, w),
        |p: &Self, s, d, w| p.collection_preflight(Direction::Inverse, s, d, w), collection_preflight_in_place, Self::collection_descriptor);
}
impl<T: R2rScalar + Equivalence, const N: usize, const M: usize> DhtPlan<T, N, M> {
    methods!(T, T, R2rWorkspace<T,N,M>, R2rInPlaceArray<T,N,M>, R2rInPlaceWorkspace<T,N,M>, R2rError, 4,
        |p: &Self, s, d, w| p.collection_preflight(Direction::Forward, s, d, w),
        |p: &Self, s, d, w| p.collection_preflight(Direction::Inverse, s, d, w), collection_preflight_in_place, Self::collection_descriptor);
}
impl<R: FftReal, const N: usize, const M: usize> MixedC2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    methods!(Complex<R>, Complex<R>, MixedC2cWorkspace<R,N,M>, MixedC2cInPlaceArray<R,N,M>, MixedC2cInPlaceWorkspace<R,N,M>, MixedError, 5,
        Self::collection_preflight_forward, Self::collection_preflight_inverse, collection_preflight_in_place, Self::collection_descriptor);
}

#[cfg(test)]
std::thread_local! {
    static FAIL_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static PANIC_MEMBER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Focused hook inside the existing poison transaction, not a substitute kernel.
#[cfg(test)]
pub(super) fn inject_in_place_failure() -> Result<(), FftError> {
    PANIC_MEMBER.with(|slot| {
        assert!(!slot.replace(false), "COLLECTION_RANK_ASYMMETRIC_PANIC");
    });
    FAIL_AFTER.with(|slot| match slot.get() {
        None => Ok(()),
        Some(0) => {
            slot.set(None);
            Err(FftError::PreparationFailed)
        }
        Some(n) => {
            slot.set(Some(n - 1));
            Ok(())
        }
    })
}

/// Invoked only in a fresh fault subprocess, before the one-rank private suite.
#[cfg(test)]
pub(super) fn panic_regression(topology: &Arc<MpiTopology<1>>) {
    let comm = topology.communicator();
    let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [comm.size() as usize * 4, 3],
        ExtraShape::scalar(),
        TransposeMethod::PointToPoint,
    )
    .unwrap();
    let mut arrays = (0..2)
        .map(|_| plan.allocate_in_place().unwrap())
        .collect::<Vec<_>>();
    let mut workspace = plan.allocate_in_place_workspace().unwrap();
    // Rank zero alone panics inside Single's owned poison transaction, while
    // peers can still be executing the first member (including MPI traffic).
    PANIC_MEMBER.with(|slot| slot.set(comm.rank() == 0));
    let _ = plan.forward_many_in_place(&mut arrays, &mut workspace);
    panic!("collection unexpectedly survived member panic");
}

#[cfg(test)]
pub(super) fn regression_cases(topology: &Arc<MpiTopology<1>>) {
    let plan = C2cPlan::<f64, 2, 1>::from_shape_with_method(
        Arc::clone(topology),
        [4, 3],
        ExtraShape::scalar(),
        TransposeMethod::PointToPoint,
    )
    .unwrap();
    let mut arrays = (0..3)
        .map(|_| {
            let mut a = plan.allocate_in_place().unwrap();
            a.view_mut()
                .unwrap()
                .as_mut_slice()
                .fill(Complex::new(2.0, 1.0));
            a
        })
        .collect::<Vec<_>>();
    let mut workspace = plan.allocate_in_place_workspace().unwrap();
    let before = arrays[2].view().unwrap().as_slice().to_vec();
    FAIL_AFTER.with(|slot| slot.set(Some(1)));
    assert!(matches!(
        plan.forward_many_in_place(&mut arrays, &mut workspace),
        Err(CollectionError::Member {
            index: 1,
            error: FftError::PreparationFailed
        })
    ));
    assert_eq!(arrays[0].state(), C2cState::Output);
    assert_eq!(arrays[1].state(), C2cState::Poisoned);
    assert_eq!(arrays[2].state(), C2cState::Input);
    assert_eq!(arrays[2].view().unwrap().as_slice(), before);
    assert!(arrays[1].view().is_err());
    assert!(matches!(
        plan.forward_many_in_place(&mut arrays[1..], &mut workspace),
        Err(CollectionError::Member {
            index: 0,
            error: FftError::Array(ArrayError::Poisoned)
        })
    ));

    let sources = (0..3)
        .map(|_| plan.allocate_input().unwrap())
        .collect::<Vec<_>>();
    let mut destinations = (0..3)
        .map(|_| {
            let mut a = plan.allocate_output().unwrap();
            a.as_mut_slice().fill(Complex::new(77.0, -2.0));
            a
        })
        .collect::<Vec<_>>();
    let before = destinations
        .iter()
        .map(|a| a.as_slice().to_vec())
        .collect::<Vec<_>>();
    let mut ws = plan.allocate_out_of_place_workspace().unwrap();
    ws.fft_scratch.clear();
    if plan.core.fft_scratch_len > 0 {
        assert!(
            plan.forward_many(&sources, &mut destinations, &mut ws)
                .is_err()
        );
        assert_eq!(
            before,
            destinations
                .iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
    }
    let mut ws = plan.allocate_out_of_place_workspace().unwrap();
    C2C_CALLBACK_INJECTION.with(|slot| slot.set(Some(C2cCallbackInjection::Error)));
    assert!(
        plan.forward_with_overlap(&sources[0], &mut destinations[0], &mut ws)
            .is_err()
    );
    let before = destinations
        .iter()
        .map(|a| a.as_slice().to_vec())
        .collect::<Vec<_>>();
    for _ in 0..2 {
        assert!(
            plan.forward_many(&sources, &mut destinations, &mut ws)
                .is_err()
        );
        assert_eq!(
            before,
            destinations
                .iter()
                .map(|a| a.as_slice().to_vec())
                .collect::<Vec<_>>()
        );
    }
    println!("COLLECTION_PRIVATE_OK");
}

impl<R: FftReal + Equivalence, const N: usize, const M: usize> MixedR2cPlan<R, N, M>
where
    Complex<R>: Equivalence,
{
    methods!(R, Complex<R>, MixedR2cWorkspace<R,N,M>, MixedR2cInPlaceArray<R,N,M>, MixedR2cInPlaceWorkspace<R,N,M>, MixedError, 6,
        Self::collection_preflight_forward, Self::collection_preflight_inverse, collection_preflight_in_place, Self::collection_descriptor);
}
