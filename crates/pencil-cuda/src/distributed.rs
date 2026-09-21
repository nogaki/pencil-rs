//! Host-staged distributed cuFFT. All transforms are out of place.
//!
//! Plan construction, plan allocation methods and execution are collective on
//! the input pencil topology. Use the same operation/order on every rank.
//! Arrays and workspaces are resident/reusable; local packing and MPI use host
//! memory, not CUDA-aware MPI. No CPU FFT execution or root gather is used.
//!
//! ```no_run
//! use pencil_array::{MpiTopology, Pencil, ExtraShape};
//! use pencil_cuda::distributed::DistributedPlan;
//! use pencil_fft::{AxisSelection, DistributedLayout, FourierDirections};
//! use mpi::traits::*;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mpi = mpi::initialize().expect("MPI not initialized");
//! let world = mpi.world();
//! let topology = MpiTopology::<1>::new(&world, [world.size() as usize])?;
//! let input = Pencil::<2, 1>::new_default(topology, [5, 7])?;
//! let plan = DistributedPlan::<f64, 2, 1>::c2c(input, ExtraShape::scalar(),
//!     AxisSelection::all(), DistributedLayout::default(),
//!     FourierDirections::forward(), 0)?;
//! let mut spatial = plan.allocate_input()?; // zero initialized
//! let mut spectral = plan.allocate_output()?;
//! let mut workspace = plan.allocate_workspace()?;
//! plan.forward(&spatial, &mut spectral, &mut workspace)?;
//! plan.inverse(&spectral, &mut spatial, &mut workspace)?;
//! # Ok(()) }
//! ```
use crate::{C2CPlan, C2RPlan, CudaBuffer, CudaDevice, CudaError, DeviceScalar, R2CPlan};
use mpi::{
    collective::{CommunicatorCollectives, SystemOperation},
    datatype::Equivalence,
};
use num_complex::Complex;
use pencil_array::{
    AllToAllvTransposePlan, ExtraShape, LocalTransposePlan, Pencil, PencilArray,
    PointToPointTransposePlan, TransposeWorkspace,
};
use pencil_fft::{
    AxisSelection, DistributedLayout, FftReal, FourierDirection, FourierDirections, StageGeometry,
    TransposeMethod,
};
use std::{fmt::Debug, sync::Arc};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("distributed CUDA descriptor mismatch")]
    Descriptor,
    #[error("another rank failed distributed CUDA preparation")]
    Peer,
    #[error("{0}")]
    Invalid(String),
}
type Result<T> = std::result::Result<T, Error>;
fn invalid(e: impl std::fmt::Display) -> Error {
    Error::Invalid(e.to_string())
}
fn agree<T>(comm: &impl CommunicatorCollectives, value: Result<T>) -> Result<T> {
    let flag = u64::from(value.is_err());
    let mut any = 0;
    comm.all_reduce_into(&flag, &mut any, SystemOperation::max());
    if any != 0 {
        Err(value.err().unwrap_or(Error::Peer))
    } else {
        value
    }
}
// This exact first pair is shared by every public GPU collective, including
// construction and workspace allocation. No CUDA call precedes it.
fn header(comm: &impl CommunicatorCollectives, op: u64, descriptor: &[u64]) -> Result<()> {
    let h = [0x47505550454e434c, 1, op, descriptor.len() as u64, 0];
    let mut lo = [0; 5];
    let mut hi = [0; 5];
    comm.all_reduce_into(&h, &mut lo, SystemOperation::min());
    comm.all_reduce_into(&h, &mut hi, SystemOperation::max());
    if lo != hi {
        return Err(Error::Descriptor);
    }
    let buffers = (|| {
        Ok((
            zeros::<u64>(descriptor.len())?,
            zeros::<u64>(descriptor.len())?,
        ))
    })();
    let (mut lo, mut hi) = agree(comm, buffers)?;
    comm.all_reduce_into(descriptor, &mut lo[..], SystemOperation::min());
    comm.all_reduce_into(descriptor, &mut hi[..], SystemOperation::max());
    if lo != hi {
        Err(Error::Descriptor)
    } else {
        Ok(())
    }
}
fn zeros<T: Default + Clone>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(invalid)?;
    v.resize(n, T::default());
    Ok(v)
}

/// Scalar transfer operations. Implemented only for the four native CUDA types.
pub trait HostScalar: DeviceScalar + Default + Debug + Equivalence {
    fn upload(b: &mut CudaBuffer<Self>, v: &[Self]) -> std::result::Result<(), CudaError>;
    fn download(b: &CudaBuffer<Self>) -> std::result::Result<Vec<Self>, CudaError>;
}
macro_rules! host {($($t:ty),*)=>{$(impl HostScalar for $t {
    fn upload(b:&mut CudaBuffer<Self>,v:&[Self])->std::result::Result<(),CudaError>{b.upload(v)}
    fn download(b:&CudaBuffer<Self>)->std::result::Result<Vec<Self>,CudaError>{b.download()}
})*};}
host!(f32, f64, Complex<f32>, Complex<f64>);
/// Native precision adapter; no CPU transform is executed.
pub trait CudaReal: HostScalar + FftReal + Into<f64> {
    const EPS: f64;
    const MIN: f64;
    const TAG: u64;
    fn complex(
        d: &CudaDevice,
        n: usize,
        b: usize,
    ) -> std::result::Result<C2CPlan<Complex<Self>>, CudaError>
    where
        Complex<Self>: DeviceScalar;
    fn real(
        d: &CudaDevice,
        n: usize,
        b: usize,
    ) -> std::result::Result<R2CPlan<Self, Complex<Self>>, CudaError>
    where
        Complex<Self>: DeviceScalar;
    fn reverse(
        d: &CudaDevice,
        n: usize,
        b: usize,
    ) -> std::result::Result<C2RPlan<Complex<Self>, Self>, CudaError>
    where
        Complex<Self>: DeviceScalar;
}
macro_rules! real {
    ($t:ty,$tag:expr) => {
        impl CudaReal for $t {
            const EPS: f64 = <$t>::EPSILON as f64;
            const MIN: f64 = <$t>::from_bits(1) as f64;
            const TAG: u64 = $tag;
            fn complex(
                d: &CudaDevice,
                n: usize,
                b: usize,
            ) -> std::result::Result<C2CPlan<Complex<Self>>, CudaError> {
                C2CPlan::<Complex<$t>>::new(d, n, b)
            }
            fn real(
                d: &CudaDevice,
                n: usize,
                b: usize,
            ) -> std::result::Result<R2CPlan<Self, Complex<Self>>, CudaError> {
                R2CPlan::<$t, Complex<$t>>::new(d, n, b)
            }
            fn reverse(
                d: &CudaDevice,
                n: usize,
                b: usize,
            ) -> std::result::Result<C2RPlan<Complex<Self>, Self>, CudaError> {
                C2RPlan::<Complex<$t>, $t>::new(d, n, b)
            }
        }
    };
}
real!(f32, 32);
real!(f64, 64);

/// Owning device array, in extra-dimensions-first physical row-major order.
#[derive(Debug)]
pub struct CudaPencilArray<T: HostScalar, const N: usize, const M: usize> {
    buffer: CudaBuffer<T>,
    pencil: Arc<Pencil<N, M>>,
    extra: ExtraShape,
}
impl<T: HostScalar, const N: usize, const M: usize> CudaPencilArray<T, N, M> {
    pub fn new(device: &CudaDevice, pencil: Arc<Pencil<N, M>>, extra: ExtraShape) -> Result<Self> {
        let n = pencil
            .local_len()
            .checked_mul(extra.element_count())
            .ok_or_else(|| invalid("array size overflow"))?;
        Ok(Self {
            buffer: CudaBuffer::new(device, n)?,
            pencil,
            extra,
        })
    }
    pub fn upload(&mut self, values: &[T]) -> Result<()> {
        T::upload(&mut self.buffer, values)?;
        Ok(())
    }
    pub fn download(&self) -> Result<Vec<T>> {
        Ok(T::download(&self.buffer)?)
    }
    pub fn pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.pencil
    }
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.extra
    }
    pub fn device(&self) -> CudaDevice {
        self.buffer.device()
    }
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[derive(Debug)]
struct Host<T: HostScalar, const N: usize, const M: usize> {
    array: PencilArray<T, N, M>,
    packed: Vec<T>,
    device: CudaBuffer<T>,
}
impl<T: HostScalar, const N: usize, const M: usize> Host<T, N, M> {
    fn new(d: &CudaDevice, p: Arc<Pencil<N, M>>, extra: &ExtraShape) -> Result<Self> {
        let n = p
            .local_len()
            .checked_mul(extra.element_count())
            .ok_or_else(|| invalid("size overflow"))?;
        Ok(Self {
            array: PencilArray::from_vec(p, extra.clone(), zeros(n)?).map_err(invalid)?,
            packed: zeros(n)?,
            device: CudaBuffer::new(d, n)?,
        })
    }
    fn pack(&mut self, n: usize, stride: usize) -> Result<()> {
        pack(self.array.as_slice(), &mut self.packed, n, stride, false)?;
        T::upload(&mut self.device, &self.packed)?;
        Ok(())
    }
    fn unpack(&mut self, n: usize, stride: usize) -> Result<()> {
        let v = T::download(&self.device)?;
        pack(&v, self.array.as_mut_slice(), n, stride, true)
    }
}
// Each block contains `stride` interleaved lines. Extra batches are leading
// dimensions and therefore require no special indexing.
fn pack<T: Copy>(src: &[T], dst: &mut [T], n: usize, stride: usize, reverse: bool) -> Result<()> {
    let block = n
        .checked_mul(stride)
        .filter(|&x| x > 0)
        .ok_or_else(|| invalid("invalid line geometry"))?;
    if src.len() != dst.len() || src.len() % block != 0 {
        return Err(invalid("line buffer size"));
    }
    for (s, d) in src.chunks_exact(block).zip(dst.chunks_exact_mut(block)) {
        for j in 0..stride {
            for k in 0..n {
                if reverse {
                    d[k * stride + j] = s[j * n + k]
                } else {
                    d[j * n + k] = s[k * stride + j]
                }
            }
        }
    }
    Ok(())
}
fn real_boundary<const N: usize, const M: usize>(
    geometry: &[StageGeometry<N, M>],
    selection: AxisSelection<N>,
) -> Option<usize> {
    let axis = (0..N).rfind(|&a| selection.contains(a))?;
    geometry.iter().position(|g| g.axis == axis)
}
fn stride<const N: usize, const M: usize>(p: &Pencil<N, M>, axis: usize) -> usize {
    let pos = p
        .permutation()
        .axes()
        .iter()
        .position(|a| a.index() == axis)
        .expect("checked axis");
    // A zero trailing local dimension means there are no lines on this rank.
    p.local_shape_memory()[pos + 1..]
        .iter()
        .product::<usize>()
        .max(1)
}
#[derive(Debug)]
enum Data<R: CudaReal, const N: usize, const M: usize>
where
    Complex<R>: HostScalar,
{
    Real(Host<R, N, M>),
    Complex(Host<Complex<R>, N, M>),
}
impl<R: CudaReal, const N: usize, const M: usize> Data<R, N, M>
where
    Complex<R>: HostScalar,
{
    fn new(real: bool, d: &CudaDevice, p: Arc<Pencil<N, M>>, e: &ExtraShape) -> Result<Self> {
        if real {
            Ok(Self::Real(Host::new(d, p, e)?))
        } else {
            Ok(Self::Complex(Host::new(d, p, e)?))
        }
    }
}
#[derive(Debug)]
enum Transition<const N: usize, const M: usize> {
    Local(LocalTransposePlan<N, M>),
    All(AllToAllvTransposePlan<N, M>),
    Point(PointToPointTransposePlan<N, M>),
}
impl<const N: usize, const M: usize> Transition<N, M> {
    fn new(a: Arc<Pencil<N, M>>, b: Arc<Pencil<N, M>>, method: TransposeMethod) -> Result<Self> {
        if a.decomposition() == b.decomposition() {
            Ok(Self::Local(LocalTransposePlan::new(a, b).map_err(invalid)?))
        } else {
            match method {
                TransposeMethod::AllToAllv => Ok(Self::All(
                    AllToAllvTransposePlan::new(a, b).map_err(invalid)?,
                )),
                TransposeMethod::PointToPoint => Ok(Self::Point(
                    PointToPointTransposePlan::new(a, b).map_err(invalid)?,
                )),
            }
        }
    }
    fn workspace<T: HostScalar>(&self, extra: &ExtraShape) -> Result<TransposeWorkspace<T>> {
        let (s, r) = match self {
            Self::Local(_) => (0, 0),
            Self::All(p) => {
                let r = p.workspace_requirements(extra).map_err(invalid)?;
                (r.send_len, r.receive_len)
            }
            Self::Point(p) => {
                let r = p.workspace_requirements(extra).map_err(invalid)?;
                (r.send_len, r.receive_len)
            }
        };
        Ok(TransposeWorkspace::from_vecs(zeros(s)?, zeros(r)?))
    }
    fn execute<T: HostScalar>(
        &self,
        a: &Host<T, N, M>,
        b: &mut Host<T, N, M>,
        w: &mut TransposeWorkspace<T>,
    ) -> Result<()> {
        match self {
            Self::Local(p) => p
                .execute_views(a.array.view(), b.array.view_mut())
                .map_err(invalid),
            Self::All(p) => p
                .execute_views(a.array.view(), b.array.view_mut(), w)
                .map_err(invalid),
            Self::Point(p) => p
                .execute_views(a.array.view(), b.array.view_mut(), w)
                .map_err(invalid),
        }
    }
}
#[derive(Debug)]
enum Native<R: CudaReal>
where
    Complex<R>: HostScalar,
{
    Identity,
    Complex(C2CPlan<Complex<R>>),
    Real(R2CPlan<R, Complex<R>>, C2RPlan<Complex<R>, R>),
}
#[derive(Debug)]
struct Stage<R: CudaReal, const N: usize, const M: usize>
where
    Complex<R>: HostScalar,
{
    geometry: StageGeometry<N, M>,
    native: Native<R>,
    n: usize,
    stride: usize,
}
#[derive(Debug)]
struct Edge<const N: usize, const M: usize> {
    forward: Transition<N, M>,
    reverse: Transition<N, M>,
}
#[derive(Debug)]
enum Scratch<R: CudaReal>
where
    Complex<R>: HostScalar,
{
    Real(TransposeWorkspace<R>, TransposeWorkspace<R>),
    Complex(
        TransposeWorkspace<Complex<R>>,
        TransposeWorkspace<Complex<R>>,
    ),
}
/// Cached GPU plans and checked canonical route. The CPU plan is used only
/// during construction to obtain immutable geometry; never to execute FFTs.
#[derive(Debug)]
pub struct DistributedPlan<R: CudaReal, const N: usize, const M: usize>
where
    Complex<R>: HostScalar,
{
    device: CudaDevice,
    stages: Vec<Stage<R, N, M>>,
    edges: Vec<Edge<N, M>>,
    extra: ExtraShape,
    descriptor: Vec<u64>,
    boundary: Option<usize>,
    selection: AxisSelection<N>,
    signs: FourierDirections<N>,
    identity: Arc<()>,
}
/// Caller-owned reusable staging and device buffers. Execution failures after
/// preflight poison this workspace. Allocate a fresh workspace to recover.
#[derive(Debug)]
pub struct DistributedWorkspace<R: CudaReal, const N: usize, const M: usize>
where
    Complex<R>: HostScalar,
{
    inputs: Vec<Data<R, N, M>>,
    outputs: Vec<Data<R, N, M>>,
    edges: Vec<Scratch<R>>,
    identity: Arc<()>,
    poisoned: bool,
}
impl<R: CudaReal, const N: usize, const M: usize> DistributedWorkspace<R, N, M>
where
    Complex<R>: HostScalar,
{
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}
impl<R: CudaReal, const N: usize, const M: usize> DistributedPlan<R, N, M>
where
    Complex<R>: HostScalar,
{
    pub fn c2c(
        input: Arc<Pencil<N, M>>,
        extra: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
        signs: FourierDirections<N>,
        ordinal: usize,
    ) -> Result<Self> {
        Self::new(input, extra, selection, layout, signs, ordinal, false)
    }
    pub fn r2c(
        input: Arc<Pencil<N, M>>,
        extra: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
        ordinal: usize,
    ) -> Result<Self> {
        Self::new(
            input,
            extra,
            selection,
            layout,
            FourierDirections::forward(),
            ordinal,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn new(
        input: Arc<Pencil<N, M>>,
        extra: ExtraShape,
        selection: AxisSelection<N>,
        layout: DistributedLayout,
        signs: FourierDirections<N>,
        ordinal: usize,
        is_real: bool,
    ) -> Result<Self> {
        let comm = input.topology().communicator();
        // Fixed stack header precedes even descriptor allocation. The extra
        // rank/device ordinal intentionally is not part of the descriptor.
        header(comm, 1, &[N as u64, M as u64, R::TAG, u64::from(is_real)])?;
        let descriptor = agree(
            comm,
            (|| {
                let mut d = Vec::new();
                d.try_reserve_exact(
                    16usize
                        .checked_add(N.checked_mul(4).ok_or_else(|| invalid("descriptor size"))?)
                        .and_then(|v| v.checked_add(M.checked_mul(2)?))
                        .and_then(|v| v.checked_add(extra.dimensions().len()))
                        .ok_or_else(|| invalid("descriptor size"))?,
                )
                .map_err(invalid)?;
                d.extend([
                    N as u64,
                    M as u64,
                    R::TAG,
                    u64::from(is_real),
                    u64::from(layout.permute_dims),
                    match layout.transpose_method {
                        TransposeMethod::AllToAllv => 0,
                        TransposeMethod::PointToPoint => 1,
                    },
                ]);
                d.extend(input.global_shape().iter().map(|&v| v as u64));
                d.extend(input.topology().process_grid().iter().map(|&v| v as u64));
                d.extend(input.decomposition().iter().map(|v| v.index() as u64));
                d.extend(input.permutation().axes().iter().map(|v| v.index() as u64));
                for a in 0..N {
                    d.push(u64::from(selection.contains(a)));
                    d.push(u64::from(signs.get(a) == Some(FourierDirection::Backward)));
                }
                d.push(extra.dimensions().len() as u64);
                d.extend(extra.dimensions().iter().map(|&v| v as u64));
                Ok(d)
            })(),
        )?;
        header(comm, 2, &descriptor)?;
        // Device availability is agreed before any CPU routing constructors
        // enter their MPI protocols.
        agree(
            comm,
            if (0..N)
                .any(|a| !selection.contains(a) && signs.get(a) == Some(FourierDirection::Backward))
            {
                Err(invalid("backward sign on unselected axis"))
            } else {
                Ok(())
            },
        )?;
        let device = agree(comm, CudaDevice::with_ordinal(ordinal).map_err(Error::from))?;
        let geometry = if is_real {
            pencil_fft::R2cPlan::<R, N, M>::from_pencil_with_selection_and_layout(
                input.clone(),
                extra.clone(),
                selection,
                layout,
            )
            .map_err(invalid)?
            .stage_geometry()
        } else {
            pencil_fft::C2cPlan::<R, N, M>::from_pencil_with_selection_and_layout(
                input.clone(),
                extra.clone(),
                selection,
                layout,
            )
            .map_err(invalid)?
            .stage_geometry()
        };
        let boundary = if is_real {
            real_boundary(&geometry, selection)
        } else {
            None
        };
        if is_real && boundary.is_none() {
            return Err(invalid("real route has no boundary"));
        }
        let mut stages = agree(
            comm,
            (|| {
                let mut v = Vec::new();
                v.try_reserve_exact(geometry.len()).map_err(invalid)?;
                Ok(v)
            })(),
        )?;
        for (i, g) in geometry.into_vec().into_iter().enumerate() {
            let stage = (|| {
                let n = g.source.global_shape()[g.axis];
                let count = g
                    .source
                    .local_len()
                    .checked_mul(extra.element_count())
                    .ok_or_else(|| invalid("batch size overflow"))?;
                let s = stride(&g.source, g.axis);
                let native = if boundary == Some(i) {
                    Native::Real(
                        R::real(&device, n, count / n)?,
                        R::reverse(&device, n, count / n)?,
                    )
                } else if selection.contains(g.axis) {
                    Native::Complex(R::complex(&device, n, count / n)?)
                } else {
                    Native::Identity
                };
                Ok(Stage {
                    geometry: g,
                    n,
                    stride: s,
                    native,
                })
            })();
            stages.push(agree(comm, stage)?);
        }
        let mut edges = agree(
            comm,
            (|| {
                let mut v = Vec::new();
                v.try_reserve_exact(stages.len().saturating_sub(1))
                    .map_err(invalid)?;
                Ok(v)
            })(),
        )?;
        for pair in stages.windows(2) {
            let a = pair[0].geometry.output.clone();
            let b = pair[1].geometry.source.clone();
            let forward = agree(
                comm,
                Transition::new(a.clone(), b.clone(), layout.transpose_method),
            )?;
            let reverse = agree(comm, Transition::new(b, a, layout.transpose_method))?;
            edges.push(Edge { forward, reverse });
        }
        Ok(Self {
            device,
            stages,
            edges,
            extra,
            descriptor,
            boundary,
            selection,
            signs,
            identity: Arc::new(()),
        })
    }
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
    pub fn input_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.stages[0].geometry.source
    }
    pub fn output_pencil(&self) -> &Arc<Pencil<N, M>> {
        &self.stages.last().expect("route").geometry.output
    }
    pub fn extra_shape(&self) -> &ExtraShape {
        &self.extra
    }
    /// Collectively allocates a C2C input (or reverse destination).
    pub fn allocate_input(&self) -> Result<CudaPencilArray<Complex<R>, N, M>> {
        let comm = self.input_pencil().topology().communicator();
        header(comm, 4, &self.descriptor)?;
        agree(
            comm,
            if self.boundary.is_some() {
                Err(invalid("real plan requires allocate_real_input"))
            } else {
                CudaPencilArray::new(
                    &self.device,
                    self.input_pencil().clone(),
                    self.extra.clone(),
                )
            },
        )
    }
    /// Collectively allocates an R2C input (or C2R destination).
    pub fn allocate_real_input(&self) -> Result<CudaPencilArray<R, N, M>> {
        let comm = self.input_pencil().topology().communicator();
        header(comm, 5, &self.descriptor)?;
        agree(
            comm,
            if self.boundary.is_none() {
                Err(invalid("complex plan requires allocate_input"))
            } else {
                CudaPencilArray::new(
                    &self.device,
                    self.input_pencil().clone(),
                    self.extra.clone(),
                )
            },
        )
    }
    /// Collectively allocates the complex output of either plan kind.
    pub fn allocate_output(&self) -> Result<CudaPencilArray<Complex<R>, N, M>> {
        let comm = self.input_pencil().topology().communicator();
        header(comm, 6, &self.descriptor)?;
        agree(
            comm,
            CudaPencilArray::new(
                &self.device,
                self.output_pencil().clone(),
                self.extra.clone(),
            ),
        )
    }
    pub fn allocate_workspace(&self) -> Result<DistributedWorkspace<R, N, M>> {
        let comm = self.input_pencil().topology().communicator();
        header(comm, 3, &self.descriptor)?;
        agree(
            comm,
            (|| {
                let mut inputs = Vec::new();
                let mut outputs = Vec::new();
                let mut edges = Vec::new();
                inputs
                    .try_reserve_exact(self.stages.len())
                    .map_err(invalid)?;
                outputs
                    .try_reserve_exact(self.stages.len())
                    .map_err(invalid)?;
                edges.try_reserve_exact(self.edges.len()).map_err(invalid)?;
                for (i, s) in self.stages.iter().enumerate() {
                    inputs.push(Data::new(
                        self.boundary.is_some_and(|b| i <= b),
                        &self.device,
                        s.geometry.source.clone(),
                        &self.extra,
                    )?);
                    outputs.push(Data::new(
                        self.boundary.is_some_and(|b| i < b),
                        &self.device,
                        s.geometry.output.clone(),
                        &self.extra,
                    )?);
                }
                for (i, e) in self.edges.iter().enumerate() {
                    edges.push(if self.boundary.is_some_and(|b| i < b) {
                        Scratch::Real(
                            e.forward.workspace(&self.extra)?,
                            e.reverse.workspace(&self.extra)?,
                        )
                    } else {
                        Scratch::Complex(
                            e.forward.workspace(&self.extra)?,
                            e.reverse.workspace(&self.extra)?,
                        )
                    });
                }
                Ok(DistributedWorkspace {
                    inputs,
                    outputs,
                    edges,
                    identity: self.identity.clone(),
                    poisoned: false,
                })
            })(),
        )
    }
    fn preflight<A: HostScalar, B: HostScalar>(
        &self,
        op: u64,
        src: &CudaPencilArray<A, N, M>,
        dst: &CudaPencilArray<B, N, M>,
        w: &DistributedWorkspace<R, N, M>,
        reverse: bool,
        real: bool,
    ) -> Result<()> {
        let comm = self.input_pencil().topology().communicator();
        header(comm, op, &self.descriptor)?;
        let (a, b) = if reverse {
            (self.output_pencil(), self.input_pencil())
        } else {
            (self.input_pencil(), self.output_pencil())
        };
        agree(
            comm,
            if self.boundary.is_some() != real
                || w.poisoned
                || !Arc::ptr_eq(&w.identity, &self.identity)
                || !a.same_layout(&src.pencil)
                || !b.same_layout(&dst.pencil)
                || src.extra != self.extra
                || dst.extra != self.extra
                || !self.device.same(&src.device())
                || !self.device.same(&dst.device())
            {
                Err(invalid("wrong plan, layout, device, or poisoned workspace"))
            } else {
                Ok(())
            },
        )
    }
    pub fn forward(
        &self,
        src: &CudaPencilArray<Complex<R>, N, M>,
        dst: &mut CudaPencilArray<Complex<R>, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
    ) -> Result<()> {
        self.c2c_execute(src, dst, w, false, false)
    }
    pub fn inverse(
        &self,
        src: &CudaPencilArray<Complex<R>, N, M>,
        dst: &mut CudaPencilArray<Complex<R>, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
    ) -> Result<()> {
        self.c2c_execute(src, dst, w, true, true)
    }
    pub fn backward(
        &self,
        src: &CudaPencilArray<Complex<R>, N, M>,
        dst: &mut CudaPencilArray<Complex<R>, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
    ) -> Result<()> {
        self.c2c_execute(src, dst, w, true, false)
    }
    fn c2c_execute(
        &self,
        src: &CudaPencilArray<Complex<R>, N, M>,
        dst: &mut CudaPencilArray<Complex<R>, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
        reverse: bool,
        normalize: bool,
    ) -> Result<()> {
        self.preflight(
            if !reverse {
                10
            } else if normalize {
                11
            } else {
                12
            },
            src,
            dst,
            w,
            reverse,
            false,
        )?;
        w.poisoned = true;
        let comm = self.input_pencil().topology().communicator();
        let values = agree(comm, src.download())?;
        let last = self.stages.len() - 1;
        if let Data::Complex(h) = if reverse {
            &mut w.outputs[last]
        } else {
            &mut w.inputs[0]
        } {
            h.array.as_mut_slice().copy_from_slice(&values)
        }
        self.execute(w, reverse, normalize)?;
        let Data::Complex(h) = (if reverse {
            &w.inputs[0]
        } else {
            &w.outputs[last]
        }) else {
            unreachable!()
        };
        agree(comm, dst.upload(h.array.as_slice()))?;
        w.poisoned = false;
        Ok(())
    }
    pub fn forward_real(
        &self,
        src: &CudaPencilArray<R, N, M>,
        dst: &mut CudaPencilArray<Complex<R>, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
    ) -> Result<()> {
        self.preflight(20, src, dst, w, false, true)?;
        w.poisoned = true;
        let comm = self.input_pencil().topology().communicator();
        let v = agree(comm, src.download())?;
        let Data::Real(h) = &mut w.inputs[0] else {
            unreachable!()
        };
        h.array.as_mut_slice().copy_from_slice(&v);
        self.execute(w, false, false)?;
        let Data::Complex(h) = w.outputs.last().unwrap() else {
            unreachable!()
        };
        agree(comm, dst.upload(h.array.as_slice()))?;
        w.poisoned = false;
        Ok(())
    }
    pub fn inverse_real(
        &self,
        src: &CudaPencilArray<Complex<R>, N, M>,
        dst: &mut CudaPencilArray<R, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
    ) -> Result<()> {
        self.reverse_real(src, dst, w, true)
    }
    pub fn backward_real(
        &self,
        src: &CudaPencilArray<Complex<R>, N, M>,
        dst: &mut CudaPencilArray<R, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
    ) -> Result<()> {
        self.reverse_real(src, dst, w, false)
    }
    fn reverse_real(
        &self,
        src: &CudaPencilArray<Complex<R>, N, M>,
        dst: &mut CudaPencilArray<R, N, M>,
        w: &mut DistributedWorkspace<R, N, M>,
        normalize: bool,
    ) -> Result<()> {
        self.preflight(if normalize { 21 } else { 22 }, src, dst, w, true, true)?;
        w.poisoned = true;
        let comm = self.input_pencil().topology().communicator();
        let v = agree(comm, src.download())?;
        let Data::Complex(h) = w.outputs.last_mut().unwrap() else {
            unreachable!()
        };
        h.array.as_mut_slice().copy_from_slice(&v);
        self.execute(w, true, normalize)?;
        let Data::Real(h) = &w.inputs[0] else {
            unreachable!()
        };
        agree(comm, dst.upload(h.array.as_slice()))?;
        w.poisoned = false;
        Ok(())
    }
    fn execute(
        &self,
        w: &mut DistributedWorkspace<R, N, M>,
        reverse: bool,
        normalize: bool,
    ) -> Result<()> {
        let comm = self.input_pencil().topology().communicator();
        for step in 0..self.stages.len() {
            let i = if reverse {
                self.stages.len() - 1 - step
            } else {
                step
            };
            let s = &self.stages[i];
            if reverse && self.boundary == Some(i) {
                let Data::Complex(h) = &mut w.outputs[i] else {
                    unreachable!()
                };
                let mut depth = 1.0;
                let mut scale = 1.0;
                for a in 0..s.geometry.axis {
                    if self.selection.contains(a) {
                        let n = self.input_pencil().global_shape()[a];
                        if n > 1 {
                            depth += (usize::BITS - (n - 1).leading_zeros()) as f64;
                        }
                        if !normalize {
                            scale *= n as f64;
                        }
                    }
                }
                crate::distributed_real::validate_project(
                    h.array.as_mut_slice(),
                    s.n,
                    s.geometry.output.local_len(),
                    s.stride,
                    self.extra.element_count(),
                    128.0 * R::EPS * depth,
                    128.0 * R::MIN * depth * scale,
                    comm,
                )
                .map_err(invalid)?;
            }
            let result = if reverse {
                self.local(i, &mut w.outputs[i], &mut w.inputs[i], true, normalize)
            } else {
                self.local(i, &mut w.inputs[i], &mut w.outputs[i], false, false)
            };
            agree(comm, result)?;
            if step + 1 < self.stages.len() {
                let e = if reverse { i - 1 } else { i };
                let edge = &self.edges[e];
                let trans = if reverse {
                    &edge.reverse
                } else {
                    &edge.forward
                };
                let (a, b) = if reverse {
                    (&w.inputs[i], &mut w.outputs[i - 1])
                } else {
                    (&w.outputs[i], &mut w.inputs[i + 1])
                };
                let result = match (a, b, &mut w.edges[e]) {
                    (Data::Real(a), Data::Real(b), Scratch::Real(f, r)) => {
                        trans.execute(a, b, if reverse { r } else { f })
                    }
                    (Data::Complex(a), Data::Complex(b), Scratch::Complex(f, r)) => {
                        trans.execute(a, b, if reverse { r } else { f })
                    }
                    _ => Err(invalid("route type mismatch")),
                };
                agree(comm, result)?;
            }
        }
        Ok(())
    }
    fn local(
        &self,
        i: usize,
        src: &mut Data<R, N, M>,
        dst: &mut Data<R, N, M>,
        reverse: bool,
        normalize: bool,
    ) -> Result<()> {
        let s = &self.stages[i];
        match (&s.native, src, dst) {
            (Native::Identity, Data::Real(a), Data::Real(b)) => {
                b.array.as_mut_slice().copy_from_slice(a.array.as_slice())
            }
            (Native::Identity, Data::Complex(a), Data::Complex(b)) => {
                b.array.as_mut_slice().copy_from_slice(a.array.as_slice())
            }
            (Native::Complex(p), Data::Complex(a), Data::Complex(b)) => {
                a.pack(s.n, s.stride)?;
                let positive =
                    (self.signs.get(s.geometry.axis) == Some(FourierDirection::Backward)) ^ reverse;
                p.run(a.device.ptr, b.device.ptr, positive, normalize)?;
                b.unpack(s.n, s.stride)?;
            }
            (Native::Real(p, _), Data::Real(a), Data::Complex(b)) => {
                a.pack(s.n, s.stride)?;
                p.execute(&a.device, &mut b.device)?;
                b.unpack(s.n / 2 + 1, s.stride)?;
            }
            (Native::Real(_, p), Data::Complex(a), Data::Real(b)) => {
                a.pack(s.n / 2 + 1, s.stride)?;
                if normalize {
                    p.execute_inverse(&a.device, &mut b.device)?
                } else {
                    p.execute_backward(&a.device, &mut b.device)?
                }
                b.unpack(s.n, s.stride)?;
            }
            _ => return Err(invalid("native route mismatch")),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packing_roundtrip() {
        for n in [1, 2, 5] {
            for stride in [1, 3] {
                for batches in [0, 1, 4] {
                    let a: Vec<_> = (0..n * stride * batches).collect();
                    let mut b = vec![0; a.len()];
                    let mut c = b.clone();
                    pack(&a, &mut b, n, stride, false).unwrap();
                    pack(&b, &mut c, n, stride, true).unwrap();
                    assert_eq!(a, c);
                    for block in 0..batches {
                        for line in 0..stride {
                            for k in 0..n {
                                assert_eq!(
                                    b[block * n * stride + line * n + k],
                                    a[block * n * stride + k * stride + line]
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(pack(&[1], &mut [0], 2, 1, false).is_err());
        assert!(pack::<u8>(&[], &mut [], 0, 1, false).is_err());
    }
}
