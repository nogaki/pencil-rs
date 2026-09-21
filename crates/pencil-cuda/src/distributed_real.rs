//! Finite, per-extra-batch/per-plane normwise C2R endpoint validation.
use mpi::collective::{CommunicatorCollectives, SystemOperation};
use num_complex::Complex;

fn geometry(
    values: usize,
    n: usize,
    local: usize,
    stride: usize,
    extra: usize,
) -> Option<(usize, usize)> {
    if n == 0 || stride == 0 {
        return None;
    }
    let bins = n / 2 + 1;
    let block = bins.checked_mul(stride)?;
    if local % block != 0 || local.checked_mul(extra)? != values {
        return None;
    }
    Some((bins, block))
}
#[derive(Debug, Default, Clone, Copy)]
struct Stats {
    max: f64,
    imag: f64,
    norm_sq: f64,
    imag_sq: f64,
    invalid: bool,
}
fn stats<R: Copy + Into<f64>>(
    values: &[Complex<R>],
    bins: usize,
    stride: usize,
    plane: usize,
    scale: f64,
) -> Stats {
    let mut s = Stats::default();
    let k = if plane == 0 { 0 } else { bins - 1 };
    for block in values.chunks_exact(bins * stride) {
        for inner in 0..stride {
            let z = block[k * stride + inner];
            let re = z.re.into();
            let im = z.im.into();
            if !re.is_finite() || !im.is_finite() {
                s.invalid = true;
                continue;
            }
            s.max = s.max.max(re.abs().max(im.abs()));
            s.imag = s.imag.max(im.abs());
            if scale != 0.0 {
                let r = re / scale;
                let i = im / scale;
                s.norm_sq += r * r + i * i;
                s.imag_sq += i * i;
            }
        }
    }
    s
}
fn accepted(imag: f64, imag_sq: f64, norm_sq: f64, relative: f64, absolute: f64) -> bool {
    [imag, imag_sq, norm_sq, relative, absolute]
        .iter()
        .all(|v| v.is_finite())
        && relative >= 0.0
        && absolute >= 0.0
        && (imag <= absolute || imag_sq.sqrt() <= relative * norm_sq.sqrt())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_project<R: pencil_fft::FftReal + Into<f64> + Default>(
    values: &mut [Complex<R>],
    n: usize,
    local_len: usize,
    stride: usize,
    extra: usize,
    relative: f64,
    absolute: f64,
    comm: &impl CommunicatorCollectives,
) -> Result<(), String> {
    // Stride and local extent legitimately differ between ranks; never place
    // them in the global descriptor. Agree local geometry before reductions.
    let config = [
        n as u64,
        extra as u64,
        relative.to_bits(),
        absolute.to_bits(),
        1,
    ];
    let mut lo = [0; 5];
    let mut hi = [0; 5];
    comm.all_reduce_into(&config, &mut lo, SystemOperation::min());
    comm.all_reduce_into(&config, &mut hi, SystemOperation::max());
    let g = geometry(values.len(), n, local_len, stride, extra);
    let ok = i32::from(g.is_some() && lo == hi);
    let mut all = 0;
    comm.all_reduce_into(&ok, &mut all, SystemOperation::min());
    if all == 0 {
        return Err("distributed real boundary geometry mismatch".into());
    }
    let (bins, block) = g.expect("collectively checked geometry");
    let planes = if n % 2 == 0 { 2 } else { 1 };
    let mut invalid = false;
    // Same policy as CPU: independent statistics per batch and endpoint
    // plane, not one norm that could hide an invalid batch/plane in another.
    for batch in 0..extra {
        let values = &values[batch * local_len..(batch + 1) * local_len];
        let mut local_max = [0.0; 4];
        for plane in 0..planes {
            let s = stats(values, bins, stride, plane, 0.0);
            invalid |= s.invalid;
            local_max[2 * plane] = s.max;
            local_max[2 * plane + 1] = s.imag;
        }
        let mut global_max = [0.0; 4];
        comm.all_reduce_into(&local_max, &mut global_max, SystemOperation::max());
        let mut local_sum = [0.0; 4];
        for plane in 0..planes {
            let s = stats(values, bins, stride, plane, global_max[2 * plane]);
            local_sum[2 * plane] = s.norm_sq;
            local_sum[2 * plane + 1] = s.imag_sq;
        }
        let mut global_sum = [0.0; 4];
        comm.all_reduce_into(&local_sum, &mut global_sum, SystemOperation::sum());
        for plane in 0..planes {
            invalid |= !accepted(
                global_max[2 * plane + 1],
                global_sum[2 * plane + 1],
                global_sum[2 * plane],
                relative,
                absolute,
            );
        }
    }
    let ok = i32::from(!invalid);
    comm.all_reduce_into(&ok, &mut all, SystemOperation::min());
    if all == 0 {
        return Err("distributed real boundary failed endpoint validation".into());
    }
    // No mutation before the final agreement. The native C2R API requires
    // strictly zero imaginary endpoints, even for accepted distributed noise.
    for values in values.chunks_exact_mut(block) {
        for inner in 0..stride {
            values[inner].im = R::default();
            if planes == 2 {
                values[(bins - 1) * stride + inner].im = R::default();
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pure_geometry_covers_lengths_and_overflow() {
        for n in [1, 2, 3, 4, 5] {
            let bins = n / 2 + 1;
            for stride in [1, 3] {
                assert_eq!(
                    geometry(bins * stride * 4, n, bins * stride * 2, stride, 2),
                    Some((bins, bins * stride))
                );
                assert!(geometry(0, n, 0, stride, 2).is_some());
            }
        }
        assert!(geometry(0, 0, 0, 1, 1).is_none());
        assert!(geometry(0, 1, 0, 0, 1).is_none());
        assert!(geometry(0, usize::MAX, 0, usize::MAX, 1).is_none());
        assert!(geometry(0, 1, usize::MAX, 1, 2).is_none());
        assert!(geometry(1, 4, 1, 1, 1).is_none());
    }
    #[test]
    fn pure_stats_handles_finite_nonfinite_and_empty() {
        let v = [
            Complex::new(3.0, 4.0),
            Complex::new(f64::NAN, f64::INFINITY),
            Complex::new(8.0, 0.0),
        ];
        let dc = stats(&v, 3, 1, 0, 4.0);
        assert_eq!(dc.norm_sq, 25.0 / 16.0);
        assert_eq!(dc.imag_sq, 1.0);
        assert!(!dc.invalid);
        assert_eq!(stats(&v, 3, 1, 1, 8.0).norm_sq, 1.0);
        assert!(stats(&v[1..2], 1, 1, 0, 1.0).invalid);
        assert_eq!(stats::<f64>(&[], 1, 1, 0, 0.0).norm_sq, 0.0);
    }
    #[test]
    fn pure_acceptance_is_normwise_absolute_or_relative() {
        assert!(accepted(1e-9, 1.0, 1.0, 0.0, 1e-9));
        assert!(accepted(1.0, 1.0, 1e8, 1e-4, 0.0));
        assert!(!accepted(1.0, 1.0, 1e8, 1e-5, 0.0));
        assert!(!accepted(f64::NAN, 0.0, 1.0, 1.0, 1.0));
        assert!(!accepted(0.0, f64::INFINITY, 1.0, 1.0, 1.0));
        assert!(!accepted(0.0, 0.0, 0.0, -1.0, 0.0));
        // Scaling by global maximum avoids overflow near f64::MAX.
        let z = [Complex::new(f64::MAX, 1.0)];
        let s = stats(&z, 1, 1, 0, f64::MAX);
        assert!(accepted(s.imag, s.imag_sq, s.norm_sq, 1e-12, 0.0));
    }
    #[test]
    fn mpi_helper_collective_geometry_and_projection() {
        use mpi::topology::Communicator;
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();
        let mut bad = vec![Complex::new(1.0f64, 9.0)];
        if rank == 0 {
            bad.clear()
        }
        assert!(validate_project(&mut bad, 1, 1, 1, 1, 0.0, 0.0, &world).is_err());
        for n in [1, 2, 3, 4, 5] {
            let stride = (rank + 1) as usize;
            let bins = n / 2 + 1;
            let mut good = vec![Complex::new(1.0, 1e-14); bins * stride * 2];
            validate_project(&mut good, n, bins * stride, stride, 2, 1e-12, 0.0, &world).unwrap();
            for b in good.chunks_exact(bins * stride) {
                for j in 0..stride {
                    assert_eq!(b[j].im, 0.0);
                    if n % 2 == 0 {
                        assert_eq!(b[(bins - 1) * stride + j].im, 0.0);
                    }
                }
            }
        }
        let mut bad_plane = vec![Complex::new(1e30, 0.0), Complex::new(0.0, 1.0)];
        let original = bad_plane.clone();
        assert!(validate_project(&mut bad_plane, 2, 2, 1, 1, 1e-12, 0.0, &world).is_err());
        assert_eq!(bad_plane, original);
        let mut bad_batch = original.clone();
        assert!(validate_project(&mut bad_batch, 1, 1, 1, 2, 1e-12, 0.0, &world).is_err());
        assert_eq!(bad_batch, original);
        let mut nan = vec![Complex::new(f64::NAN, 0.0)];
        assert!(validate_project(&mut nan, 1, 1, 1, 1, 0.0, 1.0, &world).is_err());
        assert!(nan[0].re.is_nan());
        if rank == 0 {
            println!("CUDA_ENDPOINT_HOST_OK ranks={}", world.size());
        }
    }
}
