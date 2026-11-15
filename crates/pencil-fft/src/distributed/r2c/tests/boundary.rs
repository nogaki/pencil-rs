use super::*;

fn endpoint_plane_for_test(real_len: usize, k: usize) -> Option<usize> {
    let complex_len = real_len / 2 + 1;
    if k == 0 {
        Some(0)
    } else if real_len.is_multiple_of(2) && k == complex_len - 1 {
        Some(1)
    } else {
        None
    }
}

#[test]
fn shared_endpoint_visit_matches_full_scan_for_owners_and_edge_shapes() {
    for (real_len, global_start, local_axis_len, stride, local_len, value_count) in [
        (5, 0, 3, 1, 3, 6), // odd: DC only, across two batches
        (6, 0, 2, 2, 4, 8), // even: DC owner, strided across two batches
        (6, 2, 2, 2, 4, 8), // even: Nyquist owner, strided across two batches
        (1, 0, 1, 1, 1, 2), // length one: DC only
        (6, 4, 0, 0, 0, 0), // rank with no reduced-axis elements
        (6, 0, 4, 1, 4, 0), // zero extra batches
    ] {
        let original: Vec<_> = (0..value_count)
            .map(|index| Complex::new(index as f64, index as f64 + 10.0))
            .collect();
        let mut actual = original.clone();
        let mut actual_visits = Vec::new();
        for_each_boundary_endpoint(
            &mut actual,
            local_len,
            local_axis_len,
            global_start,
            stride,
            real_len,
            |offset, global_k, plane, value| {
                actual_visits.push((offset, global_k, plane));
                value.im = 0.0;
            },
        )
        .unwrap();

        let mut expected = original;
        let mut expected_visits = Vec::new();
        if local_len > 0 {
            for (offset, value) in expected.iter_mut().enumerate() {
                let local_offset = offset % local_len;
                let global_k = global_start + (local_offset / stride) % local_axis_len;
                if let Some(plane) = endpoint_plane_for_test(real_len, global_k) {
                    expected_visits.push((offset, global_k, plane));
                    value.im = 0.0;
                }
            }
        }
        assert_eq!(actual_visits, expected_visits, "real_len={real_len}");
        assert_eq!(actual, expected, "real_len={real_len}");
    }
}
