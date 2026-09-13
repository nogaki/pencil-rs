use pencil_array::{ExtraShape, Pencil, PencilArrayView, PencilArrayViewMut};

fn construct_views<'a, T>(
    pencil: &'a Pencil<3, 2>,
    extra_shape: &'a ExtraShape,
    storage: &'a mut [T],
) {
    let _ = PencilArrayView::new(pencil, extra_shape, storage);
    let _ = PencilArrayViewMut::new(pencil, extra_shape, storage);
}

fn main() {}
