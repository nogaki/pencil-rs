use pencil_array::{PencilArray, PencilArrayView};

fn outlive_owner<'a, T>(array: PencilArray<T, 3, 2>) -> PencilArrayView<'a, T, 3, 2> {
    array.view()
}

fn main() {}
