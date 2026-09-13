use pencil_array::ManyPencilArray;

fn start_internal_write<T>(many: &mut ManyPencilArray<T, 3, 2>) {
    let _ = many.begin_in_place_write();
}

fn main() {}
