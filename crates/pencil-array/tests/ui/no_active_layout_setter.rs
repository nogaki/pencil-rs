use pencil_array::ManyPencilArray;

fn set_active<T>(many: &mut ManyPencilArray<T, 3, 2>) {
    many.set_active_layout(1);
}

fn main() {}
