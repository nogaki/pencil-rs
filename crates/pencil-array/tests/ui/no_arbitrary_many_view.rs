use pencil_array::ManyPencilArray;

fn arbitrary_view<T>(many: &ManyPencilArray<T, 3, 2>) {
    let _ = many.view_at(1);
}

fn main() {}
