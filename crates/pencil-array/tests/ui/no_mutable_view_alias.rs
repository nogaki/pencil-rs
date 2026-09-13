use pencil_array::ManyPencilArray;

fn alias_mutable_view<T>(many: &mut ManyPencilArray<T, 3, 2>) {
    let first = many.active_view_mut().unwrap();
    let second = many.active_view_mut().unwrap();
    let _ = (first, second);
}

fn main() {}
