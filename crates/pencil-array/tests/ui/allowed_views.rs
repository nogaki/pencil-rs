use pencil_array::{ManyPencilArray, PencilArray};

fn borrow_active<T>(many: &mut ManyPencilArray<T, 3, 2>) {
    let _view = many.active_view().unwrap();
    let _view_mut = many.active_view_mut().unwrap();
}

fn borrow_owner<T>(array: &mut PencilArray<T, 3, 2>) {
    let _view = array.view();
    let _view_mut = array.view_mut();
}

fn main() {}
