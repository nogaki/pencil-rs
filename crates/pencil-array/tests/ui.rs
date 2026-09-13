#[test]
fn public_api_preserves_array_ownership_and_many_layout_state() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/allowed_views.rs");
    cases.compile_fail("tests/ui/no_arbitrary_many_view.rs");
    cases.compile_fail("tests/ui/no_active_layout_setter.rs");
    cases.compile_fail("tests/ui/no_mutable_view_alias.rs");
    cases.compile_fail("tests/ui/no_view_outliving_owner.rs");
    cases.compile_fail("tests/ui/no_public_view_constructors.rs");
    cases.compile_fail("tests/ui/no_internal_layout_write.rs");
    cases.compile_fail("tests/ui/no_raw_communicator_access.rs");
    cases.compile_fail("tests/ui/no_internal_layout_types.rs");
}
