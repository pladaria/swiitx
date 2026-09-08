use std::{fs, path::Path};

#[test]
fn jit_owns_cranelift_without_a_production_interpreter_or_runtime_dependency() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let production_dependencies = manifest
        .split("[dev-dependencies]")
        .next()
        .expect("the manifest has a production dependency section");

    for required in [
        "cranelift-codegen = ",
        "cranelift-frontend = ",
        "cranelift-jit = ",
        "cranelift-module = ",
        "cranelift-native = ",
        "nixe-cpu.workspace = true",
        "nixe-memory.workspace = true",
    ] {
        assert!(
            production_dependencies.contains(required),
            "missing dependency: {required}"
        );
    }
    for forbidden in [
        "nixe-cpu-interpreter.workspace",
        "nixe-runtime.workspace",
        "nixe-horizon.workspace",
        "nixe-scheduler.workspace",
        "nixe-gpu.workspace",
    ] {
        assert!(
            !production_dependencies.contains(forbidden),
            "forbidden production dependency: {forbidden}"
        );
    }
    assert!(
        manifest
            .split_once("[dev-dependencies]")
            .unwrap()
            .1
            .contains("nixe-cpu-interpreter.workspace = true"),
        "the reference interpreter must remain test-only"
    );
    for forbidden in ["dynasm", "iced-x86"] {
        assert!(
            !manifest.contains(forbidden),
            "unexpected native-code dependency: {forbidden}"
        );
    }

    let library = fs::read_to_string(root.join("src/lib.rs")).unwrap();
    assert!(library.contains("pub use direct::{JitProcess, JitThread};"));
}

#[test]
fn neutral_crates_do_not_depend_on_cranelift_or_the_jit_backend() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for relative in ["crates/cpu/Cargo.toml", "crates/memory/Cargo.toml"] {
        let manifest = fs::read_to_string(workspace.join(relative)).unwrap();
        assert!(
            !manifest.contains("cranelift"),
            "{relative} imports Cranelift"
        );
        assert!(
            !manifest.contains("nixe-cpu-jit"),
            "{relative} imports the concrete JIT backend"
        );
    }

    let runtime = fs::read_to_string(workspace.join("crates/runtime/Cargo.toml")).unwrap();
    assert!(runtime.contains("nixe-cpu-interpreter.workspace = true"));
    assert!(runtime.contains("nixe-cpu-jit.workspace = true"));
}
