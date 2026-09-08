# Nixe

This is an educational and experimental project written in Rust. Its long-term goal is to research and build
functional emulators for Nintendo Switch and Nintendo Switch 2 while sharing well-defined components between
both platforms whenever technically appropriate.

## Goals

- Study modern console emulation and low-level systems programming.
- Build a functional Nintendo Switch emulator incrementally.
- Prepare an extensible foundation for Nintendo Switch 2 research and emulation.
- Reuse CPU, memory, graphics, tooling, and other infrastructure when the underlying behavior is genuinely
  shared.
- Favor testable, documented, and maintainable Rust code.

## Project Status

The project is in active development. Some homebrew applications run at up to 60 FPS, and gamepad input is
supported through the emulated HID services.

### Screenshots

<table>
  <tr>
    <td><img src="docs/screenshots/es2gears.png" alt="ES2Gears"></td>
    <td><img src="docs/screenshots/textured_cube.png" alt="Textured cube"></td>
  </tr>
</table>

## Modified Cranelift backend

Nixe uses a modified [Cranelift](https://cranelift.dev/) compiler to translate console instructions into native CPU code. Main changes:

- **Less repeated setup:** entering and leaving a chain of compiled blocks shares one setup and cleanup step for the whole chain.
- **Context kept close:** dedicated CPU registers keep frequently needed emulator data ready to use, for example, the guest-memory base address, the remaining instruction budget, a pointer to shared working space and more.
- **Shared working space:** blocks use a fixed area for temporary values without growing the host stack.
- **Multiple entry points:** execution can enter an optimized region at selected positions without duplicating its code.
- **Fewer data transfers:** blocks know where their inputs and outputs live, so transitions move only what is needed.
- **Direct links:** jumps in generated machine code can be patched to point directly to compiled destinations, so linked blocks jump straight to one another without returning to the emulator's control code or repeating destination lookups.
- **Precise fault recovery data:** memory accesses carry the information needed to reconstruct the emulated CPU state if they fault.
- **Easier maintenance:** readable compiler output and focused tests help catch regressions when updating Cranelift.

See [Cranelift modifications](docs/cranelift-modifications.md) for implementation details.

## Running

See [host requirements](docs/host-requirements.md) for the required CPU capabilities.

The default configuration is in [`nixe.toml`](nixe.toml). List available titles with:

```bash
cargo cli list
```

Run a title by its ID or name:

```bash
cargo cli run <id | name>
```

## Testing

Run

```
cargo test-all
```

### Integration tests against real titles

To run integration tests against caller-owned titles, copy `.env.integration.example` to
`.env.integration`, configure the paths, and run `./scripts/test-integration.sh`.

### Differential tests

Optional CPU differential tests require QEMU user-mode (`qemu-aarch64`), the Rust
`aarch64-unknown-linux-gnu` target, and its cross-linker.

```bash
sudo apt update && sudo apt install qemu-user gcc-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu
```

Verify with

```bash
qemu-aarch64 --version
aarch64-linux-gnu-gcc --version
```

Then run

```bash
cargo test-diff
```

To run a fast, focused A64 differential test for one instruction family or coverage ID, use:

```bash
NIXE_DIFF_FAMILY=simd-duplicate-element cargo test-diff-a64
NIXE_DIFF_COVERAGE_ID=0x8c cargo test-diff-a64
```

### Fuzz tests

CPU decoder and memory fuzz targets require a nightly Rust toolchain
and `cargo-fuzz`:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
cargo fuzz-all
```

See [fuzz/README.md](fuzz/README.md) for target-specific commands and configuration.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Legal Notice

This project is intended for lawful education, research, interoperability, and preservation work. It does not
provide or distribute games, firmware, cryptographic keys, copyrighted console files, or leaked confidential
material.

Users and contributors are responsible for complying with the laws applicable in their jurisdictions and for
using only software and data they are legally entitled to use.

Nintendo Switch and Nintendo Switch 2 are trademarks of Nintendo. This project is independent and is not
affiliated with, sponsored by, or endorsed by Nintendo or NVIDIA.

## License

Nixe is licensed under the GNU General Public License version 3 or later. See [LICENSE.txt](LICENSE.txt).
