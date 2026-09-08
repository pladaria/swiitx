# AArch64 tests under QEMU

Use **QEMU 11.1.1**, Linux user-mode target `aarch64-linux-user`, with CPU
model `max`. QEMU 6.2 can execute stale code in our dual-mapped JIT reuse tests;
newer QEMU implements the required [IC IVAU handling](https://patchew.org/QEMU/168722304495.6281.8113287217736957231-0%40git.sr.ht/).
Keep binaries and build directories outside the repository.

## Build

This recipe was validated on Linux Mint 21.3 / Ubuntu 22.04, x86-64, with
GCC 11.4 and GLib 2.72.4. Build prerequisites are `build-essential`,
`ninja-build`, `pkg-config`, `python3-venv`, `libglib2.0-dev`, `zlib1g-dev`,
`libgmp-dev`, `curl` and `xz-utils` (Debian/Ubuntu package names).

Run in Bash. The checksum pins the downloaded [official release archive](https://download.qemu.org/qemu-11.1.1.tar.xz).

```bash
set -e
qemu_build_dir="$(mktemp -d)"
cd "$qemu_build_dir"
curl -fL -o qemu-11.1.1.tar.xz https://download.qemu.org/qemu-11.1.1.tar.xz
printf '%s\n' '079ffbff8a7111bbc89022107cbabf3bbfd614d5fc9d7cc675991196aca12482  qemu-11.1.1.tar.xz' | sha256sum -c -
tar -xf qemu-11.1.1.tar.xz
python3 -m venv build-tools
build-tools/bin/python -m pip install 'tomli==2.4.1' 'meson==1.11.1'
mkdir build
cd build
../qemu-11.1.1/configure \
  --python="$qemu_build_dir/build-tools/bin/python" \
  --target-list=aarch64-linux-user --disable-system --disable-docs \
  --disable-tools --disable-guest-agent --disable-debug-info \
  --disable-plugins --disable-rust --prefix=/usr/local
ninja -j 12 qemu-aarch64
./qemu-aarch64 --version
```

Optional installation, from that build directory (leaves `/usr/bin` untouched):

```bash
sudo install -m 0755 qemu-aarch64 /usr/local/bin/qemu-aarch64
hash -r
```

The binary uses host shared libraries; copying it to another distribution is
not a substitute for building there. This pins QEMU inputs, not the complete
host kernel, compiler or guest sysroot, and does not promise bit-identical builds.

## Run the tiered-JIT foundation tests

Install the Rust target with `rustup target add aarch64-unknown-linux-gnu`.
The cross linker and guest sysroot come from `gcc-aarch64-linux-gnu` and
`libc6-dev-arm64-cross` on Debian/Ubuntu. From the Nixe repository, with Cargo
dependencies already fetched:

```bash
/usr/local/bin/qemu-aarch64 --version
cargo test --offline --locked -p nixe-cpu-jit --lib \
  --target aarch64-unknown-linux-gnu \
  --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc"' \
  --config 'target.aarch64-unknown-linux-gnu.runner=["/usr/local/bin/qemu-aarch64", "-cpu", "max", "-L", "/usr/aarch64-linux-gnu"]' \
  --quiet -- --skip direct::
```

Without installation, replace the runner path with the built binary's absolute
path. `--skip direct::` excludes the legacy JIT tests, not the new foundation.
QEMU execution does not validate native Arm cache-coherence or memory-ordering
behavior; native AArch64 hardware testing remains necessary.
