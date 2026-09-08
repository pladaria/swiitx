# Host requirements

These are required CPU capabilities, not a guarantee of title compatibility or
performance. RAM, GPU and operating-system requirements are not specified here.

## x86-64

Nixe requires **LAHF/SAHF support in 64-bit mode**, advertised by
`CPUID.80000001H:ECX[0]` (`LAHF_SAHF_64`). This is an explicit CPU requirement,
not a requirement for the entire x86-64-v2 or AVX feature sets.

The tiered JIT uses SAHF when a fragment expects guest NZCV in host condition
flags. It installs sign, zero and carry without using the host stack; overflow
is established separately because SAHF leaves it unchanged. Matching native
contracts need no conversion. See the [Intel instruction reference](https://cdrdv2-public.intel.com/782151/253667-sdm-vol-2b.pdf).

The JIT checks support when creating a process, before compiling or executing
code. An incompatible host fails initialization with a clear error; there is
no alternate JIT path for CPUs lacking this feature. Virtual machines must
expose the feature to the guest OS. There is no CPUID check per invocation,
fragment or link.

## AArch64

SAHF is an x86 instruction and imposes no requirement on AArch64 hosts. Their
native boundary reads and writes the architecture's NZCV register directly.
