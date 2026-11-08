# VXBIN and disassembly

The pinned Vortex SDK ships `llvm-objdump`, `llvm-objcopy`, and `llvm-readobj`
in `$VORTEX_TOOLDIR/llvm-vortex/bin`. Its LLVM build recognizes Vortex SIMT
instructions when disassembling with `--mattr=+xvortex`. The GNU RISC-V tools
under `riscv32-gnu-toolchain/bin` also handle ordinary RISC-V, but the bundled
GNU objdump prints the SIMT instructions in these examples as raw words.

## File format

VXBIN is a flat device memory image with a header and an optional kernel-name
footer. It is produced from a linked ELF by Vortex's
`sw/kernel/scripts/vxbin.py` and loaded by `sw/runtime/common/module.cpp`.
All integer fields below are little-endian, including 64-bit addresses on RV32.

```text
u64 min_vma                  image load address
u64 max_vma                  exclusive end of reserved device memory
byte[] payload               flat code/data/padding from the ELF

optional footer:
byte[] names                 concatenated UTF-8 names (no NUL required)
N entries, each 16 bytes:
    u32 name_offset          offset into names
    u16 name_length          length in bytes
    u16 padding
    u64 kernel_pc
u32 N
byte[8] magic = "VXSYMTAB"
```

The loader parses the footer backward from the magic. The name blob's size is
the maximum `name_offset + name_length`; subtracting the footer and 16-byte
header gives the payload length. It reserves `[min_vma, max_vma)`, copies only
the payload to `min_vma`, then zeroes any remaining region (BSS). The header and
footer are host metadata, not bytes uploaded with the image. Without a footer,
the loader supplies a `main` entry at `min_vma`.

## Our multiply and gather images

Both generated files have the following structure:

| File region | Offset | Bytes |
| --- | --- | --- |
| Header: `min_vma=0x80000000`, `max_vma=0x80001100` | `0x0000` | 16 |
| Device payload | `0x0010` | 4352 |
| Name blob: `main` | `0x1110` | 4 |
| Entry: name offset 0, length 4, PC `0x80000024` | `0x1114` | 16 |
| Entry count: 1 | `0x1124` | 4 |
| Magic: `VXSYMTAB` | `0x1128` | 8 |

Total: 4400 bytes. There is no extra zero-fill region for these two images.
Startup begins at `0x80000000`; the kernel body is at `0x80000024`. Startup
reads the selected kernel entry from CSR `0xce1` and arguments from `mscratch`,
then calls the entry. Both bodies return at `0x800000e8`; the remaining payload
is padding. VXBIN does not preserve ELF section boundaries, debug information,
or internal function symbols, so arbitrary payload bytes cannot all be assumed
to be instructions.

The generated `.disasm` files decode the extracted VXBIN payload, without
recompiling. The accompanying `.payload.elf` is a synthetic relocatable object
for objdump, not the original linked ELF. `main` is restored from the footer;
`image_base` is an inspection label. Converting this object back to raw bytes
was checked to reproduce the extracted payload exactly.

To reproduce for either of these specific images, from the repository:

```bash
source .cache/vortex-sdk/env.sh
kernel=multiply                    # or gather
base=build/basic/$kernel/$kernel
llvm=$VORTEX_TOOLDIR/llvm-vortex/bin
dd if="$base.vxbin" of="$base.bin" bs=1 skip=16 count=4352 status=none
"$llvm/llvm-objcopy" -I binary -O elf32-littleriscv -B riscv32 \
    --rename-section .data=.text,alloc,load,readonly,code,contents \
    "$base.bin" "$base.raw.elf"
"$llvm/llvm-objcopy" --strip-all \
    --add-symbol image_base=.text:0,function --add-symbol main=.text:0x24,function \
    "$base.raw.elf" "$base.payload.elf"
"$llvm/llvm-objdump" -d --triple=riscv32 --mattr=+m,+a,+f,+xvortex,+zicond \
    --adjust-vma=0x80000000 "$base.payload.elf" > "$base.disasm"
```

The disassembler prints `0x0000700b` at `0x8000001c` as `<unknown>`.
Vortex's `vx_start.S` identifies that exact encoding as `wsync`; the following
`vx_tmc zero` shuts down the warp. The raw disassembly retains the unknown
marker rather than silently substituting a mnemonic.
