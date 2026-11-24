# VSL 2.0 — Vulkan Shader Language & Unified `pack.vsl` Compiler

Pure-Rust, zero-C++ shader language, structured SSA compiler, direct SPIR-V 1.6 emitter, Stage A GLSL 460 oracle backend, and unified render-graph DSL (`pack.vsl`) for Vulkan 1.3+ shader packs in **Hyaloclast** (`vulkanshaders-core`).

## Architecture

VSL 2.0 replaces legacy `.vsl` v1 text preprocessing and static `pack.toml` manifests with a single statically typed compiler sharing one frontend across `.vsl`, `pack.vsl`, CLI (`vslc`), and LSP (`vsl-lsp`).

### Workspace Crates

| Crate | Role | Key Dependencies |
| :--- | :--- | :--- |
| `vsl-diagnostics` | Byte-span tracking, virtual source database, structured diagnostics, and terminal rendering | `ariadne 0.6`, `smol_str 0.3` |
| `vsl-syntax` | Shared DFA lexer, recursive-descent/Pratt parser with error recovery, and full AST for `.vsl` and `pack.vsl` | `logos 0.16`, `smol_str 0.3` |
| `vsl-mc` | Virtual `mc::*` engine contract: 14 canonical Hyaloclast passes, 41 `RES_*` resources, 16 typed `@hook` stages, 4-set Vulkan ABI, and `CompiledShaderUnit` reflection schema | `bitflags 2.10`, `indexmap 2.14`, `serde` |
| `vsl-hir` | Arena-backed `HirModule`, module graph with cycle detection, BLAKE3 L1/L2 cache keys, `scalar`/`std430`/`std140` layout engine, early tier/cap pruning, single-source-of-truth reflection, and compile-time SPIR-V legality validator | `la-arena 0.3`, `blake3 1.8`, `indexmap 2.14` |
| `vsl-pack` | Deterministic sandboxed `pack.vsl` evaluator (`100_000` step cap, `256` loop cap, zero IO), topological DAG scheduler (`SchedulePlan`), and backwards-compatible `pack.toml` v2 desugarer | `indexmap 2.14`, `toml 1.2` |
| `vsl-ir` | Structured SSA IR (`If`, `Switch`, `Loop { body, continuing, trip_count }`) and mandatory Rust passes (`ConstFold`, `UnrollLoops`, `LowerBdaOffsets`, `PropagateNonUniform`, `DCE`) | `la-arena 0.3`, `indexmap 2.14` |
| `vsl-spirv` | Flat-arena direct SPIR-V 1.6 word emitter with deduplicated types/constants, `OpName`, `OpVariable`, `OpDecorate`, `OpSelectionMerge`/`OpLoopMerge`, `PhysicalStorageBuffer64`, and `Vulkan` memory model | `spirv 0.4`, `indexmap 2.14` |
| `vsl-glsl` | Stage A GLSL 460 oracle backend for differential testing + Raster G-Buffer Hook Bridge (`vertex_displace`, `surface`, `gbuffer_pack` -> `VsMotion.packParams`) | `smol_str 0.3` |
| `vsl-driver` | End-to-end compilation pipeline, 3-level BLAKE3 cache coordinator, GLSL oracle bridge, `pack.vsl` scheduler, and `CompiledShaderUnit` builder | all core crates |
| `vsl-cli` | `vslc` command-line tool (`check`, `reflect`, `pack-eval`, `emit-spv`, `emit-glsl`) | `clap 4.5`, `tracing` |
| `vsl-lsp` | Language server (`vsl-lsp`) reusing `vsl-syntax`, `vsl-hir`, and embedded `vsl-mc` definitions for live diagnostics, hover, and completions | `tower-lsp-server 0.23`, `tokio 1` |

## Requirements

- Rust **1.88+** (Edition **2024**, Cargo resolver **3**)
- Optional for CI SPIR-V binary validation: `spirv-val` (`spirv-tools`)

## Quick Start

```sh
# Format, lint, and run all workspace unit/integration tests
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Validate a VSL 2.0 shader module
cargo run -p vsl-cli -- check examples/sunlit_vanilla/reflections.vsl --tier 2

# Dump single-source-of-truth CompiledShaderUnit reflection JSON
cargo run -p vsl-cli -- reflect examples/sunlit_vanilla/reflections.vsl --tier 2

# Emit Stage A GLSL 460 oracle source
cargo run -p vsl-cli -- emit-glsl examples/sunlit_vanilla/tiled_lighting.vsl

# Evaluate and topologically schedule pack.vsl
cargo run -p vsl-cli -- pack-eval examples/sunlit_vanilla/pack.vsl --tier 2 --json

# Run the language server over stdio
cargo run -p vsl-lsp
```

## License

Licensed under **GPL-3.0-only** to match the Hyaloclast workspace.
