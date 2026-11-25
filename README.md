# VSL 2.0

Rust compiler workspace for the VSL 2.0 shader language, SPIR-V 1.6 backend, GLSL 460 oracle, and `pack.vsl` render-graph DSL targeting Vulkan 1.3+ shader packs in Hyaloclast (`vulkanshaders-core`).

## Scope

- Owns shader-pack compilation (`extend` and `custom` modes): 16 `@hook` stages, compute/vertex/fragment passes, `pack.vsl` evaluation and scheduling, `pack.toml` v2 desugaring, native SPIR-V 1.6 emission, and `CompiledShaderUnit` reflection.
- Does not implement engine-core Tier 3 RT SBT pipelines (`rgen`/`rchit`/`rahit`/`rmiss`), `HitObject`/SER, OMM, DXIL/MSL/WGSL/CUDA backends, or runtime `spirv-opt`.
- Workspace enforces `unsafe_code = "forbid"` and `clippy::pedantic`.

## Workspace Crates

| Crate | Implemented Responsibilities |
| :--- | :--- |
| `vsl-diagnostics` | Byte-span tracking (`FileId`, `Span`), virtual `SourceDatabase`, `DiagnosticBag`, and terminal rendering via `ariadne`. |
| `vsl-syntax` | `logos` DFA lexer, recursive-descent/Pratt parser with error recovery, and shared AST for `.vsl` and `pack.vsl`. |
| `vsl-mc` | Engine contract: 4-set Vulkan ABI constants, 14 canonical passes, 41 `RES_*` built-in resources, 16 `@hook` stages (`HookAbiRoute`), static byte layouts for `WorldUBO` (544 B) / `InstanceMetaSSBO` / `SharcConfigUBO` / `EmissiveLightSSBO` / `RestirReservoirSSBO`, `PackParamType`/`PackParamLayout`/`SpecConstantDefault`, and `CompiledShaderUnit` reflection types. |
| `vsl-hir` | Arena-backed typed HIR (`HirExpr`/`HirStmt`/`HirBlock` with `TypeId` and `SymbolId`), typechecker (`E0270`–`E0275`), compile-time SPIR-V legality validator (`E0214`–`E0264`), `scalar`/`std430`/`std140` layout engine, generic struct/function/`impl<T: Trait>` monomorphization, early tier/cap pruning, `ModuleGraph` with BLAKE3 L1/L2 hashing, and `PackSchema`. |
| `vsl-ir` | Typed structured SSA IR (`IrType`, `Inst`, `StructuredStmt`, `IrLocalVar`, `IrWorkgroupVar`, `IrParam`, `IrStageIoVar`, GPU intrinsics) and deterministic pass pipeline (`SroaPass`, `Mem2RegPass`, `SelectiveInlinePass`, `UnrollLoopsPass`, `LowerBdaOffsetsPass`, `ConstFoldPass`, `CopyPropPass`, `SimplifyAndGvnPass`, `LicmPass`, `DeadStoreElimPass`, `PropagateNonUniformPass`, `DeadCodeEliminationPass`). |
| `vsl-spirv` | Flat-arena SPIR-V 1.6 emitter using `spirv 0.4` constants (no `rspirv`/`naga`): strict in-order block emission, `O(1)` constant lookup (`IndexSet<u32>`), `Block`-decorated UBO/SSBO/BDA structs, `OpFunction`/`OpFunctionParameter`/`OpFunctionCall`, `PhysicalStorageBuffer64` + `Vulkan` memory model, `SPV_KHR_ray_query`, 64-bit/float/SSBO/workgroup/image atomics + `OpMemoryBarrier`, subgroup/quad ops, `GLSL.std.450` + bit/derivative ops, vertex/fragment I/O + `OpDemoteToHelperInvocation`, `OpSpecConstantOp`, and `LocalSizeId`. |
| `vsl-pack` | Sandboxed `pack.vsl` evaluator (`MAX_EVAL_STEPS = 100_000`, `MAX_LOOP_ITERS = 256`, zero IO), `Compute`/`Raster`/`Transfer` pass nodes (`transfer_pass`, `mip_pass`), `PassDispatch` (`ForTarget`, `Explicit`, `Indirect`), topological DAG scheduler (`SchedulePlan`), `validate_pass_bindings_against_shader` descriptor cross-validation, and `pack.toml` v2 desugarer. |
| `vsl-glsl` | GLSL 460 oracle emitter with generic struct/`impl` monomorphization and Set 2 `sampler2D bindless[4096]` ABI, plus `hook_bridge` (`emit_raster_gbuffer_hook_slot_with_schema`) for `vertex_displace`/`surface`/`gbuffer_pack`. |
| `vsl-driver` | Compilation orchestrator: per-module `ModuleExports` symbol table (`pub` visibility, `struct`/`const`/`fn`/`trait`/`impl`, `E0502`/`E0503`), typed HIR-to-IR lowering with diagnostics (`E0520`–`E0529`), `mc::*` built-in lowering (`reconstruct_world_pos`, `view_dir`, `sample_sky`, typed `param`, `spec`, `resources::*`), native `@hook` chain dispatch with Porter-Duff alpha-over compositing for `@hook(clouds)`, L1 in-memory + L2 in-memory/disk BLAKE3 cache, and parallel entry-point compilation (`std::thread::scope`). |
| `vsl-cli` | `vslc` binary with `check`, `reflect`, `pack-eval`, `build-pack`, `emit-spv`, and `emit-glsl` subcommands. |
| `vsl-lsp` | Language server (`vsl-lsp`) over stdio with Windows-safe URI normalization, bounded `AnalysisCache`, workspace multi-file analysis and `pack.vsl` schema re-validation via `vsl-driver`/`vsl-hir`, context-aware completion with prefix replacement `text_edit`, HIR-backed hover, `goto_definition`, `references`, `rename`, and `semantic_tokens_full`. |

## Descriptor ABI (`vsl-mc`)

| Set | Binding(s) | Layout / Type |
| :--- | :--- | :--- |
| `0` (`DESCRIPTOR_SET_PER_PASS`) | `0` | `WorldUBO` (`std140`, 544 bytes; `cur_view_proj`/`prev_view_proj`/`inv_proj`/`inv_view` at `0..256`, `camera_pos_frac` at `256`, `frame_index` at `268`, `camera_delta` at `272`, `game_time_sec` at `284`, `sun_dir` at `288`, `rain`/`thunder`/`eye_in_fluid`/`depth_far`/`held_light` at `300..320`, `pack_params[12]` at `320..512`, `held_light_position` at `512`, `pack_params_bda: u64` at `528`, `wind_phase` at `536`, `pack_params_bda_pad` at `540`) |
| `0` (`DESCRIPTOR_SET_PER_PASS`) | `1..=14` | Pass-local textures, storage images, and buffers |
| `1` (`DESCRIPTOR_SET_SCENE_GLOBALS`) | `0` | `mc::tlas` (`AccelerationStructure`) |
| `1` (`DESCRIPTOR_SET_SCENE_GLOBALS`) | `1` | `InstanceMetaSSBO` (`std430` runtime array of 144-byte `InstanceMeta`) |
| `1` (`DESCRIPTOR_SET_SCENE_GLOBALS`) | `2` | `SharcConfigUBO` (`std140`, 48 bytes) |
| `1` (`DESCRIPTOR_SET_SCENE_GLOBALS`) | `3` | `EmissiveLightSSBO` (`std430` 16-byte header + runtime array of 48-byte `EmissiveLight`) |
| `1` (`DESCRIPTOR_SET_SCENE_GLOBALS`) | `4` | `RestirReservoirSSBO` (`std430` 16-byte header + runtime array of 48-byte `RestirReservoir`) |
| `2` (`DESCRIPTOR_SET_BINDLESS`) | `0` | `COMBINED_IMAGE_SAMPLER[4096]` (`BINDLESS_IMAGE_COUNT = 4096`, no Set 0 default sampler) |
| `3` (`DESCRIPTOR_SET_PACK_RESOURCES`) | `0..=15` | Pack resource texture slots (`PACK_RESOURCE_SLOTS = 16`) |

## Requirements

- Rust `1.88+` (Edition `2024`, Cargo resolver `3`)
- `spirv-val` (`spirv-tools`) for SPIR-V 1.6 / Vulkan 1.3 binary validation (enforced in CI via `VSL_REQUIRE_SPIRV_VAL=1`)

## Build & Verification

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
VSL_REQUIRE_SPIRV_VAL=1 cargo test --workspace
```

## CLI (`vslc`) & LSP

```sh
# Typecheck and validate a shader module
cargo run -p vsl-cli -- check examples/sunlit_vanilla/reflections.vsl --tier 2

# Emit CompiledShaderUnit reflection JSON
cargo run -p vsl-cli -- reflect examples/sunlit_vanilla/reflections.vsl --tier 2

# Emit SPIR-V 1.6 binaries
cargo run -p vsl-cli -- emit-spv examples/sunlit_vanilla/reflections.vsl --tier 2 -o reflections.spv

# Emit GLSL 460 oracle source
cargo run -p vsl-cli -- emit-glsl examples/sunlit_vanilla/tiled_lighting.vsl

# Evaluate and topologically schedule pack.vsl
cargo run -p vsl-cli -- pack-eval examples/sunlit_vanilla/pack.vsl --tier 2 --json

# Build a full shader pack (compile all modules, validate bindings, emit .spv + manifest.json)
cargo run -p vsl-cli -- build-pack examples/sunlit_vanilla --tier 2 --preset ultra -o target/sunlit_out --json

# Run the LSP server over stdio
cargo run -p vsl-lsp
```

## License

GPL-3.0-only.
