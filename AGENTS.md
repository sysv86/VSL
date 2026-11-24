# AGENTS.md

Contributor and agent guidelines for the VSL 2.0 compiler workspace. Read in full before modifying any crate.

## 1. Scope

VSL 2.0 is a pure-Rust, zero-C++ shader language, SPIR-V 1.6 compiler, and unified render-graph DSL (`pack.vsl`) for Vulkan 1.3+ shader packs in Hyaloclast (`vulkanshaders-core`).

Hard boundary between engine core and shader packs:

- Core engine shaders stay in `vulkanshaders-core`: Tier 3 RT pipelines (`pathtrace.{rgen,rchit,rahit,rmiss}`), `UberGpuBuffer` vertex decoding, `SER`/`HitObject`, `VK_EXT_opacity_micromap`, `SHaRC` resolve, `SVGF`/NRD bridge, `TSR`/DLSS/FSR3/XeSS, Optical Flow FrameGen, and raster G-Buffer passes.
- VSL 2.0 owns shader packs (`extend` and `custom` modes): all 16 `@hook` stages, custom compute and raster passes, typed `pack.vsl` authoring, `pack.toml` v2 desugaring, and native SPIR-V 1.6 emission with `CompiledShaderUnit` reflection.
- Explicit non-goals: `RayTracingKHR` SBT stages, `HitObject`/SER, OMM, DXIL/MSL/WGSL/CUDA cross-compilation, automatic differentiation, GPU dynamic dispatch.

## 2. Code standards

- English only in source, identifiers, comments, docs, diagnostics, and commit messages.
- Comments explain intent: invariants, SPIR-V encoding contracts, descriptor ABI rules, layout formulas, driver workarounds. Do not narrate syntax or leave `TODO`/`FIXME` markers.
- No stubs or fabricated APIs. Dependency versions, trait bounds, and method calls must match verified upstream documentation.
- Zero `unsafe` code; the workspace sets `unsafe_code = "forbid"`.

## 3. Licensing

GPL-3.0-only, matching Hyaloclast. Clean-room implementation: do not copy from Shader-Slang, Iris/Aperture, Sodium, OptiFine, or VulkanMod. Follow the Khronos SPIR-V Registry directly (`SPV_KHR_physical_storage_buffer`, `SPV_KHR_vulkan_memory_model`, `SPV_KHR_ray_query`, `SPV_EXT_descriptor_indexing`, `SPV_KHR_maximal_reconvergence`, `SPV_KHR_quad_control`).

## 4. Compiler invariants

1. Single-source-of-truth reflection: descriptor `(set, binding)`, push-constant offsets (`<= 128 B`), `used_specs`, and `used_params` BDA offsets are computed once from typed HIR/IR and emitted identically into SPIR-V and `CompiledShaderUnit`.
2. Hyaloclast 4-set ABI:
   - Set 0 — per-pass: `binding 0` = `WorldUBO` with 64-bit `pack_params_bda`; `bindings 1..=14` = pass targets.
   - Set 1 — scene globals: `0` = `mc::tlas`, `1` = `InstanceMetaSSBO`, `2` = `SharcConfigUBO`, `3` = `EmissiveLightSSBO`, `4` = `RestirReservoirSSBO`.
   - Set 2 — bindless: `binding 0` = 4096-entry `COMBINED_IMAGE_SAMPLER` array. This is the
     Hyaloclast engine-compatibility ABI variant; upstream VSL uses an unsized `Texture2D<f32>[]`,
     which the engine's rspirv reflection rejects. Bindless sampling uses the combined sampler
     directly, with no Set 0 default sampler.
   - Set 3 — pack assets: `bindings 0..=15` (`PACK_RESOURCE_SLOTS = 16`).
3. Compile-time SPIR-V legality: forbid `RayQuery` in structs, arrays, or `OpPhi` merges; forbid logical-pointer locals; validate `#[push_constant]` size `<= 128 B` with no overlap; gate `f16`, 64-bit atomics, float atomics, and `RayQuery` behind their `#[cap(...)]` attributes.
4. Structured IR: preserve `If`/`Switch`/`Loop { body, continuing }` through emission. Lower BDA `.offset(i)` with 64-bit integer arithmetic (`OpConvertPtrToU`, `OpIMul`, `OpIAdd`, `OpConvertUToPtr`), not `OpPtrAccessChain`. Propagate `NonUniform` onto index, `OpAccessChain`, and resource `OpLoad`. No runtime `spirv-opt`.
5. Deterministic `pack.vsl` sandbox: `100_000` evaluation steps, `256` loop iterations, zero filesystem or network IO.
6. Flat-arena emitter: `spirv` constants crate with a flat `Vec<u32>` word stream; no `rspirv` or `naga` in the backend.

## 5. Crate dependency DAG

Dependencies must remain acyclic:

```text
vsl-diagnostics ◄── vsl-syntax ◄── vsl-hir ◄── vsl-ir ◄── vsl-spirv
                       ▲
                       └── vsl-glsl
       ▲                ▲            ▲           ▲           ▲
       │                │            │           │           │
       └──────────── vsl-mc ◄────────┴── vsl-pack┘           │
                                            ▲                │
                                            └─── vsl-driver ─┘
                                                  ▲       ▲
                                            vsl-cli     vsl-lsp
```

## 6. Verification

Run before completing any change:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For SPIR-V emission changes, validate output against Khronos `spirv-val`:

```sh
spirv-val --target-env vulkan1.3 <output.spv>
```
