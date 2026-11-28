# Compiler validation fixtures

Original, small VSL programs for compiler and render-graph validation. They are not GPU runtime tests. From the repository root:

```sh
bash examples/compiler_validation/validate.sh
bash examples/rt_validation/validate.sh
```

Requires the workspace Rust toolchain, Bash, Python 3, glslang and SPIRV-Tools. Tested with glslang 16.6.0 and SPIRV-Tools v2026.4 rc2. An optional first argument supplies a parent directory for fresh validation artifacts. `VSL_CLI`, `SPIRV_VAL` and `GLSLANG_VALIDATOR` override tool paths. A supplied `VSL_CLI` skips the build, so ensure that it represents the revision under test.

| Fixture | Coverage | GLSL baseline |
| --- | --- | --- |
| `generics.vsl` | Free generic function instantiated for `u32` and `vec3`, ordinary helper call | Not run: generic free functions are not monomorphized by the GLSL emitter |
| `mutable.vsl` | Mutable scalar, mutable array and indexed buffer writes | Validated |
| `resources.vsl` | Explicit nonzero descriptor set, sparse bindings, sampled texel fetch, storage image and push constants | Validated |
| `control.vsl` | Stepped range, branch, while, switch, conditional expression | Validated |
| `graphics.vsl` | Selected vertex and fragment entries, matching locations and builtin position | Validated |
| `regressions/scalar_inout.vsl` | Scalar `inout` helper and mutable array | Validated |
| `regressions/aggregate_inout.vsl` | Aggregate `inout` helper, member updates and array writes | Validated |
| `regressions/push_constant_params.vsl` | Push-constant parameter named `params` and samplerless fetch | Validated |
| `regressions/loop_continue.vsl` | Stepped loop with `continue`, while-loop header phis | Validated |
| `pack/` | Producer/consumer graph, preset and override evaluation, inferred RAW barrier, project compilation and binding validation | Validated (both entries) |

The standalone fixtures use `--abi standalone`; the pack uses the engine ABI intentionally. The script checks native SPIR-V for Vulkan 1.3 and 1.4, verifies that reflected SPIR-V words exactly match emitted files, compiles supported selected-entry GLSL for both environments, and checks the pack's schedule and barrier JSON. The suite requires 12 native entries and 11 GLSL entries in both environments. All required entries passed on November 28, 2026, with the rebuilt debug CLI. The sole explicitly excluded GLSL path is the free generic function fixture; it is printed as not run and is not counted as a success. To require all twelve GLSL entries (including the currently unsupported generic path), run:

```sh
STRICT_GLSL=1 bash examples/compiler_validation/validate.sh
```

These are compiler-only examples. For actual execution the host must supply buffers with sufficient elements, matching descriptors, push constants, image extents compatible with the dispatch, and correct synchronization. Pack scheduling produces graph dependencies, not Vulkan resource allocation or command buffers.

## Regression coverage

The four fixtures in `regressions/` are mandatory positive tests. They preserve the original source shapes that exposed inliner remap collisions, push-constant name shadowing, and invalid loop phi predecessor lists. Both native and GLSL compilation and validation are required; a recurrence of these failures now causes a nonzero exit. The script does not infer runtime equivalence from validator success.
