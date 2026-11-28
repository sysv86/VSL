# VSL ray tracing compiler

VSL emits SPIR-V 1.6 for Vulkan 1.3 and later. The standalone ABI supports explicit descriptors and independent ray tracing shader modules. This guide covers compiler output and host integration, not a complete Vulkan renderer.

## Reproduce the validation

From the repository root, install the Rust workspace toolchain, Bash, Python 3, glslang and SPIRV-Tools with support for the extensions being tested. The reference validation used glslang 16.6.0 and SPIRV-Tools v2026.4 (rc2). An older validator may not recognize newer KHR opacity micromap or EXT invocation-reorder instructions.

```sh
bash examples/rt_validation/validate.sh
```

The script builds `vslc`, checks the sources, emits native SPIR-V and reflection, and translates each reflected entry separately to GLSL. It compiles that GLSL with `glslangValidator` for both `vulkan1.3` and `vulkan1.4`, then validates both compiler outputs with `spirv-val` for the corresponding environment. Pass an output directory as its first argument to retain artifacts at a known location. Set `VSL_CLI` to use an already built compiler instead of rebuilding; `SPIRV_VAL` and `GLSLANG_VALIDATOR` select validator executables. Any failure or unexpected entry count causes a nonzero exit. No shader is executed on a GPU. See also [non-RT compiler validation](../examples/compiler_validation/README.md).

| Module | Entry points | Compiler options (in addition to `--abi standalone`) |
| --- | --- | --- |
| [ray_query.vsl](../examples/rt_validation/ray_query.vsl) | `query_image` | `--tier 2 --cap ray_query` |
| [pipeline.vsl](../examples/rt_validation/pipeline.vsl) | `primary`, `triangle_color`, `background` | `--tier 3` |
| [callable.vsl](../examples/rt_validation/callable.vsl) | `call_color`, `swap_channels` | `--tier 3` |
| [shader_record.vsl](../examples/rt_validation/shader_record.vsl) | `record_color` | `--tier 3` |
| [ser_nv.vsl](../examples/rt_validation/ser_nv.vsl) | `deferred_nv` | `--tier 3 --cap ser_nv` |
| [ser_ext.vsl](../examples/rt_validation/ser_ext.vsl) | `deferred_ext` | `--tier 3 --cap ser_ext` |
| [omm_khr.vsl](../examples/rt_validation/omm_khr.vsl) | `micromap_positions` | `--tier 2 --cap ray_query --cap omm_khr --cap position_fetch` |

All seven modules and ten entry points passed native compilation, selected-entry GLSL compilation, and SPIR-V validation in both Vulkan 1.3 and 1.4 environments on November 28, 2026. The previously reported hardware smoke suite on an NVIDIA GeForce RTX 3050 recorded 501 passes and zero failures. That is a separate result: the examples in this directory have not been executed on a GPU, and the earlier smoke run does not establish hardware support for every extension illustrated here.

### CLI usage

```sh
cargo build -p vsl-cli --bin vslc
mkdir -p /tmp/vsl-pipeline
vslc=target/debug/vslc
src=examples/rt_validation/pipeline.vsl
"$vslc" check "$src" --abi standalone --tier 3
"$vslc" emit-spv "$src" --abi standalone --tier 3 -o /tmp/vsl-pipeline
"$vslc" reflect "$src" --abi standalone --tier 3 > /tmp/vsl-pipeline/reflection.json
spirv-val --target-env vulkan1.3 /tmp/vsl-pipeline/primary.spv
spirv-val --target-env vulkan1.4 /tmp/vsl-pipeline/primary.spv
"$vslc" emit-glsl "$src" --abi standalone --entry primary -o /tmp/vsl-pipeline/primary.glsl
glslangValidator -V --target-env vulkan1.3 -S rgen /tmp/vsl-pipeline/primary.glsl -o /tmp/vsl-pipeline/primary.glsl.spv
spirv-val --target-env vulkan1.3 /tmp/vsl-pipeline/primary.glsl.spv
```

`emit-spv --entry primary -o /tmp/primary.spv` selects one active entry point. An existing output directory receives one `<entry>.spv` file per selected entry; multiple entries with a non-directory file path receive suffixed filenames. `reflect` writes a JSON array to standard output, including SPIR-V words. `check`, `reflect` and `emit-spv` accept `--tier`, `--abi`, repeatable `--cap` and optional `--pack`. The tier defaults to 2, and the ABI defaults to `hyaloclast`; standalone is explicit throughout these examples. Optional capability names are additive, not queries of installed GPU features.

The compiler's tier-3 baseline includes ray tracing pipelines and NV SER, but EXT SER and KHR OMM have distinct capability flags. The examples explicitly request optional capabilities in source with `#[cap(...)]` and enable them in compiler options. The `emit-glsl` command is a separate GLSL 460 oracle path supporting compute, vertex, fragment and the six RT stages. It accepts `--abi` and `--entry` but not `--tier`, `--cap` or `--pack`; optional RT extensions are selected from source attributes. A module with multiple shader entries requires `--entry`. For glslang, select the matching stage (`comp`, `vert`, `frag`, `rgen`, `rchit`, `rahit`, `rmiss`, `rint`, `rcall`). The Rust driver exposes `VslCompiler::compile_shader_to_glsl_entry` and the emitter exposes `emit_glsl_460_entry_with_options`. This is a separate backend, not a proof of semantic equivalence with native SPIR-V. Unsupported features or invalid entries can produce diagnostics; some unsupported paths are still detected only by downstream GLSL compilation. Mixed stricter nested shader-record layouts are rejected by the GLSL emitter rather than silently changing the ABI.

## Ray query

`RayQuery` is a function-local traversal object. Initialize it against a bound `AccelerationStructure`, then call `proceed()` until traversal ends. Candidate triangle type is `0u`; committed triangle type is `1u`. These are different enumerations. Confirm a nonopaque triangle candidate only while it is current. The examples use triangle geometry and do not generate procedural intersections for AABB candidates.

Committed intersection reads occur inside a committed-triangle guard. The compiler does not prove that every query read is valid at runtime. Do not read candidate data after traversal has ended or read committed triangle attributes when there is no committed triangle. Ray queries can run in compute shaders without an SBT or a ray tracing pipeline; a host still needs an acceleration structure, descriptors, and the Vulkan ray-query feature.

## Native ray tracing pipeline

`@raygen`, `@closest_hit`, `@any_hit`, `@miss`, `@intersection` and `@callable` declare native RT entry points. The host creates the shader modules, pipeline groups, SBT regions and dispatch dimensions; VSL does not derive this arrangement from the names or order of functions in a module.

`trace_ray(scene, flags, mask, sbt_offset, sbt_stride, miss_index, origin, t_min, direction, t_max, payload)` invokes traversal and the selected shaders. The payload argument must be a compatible stage interface variable. `pipeline.vsl` uses one triangle hit group and one miss group, both with payload location zero. The closest-hit shader reads the fixed-function triangle barycentric hit attribute. The ray-generation shader reloads the payload after the trace and writes it to the output image.

The callable example uses `execute_callable(record_index, data)`. It expects callable SBT record zero to select `swap_channels`; the callable-data interface has location two on both sides. It does not require an acceleration structure or any hit group. A procedural hit group instead consists of an intersection shader with its optional any-hit and closest-hit shaders. Its hit attribute type must match across the producer and consumers; `report_intersection` submits an intersection to traversal.

The host must enable `VK_KHR_ray_tracing_pipeline` and its dependencies, check device limits and recursion depth, and arrange valid shader groups and SBT addresses. For all image examples, bind a writable image at the indicated descriptor slot and restrict dispatch dimensions to its extent. For compute examples, use an extent divisible by the workgroup size or add a bounds guard before dispatching partial groups. For trace examples, build a scene containing triangle geometry and use matching hit/miss indices.

## Shader record data

`#[shader_record]` exposes data stored in the SBT record, using `ShaderRecordBufferKHR`. It is not a descriptor binding. In `shader_record.vsl`, `RecordCoefficients` has two `vec4` members. The reflected record data layout is 32 bytes with 16-byte alignment, offsets zero and sixteen. Host code writes the coefficients at those reflected offsets within the shader-visible record data. Vulkan shader-group handle size, handle alignment, base alignment, SBT region stride and maximum stride are separate device properties; the compiler's record alignment and size alone do not determine a legal SBT allocation.

For nested records, use the reflected field offsets, array strides, matrix strides and nested layouts rather than assuming Rust or C struct packing. The logical byte size of a payload is not an SBT layout or a device pipeline stack size.

## Shader execution reordering

The NV and EXT examples separate traversal (`HitObject.trace_ray`) from shader invocation (`HitObject.execute_shader`), with `reorder_thread_with_hit_object` between the two. The NV example uses the hit object alone; the EXT example additionally supplies a two-bit coherence hint. The same `pipeline.vsl` hit and miss shaders can be linked by the host for these examples, provided the payload and group layout are compatible.

- `ser_nv` (aliases `ser`, `shader_invocation_reorder`) selects `SPV_NV_shader_invocation_reorder` and requires the corresponding Vulkan NV feature.
- `ser_ext` (alias `ray_tracing_invocation_reorder`) selects `SPV_EXT_shader_invocation_reorder` and requires the corresponding Vulkan EXT feature.

Do not assume extension support from the GPU vendor or from the tier setting. Select a profile after querying the physical device. Reordering is a performance facility, not a promise of faster execution, and should not change the intended shader result.

## KHR opacity micromap and position fetch

`omm_khr` (alias `opacity_micromap_khr`) requests KHR opacity micromap traversal for ray queries. The native compiler emits `SPV_KHR_opacity_micromap`, `RayTracingOpacityMicromapExecutionModeKHR` and `OpExecutionModeId ... OpacityMicromapIdKHR` with a true boolean operand. This is distinct from the legacy `omm` / `opacity_micromap` capability and the force-two-state ray flag. Enabling the KHR execution mode does not itself build or attach opacity micromaps, and does not enable opacity micromap use for ordinary RT pipeline trace calls; the host must supply the relevant Vulkan pipeline flags for those calls.

The example also requests `position_fetch` (alias `ray_tracing_position_fetch`). Its committed query accessor returns three object-space triangle vertex positions as `[vec3; 3]`. It derives a face normal for diagnostic visualization. For actual execution the host must enable the position-fetch feature, build the corresponding acceleration structure with the data-access build flag required by `VK_KHR_ray_tracing_position_fetch`, and use valid, nondegenerate triangle geometry. Enable the KHR micromap feature and attach micromap data to geometry to observe micromap-specific traversal. The sample uses neither the legacy force-two-state flag nor a GPU-specific fallback.

To inspect the emitted capabilities and instructions after validation:

```sh
spirv-dis /tmp/vsl-rt-validation-output/omm_khr/micromap_positions.spv
```

Use the actual output directory reported by the script (or pass `/tmp/vsl-rt-validation-output` to the script). The GLSL oracle maps the KHR ray-query mode to `GL_EXT_opacity_micromap_ray_query_mode` and `gl_EnableOpacityMicromapEXT`; GLSL position fetch uses an output array wrapper.

## Host-side reflection and linking

The public `vsl_mc` API exports `CompiledShaderUnit`, `RayInterfaceKind`, `RayInterfaceLink`, `RayInterfaceType`, `RayInterfaceVariable`, `ShaderRecordLayout` and `validate_ray_interface_link`. Compilation is per entry point. A module containing multiple independent pipelines is legal: entry-point co-location does not imply an interface relationship.

Validate each concrete host-selected edge before creating a pipeline. For the ordinary pipeline, call the function separately for every possible hit or miss target, including targets reachable via dynamic SBT indices:

```rust
use vsl_mc::{
    CompiledShaderUnit, RayInterfaceKind, RayInterfaceLink,
    validate_ray_interface_link,
};

fn validate_triangle_pipeline(
    raygen: &CompiledShaderUnit,
    closest_hit: &CompiledShaderUnit,
    miss: &CompiledShaderUnit,
) -> Result<(), String> {
    let link = RayInterfaceLink::Trace {
        caller_kind: RayInterfaceKind::RayPayload,
        location: 0,
    };
    validate_ray_interface_link(raygen, closest_hit, link)?;
    validate_ray_interface_link(raygen, miss, link)
}

fn validate_callable(
    caller: &CompiledShaderUnit,
    callee: &CompiledShaderUnit,
) -> Result<(), String> {
    validate_ray_interface_link(
        caller,
        callee,
        RayInterfaceLink::Callable {
            caller_kind: RayInterfaceKind::CallableData,
            location: 2,
        },
    )
}
```

For a procedural group, pass the intersection unit as the first argument, each any-hit or closest-hit consumer as the second, and `RayInterfaceLink::ProceduralHit` as the third. For nested trace/callable calls, choose the caller's actual storage kind (`IncomingRayPayload` or `IncomingCallableData` where appropriate). The validator checks stages, storage class, uniqueness, interface location and recursively matching value types, including scalar signedness, vector/matrix dimensions and array lengths. Struct names and field names are descriptive; member order and types matter. Missing or incompatible interfaces produce `Err(String)`. This check does not allocate Vulkan resources, infer SBT groups, validate runtime indices, enforce device limits or replace Vulkan validation layers.

### Reflection JSON

`vslc reflect` serializes an array of `CompiledShaderUnit` objects. The following is an excerpt of the observed `primary` entry in `pipeline.vsl` (other fields omitted):

```json
{
  "entry_point": "primary",
  "stage": "ray_gen",
  "required_caps": "RAY_TRACING_PIPELINE",
  "ray_interfaces": [
    {
      "name": "color",
      "kind": "ray_payload",
      "location": 0,
      "ty": { "kind": "vector", "scalar": "f32", "len": 4 }
    }
  ]
}
```

The full object contains `spirv`, `descriptors`, `push_constants`, `push_constant_size`, `workgroup_size`, `required_caps`, `used_specs`, `used_params`, `debug_channels`, `cache_key` and, when nonempty, `ray_interfaces`. The `required_caps` field is serialized as a bitflag string. Ordinary RT shaders reflect `RAY_TRACING_PIPELINE`; the stage itself also requires the pipeline feature. Combine stage requirements, reflected capabilities and actual SPIR-V requirements when choosing device features.

For `record_color`, the `shader_record` interface has `location: null` and an additional `record_layout` object:

```json
{
  "byte_size": 32,
  "alignment": 16,
  "fields": [
    { "name": "gain", "byte_offset": 0, "byte_size": 16 },
    { "name": "bias", "byte_offset": 16, "byte_size": 16 }
  ]
}
```

Array and matrix stride fields are emitted when relevant; `array_strides` enumerates dimensions from outermost to innermost, and `nested` describes nested struct layout. The host should consume reflection as generated for the selected entry point and compiler build rather than hard-code layouts inferred from these examples.

## References and credits

The examples are original VSL sources. The compiler is a clean-room Rust implementation; the following public specifications define the target contracts. Slang documentation is credited as comparative language and feature documentation only, not as an implementation source. No Slang compiler code or sample shader code was copied.

- Khronos: [SPIR-V 1.6 specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html), [SPV_KHR_ray_tracing](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_tracing.html), [SPV_KHR_ray_query](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_query.html).
- Khronos: [SPV_NV_shader_invocation_reorder](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_shader_invocation_reorder.html), [SPV_EXT_shader_invocation_reorder](https://github.khronos.org/SPIRV-Registry/extensions/EXT/SPV_EXT_shader_invocation_reorder.html).
- Khronos: [SPV_KHR_opacity_micromap](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_opacity_micromap.asciidoc), [SPV_EXT_opacity_micromap](https://github.khronos.org/SPIRV-Registry/extensions/EXT/SPV_EXT_opacity_micromap.html), [SPV_KHR_ray_tracing_position_fetch](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_tracing_position_fetch.html).
- Khronos GLSL: [ray tracing](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_ray_tracing.txt), [ray query](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_ray_query.txt), [position fetch](https://docs.vulkan.org/glslext/latest/glslext/ext/GLSL_EXT_ray_tracing_position_fetch.html), [opacity micromap query mode](https://docs.vulkan.org/glslext/latest/glslext/ext/GLSL_EXT_opacity_micromap_ray_query_mode.html).
- Khronos Vulkan: [VK_KHR_opacity_micromap](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_opacity_micromap.html), [position fetch proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_ray_tracing_position_fetch.html), [ray tracing and reordering overview](https://docs.vulkan.org/samples/latest/samples/extensions/ray_tracing_invocation_reorder/README.html).
- Shader Slang project: [shader execution reordering](https://docs.shader-slang.org/en/stable/external/slang/docs/shader-execution-reordering.html), [parameter blocks and shader records](https://docs.shader-slang.org/en/latest/parameter-blocks.html), [conventional language features and shader stages](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/02-conventional-features.html).
