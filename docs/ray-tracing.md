# Ray tracing compiler and host contract

VSL emits SPIR-V 1.6 directly for Vulkan ray tracing pipelines and inline ray
queries. This document covers the supported compiler interface, not a complete
Vulkan renderer. Applications remain responsible for device feature negotiation,
acceleration structure construction, pipeline groups, shader binding tables,
synchronization and resource lifetime.

## Selecting a profile

Use `--abi standalone` for ordinary Vulkan descriptor layouts. The default
Hyaloclast ABI reserves its engine-specific bindings. Tier 2 includes ray query;
tier 3 includes the ray tracing pipeline and the existing NV SER profile.
Optional features can be enabled independently with `--cap NAME`. A feature
declared in a shader must also be available in the active compilation profile.
Do not treat a compiler profile as a device capability probe.

| VSL feature | SPIR-V contract |
| --- | --- |
| `ray_query` | `SPV_KHR_ray_query` |
| Ray tracing entry points | `SPV_KHR_ray_tracing` |
| `ser_nv` | `SPV_NV_shader_invocation_reorder` |
| `ser_ext` | `SPV_EXT_shader_invocation_reorder` |
| `position_fetch` | `SPV_KHR_ray_tracing_position_fetch` |
| `omm` | Legacy opacity micromap ray flag capability |
| `omm_khr` | `SPV_KHR_opacity_micromap` and `OpacityMicromapIdKHR` execution mode |

The two OMM capabilities have different semantics: the execution mode opts
ray queries into KHR opacity micromap traversal; the legacy capability permits
the force-two-state ray flag. A shader using both needs both capabilities.
The compiler does not construct or attach opacity micromaps.

Use one SER profile per compiled module. The EXT profile is selected when
both profiles are present in its accumulated required capabilities.
Choose the profile supported by the target device, not merely its vendor name.

## Ray interfaces

Ray interface declarations are separate from descriptors. Payload and callable
locations identify logical shader interfaces, not byte offsets into the SBT.

| Attribute | Declaration and access |
| --- | --- |
| `#[ray_payload(N)]` | Outgoing ray payload |
| `#[incoming_ray_payload(N)]` | Incoming payload of a hit or miss shader |
| `#[callable_data(N)]` | Outgoing callable data |
| `#[incoming_callable_data(N)]` | Incoming data of a callable shader |
| `#[hit_attribute]` | Output in intersection shaders; read-only in hit shaders |
| `#[shader_record]` | Read-only, fixed-size struct, stored in the SBT record |

The compiler checks stage legality, location uniqueness, declaration arguments,
fixed-size types, intrinsic argument counts and the storage class of variables
passed to trace or callable operations. Mutable payload shadows are
synchronized across shader invocations, including deferred SER execution.

For procedural intersections, declare the hit attribute as `out`, write the
attribute value, then call `report_intersection`. For a built-in triangle hit,
the usual input is a two-component barycentric attribute.

## Reflection and link validation

`vslc reflect` and the pack manifest expose `ray_interfaces` for each compiled
shader unit. This contains the variable name, storage kind, optional location,
recursive value type, and (for shader records) the explicit record layout.

The layout includes member byte offsets, field sizes, array strides from
outermost to innermost dimension, matrix strides and nested struct layouts.
These offsets are relative to the shader record data following the shader
group handle; they do not include the handle itself. The application must obey
the device's shader group handle size, handle alignment, base alignment and
maximum SBT stride.

The logical size of a payload type is not a guarantee of its physical pipeline
stack allocation. Use Vulkan's pipeline stack size queries where appropriate.

Use the host-side validation API before creating shader groups:

```rust
use vsl_mc::{RayInterfaceKind, RayInterfaceLink, validate_ray_interface_link};

fn validate_primary_ray(
    raygen: &vsl_mc::CompiledShaderUnit,
    closest_hit: &vsl_mc::CompiledShaderUnit,
    miss: &vsl_mc::CompiledShaderUnit,
) -> Result<(), String> {
    let link = RayInterfaceLink::Trace {
        caller_kind: RayInterfaceKind::RayPayload,
        location: 0,
    };
    validate_ray_interface_link(raygen, closest_hit, link)?;
    validate_ray_interface_link(raygen, miss, link)
}
```

Use `RayInterfaceLink::Callable` for each callable edge and
`RayInterfaceLink::ProceduralHit` for each intersection-to-hit edge.
Type comparisons are structural and recursive: equal byte counts alone
are insufficient. Struct names and member names are descriptive.
The API validates edges explicitly selected by the application; it cannot infer
runtime SBT indices or select pipeline groups automatically.

## Validation examples

Run from the repository root:

```sh
bash examples/rt_validation/validate.sh
```

This builds the compiler, checks the seven standalone shader modules, emits
native binaries, validates them with SPIRV-Tools against Vulkan 1.3 and 1.4,
and saves reflection JSON in a temporary output directory. Pass a directory
as the first argument to retain results at a chosen location.

The GLSL oracle supports inline ray query, position fetch and KHR OMM mode for
its supported compute/raster stages. Native SPIR-V is the backend for complete
RT pipeline stages and SER. A successful static validator result does not prove
device feature availability or numerical correctness of traversal.

### Hardware smoke-test coverage

Development smoke tests on an NVIDIA GeForce RTX 3050 Laptop GPU used
independent Vulkan host programs to create acceleration structures, pipelines,
SBTs, dispatch shaders and read back hit/miss results. The native and GLSL
ray-query backends, the ordinary RT pipeline and both SER profiles produced
the expected hit/miss values of 501 and 0. A separate pipeline with nested
std140 shader record data and a callable invocation returned 526 and 9,
matching its reference values.

Position fetch was also executed with a bottom-level acceleration structure
built with `VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_BIT_KHR`.
Both the native and GLSL ray-query backends returned the expected encoded
triangle coordinates (1531 for a hit and 0 for a miss). This regression
exposed a local-array read incorrectly hoisted ahead of its guarded store;
the emitter now treats local loads as order-sensitive.

Those host programs were temporary development fixtures, not part of the
compiler or the example validation script. Actual micromap construction and
traversal and performance benchmarking were not covered by those GPU tests.
OMM shader generation was checked through SPIRV-Tools and the GLSL oracle.

## Scope and references

The implementation is original and follows public specifications. Reference
materials are credited here for semantics and interoperability:

- Khronos, [SPV_KHR_ray_query](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_query.html).
- Khronos, [SPV_KHR_ray_tracing](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_tracing.html).
- Khronos, [SPV_NV_shader_invocation_reorder](https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_shader_invocation_reorder.html).
- Khronos, [SPV_EXT_shader_invocation_reorder](https://github.khronos.org/SPIRV-Registry/extensions/EXT/SPV_EXT_shader_invocation_reorder.html).
- Khronos, [SPV_KHR_opacity_micromap](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_opacity_micromap.html).
- Khronos, [SPV_KHR_ray_tracing_position_fetch](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_tracing_position_fetch.html).
- Khronos, [GLSL ray query extension](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_ray_query.txt).
- Khronos, [GLSL opacity micromap ray query mode](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_opacity_micromap_ray_query_mode.txt).
- Khronos, [Vulkan ray tracing specification](https://docs.vulkan.org/spec/latest/chapters/raytracing.html).
- [Slang documentation](https://shader-slang.org/slang/user-guide/), for comparison of shader language concepts.

No third-party compiler implementation has been copied into VSL.
