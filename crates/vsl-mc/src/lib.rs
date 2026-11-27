//! Virtual `mc::*` engine contract and reflection ABI shared between VSL 2.0 and `vulkanshaders-core`.

mod contract;
mod layouts;
mod reflection;

pub use contract::{
    AbiProfile, BINDLESS_IMAGE_COUNT, BUILTIN_RESOURCES, BuiltinResourceId, CANONICAL_PASSES,
    CanonicalPass, DESCRIPTOR_SET_BINDLESS, DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS,
    DESCRIPTOR_SET_SCENE_GLOBALS, HardwareCaps, HookAbiRoute, HookStage, ImageFormat,
    MAX_PUSH_CONSTANT_BYTES, PACK_RESOURCE_SLOTS, RenderTier, SampleScalar, ShaderStage,
    TextureDim,
};
pub use layouts::{
    BuiltinFieldDesc, BuiltinFieldType, BuiltinStructDesc, EMISSIVE_LIGHT_FIELDS,
    EMISSIVE_LIGHT_LAYOUT, EMISSIVE_LIGHT_SSBO_FIELDS, EMISSIVE_LIGHT_SSBO_LAYOUT,
    INSTANCE_META_FIELDS, INSTANCE_META_LAYOUT, INSTANCE_META_SSBO_FIELDS,
    INSTANCE_META_SSBO_LAYOUT, PackParamLayout, PackParamType, RESTIR_RESERVOIR_FIELDS,
    RESTIR_RESERVOIR_LAYOUT, RESTIR_RESERVOIR_SSBO_FIELDS, RESTIR_RESERVOIR_SSBO_LAYOUT,
    SHARC_CONFIG_UBO_FIELDS, SHARC_CONFIG_UBO_LAYOUT, SpecConstantDefault, WORLD_UBO_FIELDS,
    WORLD_UBO_LAYOUT, builtin_descriptor_layout, builtin_struct_layout_by_name,
};
pub use reflection::{
    CompiledShaderUnit, DebugChannelBinding, DescriptorAccess, DescriptorBinding, DescriptorKind,
    PackParamBinding, PushConstantField, SpecConstantUsage,
};
