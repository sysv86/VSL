//! Virtual `mc::*` engine contract and reflection ABI shared between VSL 2.0 and `vulkanshaders-core`.

mod contract;
mod reflection;

pub use contract::{
    BINDLESS_IMAGE_COUNT, BUILTIN_RESOURCES, BuiltinResourceId, CANONICAL_PASSES, CanonicalPass,
    DESCRIPTOR_SET_BINDLESS, DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS,
    DESCRIPTOR_SET_SCENE_GLOBALS, HardwareCaps, HookAbiRoute, HookStage, ImageFormat,
    MAX_PUSH_CONSTANT_BYTES, PACK_RESOURCE_SLOTS, RenderTier, SampleScalar, ShaderStage,
    TextureDim,
};
pub use reflection::{
    CompiledShaderUnit, DebugChannelBinding, DescriptorAccess, DescriptorBinding, DescriptorKind,
    PackParamBinding, PushConstantField, SpecConstantUsage,
};
