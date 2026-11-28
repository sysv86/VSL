use crate::{HardwareCaps, ImageFormat, SampleScalar, ShaderStage, TextureDim};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// Vulkan descriptor class emitted by VSL 2.0 native reflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptorKind {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
    CombinedImageSampler,
    Sampler,
    AccelerationStructure,
    InputAttachment,
}

/// Memory access qualifiers for a reflected descriptor binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptorAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Single-source-of-truth Vulkan descriptor binding emitted alongside SPIR-V.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorBinding {
    pub name: SmolStr,
    pub set: u32,
    pub binding: u32,
    pub kind: DescriptorKind,
    #[serde(default)]
    pub dim: Option<TextureDim>,
    #[serde(default)]
    pub sample_scalar: Option<SampleScalar>,
    pub image_format: Option<ImageFormat>,
    pub access: DescriptorAccess,
    #[serde(rename = "unsized")]
    pub is_unsized: bool,
    /// Fixed descriptor array length; `Some(BINDLESS_IMAGE_COUNT)` for the Set 2 engine ABI.
    #[serde(rename = "count", default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
}

/// Reflected push-constant field with exact scalar byte offset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushConstantField {
    pub name: SmolStr,
    pub byte_offset: u32,
    pub byte_size: u32,
}

/// Reflected pack parameter byte offset inside the 64-bit BDA `PackParamBuffer`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackParamBinding {
    pub name: SmolStr,
    pub byte_offset: u32,
    pub byte_size: u32,
}

/// Specialization constant referenced by an entry point (drives selective pipeline rebuilds).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecConstantUsage {
    pub name: SmolStr,
    pub spec_id: u32,
}

/// Reflected `mc::debug_write(...)` channel binding (compute-only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugChannelBinding {
    pub channel_name: SmolStr,
    pub set: u32,
    pub binding: u32,
}

/// Complete compiled shader unit handed off from VSL 2.0 to `GpuPipelineManager`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledShaderUnit {
    pub entry_point: SmolStr,
    pub stage: ShaderStage,
    pub spirv: Vec<u32>,
    pub descriptors: Vec<DescriptorBinding>,
    pub push_constants: Vec<PushConstantField>,
    pub push_constant_size: u32,
    pub workgroup_size: Option<[u32; 3]>,
    pub required_caps: HardwareCaps,
    pub used_specs: Vec<SpecConstantUsage>,
    pub used_params: Vec<PackParamBinding>,
    pub debug_channels: Vec<DebugChannelBinding>,
    /// BLAKE3 L2 artifact hash covering `(entry_point_hir_hash, tier, caps, used_specs, driver_uuid)`.
    pub cache_key: [u8; 32],
}
