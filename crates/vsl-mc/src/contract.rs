use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// Fixed Vulkan descriptor-set indices enforced by `GpuPipelineManager` in `vulkanshaders-core`.
pub const DESCRIPTOR_SET_PER_PASS: u32 = 0;
pub const DESCRIPTOR_SET_SCENE_GLOBALS: u32 = 1;
pub const DESCRIPTOR_SET_BINDLESS: u32 = 2;
pub const DESCRIPTOR_SET_PACK_RESOURCES: u32 = 3;

/// Maximum texture slots available to shader pack assets in Set 3 (`bindings 0..=15`).
pub const PACK_RESOURCE_SLOTS: u32 = 16;

/// Set 2 bindless array length in the engine-compatibility ABI.
///
/// Hyaloclast declares global Set 2 binding 0 as `COMBINED_IMAGE_SAMPLER` with count 4096,
/// so VSL emits a fixed-size array rather than an unsized runtime array.
pub const BINDLESS_IMAGE_COUNT: u32 = 4096;

/// Vulkan 1.3 baseline guaranteed push-constant block size in bytes.
pub const MAX_PUSH_CONSTANT_BYTES: u32 = 128;

/// Hardware and algorithmic rendering tier supported by Hyaloclast (`IGNIS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RenderTier {
    /// Tier 0: Pure raster G-Buffer + CSM + Hi-Z SSR + GTAO + Volumetric Fog.
    Tier0Raster = 0,
    /// Tier 1: Raster G-Buffer + Compute Froxel GI / Screen-Space Radiance Cache.
    Tier1ComputeGi = 1,
    /// Tier 2: Hybrid Raster G-Buffer + inline `VK_KHR_ray_query` + `ReSTIR` DI.
    Tier2RayQuery = 2,
    /// Tier 3: Full Path Tracing + `ReSTIR` DI + `SHaRC` + SVGF/NRD.
    Tier3PathTracing = 3,
}

impl RenderTier {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Tier0Raster),
            1 => Some(Self::Tier1ComputeGi),
            2 => Some(Self::Tier2RayQuery),
            3 => Some(Self::Tier3PathTracing),
            _ => None,
        }
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Target descriptor/ABI profile for VSL 2.0 compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum AbiProfile {
    /// Hyaloclast 4-set engine ABI (`WorldUBO` at 0:0, Set 1 scene globals, Set 2
    /// `COMBINED_IMAGE_SAMPLER[4096]`, Set 3 pack resources).
    #[default]
    Hyaloclast = 0,
    /// Standalone Vulkan 1.3+ shader ABI with explicit `#[binding(set = N, binding = M)]`
    /// support, no implicit `WorldUBO` injection, and standard unsized bindless arrays.
    Standalone = 1,
}

impl AbiProfile {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

bitflags! {
    /// Optional Vulkan device capabilities gated via `#[cap(...)]` in VSL 2.0.
    ///
    /// Note: `device_merger.rs` enables `shader_int64`, `buffer_device_address`,
    /// `descriptor_indexing`, `scalar_block_layout`, `shader_buffer_int64_atomics`,
    /// `synchronization2`, and `dynamic_rendering` by default, but NOT `shader_float16`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
    pub struct HardwareCaps: u32 {
        const FLOAT16                     = 1 << 0;
        const INT64_ATOMICS               = 1 << 1;
        const FLOAT_ATOMICS               = 1 << 2;
        const RAY_QUERY                   = 1 << 3;
        const MAXIMAL_RECONVERGENCE       = 1 << 4;
        const QUAD_CONTROL                = 1 << 5;
        const RAY_TRACING_PIPELINE        = 1 << 6;
        const MESH_SHADER                 = 1 << 7;
        const TESSELLATION                  = 1 << 8;
        const GEOMETRY_SHADER               = 1 << 9;
        const SHADER_INVOCATION_REORDER     = 1 << 10;
        const SHADER_INVOCATION_REORDER_EXT = 1 << 11;
        const OPACITY_MICROMAP              = 1 << 12;
        const RAY_TRACING_POSITION_FETCH    = 1 << 13;
        const COOPERATIVE_MATRIX            = 1 << 14;
        const FRAGMENT_SHADING_RATE         = 1 << 15;
        const FRAGMENT_SHADER_INTERLOCK     = 1 << 16;
        const MULTIVIEW                     = 1 << 17;
        const SAMPLE_RATE_SHADING           = 1 << 18;
        const INPUT_ATTACHMENT              = 1 << 19;
    }
}

impl HardwareCaps {
    pub fn from_cap_name(name: &str) -> Option<Self> {
        match name {
            "float16" => Some(Self::FLOAT16),
            "int64_atomics" => Some(Self::INT64_ATOMICS),
            "float_atomics" => Some(Self::FLOAT_ATOMICS),
            "ray_query" => Some(Self::RAY_QUERY),
            "maximal_reconvergence" => Some(Self::MAXIMAL_RECONVERGENCE),
            "quad_control" => Some(Self::QUAD_CONTROL),
            "ray_tracing_pipeline" | "ray_tracing" => Some(Self::RAY_TRACING_PIPELINE),
            "mesh_shader" => Some(Self::MESH_SHADER),
            "tessellation" => Some(Self::TESSELLATION),
            "geometry_shader" | "geometry" => Some(Self::GEOMETRY_SHADER),
            "shader_invocation_reorder" | "ser" | "ser_nv" => Some(Self::SHADER_INVOCATION_REORDER),
            "ser_ext" | "ray_tracing_invocation_reorder" => {
                Some(Self::SHADER_INVOCATION_REORDER_EXT)
            }
            "opacity_micromap" | "omm" => Some(Self::OPACITY_MICROMAP),
            "ray_tracing_position_fetch" | "position_fetch" => {
                Some(Self::RAY_TRACING_POSITION_FETCH)
            }
            "cooperative_matrix" | "coop_matrix" | "coop_matrix_2" => {
                Some(Self::COOPERATIVE_MATRIX)
            }
            "fragment_shading_rate" | "vrs" => Some(Self::FRAGMENT_SHADING_RATE),
            "fragment_shader_interlock" | "interlock" => Some(Self::FRAGMENT_SHADER_INTERLOCK),
            "multiview" => Some(Self::MULTIVIEW),
            "sample_rate_shading" => Some(Self::SAMPLE_RATE_SHADING),
            "input_attachment" | "dynamic_rendering_local_read" => Some(Self::INPUT_ATTACHMENT),
            _ => None,
        }
    }
}

/// Dimensionality of sampled textures and storage images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureDim {
    Dim2D,
    Dim2DArray,
    Dim3D,
    DimCube,
    Dim1D,
    Dim2DMS,
    DimBuffer,
    SubpassData,
    SubpassDataMS,
}

/// Sampled scalar element type for `Texture*<T>` descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleScalar {
    F32,
    I32,
    U32,
}

/// Shader entry-point execution stage supported by VSL 2.0 (Vulkan 1.3/1.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderStage {
    Compute,
    Vertex,
    Fragment,
    // VK_KHR_ray_tracing_pipeline stages
    RayGen,
    ClosestHit,
    AnyHit,
    Miss,
    Intersection,
    Callable,
    // VK_EXT_mesh_shader stages
    Task,
    Mesh,
    // Legacy geometry pipeline stages
    TessControl,
    TessEval,
    Geometry,
}

impl ShaderStage {
    /// Returns `true` for the six `VK_KHR_ray_tracing_pipeline` stages.
    pub const fn is_ray_tracing(self) -> bool {
        matches!(
            self,
            Self::RayGen
                | Self::ClosestHit
                | Self::AnyHit
                | Self::Miss
                | Self::Intersection
                | Self::Callable
        )
    }

    /// Returns `true` for `VK_EXT_mesh_shader` task and mesh stages.
    pub const fn is_mesh_pipeline(self) -> bool {
        matches!(self, Self::Task | Self::Mesh)
    }

    /// Returns `true` for legacy tessellation and geometry shader stages.
    pub const fn is_legacy_geometry(self) -> bool {
        matches!(self, Self::TessControl | Self::TessEval | Self::Geometry)
    }

    /// Returns `true` for stages that permit `StorageClass::Workgroup` shared variables.
    pub const fn allows_workgroup_vars(self) -> bool {
        matches!(self, Self::Compute | Self::Task | Self::Mesh)
    }

    /// Returns `true` for stages that use `#[location(N)]` stage I/O variables.
    pub const fn uses_stage_io(self) -> bool {
        matches!(
            self,
            Self::Vertex
                | Self::Fragment
                | Self::Mesh
                | Self::TessControl
                | Self::TessEval
                | Self::Geometry
        )
    }
}

/// Storage image formats supported by `RWImage2D<Fmt>`, `RWImage2DArray<Fmt>`, and `RWImage3D<Fmt>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Rgba16f,
    Rg16f,
    R16f,
    R32f,
    Rgba8,
    Rg8,
    R8,
    R32ui,
    R32i,
    Rgba32f,
}

impl ImageFormat {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "rgba16f" => Some(Self::Rgba16f),
            "rg16f" => Some(Self::Rg16f),
            "r16f" => Some(Self::R16f),
            "r32f" => Some(Self::R32f),
            "rgba8" => Some(Self::Rgba8),
            "rg8" => Some(Self::Rg8),
            "r8" => Some(Self::R8),
            "r32ui" => Some(Self::R32ui),
            "r32i" => Some(Self::R32i),
            "rgba32f" => Some(Self::Rgba32f),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rgba16f => "rgba16f",
            Self::Rg16f => "rg16f",
            Self::R16f => "r16f",
            Self::R32f => "r32f",
            Self::Rgba8 => "rgba8",
            Self::Rg8 => "rg8",
            Self::R8 => "r8",
            Self::R32ui => "r32ui",
            Self::R32i => "r32i",
            Self::Rgba32f => "rgba32f",
        }
    }
}

/// Compilation lowering route for a `@hook` stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookAbiRoute {
    /// Emitted into Hyaloclast's core raster G-Buffer GLSL hook slot (routes `param` reads to
    /// `VsMotion.packParams` so VSL 2.0's native SPIR-V backend never inherits legacy raster ABI).
    RasterGbufferGlslBridge,
    /// Lowered directly through VSL 2.0's native SSA IR -> SPIR-V 1.6 pipeline.
    NativeSpirv,
}

/// All 16 composable `@hook` stages in Hyaloclast VSP Graph v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookStage {
    VertexDisplace,
    GbufferPack,
    Surface,
    EmissiveSample,
    RestirReservoir,
    DirectLight,
    BlockLight,
    IndirectGi,
    WaterWaves,
    WaterSurface,
    Sky,
    Clouds,
    VolumetricMedia,
    Fog,
    AoFilter,
    PostGrade,
}

impl HookStage {
    pub const ALL: &'static [Self] = &[
        Self::VertexDisplace,
        Self::GbufferPack,
        Self::Surface,
        Self::EmissiveSample,
        Self::RestirReservoir,
        Self::DirectLight,
        Self::BlockLight,
        Self::IndirectGi,
        Self::WaterWaves,
        Self::WaterSurface,
        Self::Sky,
        Self::Clouds,
        Self::VolumetricMedia,
        Self::Fog,
        Self::AoFilter,
        Self::PostGrade,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim() {
            "vertex_displace" => Some(Self::VertexDisplace),
            "gbuffer_pack" => Some(Self::GbufferPack),
            "surface" => Some(Self::Surface),
            "emissive_sample" => Some(Self::EmissiveSample),
            "restir_reservoir" => Some(Self::RestirReservoir),
            "direct_light" => Some(Self::DirectLight),
            "block_light" => Some(Self::BlockLight),
            "indirect_gi" => Some(Self::IndirectGi),
            "water_waves" => Some(Self::WaterWaves),
            "water_surface" => Some(Self::WaterSurface),
            "sky" => Some(Self::Sky),
            "clouds" => Some(Self::Clouds),
            "volumetric_media" => Some(Self::VolumetricMedia),
            "fog" => Some(Self::Fog),
            "ao_filter" => Some(Self::AoFilter),
            "post_grade" => Some(Self::PostGrade),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VertexDisplace => "vertex_displace",
            Self::GbufferPack => "gbuffer_pack",
            Self::Surface => "surface",
            Self::EmissiveSample => "emissive_sample",
            Self::RestirReservoir => "restir_reservoir",
            Self::DirectLight => "direct_light",
            Self::BlockLight => "block_light",
            Self::IndirectGi => "indirect_gi",
            Self::WaterWaves => "water_waves",
            Self::WaterSurface => "water_surface",
            Self::Sky => "sky",
            Self::Clouds => "clouds",
            Self::VolumetricMedia => "volumetric_media",
            Self::Fog => "fog",
            Self::AoFilter => "ao_filter",
            Self::PostGrade => "post_grade",
        }
    }

    /// Returns the lowering ABI route for this hook stage.
    pub const fn abi_route(self) -> HookAbiRoute {
        match self {
            Self::VertexDisplace | Self::Surface | Self::GbufferPack => {
                HookAbiRoute::RasterGbufferGlslBridge
            }
            _ => HookAbiRoute::NativeSpirv,
        }
    }

    /// Returns `true` for `clouds`, which returns `vec4` and is alpha-over composited across chains.
    pub const fn uses_alpha_over_blending(self) -> bool {
        matches!(self, Self::Clouds)
    }
}

/// The 14 canonical passes registered in Hyaloclast's `RenderGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalPass {
    RtBlasTlasUpdate,
    ShadowCsmPass,
    GbufferMrtPass,
    VolumetricFogPass,
    SsgiScreenSpaceGi,
    LightingRasterComputeGi,
    LightingHybridRayquery,
    LightingFullPathTracing,
    Tier3McDepthReproject,
    DenoiserSvgfNrd,
    UpscalerTsrVendor,
    PostGradeAndTonemap,
    HdrScrgbCompose,
    FramegenOpticalFlowPresent,
}

/// Complete list of all 14 canonical Hyaloclast render-graph passes.
pub const CANONICAL_PASSES: &[CanonicalPass] = &[
    CanonicalPass::RtBlasTlasUpdate,
    CanonicalPass::ShadowCsmPass,
    CanonicalPass::GbufferMrtPass,
    CanonicalPass::VolumetricFogPass,
    CanonicalPass::SsgiScreenSpaceGi,
    CanonicalPass::LightingRasterComputeGi,
    CanonicalPass::LightingHybridRayquery,
    CanonicalPass::LightingFullPathTracing,
    CanonicalPass::Tier3McDepthReproject,
    CanonicalPass::DenoiserSvgfNrd,
    CanonicalPass::UpscalerTsrVendor,
    CanonicalPass::PostGradeAndTonemap,
    CanonicalPass::HdrScrgbCompose,
    CanonicalPass::FramegenOpticalFlowPresent,
];

impl CanonicalPass {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "rt_blas_tlas_update" => Some(Self::RtBlasTlasUpdate),
            "shadow_csm_pass" => Some(Self::ShadowCsmPass),
            "gbuffer_mrt_pass" => Some(Self::GbufferMrtPass),
            "volumetric_fog_pass" => Some(Self::VolumetricFogPass),
            "ssgi_screen_space_gi" => Some(Self::SsgiScreenSpaceGi),
            "lighting_raster_compute_gi" => Some(Self::LightingRasterComputeGi),
            "lighting_hybrid_rayquery" => Some(Self::LightingHybridRayquery),
            "lighting_full_path_tracing" => Some(Self::LightingFullPathTracing),
            "tier3_mc_depth_reproject" => Some(Self::Tier3McDepthReproject),
            "denoiser_svgf_nrd" => Some(Self::DenoiserSvgfNrd),
            "upscaler_tsr_vendor" => Some(Self::UpscalerTsrVendor),
            "post_grade_and_tonemap" => Some(Self::PostGradeAndTonemap),
            "hdr_scrgb_compose" => Some(Self::HdrScrgbCompose),
            "framegen_optical_flow_present" => Some(Self::FramegenOpticalFlowPresent),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RtBlasTlasUpdate => "rt_blas_tlas_update",
            Self::ShadowCsmPass => "shadow_csm_pass",
            Self::GbufferMrtPass => "gbuffer_mrt_pass",
            Self::VolumetricFogPass => "volumetric_fog_pass",
            Self::SsgiScreenSpaceGi => "ssgi_screen_space_gi",
            Self::LightingRasterComputeGi => "lighting_raster_compute_gi",
            Self::LightingHybridRayquery => "lighting_hybrid_rayquery",
            Self::LightingFullPathTracing => "lighting_full_path_tracing",
            Self::Tier3McDepthReproject => "tier3_mc_depth_reproject",
            Self::DenoiserSvgfNrd => "denoiser_svgf_nrd",
            Self::UpscalerTsrVendor => "upscaler_tsr_vendor",
            Self::PostGradeAndTonemap => "post_grade_and_tonemap",
            Self::HdrScrgbCompose => "hdr_scrgb_compose",
            Self::FramegenOpticalFlowPresent => "framegen_optical_flow_present",
        }
    }
}

/// Built-in `RenderGraph` resource IDs (`1..=41`) matching `vulkanshaders-core::graph::RenderGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum BuiltinResourceId {
    ResTlas = 1,
    ResShadowCsm = 2,
    ResGbufferAlbedo = 3,
    ResGbufferNormal = 4,
    ResGbufferMaterial = 5,
    ResGbufferVelocity = 6,
    ResLinearDepth = 7,
    ResRawRadiance = 8,
    ResDenoisedRadiance = 9,
    ResUpscaledHdr = 10,
    ResHudlessColor = 11,
    ResMcMainDepth = 12,
    ResSwapchainOut = 13,
    ResPrevHudlessColor = 14,
    ResUiColor = 15,
    ResSharcHashTable = 16,
    ResInterpolatedFrame = 17,
    ResSvgfPing = 18,
    ResPostOutput = 19,
    ResSsgiHistory = 20,
    ResSsgiOut = 21,
    ResSvgfHistoryRadiance = 22,
    ResSvgfHistoryMoments = 23,
    ResSvgfMoments = 24,
    ResTsrHistory = 25,
    ResVolumetricFog = 26,
    ResEmissiveLights = 27,
    ResRestirReservoirs = 28,
    ResPtAlbedo = 29,
    ResVolumetricFogHistory = 30,
    ResSharpRadiance = 31,
    ResNrdInDiff = 32,
    ResNrdInSpec = 33,
    ResNrdOutSpec = 34,
    ResNrdNormalRough = 35,
    ResNrdViewz = 36,
    ResRtHitDist = 37,
    ResDhAlbedo = 38,
    ResDhNormal = 39,
    ResDhMaterial = 40,
    ResDhDepth = 41,
}

impl BuiltinResourceId {
    /// First ID reserved for pack-declared custom targets and buffers.
    pub const CUSTOM_BASE: u32 = 1000;

    /// Resolves a `mc::targets::RES_*` identifier (including `RES_DEPTH` alias for `RES_MC_MAIN_DEPTH`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "RES_TLAS" => Some(Self::ResTlas),
            "RES_SHADOW_CSM" => Some(Self::ResShadowCsm),
            "RES_GBUFFER_ALBEDO" => Some(Self::ResGbufferAlbedo),
            "RES_GBUFFER_NORMAL" => Some(Self::ResGbufferNormal),
            "RES_GBUFFER_MATERIAL" => Some(Self::ResGbufferMaterial),
            "RES_GBUFFER_VELOCITY" => Some(Self::ResGbufferVelocity),
            "RES_LINEAR_DEPTH" => Some(Self::ResLinearDepth),
            "RES_RAW_RADIANCE" => Some(Self::ResRawRadiance),
            "RES_DENOISED_RADIANCE" => Some(Self::ResDenoisedRadiance),
            "RES_UPSCALED_HDR" => Some(Self::ResUpscaledHdr),
            "RES_HUDLESS_COLOR" => Some(Self::ResHudlessColor),
            "RES_MC_MAIN_DEPTH" | "RES_DEPTH" => Some(Self::ResMcMainDepth),
            "RES_SWAPCHAIN_OUT" => Some(Self::ResSwapchainOut),
            "RES_PREV_HUDLESS_COLOR" => Some(Self::ResPrevHudlessColor),
            "RES_UI_COLOR" => Some(Self::ResUiColor),
            "RES_SHARC_HASH_TABLE" => Some(Self::ResSharcHashTable),
            "RES_INTERPOLATED_FRAME" => Some(Self::ResInterpolatedFrame),
            "RES_SVGF_PING" => Some(Self::ResSvgfPing),
            "RES_POST_OUTPUT" => Some(Self::ResPostOutput),
            "RES_SSGI_HISTORY" => Some(Self::ResSsgiHistory),
            "RES_SSGI_OUT" => Some(Self::ResSsgiOut),
            "RES_SVGF_HISTORY_RADIANCE" => Some(Self::ResSvgfHistoryRadiance),
            "RES_SVGF_HISTORY_MOMENTS" => Some(Self::ResSvgfHistoryMoments),
            "RES_SVGF_MOMENTS" => Some(Self::ResSvgfMoments),
            "RES_TSR_HISTORY" => Some(Self::ResTsrHistory),
            "RES_VOLUMETRIC_FOG" => Some(Self::ResVolumetricFog),
            "RES_EMISSIVE_LIGHTS" => Some(Self::ResEmissiveLights),
            "RES_RESTIR_RESERVOIRS" => Some(Self::ResRestirReservoirs),
            "RES_PT_ALBEDO" => Some(Self::ResPtAlbedo),
            "RES_VOLUMETRIC_FOG_HISTORY" => Some(Self::ResVolumetricFogHistory),
            "RES_SHARP_RADIANCE" => Some(Self::ResSharpRadiance),
            "RES_NRD_IN_DIFF" => Some(Self::ResNrdInDiff),
            "RES_NRD_IN_SPEC" => Some(Self::ResNrdInSpec),
            "RES_NRD_OUT_SPEC" => Some(Self::ResNrdOutSpec),
            "RES_NRD_NORMAL_ROUGH" => Some(Self::ResNrdNormalRough),
            "RES_NRD_VIEWZ" => Some(Self::ResNrdViewz),
            "RES_RT_HIT_DIST" => Some(Self::ResRtHitDist),
            "RES_DH_ALBEDO" => Some(Self::ResDhAlbedo),
            "RES_DH_NORMAL" => Some(Self::ResDhNormal),
            "RES_DH_MATERIAL" => Some(Self::ResDhMaterial),
            "RES_DH_DEPTH" => Some(Self::ResDhDepth),
            _ => None,
        }
    }

    pub const fn raw_id(self) -> u32 {
        self as u32
    }
}

/// Complete canonical list of Hyaloclast `mc::targets::RES_*` identifiers (including `RES_DEPTH` alias).
pub const BUILTIN_RESOURCES: &[(&str, BuiltinResourceId)] = &[
    ("RES_TLAS", BuiltinResourceId::ResTlas),
    ("RES_SHADOW_CSM", BuiltinResourceId::ResShadowCsm),
    ("RES_GBUFFER_ALBEDO", BuiltinResourceId::ResGbufferAlbedo),
    ("RES_GBUFFER_NORMAL", BuiltinResourceId::ResGbufferNormal),
    (
        "RES_GBUFFER_MATERIAL",
        BuiltinResourceId::ResGbufferMaterial,
    ),
    (
        "RES_GBUFFER_VELOCITY",
        BuiltinResourceId::ResGbufferVelocity,
    ),
    ("RES_LINEAR_DEPTH", BuiltinResourceId::ResLinearDepth),
    ("RES_RAW_RADIANCE", BuiltinResourceId::ResRawRadiance),
    (
        "RES_DENOISED_RADIANCE",
        BuiltinResourceId::ResDenoisedRadiance,
    ),
    ("RES_UPSCALED_HDR", BuiltinResourceId::ResUpscaledHdr),
    ("RES_HUDLESS_COLOR", BuiltinResourceId::ResHudlessColor),
    ("RES_MC_MAIN_DEPTH", BuiltinResourceId::ResMcMainDepth),
    ("RES_DEPTH", BuiltinResourceId::ResMcMainDepth),
    ("RES_SWAPCHAIN_OUT", BuiltinResourceId::ResSwapchainOut),
    (
        "RES_PREV_HUDLESS_COLOR",
        BuiltinResourceId::ResPrevHudlessColor,
    ),
    ("RES_UI_COLOR", BuiltinResourceId::ResUiColor),
    ("RES_SHARC_HASH_TABLE", BuiltinResourceId::ResSharcHashTable),
    (
        "RES_INTERPOLATED_FRAME",
        BuiltinResourceId::ResInterpolatedFrame,
    ),
    ("RES_SVGF_PING", BuiltinResourceId::ResSvgfPing),
    ("RES_POST_OUTPUT", BuiltinResourceId::ResPostOutput),
    ("RES_SSGI_HISTORY", BuiltinResourceId::ResSsgiHistory),
    ("RES_SSGI_OUT", BuiltinResourceId::ResSsgiOut),
    (
        "RES_SVGF_HISTORY_RADIANCE",
        BuiltinResourceId::ResSvgfHistoryRadiance,
    ),
    (
        "RES_SVGF_HISTORY_MOMENTS",
        BuiltinResourceId::ResSvgfHistoryMoments,
    ),
    ("RES_SVGF_MOMENTS", BuiltinResourceId::ResSvgfMoments),
    ("RES_TSR_HISTORY", BuiltinResourceId::ResTsrHistory),
    ("RES_VOLUMETRIC_FOG", BuiltinResourceId::ResVolumetricFog),
    ("RES_EMISSIVE_LIGHTS", BuiltinResourceId::ResEmissiveLights),
    (
        "RES_RESTIR_RESERVOIRS",
        BuiltinResourceId::ResRestirReservoirs,
    ),
    ("RES_PT_ALBEDO", BuiltinResourceId::ResPtAlbedo),
    (
        "RES_VOLUMETRIC_FOG_HISTORY",
        BuiltinResourceId::ResVolumetricFogHistory,
    ),
    ("RES_SHARP_RADIANCE", BuiltinResourceId::ResSharpRadiance),
    ("RES_NRD_IN_DIFF", BuiltinResourceId::ResNrdInDiff),
    ("RES_NRD_IN_SPEC", BuiltinResourceId::ResNrdInSpec),
    ("RES_NRD_OUT_SPEC", BuiltinResourceId::ResNrdOutSpec),
    ("RES_NRD_NORMAL_ROUGH", BuiltinResourceId::ResNrdNormalRough),
    ("RES_NRD_VIEWZ", BuiltinResourceId::ResNrdViewz),
    ("RES_RT_HIT_DIST", BuiltinResourceId::ResRtHitDist),
    ("RES_DH_ALBEDO", BuiltinResourceId::ResDhAlbedo),
    ("RES_DH_NORMAL", BuiltinResourceId::ResDhNormal),
    ("RES_DH_MATERIAL", BuiltinResourceId::ResDhMaterial),
    ("RES_DH_DEPTH", BuiltinResourceId::ResDhDepth),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_counts_match_hyaloclast_spec() {
        assert_eq!(HookStage::ALL.len(), 16);
        assert_eq!(CANONICAL_PASSES.len(), 14);
        assert_eq!(BuiltinResourceId::ResDhDepth.raw_id(), 41);
        assert!(HookStage::Clouds.uses_alpha_over_blending());
        assert_eq!(
            HookStage::VertexDisplace.abi_route(),
            HookAbiRoute::RasterGbufferGlslBridge
        );
        assert_eq!(HookStage::Clouds.abi_route(), HookAbiRoute::NativeSpirv);
    }
}
