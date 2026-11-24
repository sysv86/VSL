use la_arena::{Arena, Idx};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use vsl_hir::TypeId;
use vsl_mc::{
    DescriptorAccess, DescriptorKind, HardwareCaps, ImageFormat, SampleScalar, ShaderStage,
    TextureDim,
};

pub type InstId = Idx<Inst>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomicScope {
    Device,
    Workgroup,
    Subgroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtomicOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
}

/// Built-in shader stage input variable kind lowered to SPIR-V `OpDecorate BuiltIn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinKind {
    GlobalInvocationId,
    LocalInvocationId,
    LocalInvocationIndex,
    WorkgroupId,
    NumWorkgroups,
    VertexIndex,
    InstanceIndex,
    FragCoord,
}

/// Memory qualifiers lowered onto SPIR-V BDA pointer loads/stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryAccess {
    pub align: u32,
    pub non_writable: bool,
    pub restrict: bool,
}

/// Descriptor variable declaration attached to an [`IrFunction`].
///
/// Mirrors the single-source-of-truth Vulkan descriptor-set ABI so SPIR-V emission
/// and native reflection emit identical `(set, binding)` decorations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrDescriptorVar {
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
    #[serde(rename = "count", default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
}

/// Shared workgroup (`StorageClass::Workgroup`) variable declaration attached to an [`IrFunction`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrWorkgroupVar {
    pub name: SmolStr,
    pub element_is_u32: bool,
    pub array_len: Option<u32>,
}

/// Extended math operation lowered to `GLSL.std.450` (`OpExtInst`) or core `OpDot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlslStd450Op {
    FAbs,
    SAbs,
    Floor,
    Ceil,
    Fract,
    Sin,
    Cos,
    Pow,
    Exp,
    Log,
    Sqrt,
    FMin,
    UMin,
    SMin,
    FMax,
    UMax,
    SMax,
    FClamp,
    UClamp,
    SClamp,
    FMix,
    Fma,
    Dot,
    Length,
    Cross,
    Normalize,
    Reflect,
}

/// Scalar element type for typed IR blocks (push constants, BDA, and specialization constants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarIrTy {
    Bool,
    I32,
    U32,
    F32,
    F16,
    I64,
    U64,
}

/// Typed field descriptor inside an [`IrPushConstantBlock`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IrPushConstantField {
    pub name: SmolStr,
    pub byte_offset: u32,
    pub byte_size: u32,
    pub scalar: ScalarIrTy,
}

/// Push-constant block layout attached to an [`IrFunction`].
///
/// Each entry in `fields` records the field name, byte offset, byte size, and scalar type
/// computed with scalar block layout rules (`total_size <= 128` bytes per Vulkan 1.3 baseline).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrPushConstantBlock {
    pub name: SmolStr,
    pub fields: Vec<IrPushConstantField>,
    pub total_size: u32,
}

/// Structured SSA instruction payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstData {
    ConstBool(bool),
    ConstI32(i32),
    ConstU32(u32),
    /// IEEE-754 single-precision float constant stored as `f32::to_bits()` for deterministic `Eq`.
    ConstF32(u32),
    ConstI64(i64),
    ConstU64(u64),
    IAdd32(InstId, InstId),
    ISub32(InstId, InstId),
    IMul32(InstId, InstId),
    IDiv32(InstId, InstId),
    IRem32(InstId, InstId),
    FAdd32(InstId, InstId),
    FSub32(InstId, InstId),
    FMul32(InstId, InstId),
    FDiv32(InstId, InstId),
    ICmpEq(InstId, InstId),
    ICmpNe(InstId, InstId),
    ICmpLt(InstId, InstId),
    ICmpLe(InstId, InstId),
    ICmpGt(InstId, InstId),
    ICmpGe(InstId, InstId),
    UCmpLt(InstId, InstId),
    FCmpEq(InstId, InstId),
    FCmpNe(InstId, InstId),
    FCmpLt(InstId, InstId),
    FCmpLe(InstId, InstId),
    FCmpGt(InstId, InstId),
    FCmpGe(InstId, InstId),
    LogicalAnd(InstId, InstId),
    LogicalOr(InstId, InstId),
    LogicalNot(InstId),
    Select {
        cond: InstId,
        on_true: InstId,
        on_false: InstId,
    },
    BitAnd32(InstId, InstId),
    BitOr32(InstId, InstId),
    BitXor32(InstId, InstId),
    BitNot32(InstId),
    Shl32(InstId, InstId),
    Shr32(InstId, InstId),
    /// Bit-preserving conversion from signed 64-bit integer to unsigned 64-bit integer.
    CastI64ToU64(InstId),
    /// Explicit conversion from `u64`/`i64`/`i32`/`f32`/`bool` to `u32`.
    CastToU32(InstId),
    /// Explicit conversion to `i32`.
    CastToI32(InstId),
    /// Explicit conversion to `f32`.
    CastToF32(InstId),
    /// Explicit conversion to `i64`.
    CastToI64(InstId),
    /// Explicit conversion to `u64`.
    CastToU64(InstId),
    /// Specialization constant (`OpSpecConstant` / `OpSpecConstantTrue`/`False`) with `SpecId`.
    SpecConstant {
        spec_id: u32,
        scalar: ScalarIrTy,
        default_bits: u32,
    },
    /// Extended math operation (`GLSL.std.450` `OpExtInst` or core `OpDot`).
    GlslStd450 {
        op: GlslStd450Op,
        args: Vec<InstId>,
    },
    /// SSA merge node over incoming `(value, predecessor_tag)` pairs from structured control-flow paths.
    Phi {
        incoming: Vec<(InstId, u32)>,
    },
    /// High-level typed BDA offset (`p.offset(i)`), legalized by [`crate::LowerBdaOffsetsPass`]
    /// into explicit 64-bit integer arithmetic to avoid NVIDIA's `OpPtrAccessChain` +4 GB bug.
    BdaOffset {
        base_ptr: InstId,
        index_i64: InstId,
        stride_bytes: u64,
        align: u32,
    },
    ConvertPtrToU(InstId),
    ConvertUToPtr {
        addr_u64: InstId,
        align: u32,
    },
    IMul64(InstId, InstId),
    IAdd64(InstId, InstId),
    BdaLoad {
        ptr: InstId,
        access: MemoryAccess,
    },
    BdaStore {
        ptr: InstId,
        value: InstId,
        access: MemoryAccess,
    },
    /// Samples or fetches an `f32` scalar from an image descriptor declared at `func.descriptors[descriptor_index]`.
    /// When `is_fetch` is `true`, integer texel fetch (`OpImageFetch`) is emitted; otherwise
    /// normalized UV/cube sampling (`OpImageSampleExplicitLod`) is emitted.
    ImageSample {
        descriptor_index: u32,
        image_inst: Option<InstId>,
        coord: InstId,
        lod: InstId,
        is_fetch: bool,
    },
    /// Writes a texel value to a storage image descriptor declared at `func.descriptors[descriptor_index]`.
    ImageStore {
        descriptor_index: u32,
        coord: InstId,
        value: InstId,
    },
    /// Produces a `StorageClass::Workgroup` pointer to a scalar or indexed element of `func.workgroup_vars[var_index]`.
    WorkgroupAddr {
        var_index: u32,
        index: Option<InstId>,
    },
    /// Loads a scalar or indexed element from `func.workgroup_vars[var_index]`.
    WorkgroupLoad {
        var_index: u32,
        index: Option<InstId>,
    },
    /// Stores a value to a scalar or indexed element of `func.workgroup_vars[var_index]`.
    WorkgroupStore {
        var_index: u32,
        index: Option<InstId>,
        value: InstId,
    },
    /// Marks a dynamically non-uniform descriptor index (`nonuniform(idx)`).
    NonUniform(InstId),
    /// Indexes into an unsized or arrayed descriptor declared at `func.descriptors[descriptor_index]`.
    ResourceAccessChain {
        descriptor_index: u32,
        index: InstId,
    },
    ResourceLoad {
        ptr: InstId,
    },
    AtomicAdd {
        ptr: InstId,
        value: InstId,
        scope: AtomicScope,
        order: AtomicOrder,
    },
    AtomicMin {
        ptr: InstId,
        value: InstId,
        scope: AtomicScope,
        order: AtomicOrder,
    },
    AtomicMax {
        ptr: InstId,
        value: InstId,
        scope: AtomicScope,
        order: AtomicOrder,
    },
    AtomicCas {
        ptr: InstId,
        expected: InstId,
        desired: InstId,
        scope: AtomicScope,
        order: AtomicOrder,
    },
    WorkgroupBarrier,
    SubgroupBallot(InstId),
    SubgroupBallotBitCount(InstId),
    SubgroupBroadcastFirst(InstId),
    /// Loads a scalar field at 32-bit `word_index` from a UBO or SSBO descriptor.
    DescriptorLoad {
        descriptor_index: u32,
        word_index: u32,
        scalar: ScalarIrTy,
    },
    /// Loads a SPIR-V `BuiltIn` input variable (optionally extracting `component` from a vector built-in).
    LoadBuiltin {
        kind: BuiltinKind,
        component: Option<u32>,
    },
    /// Loads a `u32` scalar from member `field_index` of the function's `PushConstant` block.
    LoadPushConstant {
        field_index: u32,
    },
    /// Initializes a `SPV_KHR_ray_query` traversal against `func.descriptors[tlas_descriptor_index]`.
    RayQueryInitialize {
        ray_query_id: u32,
        tlas_descriptor_index: u32,
        flags: InstId,
        cull_mask: InstId,
        origin: InstId,
        t_min: InstId,
        dir: InstId,
        t_max: InstId,
    },
    /// Advances the function-local ray query traversal (`OpRayQueryProceedKHR`), returning `bool`.
    RayQueryProceed(u32),
    /// Queries the committed intersection type (`OpRayQueryGetIntersectionTypeKHR`), returning `u32`.
    RayQueryCommittedType(u32),
    /// Queries the committed instance custom index (`OpRayQueryGetIntersectionInstanceCustomIndexKHR`), returning `u32`.
    RayQueryCommittedInstanceCustomIndex(u32),
    /// Queries the committed triangle barycentrics (`OpRayQueryGetIntersectionBarycentricsKHR`), returning component 0 as `f32`.
    RayQueryCommittedBarycentrics(u32),
    /// Constructs a 2-component vector `(x, y)` (e.g., 2D image coordinates).
    MakeVec2(InstId, InstId),
    /// Constructs a 3-component vector `(x, y, z)`.
    MakeVec3(InstId, InstId, InstId),
    /// Constructs a 4-component vector `(x, y, z, w)`.
    MakeVec4(InstId, InstId, InstId, InstId),
    /// Extracts scalar component `index` (`0..=3`) from a vector value.
    ExtractComponent {
        vec: InstId,
        index: u32,
    },
    Nop,
}

impl InstData {
    /// Constructs a [`Self::ConstF32`] from an `f32` value using its IEEE-754 bit pattern.
    pub const fn const_f32(value: f32) -> Self {
        Self::ConstF32(value.to_bits())
    }

    /// Returns all direct SSA operand instruction IDs referenced by this instruction.
    pub fn operands(&self) -> Vec<InstId> {
        match self {
            Self::BdaOffset {
                base_ptr,
                index_i64,
                ..
            } => vec![*base_ptr, *index_i64],
            Self::ConvertPtrToU(v)
            | Self::ConvertUToPtr { addr_u64: v, .. }
            | Self::BdaLoad { ptr: v, .. }
            | Self::NonUniform(v)
            | Self::ResourceAccessChain { index: v, .. }
            | Self::ResourceLoad { ptr: v }
            | Self::SubgroupBallot(v)
            | Self::SubgroupBallotBitCount(v)
            | Self::SubgroupBroadcastFirst(v)
            | Self::LogicalNot(v)
            | Self::BitNot32(v)
            | Self::CastI64ToU64(v)
            | Self::CastToU32(v)
            | Self::CastToI32(v)
            | Self::CastToF32(v)
            | Self::CastToI64(v)
            | Self::CastToU64(v)
            | Self::ExtractComponent { vec: v, .. } => vec![*v],
            Self::IAdd32(a, b)
            | Self::ISub32(a, b)
            | Self::IMul32(a, b)
            | Self::IDiv32(a, b)
            | Self::IRem32(a, b)
            | Self::FAdd32(a, b)
            | Self::FSub32(a, b)
            | Self::FMul32(a, b)
            | Self::FDiv32(a, b)
            | Self::ICmpEq(a, b)
            | Self::ICmpNe(a, b)
            | Self::ICmpLt(a, b)
            | Self::ICmpLe(a, b)
            | Self::ICmpGt(a, b)
            | Self::ICmpGe(a, b)
            | Self::UCmpLt(a, b)
            | Self::FCmpEq(a, b)
            | Self::FCmpNe(a, b)
            | Self::FCmpLt(a, b)
            | Self::FCmpLe(a, b)
            | Self::FCmpGt(a, b)
            | Self::FCmpGe(a, b)
            | Self::LogicalAnd(a, b)
            | Self::LogicalOr(a, b)
            | Self::BitAnd32(a, b)
            | Self::BitOr32(a, b)
            | Self::BitXor32(a, b)
            | Self::Shl32(a, b)
            | Self::Shr32(a, b)
            | Self::IMul64(a, b)
            | Self::IAdd64(a, b)
            | Self::MakeVec2(a, b)
            | Self::BdaStore {
                ptr: a, value: b, ..
            }
            | Self::ImageStore {
                coord: a, value: b, ..
            }
            | Self::AtomicAdd {
                ptr: a, value: b, ..
            }
            | Self::AtomicMin {
                ptr: a, value: b, ..
            }
            | Self::AtomicMax {
                ptr: a, value: b, ..
            } => vec![*a, *b],
            Self::AtomicCas {
                ptr,
                expected,
                desired,
                ..
            } => vec![*ptr, *expected, *desired],
            Self::MakeVec3(a, b, c) => vec![*a, *b, *c],
            Self::MakeVec4(a, b, c, d) => vec![*a, *b, *c, *d],
            Self::GlslStd450 { args, .. } => args.clone(),
            Self::WorkgroupAddr { index, .. } | Self::WorkgroupLoad { index, .. } => {
                index.iter().copied().collect()
            }
            Self::WorkgroupStore { index, value, .. } => {
                if let Some(idx) = index {
                    vec![*idx, *value]
                } else {
                    vec![*value]
                }
            }
            Self::ImageSample {
                image_inst,
                coord,
                lod,
                ..
            } => {
                if let Some(img) = image_inst {
                    vec![*img, *coord, *lod]
                } else {
                    vec![*coord, *lod]
                }
            }
            Self::RayQueryInitialize {
                flags,
                cull_mask,
                origin,
                t_min,
                dir,
                t_max,
                ..
            } => vec![*flags, *cull_mask, *origin, *t_min, *dir, *t_max],
            Self::Select {
                cond,
                on_true,
                on_false,
            } => vec![*cond, *on_true, *on_false],
            Self::Phi { incoming } => incoming.iter().map(|&(v, _)| v).collect(),
            Self::ConstBool(_)
            | Self::ConstI32(_)
            | Self::ConstU32(_)
            | Self::ConstF32(_)
            | Self::ConstI64(_)
            | Self::ConstU64(_)
            | Self::WorkgroupBarrier
            | Self::LoadBuiltin { .. }
            | Self::LoadPushConstant { .. }
            | Self::SpecConstant { .. }
            | Self::DescriptorLoad { .. }
            | Self::RayQueryProceed(_)
            | Self::RayQueryCommittedType(_)
            | Self::RayQueryCommittedInstanceCustomIndex(_)
            | Self::RayQueryCommittedBarycentrics(_)
            | Self::Nop => Vec::new(),
        }
    }
}

/// Single SSA instruction stored in [`IrFunction::insts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst {
    pub ty: Option<TypeId>,
    pub data: InstData,
    /// Set by [`crate::PropagateNonUniformPass`] on tainted indices, access chains, and loads.
    pub non_uniform: bool,
}

impl Inst {
    pub const fn new(ty: Option<TypeId>, data: InstData) -> Self {
        Self {
            ty,
            data,
            non_uniform: false,
        }
    }

    /// Returns `true` if this instruction cannot be eliminated by DCE even when unused.
    pub const fn has_side_effects(&self) -> bool {
        matches!(
            self.data,
            InstData::BdaStore { .. }
                | InstData::ImageStore { .. }
                | InstData::WorkgroupStore { .. }
                | InstData::AtomicAdd { .. }
                | InstData::AtomicMin { .. }
                | InstData::AtomicMax { .. }
                | InstData::AtomicCas { .. }
                | InstData::WorkgroupBarrier
                | InstData::SubgroupBallot(_)
                | InstData::SubgroupBallotBitCount(_)
                | InstData::SubgroupBroadcastFirst(_)
                | InstData::RayQueryInitialize { .. }
                | InstData::RayQueryProceed(_)
        )
    }
}

/// Structured control-flow node preserved all the way to SPIR-V emission
/// (avoids post-hoc CFG structurization).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredStmt {
    Eval(InstId),
    If {
        cond: InstId,
        then_body: Vec<StructuredStmt>,
        else_body: Vec<StructuredStmt>,
    },
    Switch {
        selector: InstId,
        cases: Vec<(i64, Vec<StructuredStmt>)>,
        default: Vec<StructuredStmt>,
    },
    Loop {
        body: Vec<StructuredStmt>,
        continuing: Vec<StructuredStmt>,
        unroll: bool,
        /// Optional compile-time constant trip count used by [`crate::UnrollLoopsPass`].
        trip_count: Option<u32>,
        /// Optional induction variable SSA instruction initialized before the loop.
        induction_init: Option<InstId>,
        /// Constant signed step applied to `induction_init` on each unrolled iteration.
        induction_step: i64,
    },
    Break,
    Continue,
    Return(Option<InstId>),
}

/// Structured SSA function body owned by an [`IrModule`].
#[derive(Debug, Clone)]
pub struct IrFunction {
    pub name: SmolStr,
    pub stage: Option<ShaderStage>,
    pub workgroup_size: Option<[u32; 3]>,
    pub required_caps: HardwareCaps,
    pub uses_bda: bool,
    pub uses_subgroups: bool,
    pub descriptors: Vec<IrDescriptorVar>,
    pub workgroup_vars: Vec<IrWorkgroupVar>,
    pub push_constants: Option<IrPushConstantBlock>,
    pub insts: Arena<Inst>,
    pub body: Vec<StructuredStmt>,
}

impl IrFunction {
    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self {
            name: name.into(),
            stage: None,
            workgroup_size: None,
            required_caps: HardwareCaps::empty(),
            uses_bda: false,
            uses_subgroups: false,
            descriptors: Vec::new(),
            workgroup_vars: Vec::new(),
            push_constants: None,
            insts: Arena::new(),
            body: Vec::new(),
        }
    }

    pub fn emit(&mut self, ty: Option<TypeId>, data: InstData) -> InstId {
        self.insts.alloc(Inst::new(ty, data))
    }
}

/// Complete structured SSA IR module ready for pass execution and SPIR-V emission.
#[derive(Debug, Default, Clone)]
pub struct IrModule {
    pub name: SmolStr,
    pub functions: Vec<IrFunction>,
}
