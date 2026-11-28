use la_arena::{Arena, Idx};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use vsl_hir::{LayoutKind, ScalarKind, Type, TypeArena, TypeId};
use vsl_mc::{
    BuiltinFieldType, BuiltinStructDesc, DescriptorAccess, DescriptorKind, HardwareCaps,
    ImageFormat, PackParamType, SampleScalar, ShaderStage, TextureDim,
    builtin_struct_layout_by_name,
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

/// Built-in shader stage input/output variable kind lowered to SPIR-V `OpDecorate BuiltIn`.
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
    SubgroupLocalInvocationId,
    SubgroupSize,
    Position,
    FragDepth,
    FrontFacing,
    // VK_KHR_ray_tracing_pipeline builtins
    LaunchIdKhr,
    LaunchSizeKhr,
    WorldRayOriginKhr,
    WorldRayDirectionKhr,
    ObjectRayOriginKhr,
    ObjectRayDirectionKhr,
    RayTminKhr,
    RayTmaxKhr,
    InstanceCustomIndexKhr,
    ObjectToWorldKhr,
    WorldToObjectKhr,
    HitKindKhr,
    IncomingRayFlagsKhr,
    RayGeometryIndexKhr,
    // VK_EXT_mesh_shader builtins
    PrimitivePointIndicesExt,
    PrimitiveLineIndicesExt,
    PrimitiveTriangleIndicesExt,
    CullPrimitiveExt,
    // Legacy geometry / tessellation builtins
    InvocationId,
    PatchVertices,
    TessLevelOuter,
    TessLevelInner,
    TessCoord,
    PrimitiveId,
    Layer,
}

/// Memory qualifiers lowered onto SPIR-V BDA pointer loads/stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryAccess {
    pub align: u32,
    pub non_writable: bool,
    pub restrict: bool,
}

/// SPIR-V storage class associated with an [`IrType::Pointer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrStorageClass {
    Function,
    Workgroup,
    Uniform,
    StorageBuffer,
    PhysicalStorageBuffer,
    Image,
    RayPayloadKhr,
    IncomingRayPayloadKhr,
    HitAttributeKhr,
    CallableDataKhr,
    IncomingCallableDataKhr,
    ShaderRecordBufferKhr,
    TaskPayloadWorkgroupExt,
}

/// Scalar element type for typed IR values, push constants, BDA, and specialization constants.
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

impl ScalarIrTy {
    /// Byte size of this scalar in GPU memory layouts.
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::F16 => 2,
            Self::Bool | Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }

    /// Converts an HIR [`ScalarKind`] into its IR scalar counterpart.
    pub const fn from_hir_scalar(kind: ScalarKind) -> Self {
        match kind {
            ScalarKind::Bool => Self::Bool,
            ScalarKind::I32 => Self::I32,
            ScalarKind::U32 => Self::U32,
            ScalarKind::I64 => Self::I64,
            ScalarKind::U64 => Self::U64,
            ScalarKind::F32 => Self::F32,
            ScalarKind::F16 => Self::F16,
        }
    }
}

/// Typed field layout inside an [`IrStructLayout`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IrStructFieldLayout {
    pub name: SmolStr,
    pub byte_offset: u32,
    pub byte_size: u32,
    pub ty: IrType,
}

/// Complete memory layout for an IR struct (local, workgroup, UBO, SSBO, or BDA pointee).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IrStructLayout {
    pub name: SmolStr,
    pub fields: Vec<IrStructFieldLayout>,
    pub size: u32,
    pub align: u32,
    pub layout: LayoutKind,
}

/// First-class SSA value and memory type in VSL 2.0 IR.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrType {
    #[default]
    Void,
    Scalar(ScalarIrTy),
    Vector {
        scalar: ScalarIrTy,
        len: u8,
    },
    Matrix {
        scalar: ScalarIrTy,
        cols: u8,
        rows: u8,
    },
    Array {
        element: Box<IrType>,
        len: u32,
    },
    RuntimeArray(Box<IrType>),
    Struct(IrStructLayout),
    Pointer {
        pointee: Box<IrType>,
        align: u32,
        storage: IrStorageClass,
    },
}

impl IrType {
    /// Returns the underlying scalar element type for scalar, vector, and matrix types.
    pub const fn scalar_ty(&self) -> Option<ScalarIrTy> {
        match self {
            Self::Scalar(s) | Self::Vector { scalar: s, .. } | Self::Matrix { scalar: s, .. } => {
                Some(*s)
            }
            Self::Void
            | Self::Array { .. }
            | Self::RuntimeArray(_)
            | Self::Struct(_)
            | Self::Pointer { .. } => None,
        }
    }

    /// Returns `true` if this type is a floating-point scalar, vector, or matrix (`f32`/`f16`).
    pub const fn is_float(&self) -> bool {
        matches!(self.scalar_ty(), Some(ScalarIrTy::F32 | ScalarIrTy::F16))
    }

    /// Returns `true` if this type has an `f16` element type.
    pub const fn is_f16(&self) -> bool {
        matches!(self.scalar_ty(), Some(ScalarIrTy::F16))
    }

    /// Returns `true` if this type has a 64-bit integer element type (`i64`/`u64`).
    pub const fn is_64bit(&self) -> bool {
        matches!(self.scalar_ty(), Some(ScalarIrTy::I64 | ScalarIrTy::U64))
    }

    /// Returns `true` if this type has a signed integer element type (`i32`/`i64`).
    pub const fn is_signed_int(&self) -> bool {
        matches!(self.scalar_ty(), Some(ScalarIrTy::I32 | ScalarIrTy::I64))
    }

    /// Returns `true` if this type has an unsigned integer element type (`u32`/`u64`).
    pub const fn is_unsigned_int(&self) -> bool {
        matches!(self.scalar_ty(), Some(ScalarIrTy::U32 | ScalarIrTy::U64))
    }

    /// Returns the component count if this type is a vector.
    pub const fn vector_len(&self) -> Option<u8> {
        match self {
            Self::Vector { len, .. } => Some(*len),
            _ => None,
        }
    }

    /// Returns the tight byte size of this type (or `0` for `Void`, runtime arrays, and logical pointers).
    pub fn byte_size(&self) -> u32 {
        match self {
            Self::Void | Self::RuntimeArray(_) => 0,
            Self::Scalar(s) => s.byte_size(),
            Self::Vector { scalar, len } => scalar.byte_size().saturating_mul(u32::from(*len)),
            Self::Matrix { scalar, cols, rows } => scalar
                .byte_size()
                .saturating_mul(u32::from(*cols))
                .saturating_mul(u32::from(*rows)),
            Self::Array { element, len } => element.byte_size().saturating_mul(*len),
            Self::Struct(layout) => layout.size,
            Self::Pointer {
                storage: IrStorageClass::PhysicalStorageBuffer,
                ..
            } => 8,
            Self::Pointer { .. } => 0,
        }
    }

    /// Lowers a resolved HIR [`TypeId`] into a self-contained [`IrType`].
    pub fn from_hir_type(arena: &TypeArena, id: TypeId) -> Self {
        let mut visited = Vec::new();
        Self::from_hir_type_inner(arena, id, &mut visited)
    }

    fn from_hir_type_inner(arena: &TypeArena, id: TypeId, visited: &mut Vec<TypeId>) -> Self {
        if visited.contains(&id) {
            return Self::Void;
        }
        visited.push(id);
        let result = match arena.get(id) {
            Type::Void
            | Type::Texture2D { .. }
            | Type::Texture2DArray { .. }
            | Type::Texture3D { .. }
            | Type::TextureCube { .. }
            | Type::RwImage2D(_)
            | Type::RwImage2DArray(_)
            | Type::RwImage3D(_)
            | Type::Sampler
            | Type::SamplerShadow
            | Type::AccelStruct
            | Type::RayQuery
            | Type::TypeParam(_) => Self::Void,
            Type::Scalar(s) => Self::Scalar(ScalarIrTy::from_hir_scalar(*s)),
            Type::Vector { scalar, len } => Self::Vector {
                scalar: ScalarIrTy::from_hir_scalar(*scalar),
                len: *len,
            },
            Type::Matrix { scalar, cols, rows } => Self::Matrix {
                scalar: ScalarIrTy::from_hir_scalar(*scalar),
                cols: *cols,
                rows: *rows,
            },
            Type::Array { element, len } => Self::Array {
                element: Box::new(Self::from_hir_type_inner(arena, *element, visited)),
                len: *len,
            },
            Type::Pointer { pointee, align, .. } => Self::Pointer {
                pointee: Box::new(Self::from_hir_type_inner(arena, *pointee, visited)),
                align: *align,
                storage: IrStorageClass::PhysicalStorageBuffer,
            },
            Type::Buffer { element } | Type::RwBuffer { element } => Self::RuntimeArray(Box::new(
                Self::from_hir_type_inner(arena, *element, visited),
            )),
            Type::Struct {
                name,
                layout,
                fields,
            } => {
                if let Some(computed) = arena.compute_struct_layout(fields, *layout) {
                    let ir_fields = fields
                        .iter()
                        .zip(computed.field_offsets.iter())
                        .map(|((field_name, field_ty_id), (_, byte_offset, byte_size))| {
                            IrStructFieldLayout {
                                name: field_name.clone(),
                                byte_offset: *byte_offset,
                                byte_size: *byte_size,
                                ty: Self::from_hir_type_inner(arena, *field_ty_id, visited),
                            }
                        })
                        .collect();
                    Self::Struct(IrStructLayout {
                        name: name.clone(),
                        fields: ir_fields,
                        size: computed.size,
                        align: computed.align,
                        layout: *layout,
                    })
                } else {
                    let mut cursor = 0u32;
                    let ir_fields = fields
                        .iter()
                        .map(|(field_name, field_ty_id)| {
                            let ty = Self::from_hir_type_inner(arena, *field_ty_id, visited);
                            let sz = ty.byte_size();
                            let entry = IrStructFieldLayout {
                                name: field_name.clone(),
                                byte_offset: cursor,
                                byte_size: sz,
                                ty,
                            };
                            cursor = cursor.saturating_add(sz);
                            entry
                        })
                        .collect();
                    Self::Struct(IrStructLayout {
                        name: name.clone(),
                        fields: ir_fields,
                        size: cursor,
                        align: 4,
                        layout: *layout,
                    })
                }
            }
        };
        visited.pop();
        result
    }

    /// Converts a built-in engine buffer field type into an [`IrType`].
    pub fn from_builtin_field_type(ft: BuiltinFieldType) -> Self {
        match ft {
            BuiltinFieldType::Bool => Self::Scalar(ScalarIrTy::Bool),
            BuiltinFieldType::I32 => Self::Scalar(ScalarIrTy::I32),
            BuiltinFieldType::U32 => Self::Scalar(ScalarIrTy::U32),
            BuiltinFieldType::I64 => Self::Scalar(ScalarIrTy::I64),
            BuiltinFieldType::U64 => Self::Scalar(ScalarIrTy::U64),
            BuiltinFieldType::F32 => Self::Scalar(ScalarIrTy::F32),
            BuiltinFieldType::Vec2 => Self::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            BuiltinFieldType::Vec3 => Self::Vector {
                scalar: ScalarIrTy::F32,
                len: 3,
            },
            BuiltinFieldType::Vec4 => Self::Vector {
                scalar: ScalarIrTy::F32,
                len: 4,
            },
            BuiltinFieldType::UVec2 => Self::Vector {
                scalar: ScalarIrTy::U32,
                len: 2,
            },
            BuiltinFieldType::UVec3 => Self::Vector {
                scalar: ScalarIrTy::U32,
                len: 3,
            },
            BuiltinFieldType::UVec4 => Self::Vector {
                scalar: ScalarIrTy::U32,
                len: 4,
            },
            BuiltinFieldType::Mat4 => Self::Matrix {
                scalar: ScalarIrTy::F32,
                cols: 4,
                rows: 4,
            },
            BuiltinFieldType::RuntimeArrayStruct(name) => {
                let elem = builtin_struct_layout_by_name(name).map_or_else(
                    || {
                        Self::Struct(IrStructLayout {
                            name: SmolStr::new(name),
                            fields: Vec::new(),
                            size: 0,
                            align: 16,
                            layout: LayoutKind::Std430,
                        })
                    },
                    Self::from_builtin_struct,
                );
                Self::RuntimeArray(Box::new(elem))
            }
        }
    }

    /// Converts a built-in engine UBO/SSBO struct descriptor into an [`IrType::Struct`].
    pub fn from_builtin_struct(desc: &BuiltinStructDesc) -> Self {
        let layout = if desc.name.ends_with("UBO") {
            LayoutKind::Std140
        } else {
            LayoutKind::Std430
        };
        let fields = desc
            .fields
            .iter()
            .map(|f| IrStructFieldLayout {
                name: SmolStr::new_static(f.name),
                byte_offset: f.byte_offset,
                byte_size: f.byte_size,
                ty: Self::from_builtin_field_type(f.ty),
            })
            .collect();
        Self::Struct(IrStructLayout {
            name: SmolStr::new_static(desc.name),
            fields,
            size: desc.total_size,
            align: desc.align,
            layout,
        })
    }

    /// Converts a `pack.vsl` setting [`PackParamType`] into its [`IrType`].
    pub const fn from_pack_param_type(pt: PackParamType) -> Self {
        match pt {
            PackParamType::Bool => Self::Scalar(ScalarIrTy::Bool),
            PackParamType::I32 | PackParamType::Enum => Self::Scalar(ScalarIrTy::I32),
            PackParamType::U32 => Self::Scalar(ScalarIrTy::U32),
            PackParamType::F32 => Self::Scalar(ScalarIrTy::F32),
            PackParamType::Vec3 => Self::Vector {
                scalar: ScalarIrTy::F32,
                len: 3,
            },
        }
    }
}

/// Function-local (`StorageClass::Function`) variable declaration attached to an [`IrFunction`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrLocalVar {
    pub name: SmolStr,
    pub ty: IrType,
}

/// Formal parameter declaration attached to an [`IrFunction`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrParam {
    pub name: SmolStr,
    pub ty: IrType,
    pub is_inout: bool,
    pub is_out: bool,
}

/// Interpolation qualifier on a raster stage input/output variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrInterpolation {
    Smooth,
    Flat,
    NoPerspective,
    Centroid,
}

/// Raster / Mesh / Tessellation / Geometry stage input/output variable declaration (`StorageClass::Input` / `Output`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrStageIoVar {
    pub name: SmolStr,
    pub location: u32,
    pub is_output: bool,
    pub ty: IrType,
    pub interpolation: IrInterpolation,
    #[serde(default)]
    pub per_primitive: bool,
}

/// Non-descriptor stage interface variable (RT payload, hit attribute, callable data, SBT record, or task payload).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrStageVar {
    pub name: SmolStr,
    pub storage: IrStorageClass,
    pub location: Option<u32>,
    pub ty: IrType,
}

/// Primitive topology for Mesh and Geometry stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrTopology {
    Points,
    Lines,
    Triangles,
    LineStrip,
    TriangleStrip,
}

/// Tessellation domain mode (`ExecutionMode::Triangles` / `Quads` / `Isolines`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrTessDomain {
    Triangles,
    Quads,
    Isolines,
}

/// Tessellation spacing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrTessSpacing {
    Equal,
    FractionalEven,
    FractionalOdd,
}

/// Tessellation vertex winding order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrTessOrder {
    Cw,
    Ccw,
}

/// Stage execution-mode metadata for Mesh, Tessellation, and Geometry shaders.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrStageExecMeta {
    pub max_vertices: Option<u32>,
    pub max_primitives: Option<u32>,
    pub output_topology: Option<IrTopology>,
    pub input_topology: Option<IrTopology>,
    pub invocations: Option<u32>,
    pub tess_domain: Option<IrTessDomain>,
    pub tess_spacing: Option<IrTessSpacing>,
    pub tess_order: Option<IrTessOrder>,
    pub point_mode: bool,
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
    #[serde(default)]
    pub struct_layout: Option<IrStructLayout>,
}

/// Shared workgroup (`StorageClass::Workgroup`) variable declaration attached to an [`IrFunction`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrWorkgroupVar {
    pub name: SmolStr,
    pub element_is_u32: bool,
    pub array_len: Option<u32>,
    #[serde(default)]
    pub ty: IrType,
}

impl IrWorkgroupVar {
    /// Constructs a workgroup variable with an explicit [`IrType`], keeping legacy
    /// `element_is_u32` and `array_len` fields synchronized.
    pub fn typed(name: impl Into<SmolStr>, ty: IrType) -> Self {
        let (element_is_u32, array_len) = match &ty {
            IrType::Array { element, len } => (
                matches!(element.as_ref(), IrType::Scalar(ScalarIrTy::U32)),
                Some(*len),
            ),
            IrType::Scalar(ScalarIrTy::U32) => (true, None),
            _ => (false, None),
        };
        Self {
            name: name.into(),
            element_is_u32,
            array_len,
            ty,
        }
    }

    /// Returns the effective [`IrType`] of this workgroup variable, falling back to the
    /// legacy `element_is_u32`/`array_len` representation when `ty` is `IrType::Void`.
    pub fn effective_type(&self) -> IrType {
        if self.ty != IrType::Void {
            return self.ty.clone();
        }
        let elem = if self.element_is_u32 {
            IrType::Scalar(ScalarIrTy::U32)
        } else {
            IrType::Scalar(ScalarIrTy::F32)
        };
        match self.array_len {
            Some(len) => IrType::Array {
                element: Box::new(elem),
                len,
            },
            None => elem,
        }
    }
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
    Atan,
    Atan2,
    Asin,
    Acos,
    Tan,
    Exp2,
    Log2,
    InverseSqrt,
    Sign,
    Step,
    SmoothStep,
    Mod,
    Trunc,
    Round,
    PackHalf2x16,
    UnpackHalf2x16,
    PackUnorm4x8,
    UnpackUnorm4x8,
    FindILsb,
    FindSMsb,
    FindUMsb,
}

/// SPIR-V `OpSpecConstantOp` operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecConstantOpKind {
    IAdd,
    ISub,
    IMul,
    UDiv,
    SDiv,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeftLogical,
    ShiftRightLogical,
    Select,
    IEqual,
    INotEqual,
    ULessThan,
    SLessThan,
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
    /// IEEE-754 half-precision float constant stored as raw `u16` bit pattern.
    ConstF16(u16),
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
    /// Floating-point precision conversion (`OpFConvert`, e.g. `f32 <-> f16`).
    FConvert {
        value: InstId,
        target: ScalarIrTy,
    },
    /// Specialization constant (`OpSpecConstant` / `OpSpecConstantTrue`/`False`) with `SpecId`.
    SpecConstant {
        spec_id: u32,
        scalar: ScalarIrTy,
        default_bits: u32,
    },
    /// Specialization constant expression (`OpSpecConstantOp`).
    SpecConstantOp {
        op: SpecConstantOpKind,
        operands: Vec<InstId>,
    },
    /// Extended math operation (`GLSL.std.450` `OpExtInst` or core `OpDot`).
    GlslStd450 {
        op: GlslStd450Op,
        args: Vec<InstId>,
    },
    /// Vector scaled by scalar (`OpVectorTimesScalar`).
    VectorTimesScalar {
        vec: InstId,
        scalar: InstId,
    },
    /// Matrix multiplied by column vector (`OpMatrixTimesVector`).
    MatrixTimesVector {
        mat: InstId,
        vec: InstId,
    },
    /// Matrix multiplied by matrix (`OpMatrixTimesMatrix`).
    MatrixTimesMatrix {
        lhs: InstId,
        rhs: InstId,
    },
    /// Vector component swizzle/shuffle (`OpVectorShuffle`).
    VectorShuffle {
        lhs: InstId,
        rhs: InstId,
        components: Vec<u32>,
    },
    /// General composite construction (`OpCompositeConstruct`) for vectors, matrices, arrays, and structs.
    CompositeConstruct {
        elems: Vec<InstId>,
    },
    /// General composite extraction (`OpCompositeExtract`) along constant member/element indices.
    CompositeExtract {
        composite: InstId,
        indices: Vec<u32>,
    },
    /// General composite insertion (`OpCompositeInsert`) along constant member/element indices.
    CompositeInsert {
        composite: InstId,
        value: InstId,
        indices: Vec<u32>,
    },
    /// SSA merge node over incoming `(value, predecessor_tag)` pairs from structured control-flow paths.
    Phi {
        incoming: Vec<(InstId, u32)>,
    },
    /// Address of a function-local variable `func.local_vars[local_index]`.
    LocalAddr {
        local_index: u32,
    },
    /// Loads a value from a function-local pointer.
    LocalLoad {
        ptr: InstId,
    },
    /// Stores a value to a function-local pointer.
    LocalStore {
        ptr: InstId,
        value: InstId,
    },
    /// Generic pointer indexing (`OpAccessChain`) over local, workgroup, or buffer pointers.
    AccessChain {
        base_ptr: InstId,
        indices: Vec<InstId>,
    },
    /// Access chain directly rooted at descriptor `func.descriptors[descriptor_index]`.
    DescriptorAccessChain {
        descriptor_index: u32,
        indices: Vec<InstId>,
    },
    /// Stores a value through a descriptor pointer (e.g., SSBO element pointer).
    DescriptorStore {
        ptr: InstId,
        value: InstId,
    },
    /// References formal parameter `func.params[index]`.
    Param {
        index: u32,
    },
    /// Calls another [`IrFunction`] in the same [`IrModule`] by name.
    FunctionCall {
        callee: SmolStr,
        args: Vec<InstId>,
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
    /// Samples or fetches from an image descriptor declared at `func.descriptors[descriptor_index]`.
    /// When `is_fetch` is `true`, integer texel fetch (`OpImageFetch`) is emitted; otherwise
    /// normalized UV/cube sampling (`OpImageSampleExplicitLod`) is emitted.
    ImageSample {
        descriptor_index: u32,
        image_inst: Option<InstId>,
        coord: InstId,
        lod: InstId,
        is_fetch: bool,
    },
    /// Fragment-stage implicit-LOD image sample (`OpImageSampleImplicitLod`).
    ImageSampleImplicitLod {
        descriptor_index: u32,
        image_inst: Option<InstId>,
        coord: InstId,
    },
    /// Explicit-gradient image sample (`OpImageSampleExplicitLod` with `Grad`).
    ImageSampleGrad {
        descriptor_index: u32,
        image_inst: Option<InstId>,
        coord: InstId,
        ddx: InstId,
        ddy: InstId,
    },
    /// Depth-comparison image sample with explicit LOD (`OpImageSampleDrefExplicitLod`).
    ImageSampleDrefExplicitLod {
        descriptor_index: u32,
        image_inst: Option<InstId>,
        coord: InstId,
        dref: InstId,
        lod: InstId,
    },
    /// 4-texel component gather (`OpImageGather`).
    ImageGather {
        descriptor_index: u32,
        image_inst: Option<InstId>,
        coord: InstId,
        component: u32,
    },
    /// Image dimension query (`OpImageQuerySize` / `OpImageQuerySizeLod`).
    ImageQuerySize {
        descriptor_index: u32,
        lod: Option<InstId>,
    },
    /// Image mipmap level count query (`OpImageQueryLevels`).
    ImageQueryLevels {
        descriptor_index: u32,
    },
    /// Texel pointer for storage image atomics (`OpImageTexelPointer`).
    ImageTexelPointer {
        descriptor_index: u32,
        coord: InstId,
        sample: InstId,
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
    /// Explicit memory barrier (`OpMemoryBarrier`).
    MemoryBarrier {
        scope: AtomicScope,
        order: AtomicOrder,
    },
    SubgroupBallot(InstId),
    SubgroupBallotBitCount(InstId),
    SubgroupBallotExclusiveBitCount(InstId),
    SubgroupBallotInclusiveBitCount(InstId),
    SubgroupBroadcastFirst(InstId),
    SubgroupAdd(InstId),
    SubgroupMin(InstId),
    SubgroupMax(InstId),
    SubgroupExclusiveAdd(InstId),
    SubgroupInclusiveAdd(InstId),
    SubgroupShuffle {
        value: InstId,
        lane: InstId,
    },
    SubgroupShuffleXor {
        value: InstId,
        mask: InstId,
    },
    SubgroupAny(InstId),
    SubgroupAll(InstId),
    SubgroupElect,
    QuadSwapHorizontal(InstId),
    QuadSwapVertical(InstId),
    QuadSwapDiagonal(InstId),
    /// Bitfield extract (`OpBitFieldSExtract` / `OpBitFieldUExtract`).
    BitFieldExtract {
        base: InstId,
        offset: InstId,
        count: InstId,
        signed: bool,
    },
    /// Bitfield insert (`OpBitFieldInsert`).
    BitFieldInsert {
        base: InstId,
        insert: InstId,
        offset: InstId,
        count: InstId,
    },
    /// Population count (`OpBitCount`).
    BitCount(InstId),
    /// Bit-preserving reinterpretation (`OpBitcast`).
    Bitcast {
        value: InstId,
        target: IrType,
    },
    /// NaN predicate (`OpIsNan`).
    IsNan(InstId),
    /// Infinity predicate (`OpIsInf`).
    IsInf(InstId),
    /// Fragment screen-space X derivative (`OpDPdx`).
    DPdx(InstId),
    /// Fragment screen-space Y derivative (`OpDPdy`).
    DPdy(InstId),
    /// Fragment screen-space sum of absolute derivatives (`OpFwidth`).
    Fwidth(InstId),
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
    /// Loads a raster stage input variable declared at `func.stage_io[var_index]`.
    LoadStageInput {
        var_index: u32,
    },
    /// Stores a raster stage output variable declared at `func.stage_io[var_index]`.
    StoreStageOutput {
        var_index: u32,
        value: InstId,
    },
    /// Stores a SPIR-V `BuiltIn` output variable (such as `Position` or `FragDepth`).
    StoreBuiltin {
        kind: BuiltinKind,
        value: InstId,
    },
    /// Fragment helper-invocation demotion (`OpDemoteToHelperInvocation`).
    DemoteToHelperInvocation,
    /// Loads a scalar from member `field_index` of the function's `PushConstant` block.
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
    /// Queries the candidate intersection type (`OpRayQueryGetIntersectionTypeKHR`), returning `u32`.
    RayQueryCandidateType(u32),
    /// Confirms the current candidate triangle intersection (`OpRayQueryConfirmIntersectionKHR`).
    RayQueryConfirmIntersection(u32),
    /// Terminates the ray query traversal (`OpRayQueryTerminateKHR`).
    RayQueryTerminate(u32),
    /// Generates a procedural candidate intersection at `t_hit` (`OpRayQueryGenerateIntersectionKHR`).
    RayQueryGenerateIntersection {
        ray_query_id: u32,
        t_hit: InstId,
    },
    /// Queries committed intersection distance `t` (`OpRayQueryGetIntersectionTKHR`).
    RayQueryCommittedT(u32),
    /// Queries candidate intersection distance `t` (`OpRayQueryGetIntersectionTKHR`).
    RayQueryCandidateT(u32),
    /// Queries committed instance ID (`OpRayQueryGetIntersectionInstanceIdKHR`).
    RayQueryCommittedInstanceId(u32),
    /// Queries the committed instance custom index (`OpRayQueryGetIntersectionInstanceCustomIndexKHR`), returning `u32`.
    RayQueryCommittedInstanceCustomIndex(u32),
    /// Queries committed primitive index (`OpRayQueryGetIntersectionPrimitiveIndexKHR`).
    RayQueryCommittedPrimitiveIndex(u32),
    /// Queries committed geometry index (`OpRayQueryGetIntersectionGeometryIndexKHR`).
    RayQueryCommittedGeometryIndex(u32),
    /// Queries committed object-to-world 4x3 matrix (`OpRayQueryGetIntersectionObjectToWorldKHR`).
    RayQueryCommittedObjectToWorld(u32),
    /// Queries committed world-to-object 4x3 matrix (`OpRayQueryGetIntersectionWorldToObjectKHR`).
    RayQueryCommittedWorldToObject(u32),
    /// Queries committed triangle front-face flag (`OpRayQueryGetIntersectionFrontFaceKHR`).
    RayQueryCommittedFrontFace(u32),
    /// Queries the committed triangle barycentrics (`OpRayQueryGetIntersectionBarycentricsKHR`).
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
    /// Traces a ray in `VK_KHR_ray_tracing_pipeline` (`OpTraceRayKHR`).
    TraceRayKhr {
        tlas_descriptor_index: u32,
        ray_flags: InstId,
        cull_mask: InstId,
        sbt_offset: InstId,
        sbt_stride: InstId,
        miss_index: InstId,
        origin: InstId,
        t_min: InstId,
        direction: InstId,
        t_max: InstId,
        payload_var_index: u32,
    },
    /// Ignores the current candidate intersection in `AnyHitKHR` (`OpIgnoreIntersectionKHR`).
    IgnoreIntersectionKhr,
    /// Terminates ray traversal from `AnyHitKHR` (`OpTerminateRayKHR`).
    TerminateRayKhr,
    /// Reports a procedural intersection from `IntersectionKHR` (`OpReportIntersectionKHR`).
    ReportIntersectionKhr {
        hit_t: InstId,
        hit_kind: InstId,
    },
    /// Invokes a callable shader via SBT index (`OpExecuteCallableKHR`).
    ExecuteCallableKhr {
        sbt_index: InstId,
        callable_var_index: u32,
    },
    /// Launches mesh shader workgroups from `TaskEXT` (`OpEmitMeshTasksEXT`).
    EmitMeshTasksExt {
        group_x: InstId,
        group_y: InstId,
        group_z: InstId,
        payload_var_index: Option<u32>,
    },
    /// Declares output vertex and primitive counts in `MeshEXT` (`OpSetMeshOutputsEXT`).
    SetMeshOutputsExt {
        vertex_count: InstId,
        primitive_count: InstId,
    },
    /// Emits a vertex in a `Geometry` shader (`OpEmitVertex`).
    EmitVertex,
    /// Completes the current primitive in a `Geometry` shader (`OpEndPrimitive`).
    EndPrimitive,
    /// Loads a stage variable (`RayPayloadKHR`, `HitAttributeKHR`, `CallableDataKHR`, `ShaderRecordBufferKHR`, `TaskPayloadWorkgroupEXT`).
    LoadStageVar {
        var_index: u32,
    },
    /// Stores a value into a stage variable.
    StoreStageVar {
        var_index: u32,
        value: InstId,
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
            | Self::LocalLoad { ptr: v }
            | Self::NonUniform(v)
            | Self::ResourceAccessChain { index: v, .. }
            | Self::ResourceLoad { ptr: v }
            | Self::SubgroupBallot(v)
            | Self::SubgroupBallotBitCount(v)
            | Self::SubgroupBallotExclusiveBitCount(v)
            | Self::SubgroupBallotInclusiveBitCount(v)
            | Self::SubgroupBroadcastFirst(v)
            | Self::SubgroupAdd(v)
            | Self::SubgroupMin(v)
            | Self::SubgroupMax(v)
            | Self::SubgroupExclusiveAdd(v)
            | Self::SubgroupInclusiveAdd(v)
            | Self::SubgroupAny(v)
            | Self::SubgroupAll(v)
            | Self::QuadSwapHorizontal(v)
            | Self::QuadSwapVertical(v)
            | Self::QuadSwapDiagonal(v)
            | Self::LogicalNot(v)
            | Self::BitNot32(v)
            | Self::BitCount(v)
            | Self::Bitcast { value: v, .. }
            | Self::IsNan(v)
            | Self::IsInf(v)
            | Self::DPdx(v)
            | Self::DPdy(v)
            | Self::Fwidth(v)
            | Self::CastI64ToU64(v)
            | Self::CastToU32(v)
            | Self::CastToI32(v)
            | Self::CastToF32(v)
            | Self::CastToI64(v)
            | Self::CastToU64(v)
            | Self::FConvert { value: v, .. }
            | Self::ExtractComponent { vec: v, .. }
            | Self::CompositeExtract { composite: v, .. }
            | Self::StoreStageOutput { value: v, .. }
            | Self::StoreBuiltin { value: v, .. }
            | Self::RayQueryGenerateIntersection { t_hit: v, .. } => vec![*v],
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
            | Self::VectorTimesScalar { vec: a, scalar: b }
            | Self::MatrixTimesVector { mat: a, vec: b }
            | Self::MatrixTimesMatrix { lhs: a, rhs: b }
            | Self::VectorShuffle { lhs: a, rhs: b, .. }
            | Self::CompositeInsert {
                composite: a,
                value: b,
                ..
            }
            | Self::LocalStore { ptr: a, value: b }
            | Self::DescriptorStore { ptr: a, value: b }
            | Self::BdaStore {
                ptr: a, value: b, ..
            }
            | Self::ImageStore {
                coord: a, value: b, ..
            }
            | Self::ImageTexelPointer {
                coord: a,
                sample: b,
                ..
            }
            | Self::AtomicAdd {
                ptr: a, value: b, ..
            }
            | Self::AtomicMin {
                ptr: a, value: b, ..
            }
            | Self::AtomicMax {
                ptr: a, value: b, ..
            }
            | Self::SubgroupShuffle { value: a, lane: b }
            | Self::SubgroupShuffleXor { value: a, mask: b } => vec![*a, *b],
            Self::AtomicCas {
                ptr,
                expected,
                desired,
                ..
            } => vec![*ptr, *expected, *desired],
            Self::BitFieldExtract {
                base,
                offset,
                count,
                ..
            } => vec![*base, *offset, *count],
            Self::BitFieldInsert {
                base,
                insert,
                offset,
                count,
            } => vec![*base, *insert, *offset, *count],
            Self::MakeVec3(a, b, c) => vec![*a, *b, *c],
            Self::MakeVec4(a, b, c, d) => vec![*a, *b, *c, *d],
            Self::GlslStd450 { args, .. } | Self::FunctionCall { args, .. } => args.clone(),
            Self::SpecConstantOp { operands, .. } => operands.clone(),
            Self::CompositeConstruct { elems } => elems.clone(),
            Self::DescriptorAccessChain { indices, .. } => indices.clone(),
            Self::AccessChain { base_ptr, indices } => {
                let mut out = Vec::with_capacity(1 + indices.len());
                out.push(*base_ptr);
                out.extend(indices.iter().copied());
                out
            }
            Self::WorkgroupAddr { index, .. }
            | Self::WorkgroupLoad { index, .. }
            | Self::ImageQuerySize { lod: index, .. } => index.iter().copied().collect(),
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
            Self::ImageSampleImplicitLod {
                image_inst, coord, ..
            }
            | Self::ImageGather {
                image_inst, coord, ..
            } => {
                if let Some(img) = image_inst {
                    vec![*img, *coord]
                } else {
                    vec![*coord]
                }
            }
            Self::ImageSampleGrad {
                image_inst,
                coord,
                ddx,
                ddy,
                ..
            } => {
                if let Some(img) = image_inst {
                    vec![*img, *coord, *ddx, *ddy]
                } else {
                    vec![*coord, *ddx, *ddy]
                }
            }
            Self::ImageSampleDrefExplicitLod {
                image_inst,
                coord,
                dref,
                lod,
                ..
            } => {
                if let Some(img) = image_inst {
                    vec![*img, *coord, *dref, *lod]
                } else {
                    vec![*coord, *dref, *lod]
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
            | Self::ConstF16(_)
            | Self::ConstI64(_)
            | Self::ConstU64(_)
            | Self::LocalAddr { .. }
            | Self::Param { .. }
            | Self::WorkgroupBarrier
            | Self::MemoryBarrier { .. }
            | Self::SubgroupElect
            | Self::LoadBuiltin { .. }
            | Self::LoadStageInput { .. }
            | Self::DemoteToHelperInvocation
            | Self::LoadPushConstant { .. }
            | Self::SpecConstant { .. }
            | Self::DescriptorLoad { .. }
            | Self::ImageQueryLevels { .. }
            | Self::RayQueryProceed(_)
            | Self::RayQueryCommittedType(_)
            | Self::RayQueryCandidateType(_)
            | Self::RayQueryConfirmIntersection(_)
            | Self::RayQueryTerminate(_)
            | Self::RayQueryCommittedT(_)
            | Self::RayQueryCandidateT(_)
            | Self::RayQueryCommittedInstanceId(_)
            | Self::RayQueryCommittedInstanceCustomIndex(_)
            | Self::RayQueryCommittedPrimitiveIndex(_)
            | Self::RayQueryCommittedGeometryIndex(_)
            | Self::RayQueryCommittedObjectToWorld(_)
            | Self::RayQueryCommittedWorldToObject(_)
            | Self::RayQueryCommittedFrontFace(_)
            | Self::RayQueryCommittedBarycentrics(_)
            | Self::IgnoreIntersectionKhr
            | Self::TerminateRayKhr
            | Self::EmitVertex
            | Self::EndPrimitive
            | Self::LoadStageVar { .. }
            | Self::Nop => Vec::new(),
            Self::TraceRayKhr {
                ray_flags,
                cull_mask,
                sbt_offset,
                sbt_stride,
                miss_index,
                origin,
                t_min,
                direction,
                t_max,
                ..
            } => vec![
                *ray_flags,
                *cull_mask,
                *sbt_offset,
                *sbt_stride,
                *miss_index,
                *origin,
                *t_min,
                *direction,
                *t_max,
            ],
            Self::ReportIntersectionKhr { hit_t, hit_kind } => vec![*hit_t, *hit_kind],
            Self::ExecuteCallableKhr { sbt_index, .. } => vec![*sbt_index],
            Self::EmitMeshTasksExt {
                group_x,
                group_y,
                group_z,
                ..
            } => vec![*group_x, *group_y, *group_z],
            Self::SetMeshOutputsExt {
                vertex_count,
                primitive_count,
            } => vec![*vertex_count, *primitive_count],
            Self::StoreStageVar { value, .. } => vec![*value],
        }
    }
}

/// Infers a reasonable default [`IrType`] directly from an [`InstData`] variant without arena context.
fn infer_static_ir_type(data: &InstData) -> IrType {
    match data {
        InstData::ConstBool(_)
        | InstData::ICmpEq(_, _)
        | InstData::ICmpNe(_, _)
        | InstData::ICmpLt(_, _)
        | InstData::ICmpLe(_, _)
        | InstData::ICmpGt(_, _)
        | InstData::ICmpGe(_, _)
        | InstData::UCmpLt(_, _)
        | InstData::FCmpEq(_, _)
        | InstData::FCmpNe(_, _)
        | InstData::FCmpLt(_, _)
        | InstData::FCmpLe(_, _)
        | InstData::FCmpGt(_, _)
        | InstData::FCmpGe(_, _)
        | InstData::LogicalAnd(_, _)
        | InstData::LogicalOr(_, _)
        | InstData::LogicalNot(_)
        | InstData::IsNan(_)
        | InstData::IsInf(_)
        | InstData::SubgroupAny(_)
        | InstData::SubgroupAll(_)
        | InstData::SubgroupElect
        | InstData::RayQueryProceed(_)
        | InstData::RayQueryCommittedFrontFace(_)
        | InstData::ReportIntersectionKhr { .. } => IrType::Scalar(ScalarIrTy::Bool),
        InstData::ConstI32(_) | InstData::CastToI32(_) => IrType::Scalar(ScalarIrTy::I32),
        InstData::ConstU32(_)
        | InstData::CastToU32(_)
        | InstData::SubgroupBallotBitCount(_)
        | InstData::SubgroupBallotExclusiveBitCount(_)
        | InstData::SubgroupBallotInclusiveBitCount(_)
        | InstData::BitCount(_)
        | InstData::ImageQueryLevels { .. }
        | InstData::RayQueryCommittedType(_)
        | InstData::RayQueryCandidateType(_)
        | InstData::RayQueryCommittedInstanceId(_)
        | InstData::RayQueryCommittedInstanceCustomIndex(_)
        | InstData::RayQueryCommittedPrimitiveIndex(_)
        | InstData::RayQueryCommittedGeometryIndex(_) => IrType::Scalar(ScalarIrTy::U32),
        InstData::ConstF32(_)
        | InstData::FAdd32(_, _)
        | InstData::FSub32(_, _)
        | InstData::FMul32(_, _)
        | InstData::FDiv32(_, _)
        | InstData::CastToF32(_)
        | InstData::ImageSampleDrefExplicitLod { .. }
        | InstData::RayQueryCommittedT(_)
        | InstData::RayQueryCandidateT(_) => IrType::Scalar(ScalarIrTy::F32),
        InstData::ConstF16(_) => IrType::Scalar(ScalarIrTy::F16),
        InstData::ConstI64(_) | InstData::CastToI64(_) => IrType::Scalar(ScalarIrTy::I64),
        InstData::ConstU64(_)
        | InstData::CastI64ToU64(_)
        | InstData::CastToU64(_)
        | InstData::ConvertPtrToU(_) => IrType::Scalar(ScalarIrTy::U64),
        InstData::FConvert { target, .. } => IrType::Scalar(*target),
        InstData::SpecConstant { scalar, .. } | InstData::DescriptorLoad { scalar, .. } => {
            IrType::Scalar(*scalar)
        }
        InstData::Bitcast { target, .. } => target.clone(),
        InstData::SubgroupBallot(_) => IrType::Vector {
            scalar: ScalarIrTy::U32,
            len: 4,
        },
        InstData::RayQueryCommittedBarycentrics(_) => IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 2,
        },
        InstData::RayQueryCommittedObjectToWorld(_)
        | InstData::RayQueryCommittedWorldToObject(_) => IrType::Matrix {
            scalar: ScalarIrTy::F32,
            cols: 4,
            rows: 3,
        },
        InstData::ConvertUToPtr { align, .. } | InstData::BdaOffset { align, .. } => {
            IrType::Pointer {
                pointee: Box::new(IrType::Void),
                align: *align,
                storage: IrStorageClass::PhysicalStorageBuffer,
            }
        }
        InstData::LoadBuiltin { kind, component } => match (kind, component) {
            (
                BuiltinKind::GlobalInvocationId
                | BuiltinKind::LocalInvocationId
                | BuiltinKind::WorkgroupId
                | BuiltinKind::NumWorkgroups,
                None,
            ) => IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 3,
            },
            (
                BuiltinKind::GlobalInvocationId
                | BuiltinKind::LocalInvocationId
                | BuiltinKind::WorkgroupId
                | BuiltinKind::NumWorkgroups
                | BuiltinKind::LocalInvocationIndex
                | BuiltinKind::SubgroupLocalInvocationId
                | BuiltinKind::SubgroupSize,
                _,
            ) => IrType::Scalar(ScalarIrTy::U32),
            (BuiltinKind::VertexIndex | BuiltinKind::InstanceIndex, _) => {
                IrType::Scalar(ScalarIrTy::I32)
            }
            (BuiltinKind::FragCoord | BuiltinKind::Position, None) => IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 4,
            },
            (BuiltinKind::FragCoord | BuiltinKind::Position | BuiltinKind::FragDepth, _) => {
                IrType::Scalar(ScalarIrTy::F32)
            }
            (BuiltinKind::FrontFacing, _) => IrType::Scalar(ScalarIrTy::Bool),
        },
        _ => IrType::Void,
    }
}

/// Single SSA instruction stored in [`IrFunction::insts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst {
    pub ty: Option<TypeId>,
    pub ir_ty: IrType,
    pub data: InstData,
    /// Set by [`crate::PropagateNonUniformPass`] on tainted indices, access chains, and loads.
    pub non_uniform: bool,
}

impl Inst {
    pub fn new(ty: Option<TypeId>, data: InstData) -> Self {
        let ir_ty = infer_static_ir_type(&data);
        Self {
            ty,
            ir_ty,
            data,
            non_uniform: false,
        }
    }

    pub const fn new_typed(ty: Option<TypeId>, ir_ty: IrType, data: InstData) -> Self {
        Self {
            ty,
            ir_ty,
            data,
            non_uniform: false,
        }
    }

    /// Returns `true` if this instruction cannot be eliminated by DCE even when unused.
    pub const fn has_side_effects(&self) -> bool {
        matches!(
            self.data,
            InstData::BdaStore { .. }
                | InstData::LocalStore { .. }
                | InstData::DescriptorStore { .. }
                | InstData::ImageStore { .. }
                | InstData::WorkgroupStore { .. }
                | InstData::StoreStageOutput { .. }
                | InstData::StoreBuiltin { .. }
                | InstData::DemoteToHelperInvocation
                | InstData::FunctionCall { .. }
                | InstData::AtomicAdd { .. }
                | InstData::AtomicMin { .. }
                | InstData::AtomicMax { .. }
                | InstData::AtomicCas { .. }
                | InstData::WorkgroupBarrier
                | InstData::MemoryBarrier { .. }
                | InstData::SubgroupBallot(_)
                | InstData::SubgroupBallotBitCount(_)
                | InstData::SubgroupBallotExclusiveBitCount(_)
                | InstData::SubgroupBallotInclusiveBitCount(_)
                | InstData::SubgroupBroadcastFirst(_)
                | InstData::SubgroupAdd(_)
                | InstData::SubgroupMin(_)
                | InstData::SubgroupMax(_)
                | InstData::SubgroupExclusiveAdd(_)
                | InstData::SubgroupInclusiveAdd(_)
                | InstData::SubgroupShuffle { .. }
                | InstData::SubgroupShuffleXor { .. }
                | InstData::SubgroupAny(_)
                | InstData::SubgroupAll(_)
                | InstData::SubgroupElect
                | InstData::QuadSwapHorizontal(_)
                | InstData::QuadSwapVertical(_)
                | InstData::QuadSwapDiagonal(_)
                | InstData::RayQueryInitialize { .. }
                | InstData::RayQueryProceed(_)
                | InstData::RayQueryConfirmIntersection(_)
                | InstData::RayQueryTerminate(_)
                | InstData::RayQueryGenerateIntersection { .. }
                | InstData::TraceRayKhr { .. }
                | InstData::IgnoreIntersectionKhr
                | InstData::TerminateRayKhr
                | InstData::ReportIntersectionKhr { .. }
                | InstData::ExecuteCallableKhr { .. }
                | InstData::EmitMeshTasksExt { .. }
                | InstData::SetMeshOutputsExt { .. }
                | InstData::EmitVertex
                | InstData::EndPrimitive
                | InstData::StoreStageVar { .. }
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
    pub local_size_spec_ids: Option<[u32; 3]>,
    pub required_caps: HardwareCaps,
    pub uses_bda: bool,
    pub uses_subgroups: bool,
    pub uses_quad_control: bool,
    pub maximally_reconverges: bool,
    pub params: Vec<IrParam>,
    pub return_ty: IrType,
    pub local_vars: Vec<IrLocalVar>,
    pub stage_io: Vec<IrStageIoVar>,
    pub stage_vars: Vec<IrStageVar>,
    pub stage_exec_meta: IrStageExecMeta,
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
            local_size_spec_ids: None,
            required_caps: HardwareCaps::empty(),
            uses_bda: false,
            uses_subgroups: false,
            uses_quad_control: false,
            maximally_reconverges: false,
            params: Vec::new(),
            return_ty: IrType::Void,
            local_vars: Vec::new(),
            stage_io: Vec::new(),
            stage_vars: Vec::new(),
            stage_exec_meta: IrStageExecMeta::default(),
            descriptors: Vec::new(),
            workgroup_vars: Vec::new(),
            push_constants: None,
            insts: Arena::new(),
            body: Vec::new(),
        }
    }

    /// Emits an instruction with an inferred [`IrType`] based on `data` and existing arena types.
    pub fn emit(&mut self, ty: Option<TypeId>, data: InstData) -> InstId {
        let ir_ty = self.infer_ir_type(&data);
        self.insts.alloc(Inst::new_typed(ty, ir_ty, data))
    }

    /// Emits an instruction with an explicit [`IrType`].
    pub fn emit_typed(&mut self, ir_ty: IrType, data: InstData) -> InstId {
        self.insts.alloc(Inst::new_typed(None, ir_ty, data))
    }

    fn infer_ir_type(&self, data: &InstData) -> IrType {
        match data {
            InstData::IAdd32(a, _)
            | InstData::ISub32(a, _)
            | InstData::IMul32(a, _)
            | InstData::IDiv32(a, _)
            | InstData::IRem32(a, _)
            | InstData::FAdd32(a, _)
            | InstData::FSub32(a, _)
            | InstData::FMul32(a, _)
            | InstData::FDiv32(a, _)
            | InstData::BitAnd32(a, _)
            | InstData::BitOr32(a, _)
            | InstData::BitXor32(a, _)
            | InstData::BitNot32(a)
            | InstData::Shl32(a, _)
            | InstData::Shr32(a, _)
            | InstData::IAdd64(a, _)
            | InstData::IMul64(a, _)
            | InstData::NonUniform(a)
            | InstData::VectorTimesScalar { vec: a, .. }
            | InstData::CompositeInsert { composite: a, .. }
            | InstData::SubgroupBroadcastFirst(a)
            | InstData::SubgroupAdd(a)
            | InstData::SubgroupMin(a)
            | InstData::SubgroupMax(a)
            | InstData::SubgroupExclusiveAdd(a)
            | InstData::SubgroupInclusiveAdd(a)
            | InstData::SubgroupShuffle { value: a, .. }
            | InstData::SubgroupShuffleXor { value: a, .. }
            | InstData::QuadSwapHorizontal(a)
            | InstData::QuadSwapVertical(a)
            | InstData::QuadSwapDiagonal(a)
            | InstData::BitFieldExtract { base: a, .. }
            | InstData::BitFieldInsert { base: a, .. }
            | InstData::DPdx(a)
            | InstData::DPdy(a)
            | InstData::Fwidth(a) => self.insts[*a].ir_ty.clone(),
            InstData::Select { on_true, .. } => self.insts[*on_true].ir_ty.clone(),
            InstData::Phi { incoming } => incoming
                .first()
                .map_or(IrType::Void, |&(v, _)| self.insts[v].ir_ty.clone()),
            InstData::MakeVec2(a, _) => self.insts[*a]
                .ir_ty
                .scalar_ty()
                .map_or(IrType::Void, |scalar| IrType::Vector { scalar, len: 2 }),
            InstData::MakeVec3(a, _, _) => self.insts[*a]
                .ir_ty
                .scalar_ty()
                .map_or(IrType::Void, |scalar| IrType::Vector { scalar, len: 3 }),
            InstData::MakeVec4(a, _, _, _) => self.insts[*a]
                .ir_ty
                .scalar_ty()
                .map_or(IrType::Void, |scalar| IrType::Vector { scalar, len: 4 }),
            InstData::ExtractComponent { vec, .. } => self.insts[*vec]
                .ir_ty
                .scalar_ty()
                .map_or(IrType::Void, IrType::Scalar),
            InstData::VectorShuffle {
                lhs, components, ..
            } => {
                if let Some(scalar) = self.insts[*lhs].ir_ty.scalar_ty() {
                    if components.len() == 1 {
                        IrType::Scalar(scalar)
                    } else {
                        IrType::Vector {
                            scalar,
                            len: components.len() as u8,
                        }
                    }
                } else {
                    IrType::Void
                }
            }
            InstData::MatrixTimesVector { mat, .. } => {
                if let IrType::Matrix { scalar, rows, .. } = self.insts[*mat].ir_ty {
                    IrType::Vector { scalar, len: rows }
                } else {
                    IrType::Void
                }
            }
            InstData::MatrixTimesMatrix { lhs, rhs } => {
                if let (IrType::Matrix { scalar, rows, .. }, IrType::Matrix { cols, .. }) =
                    (&self.insts[*lhs].ir_ty, &self.insts[*rhs].ir_ty)
                {
                    IrType::Matrix {
                        scalar: *scalar,
                        cols: *cols,
                        rows: *rows,
                    }
                } else {
                    IrType::Void
                }
            }
            InstData::LocalAddr { local_index } => self
                .local_vars
                .get(*local_index as usize)
                .map_or(IrType::Void, |lv| IrType::Pointer {
                    pointee: Box::new(lv.ty.clone()),
                    align: 4,
                    storage: IrStorageClass::Function,
                }),
            InstData::LocalLoad { ptr } => {
                if let IrType::Pointer { pointee, .. } = &self.insts[*ptr].ir_ty {
                    pointee.as_ref().clone()
                } else {
                    IrType::Void
                }
            }
            InstData::Param { index } => self
                .params
                .get(*index as usize)
                .map_or(IrType::Void, |p| p.ty.clone()),
            InstData::LoadStageInput { var_index } => self
                .stage_io
                .get(*var_index as usize)
                .map_or(IrType::Void, |v| v.ty.clone()),
            InstData::LoadPushConstant { field_index } => self
                .push_constants
                .as_ref()
                .and_then(|pc| pc.fields.get(*field_index as usize))
                .map_or(IrType::Scalar(ScalarIrTy::U32), |f| {
                    IrType::Scalar(f.scalar)
                }),
            InstData::WorkgroupLoad { var_index, index } => self
                .workgroup_vars
                .get(*var_index as usize)
                .map_or(IrType::Void, |wv| {
                    let ty = wv.effective_type();
                    if index.is_some()
                        && let IrType::Array { element, .. } = ty
                    {
                        *element
                    } else {
                        ty
                    }
                }),
            InstData::WorkgroupAddr { var_index, index } => self
                .workgroup_vars
                .get(*var_index as usize)
                .map_or(IrType::Void, |wv| {
                    let ty = wv.effective_type();
                    let pointee = if index.is_some()
                        && let IrType::Array { element, .. } = ty
                    {
                        *element
                    } else {
                        ty
                    };
                    IrType::Pointer {
                        pointee: Box::new(pointee),
                        align: 4,
                        storage: IrStorageClass::Workgroup,
                    }
                }),
            _ => infer_static_ir_type(data),
        }
    }
}

/// Complete structured SSA IR module ready for pass execution and SPIR-V emission.
#[derive(Debug, Default, Clone)]
pub struct IrModule {
    pub name: SmolStr,
    pub functions: Vec<IrFunction>,
}
