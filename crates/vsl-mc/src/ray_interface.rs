//! Host-facing ray tracing interface reflection and explicit shader-link validation.
//!
//! The engine selects shader groups and SBT indices at runtime. This module does not
//! infer groups from entry-point names or require unrelated ray pipelines to share an ABI.

use crate::{CompiledShaderUnit, ShaderStage};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// Storage classes participating in a Vulkan ray tracing pipeline interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RayInterfaceKind {
    RayPayload,
    IncomingRayPayload,
    CallableData,
    IncomingCallableData,
    HitAttribute,
    ShaderRecord,
}

impl RayInterfaceKind {
    #[must_use]
    pub const fn allows_stage(self, stage: ShaderStage) -> bool {
        use ShaderStage::{AnyHit, Callable, ClosestHit, Intersection, Miss, RayGen};
        match self {
            Self::RayPayload => matches!(stage, RayGen | ClosestHit | Miss),
            Self::IncomingRayPayload => matches!(stage, ClosestHit | AnyHit | Miss),
            Self::CallableData => matches!(stage, RayGen | ClosestHit | Miss | Callable),
            Self::IncomingCallableData => matches!(stage, Callable),
            Self::HitAttribute => matches!(stage, Intersection | AnyHit | ClosestHit),
            Self::ShaderRecord => stage.is_ray_tracing(),
        }
    }

    #[must_use]
    pub const fn allows_multiple(self) -> bool {
        matches!(self, Self::RayPayload | Self::CallableData)
    }

    #[must_use]
    pub const fn has_location(self) -> bool {
        matches!(
            self,
            Self::RayPayload
                | Self::IncomingRayPayload
                | Self::CallableData
                | Self::IncomingCallableData
        )
    }
}

/// Scalar types of a ray payload or record. Integer signedness is preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RayScalar {
    Bool,
    I32,
    U32,
    F16,
    F32,
    I64,
    U64,
}

impl RayScalar {
    #[must_use]
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::F16 => 2,
            Self::I64 | Self::U64 => 8,
            Self::Bool | Self::I32 | Self::U32 | Self::F32 => 4,
        }
    }
}

/// Structural shader interface type, independent of SPIR-V result IDs.
/// Field and struct names are descriptive and do not determine link compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RayInterfaceType {
    Scalar {
        scalar: RayScalar,
    },
    Vector {
        scalar: RayScalar,
        len: u8,
    },
    Matrix {
        scalar: RayScalar,
        columns: u8,
        rows: u8,
    },
    Array {
        element: Box<Self>,
        len: u32,
    },
    Struct {
        name: SmolStr,
        fields: Vec<RayInterfaceMember>,
    },
    DeviceAddress {
        pointee: Box<Self>,
    },
    /// An IR type without a supported fixed-size ray interface representation.
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RayInterfaceMember {
    pub name: SmolStr,
    pub ty: RayInterfaceType,
}

impl RayInterfaceType {
    /// Compares the recursive value types, not merely their byte counts.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar { scalar: a }, Self::Scalar { scalar: b }) => a == b,
            (Self::Vector { scalar: a, len: x }, Self::Vector { scalar: b, len: y }) => {
                a == b && x == y
            }
            (
                Self::Matrix {
                    scalar: lhs_scalar,
                    columns: lhs_cols,
                    rows: lhs_rows,
                },
                Self::Matrix {
                    scalar: rhs_scalar,
                    columns: rhs_cols,
                    rows: rhs_rows,
                },
            ) => lhs_scalar == rhs_scalar && lhs_cols == rhs_cols && lhs_rows == rhs_rows,
            (Self::Array { element: a, len: x }, Self::Array { element: b, len: y }) => {
                x == y && a.matches(b)
            }
            (Self::Struct { fields: a, .. }, Self::Struct { fields: b, .. }) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.ty.matches(&b.ty))
            }
            (Self::DeviceAddress { pointee: a }, Self::DeviceAddress { pointee: b }) => {
                a.matches(b)
            }
            _ => false,
        }
    }

    /// Tight logical size. This is not the implementation-defined payload allocation
    /// size and must not be used to choose a device's pipeline stack size.
    #[must_use]
    pub fn logical_byte_size(&self) -> Option<u32> {
        match self {
            Self::Scalar { scalar } => Some(scalar.byte_size()),
            Self::Vector { scalar, len } => scalar.byte_size().checked_mul(u32::from(*len)),
            Self::Matrix {
                scalar,
                columns,
                rows,
            } => scalar
                .byte_size()
                .checked_mul(u32::from(*columns))?
                .checked_mul(u32::from(*rows)),
            Self::Array { element, len } => element.logical_byte_size()?.checked_mul(*len),
            Self::Struct { fields, .. } => fields.iter().try_fold(0u32, |sum, field| {
                sum.checked_add(field.ty.logical_byte_size()?)
            }),
            Self::DeviceAddress { .. } => Some(8),
            Self::Unsupported => None,
        }
    }
}

/// Explicit byte layout of data following a shader-group handle in an SBT record.
/// Strides correspond to the SPIR-V decorations, not host-language struct packing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShaderRecordLayout {
    pub byte_size: u32,
    pub alignment: u32,
    pub fields: Vec<ShaderRecordField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShaderRecordField {
    pub name: SmolStr,
    pub byte_offset: u32,
    pub byte_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub array_stride: Option<u32>,
    /// Strides of all array dimensions, outermost first (including `array_stride`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub array_strides: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_stride: Option<u32>,
    /// Nested struct layout, including the element layout of arrays of structs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested: Option<Box<ShaderRecordLayout>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RayInterfaceVariable {
    pub name: SmolStr,
    pub kind: RayInterfaceKind,
    pub location: Option<u32>,
    pub ty: RayInterfaceType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_layout: Option<ShaderRecordLayout>,
}

/// An actual host-selected shader call or procedural hit-group edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayInterfaceLink {
    /// The caller's selected `RayPayload` or `IncomingRayPayload` variable.
    Trace {
        caller_kind: RayInterfaceKind,
        location: u32,
    },
    /// The caller's selected `CallableData` or `IncomingCallableData` variable.
    Callable {
        caller_kind: RayInterfaceKind,
        location: u32,
    },
    /// The hit attributes written by a procedural intersection shader.
    ProceduralHit,
}

/// Validates one concrete caller/target relation before creating an RT pipeline.
/// Pass each possible runtime target separately, including targets selected through
/// dynamic SBT indices. Missing incoming variables are reported, rather than silently
/// treating unrelated stages with the same location as compatible.
pub fn validate_ray_interface_link(
    caller: &CompiledShaderUnit,
    target: &CompiledShaderUnit,
    link: RayInterfaceLink,
) -> Result<(), String> {
    let (source_kind, target_kind, location, stages_allowed) = match link {
        RayInterfaceLink::Trace {
            caller_kind,
            location,
        } => (
            caller_kind,
            RayInterfaceKind::IncomingRayPayload,
            Some(location),
            matches!(
                caller_kind,
                RayInterfaceKind::RayPayload | RayInterfaceKind::IncomingRayPayload
            ) && matches!(
                caller.stage,
                ShaderStage::RayGen | ShaderStage::ClosestHit | ShaderStage::Miss
            ) && matches!(
                target.stage,
                ShaderStage::AnyHit | ShaderStage::ClosestHit | ShaderStage::Miss
            ),
        ),
        RayInterfaceLink::Callable {
            caller_kind,
            location,
        } => (
            caller_kind,
            RayInterfaceKind::IncomingCallableData,
            Some(location),
            matches!(
                caller_kind,
                RayInterfaceKind::CallableData | RayInterfaceKind::IncomingCallableData
            ) && matches!(
                caller.stage,
                ShaderStage::RayGen
                    | ShaderStage::ClosestHit
                    | ShaderStage::Miss
                    | ShaderStage::Callable
            ) && target.stage == ShaderStage::Callable,
        ),
        RayInterfaceLink::ProceduralHit => (
            RayInterfaceKind::HitAttribute,
            RayInterfaceKind::HitAttribute,
            None,
            caller.stage == ShaderStage::Intersection
                && matches!(target.stage, ShaderStage::ClosestHit | ShaderStage::AnyHit),
        ),
    };
    let prefix = format!("{} -> {}", caller.entry_point, target.entry_point);
    if !stages_allowed || !source_kind.allows_stage(caller.stage) {
        return Err(format!(
            "{prefix}: invalid shader stages or storage class for {link:?}"
        ));
    }
    let mut sources = caller
        .ray_interfaces
        .iter()
        .filter(|v| v.kind == source_kind && v.location == location);
    let source = sources
        .next()
        .ok_or_else(|| format!("{prefix}: caller is missing {source_kind:?} at {location:?}"))?;
    if sources.next().is_some() {
        return Err(format!(
            "{prefix}: caller has an ambiguous {source_kind:?} interface at {location:?}"
        ));
    }
    let mut targets = target
        .ray_interfaces
        .iter()
        .filter(|v| v.kind == target_kind);
    let incoming = targets
        .next()
        .ok_or_else(|| format!("{prefix}: target is missing {target_kind:?}"))?;
    if targets.next().is_some() {
        return Err(format!(
            "{prefix}: target has multiple {target_kind:?} interfaces"
        ));
    }
    if incoming.location != source.location {
        return Err(format!(
            "{prefix}: interface location mismatch ({:?} vs {:?})",
            source.location, incoming.location
        ));
    }
    if !source.ty.matches(&incoming.ty) {
        return Err(format!(
            "{prefix}: interface type mismatch ({} vs {})",
            source.name, incoming.name
        ));
    }
    Ok(())
}
