//! High-level intermediate representation (HIR), type layout engine, module graph,
//! BLAKE3 cache keys, lowering with early tier pruning, and compile-time SPIR-V legality validation.

pub mod hir;
pub mod lower;
pub mod module;
pub mod ty;
pub mod validate;

pub use hir::{
    EntryReflection, HirFnId, HirFunction, HirModule, HirParam, HirStruct, HirStructField,
    HirStructId, HirWorkgroupVar,
};
pub use lower::{PackSchema, lower_unit, lower_unit_with_schema};
pub use module::{L1ModuleHash, L2ArtifactHash, ModuleGraph, ModuleGraphError, ModuleId};
pub use ty::{LayoutKind, ScalarKind, StructLayout, Type, TypeArena, TypeId};
pub use validate::{ValidationSummary, validate_unit};
