//! High-level intermediate representation (HIR), type layout engine, module graph,
//! BLAKE3 cache keys, lowering with early tier pruning, and compile-time SPIR-V legality validation.

pub mod hir;
pub mod lower;
pub mod module;
pub mod ty;
pub mod validate;

pub use hir::{
    EntryReflection, HirBlock, HirCallArg, HirConst, HirExpr, HirExprKind, HirFnId, HirFunction,
    HirImpl, HirModule, HirParam, HirStmt, HirStruct, HirStructField, HirStructId, HirSwitchCase,
    HirSymbol, HirTrait, HirTraitMethod, HirWorkgroupVar, SymbolId, SymbolKind,
};
pub use lower::{PackSchema, lower_unit, lower_unit_with_schema, lower_unit_with_schema_and_abi};
pub use module::{L1ModuleHash, L2ArtifactHash, ModuleGraph, ModuleGraphError, ModuleId};
pub use ty::{LayoutKind, CoopMatScalar, ScalarKind, StructLayout, Type, TypeArena, TypeId};
pub use validate::{ValidationSummary, validate_unit};
pub use vsl_mc::AbiProfile;
