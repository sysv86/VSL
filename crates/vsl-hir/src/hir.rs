//! Arena-backed High-Level Intermediate Representation (HIR) and single-source-of-truth
//! entry-point reflection structures for VSL 2.0.

use crate::ty::{LayoutKind, StructLayout, TypeArena, TypeId};
use indexmap::IndexMap;
use la_arena::{Arena, Idx};
use smol_str::SmolStr;
use vsl_diagnostics::Span;
use vsl_mc::{
    DebugChannelBinding, DescriptorBinding, HardwareCaps, HookStage, PackParamBinding,
    PushConstantField, ShaderStage, SpecConstantUsage,
};
use vsl_syntax::ast::{Block, ParamMode};

/// Dense arena index for a lowered [`HirStruct`].
pub type HirStructId = Idx<HirStruct>;

/// Dense arena index for a lowered [`HirFunction`].
pub type HirFnId = Idx<HirFunction>;

/// Arena-backed HIR module produced by lowering a parsed [`vsl_syntax::ast::SourceUnit`].
#[derive(Debug, Clone)]
pub struct HirModule {
    pub module_name: SmolStr,
    pub structs: Arena<HirStruct>,
    pub functions: Arena<HirFunction>,
    pub workgroup_vars: Vec<HirWorkgroupVar>,
    pub types: TypeArena,
    /// Number of functions pruned during early tier filtering (Spec §3.4).
    pub pruned_functions: usize,
}

impl HirModule {
    /// Creates an empty HIR module with the given module name.
    #[must_use]
    pub fn new(module_name: impl Into<SmolStr>) -> Self {
        Self {
            module_name: module_name.into(),
            structs: Arena::new(),
            functions: Arena::new(),
            workgroup_vars: Vec::new(),
            types: TypeArena::new(),
            pruned_functions: 0,
        }
    }

    /// Iterates over all entry-point functions (annotated with `@stage(...)`).
    pub fn entry_points(&self) -> impl Iterator<Item = (HirFnId, &HirFunction)> {
        self.functions.iter().filter(|(_, f)| f.stage.is_some())
    }

    /// Looks up a lowered function by its identifier name.
    pub fn find_function(&self, name: &str) -> Option<&HirFunction> {
        self.functions
            .iter()
            .find_map(|(_, f)| (f.name == name).then_some(f))
    }

    /// Looks up a lowered struct by its identifier name.
    pub fn find_struct(&self, name: &str) -> Option<&HirStruct> {
        self.structs
            .iter()
            .find_map(|(_, s)| (s.name == name).then_some(s))
    }
}

/// Lowered struct field with resolved [`TypeId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirStructField {
    pub name: SmolStr,
    pub ty: TypeId,
    pub span: Span,
}

/// Lowered struct declaration with memory layout rules and computed byte offsets.
#[derive(Debug, Clone)]
pub struct HirStruct {
    pub name: SmolStr,
    pub generic_params: Vec<SmolStr>,
    pub layout: LayoutKind,
    pub is_push_constant: bool,
    pub required_caps: HardwareCaps,
    pub fields: Vec<HirStructField>,
    pub computed_layout: Option<StructLayout>,
    pub span: Span,
}

/// Lowered `workgroup var` shared-memory declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirWorkgroupVar {
    pub name: SmolStr,
    pub ty: TypeId,
    pub span: Span,
}

/// Lowered function parameter with resolved type and optional `#[builtin(...)]` binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirParam {
    pub name: SmolStr,
    pub mode: ParamMode,
    pub ty: Option<TypeId>,
    pub builtin: Option<SmolStr>,
    pub is_push_constant: bool,
    pub pack_resource_slot: Option<u32>,
    pub span: Span,
}

/// Lowered function or shader entry point.
#[derive(Debug, Clone)]
pub struct HirFunction {
    pub name: SmolStr,
    pub stage: Option<ShaderStage>,
    pub workgroup_size: Option<[u32; 3]>,
    pub hook: Option<HookStage>,
    pub tier_range: (u8, u8),
    pub required_caps: HardwareCaps,
    pub params: Vec<HirParam>,
    pub return_ty: Option<TypeId>,
    pub local_types: IndexMap<SmolStr, TypeId>,
    pub body_ast: Option<Block>,
    pub reflection: EntryReflection,
    pub span: Span,
}

/// Single-source-of-truth Vulkan reflection computed per entry point.
///
/// Emitted identically to both the SPIR-V backend (`vsl-spirv`) and `CompiledShaderUnit`
/// so descriptor sets, push-constant byte offsets, and debug channels never diverge.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EntryReflection {
    pub descriptors: Vec<DescriptorBinding>,
    pub push_constants: Vec<PushConstantField>,
    pub push_constant_size: u32,
    pub used_specs: Vec<SpecConstantUsage>,
    pub used_params: Vec<PackParamBinding>,
    pub debug_channels: Vec<DebugChannelBinding>,
    pub uses_bda: bool,
    pub uses_subgroups: bool,
}
