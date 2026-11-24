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
use vsl_syntax::ast::{BinaryOp, Block, ParamMode, UnaryOp};

/// Dense arena index for a lowered [`HirStruct`].
pub type HirStructId = Idx<HirStruct>;

/// Dense arena index for a lowered [`HirFunction`].
pub type HirFnId = Idx<HirFunction>;

/// Dense identifier for a resolved symbol in [`HirModule::symbols`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

/// Classification of a resolved [`HirSymbol`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Local,
    Param,
    Function,
    Struct,
    Const,
    WorkgroupVar,
    Descriptor,
    Builtin,
}

/// Resolved symbol metadata recorded in [`HirModule::symbols`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirSymbol {
    pub id: SymbolId,
    pub name: SmolStr,
    pub kind: SymbolKind,
    pub ty: TypeId,
    pub is_mut: bool,
    pub span: Span,
}

/// Lowered function or method call argument with optional keyword label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirCallArg {
    pub name: Option<SmolStr>,
    pub value: HirExpr,
    pub span: Span,
}

/// Typed HIR expression node with resolved [`TypeId`] and optional [`SymbolId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: TypeId,
    pub symbol: Option<SymbolId>,
    pub span: Span,
}

/// Typed HIR expression variants covering all lowered VSL 2.0 expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HirExprKind {
    IntLit(SmolStr),
    FloatLit(SmolStr),
    BoolLit(bool),
    StringLit(SmolStr),
    NoneLit,
    Path {
        path: SmolStr,
        turbofish: Vec<TypeId>,
    },
    ArrayLit {
        elements: Vec<HirExpr>,
    },
    StructLit {
        struct_name: SmolStr,
        fields: Vec<(SmolStr, HirExpr)>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<HirExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<HirExpr>,
        rhs: Box<HirExpr>,
    },
    Call {
        callee: Box<HirExpr>,
        resolved_fn: Option<HirFnId>,
        args: Vec<HirCallArg>,
    },
    MethodCall {
        receiver: Box<HirExpr>,
        method: SmolStr,
        resolved_fn: Option<HirFnId>,
        turbofish: Vec<TypeId>,
        args: Vec<HirCallArg>,
    },
    Field {
        receiver: Box<HirExpr>,
        field: SmolStr,
        field_index: Option<u32>,
    },
    Index {
        receiver: Box<HirExpr>,
        index: Box<HirExpr>,
    },
    If {
        cond: Box<HirExpr>,
        then_branch: HirBlock,
        else_branch: Option<Box<HirExpr>>,
    },
    Block(HirBlock),
    Swizzle {
        base: Box<HirExpr>,
        components: Vec<u8>,
        is_write: bool,
    },
}

/// Single `case` arm inside a [`HirStmt::Switch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirSwitchCase {
    pub value: HirExpr,
    pub body: HirBlock,
    pub span: Span,
}

/// Typed HIR statement node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HirStmt {
    Let {
        symbol: SymbolId,
        is_mut: bool,
        name: SmolStr,
        ty: TypeId,
        init: Option<HirExpr>,
        span: Span,
    },
    Return {
        value: Option<HirExpr>,
        span: Span,
    },
    For {
        binding_symbol: SymbolId,
        binding: SmolStr,
        binding_ty: TypeId,
        iter: HirExpr,
        body: HirBlock,
        span: Span,
    },
    While {
        cond: HirExpr,
        body: HirBlock,
        span: Span,
    },
    Loop {
        body: HirBlock,
        span: Span,
    },
    Switch {
        selector: HirExpr,
        cases: Vec<HirSwitchCase>,
        default: Option<HirBlock>,
        span: Span,
    },
    Break(Span),
    Continue(Span),
    Expr {
        expr: HirExpr,
        has_semi: bool,
        span: Span,
    },
}

/// Typed HIR block with resolved tail expression type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirBlock {
    pub stmts: Vec<HirStmt>,
    pub tail_expr: Option<Box<HirExpr>>,
    pub ty: TypeId,
    pub span: Span,
}

/// Lowered method signature inside a [`HirTrait`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirTraitMethod {
    pub name: SmolStr,
    pub params: Vec<HirParam>,
    pub return_ty: Option<TypeId>,
    pub span: Span,
}

/// Lowered trait declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirTrait {
    pub name: SmolStr,
    pub methods: Vec<HirTraitMethod>,
    pub span: Span,
}

/// Lowered inherent or trait `impl` block linking methods to a target struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirImpl {
    pub generic_params: Vec<(SmolStr, Option<SmolStr>)>,
    pub trait_name: Option<SmolStr>,
    pub target_struct: SmolStr,
    pub methods: Vec<HirFnId>,
    pub span: Span,
}

/// Lowered module-level `const` declaration with typed initializer expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirConst {
    pub name: SmolStr,
    pub ty: TypeId,
    pub value_expr: HirExpr,
    pub is_pub: bool,
    pub span: Span,
}

/// Arena-backed HIR module produced by lowering a parsed [`vsl_syntax::ast::SourceUnit`].
#[derive(Debug, Clone)]
pub struct HirModule {
    pub module_name: SmolStr,
    pub structs: Arena<HirStruct>,
    pub functions: Arena<HirFunction>,
    pub workgroup_vars: Vec<HirWorkgroupVar>,
    pub traits: Vec<HirTrait>,
    pub impls: Vec<HirImpl>,
    pub consts: Vec<HirConst>,
    pub symbols: Vec<HirSymbol>,
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
            traits: Vec::new(),
            impls: Vec::new(),
            consts: Vec::new(),
            symbols: Vec::new(),
            types: TypeArena::new(),
            pruned_functions: 0,
        }
    }

    /// Allocates a new [`HirSymbol`] in [`Self::symbols`] and returns its [`SymbolId`].
    pub fn alloc_symbol(
        &mut self,
        name: SmolStr,
        kind: SymbolKind,
        ty: TypeId,
        is_mut: bool,
        span: Span,
    ) -> SymbolId {
        let id = SymbolId(self.symbols.len() as u32);
        self.symbols.push(HirSymbol {
            id,
            name,
            kind,
            ty,
            is_mut,
            span,
        });
        id
    }

    /// Returns the [`HirSymbol`] for a previously allocated [`SymbolId`].
    pub fn symbol(&self, id: SymbolId) -> &HirSymbol {
        &self.symbols[id.0 as usize]
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
    pub generic_params: Vec<(SmolStr, Option<SmolStr>)>,
    pub stage: Option<ShaderStage>,
    pub workgroup_size: Option<[u32; 3]>,
    pub hook: Option<HookStage>,
    pub tier_range: (u8, u8),
    pub required_caps: HardwareCaps,
    pub params: Vec<HirParam>,
    pub return_ty: Option<TypeId>,
    pub local_types: IndexMap<SmolStr, TypeId>,
    pub body_hir: Option<HirBlock>,
    pub body_ast: Option<Block>,
    pub is_pub: bool,
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
