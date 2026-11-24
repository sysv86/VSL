use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use vsl_diagnostics::Span;

/// Top-level parsed compilation unit (either a `.vsl` shader module or a `pack.vsl` manifest).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceUnit {
    pub module_decl: Option<ModuleDecl>,
    pub imports: Vec<ImportDecl>,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ident {
    pub name: SmolStr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Path {
    pub segments: Vec<Ident>,
    pub span: Span,
}

impl Path {
    pub fn joined(&self) -> String {
        self.segments
            .iter()
            .map(|seg| seg.name.as_str())
            .collect::<Vec<_>>()
            .join("::")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleDecl {
    pub path: Path,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportDecl {
    pub prefix: Path,
    pub tree: ImportTree,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportTree {
    Single,
    Group(Vec<Ident>),
    Glob,
}

/// Rust-style `#[...]` attribute or VSL `@stage(...)` / `@hook(...)` decorator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attribute {
    pub is_at_decorator: bool,
    pub entries: Vec<AttributeEntry>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributeEntry {
    pub name: Ident,
    pub args: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Item {
    Const(ConstDecl),
    WorkgroupVar(WorkgroupVarDecl),
    Struct(StructDecl),
    Trait(TraitDecl),
    Impl(ImplBlock),
    Function(FnDecl),
    Pack(PackDecl),
    Settings(SettingsDecl),
    Presets(PresetsDecl),
    Pipeline(PipelineDecl),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstDecl {
    pub attrs: Vec<Attribute>,
    pub is_pub: bool,
    pub name: Ident,
    pub ty: TypeExpr,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkgroupVarDecl {
    pub attrs: Vec<Attribute>,
    pub name: Ident,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericParam {
    pub name: Ident,
    pub bound: Option<Path>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructDecl {
    pub attrs: Vec<Attribute>,
    pub is_pub: bool,
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<StructField>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructField {
    pub attrs: Vec<Attribute>,
    pub is_pub: bool,
    pub name: Ident,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraitDecl {
    pub attrs: Vec<Attribute>,
    pub is_pub: bool,
    pub name: Ident,
    pub methods: Vec<FnDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImplBlock {
    pub attrs: Vec<Attribute>,
    pub generics: Vec<GenericParam>,
    /// Present for `impl Trait for Type`; `None` for inherent `impl Type`.
    pub trait_path: Option<Path>,
    pub target: TypeExpr,
    pub methods: Vec<FnDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamMode {
    In,
    InOut,
    Out,
    Mut,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnParam {
    pub attrs: Vec<Attribute>,
    pub mode: ParamMode,
    pub name: Ident,
    pub ty: Option<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnDecl {
    pub attrs: Vec<Attribute>,
    pub is_pub: bool,
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub params: Vec<FnParam>,
    pub return_ty: Option<TypeExpr>,
    pub body: Option<Block>,
    pub span: Span,
}

/// Type expression supporting scalars, vectors, matrices, generics, BDA `ptr<T, align = N>`,
/// fixed arrays `[T; N]`, and unsized bindless arrays `Texture2D<f32>[]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExpr {
    Named {
        path: Path,
        args: Vec<TypeArg>,
        unsized_array: bool,
        span: Span,
    },
    Array {
        element: Box<TypeExpr>,
        len: Box<Expr>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeArg {
    Type(TypeExpr),
    NamedExpr {
        name: Ident,
        value: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail_expr: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    Let {
        is_mut: bool,
        name: Ident,
        ty: Option<TypeExpr>,
        init: Option<Expr>,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    For {
        attrs: Vec<Attribute>,
        binding: Ident,
        iter: Expr,
        body: Block,
        span: Span,
    },
    While {
        attrs: Vec<Attribute>,
        cond: Expr,
        body: Block,
        span: Span,
    },
    Loop {
        attrs: Vec<Attribute>,
        body: Block,
        span: Span,
    },
    Switch {
        selector: Expr,
        cases: Vec<SwitchCase>,
        default: Option<Block>,
        span: Span,
    },
    Break(Span),
    Continue(Span),
    Expr {
        expr: Expr,
        has_semi: bool,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwitchCase {
    pub value: Expr,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    AddrOf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    Range,
    RangeInclusive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallArg {
    pub name: Option<Ident>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    IntLit(SmolStr, Span),
    FloatLit(SmolStr, Span),
    BoolLit(bool, Span),
    StringLit(SmolStr, Span),
    NoneLit(Span),
    Path {
        path: Path,
        turbofish: Vec<TypeArg>,
        span: Span,
    },
    ArrayLit {
        elements: Vec<Expr>,
        span: Span,
    },
    StructLit {
        path: Path,
        fields: Vec<(Ident, Expr)>,
        span: Span,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<CallArg>,
        span: Span,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        turbofish: Vec<TypeArg>,
        args: Vec<CallArg>,
        span: Span,
    },
    Field {
        receiver: Box<Expr>,
        field: Ident,
        span: Span,
    },
    Index {
        receiver: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    If {
        cond: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Box<Expr>>,
        span: Span,
    },
    Block(Block),
}

impl Expr {
    pub const fn span(&self) -> Span {
        match self {
            Self::IntLit(_, s)
            | Self::FloatLit(_, s)
            | Self::BoolLit(_, s)
            | Self::StringLit(_, s)
            | Self::NoneLit(s)
            | Self::Path { span: s, .. }
            | Self::ArrayLit { span: s, .. }
            | Self::StructLit { span: s, .. }
            | Self::Unary { span: s, .. }
            | Self::Binary { span: s, .. }
            | Self::Call { span: s, .. }
            | Self::MethodCall { span: s, .. }
            | Self::Field { span: s, .. }
            | Self::Index { span: s, .. }
            | Self::If { span: s, .. } => *s,
            Self::Block(block) => block.span,
        }
    }
}

// ============================================================================
// `pack.vsl` Manifest & Pipeline DSL AST Nodes
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackField {
    pub key: Ident,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackDecl {
    pub id: SmolStr,
    pub fields: Vec<PackField>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingsDecl {
    pub categories: Vec<SettingCategoryDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingCategoryDecl {
    pub name: SmolStr,
    pub entries: Vec<SettingEntryDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettingKind {
    Param,
    Spec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SettingTypeAst {
    Named(Ident),
    Enum { variants: Vec<Ident>, span: Span },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingEntryDecl {
    pub kind: SettingKind,
    pub name: Ident,
    pub ty: SettingTypeAst,
    pub default_value: Expr,
    pub metadata: Vec<PackField>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetsDecl {
    pub presets: Vec<PresetDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetDecl {
    pub name: Ident,
    pub metadata: Vec<PackField>,
    pub assignments: Vec<(Ident, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineModeAst {
    Extend,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineDecl {
    pub mode: PipelineModeAst,
    pub params: Vec<FnParam>,
    pub body: Block,
    pub span: Span,
}
