use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use vsl_mc::{
    AbiProfile, BINDLESS_IMAGE_COUNT, BuiltinFieldType, BuiltinStructDesc, DESCRIPTOR_SET_BINDLESS,
    DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS, DESCRIPTOR_SET_SCENE_GLOBALS,
    EMISSIVE_LIGHT_LAYOUT, EMISSIVE_LIGHT_SSBO_LAYOUT, INSTANCE_META_LAYOUT, ImageFormat,
    RESTIR_RESERVOIR_LAYOUT, RESTIR_RESERVOIR_SSBO_LAYOUT, SHARC_CONFIG_UBO_LAYOUT,
    WORLD_UBO_LAYOUT,
};
use vsl_syntax::ast::{
    Attribute, BinaryOp, Block, CallArg, ConstDecl, Expr, FnDecl, FnParam, GenericParam, Ident,
    ImplBlock, ImportTree, Item, ParamMode, Path, SourceUnit, Stmt, StructDecl, TraitDecl, TypeArg,
    TypeExpr, UnaryOp, WorkgroupVarDecl,
};

/// Maximum Set 0 binding reserved for per-pass targets before optional fallback sampler slots.
const MAX_PASS_TARGET_BINDING: u32 = 14;

/// Set 2 bindless descriptor ABI variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BindlessAbi {
    /// Hyaloclast engine ABI: `sampler2D bindless[4096]` combined image-sampler array.
    #[default]
    Hyaloclast4096,
    /// Upstream VSL ABI: unsized `texture2D bindless[]` separated texture array.
    UpstreamUnsized,
}

/// Lowering options controlling GLSL 460 oracle vs. raster G-Buffer hook bridge behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmitOptions {
    /// When `true`, rewrites pack `param` reads to `VsMotion.packParams.<key>` for
    /// `HookAbiRoute::RasterGbufferGlslBridge` slots.
    pub route_params_to_vs_motion: bool,
    /// Controls whether Set 2 binding 0 emits Hyaloclast's `sampler2D bindless[4096]`
    /// or upstream VSL's unsized `texture2D bindless[]`.
    pub bindless_abi: BindlessAbi,
    /// Target ABI profile (`Hyaloclast` by default, or `Standalone`).
    pub abi: AbiProfile,
}

/// Memory layout mode declared on a struct or buffer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LayoutMode {
    #[default]
    Scalar,
    Std430,
    Std140,
}

impl LayoutMode {
    pub(crate) const fn as_glsl(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Std430 => "std430",
            Self::Std140 => "std140",
        }
    }

    pub(crate) fn from_attr_str(s: &str) -> Option<Self> {
        match s {
            "scalar" => Some(Self::Scalar),
            "std430" => Some(Self::Std430),
            "std140" => Some(Self::Std140),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BdaRefKey {
    pub is_mut: bool,
    pub elem_glsl: String,
    pub align: u32,
}

impl BdaRefKey {
    pub(crate) fn type_name(&self) -> String {
        let access = if self.is_mut { "Rw" } else { "Ro" };
        let sanitized: String = self
            .elem_glsl
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        format!("VslPtr{access}_{sanitized}_a{}", self.align)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MonomorphKey {
    pub struct_name: SmolStr,
    pub type_args: Vec<String>,
}

impl MonomorphKey {
    pub(crate) fn mangled_name(&self) -> String {
        mangle_generic_name(&self.struct_name, &self.type_args)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GenericStructMeta {
    pub generic_param_count: usize,
    pub param_fields: IndexMap<SmolStr, usize>,
}

#[derive(Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct FeatureScan {
    pub extensions: BTreeSet<&'static str>,
    pub bda_refs: BTreeSet<BdaRefKey>,
    pub generic_instantiations: IndexSet<MonomorphKey>,
    pub uses_tlas: bool,
    pub uses_instances: bool,
    pub uses_sharc_config: bool,
    pub uses_lights: bool,
    pub uses_restir_reservoir: bool,
    pub uses_bindless: bool,
    pub uses_sampled_textures: bool,
    pub imported_mc_world: bool,
    pub pack_param_keys: BTreeSet<String>,
    pub pack_param_types: IndexMap<SmolStr, String>,
    pub debug_channels: IndexSet<String>,
    pub next_per_pass_binding: u32,
    pub mc_helpers: BTreeSet<&'static str>,
}

/// Scoped emission context tracking local/parameter GLSL types, BDA metadata, and struct layouts.
#[derive(Clone)]
pub(crate) struct EmitCtx {
    pub opts: EmitOptions,
    pub local_types: IndexMap<SmolStr, String>,
    pub bda_types: IndexMap<String, BdaRefKey>,
    pub struct_fields: IndexMap<SmolStr, IndexMap<SmolStr, String>>,
    pub struct_field_dims: IndexMap<SmolStr, IndexMap<SmolStr, Vec<u64>>>,
    pub struct_layouts: IndexMap<SmolStr, LayoutMode>,
    pub push_constant_instances: IndexMap<SmolStr, String>,
    pub generic_structs: IndexMap<SmolStr, GenericStructMeta>,
    pub fn_returns: IndexMap<SmolStr, String>,
    pub generic_subst: IndexMap<SmolStr, String>,
    pub pack_param_types: IndexMap<SmolStr, String>,
    pub imported_mc_world: bool,
    pub shadowed_world: bool,
    pub tmp_counter: u32,
}

impl EmitCtx {
    pub(crate) fn new(opts: EmitOptions) -> Self {
        Self {
            opts,
            local_types: IndexMap::new(),
            bda_types: IndexMap::new(),
            struct_fields: IndexMap::new(),
            struct_field_dims: IndexMap::new(),
            struct_layouts: IndexMap::new(),
            push_constant_instances: IndexMap::new(),
            generic_structs: IndexMap::new(),
            fn_returns: IndexMap::new(),
            generic_subst: IndexMap::new(),
            pack_param_types: IndexMap::new(),
            imported_mc_world: false,
            shadowed_world: false,
            tmp_counter: 0,
        }
    }

    fn next_tmp(&mut self) -> String {
        let id = self.tmp_counter;
        self.tmp_counter += 1;
        format!("_vsl_tmp_{id}")
    }
}

/// Prefixes GLSL 460 reserved keywords with `_vsl_` so VSL identifiers remain valid GLSL 460.
pub(crate) fn escape_glsl_ident(name: &str) -> String {
    match name {
        "sample" | "input" | "output" | "buffer" | "shared" | "coherent" | "volatile"
        | "restrict" | "readonly" | "writeonly" | "resource" | "patch" | "subroutine"
        | "common" | "partition" | "active" | "asm" | "class" | "union" | "enum" | "typedef"
        | "template" | "this" | "goto" | "inline" | "noinline" | "public" | "static" | "extern"
        | "external" | "interface" | "long" | "short" | "half" | "fixed" | "unsigned"
        | "superp" | "filter" | "sizeof" | "cast" | "namespace" | "using" => {
            format!("_vsl_{name}")
        }
        _ => name.to_owned(),
    }
}

/// Emits a Stage A GLSL 460 oracle shader source string from a parsed VSL 2.0 [`SourceUnit`].
///
/// Translates:
/// - `#[layout(std430)]` / `#[layout(scalar)]` structs and `#[push_constant]` uniform blocks
/// - `mc::*` global scene resources (`WorldUBO`, `tlas`, `InstanceMetaSSBO`, `EmissiveLightSSBO`, `bindless[]`)
/// - `workgroup var` declarations to GLSL `shared` variables
/// - `@stage(compute(x, y, z))` entry points to `layout(local_size_x = ..., ...) in; void main()`
/// - BDA `ptr<T, align = N>` / `ptr_mut<T, align = N>` types to `GL_EXT_buffer_reference2` blocks
/// - `trait` / `impl` blocks to mangled GLSL helper functions
/// - VSL 2.0 statements, structured loops, `switch`, struct literals, and expressions into clean GLSL 460
pub fn emit_glsl_460(unit: &SourceUnit) -> String {
    emit_glsl_460_with_options(unit, EmitOptions::default())
}

/// Emits a Stage A GLSL 460 oracle shader source string with explicit [`EmitOptions`].
pub fn emit_glsl_460_with_options(unit: &SourceUnit, opts: EmitOptions) -> String {
    let mut scan = FeatureScan {
        generic_instantiations: collect_generic_instantiations(unit),
        ..FeatureScan::default()
    };
    scan_unit_with_options(unit, opts, &mut scan);
    // Canonical `WorldUBO` carries `uint64_t pack_params_bda` for every entry point.
    scan.extensions
        .insert("GL_EXT_shader_explicit_arithmetic_types_int64");

    let mut ctx = EmitCtx::new(opts);
    ctx.imported_mc_world = scan.imported_mc_world;
    for bda in &scan.bda_refs {
        ctx.bda_types.insert(bda.type_name(), bda.clone());
    }
    seed_mc_builtin_types(&mut ctx);

    // Pre-collect struct field types, dimensions, layouts, generic struct metadata,
    // push-constant instance names, constants, workgroup vars, and function return types.
    for item in &unit.items {
        match item {
            Item::Const(c) => {
                ctx.local_types
                    .insert(c.name.name.clone(), type_to_glsl(&c.ty).0);
            }
            Item::WorkgroupVar(w) => {
                ctx.local_types
                    .insert(w.name.name.clone(), type_to_glsl(&w.ty).0);
            }
            Item::Struct(s) => {
                let layout_mode = extract_layout_attr(&s.attrs)
                    .and_then(LayoutMode::from_attr_str)
                    .unwrap_or(LayoutMode::Scalar);
                if s.generics.is_empty() {
                    let mut fields = IndexMap::new();
                    let mut dims = IndexMap::new();
                    for f in &s.fields {
                        fields.insert(f.name.name.clone(), type_to_glsl(&f.ty).0);
                        let field_dims = extract_array_dims(&f.ty);
                        if !field_dims.is_empty() {
                            dims.insert(f.name.name.clone(), field_dims);
                        }
                    }
                    ctx.struct_fields.insert(s.name.name.clone(), fields);
                    ctx.struct_field_dims.insert(s.name.name.clone(), dims);
                    ctx.struct_layouts.insert(s.name.name.clone(), layout_mode);
                } else {
                    let generic_params: Vec<SmolStr> =
                        s.generics.iter().map(|g| g.name.name.clone()).collect();
                    let mut param_fields = IndexMap::new();
                    for f in &s.fields {
                        if let TypeExpr::Named { path, args, .. } = &f.ty
                            && path.segments.len() == 1
                            && args.is_empty()
                            && let Some(idx) = generic_params
                                .iter()
                                .position(|gp| gp == &path.segments[0].name)
                        {
                            param_fields.insert(f.name.name.clone(), idx);
                        }
                    }
                    ctx.generic_structs.insert(
                        s.name.name.clone(),
                        GenericStructMeta {
                            generic_param_count: generic_params.len(),
                            param_fields,
                        },
                    );
                    for key in scan
                        .generic_instantiations
                        .iter()
                        .filter(|k| k.struct_name == s.name.name)
                    {
                        let mut subst = IndexMap::new();
                        for (g, concrete) in s.generics.iter().zip(&key.type_args) {
                            subst.insert(g.name.name.clone(), concrete.clone());
                        }
                        let mut fields = IndexMap::new();
                        let mut dims = IndexMap::new();
                        for f in &s.fields {
                            fields.insert(
                                f.name.name.clone(),
                                type_to_glsl_with_subst(&f.ty, &subst).0,
                            );
                            let field_dims = extract_array_dims(&f.ty);
                            if !field_dims.is_empty() {
                                dims.insert(f.name.name.clone(), field_dims);
                            }
                        }
                        let mangled = SmolStr::new(key.mangled_name());
                        ctx.struct_fields.insert(mangled.clone(), fields);
                        ctx.struct_field_dims.insert(mangled.clone(), dims);
                        ctx.struct_layouts.insert(mangled, layout_mode);
                    }
                }
            }
            Item::Function(f) => {
                ctx.fn_returns
                    .insert(f.name.name.clone(), fn_return_glsl_type(f));
                if extract_entry_stage(&f.attrs).is_some() {
                    for p in &f.params {
                        if has_attr(&p.attrs, "push_constant")
                            && let Some(ref ty) = p.ty
                        {
                            let ty_name = type_to_glsl(ty).0;
                            ctx.push_constant_instances
                                .insert(SmolStr::new(ty_name), escape_glsl_ident(&p.name.name));
                        }
                    }
                }
            }
            Item::Impl(imp) => {
                if imp.generics.is_empty() {
                    let empty = IndexMap::new();
                    let target_name = impl_target_name_with_subst(&imp.target, &empty);
                    for m in &imp.methods {
                        let mangled = SmolStr::new(format!("{target_name}_{}", m.name.name));
                        let ret = fn_return_glsl_type(m);
                        ctx.fn_returns.insert(mangled, ret.clone());
                        ctx.fn_returns.entry(m.name.name.clone()).or_insert(ret);
                    }
                } else if let Some(base_name) = impl_target_base_name(&imp.target) {
                    for key in scan
                        .generic_instantiations
                        .iter()
                        .filter(|k| k.struct_name.as_str() == base_name)
                    {
                        let mut subst = IndexMap::new();
                        apply_impl_subst(imp, &key.type_args, &mut subst);
                        let target_name = impl_target_name_with_subst(&imp.target, &subst);
                        for m in &imp.methods {
                            let mangled = SmolStr::new(format!("{target_name}_{}", m.name.name));
                            let ret = fn_return_glsl_type_with_subst(m, &subst);
                            ctx.fn_returns.insert(mangled, ret.clone());
                            ctx.fn_returns.entry(m.name.name.clone()).or_insert(ret);
                        }
                    }
                }
            }
            Item::Trait(t) => {
                for m in &t.methods {
                    ctx.fn_returns
                        .entry(m.name.name.clone())
                        .or_insert_with(|| fn_return_glsl_type(m));
                }
            }
            _ => {}
        }
    }

    // Require `GL_EXT_scalar_block_layout` when any BDA reference uses `scalar` layout.
    for bda in &scan.bda_refs {
        let layout = ctx
            .struct_layouts
            .get(bda.elem_glsl.as_str())
            .copied()
            .unwrap_or(LayoutMode::Scalar);
        if layout == LayoutMode::Scalar {
            scan.extensions.insert("GL_EXT_scalar_block_layout");
        }
    }

    let mut out = String::with_capacity(2048);
    out.push_str("#version 460 core\n");

    for ext in &scan.extensions {
        let _ = writeln!(out, "#extension {ext} : require");
    }

    if let Some(ref module_decl) = unit.module_decl {
        let _ = writeln!(out, "// module {}", module_decl.path.joined());
    }
    out.push('\n');

    // 0. Global `mc::*` scene & bindless resource declarations.
    emit_mc_global_decls(&scan, opts, &mut out);

    // 1. Constants.
    for item in &unit.items {
        if let Item::Const(c) = item {
            emit_const_decl(c, &ctx, &mut out);
        }
    }

    // 2. Concrete structs / push-constant blocks first, then monomorphized generic structs.
    for item in &unit.items {
        if let Item::Struct(s) = item
            && s.generics.is_empty()
        {
            emit_struct_decl(s, &scan.generic_instantiations, &mut ctx, &mut out);
        }
    }
    for item in &unit.items {
        if let Item::Struct(s) = item
            && !s.generics.is_empty()
        {
            emit_struct_decl(s, &scan.generic_instantiations, &mut ctx, &mut out);
        }
    }

    // 3. BDA `buffer_reference` declarations (emitted after structs so struct element types resolve).
    for bda in &scan.bda_refs {
        let qualifier = if bda.is_mut {
            "buffer"
        } else {
            "readonly buffer"
        };
        let layout_qual = ctx
            .struct_layouts
            .get(bda.elem_glsl.as_str())
            .copied()
            .unwrap_or(LayoutMode::Scalar)
            .as_glsl();
        let _ = writeln!(
            out,
            "layout(buffer_reference, {layout_qual}, buffer_reference_align = {}) {qualifier} {} {{",
            bda.align,
            bda.type_name()
        );
        let _ = writeln!(out, "    {} value;", bda.elem_glsl);
        out.push_str("};\n\n");
    }

    // 4. Workgroup shared variables.
    for item in &unit.items {
        if let Item::WorkgroupVar(w) = item {
            emit_workgroup_var(w, &mut out);
        }
    }

    // 5. Trait interfaces and `impl` methods (concrete impls before monomorphized generic impls
    //    so trait method definitions precede monomorphized callers in GLSL).
    for item in &unit.items {
        if let Item::Trait(t) = item {
            emit_trait_decl(t, &mut out);
        }
    }
    for item in &unit.items {
        if let Item::Impl(imp) = item
            && imp.generics.is_empty()
        {
            emit_impl_block(imp, &scan.generic_instantiations, &mut ctx, &mut out);
        }
    }
    for item in &unit.items {
        if let Item::Impl(imp) = item
            && !imp.generics.is_empty()
        {
            emit_impl_block(imp, &scan.generic_instantiations, &mut ctx, &mut out);
        }
    }

    // 6. Functions and `@stage(compute(x, y, z))` entry points.
    for item in &unit.items {
        if let Item::Function(f) = item {
            emit_fn_item(f, &mut ctx, &mut out);
        }
    }

    out
}

fn builtin_field_glsl_type(ty: BuiltinFieldType) -> (String, &'static str) {
    match ty {
        BuiltinFieldType::Bool => ("bool".to_owned(), ""),
        BuiltinFieldType::I32 => ("int".to_owned(), ""),
        BuiltinFieldType::U32 => ("uint".to_owned(), ""),
        BuiltinFieldType::I64 => ("int64_t".to_owned(), ""),
        BuiltinFieldType::U64 => ("uint64_t".to_owned(), ""),
        BuiltinFieldType::F32 => ("float".to_owned(), ""),
        BuiltinFieldType::Vec2 => ("vec2".to_owned(), ""),
        BuiltinFieldType::Vec3 => ("vec3".to_owned(), ""),
        BuiltinFieldType::Vec4 => ("vec4".to_owned(), ""),
        BuiltinFieldType::UVec2 => ("uvec2".to_owned(), ""),
        BuiltinFieldType::UVec3 => ("uvec3".to_owned(), ""),
        BuiltinFieldType::UVec4 => ("uvec4".to_owned(), ""),
        BuiltinFieldType::Mat4 => ("mat4".to_owned(), ""),
        BuiltinFieldType::RuntimeArrayStruct(elem) => (elem.to_owned(), "[]"),
    }
}

fn seed_builtin_struct(desc: &BuiltinStructDesc, layout: LayoutMode, ctx: &mut EmitCtx) {
    let mut fields = IndexMap::new();
    for field in desc.fields {
        let (glsl_ty, _) = builtin_field_glsl_type(field.ty);
        fields.insert(SmolStr::new(field.name), glsl_ty);
    }
    let name = SmolStr::new(desc.name);
    ctx.struct_fields.insert(name.clone(), fields);
    ctx.struct_layouts.insert(name, layout);
}

fn emit_builtin_fields(desc: &BuiltinStructDesc, out: &mut String) {
    for field in desc.fields {
        let (glsl_ty, suffix) = builtin_field_glsl_type(field.ty);
        let _ = writeln!(out, "    {glsl_ty} {}{suffix};", field.name);
    }
}

pub(crate) fn seed_mc_builtin_types(ctx: &mut EmitCtx) {
    seed_builtin_struct(&WORLD_UBO_LAYOUT, LayoutMode::Std140, ctx);
    seed_builtin_struct(&INSTANCE_META_LAYOUT, LayoutMode::Std430, ctx);
    seed_builtin_struct(&SHARC_CONFIG_UBO_LAYOUT, LayoutMode::Std140, ctx);
    seed_builtin_struct(&EMISSIVE_LIGHT_LAYOUT, LayoutMode::Std430, ctx);
    seed_builtin_struct(&EMISSIVE_LIGHT_SSBO_LAYOUT, LayoutMode::Std430, ctx);
    seed_builtin_struct(&RESTIR_RESERVOIR_LAYOUT, LayoutMode::Std430, ctx);
    seed_builtin_struct(&RESTIR_RESERVOIR_SSBO_LAYOUT, LayoutMode::Std430, ctx);

    ctx.local_types
        .insert(SmolStr::new_static("mc_world"), "WorldUBO".to_owned());
    ctx.local_types
        .insert(SmolStr::new_static("world"), "WorldUBO".to_owned());
    ctx.local_types.insert(
        SmolStr::new_static("tlas"),
        "accelerationStructureEXT".to_owned(),
    );
    ctx.local_types.insert(
        SmolStr::new_static("mc_tlas"),
        "accelerationStructureEXT".to_owned(),
    );
    ctx.local_types.insert(
        SmolStr::new_static("mc_instances"),
        "InstanceMeta".to_owned(),
    );
    ctx.local_types.insert(
        SmolStr::new_static("mc_sharc_config"),
        "SharcConfigUBO".to_owned(),
    );
    ctx.local_types.insert(
        SmolStr::new_static("mc_lights"),
        "EmissiveLightSSBO".to_owned(),
    );
    ctx.local_types.insert(
        SmolStr::new_static("mc_restir_reservoir"),
        "RestirReservoirSSBO".to_owned(),
    );
    let bindless_ty = match ctx.opts.bindless_abi {
        BindlessAbi::Hyaloclast4096 => "sampler2D",
        BindlessAbi::UpstreamUnsized => "texture2D",
    };
    ctx.local_types
        .insert(SmolStr::new_static("bindless"), bindless_ty.to_owned());
    ctx.local_types
        .insert(SmolStr::new_static("mc_bindless"), bindless_ty.to_owned());
}

fn emit_mc_global_decls(scan: &FeatureScan, opts: EmitOptions, out: &mut String) {
    // Set 0, binding 0: Canonical `WorldUBO` (`std140`, 544 bytes) from `vsl_mc::layouts`.
    let emit_world_ubo = opts.abi == AbiProfile::Hyaloclast
        || scan.imported_mc_world
        || !scan.pack_param_keys.is_empty();
    let mut emitted_any = emit_world_ubo;
    if emit_world_ubo {
        let world_set = WORLD_UBO_LAYOUT.set.unwrap_or(DESCRIPTOR_SET_PER_PASS);
        let world_binding = WORLD_UBO_LAYOUT.binding.unwrap_or(0);
        let _ = writeln!(
            out,
            "layout(set = {world_set}, binding = {world_binding}, std140) uniform {} {{",
            WORLD_UBO_LAYOUT.name
        );
        emit_builtin_fields(&WORLD_UBO_LAYOUT, out);
        out.push_str("} mc_world;\n\n");
    }
    // Set 0, ascending bindings after pass resource parameters: debug output images from `mc::debug_write(...)`.
    let mut next_free_binding = if emit_world_ubo {
        scan.next_per_pass_binding.max(1)
    } else {
        scan.next_per_pass_binding
    };
    for channel in &scan.debug_channels {
        let _ = writeln!(
            out,
            "layout(set = {DESCRIPTOR_SET_PER_PASS}, binding = {next_free_binding}, rgba16f) writeonly uniform image2D vsl_debug_{channel};"
        );
        next_free_binding += 1;
        emitted_any = true;
    }
    if !scan.debug_channels.is_empty() {
        out.push('\n');
    }
    // Set 0 default sampler: emitted only when separated textures are actually sampled,
    // placed at `max(next_free_binding, MAX_PASS_TARGET_BINDING)` so it never collides with
    // per-pass resource parameters or debug channels.
    if scan.uses_sampled_textures {
        let sampler_binding = next_free_binding.max(MAX_PASS_TARGET_BINDING);
        let _ = writeln!(
            out,
            "layout(set = {DESCRIPTOR_SET_PER_PASS}, binding = {sampler_binding}) uniform sampler vsl_default_sampler;\n"
        );
        emitted_any = true;
    }
    // Set 1, binding 0: TLAS.
    if scan.uses_tlas {
        let _ = writeln!(
            out,
            "layout(set = {DESCRIPTOR_SET_SCENE_GLOBALS}, binding = 0) uniform accelerationStructureEXT tlas;\n"
        );
        emitted_any = true;
    }
    // Set 1, binding 1: Canonical `InstanceMetaSSBO` (`std430`).
    if scan.uses_instances {
        let _ = writeln!(out, "struct {} {{", INSTANCE_META_LAYOUT.name);
        emit_builtin_fields(&INSTANCE_META_LAYOUT, out);
        out.push_str("};\n");
        let _ = writeln!(
            out,
            "layout(set = {DESCRIPTOR_SET_SCENE_GLOBALS}, binding = 1, std430) readonly buffer InstanceMetaSSBO {{"
        );
        out.push_str("    InstanceMeta mc_instances[];\n");
        out.push_str("};\n\n");
        emitted_any = true;
    }
    // Set 1, binding 2: Canonical `SharcConfigUBO` (`std140`, 48 bytes).
    if scan.uses_sharc_config {
        let sharc_set = SHARC_CONFIG_UBO_LAYOUT
            .set
            .unwrap_or(DESCRIPTOR_SET_SCENE_GLOBALS);
        let sharc_binding = SHARC_CONFIG_UBO_LAYOUT.binding.unwrap_or(2);
        let _ = writeln!(
            out,
            "layout(set = {sharc_set}, binding = {sharc_binding}, std140) uniform {} {{",
            SHARC_CONFIG_UBO_LAYOUT.name
        );
        emit_builtin_fields(&SHARC_CONFIG_UBO_LAYOUT, out);
        out.push_str("} mc_sharc_config;\n\n");
        emitted_any = true;
    }
    // Set 1, binding 3: Canonical `EmissiveLightSSBO` (`std430`).
    if scan.uses_lights {
        let _ = writeln!(out, "struct {} {{", EMISSIVE_LIGHT_LAYOUT.name);
        emit_builtin_fields(&EMISSIVE_LIGHT_LAYOUT, out);
        out.push_str("};\n");
        let lights_set = EMISSIVE_LIGHT_SSBO_LAYOUT
            .set
            .unwrap_or(DESCRIPTOR_SET_SCENE_GLOBALS);
        let lights_binding = EMISSIVE_LIGHT_SSBO_LAYOUT.binding.unwrap_or(3);
        let _ = writeln!(
            out,
            "layout(set = {lights_set}, binding = {lights_binding}, std430) readonly buffer {} {{",
            EMISSIVE_LIGHT_SSBO_LAYOUT.name
        );
        emit_builtin_fields(&EMISSIVE_LIGHT_SSBO_LAYOUT, out);
        out.push_str("} mc_lights;\n\n");
        emitted_any = true;
    }
    // Set 1, binding 4: Canonical `RestirReservoirSSBO` (`std430`).
    if scan.uses_restir_reservoir {
        let _ = writeln!(out, "struct {} {{", RESTIR_RESERVOIR_LAYOUT.name);
        emit_builtin_fields(&RESTIR_RESERVOIR_LAYOUT, out);
        out.push_str("};\n");
        let restir_set = RESTIR_RESERVOIR_SSBO_LAYOUT
            .set
            .unwrap_or(DESCRIPTOR_SET_SCENE_GLOBALS);
        let restir_binding = RESTIR_RESERVOIR_SSBO_LAYOUT.binding.unwrap_or(4);
        let _ = writeln!(
            out,
            "layout(set = {restir_set}, binding = {restir_binding}, std430) buffer {} {{",
            RESTIR_RESERVOIR_SSBO_LAYOUT.name
        );
        emit_builtin_fields(&RESTIR_RESERVOIR_SSBO_LAYOUT, out);
        out.push_str("} mc_restir_reservoir;\n\n");
        emitted_any = true;
    }
    // Set 2, binding 0: Global bindless descriptor array (Hyaloclast 4096 or upstream unsized).
    if scan.uses_bindless {
        match opts.bindless_abi {
            BindlessAbi::Hyaloclast4096 => {
                let _ = writeln!(
                    out,
                    "layout(set = {DESCRIPTOR_SET_BINDLESS}, binding = 0) uniform sampler2D bindless[{BINDLESS_IMAGE_COUNT}];\n"
                );
            }
            BindlessAbi::UpstreamUnsized => {
                let _ = writeln!(
                    out,
                    "layout(set = {DESCRIPTOR_SET_BINDLESS}, binding = 0) uniform texture2D bindless[];\n"
                );
            }
        }
        emitted_any = true;
    }
    // Forward declarations for referenced `mc::*` utility helpers.
    if scan.mc_helpers.contains("mc_reconstruct_world_pos") {
        out.push_str("vec3 mc_reconstruct_world_pos(ivec2 coord, texture2D depth_tex);\n");
        emitted_any = true;
    }
    if scan.mc_helpers.contains("mc_view_dir") {
        out.push_str("vec3 mc_view_dir(ivec2 coord);\n");
        emitted_any = true;
    }
    if scan.mc_helpers.contains("mc_sample_sky") {
        out.push_str("vec3 mc_sample_sky(vec3 dir);\n");
        emitted_any = true;
    }
    if emitted_any {
        out.push('\n');
    }
}

/// Scans a single [`FnDecl`] for required GLSL extensions and referenced pack parameter keys,
/// including AST-based pack parameter type inference.
pub(crate) fn scan_fn_decl(func: &FnDecl) -> FeatureScan {
    let mut scan = scan_fn_decl_without_param_inference(func);
    infer_pack_param_types(func, &mut scan);
    scan
}

/// Scans a single [`FnDecl`] for required GLSL extensions and referenced pack parameter keys
/// without AST-based pack parameter type inference (used when a `pack.vsl` schema supplies types).
pub(crate) fn scan_fn_decl_without_param_inference(func: &FnDecl) -> FeatureScan {
    let mut scan = FeatureScan::default();
    scan_fn_into(func, &mut scan);
    scan
}

pub(crate) fn emit_fn_decl_with_ctx(
    func: &FnDecl,
    self_ty: Option<&str>,
    ctx: &mut EmitCtx,
    out: &mut String,
) {
    let saved_locals = ctx.local_types.clone();
    let saved_shadowed_world = ctx.shadowed_world;

    for param in &func.params {
        if param.name.name == "world" {
            ctx.shadowed_world = true;
        }
        let ty_str = if param.name.name == "self" && param.ty.is_none() {
            self_ty.unwrap_or("float").to_owned()
        } else {
            param.ty.as_ref().map_or_else(
                || "float".to_owned(),
                |t| type_to_glsl_with_subst(t, &ctx.generic_subst).0,
            )
        };
        ctx.local_types.insert(param.name.name.clone(), ty_str);
    }

    let fn_name = self_ty.map_or_else(
        || escape_glsl_ident(&func.name.name),
        |t| format!("{t}_{}", func.name.name),
    );
    let ret_ty = fn_return_glsl_type_with_subst(func, &ctx.generic_subst);
    let params = format_param_list_with_self(&func.params, self_ty, &ctx.generic_subst);
    let _ = writeln!(out, "{ret_ty} {fn_name}({params}) {{");
    if let Some(ref body) = func.body {
        emit_block_contents(body, 1, ret_ty != "void", ctx, out);
    }
    out.push_str("}\n");

    ctx.local_types = saved_locals;
    ctx.shadowed_world = saved_shadowed_world;
}

/// Formats a VSL parameter list into GLSL 460 parameter syntax.
pub(crate) fn format_param_list(params: &[FnParam]) -> String {
    let empty = IndexMap::new();
    format_param_list_with_self(params, None, &empty)
}

fn format_param_list_with_self(
    params: &[FnParam],
    self_ty: Option<&str>,
    generic_subst: &IndexMap<SmolStr, String>,
) -> String {
    params
        .iter()
        .map(|p| {
            let mode = match p.mode {
                ParamMode::In => "in ",
                ParamMode::InOut | ParamMode::Mut => "inout ",
                ParamMode::Out => "out ",
            };
            let (ty_str, arr_suffix) = if p.name.name == "self" && p.ty.is_none() {
                (self_ty.unwrap_or("float").to_owned(), String::new())
            } else {
                p.ty.as_ref().map_or_else(
                    || ("float".to_owned(), String::new()),
                    |t| type_to_glsl_with_subst(t, generic_subst),
                )
            };
            let pname = escape_glsl_ident(&p.name.name);
            format!("{mode}{ty_str} {pname}{arr_suffix}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Returns the GLSL 460 return type string for a [`FnDecl`].
pub(crate) fn fn_return_glsl_type(func: &FnDecl) -> String {
    let empty = IndexMap::new();
    fn_return_glsl_type_with_subst(func, &empty)
}

fn fn_return_glsl_type_with_subst(
    func: &FnDecl,
    generic_subst: &IndexMap<SmolStr, String>,
) -> String {
    func.return_ty.as_ref().map_or_else(
        || "void".to_owned(),
        |ty| type_to_glsl_with_subst(ty, generic_subst).0,
    )
}

fn mangle_generic_name(base: &str, type_args: &[String]) -> String {
    if type_args.is_empty() {
        return base.to_owned();
    }
    let suffix = type_args
        .iter()
        .map(|arg| sanitize_glsl_ident(arg))
        .collect::<Vec<_>>()
        .join("__");
    format!("{base}__{suffix}")
}

fn format_generic_params(generics: &[GenericParam]) -> String {
    generics
        .iter()
        .map(|g| match &g.bound {
            Some(bound) => format!("{}: {}", g.name.name, bound.joined()),
            None => g.name.name.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_type_expr_comment(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Array { element, .. } => format!("[{}]", format_type_expr_comment(element)),
        TypeExpr::Named { path, args, .. } => {
            let base = path.joined();
            if args.is_empty() {
                base
            } else {
                let rendered: Vec<String> = args
                    .iter()
                    .filter_map(|a| match a {
                        TypeArg::Type(t) => Some(format_type_expr_comment(t)),
                        TypeArg::NamedExpr { name, .. } => Some(name.name.to_string()),
                    })
                    .collect();
                format!("{base}<{}>", rendered.join(", "))
            }
        }
    }
}

fn impl_target_base_name(target: &TypeExpr) -> Option<String> {
    match target {
        TypeExpr::Named { path, .. } => Some(path.joined().replace("::", "_")),
        TypeExpr::Array { .. } => None,
    }
}

fn impl_target_name_with_subst(
    target: &TypeExpr,
    generic_subst: &IndexMap<SmolStr, String>,
) -> String {
    match target {
        TypeExpr::Named { .. } => type_to_glsl_with_subst(target, generic_subst).0,
        TypeExpr::Array { .. } => "Array".to_owned(),
    }
}

fn apply_impl_subst(imp: &ImplBlock, type_args: &[String], subst: &mut IndexMap<SmolStr, String>) {
    let mut bound_from_target = false;
    if let TypeExpr::Named { args, .. } = &imp.target {
        let target_type_args: Vec<&TypeExpr> = args
            .iter()
            .filter_map(|a| match a {
                TypeArg::Type(t) => Some(t),
                TypeArg::NamedExpr { .. } => None,
            })
            .collect();
        if target_type_args.len() == type_args.len() {
            for (target_arg, concrete) in target_type_args.iter().zip(type_args) {
                if let TypeExpr::Named {
                    path,
                    args: inner_args,
                    ..
                } = target_arg
                    && path.segments.len() == 1
                    && inner_args.is_empty()
                {
                    let param_name = &path.segments[0].name;
                    if imp.generics.iter().any(|g| &g.name.name == param_name) {
                        subst.insert(param_name.clone(), concrete.clone());
                        bound_from_target = true;
                    }
                }
            }
        }
    }
    if !bound_from_target {
        for (g, concrete) in imp.generics.iter().zip(type_args) {
            subst.insert(g.name.name.clone(), concrete.clone());
        }
    }
}

fn emit_trait_decl(decl: &TraitDecl, out: &mut String) {
    let _ = writeln!(out, "// trait {}", decl.name.name);
    let empty = IndexMap::new();
    for method in &decl.methods {
        let ret = fn_return_glsl_type(method);
        let params = format_param_list_with_self(&method.params, Some("Self"), &empty);
        let _ = writeln!(out, "//   {ret} {}({params});", method.name.name);
    }
    out.push('\n');
}

fn emit_impl_block(
    imp: &ImplBlock,
    instantiations: &IndexSet<MonomorphKey>,
    ctx: &mut EmitCtx,
    out: &mut String,
) {
    if !imp.generics.is_empty() {
        let base_name = impl_target_base_name(&imp.target);
        let matching: Vec<&MonomorphKey> = base_name.as_deref().map_or_else(Vec::new, |base| {
            instantiations
                .iter()
                .filter(|k| k.struct_name == base)
                .collect()
        });
        if matching.is_empty() {
            let gen_list = format_generic_params(&imp.generics);
            let target_desc = format_type_expr_comment(&imp.target);
            if let Some(ref trait_path) = imp.trait_path {
                let _ = writeln!(
                    out,
                    "// generic impl<{gen_list}> {} for {target_desc} (no concrete instantiations in this unit)\n",
                    trait_path.joined()
                );
            } else {
                let _ = writeln!(
                    out,
                    "// generic impl<{gen_list}> {target_desc} (no concrete instantiations in this unit)\n"
                );
            }
            return;
        }
        for key in matching {
            let saved_subst = ctx.generic_subst.clone();
            apply_impl_subst(imp, &key.type_args, &mut ctx.generic_subst);
            let target_name = impl_target_name_with_subst(&imp.target, &ctx.generic_subst);
            if let Some(ref trait_path) = imp.trait_path {
                let _ = writeln!(out, "// impl {} for {target_name}", trait_path.joined());
            } else {
                let _ = writeln!(out, "// impl {target_name}");
            }
            for method in &imp.methods {
                emit_fn_decl_with_ctx(method, Some(&target_name), ctx, out);
                out.push('\n');
            }
            ctx.generic_subst = saved_subst;
        }
        return;
    }

    let empty = IndexMap::new();
    let target_name = impl_target_name_with_subst(&imp.target, &empty);
    if let Some(ref trait_path) = imp.trait_path {
        let _ = writeln!(out, "// impl {} for {target_name}", trait_path.joined());
    } else {
        let _ = writeln!(out, "// impl {target_name}");
    }
    for method in &imp.methods {
        emit_fn_decl_with_ctx(method, Some(&target_name), ctx, out);
        out.push('\n');
    }
}

pub(crate) fn emit_const_decl(decl: &ConstDecl, ctx: &EmitCtx, out: &mut String) {
    let (ty_str, arr_suffix) = type_to_glsl(&decl.ty);
    let val_str = emit_expr_const(&decl.value, ctx);
    let name = escape_glsl_ident(&decl.name.name);
    let _ = writeln!(out, "const {ty_str} {name}{arr_suffix} = {val_str};\n");
}

fn emit_workgroup_var(decl: &WorkgroupVarDecl, out: &mut String) {
    let (ty_str, arr_suffix) = type_to_glsl(&decl.ty);
    let name = escape_glsl_ident(&decl.name.name);
    let _ = writeln!(out, "shared {ty_str} {name}{arr_suffix};");
}

fn emit_struct_decl(
    decl: &StructDecl,
    instantiations: &IndexSet<MonomorphKey>,
    ctx: &mut EmitCtx,
    out: &mut String,
) {
    if !decl.generics.is_empty() {
        let matching: Vec<&MonomorphKey> = instantiations
            .iter()
            .filter(|k| k.struct_name == decl.name.name)
            .collect();
        if matching.is_empty() {
            if let Some(layout) = extract_layout_attr(&decl.attrs) {
                let _ = writeln!(out, "// layout({layout})");
            }
            let gen_list = format_generic_params(&decl.generics);
            let _ = writeln!(
                out,
                "// generic struct {}<{gen_list}> (no concrete instantiations in this unit)\n",
                decl.name.name
            );
            return;
        }
        for key in matching {
            let saved_subst = ctx.generic_subst.clone();
            for (g, concrete) in decl.generics.iter().zip(&key.type_args) {
                ctx.generic_subst
                    .insert(g.name.name.clone(), concrete.clone());
            }
            let mangled = key.mangled_name();
            emit_concrete_struct_decl(decl, &mangled, ctx, out);
            ctx.generic_subst = saved_subst;
        }
        return;
    }

    emit_concrete_struct_decl(decl, &decl.name.name, ctx, out);
}

pub(crate) fn emit_concrete_struct_decl(
    decl: &StructDecl,
    glsl_name: &str,
    ctx: &EmitCtx,
    out: &mut String,
) {
    let is_push_constant = has_attr(&decl.attrs, "push_constant");
    let layout_mode = extract_layout_attr(&decl.attrs);

    if is_push_constant {
        let layout_qual = layout_mode.unwrap_or("scalar");
        let _ = writeln!(
            out,
            "layout(push_constant, {layout_qual}) uniform {glsl_name} {{"
        );
        for field in &decl.fields {
            let (ty_str, arr_suffix) = type_to_glsl_with_subst(&field.ty, &ctx.generic_subst);
            let fname = escape_glsl_ident(&field.name.name);
            let _ = writeln!(out, "    {ty_str} {fname}{arr_suffix};");
        }
        if let Some(inst_name) = ctx.push_constant_instances.get(glsl_name) {
            let _ = writeln!(out, "}} {inst_name};\n");
        } else {
            out.push_str("};\n\n");
        }
        return;
    }

    if let Some(layout) = layout_mode {
        let _ = writeln!(out, "// layout({layout})");
    }
    let _ = writeln!(out, "struct {glsl_name} {{");
    for field in &decl.fields {
        let (ty_str, arr_suffix) = type_to_glsl_with_subst(&field.ty, &ctx.generic_subst);
        let fname = escape_glsl_ident(&field.name.name);
        let _ = writeln!(out, "    {ty_str} {fname}{arr_suffix};");
    }
    out.push_str("};\n\n");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryStage {
    Compute(u32, u32, u32),
    Vertex,
    Fragment,
}

fn emit_fn_item(func: &FnDecl, ctx: &mut EmitCtx, out: &mut String) {
    if let Some(stage) = extract_entry_stage(&func.attrs) {
        let saved_locals = ctx.local_types.clone();
        let saved_shadowed_world = ctx.shadowed_world;

        let world_reserved = ctx.opts.abi == AbiProfile::Hyaloclast || ctx.imported_mc_world;
        let mut binding_slot = u32::from(world_reserved);
        let mut pack_resource_slot = 0u32;
        let mut builtin_inits: Vec<(String, String, Option<&'static str>)> = Vec::new();
        let mut builtin_writebacks: Vec<(&'static str, String)> = Vec::new();

        for param in &func.params {
            if let Some(builtin) = extract_builtin_attr(&param.attrs) {
                let (ty_str, _) = param
                    .ty
                    .as_ref()
                    .map_or_else(|| ("uvec3".to_owned(), String::new()), type_to_glsl);
                ctx.local_types
                    .insert(param.name.name.clone(), ty_str.clone());
                if let Some(gl_builtin) = map_builtin_var(builtin) {
                    let local_name = escape_glsl_ident(&param.name.name);
                    match param.mode {
                        ParamMode::Out => {
                            builtin_inits.push((ty_str, local_name.clone(), None));
                            builtin_writebacks.push((gl_builtin, local_name));
                        }
                        ParamMode::InOut | ParamMode::Mut => {
                            builtin_inits.push((ty_str, local_name.clone(), Some(gl_builtin)));
                            builtin_writebacks.push((gl_builtin, local_name));
                        }
                        ParamMode::In => {
                            builtin_inits.push((ty_str, local_name, Some(gl_builtin)));
                        }
                    }
                }
            } else if let Some(location) = extract_location_attr(&param.attrs) {
                if let Some(ref ty) = param.ty {
                    let (ty_str, arr_suffix) = type_to_glsl(ty);
                    ctx.local_types
                        .insert(param.name.name.clone(), ty_str.clone());
                    let escaped_name = escape_glsl_ident(&param.name.name);
                    let dir = match param.mode {
                        ParamMode::Out | ParamMode::InOut | ParamMode::Mut => "out",
                        ParamMode::In => "in",
                    };
                    let _ = writeln!(
                        out,
                        "layout(location = {location}) {dir} {ty_str} {escaped_name}{arr_suffix};"
                    );
                }
            } else if has_attr(&param.attrs, "push_constant") {
                if let Some(ref ty) = param.ty {
                    ctx.local_types
                        .insert(param.name.name.clone(), type_to_glsl(ty).0);
                }
            } else if let Some((set, binding)) = extract_binding_attr(&param.attrs) {
                if let Some(ref ty) = param.ty {
                    ctx.local_types
                        .insert(param.name.name.clone(), type_to_glsl(ty).0);
                    let escaped_name = escape_glsl_ident(&param.name.name);
                    emit_entry_resource_binding(set, binding, &escaped_name, ty, ctx, out);
                }
            } else if has_attr(&param.attrs, "pack_resource") {
                if let Some(ref ty) = param.ty {
                    ctx.local_types
                        .insert(param.name.name.clone(), type_to_glsl(ty).0);
                    let escaped_name = escape_glsl_ident(&param.name.name);
                    emit_entry_resource_binding(
                        DESCRIPTOR_SET_PACK_RESOURCES,
                        pack_resource_slot,
                        &escaped_name,
                        ty,
                        ctx,
                        out,
                    );
                    pack_resource_slot += 1;
                }
            } else if let Some(ref ty) = param.ty {
                ctx.local_types
                    .insert(param.name.name.clone(), type_to_glsl(ty).0);
                let escaped_name = escape_glsl_ident(&param.name.name);
                emit_entry_resource_binding(
                    DESCRIPTOR_SET_PER_PASS,
                    binding_slot,
                    &escaped_name,
                    ty,
                    ctx,
                    out,
                );
                binding_slot += 1;
            }
        }

        match stage {
            EntryStage::Compute(lx, ly, lz) => {
                let _ = writeln!(
                    out,
                    "layout(local_size_x = {lx}, local_size_y = {ly}, local_size_z = {lz}) in;"
                );
            }
            EntryStage::Vertex => {
                out.push_str("// stage: vertex\n");
            }
            EntryStage::Fragment => {
                out.push_str("// stage: fragment\n");
            }
        }
        if has_attr(&func.attrs, "maximally_reconverges") {
            out.push_str("// maximally_reconverges\n");
        }
        let _ = writeln!(out, "// entry: {}", func.name.name);
        out.push_str("void main() {\n");
        for (ty_str, local_name, gl_var) in builtin_inits {
            if let Some(var) = gl_var {
                let _ = writeln!(out, "    {ty_str} {local_name} = {var};");
            } else {
                let _ = writeln!(out, "    {ty_str} {local_name};");
            }
        }
        if let Some(ref body) = func.body {
            emit_block_contents(body, 1, false, ctx, out);
        }
        for (gl_var, local_name) in builtin_writebacks {
            let _ = writeln!(out, "    {gl_var} = {local_name};");
        }
        out.push_str("}\n\n");

        ctx.local_types = saved_locals;
        ctx.shadowed_world = saved_shadowed_world;
    } else {
        emit_fn_decl_with_ctx(func, None, ctx, out);
        out.push('\n');
    }
}

fn emit_entry_resource_binding(
    set: u32,
    binding: u32,
    name: &str,
    ty: &TypeExpr,
    ctx: &EmitCtx,
    out: &mut String,
) {
    if let TypeExpr::Named { path, args, .. } = ty {
        let base = path.joined();
        match base.as_str() {
            "Texture2D" => {
                let _ = writeln!(
                    out,
                    "layout(set = {set}, binding = {binding}) uniform texture2D {name};"
                );
                return;
            }
            "Texture3D" => {
                let _ = writeln!(
                    out,
                    "layout(set = {set}, binding = {binding}) uniform texture3D {name};"
                );
                return;
            }
            "Texture2DArray" => {
                let _ = writeln!(
                    out,
                    "layout(set = {set}, binding = {binding}) uniform texture2DArray {name};"
                );
                return;
            }
            "TextureCube" => {
                let _ = writeln!(
                    out,
                    "layout(set = {set}, binding = {binding}) uniform textureCube {name};"
                );
                return;
            }
            "Sampler" => {
                let _ = writeln!(
                    out,
                    "layout(set = {set}, binding = {binding}) uniform sampler {name};"
                );
                return;
            }
            "RWImage2D" | "RWImage2DArray" | "RWImage3D" => {
                let fmt = args
                    .first()
                    .and_then(|ta| match ta {
                        TypeArg::Type(TypeExpr::Named { path: p, .. }) => {
                            ImageFormat::from_name(&p.joined()).map(ImageFormat::as_str)
                        }
                        _ => None,
                    })
                    .unwrap_or("rgba16f");
                let img_ty = match base.as_str() {
                    "RWImage2DArray" => "image2DArray",
                    "RWImage3D" => "image3D",
                    _ => "image2D",
                };
                let _ = writeln!(
                    out,
                    "layout(set = {set}, binding = {binding}, {fmt}) uniform {img_ty} {name};"
                );
                return;
            }
            "Buffer" | "RWBuffer" => {
                let elem_ty = args
                    .first()
                    .and_then(|ta| match ta {
                        TypeArg::Type(t) => Some(type_to_glsl(t).0),
                        TypeArg::NamedExpr { .. } => None,
                    })
                    .unwrap_or_else(|| "uint".to_owned());
                let qual = if base == "Buffer" {
                    "readonly buffer"
                } else {
                    "buffer"
                };
                let layout_qual = ctx
                    .struct_layouts
                    .get(elem_ty.as_str())
                    .copied()
                    .unwrap_or(LayoutMode::Scalar)
                    .as_glsl();
                let _ = writeln!(
                    out,
                    "layout(set = {set}, binding = {binding}, {layout_qual}) {qual} VslBuf_{name} {{ {elem_ty} {name}[]; }};"
                );
                return;
            }
            _ => {}
        }
    }
    let (ty_str, arr_suffix) = type_to_glsl(ty);
    let _ = writeln!(
        out,
        "layout(set = {set}, binding = {binding}) uniform {ty_str} {name}{arr_suffix};"
    );
}

fn emit_block_contents(
    block: &Block,
    indent: usize,
    return_tail: bool,
    ctx: &mut EmitCtx,
    out: &mut String,
) {
    let saved_locals = ctx.local_types.clone();
    let saved_shadowed_world = ctx.shadowed_world;
    for stmt in &block.stmts {
        emit_stmt(stmt, indent, ctx, out);
    }
    if let Some(ref tail) = block.tail_expr {
        if return_tail {
            if let Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            } = tail.as_ref()
                && if_expr_needs_hoisting(then_branch, else_branch.as_deref())
            {
                let tmp = ctx.next_tmp();
                let ty_str = infer_glsl_type(tail, ctx);
                write_indent(indent, out);
                let _ = writeln!(out, "{ty_str} {tmp};");
                emit_hoisted_if_assign(
                    &tmp,
                    cond,
                    then_branch,
                    else_branch.as_deref(),
                    indent,
                    ctx,
                    out,
                );
                write_indent(indent, out);
                let _ = writeln!(out, "return {tmp};");
            } else {
                let mut pre = String::new();
                let tail_str = emit_expr_with_hoist(tail, indent, ctx, &mut pre);
                out.push_str(&pre);
                write_indent(indent, out);
                let _ = writeln!(out, "return {tail_str};");
            }
        } else {
            emit_stmt(
                &Stmt::Expr {
                    expr: (**tail).clone(),
                    has_semi: true,
                    span: tail.span(),
                },
                indent,
                ctx,
                out,
            );
        }
    }
    ctx.local_types = saved_locals;
    ctx.shadowed_world = saved_shadowed_world;
}

fn if_expr_needs_hoisting(then_branch: &Block, else_branch: Option<&Expr>) -> bool {
    if !then_branch.stmts.is_empty() {
        return true;
    }
    match else_branch {
        Some(Expr::Block(b)) => !b.stmts.is_empty(),
        Some(Expr::If {
            then_branch: et,
            else_branch: ee,
            ..
        }) => if_expr_needs_hoisting(et, ee.as_deref()),
        _ => false,
    }
}

fn emit_hoisted_if_assign(
    target: &str,
    cond: &Expr,
    then_branch: &Block,
    else_branch: Option<&Expr>,
    indent: usize,
    ctx: &mut EmitCtx,
    out: &mut String,
) {
    let mut pre = String::new();
    let cond_str = emit_expr_with_hoist(cond, indent, ctx, &mut pre);
    out.push_str(&pre);
    write_indent(indent, out);
    let _ = writeln!(out, "if ({cond_str}) {{");
    emit_branch_assign(target, then_branch, indent + 1, ctx, out);
    if let Some(eb) = else_branch {
        write_indent(indent, out);
        match eb {
            Expr::Block(else_block) => {
                out.push_str("} else {\n");
                emit_branch_assign(target, else_block, indent + 1, ctx, out);
                write_indent(indent, out);
                out.push_str("}\n");
            }
            Expr::If {
                cond: ec,
                then_branch: et,
                else_branch: ee,
                ..
            } => {
                out.push_str("} else ");
                let mut nested = String::new();
                emit_hoisted_if_assign(target, ec, et, ee.as_deref(), indent, ctx, &mut nested);
                out.push_str(nested.trim_start());
            }
            other => {
                out.push_str("} else {\n");
                let mut branch_pre = String::new();
                let val_str = emit_expr_with_hoist(other, indent + 1, ctx, &mut branch_pre);
                out.push_str(&branch_pre);
                write_indent(indent + 1, out);
                let _ = writeln!(out, "{target} = {val_str};");
                write_indent(indent, out);
                out.push_str("}\n");
            }
        }
    } else {
        write_indent(indent, out);
        out.push_str("}\n");
    }
}

fn emit_branch_assign(
    target: &str,
    block: &Block,
    indent: usize,
    ctx: &mut EmitCtx,
    out: &mut String,
) {
    let saved_locals = ctx.local_types.clone();
    let saved_shadowed_world = ctx.shadowed_world;
    for stmt in &block.stmts {
        emit_stmt(stmt, indent, ctx, out);
    }
    if let Some(ref tail) = block.tail_expr {
        let mut pre = String::new();
        let val_str = emit_expr_with_hoist(tail, indent, ctx, &mut pre);
        out.push_str(&pre);
        write_indent(indent, out);
        let _ = writeln!(out, "{target} = {val_str};");
    }
    ctx.local_types = saved_locals;
    ctx.shadowed_world = saved_shadowed_world;
}

fn emit_stmt(stmt: &Stmt, indent: usize, ctx: &mut EmitCtx, out: &mut String) {
    match stmt {
        Stmt::Let { name, ty, init, .. } => {
            let (ty_str, arr_suffix) = if let Some(t) = ty {
                type_to_glsl_with_subst(t, &ctx.generic_subst)
            } else if let Some(e) = init {
                (infer_glsl_type(e, ctx), String::new())
            } else {
                ("float".to_owned(), String::new())
            };
            ctx.local_types.insert(name.name.clone(), ty_str.clone());
            let glsl_name = escape_glsl_ident(&name.name);

            if let Some(Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            }) = init
                && if_expr_needs_hoisting(then_branch, else_branch.as_deref())
            {
                write_indent(indent, out);
                let _ = writeln!(out, "{ty_str} {glsl_name}{arr_suffix};");
                emit_hoisted_if_assign(
                    &glsl_name,
                    cond,
                    then_branch,
                    else_branch.as_deref(),
                    indent,
                    ctx,
                    out,
                );
                if name.name == "world" {
                    ctx.shadowed_world = true;
                }
                return;
            }

            if let Some(expr) = init {
                let mut pre = String::new();
                let init_str = emit_expr_with_hoist(expr, indent, ctx, &mut pre);
                out.push_str(&pre);
                write_indent(indent, out);
                let _ = writeln!(out, "{ty_str} {glsl_name}{arr_suffix} = {init_str};");
            } else {
                write_indent(indent, out);
                let _ = writeln!(out, "{ty_str} {glsl_name}{arr_suffix};");
            }
            if name.name == "world" {
                ctx.shadowed_world = true;
            }
        }
        Stmt::Return { value, .. } => {
            if let Some(Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            }) = value
                && if_expr_needs_hoisting(then_branch, else_branch.as_deref())
            {
                let tmp = ctx.next_tmp();
                let ty_str = value
                    .as_ref()
                    .map_or_else(|| "float".to_owned(), |e| infer_glsl_type(e, ctx));
                write_indent(indent, out);
                let _ = writeln!(out, "{ty_str} {tmp};");
                emit_hoisted_if_assign(
                    &tmp,
                    cond,
                    then_branch,
                    else_branch.as_deref(),
                    indent,
                    ctx,
                    out,
                );
                write_indent(indent, out);
                let _ = writeln!(out, "return {tmp};");
                return;
            }
            if let Some(expr) = value {
                let mut pre = String::new();
                let val_str = emit_expr_with_hoist(expr, indent, ctx, &mut pre);
                out.push_str(&pre);
                write_indent(indent, out);
                let _ = writeln!(out, "return {val_str};");
            } else {
                write_indent(indent, out);
                out.push_str("return;\n");
            }
        }
        Stmt::For {
            attrs,
            binding,
            iter,
            body,
            ..
        } => {
            let saved_locals = ctx.local_types.clone();
            let saved_shadowed_world = ctx.shadowed_world;
            if binding.name == "world" {
                ctx.shadowed_world = true;
            }
            let var_name = if binding.name == "_" {
                "_vsl_iter".to_owned()
            } else {
                escape_glsl_ident(&binding.name)
            };
            let mut pre = String::new();
            let header = if let Some((start, end, inclusive, step)) = decompose_range_iter(iter) {
                let cmp = if inclusive { "<=" } else { "<" };
                let iter_ty = infer_range_type(start, end, ctx);
                ctx.local_types
                    .insert(binding.name.clone(), iter_ty.to_owned());
                let start_s = emit_expr_with_hoist(start, indent, ctx, &mut pre);
                let end_s = emit_expr_with_hoist(end, indent, ctx, &mut pre);
                let step_clause = step.map_or_else(
                    || format!("++{var_name}"),
                    |s| {
                        let s_str = emit_expr_with_hoist(s, indent, ctx, &mut pre);
                        format!("{var_name} += {s_str}")
                    },
                );
                format!(
                    "for ({iter_ty} {var_name} = {start_s}; {var_name} {cmp} {end_s}; {step_clause}) {{"
                )
            } else {
                ctx.local_types
                    .insert(binding.name.clone(), "uint".to_owned());
                let iter_s = emit_expr_with_hoist(iter, indent, ctx, &mut pre);
                format!("for (uint {var_name} = 0u; {var_name} < {iter_s}; ++{var_name}) {{")
            };
            out.push_str(&pre);
            write_indent(indent, out);
            if has_attr(attrs, "unroll") {
                out.push_str("[[unroll]] ");
            }
            let _ = writeln!(out, "{header}");
            emit_block_contents(body, indent + 1, false, ctx, out);
            write_indent(indent, out);
            out.push_str("}\n");
            ctx.local_types = saved_locals;
            ctx.shadowed_world = saved_shadowed_world;
        }
        Stmt::While { cond, body, .. } => {
            let mut pre = String::new();
            let cond_s = emit_expr_with_hoist(cond, indent, ctx, &mut pre);
            out.push_str(&pre);
            write_indent(indent, out);
            let _ = writeln!(out, "while ({cond_s}) {{");
            emit_block_contents(body, indent + 1, false, ctx, out);
            write_indent(indent, out);
            out.push_str("}\n");
        }
        Stmt::Loop { body, .. } => {
            write_indent(indent, out);
            out.push_str("while (true) {\n");
            emit_block_contents(body, indent + 1, false, ctx, out);
            write_indent(indent, out);
            out.push_str("}\n");
        }
        Stmt::Switch {
            selector,
            cases,
            default,
            ..
        } => {
            let mut pre = String::new();
            let sel_s = emit_expr_with_hoist(selector, indent, ctx, &mut pre);
            out.push_str(&pre);
            write_indent(indent, out);
            let _ = writeln!(out, "switch ({sel_s}) {{");
            for case in cases {
                write_indent(indent + 1, out);
                let _ = writeln!(out, "case {}: {{", emit_expr_const(&case.value, ctx));
                emit_block_contents(&case.body, indent + 2, false, ctx, out);
                write_indent(indent + 2, out);
                out.push_str("break;\n");
                write_indent(indent + 1, out);
                out.push_str("}\n");
            }
            if let Some(def_block) = default {
                write_indent(indent + 1, out);
                out.push_str("default: {\n");
                emit_block_contents(def_block, indent + 2, false, ctx, out);
                write_indent(indent + 2, out);
                out.push_str("break;\n");
                write_indent(indent + 1, out);
                out.push_str("}\n");
            }
            write_indent(indent, out);
            out.push_str("}\n");
        }
        Stmt::Break(_) => {
            write_indent(indent, out);
            out.push_str("break;\n");
        }
        Stmt::Continue(_) => {
            write_indent(indent, out);
            out.push_str("continue;\n");
        }
        Stmt::Expr { expr, .. } => match expr {
            Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                emit_if_stmt(cond, then_branch, else_branch.as_deref(), indent, ctx, out);
            }
            Expr::Block(block) => {
                write_indent(indent, out);
                out.push_str("{\n");
                emit_block_contents(block, indent + 1, false, ctx, out);
                write_indent(indent, out);
                out.push_str("}\n");
            }
            Expr::Binary {
                op: BinaryOp::Assign,
                lhs,
                rhs,
                ..
            } if matches!(
                rhs.as_ref(),
                Expr::If {
                    then_branch,
                    else_branch,
                    ..
                } if if_expr_needs_hoisting(then_branch, else_branch.as_deref())
            ) =>
            {
                let mut pre = String::new();
                let lhs_str = emit_expr_with_hoist(lhs, indent, ctx, &mut pre);
                out.push_str(&pre);
                if let Expr::If {
                    cond,
                    then_branch,
                    else_branch,
                    ..
                } = rhs.as_ref()
                {
                    emit_hoisted_if_assign(
                        &lhs_str,
                        cond,
                        then_branch,
                        else_branch.as_deref(),
                        indent,
                        ctx,
                        out,
                    );
                }
            }
            _ => {
                let mut pre = String::new();
                let expr_str = emit_expr_with_hoist(expr, indent, ctx, &mut pre);
                out.push_str(&pre);
                write_indent(indent, out);
                let _ = writeln!(out, "{expr_str};");
            }
        },
    }
}

fn emit_if_stmt(
    cond: &Expr,
    then_branch: &Block,
    else_branch: Option<&Expr>,
    indent: usize,
    ctx: &mut EmitCtx,
    out: &mut String,
) {
    let mut pre = String::new();
    let cond_str = emit_expr_with_hoist(cond, indent, ctx, &mut pre);
    out.push_str(&pre);
    write_indent(indent, out);
    let _ = writeln!(out, "if ({cond_str}) {{");
    emit_block_contents(then_branch, indent + 1, false, ctx, out);
    if let Some(eb) = else_branch {
        write_indent(indent, out);
        match eb {
            Expr::Block(else_block) => {
                out.push_str("} else {\n");
                emit_block_contents(else_block, indent + 1, false, ctx, out);
                write_indent(indent, out);
                out.push_str("}\n");
            }
            Expr::If {
                cond: ec,
                then_branch: et,
                else_branch: ee,
                ..
            } => {
                out.push_str("} else ");
                let mut nested = String::new();
                emit_if_stmt(ec, et, ee.as_deref(), indent, ctx, &mut nested);
                out.push_str(nested.trim_start());
            }
            other => {
                out.push_str("} else {\n");
                let mut branch_pre = String::new();
                let other_str = emit_expr_with_hoist(other, indent + 1, ctx, &mut branch_pre);
                out.push_str(&branch_pre);
                write_indent(indent + 1, out);
                let _ = writeln!(out, "{other_str};");
                write_indent(indent, out);
                out.push_str("}\n");
            }
        }
    } else {
        write_indent(indent, out);
        out.push_str("}\n");
    }
}

fn emit_expr_const(expr: &Expr, ctx: &EmitCtx) -> String {
    let mut dummy_ctx = ctx.clone();
    let mut discard = String::new();
    emit_expr_with_hoist(expr, 0, &mut dummy_ctx, &mut discard)
}

fn emit_expr_with_hoist(
    expr: &Expr,
    indent: usize,
    ctx: &mut EmitCtx,
    pre_stmts: &mut String,
) -> String {
    match expr {
        Expr::IntLit(raw, _) => format_int_literal(raw),
        Expr::FloatLit(raw, _) => format_float_literal(raw),
        Expr::BoolLit(b, _) => {
            if *b {
                "true".to_owned()
            } else {
                "false".to_owned()
            }
        }
        Expr::StringLit(s, _) => sanitize_glsl_ident(s),
        Expr::NoneLit(_) => "0".to_owned(),
        Expr::Path { path, .. } => format_path_expr(path, ctx),
        Expr::ArrayLit { elements, .. } => {
            let items: Vec<String> = elements
                .iter()
                .map(|e| emit_expr_with_hoist(e, indent, ctx, pre_stmts))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Expr::StructLit { path, fields, .. } => {
            let struct_name = resolve_struct_lit_name(path, fields, ctx);
            let decl_keys: Option<Vec<SmolStr>> = ctx
                .struct_fields
                .get(struct_name.as_str())
                .map(|m| m.keys().cloned().collect());
            let ordered_vals: Vec<String> = if let Some(keys) = decl_keys {
                keys.iter()
                    .map(|fname| {
                        let expected = escape_glsl_ident(fname);
                        fields
                            .iter()
                            .find(|(id, _)| escape_glsl_ident(&id.name) == expected)
                            .map_or_else(
                                || "0".to_owned(),
                                |(_, v)| emit_expr_with_hoist(v, indent, ctx, pre_stmts),
                            )
                    })
                    .collect()
            } else {
                fields
                    .iter()
                    .map(|(_, v)| emit_expr_with_hoist(v, indent, ctx, pre_stmts))
                    .collect()
            };
            format!("{struct_name}({})", ordered_vals.join(", "))
        }
        Expr::Unary { op, operand, .. } => {
            let inner = emit_expr_with_hoist(operand, indent, ctx, pre_stmts);
            match op {
                UnaryOp::Neg => format!("-({inner})"),
                UnaryOp::Not => format!("!({inner})"),
                UnaryOp::BitNot => format!("~({inner})"),
                // GLSL atomic intrinsics accept lvalue references directly without `&`.
                UnaryOp::AddrOf => inner,
            }
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let l = emit_expr_with_hoist(lhs, indent, ctx, pre_stmts);
            let r = emit_expr_with_hoist(rhs, indent, ctx, pre_stmts);
            let op_str = binary_op_glsl(*op);
            if is_assignment_op(*op) {
                format!("{l} {op_str} {r}")
            } else {
                format!("({l} {op_str} {r})")
            }
        }
        Expr::Call { callee, args, .. } => {
            emit_call_expr_with_hoist(callee, args, indent, ctx, pre_stmts)
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
            ..
        } => emit_method_call_expr_with_hoist(receiver, &method.name, args, indent, ctx, pre_stmts),
        Expr::Field {
            receiver, field, ..
        } => {
            let field_name = escape_glsl_ident(&field.name);
            if ctx.opts.route_params_to_vs_motion
                && let Expr::Path { path, .. } = receiver.as_ref()
                && matches!(
                    path.joined().as_str(),
                    "mc::params" | "params" | "pack::params"
                )
            {
                return format!("VsMotion.packParams.{field_name}");
            }
            let recv_ty = infer_glsl_type(receiver, ctx);
            let recv_str = emit_expr_with_hoist(receiver, indent, ctx, pre_stmts);
            if recv_ty.starts_with("VslPtr") {
                format!("{recv_str}.value.{field_name}")
            } else {
                format!("{recv_str}.{field_name}")
            }
        }
        Expr::Index {
            receiver, index, ..
        } => {
            let recv_s = emit_expr_with_hoist(receiver, indent, ctx, pre_stmts);
            let idx_s = emit_expr_with_hoist(index, indent, ctx, pre_stmts);
            format!("{recv_s}[{idx_s}]")
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            if if_expr_needs_hoisting(then_branch, else_branch.as_deref()) {
                let tmp = ctx.next_tmp();
                let ty_str = infer_glsl_type(expr, ctx);
                write_indent(indent, pre_stmts);
                let _ = writeln!(pre_stmts, "{ty_str} {tmp};");
                emit_hoisted_if_assign(
                    &tmp,
                    cond,
                    then_branch,
                    else_branch.as_deref(),
                    indent,
                    ctx,
                    pre_stmts,
                );
                return tmp;
            }
            let c = emit_expr_with_hoist(cond, indent, ctx, pre_stmts);
            let t = then_branch.tail_expr.as_ref().map_or_else(
                || "0".to_owned(),
                |e| emit_expr_with_hoist(e, indent, ctx, pre_stmts),
            );
            let f = else_branch.as_deref().map_or_else(
                || "0".to_owned(),
                |e| emit_expr_with_hoist(e, indent, ctx, pre_stmts),
            );
            format!("({c} ? {t} : {f})")
        }
        Expr::Block(block) => {
            if !block.stmts.is_empty() {
                let tmp = ctx.next_tmp();
                let ty_str = infer_glsl_type(expr, ctx);
                write_indent(indent, pre_stmts);
                let _ = writeln!(pre_stmts, "{ty_str} {tmp};");
                write_indent(indent, pre_stmts);
                pre_stmts.push_str("{\n");
                emit_branch_assign(&tmp, block, indent + 1, ctx, pre_stmts);
                write_indent(indent, pre_stmts);
                pre_stmts.push_str("}\n");
                return tmp;
            }
            block.tail_expr.as_ref().map_or_else(
                || "0".to_owned(),
                |e| emit_expr_with_hoist(e, indent, ctx, pre_stmts),
            )
        }
    }
}

fn sanitize_glsl_ident(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "default".to_owned()
    } else if out.as_bytes()[0].is_ascii_digit() {
        format!("_{out}")
    } else {
        out
    }
}

fn emit_call_expr_with_hoist(
    callee: &Expr,
    args: &[CallArg],
    indent: usize,
    ctx: &mut EmitCtx,
    pre_stmts: &mut String,
) -> String {
    if let Expr::Path { path, .. } = callee {
        let joined = path.joined();

        // Lower `mc::debug_write("channel", coord, color)` into `imageStore(vsl_debug_<channel>, ...)`.
        if joined == "mc::debug_write" && args.len() >= 3 {
            let channel = match &args[0].value {
                Expr::StringLit(s, _) => sanitize_glsl_ident(s),
                Expr::Path { path: p, .. } => sanitize_glsl_ident(&p.joined()),
                _ => "default".to_owned(),
            };
            let coord = emit_expr_with_hoist(&args[1].value, indent, ctx, pre_stmts);
            let val_ty = infer_glsl_type(&args[2].value, ctx);
            let val = emit_expr_with_hoist(&args[2].value, indent, ctx, pre_stmts);
            let vec4_val = match val_ty.as_str() {
                "vec4" => val,
                "vec3" => format!("vec4({val}, 1.0)"),
                "vec2" => format!("vec4({val}, 0.0, 1.0)"),
                _ => format!("vec4({val})"),
            };
            return format!("imageStore(vsl_debug_{channel}, {coord}, {vec4_val})");
        }

        // Route `param(key)` / `mc::param(key)` / `pack::param(key)` to `VsMotion.packParams.<key>`.
        if ctx.opts.route_params_to_vs_motion
            && matches!(joined.as_str(), "param" | "mc::param" | "pack::param")
        {
            if let Some(first) = args.first() {
                let key = match &first.value {
                    Expr::Path { path: p, .. } => p.joined().replace("::", "_"),
                    Expr::StringLit(s, _) => s.to_string(),
                    other => emit_expr_with_hoist(other, indent, ctx, pre_stmts),
                };
                let escaped_key = escape_glsl_ident(&key);
                return format!("VsMotion.packParams.{escaped_key}");
            }
            return "VsMotion.packParams".to_owned();
        }

        let mapped = map_call_name(&joined);
        let is_atomic = matches!(
            joined.as_str(),
            "atomic_add"
                | "atomic_cas"
                | "atomic_min"
                | "atomic_max"
                | "atomic_and"
                | "atomic_or"
                | "atomic_xor"
                | "atomic_exchange"
        );
        let rendered_args: Vec<String> = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let s = emit_expr_with_hoist(&a.value, indent, ctx, pre_stmts);
                if is_atomic && i == 0 && infer_glsl_type(&a.value, ctx).starts_with("VslPtr") {
                    format!("{s}.value")
                } else {
                    s
                }
            })
            .collect();
        return format!("{mapped}({})", rendered_args.join(", "));
    }

    let c = emit_expr_with_hoist(callee, indent, ctx, pre_stmts);
    let rendered_args: Vec<String> = args
        .iter()
        .map(|a| emit_expr_with_hoist(&a.value, indent, ctx, pre_stmts))
        .collect();
    format!("{c}({})", rendered_args.join(", "))
}

fn is_combined_sampler_type(ty: &str) -> bool {
    matches!(
        ty,
        "sampler2D" | "sampler3D" | "sampler2DArray" | "samplerCube"
    )
}

fn combined_sampler_ctor(tex_ty: &str) -> &'static str {
    match tex_ty {
        "texture3D" | "sampler3D" => "sampler3D",
        "texture2DArray" | "sampler2DArray" => "sampler2DArray",
        "textureCube" | "samplerCube" => "samplerCube",
        _ => "sampler2D",
    }
}

const fn align_up(offset: u64, align: u64) -> u64 {
    let a = if align == 0 { 1 } else { align };
    offset.div_ceil(a) * a
}

fn extract_array_dims(ty: &TypeExpr) -> Vec<u64> {
    let mut dims = Vec::new();
    let mut cur = ty;
    while let TypeExpr::Array { element, len, .. } = cur {
        let count = parse_u32_lit(len).map_or(1, u64::from).max(1);
        dims.push(count);
        cur = element;
    }
    dims
}

fn bda_element_stride(bda: &BdaRefKey, ctx: &EmitCtx) -> u64 {
    let layout = ctx
        .struct_layouts
        .get(bda.elem_glsl.as_str())
        .copied()
        .unwrap_or(LayoutMode::Scalar);
    let (size, natural_align) = glsl_type_size_align(&bda.elem_glsl, &[], layout, ctx, 0);
    let align = natural_align.max(u64::from(bda.align.max(1)));
    align_up(size, align)
}

fn glsl_type_size_align(
    ty: &str,
    array_dims: &[u64],
    layout: LayoutMode,
    ctx: &EmitCtx,
    depth: usize,
) -> (u64, u64) {
    let (elem_size, elem_align) = match ty {
        "float16_t" => (2, 2),
        "f16vec2" => (
            4,
            match layout {
                LayoutMode::Scalar => 2,
                LayoutMode::Std430 | LayoutMode::Std140 => 4,
            },
        ),
        "f16vec3" => (
            6,
            match layout {
                LayoutMode::Scalar => 2,
                LayoutMode::Std430 | LayoutMode::Std140 => 8,
            },
        ),
        "f16vec4" => (
            8,
            match layout {
                LayoutMode::Scalar => 2,
                LayoutMode::Std430 | LayoutMode::Std140 => 8,
            },
        ),
        "float" | "int" | "uint" | "bool" => (4, 4),
        "double" | "int64_t" | "uint64_t" => (8, 8),
        other if other.starts_with("VslPtr") => (8, 8),
        "vec2" | "ivec2" | "uvec2" | "bvec2" => (
            8,
            match layout {
                LayoutMode::Scalar => 4,
                LayoutMode::Std430 | LayoutMode::Std140 => 8,
            },
        ),
        "vec3" | "ivec3" | "uvec3" | "bvec3" => (
            12,
            match layout {
                LayoutMode::Scalar => 4,
                LayoutMode::Std430 | LayoutMode::Std140 => 16,
            },
        ),
        "vec4" | "ivec4" | "uvec4" | "bvec4" => (
            16,
            match layout {
                LayoutMode::Scalar => 4,
                LayoutMode::Std430 | LayoutMode::Std140 => 16,
            },
        ),
        "dvec2" => (
            16,
            match layout {
                LayoutMode::Scalar => 8,
                LayoutMode::Std430 | LayoutMode::Std140 => 16,
            },
        ),
        "dvec3" => (
            24,
            match layout {
                LayoutMode::Scalar => 8,
                LayoutMode::Std430 | LayoutMode::Std140 => 32,
            },
        ),
        "dvec4" => (
            32,
            match layout {
                LayoutMode::Scalar => 8,
                LayoutMode::Std430 | LayoutMode::Std140 => 32,
            },
        ),
        "mat2" => match layout {
            LayoutMode::Scalar => (16, 4),
            LayoutMode::Std430 => (16, 8),
            LayoutMode::Std140 => (32, 16),
        },
        "mat3" => match layout {
            LayoutMode::Scalar => (36, 4),
            LayoutMode::Std430 | LayoutMode::Std140 => (48, 16),
        },
        "mat4" => (
            64,
            match layout {
                LayoutMode::Scalar => 4,
                LayoutMode::Std430 | LayoutMode::Std140 => 16,
            },
        ),
        other => {
            if depth < 16
                && let Some(fields) = ctx.struct_fields.get(other)
            {
                let field_dims_map = ctx.struct_field_dims.get(other);
                let mut offset = 0u64;
                let mut max_align = 1u64;
                for (fname, f_ty) in fields {
                    let f_dims = field_dims_map
                        .and_then(|m| m.get(fname))
                        .map_or(&[][..], Vec::as_slice);
                    let (f_size, f_align) =
                        glsl_type_size_align(f_ty, f_dims, layout, ctx, depth + 1);
                    offset = align_up(offset, f_align);
                    offset += f_size;
                    max_align = max_align.max(f_align);
                }
                if layout == LayoutMode::Std140 {
                    max_align = align_up(max_align, 16);
                }
                (align_up(offset, max_align).max(4), max_align)
            } else {
                (4, 4)
            }
        }
    };

    if array_dims.is_empty() {
        return (elem_size, elem_align);
    }

    let arr_align = match layout {
        LayoutMode::Std140 => align_up(elem_align, 16),
        LayoutMode::Scalar | LayoutMode::Std430 => elem_align,
    };
    let elem_stride = align_up(elem_size, arr_align);
    let total_elems: u64 = array_dims
        .iter()
        .copied()
        .fold(1u64, |acc, d| acc.saturating_mul(d.max(1)));
    (elem_stride.saturating_mul(total_elems), arr_align)
}

fn emit_method_call_expr_with_hoist(
    receiver: &Expr,
    method: &str,
    args: &[CallArg],
    indent: usize,
    ctx: &mut EmitCtx,
    pre_stmts: &mut String,
) -> String {
    let recv_ty = infer_glsl_type(receiver, ctx);
    let recv = emit_expr_with_hoist(receiver, indent, ctx, pre_stmts);
    let rendered_args: Vec<String> = args
        .iter()
        .map(|a| emit_expr_with_hoist(&a.value, indent, ctx, pre_stmts))
        .collect();

    match method {
        "store" => {
            if rendered_args.len() == 2 {
                format!(
                    "imageStore({recv}, {}, {})",
                    rendered_args[0], rendered_args[1]
                )
            } else if rendered_args.len() == 1 {
                format!("({recv}).value = {}", rendered_args[0])
            } else {
                format!("imageStore({recv}, {})", rendered_args.join(", "))
            }
        }
        "load" => {
            if rendered_args.is_empty() {
                format!("({recv}).value")
            } else {
                format!("imageLoad({recv}, {})", rendered_args.join(", "))
            }
        }
        "fetch" => format!("texelFetch({recv}, {})", rendered_args.join(", ")),
        "sample_lod" => {
            if is_combined_sampler_type(&recv_ty) {
                if args.len() == 3 && infer_glsl_type(&args[0].value, ctx) == "sampler" {
                    format!(
                        "textureLod({recv}, {}, {})",
                        rendered_args[1], rendered_args[2]
                    )
                } else {
                    format!("textureLod({recv}, {})", rendered_args.join(", "))
                }
            } else {
                let ctor = combined_sampler_ctor(&recv_ty);
                if args.len() == 3 && infer_glsl_type(&args[0].value, ctx) == "sampler" {
                    format!(
                        "textureLod({ctor}({recv}, {}), {}, {})",
                        rendered_args[0], rendered_args[1], rendered_args[2]
                    )
                } else {
                    format!(
                        "textureLod({ctor}({recv}, vsl_default_sampler), {})",
                        rendered_args.join(", ")
                    )
                }
            }
        }
        "sample" => {
            if is_combined_sampler_type(&recv_ty) {
                if args.len() == 2 && infer_glsl_type(&args[0].value, ctx) == "sampler" {
                    format!("texture({recv}, {})", rendered_args[1])
                } else {
                    format!("texture({recv}, {})", rendered_args.join(", "))
                }
            } else {
                let ctor = combined_sampler_ctor(&recv_ty);
                if args.len() == 2 && infer_glsl_type(&args[0].value, ctx) == "sampler" {
                    format!(
                        "texture({ctor}({recv}, {}), {})",
                        rendered_args[0], rendered_args[1]
                    )
                } else {
                    format!(
                        "texture({ctor}({recv}, vsl_default_sampler), {})",
                        rendered_args.join(", ")
                    )
                }
            }
        }
        "offset" => {
            if let Some(off) = rendered_args.first() {
                if let Some(bda) = ctx.bda_types.get(&recv_ty) {
                    let stride = bda_element_stride(bda, ctx);
                    format!("{recv_ty}(uint64_t({recv}) + uint64_t({off}) * {stride}ul)")
                } else {
                    format!("({recv} + {off})")
                }
            } else {
                recv
            }
        }
        "initialize" => format!(
            "rayQueryInitializeEXT({recv}, {})",
            rendered_args.join(", ")
        ),
        "proceed" => format!("rayQueryProceedEXT({recv})"),
        "committed_type" => format!("rayQueryGetIntersectionTypeEXT({recv}, true)"),
        "candidate_type" => format!("rayQueryGetIntersectionTypeEXT({recv}, false)"),
        "confirm_intersection" => format!("rayQueryConfirmIntersectionEXT({recv})"),
        "terminate" => format!("rayQueryTerminateEXT({recv})"),
        "generate_intersection" => format!(
            "rayQueryGenerateIntersectionEXT({recv}, {})",
            rendered_args.join(", ")
        ),
        "committed_t" => format!("rayQueryGetIntersectionTEXT({recv}, true)"),
        "candidate_t" => format!("rayQueryGetIntersectionTEXT({recv}, false)"),
        "committed_instance_id" => {
            format!("uint(rayQueryGetIntersectionInstanceIdEXT({recv}, true))")
        }
        "committed_primitive_index" => {
            format!("uint(rayQueryGetIntersectionPrimitiveIndexEXT({recv}, true))")
        }
        "committed_geometry_index" => {
            format!("uint(rayQueryGetIntersectionGeometryIndexEXT({recv}, true))")
        }
        "committed_object_to_world" => {
            format!("rayQueryGetIntersectionObjectToWorldEXT({recv}, true)")
        }
        "committed_world_to_object" => {
            format!("rayQueryGetIntersectionWorldToObjectEXT({recv}, true)")
        }
        "committed_front_face" => format!("rayQueryGetIntersectionFrontFaceEXT({recv}, true)"),
        "committed_instance_custom_index" => {
            format!("uint(rayQueryGetIntersectionInstanceCustomIndexEXT({recv}, true))")
        }
        "committed_barycentrics" => {
            format!("rayQueryGetIntersectionBarycentricsEXT({recv}, true)")
        }
        _ => {
            let mangled = format!("{recv_ty}_{method}");
            let fn_call = if ctx.fn_returns.contains_key(mangled.as_str()) {
                mangled
            } else {
                method.to_owned()
            };
            if rendered_args.is_empty() {
                format!("{fn_call}({recv})")
            } else {
                format!("{fn_call}({recv}, {})", rendered_args.join(", "))
            }
        }
    }
}

pub(crate) fn type_to_glsl(ty: &TypeExpr) -> (String, String) {
    let empty = IndexMap::new();
    type_to_glsl_with_subst(ty, &empty)
}

fn type_to_glsl_with_subst(
    ty: &TypeExpr,
    generic_subst: &IndexMap<SmolStr, String>,
) -> (String, String) {
    match ty {
        TypeExpr::Array { element, len, .. } => {
            let (elem_ty, inner_suffix) = type_to_glsl_with_subst(element, generic_subst);
            let empty_ctx = EmitCtx::new(EmitOptions::default());
            let len_str = emit_expr_const(len, &empty_ctx);
            (elem_ty, format!("{inner_suffix}[{len_str}]"))
        }
        TypeExpr::Named {
            path,
            args,
            unsized_array,
            ..
        } => {
            let joined = path.joined();
            if let Some(subst_ty) = generic_subst.get(joined.as_str()) {
                let suffix = if *unsized_array {
                    "[]".to_owned()
                } else {
                    String::new()
                };
                return (subst_ty.clone(), suffix);
            }
            if let Some(bda) = parse_bda_type_with_subst(&joined, args, generic_subst) {
                return (bda.type_name(), String::new());
            }
            let is_builtin_generic = matches!(
                joined.as_str(),
                "Texture2D"
                    | "Texture3D"
                    | "Texture2DArray"
                    | "TextureCube"
                    | "RWImage2D"
                    | "RWImage2DArray"
                    | "RWImage3D"
                    | "Buffer"
                    | "RWBuffer"
            );
            let mut base = match joined.as_str() {
                "f32" => "float".to_owned(),
                "f64" => "double".to_owned(),
                "f16" => "float16_t".to_owned(),
                "i32" => "int".to_owned(),
                "u32" => "uint".to_owned(),
                "i64" => "int64_t".to_owned(),
                "u64" => "uint64_t".to_owned(),
                "bool" => "bool".to_owned(),
                "RayQuery" => "rayQueryEXT".to_owned(),
                "AccelerationStructure" => "accelerationStructureEXT".to_owned(),
                "Texture2D" => "texture2D".to_owned(),
                "Texture3D" => "texture3D".to_owned(),
                "Texture2DArray" => "texture2DArray".to_owned(),
                "TextureCube" => "textureCube".to_owned(),
                "Sampler" => "sampler".to_owned(),
                "RWImage2D" => "image2D".to_owned(),
                "RWImage2DArray" => "image2DArray".to_owned(),
                "RWImage3D" => "image3D".to_owned(),
                other => other.replace("::", "_"),
            };
            if !is_builtin_generic && !args.is_empty() {
                let resolved_args: Vec<String> = args
                    .iter()
                    .filter_map(|arg| match arg {
                        TypeArg::Type(inner) => {
                            Some(type_to_glsl_with_subst(inner, generic_subst).0)
                        }
                        TypeArg::NamedExpr { .. } => None,
                    })
                    .collect();
                if !resolved_args.is_empty() {
                    base = mangle_generic_name(&base, &resolved_args);
                }
            }
            let suffix = if *unsized_array {
                "[]".to_owned()
            } else {
                String::new()
            };
            (base, suffix)
        }
    }
}

fn parse_bda_type_with_subst(
    name: &str,
    args: &[TypeArg],
    generic_subst: &IndexMap<SmolStr, String>,
) -> Option<BdaRefKey> {
    let is_mut = match name {
        "ptr" => false,
        "ptr_mut" => true,
        _ => return None,
    };
    let elem_glsl = if let Some(TypeArg::Type(t)) = args.first() {
        type_to_glsl_with_subst(t, generic_subst).0
    } else {
        "uint".to_owned()
    };
    let mut align = 4u32;
    for arg in args.iter().skip(1) {
        if let TypeArg::NamedExpr { name, value, .. } = arg
            && name.name == "align"
            && let Some(v) = parse_u32_lit(value)
        {
            align = v;
        }
    }
    Some(BdaRefKey {
        is_mut,
        elem_glsl,
        align,
    })
}

fn swizzle_result_type(base_ty: &str, field: &str) -> Option<String> {
    if field.is_empty()
        || field.len() > 4
        || !field.chars().all(|c| {
            matches!(
                c,
                'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a' | 's' | 't' | 'p' | 'q'
            )
        })
    {
        return None;
    }
    let (scalar, prefix) = match base_ty {
        "vec2" | "vec3" | "vec4" => ("float", "vec"),
        "ivec2" | "ivec3" | "ivec4" => ("int", "ivec"),
        "uvec2" | "uvec3" | "uvec4" => ("uint", "uvec"),
        "bvec2" | "bvec3" | "bvec4" => ("bool", "bvec"),
        _ => return None,
    };
    if field.len() == 1 {
        Some(scalar.to_owned())
    } else {
        Some(format!("{prefix}{}", field.len()))
    }
}

fn resolve_struct_lit_name(path: &Path, fields: &[(Ident, Expr)], ctx: &EmitCtx) -> String {
    let base = path.joined().replace("::", "_");
    let Some(meta) = ctx.generic_structs.get(base.as_str()) else {
        return base;
    };
    let mut resolved: Vec<Option<String>> = vec![None; meta.generic_param_count];
    for (field_name, field_idx) in &meta.param_fields {
        if let Some((_, val_expr)) = fields.iter().find(|(id, _)| &id.name == field_name) {
            let ty = infer_glsl_type(val_expr, ctx);
            if !ty.is_empty() {
                resolved[*field_idx] = Some(ty);
            }
        }
    }
    if resolved.iter().all(Option::is_some) {
        let args: Vec<String> = resolved.into_iter().flatten().collect();
        return mangle_generic_name(&base, &args);
    }
    base
}

fn infer_glsl_type(expr: &Expr, ctx: &EmitCtx) -> String {
    match expr {
        Expr::BoolLit(_, _) => "bool".to_owned(),
        Expr::IntLit(raw, _) => {
            if raw.ends_with("u64") {
                "uint64_t".to_owned()
            } else if raw.ends_with("i64") {
                "int64_t".to_owned()
            } else if raw.ends_with('u') || raw.ends_with("u32") {
                "uint".to_owned()
            } else {
                "int".to_owned()
            }
        }
        Expr::FloatLit(raw, _) => {
            if raw.ends_with("f16") {
                "float16_t".to_owned()
            } else {
                "float".to_owned()
            }
        }
        Expr::Path { path, .. } => {
            let joined = path.joined().replace("::", "_");
            if let Some(ty) = ctx.local_types.get(joined.as_str()) {
                return ty.clone();
            }
            if path.segments.len() == 1
                && let Some(ty) = ctx.local_types.get(&path.segments[0].name)
            {
                return ty.clone();
            }
            "float".to_owned()
        }
        Expr::StructLit { path, fields, .. } => resolve_struct_lit_name(path, fields, ctx),
        Expr::Field {
            receiver, field, ..
        } => {
            if let Expr::Path { path, .. } = receiver.as_ref()
                && matches!(
                    path.joined().as_str(),
                    "mc::params" | "params" | "pack::params"
                )
                && let Some(param_ty) = ctx.pack_param_types.get(&field.name)
            {
                return param_ty.clone();
            }
            let recv_ty = infer_glsl_type(receiver, ctx);
            if let Some(swiz) = swizzle_result_type(&recv_ty, &field.name) {
                return swiz;
            }
            let struct_key = ctx
                .bda_types
                .get(&recv_ty)
                .map_or(recv_ty.as_str(), |b| b.elem_glsl.as_str());
            if let Some(fields) = ctx.struct_fields.get(struct_key)
                && let Some(f_ty) = fields.get(&field.name)
            {
                return f_ty.clone();
            }
            "float".to_owned()
        }
        Expr::Index { receiver, .. } => infer_glsl_type(receiver, ctx),
        Expr::Unary { op, operand, .. } => match op {
            UnaryOp::Not => "bool".to_owned(),
            _ => infer_glsl_type(operand, ctx),
        },
        Expr::MethodCall {
            receiver, method, ..
        } => {
            let recv_ty = infer_glsl_type(receiver, ctx);
            match method.name.as_str() {
                "offset" => recv_ty,
                "load" => ctx
                    .bda_types
                    .get(&recv_ty)
                    .map_or_else(|| "vec4".to_owned(), |b| b.elem_glsl.clone()),
                "fetch" | "sample" | "sample_lod" => "vec4".to_owned(),
                "committed_type"
                | "candidate_type"
                | "committed_instance_custom_index"
                | "committed_instance_id"
                | "committed_primitive_index"
                | "committed_geometry_index" => "uint".to_owned(),
                "committed_object_to_world" | "committed_world_to_object" => "mat4x3".to_owned(),
                "committed_t" | "candidate_t" => "float".to_owned(),
                "initialize" | "confirm_intersection" | "terminate" | "generate_intersection" => {
                    "void".to_owned()
                }
                "committed_barycentrics" => "vec2".to_owned(),
                "proceed" | "committed_front_face" => "bool".to_owned(),
                other => {
                    let mangled = format!("{recv_ty}_{other}");
                    ctx.fn_returns
                        .get(mangled.as_str())
                        .or_else(|| ctx.fn_returns.get(other))
                        .cloned()
                        .unwrap_or_else(|| "float".to_owned())
                }
            }
        }
        Expr::Call { callee, args, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref() {
                let name = path.joined();
                if matches!(name.as_str(), "param" | "mc::param" | "pack::param")
                    && let Some(first) = args.first()
                {
                    let key = match &first.value {
                        Expr::Path { path: p, .. } => Some(p.joined().replace("::", "_")),
                        Expr::StringLit(s, _) => Some(s.to_string()),
                        _ => None,
                    };
                    if let Some(k) = key
                        && let Some(param_ty) = ctx.pack_param_types.get(k.as_str())
                    {
                        return param_ty.clone();
                    }
                }
                match name.as_str() {
                    "atomic_cas" => {
                        if let Some(val_arg) = args.get(2).or_else(|| args.get(1)) {
                            return infer_glsl_type(&val_arg.value, ctx);
                        }
                        if let Some(ptr_arg) = args.first() {
                            let ptr_ty = infer_glsl_type(&ptr_arg.value, ctx);
                            if let Some(bda) = ctx.bda_types.get(&ptr_ty) {
                                return bda.elem_glsl.clone();
                            }
                        }
                        return "uint".to_owned();
                    }
                    "atomic_add" | "atomic_min" | "atomic_max" | "atomic_and" | "atomic_or"
                    | "atomic_xor" | "atomic_exchange" => {
                        if let Some(val_arg) = args.get(1) {
                            return infer_glsl_type(&val_arg.value, ctx);
                        }
                        return "uint".to_owned();
                    }
                    "min" | "max" | "clamp" | "abs" | "mix" | "nonuniform" => {
                        if let Some(first) = args.first() {
                            return infer_glsl_type(&first.value, ctx);
                        }
                        return "float".to_owned();
                    }
                    "u32"
                    | "subgroup_ballot_bit_count"
                    | "subgroup_ballot_exclusive_bit_count"
                    | "subgroup_broadcast_first" => return "uint".to_owned(),
                    "i32" => return "int".to_owned(),
                    "f32" => return "float".to_owned(),
                    "u64" => return "uint64_t".to_owned(),
                    "i64" => return "int64_t".to_owned(),
                    "vec2" | "vec3" | "vec4" | "ivec2" | "ivec3" | "ivec4" | "uvec2" | "uvec3"
                    | "uvec4" | "bvec2" | "bvec3" | "bvec4" | "mat2" | "mat3" | "mat4" => {
                        return name;
                    }
                    "subgroup_ballot" => return "uvec4".to_owned(),
                    "reflect"
                    | "normalize"
                    | "cross"
                    | "mc::reconstruct_world_pos"
                    | "mc::view_dir"
                    | "mc::sample_sky" => return "vec3".to_owned(),
                    other => {
                        let key = other.replace("::", "_");
                        if let Some(ret) = ctx.fn_returns.get(key.as_str()) {
                            return ret.clone();
                        }
                    }
                }
            }
            "float".to_owned()
        }
        Expr::Binary { op, lhs, .. } => match op {
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::And
            | BinaryOp::Or => "bool".to_owned(),
            _ => infer_glsl_type(lhs, ctx),
        },
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            if let Some(ref tail) = then_branch.tail_expr {
                infer_glsl_type(tail, ctx)
            } else if let Some(eb) = else_branch {
                infer_glsl_type(eb, ctx)
            } else {
                "float".to_owned()
            }
        }
        Expr::Block(b) => b
            .tail_expr
            .as_ref()
            .map_or_else(|| "float".to_owned(), |t| infer_glsl_type(t, ctx)),
        _ => "float".to_owned(),
    }
}

fn infer_range_type(start: &Expr, end: &Expr, ctx: &EmitCtx) -> &'static str {
    let s = infer_glsl_type(start, ctx);
    let e = infer_glsl_type(end, ctx);
    if s == "int" && e == "int" {
        "int"
    } else {
        "uint"
    }
}

fn decompose_range_iter(expr: &Expr) -> Option<(&Expr, &Expr, bool, Option<&Expr>)> {
    if let Expr::MethodCall {
        receiver,
        method,
        args,
        ..
    } = expr
        && method.name == "step_by"
        && let Some(step) = args.first().map(|a| &a.value)
        && let Some((start, end, inclusive, _)) = decompose_range_iter(receiver)
    {
        return Some((start, end, inclusive, Some(step)));
    }
    if let Expr::Binary {
        op: op @ (BinaryOp::Range | BinaryOp::RangeInclusive),
        lhs,
        rhs,
        ..
    } = expr
    {
        return Some((lhs, rhs, matches!(op, BinaryOp::RangeInclusive), None));
    }
    None
}

fn format_int_literal(raw: &str) -> String {
    let (body, glsl_suffix) = if let Some(s) = raw.strip_suffix("u64") {
        (s, "ul")
    } else if let Some(s) = raw.strip_suffix("i64") {
        (s, "l")
    } else if let Some(s) = raw.strip_suffix("u32") {
        (s, "u")
    } else if let Some(s) = raw.strip_suffix("i32") {
        (s, "")
    } else if let Some(s) = raw.strip_suffix('u') {
        (s, "u")
    } else if let Some(s) = raw.strip_suffix('i') {
        (s, "")
    } else {
        (raw, "")
    };

    if let Some(bin) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        let digits = bin.replace('_', "");
        if let Ok(val) = u128::from_str_radix(&digits, 2) {
            return format!("0x{val:X}{glsl_suffix}");
        }
    } else if let Some(oct) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        let digits = oct.replace('_', "");
        if let Ok(val) = u128::from_str_radix(&digits, 8) {
            return format!("0x{val:X}{glsl_suffix}");
        }
    }

    format!("{body}{glsl_suffix}")
}

fn format_float_literal(raw: &str) -> String {
    if let Some(s) = raw.strip_suffix("f32") {
        s.to_owned()
    } else if let Some(s) = raw.strip_suffix("f16") {
        format!("{s}hf")
    } else {
        raw.to_owned()
    }
}

fn format_path_expr(path: &Path, ctx: &EmitCtx) -> String {
    let joined = path.joined();
    match joined.as_str() {
        "mc::RayFlags::OPAQUE" => "gl_RayFlagsOpaqueEXT".to_owned(),
        "mc::HitType::Triangle" => "gl_RayQueryCommittedIntersectionTriangleEXT".to_owned(),
        "mc::tlas" => "tlas".to_owned(),
        "mc::bindless" => "bindless".to_owned(),
        "mc::sharc_config" | "mc::sharc" => "mc_sharc_config".to_owned(),
        "mc::restir_reservoir" | "mc::reservoirs" => "mc_restir_reservoir".to_owned(),
        "world" if ctx.imported_mc_world && !ctx.shadowed_world => "mc_world".to_owned(),
        _ => escape_glsl_ident(&joined.replace("::", "_")),
    }
}

fn map_call_name(name: &str) -> String {
    match name {
        "workgroup_barrier" => "barrier".to_owned(),
        "atomic_add" => "atomicAdd".to_owned(),
        "atomic_cas" => "atomicCompSwap".to_owned(),
        "subgroup_ballot" => "subgroupBallot".to_owned(),
        "subgroup_ballot_bit_count" => "subgroupBallotBitCount".to_owned(),
        "subgroup_ballot_exclusive_bit_count" => "subgroupBallotExclusiveBitCount".to_owned(),
        "subgroup_broadcast_first" => "subgroupBroadcastFirst".to_owned(),
        "nonuniform" => "nonuniformEXT".to_owned(),
        "u32" => "uint".to_owned(),
        "i32" => "int".to_owned(),
        "f32" => "float".to_owned(),
        "u64" => "uint64_t".to_owned(),
        "i64" => "int64_t".to_owned(),
        other => other.replace("::", "_"),
    }
}

const fn binary_op_glsl(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Rem => "%",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitOr => "|",
        BinaryOp::BitXor => "^",
        BinaryOp::Shl => "<<",
        BinaryOp::Shr => ">>",
        BinaryOp::Eq => "==",
        BinaryOp::Ne => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
        BinaryOp::And => "&&",
        BinaryOp::Or => "||",
        BinaryOp::Assign => "=",
        BinaryOp::AddAssign => "+=",
        BinaryOp::SubAssign => "-=",
        BinaryOp::MulAssign => "*=",
        BinaryOp::DivAssign => "/=",
        BinaryOp::RemAssign => "%=",
        BinaryOp::BitAndAssign => "&=",
        BinaryOp::BitOrAssign => "|=",
        BinaryOp::BitXorAssign => "^=",
        BinaryOp::Range => "..",
        BinaryOp::RangeInclusive => "..=",
    }
}

const fn is_assignment_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Assign
            | BinaryOp::AddAssign
            | BinaryOp::SubAssign
            | BinaryOp::MulAssign
            | BinaryOp::DivAssign
            | BinaryOp::RemAssign
            | BinaryOp::BitAndAssign
            | BinaryOp::BitOrAssign
            | BinaryOp::BitXorAssign
    )
}

fn map_builtin_var(name: &str) -> Option<&'static str> {
    match name {
        "global_invocation_id" => Some("gl_GlobalInvocationID"),
        "local_invocation_id" => Some("gl_LocalInvocationID"),
        "local_invocation_index" => Some("gl_LocalInvocationIndex"),
        "workgroup_id" => Some("gl_WorkGroupID"),
        "num_workgroups" => Some("gl_NumWorkGroups"),
        "vertex_index" => Some("gl_VertexIndex"),
        "instance_index" => Some("gl_InstanceIndex"),
        "position" => Some("gl_Position"),
        "frag_coord" => Some("gl_FragCoord"),
        "front_facing" => Some("gl_FrontFacing"),
        "point_coord" => Some("gl_PointCoord"),
        _ => None,
    }
}

pub(crate) fn has_attr(attrs: &[Attribute], name: &str) -> bool {
    attrs
        .iter()
        .any(|a| a.entries.iter().any(|e| e.name.name == name))
}

fn extract_layout_attr(attrs: &[Attribute]) -> Option<&'static str> {
    for attr in attrs {
        for entry in &attr.entries {
            if entry.name.name == "layout"
                && let Some(Expr::Path { path, .. }) = entry.args.first()
            {
                return match path.joined().as_str() {
                    "scalar" => Some("scalar"),
                    "std430" => Some("std430"),
                    "std140" => Some("std140"),
                    _ => None,
                };
            }
        }
    }
    None
}

fn extract_builtin_attr(attrs: &[Attribute]) -> Option<&str> {
    for attr in attrs {
        for entry in &attr.entries {
            if entry.name.name == "builtin"
                && let Some(Expr::Path { path, .. }) = entry.args.first()
                && let Some(seg) = path.segments.last()
            {
                return Some(seg.name.as_str());
            }
        }
    }
    None
}

fn extract_location_attr(attrs: &[Attribute]) -> Option<u32> {
    for attr in attrs {
        for entry in &attr.entries {
            if entry.name.name == "location"
                && let Some(first) = entry.args.first()
            {
                return parse_u32_lit(first);
            }
        }
    }
    None
}

fn extract_binding_attr(attrs: &[Attribute]) -> Option<(u32, u32)> {
    for attr in attrs {
        for entry in &attr.entries {
            if entry.name.name == "binding" && entry.args.len() == 2 {
                let mut set: Option<u32> = None;
                let mut binding: Option<u32> = None;
                let mut all_named = true;
                for arg in &entry.args {
                    if let Expr::Binary {
                        op: BinaryOp::Assign,
                        lhs,
                        rhs,
                        ..
                    } = arg
                        && let Expr::Path { path, .. } = lhs.as_ref()
                        && let Some(val) = parse_u32_lit(rhs.as_ref())
                    {
                        match path.joined().as_str() {
                            "set" if set.is_none() => set = Some(val),
                            "binding" if binding.is_none() => binding = Some(val),
                            _ => {
                                all_named = false;
                                break;
                            }
                        }
                    } else {
                        all_named = false;
                        break;
                    }
                }
                if all_named && let (Some(s), Some(b)) = (set, binding) {
                    return Some((s, b));
                }
                if let (Some(s), Some(b)) = (
                    entry.args.first().and_then(parse_u32_lit),
                    entry.args.get(1).and_then(parse_u32_lit),
                ) {
                    return Some((s, b));
                }
            }
        }
    }
    None
}

fn extract_entry_stage(attrs: &[Attribute]) -> Option<EntryStage> {
    for attr in attrs {
        for entry in &attr.entries {
            if entry.name.name == "stage" {
                match entry.args.first() {
                    Some(Expr::Call { callee, args, .. }) => {
                        if let Expr::Path { path, .. } = callee.as_ref() {
                            match path.joined().as_str() {
                                "compute" => {
                                    let x = args
                                        .first()
                                        .and_then(|a| parse_u32_lit(&a.value))
                                        .unwrap_or(1);
                                    let y = args
                                        .get(1)
                                        .and_then(|a| parse_u32_lit(&a.value))
                                        .unwrap_or(1);
                                    let z = args
                                        .get(2)
                                        .and_then(|a| parse_u32_lit(&a.value))
                                        .unwrap_or(1);
                                    return Some(EntryStage::Compute(x, y, z));
                                }
                                "vertex" => return Some(EntryStage::Vertex),
                                "fragment" => return Some(EntryStage::Fragment),
                                _ => {}
                            }
                        }
                    }
                    Some(Expr::Path { path, .. }) => match path.joined().as_str() {
                        "vertex" => return Some(EntryStage::Vertex),
                        "fragment" => return Some(EntryStage::Fragment),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
    None
}

fn strip_u32_suffix(raw: &str) -> &str {
    for suffix in ["u64", "i64", "u32", "i32", "u", "i"] {
        if let Some(stripped) = raw.strip_suffix(suffix) {
            return stripped;
        }
    }
    raw
}

pub(crate) fn parse_u32_lit(expr: &Expr) -> Option<u32> {
    if let Expr::IntLit(raw, _) = expr {
        let clean = strip_u32_suffix(raw).replace('_', "");
        if let Some(hex) = clean
            .strip_prefix("0x")
            .or_else(|| clean.strip_prefix("0X"))
        {
            return u32::from_str_radix(hex, 16).ok();
        }
        if let Some(bin) = clean
            .strip_prefix("0b")
            .or_else(|| clean.strip_prefix("0B"))
        {
            return u32::from_str_radix(bin, 2).ok();
        }
        if let Some(oct) = clean
            .strip_prefix("0o")
            .or_else(|| clean.strip_prefix("0O"))
        {
            return u32::from_str_radix(oct, 8).ok();
        }
        return clean.parse::<u32>().ok();
    }
    None
}

fn write_indent(indent: usize, out: &mut String) {
    for _ in 0..indent {
        out.push_str("    ");
    }
}

// ============================================================================
// AST Pre-Scan for Extensions, `mc::*` Globals, and BDA Buffer References
// ============================================================================

fn scan_unit_with_options(unit: &SourceUnit, opts: EmitOptions, scan: &mut FeatureScan) {
    for imp in &unit.imports {
        let prefix = imp.prefix.joined();
        if prefix.contains("subgroup") {
            scan.extensions.insert("GL_KHR_shader_subgroup_basic");
            scan.extensions.insert("GL_KHR_shader_subgroup_ballot");
        }
        if prefix == "mc" {
            match &imp.tree {
                ImportTree::Group(names) => {
                    for id in names {
                        if id.name == "world" {
                            scan.imported_mc_world = true;
                        }
                        mark_mc_global(&id.name, scan);
                    }
                }
                ImportTree::Glob => {
                    scan.imported_mc_world = true;
                    mark_mc_global("world", scan);
                    mark_mc_global("tlas", scan);
                    mark_mc_global("instances", scan);
                    mark_mc_global("sharc_config", scan);
                    mark_mc_global("lights", scan);
                    mark_mc_global("restir_reservoir", scan);
                    mark_mc_global("bindless", scan);
                }
                ImportTree::Single => {}
            }
        } else if let Some(rest) = prefix.strip_prefix("mc::") {
            if rest == "world" {
                scan.imported_mc_world = true;
            }
            mark_mc_global(rest, scan);
        }
    }
    for item in &unit.items {
        match item {
            Item::Const(c) => {
                scan_type(&c.ty, scan);
                scan_expr_with_options(&c.value, opts, scan);
            }
            Item::WorkgroupVar(w) => {
                scan_type(&w.ty, scan);
            }
            Item::Struct(s) => {
                scan_attrs(&s.attrs, scan);
                if (has_attr(&s.attrs, "push_constant")
                    && extract_layout_attr(&s.attrs).unwrap_or("scalar") == "scalar")
                    || extract_layout_attr(&s.attrs) == Some("scalar")
                {
                    scan.extensions.insert("GL_EXT_scalar_block_layout");
                }
                if s.generics.is_empty() {
                    for f in &s.fields {
                        scan_type(&f.ty, scan);
                    }
                } else {
                    let matching: Vec<MonomorphKey> = scan
                        .generic_instantiations
                        .iter()
                        .filter(|k| k.struct_name == s.name.name)
                        .cloned()
                        .collect();
                    for key in matching {
                        let mut subst = IndexMap::new();
                        for (g, concrete) in s.generics.iter().zip(&key.type_args) {
                            subst.insert(g.name.name.clone(), concrete.clone());
                        }
                        for f in &s.fields {
                            scan_type_with_subst(&f.ty, &subst, scan);
                        }
                    }
                }
            }
            Item::Trait(t) => {
                for m in &t.methods {
                    scan_fn_into_with_options(m, opts, scan);
                }
            }
            Item::Impl(imp) => {
                if imp.generics.is_empty() {
                    for m in &imp.methods {
                        scan_fn_into_with_options(m, opts, scan);
                    }
                } else if let Some(base_name) = impl_target_base_name(&imp.target) {
                    let matching: Vec<MonomorphKey> = scan
                        .generic_instantiations
                        .iter()
                        .filter(|k| k.struct_name.as_str() == base_name)
                        .cloned()
                        .collect();
                    for key in matching {
                        let mut subst = IndexMap::new();
                        apply_impl_subst(imp, &key.type_args, &mut subst);
                        for m in &imp.methods {
                            scan_fn_into_with_subst(m, &subst, opts, scan);
                        }
                    }
                }
            }
            Item::Function(f) => {
                if extract_entry_stage(&f.attrs).is_some() {
                    let mut next_binding = 1u32;
                    for p in &f.params {
                        if extract_builtin_attr(&p.attrs).is_none()
                            && extract_location_attr(&p.attrs).is_none()
                            && !has_attr(&p.attrs, "push_constant")
                            && !has_attr(&p.attrs, "pack_resource")
                            && p.ty.is_some()
                        {
                            next_binding += 1;
                        }
                    }
                    scan.next_per_pass_binding = scan.next_per_pass_binding.max(next_binding);
                }
                scan_fn_into_with_options(f, opts, scan);
            }
            _ => {}
        }
    }
}

fn mark_mc_global(name: &str, scan: &mut FeatureScan) {
    match name {
        "world" => {
            scan.extensions
                .insert("GL_EXT_shader_explicit_arithmetic_types_int64");
        }
        "tlas" => {
            scan.uses_tlas = true;
            scan.extensions.insert("GL_EXT_ray_query");
        }
        "instances" => {
            scan.uses_instances = true;
        }
        "sharc_config" | "sharc" => {
            scan.uses_sharc_config = true;
            scan.extensions
                .insert("GL_EXT_shader_explicit_arithmetic_types_int64");
        }
        "lights" => {
            scan.uses_lights = true;
        }
        "restir_reservoir" | "reservoirs" => {
            scan.uses_restir_reservoir = true;
        }
        "bindless" => {
            scan.uses_bindless = true;
            scan.extensions.insert("GL_EXT_nonuniform_qualifier");
        }
        "reconstruct_world_pos" => {
            scan.mc_helpers.insert("mc_reconstruct_world_pos");
        }
        "view_dir" => {
            scan.mc_helpers.insert("mc_view_dir");
        }
        "sample_sky" => {
            scan.mc_helpers.insert("mc_sample_sky");
        }
        _ => {}
    }
}

pub(crate) fn scan_fn_into(f: &FnDecl, scan: &mut FeatureScan) {
    scan_fn_into_with_options(f, EmitOptions::default(), scan);
}

fn scan_fn_into_with_options(f: &FnDecl, opts: EmitOptions, scan: &mut FeatureScan) {
    let empty = IndexMap::new();
    scan_fn_into_with_subst(f, &empty, opts, scan);
}

fn scan_fn_into_with_subst(
    f: &FnDecl,
    subst: &IndexMap<SmolStr, String>,
    opts: EmitOptions,
    scan: &mut FeatureScan,
) {
    scan_attrs(&f.attrs, scan);
    for p in &f.params {
        if let Some(ref ty) = p.ty {
            scan_type_with_subst(ty, subst, scan);
        }
    }
    if let Some(ref ret) = f.return_ty {
        scan_type_with_subst(ret, subst, scan);
    }
    if let Some(ref body) = f.body {
        scan_block_with_subst(body, subst, opts, scan);
    }
}

fn scan_attrs(attrs: &[Attribute], scan: &mut FeatureScan) {
    for attr in attrs {
        for entry in &attr.entries {
            match entry.name.name.as_str() {
                "cap" => {
                    for arg in &entry.args {
                        if let Expr::Path { path, .. } = arg {
                            match path.joined().as_str() {
                                "ray_query" => {
                                    scan.extensions.insert("GL_EXT_ray_query");
                                }
                                "int64_atomics" => {
                                    scan.extensions
                                        .insert("GL_EXT_shader_explicit_arithmetic_types_int64");
                                    scan.extensions.insert("GL_EXT_shader_atomic_int64");
                                }
                                "float16" => {
                                    scan.extensions
                                        .insert("GL_EXT_shader_explicit_arithmetic_types_float16");
                                }
                                "maximal_reconvergence" => {
                                    scan.extensions.insert("GL_EXT_maximal_reconvergence");
                                }
                                "float_atomics" => {
                                    scan.extensions.insert("GL_EXT_shader_atomic_float");
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "maximally_reconverges" => {
                    scan.extensions.insert("GL_EXT_maximal_reconvergence");
                }
                "unroll" => {
                    scan.extensions.insert("GL_EXT_control_flow_attributes");
                }
                _ => {}
            }
        }
    }
}

pub(crate) fn scan_type(ty: &TypeExpr, scan: &mut FeatureScan) {
    let empty = IndexMap::new();
    scan_type_with_subst(ty, &empty, scan);
}

fn scan_type_with_subst(ty: &TypeExpr, subst: &IndexMap<SmolStr, String>, scan: &mut FeatureScan) {
    match ty {
        TypeExpr::Array { element, .. } => scan_type_with_subst(element, subst, scan),
        TypeExpr::Named { path, args, .. } => {
            let joined = path.joined();
            if let Some(bda) = parse_bda_type_with_subst(&joined, args, subst) {
                scan.extensions.insert("GL_EXT_buffer_reference");
                scan.extensions.insert("GL_EXT_buffer_reference2");
                scan.extensions
                    .insert("GL_EXT_shader_explicit_arithmetic_types_int64");
                scan.bda_refs.insert(bda);
            }
            match joined.as_str() {
                "u64" | "i64" => {
                    scan.extensions
                        .insert("GL_EXT_shader_explicit_arithmetic_types_int64");
                }
                "f16" | "f16vec2" | "f16vec3" | "f16vec4" => {
                    scan.extensions
                        .insert("GL_EXT_shader_explicit_arithmetic_types_float16");
                }
                "RayQuery" => {
                    scan.extensions.insert("GL_EXT_ray_query");
                }
                "Buffer" | "RWBuffer" => {
                    scan.extensions.insert("GL_EXT_scalar_block_layout");
                }
                _ => {}
            }
            for arg in args {
                if let TypeArg::Type(inner) = arg {
                    scan_type_with_subst(inner, subst, scan);
                }
            }
        }
    }
}

fn scan_block_with_subst(
    block: &Block,
    subst: &IndexMap<SmolStr, String>,
    opts: EmitOptions,
    scan: &mut FeatureScan,
) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { ty, init, .. } => {
                if let Some(t) = ty {
                    scan_type_with_subst(t, subst, scan);
                }
                if let Some(e) = init {
                    scan_expr_with_subst(e, subst, opts, scan);
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(e) = value {
                    scan_expr_with_subst(e, subst, opts, scan);
                }
            }
            Stmt::For {
                attrs, iter, body, ..
            } => {
                scan_attrs(attrs, scan);
                scan_expr_with_subst(iter, subst, opts, scan);
                scan_block_with_subst(body, subst, opts, scan);
            }
            Stmt::While {
                attrs, cond, body, ..
            } => {
                scan_attrs(attrs, scan);
                scan_expr_with_subst(cond, subst, opts, scan);
                scan_block_with_subst(body, subst, opts, scan);
            }
            Stmt::Loop { attrs, body, .. } => {
                scan_attrs(attrs, scan);
                scan_block_with_subst(body, subst, opts, scan);
            }
            Stmt::Switch {
                selector,
                cases,
                default,
                ..
            } => {
                scan_expr_with_subst(selector, subst, opts, scan);
                for case in cases {
                    scan_expr_with_subst(&case.value, subst, opts, scan);
                    scan_block_with_subst(&case.body, subst, opts, scan);
                }
                if let Some(def) = default {
                    scan_block_with_subst(def, subst, opts, scan);
                }
            }
            Stmt::Expr { expr, .. } => scan_expr_with_subst(expr, subst, opts, scan),
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }
    if let Some(ref tail) = block.tail_expr {
        scan_expr_with_subst(tail, subst, opts, scan);
    }
}

fn is_bindless_receiver(expr: &Expr) -> bool {
    match expr {
        Expr::Path { path, .. } => matches!(path.joined().as_str(), "bindless" | "mc::bindless"),
        Expr::Index { receiver, .. } => is_bindless_receiver(receiver),
        _ => false,
    }
}

pub(crate) fn scan_expr(expr: &Expr, scan: &mut FeatureScan) {
    scan_expr_with_options(expr, EmitOptions::default(), scan);
}

fn scan_expr_with_options(expr: &Expr, opts: EmitOptions, scan: &mut FeatureScan) {
    let empty = IndexMap::new();
    scan_expr_with_subst(expr, &empty, opts, scan);
}

fn scan_expr_with_subst(
    expr: &Expr,
    subst: &IndexMap<SmolStr, String>,
    opts: EmitOptions,
    scan: &mut FeatureScan,
) {
    match expr {
        Expr::IntLit(raw, _) => {
            if raw.ends_with("u64") || raw.ends_with("i64") {
                scan.extensions
                    .insert("GL_EXT_shader_explicit_arithmetic_types_int64");
            }
        }
        Expr::FloatLit(raw, _) => {
            if raw.ends_with("f16") {
                scan.extensions
                    .insert("GL_EXT_shader_explicit_arithmetic_types_float16");
            }
        }
        Expr::Path { path, .. } => {
            let joined = path.joined();
            if joined == "nonuniform" {
                scan.extensions.insert("GL_EXT_nonuniform_qualifier");
            } else if joined.starts_with("subgroup_") {
                scan.extensions.insert("GL_KHR_shader_subgroup_basic");
                scan.extensions.insert("GL_KHR_shader_subgroup_ballot");
            } else if let Some(rest) = joined.strip_prefix("mc::") {
                mark_mc_global(rest, scan);
            }
        }
        Expr::ArrayLit { elements, .. } => {
            for e in elements {
                scan_expr_with_subst(e, subst, opts, scan);
            }
        }
        Expr::StructLit { fields, .. } => {
            for (_, v) in fields {
                scan_expr_with_subst(v, subst, opts, scan);
            }
        }
        Expr::Unary { operand, .. } => scan_expr_with_subst(operand, subst, opts, scan),
        Expr::Binary { lhs, rhs, .. } => {
            scan_expr_with_subst(lhs, subst, opts, scan);
            scan_expr_with_subst(rhs, subst, opts, scan);
        }
        Expr::Call { callee, args, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref() {
                let joined = path.joined();
                if joined == "mc::debug_write"
                    && let Some(first) = args.first()
                {
                    let channel = match &first.value {
                        Expr::StringLit(s, _) => sanitize_glsl_ident(s),
                        Expr::Path { path: p, .. } => sanitize_glsl_ident(&p.joined()),
                        _ => "default".to_owned(),
                    };
                    scan.debug_channels.insert(channel);
                } else if matches!(joined.as_str(), "param" | "mc::param" | "pack::param")
                    && let Some(first) = args.first()
                {
                    match &first.value {
                        Expr::Path { path: p, .. } => {
                            scan.pack_param_keys.insert(p.joined().replace("::", "_"));
                        }
                        Expr::StringLit(s, _) => {
                            scan.pack_param_keys.insert(s.to_string());
                        }
                        _ => {}
                    }
                }
            }
            scan_expr_with_subst(callee, subst, opts, scan);
            for a in args {
                scan_expr_with_subst(&a.value, subst, opts, scan);
            }
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
            ..
        } => {
            // Only `.sample()` / `.sample_lod()` require a sampler; `.fetch()` lowers to
            // `texelFetch(...)` directly without `vsl_default_sampler`.
            if matches!(method.name.as_str(), "sample" | "sample_lod")
                && (opts.bindless_abi == BindlessAbi::UpstreamUnsized
                    || !is_bindless_receiver(receiver))
            {
                scan.uses_sampled_textures = true;
            }
            scan_expr_with_subst(receiver, subst, opts, scan);
            for a in args {
                scan_expr_with_subst(&a.value, subst, opts, scan);
            }
        }
        Expr::Field {
            receiver, field, ..
        } => {
            if let Expr::Path { path, .. } = receiver.as_ref()
                && matches!(
                    path.joined().as_str(),
                    "mc::params" | "params" | "pack::params"
                )
            {
                scan.pack_param_keys.insert(field.name.to_string());
            }
            scan_expr_with_subst(receiver, subst, opts, scan);
        }
        Expr::Index {
            receiver, index, ..
        } => {
            scan_expr_with_subst(receiver, subst, opts, scan);
            scan_expr_with_subst(index, subst, opts, scan);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            scan_expr_with_subst(cond, subst, opts, scan);
            scan_block_with_subst(then_branch, subst, opts, scan);
            if let Some(eb) = else_branch {
                scan_expr_with_subst(eb, subst, opts, scan);
            }
        }
        Expr::Block(b) => scan_block_with_subst(b, subst, opts, scan),
        Expr::BoolLit(_, _) | Expr::StringLit(_, _) | Expr::NoneLit(_) => {}
    }
}

// ============================================================================
// Generic Struct Monomorphization Discovery
// ============================================================================

fn collect_generic_instantiations(unit: &SourceUnit) -> IndexSet<MonomorphKey> {
    let mut generic_structs: IndexMap<SmolStr, &StructDecl> = IndexMap::new();
    for item in &unit.items {
        if let Item::Struct(s) = item
            && !s.generics.is_empty()
        {
            generic_structs.insert(s.name.name.clone(), s);
        }
    }
    if generic_structs.is_empty() {
        return IndexSet::new();
    }

    let mut out: IndexSet<MonomorphKey> = IndexSet::new();

    // 1. Collect concrete `impl Trait for ConcreteStruct` definitions to satisfy bounded generics.
    let mut trait_impls: IndexMap<SmolStr, IndexSet<String>> = IndexMap::new();
    for item in &unit.items {
        if let Item::Impl(imp) = item
            && imp.generics.is_empty()
            && let Some(ref trait_path) = imp.trait_path
        {
            let concrete_ty = type_to_glsl(&imp.target).0;
            if let Some(last) = trait_path.segments.last() {
                trait_impls
                    .entry(last.name.clone())
                    .or_default()
                    .insert(concrete_ty.clone());
            }
            trait_impls
                .entry(SmolStr::new(trait_path.joined()))
                .or_default()
                .insert(concrete_ty);
        }
    }

    for (struct_name, s_decl) in &generic_structs {
        let mut param_candidates: Vec<Vec<String>> = Vec::with_capacity(s_decl.generics.len());
        let mut all_bounded = true;
        for g in &s_decl.generics {
            let Some(ref bound) = g.bound else {
                all_bounded = false;
                break;
            };
            let key = bound.joined();
            let impls = trait_impls.get(key.as_str()).or_else(|| {
                bound
                    .segments
                    .last()
                    .and_then(|seg| trait_impls.get(&seg.name))
            });
            if let Some(types) = impls
                && !types.is_empty()
            {
                param_candidates.push(types.iter().cloned().collect());
            } else {
                all_bounded = false;
                break;
            }
        }
        if all_bounded {
            let mut acc = Vec::with_capacity(param_candidates.len());
            emit_cartesian_instantiations(struct_name, &param_candidates, 0, &mut acc, &mut out);
        }
    }

    // 2. Scan all type annotations, fields, signatures, and struct literals across the unit.
    let empty_generics: IndexSet<SmolStr> = IndexSet::new();
    for item in &unit.items {
        match item {
            Item::Const(c) => {
                collect_instantiations_in_type(&c.ty, &generic_structs, &empty_generics, &mut out);
            }
            Item::WorkgroupVar(w) => {
                collect_instantiations_in_type(&w.ty, &generic_structs, &empty_generics, &mut out);
            }
            Item::Struct(s) => {
                let active: IndexSet<SmolStr> =
                    s.generics.iter().map(|g| g.name.name.clone()).collect();
                for f in &s.fields {
                    collect_instantiations_in_type(&f.ty, &generic_structs, &active, &mut out);
                }
            }
            Item::Trait(t) => {
                for m in &t.methods {
                    collect_instantiations_in_fn(m, &generic_structs, &empty_generics, &mut out);
                }
            }
            Item::Impl(imp) => {
                let active: IndexSet<SmolStr> =
                    imp.generics.iter().map(|g| g.name.name.clone()).collect();
                collect_instantiations_in_type(&imp.target, &generic_structs, &active, &mut out);
                for m in &imp.methods {
                    collect_instantiations_in_fn(m, &generic_structs, &active, &mut out);
                }
            }
            Item::Function(f) => {
                collect_instantiations_in_fn(f, &generic_structs, &empty_generics, &mut out);
            }
            _ => {}
        }
    }

    // 3. Fixed-point expansion for generic structs referencing other generic structs in fields.
    let mut cursor = 0usize;
    while cursor < out.len() {
        let Some(key) = out.get_index(cursor).cloned() else {
            break;
        };
        cursor += 1;
        if let Some(s_decl) = generic_structs.get(&key.struct_name) {
            let mut subst = IndexMap::new();
            for (g, concrete) in s_decl.generics.iter().zip(&key.type_args) {
                subst.insert(g.name.name.clone(), concrete.clone());
            }
            for f in &s_decl.fields {
                collect_instantiations_in_type_with_subst(
                    &f.ty,
                    &generic_structs,
                    &subst,
                    &mut out,
                );
            }
        }
    }

    out
}

fn emit_cartesian_instantiations(
    struct_name: &SmolStr,
    candidates: &[Vec<String>],
    depth: usize,
    current: &mut Vec<String>,
    out: &mut IndexSet<MonomorphKey>,
) {
    if depth == candidates.len() {
        out.insert(MonomorphKey {
            struct_name: struct_name.clone(),
            type_args: current.clone(),
        });
        return;
    }
    for ty in &candidates[depth] {
        current.push(ty.clone());
        emit_cartesian_instantiations(struct_name, candidates, depth + 1, current, out);
        current.pop();
    }
}

fn type_contains_generic_param(ty: &TypeExpr, active_generics: &IndexSet<SmolStr>) -> bool {
    if active_generics.is_empty() {
        return false;
    }
    match ty {
        TypeExpr::Array { element, .. } => type_contains_generic_param(element, active_generics),
        TypeExpr::Named { path, args, .. } => {
            if path.segments.len() == 1 && active_generics.contains(&path.segments[0].name) {
                return true;
            }
            args.iter().any(|arg| match arg {
                TypeArg::Type(inner) => type_contains_generic_param(inner, active_generics),
                TypeArg::NamedExpr { .. } => false,
            })
        }
    }
}

fn collect_instantiations_in_type(
    ty: &TypeExpr,
    generic_structs: &IndexMap<SmolStr, &StructDecl>,
    active_generics: &IndexSet<SmolStr>,
    out: &mut IndexSet<MonomorphKey>,
) {
    match ty {
        TypeExpr::Array { element, .. } => {
            collect_instantiations_in_type(element, generic_structs, active_generics, out);
        }
        TypeExpr::Named { path, args, .. } => {
            for arg in args {
                if let TypeArg::Type(inner) = arg {
                    collect_instantiations_in_type(inner, generic_structs, active_generics, out);
                }
            }
            if args.is_empty() {
                return;
            }
            let base = path.joined().replace("::", "_");
            let Some(s_decl) = generic_structs.get(base.as_str()) else {
                return;
            };
            let type_arg_exprs: Vec<&TypeExpr> = args
                .iter()
                .filter_map(|a| match a {
                    TypeArg::Type(t) => Some(t),
                    TypeArg::NamedExpr { .. } => None,
                })
                .collect();
            if type_arg_exprs.len() != s_decl.generics.len() {
                return;
            }
            if type_arg_exprs
                .iter()
                .any(|t| type_contains_generic_param(t, active_generics))
            {
                return;
            }
            let concrete_args: Vec<String> =
                type_arg_exprs.iter().map(|t| type_to_glsl(t).0).collect();
            out.insert(MonomorphKey {
                struct_name: s_decl.name.name.clone(),
                type_args: concrete_args,
            });
        }
    }
}

fn collect_instantiations_in_type_with_subst(
    ty: &TypeExpr,
    generic_structs: &IndexMap<SmolStr, &StructDecl>,
    subst: &IndexMap<SmolStr, String>,
    out: &mut IndexSet<MonomorphKey>,
) {
    match ty {
        TypeExpr::Array { element, .. } => {
            collect_instantiations_in_type_with_subst(element, generic_structs, subst, out);
        }
        TypeExpr::Named { path, args, .. } => {
            for arg in args {
                if let TypeArg::Type(inner) = arg {
                    collect_instantiations_in_type_with_subst(inner, generic_structs, subst, out);
                }
            }
            if args.is_empty() {
                return;
            }
            let base = path.joined().replace("::", "_");
            let Some(s_decl) = generic_structs.get(base.as_str()) else {
                return;
            };
            let concrete_args: Vec<String> = args
                .iter()
                .filter_map(|a| match a {
                    TypeArg::Type(t) => Some(type_to_glsl_with_subst(t, subst).0),
                    TypeArg::NamedExpr { .. } => None,
                })
                .collect();
            if concrete_args.len() == s_decl.generics.len() {
                out.insert(MonomorphKey {
                    struct_name: s_decl.name.name.clone(),
                    type_args: concrete_args,
                });
            }
        }
    }
}

fn collect_instantiations_in_fn(
    f: &FnDecl,
    generic_structs: &IndexMap<SmolStr, &StructDecl>,
    parent_generics: &IndexSet<SmolStr>,
    out: &mut IndexSet<MonomorphKey>,
) {
    let mut active = parent_generics.clone();
    for g in &f.generics {
        active.insert(g.name.name.clone());
    }
    let mut local_types: IndexMap<SmolStr, String> = IndexMap::new();
    for p in &f.params {
        if let Some(ref ty) = p.ty {
            collect_instantiations_in_type(ty, generic_structs, &active, out);
            if !type_contains_generic_param(ty, &active) {
                local_types.insert(p.name.name.clone(), type_to_glsl(ty).0);
            }
        }
    }
    if let Some(ref ret) = f.return_ty {
        collect_instantiations_in_type(ret, generic_structs, &active, out);
    }
    if let Some(ref body) = f.body {
        collect_instantiations_in_block(body, generic_structs, &active, &mut local_types, out);
    }
}

fn collect_instantiations_in_block(
    block: &Block,
    generic_structs: &IndexMap<SmolStr, &StructDecl>,
    active_generics: &IndexSet<SmolStr>,
    local_types: &mut IndexMap<SmolStr, String>,
    out: &mut IndexSet<MonomorphKey>,
) {
    let saved_locals = local_types.clone();
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { name, ty, init, .. } => {
                if let Some(t) = ty {
                    collect_instantiations_in_type(t, generic_structs, active_generics, out);
                    if !type_contains_generic_param(t, active_generics) {
                        local_types.insert(name.name.clone(), type_to_glsl(t).0);
                    }
                }
                if let Some(e) = init {
                    collect_instantiations_in_expr(
                        e,
                        generic_structs,
                        active_generics,
                        local_types,
                        out,
                    );
                    if ty.is_none()
                        && let Some(inferred) =
                            infer_simple_concrete_expr_type(e, generic_structs, local_types)
                    {
                        local_types.insert(name.name.clone(), inferred);
                    }
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(e) = value {
                    collect_instantiations_in_expr(
                        e,
                        generic_structs,
                        active_generics,
                        local_types,
                        out,
                    );
                }
            }
            Stmt::For { iter, body, .. } => {
                collect_instantiations_in_expr(
                    iter,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
                collect_instantiations_in_block(
                    body,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
            Stmt::While { cond, body, .. } => {
                collect_instantiations_in_expr(
                    cond,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
                collect_instantiations_in_block(
                    body,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
            Stmt::Loop { body, .. } => {
                collect_instantiations_in_block(
                    body,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
            Stmt::Switch {
                selector,
                cases,
                default,
                ..
            } => {
                collect_instantiations_in_expr(
                    selector,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
                for c in cases {
                    collect_instantiations_in_expr(
                        &c.value,
                        generic_structs,
                        active_generics,
                        local_types,
                        out,
                    );
                    collect_instantiations_in_block(
                        &c.body,
                        generic_structs,
                        active_generics,
                        local_types,
                        out,
                    );
                }
                if let Some(d) = default {
                    collect_instantiations_in_block(
                        d,
                        generic_structs,
                        active_generics,
                        local_types,
                        out,
                    );
                }
            }
            Stmt::Expr { expr, .. } => {
                collect_instantiations_in_expr(
                    expr,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }
    if let Some(ref tail) = block.tail_expr {
        collect_instantiations_in_expr(tail, generic_structs, active_generics, local_types, out);
    }
    *local_types = saved_locals;
}

fn infer_simple_concrete_expr_type(
    expr: &Expr,
    generic_structs: &IndexMap<SmolStr, &StructDecl>,
    local_types: &IndexMap<SmolStr, String>,
) -> Option<String> {
    match expr {
        Expr::StructLit { path, fields, .. } => {
            let base = path.joined().replace("::", "_");
            if let Some(s_decl) = generic_structs.get(base.as_str()) {
                let args =
                    infer_generic_struct_lit_args(s_decl, fields, generic_structs, local_types)?;
                return Some(mangle_generic_name(&base, &args));
            }
            Some(base)
        }
        Expr::Path { path, .. } if path.segments.len() == 1 => {
            local_types.get(&path.segments[0].name).cloned()
        }
        Expr::BoolLit(_, _) => Some("bool".to_owned()),
        Expr::FloatLit(_, _) => Some("float".to_owned()),
        Expr::IntLit(raw, _) => {
            if raw.ends_with("u64") {
                Some("uint64_t".to_owned())
            } else if raw.ends_with("i64") {
                Some("int64_t".to_owned())
            } else if raw.ends_with('u') || raw.ends_with("u32") {
                Some("uint".to_owned())
            } else {
                Some("int".to_owned())
            }
        }
        _ => None,
    }
}

fn infer_generic_struct_lit_args(
    s_decl: &StructDecl,
    fields: &[(Ident, Expr)],
    generic_structs: &IndexMap<SmolStr, &StructDecl>,
    local_types: &IndexMap<SmolStr, String>,
) -> Option<Vec<String>> {
    let mut resolved: Vec<Option<String>> = vec![None; s_decl.generics.len()];
    for f in &s_decl.fields {
        if let TypeExpr::Named { path, args, .. } = &f.ty
            && path.segments.len() == 1
            && args.is_empty()
            && let Some(idx) = s_decl
                .generics
                .iter()
                .position(|g| g.name.name == path.segments[0].name)
            && let Some((_, val_expr)) = fields.iter().find(|(id, _)| id.name == f.name.name)
            && let Some(ty) =
                infer_simple_concrete_expr_type(val_expr, generic_structs, local_types)
        {
            resolved[idx] = Some(ty);
        }
    }
    if resolved.iter().all(Option::is_some) {
        Some(resolved.into_iter().flatten().collect())
    } else {
        None
    }
}

fn collect_instantiations_in_expr(
    expr: &Expr,
    generic_structs: &IndexMap<SmolStr, &StructDecl>,
    active_generics: &IndexSet<SmolStr>,
    local_types: &mut IndexMap<SmolStr, String>,
    out: &mut IndexSet<MonomorphKey>,
) {
    match expr {
        Expr::StructLit { path, fields, .. } => {
            for (_, v) in fields {
                collect_instantiations_in_expr(
                    v,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
            let base = path.joined().replace("::", "_");
            if let Some(s_decl) = generic_structs.get(base.as_str())
                && let Some(type_args) =
                    infer_generic_struct_lit_args(s_decl, fields, generic_structs, local_types)
                && !type_args
                    .iter()
                    .any(|t| active_generics.contains(t.as_str()))
            {
                out.insert(MonomorphKey {
                    struct_name: s_decl.name.name.clone(),
                    type_args,
                });
            }
        }
        Expr::Path { turbofish, .. } => {
            for arg in turbofish {
                if let TypeArg::Type(t) = arg {
                    collect_instantiations_in_type(t, generic_structs, active_generics, out);
                }
            }
        }
        Expr::ArrayLit { elements, .. } => {
            for e in elements {
                collect_instantiations_in_expr(
                    e,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
        }
        Expr::Unary { operand, .. } => {
            collect_instantiations_in_expr(
                operand,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_instantiations_in_expr(lhs, generic_structs, active_generics, local_types, out);
            collect_instantiations_in_expr(rhs, generic_structs, active_generics, local_types, out);
        }
        Expr::Call { callee, args, .. } => {
            collect_instantiations_in_expr(
                callee,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
            for a in args {
                collect_instantiations_in_expr(
                    &a.value,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
        }
        Expr::MethodCall {
            receiver,
            turbofish,
            args,
            ..
        } => {
            collect_instantiations_in_expr(
                receiver,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
            for arg in turbofish {
                if let TypeArg::Type(t) = arg {
                    collect_instantiations_in_type(t, generic_structs, active_generics, out);
                }
            }
            for a in args {
                collect_instantiations_in_expr(
                    &a.value,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
        }
        Expr::Field { receiver, .. } => {
            collect_instantiations_in_expr(
                receiver,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
        }
        Expr::Index {
            receiver, index, ..
        } => {
            collect_instantiations_in_expr(
                receiver,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
            collect_instantiations_in_expr(
                index,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_instantiations_in_expr(
                cond,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
            collect_instantiations_in_block(
                then_branch,
                generic_structs,
                active_generics,
                local_types,
                out,
            );
            if let Some(eb) = else_branch {
                collect_instantiations_in_expr(
                    eb,
                    generic_structs,
                    active_generics,
                    local_types,
                    out,
                );
            }
        }
        Expr::Block(b) => {
            collect_instantiations_in_block(b, generic_structs, active_generics, local_types, out);
        }
        Expr::IntLit(_, _)
        | Expr::FloatLit(_, _)
        | Expr::BoolLit(_, _)
        | Expr::StringLit(_, _)
        | Expr::NoneLit(_) => {}
    }
}

// ============================================================================
// Pack Parameter Type Inference for Raster G-Buffer Hook Bridge
// ============================================================================

fn extract_pack_param_key(expr: &Expr) -> Option<SmolStr> {
    match expr {
        Expr::Call { callee, args, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref()
                && matches!(
                    path.joined().as_str(),
                    "param" | "mc::param" | "pack::param"
                )
                && let Some(first) = args.first()
            {
                return match &first.value {
                    Expr::Path { path: p, .. } => Some(SmolStr::new(p.joined().replace("::", "_"))),
                    Expr::StringLit(s, _) => Some(s.clone()),
                    _ => None,
                };
            }
            None
        }
        Expr::Field {
            receiver, field, ..
        } => {
            if let Expr::Path { path, .. } = receiver.as_ref()
                && matches!(
                    path.joined().as_str(),
                    "mc::params" | "params" | "pack::params"
                )
            {
                return Some(field.name.clone());
            }
            None
        }
        _ => None,
    }
}

fn expr_to_param_key(expr: &Expr, aliases: &IndexMap<SmolStr, SmolStr>) -> Option<SmolStr> {
    if let Some(k) = extract_pack_param_key(expr) {
        return Some(k);
    }
    if let Expr::Path { path, .. } = expr
        && path.segments.len() == 1
    {
        return aliases.get(&path.segments[0].name).cloned();
    }
    None
}

fn record_param_type(types: &mut IndexMap<SmolStr, String>, key: SmolStr, inferred: &str) {
    // Keep non-float (more specific) inferred types once discovered.
    if inferred == "float" && types.contains_key(&key) {
        return;
    }
    types.insert(key, inferred.to_owned());
}

fn infer_pack_param_types(func: &FnDecl, scan: &mut FeatureScan) {
    let mut local_types: IndexMap<SmolStr, String> = IndexMap::new();
    for p in &func.params {
        if let Some(ref ty) = p.ty {
            local_types.insert(p.name.name.clone(), type_to_glsl(ty).0);
        }
    }
    let mut aliases: IndexMap<SmolStr, SmolStr> = IndexMap::new();
    if let Some(ref body) = func.body {
        infer_param_types_block(
            body,
            &mut local_types,
            &mut aliases,
            &mut scan.pack_param_types,
        );
    }
}

fn infer_param_types_block(
    block: &Block,
    local_types: &mut IndexMap<SmolStr, String>,
    aliases: &mut IndexMap<SmolStr, SmolStr>,
    out_types: &mut IndexMap<SmolStr, String>,
) {
    let saved_locals = local_types.clone();
    let saved_aliases = aliases.clone();

    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { name, ty, init, .. } => {
                if let Some(e) = init {
                    let rhs_param_key = expr_to_param_key(e, aliases);
                    let rhs_simple_ty = literal_or_local_type(e, local_types);
                    infer_param_types_expr(e, local_types, aliases, out_types);
                    if let Some(t) = ty {
                        let glsl_ty = type_to_glsl(t).0;
                        if let Some(key) = rhs_param_key {
                            record_param_type(out_types, key, &glsl_ty);
                        }
                        aliases.swap_remove(&name.name);
                        local_types.insert(name.name.clone(), glsl_ty);
                    } else if let Some(key) = rhs_param_key {
                        local_types.swap_remove(&name.name);
                        aliases.insert(name.name.clone(), key);
                    } else {
                        aliases.swap_remove(&name.name);
                        if let Some(inferred) = rhs_simple_ty {
                            local_types.insert(name.name.clone(), inferred.to_owned());
                        } else {
                            local_types.swap_remove(&name.name);
                        }
                    }
                } else if let Some(t) = ty {
                    aliases.swap_remove(&name.name);
                    local_types.insert(name.name.clone(), type_to_glsl(t).0);
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(e) = value {
                    infer_param_types_expr(e, local_types, aliases, out_types);
                }
            }
            Stmt::For { iter, body, .. } => {
                infer_param_types_expr(iter, local_types, aliases, out_types);
                infer_param_types_block(body, local_types, aliases, out_types);
            }
            Stmt::While { cond, body, .. } => {
                if let Some(key) = expr_to_param_key(cond, aliases) {
                    record_param_type(out_types, key, "bool");
                }
                infer_param_types_expr(cond, local_types, aliases, out_types);
                infer_param_types_block(body, local_types, aliases, out_types);
            }
            Stmt::Loop { body, .. } => {
                infer_param_types_block(body, local_types, aliases, out_types);
            }
            Stmt::Switch {
                selector,
                cases,
                default,
                ..
            } => {
                if let Some(key) = expr_to_param_key(selector, aliases) {
                    record_param_type(out_types, key, "int");
                }
                infer_param_types_expr(selector, local_types, aliases, out_types);
                for c in cases {
                    infer_param_types_expr(&c.value, local_types, aliases, out_types);
                    infer_param_types_block(&c.body, local_types, aliases, out_types);
                }
                if let Some(d) = default {
                    infer_param_types_block(d, local_types, aliases, out_types);
                }
            }
            Stmt::Expr { expr, .. } => {
                infer_param_types_expr(expr, local_types, aliases, out_types);
            }
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }
    if let Some(ref tail) = block.tail_expr {
        infer_param_types_expr(tail, local_types, aliases, out_types);
    }

    *local_types = saved_locals;
    *aliases = saved_aliases;
}

fn literal_or_local_type(
    expr: &Expr,
    local_types: &IndexMap<SmolStr, String>,
) -> Option<&'static str> {
    match expr {
        Expr::BoolLit(_, _) => Some("bool"),
        Expr::IntLit(raw, _) => {
            if raw.ends_with('u') || raw.ends_with("u32") {
                Some("uint")
            } else {
                Some("int")
            }
        }
        Expr::FloatLit(_, _) => Some("float"),
        Expr::Call { callee, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref() {
                return match path.joined().as_str() {
                    "vec2" => Some("vec2"),
                    "vec3" => Some("vec3"),
                    "vec4" => Some("vec4"),
                    "ivec2" => Some("ivec2"),
                    "ivec3" => Some("ivec3"),
                    "ivec4" => Some("ivec4"),
                    _ => None,
                };
            }
            None
        }
        Expr::Path { path, .. } if path.segments.len() == 1 => {
            match local_types.get(&path.segments[0].name).map(String::as_str) {
                Some("bool") => Some("bool"),
                Some("int") => Some("int"),
                Some("uint") => Some("uint"),
                Some("vec2") => Some("vec2"),
                Some("vec3") => Some("vec3"),
                Some("vec4") => Some("vec4"),
                Some("float") => Some("float"),
                _ => None,
            }
        }
        _ => None,
    }
}

fn infer_param_types_expr(
    expr: &Expr,
    local_types: &mut IndexMap<SmolStr, String>,
    aliases: &mut IndexMap<SmolStr, SmolStr>,
    out_types: &mut IndexMap<SmolStr, String>,
) {
    match expr {
        Expr::Unary { op, operand, .. } => {
            if let Some(key) = expr_to_param_key(operand, aliases) {
                match op {
                    UnaryOp::Not => record_param_type(out_types, key, "bool"),
                    UnaryOp::BitNot => record_param_type(out_types, key, "int"),
                    _ => {}
                }
            }
            infer_param_types_expr(operand, local_types, aliases, out_types);
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            match op {
                BinaryOp::And | BinaryOp::Or => {
                    if let Some(k) = expr_to_param_key(lhs, aliases) {
                        record_param_type(out_types, k, "bool");
                    }
                    if let Some(k) = expr_to_param_key(rhs, aliases) {
                        record_param_type(out_types, k, "bool");
                    }
                }
                BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::Rem => {
                    if let Some(k) = expr_to_param_key(lhs, aliases) {
                        record_param_type(out_types, k, "int");
                    }
                    if let Some(k) = expr_to_param_key(rhs, aliases) {
                        record_param_type(out_types, k, "int");
                    }
                }
                BinaryOp::Assign
                | BinaryOp::AddAssign
                | BinaryOp::SubAssign
                | BinaryOp::MulAssign
                | BinaryOp::DivAssign
                | BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div => {
                    if let Some(k) = expr_to_param_key(lhs, aliases)
                        && let Some(ty) = literal_or_local_type(rhs, local_types)
                    {
                        record_param_type(out_types, k, ty);
                    }
                    if let Some(k) = expr_to_param_key(rhs, aliases)
                        && let Some(ty) = literal_or_local_type(lhs, local_types)
                    {
                        record_param_type(out_types, k, ty);
                    }
                }
                _ => {}
            }
            infer_param_types_expr(lhs, local_types, aliases, out_types);
            infer_param_types_expr(rhs, local_types, aliases, out_types);
        }
        Expr::Field {
            receiver, field, ..
        } => {
            if let Some(key) = expr_to_param_key(receiver, aliases) {
                let swiz = field.name.as_str();
                if !swiz.is_empty()
                    && swiz.len() <= 4
                    && swiz
                        .chars()
                        .all(|c| matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a'))
                {
                    let inferred = if swiz.contains('w') || swiz.contains('a') {
                        "vec4"
                    } else {
                        "vec3"
                    };
                    record_param_type(out_types, key, inferred);
                }
            }
            infer_param_types_expr(receiver, local_types, aliases, out_types);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            if let Some(key) = expr_to_param_key(cond, aliases) {
                record_param_type(out_types, key, "bool");
            }
            infer_param_types_expr(cond, local_types, aliases, out_types);
            infer_param_types_block(then_branch, local_types, aliases, out_types);
            if let Some(eb) = else_branch {
                infer_param_types_expr(eb, local_types, aliases, out_types);
            }
        }
        Expr::Block(b) => infer_param_types_block(b, local_types, aliases, out_types),
        Expr::Call { callee, args, .. } => {
            infer_param_types_expr(callee, local_types, aliases, out_types);
            for a in args {
                infer_param_types_expr(&a.value, local_types, aliases, out_types);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            infer_param_types_expr(receiver, local_types, aliases, out_types);
            for a in args {
                infer_param_types_expr(&a.value, local_types, aliases, out_types);
            }
        }
        Expr::Index {
            receiver, index, ..
        } => {
            infer_param_types_expr(receiver, local_types, aliases, out_types);
            infer_param_types_expr(index, local_types, aliases, out_types);
        }
        Expr::ArrayLit { elements, .. } => {
            for e in elements {
                infer_param_types_expr(e, local_types, aliases, out_types);
            }
        }
        Expr::StructLit { fields, .. } => {
            for (_, v) in fields {
                infer_param_types_expr(v, local_types, aliases, out_types);
            }
        }
        Expr::IntLit(_, _)
        | Expr::FloatLit(_, _)
        | Expr::BoolLit(_, _)
        | Expr::StringLit(_, _)
        | Expr::NoneLit(_)
        | Expr::Path { .. } => {}
    }
}
