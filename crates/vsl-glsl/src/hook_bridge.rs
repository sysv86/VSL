use crate::emit::{
    BindlessAbi, EmitCtx, EmitOptions, LayoutMode, emit_concrete_struct_decl, emit_const_decl,
    emit_fn_decl_with_ctx, escape_glsl_ident, fn_return_glsl_type, format_param_list, has_attr,
    scan_expr, scan_fn_decl, scan_fn_decl_without_param_inference, scan_fn_into, scan_type,
    seed_mc_builtin_types, type_to_glsl,
};
use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use std::fmt::Write as _;
use vsl_mc::{DESCRIPTOR_SET_PER_PASS, HookAbiRoute, HookStage, PackParamType};
use vsl_syntax::ast::{
    Block, ConstDecl, Expr, FnDecl, Item, SourceUnit, Stmt, StructDecl, TypeArg, TypeExpr,
};

/// Dedicated Set 0 binding for `VsMotionBlock`, avoiding Set 0 binding 0 (`WorldUBO`)
/// and per-pass target slots `1..=14`.
const VS_MOTION_BINDING: u32 = 15;

/// Emits a Hyaloclast raster G-Buffer GLSL hook slot wrapper for a VSL 2.0 `@hook` function.
///
/// Only stages routed through [`HookAbiRoute::RasterGbufferGlslBridge`] (`vertex_displace`,
/// `surface`, `gbuffer_pack`) are permitted here. Pack parameter reads (`param(...)`,
/// `mc::param(...)`, `pack::param(...)`, `mc::params.<key>`, `params.<key>`, `pack::params.<key>`)
/// inside the hook body are routed directly to `VsMotion.packParams.<key>` so VSL 2.0's native
/// SPIR-V backend never inherits legacy raster G-Buffer ABI coupling.
///
/// Each referenced pack parameter's GLSL field type (`bool`, `int`, `uint`, `vec3`, `vec4`,
/// `float`) is inferred automatically from usage context inside `func`. To supply schema types
/// directly from `pack.vsl` settings metadata, use [`emit_raster_gbuffer_hook_slot_with_schema`]
/// or [`emit_raster_gbuffer_hook_slot_with_types`].
pub fn emit_raster_gbuffer_hook_slot(stage: HookStage, func: &FnDecl) -> Result<String, String> {
    let empty = IndexMap::new();
    emit_raster_gbuffer_hook_slot_with_types(stage, func, &empty)
}

/// Emits a Hyaloclast raster G-Buffer GLSL hook slot wrapper with optional explicit GLSL types
/// for `VsPackParams` fields.
///
/// Entries in `param_types` override inferred field types for matching parameter keys; any
/// remaining referenced parameter keys fall back to usage-based type inference (`bool`, `int`,
/// `vec3`, `float`).
pub fn emit_raster_gbuffer_hook_slot_with_types(
    stage: HookStage,
    func: &FnDecl,
    param_types: &IndexMap<SmolStr, String>,
) -> Result<String, String> {
    emit_raster_gbuffer_hook_slot_inner(stage, None, func, param_types, true)
}

/// Emits a Hyaloclast raster G-Buffer GLSL hook slot wrapper with `VsPackParams` fields typed
/// strictly from a `pack.vsl` [`PackParamType`] schema.
///
/// Emits the complete ordered `param_types` schema into `VsPackParams` so member offsets match
/// the host/BDA `PackParamBuffer` layout regardless of which subset a particular hook reads.
pub fn emit_raster_gbuffer_hook_slot_with_schema(
    stage: HookStage,
    func: &FnDecl,
    param_types: &IndexMap<SmolStr, PackParamType>,
) -> Result<String, String> {
    let schema_types: IndexMap<SmolStr, String> = param_types
        .iter()
        .map(|(k, ty)| (k.clone(), ty.glsl_type_name().to_owned()))
        .collect();
    emit_raster_gbuffer_hook_slot_inner(stage, None, func, &schema_types, false)
}

/// Emits a Hyaloclast raster G-Buffer GLSL hook slot wrapper with a full [`SourceUnit`] context,
/// including transitively referenced constants, structs, and helper functions from the same module
/// alongside the complete ordered `VsPackParams` schema.
pub fn emit_raster_gbuffer_hook_slot_from_unit(
    stage: HookStage,
    unit: &SourceUnit,
    func: &FnDecl,
    param_types: &IndexMap<SmolStr, PackParamType>,
) -> Result<String, String> {
    let schema_types: IndexMap<SmolStr, String> = param_types
        .iter()
        .map(|(k, ty)| (k.clone(), ty.glsl_type_name().to_owned()))
        .collect();
    emit_raster_gbuffer_hook_slot_inner(stage, Some(unit), func, &schema_types, false)
}

fn emit_raster_gbuffer_hook_slot_inner(
    stage: HookStage,
    unit: Option<&SourceUnit>,
    func: &FnDecl,
    param_types: &IndexMap<SmolStr, String>,
    infer_param_types: bool,
) -> Result<String, String> {
    if stage.abi_route() != HookAbiRoute::RasterGbufferGlslBridge {
        return Err(format!(
            "hook stage `{}` uses native SPIR-V lowering and is not a raster G-Buffer GLSL bridge hook",
            stage.as_str()
        ));
    }

    let dispatch_name = match stage {
        HookStage::VertexDisplace => "vs_dispatch_hook_vertex_displace",
        HookStage::Surface => "vs_dispatch_hook_surface",
        HookStage::GbufferPack => "vs_dispatch_hook_gbuffer_pack",
        _ => {
            return Err(format!(
                "unsupported raster G-Buffer hook stage `{}`",
                stage.as_str()
            ));
        }
    };

    let mut scan = if infer_param_types {
        scan_fn_decl(func)
    } else {
        scan_fn_decl_without_param_inference(func)
    };

    let deps = unit.map(|u| collect_transitive_module_deps(u, func));
    if let Some(ref deps) = deps {
        for c in &deps.consts {
            scan_type(&c.ty, &mut scan);
            scan_expr(&c.value, &mut scan);
        }
        for s in &deps.structs {
            for f in &s.fields {
                scan_type(&f.ty, &mut scan);
            }
        }
        for helper in &deps.funcs {
            scan_fn_into(helper, &mut scan);
        }
    }

    let resolved_param_types = if infer_param_types {
        // Merge caller-supplied parameter types on top of usage-inferred types.
        let mut merged = scan.pack_param_types.clone();
        for (k, v) in param_types {
            merged.insert(k.clone(), v.clone());
        }
        merged
    } else {
        // Schema-driven path: use only caller-supplied schema types without AST inference.
        param_types.clone()
    };

    let opts = EmitOptions {
        route_params_to_vs_motion: true,
        bindless_abi: BindlessAbi::Hyaloclast4096,
        ..EmitOptions::default()
    };

    let mut out = String::with_capacity(1024);
    for ext in &scan.extensions {
        let _ = writeln!(out, "#extension {ext} : require");
    }
    if !scan.extensions.is_empty() {
        out.push('\n');
    }

    let _ = writeln!(
        out,
        "// Hyaloclast Raster G-Buffer Hook Bridge: {} ({dispatch_name})",
        stage.as_str()
    );
    out.push_str("// Pack parameter reads routed to VsMotion.packParams.\n");
    out.push_str("struct VsPackParams {\n");
    if !infer_param_types && !param_types.is_empty() {
        // Emit the complete ordered `PackSchema` so `VsPackParams` matches the host/BDA layout.
        for (key, glsl_ty) in param_types {
            let field_name = escape_glsl_ident(key);
            let _ = writeln!(out, "    {glsl_ty} {field_name};");
        }
        for key in &scan.pack_param_keys {
            if !param_types.contains_key(key.as_str()) {
                let field_name = escape_glsl_ident(key);
                let _ = writeln!(out, "    float {field_name};");
            }
        }
    } else if scan.pack_param_keys.is_empty() && param_types.is_empty() {
        out.push_str("    float _reserved;\n");
    } else {
        let mut emitted = IndexSet::new();
        for (key, glsl_ty) in param_types {
            if emitted.insert(key.as_str().to_owned()) {
                let field_name = escape_glsl_ident(key);
                let _ = writeln!(out, "    {glsl_ty} {field_name};");
            }
        }
        for key in &scan.pack_param_keys {
            if emitted.insert(key.clone()) {
                let glsl_ty = resolved_param_types
                    .get(key.as_str())
                    .map_or("float", String::as_str);
                let field_name = escape_glsl_ident(key);
                let _ = writeln!(out, "    {glsl_ty} {field_name};");
            }
        }
    }
    out.push_str("};\n");
    let _ = writeln!(
        out,
        "layout(set = {DESCRIPTOR_SET_PER_PASS}, binding = {VS_MOTION_BINDING}, std430) readonly buffer VsMotionBlock {{"
    );
    out.push_str("    VsPackParams packParams;\n");
    out.push_str("} VsMotion;\n\n");

    let mut ctx = EmitCtx::new(opts);
    ctx.pack_param_types.clone_from(&resolved_param_types);
    seed_mc_builtin_types(&mut ctx);

    if let Some(ref deps) = deps {
        for c in &deps.consts {
            ctx.local_types
                .insert(c.name.name.clone(), type_to_glsl(&c.ty).0);
        }
        for s in &deps.structs {
            let mut fields = IndexMap::new();
            for f in &s.fields {
                fields.insert(f.name.name.clone(), type_to_glsl(&f.ty).0);
            }
            ctx.struct_fields.insert(s.name.name.clone(), fields);
            ctx.struct_layouts
                .insert(s.name.name.clone(), LayoutMode::Std430);
        }
        for helper in &deps.funcs {
            ctx.fn_returns
                .insert(helper.name.name.clone(), fn_return_glsl_type(helper));
        }
        ctx.fn_returns
            .insert(func.name.name.clone(), fn_return_glsl_type(func));

        for c in &deps.consts {
            emit_const_decl(c, &ctx, &mut out);
        }
        for s in &deps.structs {
            emit_concrete_struct_decl(s, &s.name.name, &ctx, &mut out);
        }
        for helper in &deps.funcs {
            emit_fn_decl_with_ctx(helper, None, &mut ctx, &mut out);
            out.push('\n');
        }
    }

    emit_fn_decl_with_ctx(func, None, &mut ctx, &mut out);
    if !ctx.errors.is_empty() {
        return Err(ctx.errors.join("; "));
    }
    out.push('\n');

    let ret_ty = fn_return_glsl_type(func);
    let params = format_param_list(&func.params);
    let call_args = func
        .params
        .iter()
        .map(|p| escape_glsl_ident(&p.name.name))
        .collect::<Vec<_>>()
        .join(", ");
    let fn_name = escape_glsl_ident(&func.name.name);

    let _ = writeln!(out, "{ret_ty} {dispatch_name}({params}) {{");
    if ret_ty == "void" {
        let _ = writeln!(out, "    {fn_name}({call_args});");
    } else {
        let _ = writeln!(out, "    return {fn_name}({call_args});");
    }
    out.push_str("}\n");

    Ok(out)
}

struct ModuleDeps<'a> {
    consts: Vec<&'a ConstDecl>,
    structs: Vec<&'a StructDecl>,
    funcs: Vec<&'a FnDecl>,
}

fn collect_transitive_module_deps<'a>(unit: &'a SourceUnit, root: &FnDecl) -> ModuleDeps<'a> {
    let mut const_map: IndexMap<&str, &'a ConstDecl> = IndexMap::new();
    let mut struct_map: IndexMap<&str, &'a StructDecl> = IndexMap::new();
    let mut fn_map: IndexMap<&str, &'a FnDecl> = IndexMap::new();

    for item in &unit.items {
        match item {
            Item::Const(c) => {
                const_map.insert(c.name.name.as_str(), c);
            }
            Item::Struct(s) if s.generics.is_empty() && !has_attr(&s.attrs, "push_constant") => {
                struct_map.insert(s.name.name.as_str(), s);
            }
            Item::Function(f) if f.name.name != root.name.name => {
                fn_map.insert(f.name.name.as_str(), f);
            }
            _ => {}
        }
    }

    let mut worklist: Vec<SmolStr> = Vec::new();
    collect_fn_refs(root, &mut worklist);

    let mut needed_consts: IndexSet<SmolStr> = IndexSet::new();
    let mut needed_structs: IndexSet<SmolStr> = IndexSet::new();
    let mut needed_funcs: IndexSet<SmolStr> = IndexSet::new();

    while let Some(ident) = worklist.pop() {
        let key = ident.as_str();
        if let Some(c) = const_map.get(key)
            && needed_consts.insert(ident.clone())
        {
            collect_type_refs(&c.ty, &mut worklist);
            collect_expr_refs(&c.value, &mut worklist);
        }
        if let Some(s) = struct_map.get(key)
            && needed_structs.insert(ident.clone())
        {
            for field in &s.fields {
                collect_type_refs(&field.ty, &mut worklist);
            }
        }
        if let Some(f) = fn_map.get(key)
            && needed_funcs.insert(ident)
        {
            collect_fn_refs(f, &mut worklist);
        }
    }

    let mut consts = Vec::new();
    let mut structs = Vec::new();
    let mut funcs = Vec::new();
    for item in &unit.items {
        match item {
            Item::Const(c) if needed_consts.contains(&c.name.name) => consts.push(c),
            Item::Struct(s) if needed_structs.contains(&s.name.name) => structs.push(s),
            Item::Function(f) if needed_funcs.contains(&f.name.name) => funcs.push(f),
            _ => {}
        }
    }

    ModuleDeps {
        consts,
        structs,
        funcs,
    }
}

fn collect_fn_refs(func: &FnDecl, out: &mut Vec<SmolStr>) {
    for param in &func.params {
        if let Some(ref ty) = param.ty {
            collect_type_refs(ty, out);
        }
    }
    if let Some(ref ret) = func.return_ty {
        collect_type_refs(ret, out);
    }
    if let Some(ref body) = func.body {
        collect_block_refs(body, out);
    }
}

fn collect_type_refs(ty: &TypeExpr, out: &mut Vec<SmolStr>) {
    match ty {
        TypeExpr::Array { element, len, .. } => {
            collect_type_refs(element, out);
            collect_expr_refs(len, out);
        }
        TypeExpr::Named { path, args, .. } => {
            if path.segments.len() == 1 {
                out.push(path.segments[0].name.clone());
            }
            for arg in args {
                match arg {
                    TypeArg::Type(t) => collect_type_refs(t, out),
                    TypeArg::NamedExpr { value, .. } => collect_expr_refs(value, out),
                }
            }
        }
    }
}

fn collect_block_refs(block: &Block, out: &mut Vec<SmolStr>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { ty, init, .. } => {
                if let Some(t) = ty {
                    collect_type_refs(t, out);
                }
                if let Some(e) = init {
                    collect_expr_refs(e, out);
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(e) = value {
                    collect_expr_refs(e, out);
                }
            }
            Stmt::For { iter, body, .. } => {
                collect_expr_refs(iter, out);
                collect_block_refs(body, out);
            }
            Stmt::While { cond, body, .. } => {
                collect_expr_refs(cond, out);
                collect_block_refs(body, out);
            }
            Stmt::Loop { body, .. } => collect_block_refs(body, out),
            Stmt::Switch {
                selector,
                cases,
                default,
                ..
            } => {
                collect_expr_refs(selector, out);
                for case in cases {
                    collect_expr_refs(&case.value, out);
                    collect_block_refs(&case.body, out);
                }
                if let Some(def) = default {
                    collect_block_refs(def, out);
                }
            }
            Stmt::Expr { expr, .. } => collect_expr_refs(expr, out),
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }
    if let Some(ref tail) = block.tail_expr {
        collect_expr_refs(tail, out);
    }
}

fn collect_expr_refs(expr: &Expr, out: &mut Vec<SmolStr>) {
    match expr {
        Expr::Path { path, .. } => {
            if path.segments.len() == 1 {
                out.push(path.segments[0].name.clone());
            }
        }
        Expr::StructLit { path, fields, .. } => {
            if path.segments.len() == 1 {
                out.push(path.segments[0].name.clone());
            }
            for (_, v) in fields {
                collect_expr_refs(v, out);
            }
        }
        Expr::ArrayLit { elements, .. } => {
            for e in elements {
                collect_expr_refs(e, out);
            }
        }
        Expr::Unary { operand, .. } => collect_expr_refs(operand, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_expr_refs(lhs, out);
            collect_expr_refs(rhs, out);
        }
        Expr::Call { callee, args, .. } => {
            collect_expr_refs(callee, out);
            for a in args {
                collect_expr_refs(&a.value, out);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            collect_expr_refs(receiver, out);
            for a in args {
                collect_expr_refs(&a.value, out);
            }
        }
        Expr::Field { receiver, .. } => collect_expr_refs(receiver, out),
        Expr::Index {
            receiver, index, ..
        } => {
            collect_expr_refs(receiver, out);
            collect_expr_refs(index, out);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_expr_refs(cond, out);
            collect_block_refs(then_branch, out);
            if let Some(eb) = else_branch {
                collect_expr_refs(eb, out);
            }
        }
        Expr::Block(b) => collect_block_refs(b, out),
        Expr::IntLit(_, _)
        | Expr::FloatLit(_, _)
        | Expr::BoolLit(_, _)
        | Expr::StringLit(_, _)
        | Expr::NoneLit(_) => {}
    }
}
