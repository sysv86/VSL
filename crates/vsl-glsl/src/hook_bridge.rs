use crate::emit::{
    EmitOptions, emit_fn_decl_with_param_types, escape_glsl_ident, fn_return_glsl_type,
    format_param_list, scan_fn_decl,
};
use indexmap::IndexMap;
use smol_str::SmolStr;
use std::fmt::Write as _;
use vsl_mc::{DESCRIPTOR_SET_PER_PASS, HookAbiRoute, HookStage};
use vsl_syntax::ast::FnDecl;

/// Emits a Hyaloclast raster G-Buffer GLSL hook slot wrapper for a VSL 2.0 `@hook` function.
///
/// Only stages routed through [`HookAbiRoute::RasterGbufferGlslBridge`] (`vertex_displace`,
/// `surface`, `gbuffer_pack`) are permitted here. Pack parameter reads (`param(...)`,
/// `mc::param(...)`, `pack::param(...)`, `mc::params.<key>`, `params.<key>`, `pack::params.<key>`)
/// inside the hook body are routed directly to `VsMotion.packParams.<key>` so VSL 2.0's native
/// SPIR-V backend never inherits legacy raster G-Buffer ABI coupling.
///
/// Each referenced pack parameter's GLSL field type (`bool`, `int`, `uint`, `vec3`, `vec4`,
/// `float`) is inferred automatically from usage context inside `func`. To override or supply
/// explicit pack parameter types from `pack.vsl` settings metadata, use
/// [`emit_raster_gbuffer_hook_slot_with_types`].
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

    let mut scan = scan_fn_decl(func);
    scan.extensions.insert("GL_EXT_scalar_block_layout");

    // Merge caller-supplied parameter types on top of usage-inferred types.
    let mut resolved_param_types = scan.pack_param_types.clone();
    for (k, v) in param_types {
        resolved_param_types.insert(k.clone(), v.clone());
    }

    let opts = EmitOptions {
        route_params_to_vs_motion: true,
    };

    let mut out = String::with_capacity(768);
    for ext in &scan.extensions {
        let _ = writeln!(out, "#extension {ext} : require");
    }
    out.push('\n');

    let _ = writeln!(
        out,
        "// Hyaloclast Raster G-Buffer Hook Bridge: {} ({dispatch_name})",
        stage.as_str()
    );
    out.push_str("// Pack parameter reads routed to VsMotion.packParams.\n");
    out.push_str("struct VsPackParams {\n");
    if scan.pack_param_keys.is_empty() {
        out.push_str("    float _reserved;\n");
    } else {
        for key in &scan.pack_param_keys {
            let glsl_ty = resolved_param_types
                .get(key.as_str())
                .map_or("float", String::as_str);
            let field_name = escape_glsl_ident(key);
            let _ = writeln!(out, "    {glsl_ty} {field_name};");
        }
    }
    out.push_str("};\n");
    let _ = writeln!(
        out,
        "layout(set = {DESCRIPTOR_SET_PER_PASS}, binding = 0, scalar) uniform VsMotionBlock {{"
    );
    out.push_str("    VsPackParams packParams;\n");
    out.push_str("} VsMotion;\n\n");

    emit_fn_decl_with_param_types(func, opts, &resolved_param_types, &mut out);
    out.push('\n');

    let ret_ty = fn_return_glsl_type(func);
    let params = format_param_list(&func.params);
    let call_args = func
        .params
        .iter()
        .map(|p| escape_glsl_ident(&p.name.name))
        .collect::<Vec<_>>()
        .join(", ");

    let _ = writeln!(out, "{ret_ty} {dispatch_name}({params}) {{");
    if ret_ty == "void" {
        let _ = writeln!(out, "    {}({call_args});", func.name.name);
    } else {
        let _ = writeln!(out, "    return {}({call_args});", func.name.name);
    }
    out.push_str("}\n");

    Ok(out)
}
