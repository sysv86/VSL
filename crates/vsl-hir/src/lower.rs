//! AST-to-HIR lowering, early tier/capability pruning (Spec §3.4), compile-time SPIR-V
//! legality validation, and Hyaloclast 4-set Vulkan ABI reflection computation.

use crate::hir::{
    EntryReflection, HirFunction, HirModule, HirParam, HirStruct, HirStructField, HirStructId,
    HirWorkgroupVar,
};
use crate::ty::{LayoutKind, ScalarKind, Type, TypeArena, TypeId};
use indexmap::{IndexMap, IndexSet};
use la_arena::Arena;
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, DiagnosticBag, Span};
use vsl_mc::{
    BINDLESS_IMAGE_COUNT, DESCRIPTOR_SET_BINDLESS, DESCRIPTOR_SET_PACK_RESOURCES,
    DESCRIPTOR_SET_PER_PASS, DESCRIPTOR_SET_SCENE_GLOBALS, DebugChannelBinding, DescriptorAccess,
    DescriptorBinding, DescriptorKind, HardwareCaps, HookStage, ImageFormat,
    MAX_PUSH_CONSTANT_BYTES, PACK_RESOURCE_SLOTS, PackParamBinding, PushConstantField, RenderTier,
    SampleScalar, ShaderStage, SpecConstantUsage, TextureDim,
};
use vsl_syntax::ast::{
    Attribute, BinaryOp, Block, Expr, FnDecl, GenericParam, ImportTree, Item, ParamMode,
    SourceUnit, Stmt, StructDecl, TypeArg, TypeExpr, UnaryOp,
};

/// Maximum Set 0 binding index reserved for per-pass resources (`bindings 1..=14`).
const MAX_PER_PASS_BINDING: u32 = 14;

/// Maximum valid rendering tier index (`Tier3PathTracing = 3`).
const MAX_TIER: u8 = 3;

const FLAG_WORLD: u8 = 1 << 0;
const FLAG_TLAS: u8 = 1 << 1;
const FLAG_INSTANCES: u8 = 1 << 2;
const FLAG_LIGHTS: u8 = 1 << 3;
const FLAG_BINDLESS: u8 = 1 << 4;
const FLAG_BDA: u8 = 1 << 5;
const FLAG_SUBGROUPS: u8 = 1 << 6;

/// Pack-wide parameter layout, specialization-constant IDs, and Set 3 resource slot table
/// supplied by `pack.vsl`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PackSchema {
    /// Maps pack parameter name -> `(byte_offset, byte_size)` in `PackParamBuffer`.
    pub params: IndexMap<SmolStr, (u32, u32)>,
    /// Maps specialization constant name -> pack-global Vulkan `SpecId`.
    pub spec_ids: IndexMap<SmolStr, u32>,
    /// Maps Set 3 pack resource texture name -> pack-global Set 3 binding slot (`0..=15`).
    pub resource_slots: IndexMap<SmolStr, u32>,
}

/// Lowers a parsed [`SourceUnit`] into an arena-backed [`HirModule`] for a target
/// `(active_tier, active_caps)` configuration using deterministic module-wide fallback
/// parameter/specialization IDs.
pub fn lower_unit(
    unit: &SourceUnit,
    active_tier: RenderTier,
    active_caps: HardwareCaps,
) -> (HirModule, DiagnosticBag) {
    lower_unit_with_schema(unit, active_tier, active_caps, None)
}

/// Lowers a parsed [`SourceUnit`] into an arena-backed [`HirModule`] for a target
/// `(active_tier, active_caps)` configuration and optional pack-wide [`PackSchema`].
///
/// Performs:
/// - Two-pass struct lowering with forward-reference support and exact `scalar`/`std430`/`std140`
///   byte layout computation.
/// - Early tier pruning (Spec §3.4): functions whose `#[tier(min..=max)]` excludes
///   `active_tier` are stripped before capability validation and reflection emission.
/// - Complete compile-time SPIR-V legality checks (`RayQuery`, `f16`, 64-bit/float atomics,
///   local logical pointers, `bool` in interface blocks, push-constant limits).
/// - Hyaloclast 4-set Vulkan ABI descriptor reflection (`Set 0` per-pass, `Set 1` scene
///   globals, `Set 2` bindless, `Set 3` pack resources) plus push-constant, debug-channel,
///   BDA, and subgroup scans.
pub fn lower_unit_with_schema(
    unit: &SourceUnit,
    active_tier: RenderTier,
    active_caps: HardwareCaps,
    schema: Option<&PackSchema>,
) -> (HirModule, DiagnosticBag) {
    lower_unit_internal(unit, Some(active_tier), active_caps, schema)
}

/// Internal entry point used by [`crate::validate::validate_unit`] to run the exact same
/// lowering and legality checks across all tiers without pruning functions or rejecting
/// valid device-gated capabilities.
pub(crate) fn lower_unit_for_validation(unit: &SourceUnit) -> (HirModule, DiagnosticBag) {
    lower_unit_internal(unit, None, HardwareCaps::all(), None)
}

struct PendingStruct<'a> {
    decl: &'a StructDecl,
    generic_params: Vec<SmolStr>,
    type_id: TypeId,
    layout: LayoutKind,
    is_push_constant: bool,
    has_explicit_block_layout: bool,
    required_caps: HardwareCaps,
    hir_fields: Vec<HirStructField>,
    type_fields: Vec<(SmolStr, TypeId)>,
    all_fields_resolved: bool,
    direct_rq_fields: IndexSet<SmolStr>,
    direct_f16_fields: IndexSet<SmolStr>,
}

fn lower_unit_internal(
    unit: &SourceUnit,
    active_tier: Option<RenderTier>,
    active_caps: HardwareCaps,
    schema: Option<&PackSchema>,
) -> (HirModule, DiagnosticBag) {
    let module_name = unit.module_decl.as_ref().map_or_else(
        || SmolStr::new_static("unnamed"),
        |decl| SmolStr::new(decl.path.joined()),
    );
    let mut module = HirModule::new(module_name);
    let mut diags = DiagnosticBag::new();

    let mc_imports = collect_mc_imports(unit);

    // Pass 0: Collect declared traits so generic bounds and `impl Trait for Type` can be validated.
    let mut trait_names: IndexSet<SmolStr> = IndexSet::new();
    for item in &unit.items {
        if let Item::Trait(decl) = item
            && !trait_names.insert(decl.name.name.clone())
        {
            diags.push(Diagnostic::error(
                "E0270",
                format!("duplicate trait declaration `{}`", decl.name.name),
                decl.span,
            ));
        }
    }

    // Pass 1 (Structs): Pre-register all struct names and attributes for forward references.
    let mut struct_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
    let mut pending_structs: Vec<PendingStruct<'_>> = Vec::new();

    for item in &unit.items {
        if let Item::Struct(decl) = item {
            if struct_types.contains_key(&decl.name.name) {
                diags.push(Diagnostic::error(
                    "E0202",
                    format!("duplicate struct declaration `{}`", decl.name.name),
                    decl.span,
                ));
                continue;
            }
            validate_generic_bounds(&decl.generics, &trait_names, &mut diags);

            let mut layout = LayoutKind::Std430;
            let mut is_push_constant = false;
            let mut has_explicit_block_layout = false;
            let mut required_caps = HardwareCaps::empty();

            for attr in &decl.attrs {
                for entry in &attr.entries {
                    match entry.name.name.as_str() {
                        "push_constant" => is_push_constant = true,
                        "layout" => {
                            if let Some(Expr::Path { path, .. }) = entry.args.first()
                                && let Some(parsed) = LayoutKind::from_name(&path.joined())
                            {
                                layout = parsed;
                                if matches!(parsed, LayoutKind::Std140 | LayoutKind::Std430) {
                                    has_explicit_block_layout = true;
                                }
                            } else {
                                diags.push(Diagnostic::error(
                                    "E0201",
                                    "invalid `#[layout(...)]` argument; expected `scalar`, `std430`, or `std140`",
                                    entry.span,
                                ));
                            }
                        }
                        "cap" => parse_cap_args(&entry.args, &mut required_caps, &mut diags),
                        _ => {}
                    }
                }
            }

            let missing_caps = required_caps.difference(active_caps);
            if !missing_caps.is_empty() {
                diags.push(Diagnostic::error(
                    "E0236",
                    format!(
                        "struct `{}` requires hardware capabilities `{missing_caps:?}` not enabled in active device profile",
                        decl.name.name
                    ),
                    decl.span,
                ));
            }

            let type_id = module.types.alloc(Type::Struct {
                name: decl.name.name.clone(),
                layout,
                fields: Vec::new(),
            });
            struct_types.insert(decl.name.name.clone(), type_id);
            let generic_params: Vec<SmolStr> =
                decl.generics.iter().map(|g| g.name.name.clone()).collect();
            pending_structs.push(PendingStruct {
                decl,
                generic_params,
                type_id,
                layout,
                is_push_constant,
                has_explicit_block_layout,
                required_caps,
                hir_fields: Vec::with_capacity(decl.fields.len()),
                type_fields: Vec::with_capacity(decl.fields.len()),
                all_fields_resolved: true,
                direct_rq_fields: IndexSet::new(),
                direct_f16_fields: IndexSet::new(),
            });
        }
    }

    // Pass 2 (Structs): Resolve all field types using the complete struct table.
    for pending in &mut pending_structs {
        let generic_names: IndexSet<SmolStr> = pending.generic_params.iter().cloned().collect();
        let mut seen_fields: IndexSet<SmolStr> = IndexSet::new();

        for field in &pending.decl.fields {
            if !seen_fields.insert(field.name.name.clone()) {
                diags.push(Diagnostic::error(
                    "E0203",
                    format!(
                        "duplicate field `{}` in struct `{}`",
                        field.name.name, pending.decl.name.name
                    ),
                    field.span,
                ));
                continue;
            }
            check_bda_pointer_alignment(&field.ty, &mut diags);
            if type_expr_contains_ident(&field.ty, "RayQuery") {
                diags.push(
                    Diagnostic::error(
                        "E0210",
                        format!(
                            "`RayQuery` is forbidden inside struct `{}` field `{}`",
                            pending.decl.name.name, field.name.name
                        ),
                        field.span,
                    )
                    .with_note(
                        "SPV_KHR_ray_query forbids RayQueryKHR in aggregates or OpPhi merges; declare it as a local variable",
                    ),
                );
                pending.direct_rq_fields.insert(field.name.name.clone());
                pending.all_fields_resolved = false;
                continue;
            }

            if type_expr_uses_f16(&field.ty)
                && !pending.required_caps.contains(HardwareCaps::FLOAT16)
            {
                diags.push(
                    Diagnostic::error(
                        "E0232",
                        format!(
                            "`f16` field `{}` in struct `{}` requires `#[cap(float16)]`",
                            field.name.name, pending.decl.name.name
                        ),
                        field.span,
                    )
                    .with_note(
                        "Hyaloclast does not enable shader_float16 unconditionally in device_merger.rs",
                    ),
                );
                pending.direct_f16_fields.insert(field.name.name.clone());
            }

            let mut type_ctx = TypeLowerCtx::types_only(&mut module.types, &mut struct_types);
            if let Some(ty_id) = lower_type_expr(&field.ty, &mut type_ctx, &generic_names) {
                pending.hir_fields.push(HirStructField {
                    name: field.name.name.clone(),
                    ty: ty_id,
                    span: field.span,
                });
                pending.type_fields.push((field.name.name.clone(), ty_id));
            } else {
                pending.all_fields_resolved = false;
                if pending.is_push_constant {
                    diags.push(Diagnostic::error(
                        "E0221",
                        format!(
                            "unresolved field type on push-constant struct `{}` field `{}`",
                            pending.decl.name.name, field.name.name
                        ),
                        field.span,
                    ));
                }
            }
        }

        module
            .types
            .set_struct_fields(pending.type_id, pending.type_fields.clone());
    }

    // Pass 2.5 (Structs): Register all declared structs in `module.structs` so generic
    // struct fields can be monomorphized before layout and legality validation.
    let mut struct_ids: IndexMap<SmolStr, HirStructId> = IndexMap::new();
    let mut pending_sids: Vec<HirStructId> = Vec::with_capacity(pending_structs.len());
    for pending in &pending_structs {
        let name = pending.decl.name.name.clone();
        let sid = module.structs.alloc(HirStruct {
            name: name.clone(),
            generic_params: pending.generic_params.clone(),
            layout: pending.layout,
            is_push_constant: pending.is_push_constant,
            required_caps: pending.required_caps,
            fields: pending.hir_fields.clone(),
            computed_layout: None,
            span: pending.decl.span,
        });
        struct_ids.insert(name, sid);
        pending_sids.push(sid);
    }

    // Pass 2.6 (Structs): Re-lower fields that instantiate generic structs now that all
    // generic struct templates have their fields populated in `module.structs`.
    for (pending, &sid) in pending_structs.iter_mut().zip(&pending_sids) {
        let generic_names: IndexSet<SmolStr> = pending.generic_params.iter().cloned().collect();
        let mut updated = false;
        for field in &pending.decl.fields {
            if !type_expr_has_type_args(&field.ty)
                || pending.direct_rq_fields.contains(&field.name.name)
            {
                continue;
            }
            let mut type_ctx = TypeLowerCtx::full(
                &mut module.types,
                &mut module.structs,
                &mut struct_types,
                &mut struct_ids,
            );
            if let Some(ty_id) = lower_type_expr(&field.ty, &mut type_ctx, &generic_names) {
                if let Some(hir_field) = pending
                    .hir_fields
                    .iter_mut()
                    .find(|f| f.name == field.name.name)
                {
                    hir_field.ty = ty_id;
                }
                if let Some(type_field) = pending
                    .type_fields
                    .iter_mut()
                    .find(|(name, _)| *name == field.name.name)
                {
                    type_field.1 = ty_id;
                }
                updated = true;
            }
        }
        if updated {
            module
                .types
                .set_struct_fields(pending.type_id, pending.type_fields.clone());
            module.structs[sid].fields.clone_from(&pending.hir_fields);
        }
    }

    // Pass 3 (Structs): Compute layouts and run transitive legality checks across all structs.
    let mut module_push_constants: Vec<HirStructId> = Vec::new();
    let mut bool_checked_structs: IndexSet<SmolStr> = IndexSet::new();

    for (pending, sid) in pending_structs.into_iter().zip(pending_sids) {
        let is_interface_block = pending.is_push_constant || pending.has_explicit_block_layout;
        if is_interface_block {
            bool_checked_structs.insert(pending.decl.name.name.clone());
        }

        for field in &pending.hir_fields {
            if !pending.direct_rq_fields.contains(&field.name)
                && module.types.contains_ray_query(field.ty)
            {
                diags.push(
                    Diagnostic::error(
                        "E0210",
                        format!(
                            "`RayQuery` is forbidden inside struct `{}` field `{}`",
                            pending.decl.name.name, field.name
                        ),
                        field.span,
                    )
                    .with_note(
                        "SPV_KHR_ray_query forbids RayQueryKHR in aggregates or OpPhi merges; declare it as a local variable",
                    ),
                );
            }

            if !pending.direct_f16_fields.contains(&field.name)
                && !pending.required_caps.contains(HardwareCaps::FLOAT16)
                && module.types.requires_float16(field.ty)
            {
                diags.push(
                    Diagnostic::error(
                        "E0232",
                        format!(
                            "`f16` field `{}` in struct `{}` requires `#[cap(float16)]`",
                            field.name, pending.decl.name.name
                        ),
                        field.span,
                    )
                    .with_note(
                        "Hyaloclast does not enable shader_float16 unconditionally in device_merger.rs",
                    ),
                );
            }

            if is_interface_block && module.types.contains_bool(field.ty) {
                diags.push(
                    Diagnostic::error(
                        "E0223",
                        format!(
                            "`bool` is forbidden inside interface (`Block`) struct `{}` field `{}`",
                            pending.decl.name.name, field.name
                        ),
                        field.span,
                    )
                    .with_note(
                        "Vulkan SPIR-V forbids OpTypeBool inside Block-decorated push-constant, UBO, and SSBO structs; use `u32` instead",
                    ),
                );
            }
        }

        let has_cycle = module.types.has_layout_cycle(pending.type_id);
        if has_cycle {
            diags.push(Diagnostic::error(
                "E0204",
                format!(
                    "recursive struct `{}` has infinite size",
                    pending.decl.name.name
                ),
                pending.decl.span,
            ));
        }

        let computed_layout = if pending.all_fields_resolved && !has_cycle {
            module
                .types
                .compute_struct_layout(&pending.type_fields, pending.layout)
        } else {
            None
        };

        if pending.is_push_constant
            && let Some(ref computed) = computed_layout
            && computed.size > MAX_PUSH_CONSTANT_BYTES
        {
            diags.push(
                Diagnostic::error(
                    "E0220",
                    format!(
                        "push constant struct `{}` is {} bytes, exceeding the {MAX_PUSH_CONSTANT_BYTES} byte limit",
                        pending.decl.name.name, computed.size
                    ),
                    pending.decl.span,
                )
                .with_primary_label(
                    pending.decl.span,
                    "exceeds 128-byte Vulkan push-constant budget",
                ),
            );
        }

        module.structs[sid].computed_layout = computed_layout;
        if pending.is_push_constant {
            module_push_constants.push(sid);
        }
    }

    // Diagnose multiple `#[push_constant]` structs in the same module instead of silently taking `.first()`.
    if module_push_constants.len() > 1 {
        for &extra_sid in &module_push_constants[1..] {
            let st = &module.structs[extra_sid];
            diags.push(
                Diagnostic::error(
                    "E0222",
                    format!(
                        "multiple `#[push_constant]` structs in module (found duplicate `{}`)",
                        st.name
                    ),
                    st.span,
                )
                .with_note("Vulkan allows at most one push-constant block per entry point"),
            );
        }
    }

    // Pass 4: Lower module constants and workgroup variables before lowering functions.
    let empty_generics: IndexSet<SmolStr> = IndexSet::new();
    let mut const_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
    let mut workgroup_var_types: IndexMap<SmolStr, TypeId> = IndexMap::new();

    for item in &unit.items {
        match item {
            Item::Const(decl) => {
                let mut type_ctx = TypeLowerCtx::full(
                    &mut module.types,
                    &mut module.structs,
                    &mut struct_types,
                    &mut struct_ids,
                );
                if let Some(ty_id) = lower_type_expr(&decl.ty, &mut type_ctx, &empty_generics) {
                    const_types.insert(decl.name.name.clone(), ty_id);
                }
            }
            Item::WorkgroupVar(var) => {
                let mut var_caps = HardwareCaps::empty();
                for attr in &var.attrs {
                    for entry in &attr.entries {
                        if entry.name.name == "cap" {
                            parse_cap_args(&entry.args, &mut var_caps, &mut diags);
                        }
                    }
                }
                let missing_caps = var_caps.difference(active_caps);
                if !missing_caps.is_empty() {
                    diags.push(Diagnostic::error(
                        "E0236",
                        format!(
                            "`workgroup var {}` requires hardware capabilities `{missing_caps:?}` not enabled in active device profile",
                            var.name.name
                        ),
                        var.span,
                    ));
                }

                let ty = {
                    let mut type_ctx = TypeLowerCtx::full(
                        &mut module.types,
                        &mut module.structs,
                        &mut struct_types,
                        &mut struct_ids,
                    );
                    lower_type_expr(&var.ty, &mut type_ctx, &empty_generics)
                }
                .unwrap_or_else(|| module.types.alloc(Type::Void));

                if type_expr_contains_ident(&var.ty, "RayQuery")
                    || module.types.contains_ray_query(ty)
                {
                    diags.push(Diagnostic::error(
                        "E0212",
                        format!(
                            "`RayQuery` is forbidden in `workgroup var {}`",
                            var.name.name
                        ),
                        var.span,
                    ));
                }

                if (type_expr_uses_f16(&var.ty) || module.types.requires_float16(ty))
                    && !var_caps.contains(HardwareCaps::FLOAT16)
                {
                    diags.push(
                        Diagnostic::error(
                            "E0232",
                            format!(
                                "`f16` in `workgroup var {}` requires `#[cap(float16)]`",
                                var.name.name
                            ),
                            var.span,
                        )
                        .with_note(
                            "Hyaloclast does not enable shader_float16 unconditionally in device_merger.rs",
                        ),
                    );
                }

                workgroup_var_types.insert(var.name.name.clone(), ty);
                module.workgroup_vars.push(HirWorkgroupVar {
                    name: var.name.name.clone(),
                    ty,
                    span: var.span,
                });
            }
            _ => {}
        }
    }

    // Initialize module-wide fallback spec/param IDs past any IDs/offsets reserved by `PackSchema`.
    let next_fallback_spec_id = schema
        .and_then(|s| s.spec_ids.values().copied().max().map(|m| m + 1))
        .unwrap_or(0);
    let next_fallback_param_offset = schema
        .and_then(|s| {
            s.params
                .values()
                .map(|&(off, sz)| off.saturating_add(sz))
                .max()
        })
        .unwrap_or(0);

    // Pass 4.5: Pre-collect declared return types for all module functions and `impl` methods
    // so call-site type inference (such as `atomic_add(get_u64_ptr(), 1u64)`) resolves regardless
    // of declaration order.
    let mut fn_return_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
    for item in &unit.items {
        match item {
            Item::Function(func) => {
                if let Some(ref ret_ty) = func.return_ty {
                    let fn_generics: IndexSet<SmolStr> =
                        func.generics.iter().map(|g| g.name.name.clone()).collect();
                    let mut type_ctx = TypeLowerCtx::full(
                        &mut module.types,
                        &mut module.structs,
                        &mut struct_types,
                        &mut struct_ids,
                    );
                    if let Some(ty_id) = lower_type_expr(ret_ty, &mut type_ctx, &fn_generics) {
                        fn_return_types.insert(func.name.name.clone(), ty_id);
                    }
                }
            }
            Item::Impl(imp) => {
                let target_name = match &imp.target {
                    TypeExpr::Named { path, .. } => Some(path.joined()),
                    TypeExpr::Array { .. } => None,
                };
                let impl_self_ty = target_name
                    .as_deref()
                    .and_then(|name| struct_types.get(name).copied())
                    .or_else(|| {
                        let mut type_ctx = TypeLowerCtx::full(
                            &mut module.types,
                            &mut module.structs,
                            &mut struct_types,
                            &mut struct_ids,
                        );
                        lower_type_expr(&imp.target, &mut type_ctx, &empty_generics)
                    });
                for method in &imp.methods {
                    let Some(ref ret_ty) = method.return_ty else {
                        continue;
                    };
                    let mut method_generics: IndexSet<SmolStr> =
                        imp.generics.iter().map(|g| g.name.name.clone()).collect();
                    method_generics.extend(method.generics.iter().map(|g| g.name.name.clone()));

                    let resolved_ret = if is_self_type_expr(ret_ty) {
                        impl_self_ty
                    } else {
                        let mut type_ctx = TypeLowerCtx::full(
                            &mut module.types,
                            &mut module.structs,
                            &mut struct_types,
                            &mut struct_ids,
                        );
                        lower_type_expr(ret_ty, &mut type_ctx, &method_generics)
                    };
                    if let Some(ty_id) = resolved_ret
                        && let Some(ref target) = target_name
                    {
                        let qualified = SmolStr::from(format!("{target}::{}", method.name.name));
                        fn_return_types.insert(qualified, ty_id);
                    }
                }
            }
            _ => {}
        }
    }

    // Pass 5: Validate trait methods, impl blocks, and lower functions.
    let mut ctx = LowerCtx {
        module: &mut module,
        diags: &mut diags,
        struct_types: &mut struct_types,
        struct_ids: &mut struct_ids,
        trait_names: &trait_names,
        const_types: &const_types,
        workgroup_var_types: &workgroup_var_types,
        fn_return_types: &fn_return_types,
        module_push_constants: &module_push_constants,
        mc_imports: &mc_imports,
        active_tier,
        active_caps,
        schema,
        fallback_spec_ids: IndexMap::new(),
        next_fallback_spec_id,
        fallback_param_layouts: IndexMap::new(),
        next_fallback_param_offset,
        bool_checked_structs: &mut bool_checked_structs,
    };

    for item in &unit.items {
        match item {
            Item::Trait(decl) => {
                for method in &decl.methods {
                    ctx.validate_trait_method(method);
                }
            }
            Item::Function(func) => ctx.lower_function(func, None, &empty_generics),
            Item::Impl(imp) => {
                validate_generic_bounds(&imp.generics, ctx.trait_names, ctx.diags);
                if let Some(ref trait_path) = imp.trait_path {
                    let trait_name = trait_path.joined();
                    if !ctx.trait_names.contains(trait_name.as_str()) {
                        ctx.diags.push(Diagnostic::error(
                            "E0272",
                            format!("unknown trait `{trait_name}` in `impl` block"),
                            trait_path.span,
                        ));
                    }
                }

                let impl_generics: IndexSet<SmolStr> =
                    imp.generics.iter().map(|g| g.name.name.clone()).collect();
                let self_ty = ctx.validate_impl_target(&imp.target, imp.span);
                for method in &imp.methods {
                    ctx.lower_function(method, self_ty, &impl_generics);
                }
            }
            _ => {}
        }
    }

    (module, diags)
}

#[derive(Default)]
struct ParsedFnAttrs {
    stage: Option<ShaderStage>,
    workgroup_size: Option<[u32; 3]>,
    hook: Option<HookStage>,
    required_caps: HardwareCaps,
}

struct LowerCtx<'a> {
    module: &'a mut HirModule,
    diags: &'a mut DiagnosticBag,
    struct_types: &'a mut IndexMap<SmolStr, TypeId>,
    struct_ids: &'a mut IndexMap<SmolStr, HirStructId>,
    trait_names: &'a IndexSet<SmolStr>,
    const_types: &'a IndexMap<SmolStr, TypeId>,
    workgroup_var_types: &'a IndexMap<SmolStr, TypeId>,
    fn_return_types: &'a IndexMap<SmolStr, TypeId>,
    module_push_constants: &'a [HirStructId],
    mc_imports: &'a IndexSet<SmolStr>,
    active_tier: Option<RenderTier>,
    active_caps: HardwareCaps,
    schema: Option<&'a PackSchema>,
    fallback_spec_ids: IndexMap<SmolStr, u32>,
    next_fallback_spec_id: u32,
    fallback_param_layouts: IndexMap<SmolStr, (u32, u32)>,
    next_fallback_param_offset: u32,
    bool_checked_structs: &'a mut IndexSet<SmolStr>,
}

impl LowerCtx<'_> {
    fn type_ctx(&mut self) -> TypeLowerCtx<'_> {
        TypeLowerCtx::full(
            &mut self.module.types,
            &mut self.module.structs,
            self.struct_types,
            self.struct_ids,
        )
    }

    fn validate_impl_target(&mut self, target: &TypeExpr, span: Span) -> Option<TypeId> {
        let empty_generics = IndexSet::new();
        match target {
            TypeExpr::Named { path, .. } => {
                let name = path.joined();
                if let Some(&id) = self.struct_types.get(name.as_str()) {
                    return Some(id);
                }
                let mut type_ctx = self.type_ctx();
                if let Some(id) = lower_type_expr(target, &mut type_ctx, &empty_generics) {
                    return Some(id);
                }
                self.diags.push(Diagnostic::error(
                    "E0273",
                    format!("unknown target type `{name}` in `impl` block"),
                    span,
                ));
                None
            }
            TypeExpr::Array { .. } => {
                let mut type_ctx = self.type_ctx();
                lower_type_expr(target, &mut type_ctx, &empty_generics)
            }
        }
    }

    fn validate_trait_method(&mut self, method: &FnDecl) {
        validate_generic_bounds(&method.generics, self.trait_names, self.diags);
        let mut parsed_attrs = ParsedFnAttrs::default();
        for attr in &method.attrs {
            parse_fn_attribute(attr, method, &mut parsed_attrs, self.diags);
        }
        for param in &method.params {
            if let Some(ref ty_expr) = param.ty {
                check_type_caps_and_legality(
                    ty_expr,
                    parsed_attrs.required_caps,
                    param.span,
                    self.diags,
                );
                if type_expr_contains_ident(ty_expr, "RayQuery") {
                    self.diags.push(Diagnostic::error(
                        "E0213",
                        "`RayQuery` is forbidden as a function parameter",
                        param.span,
                    ));
                }
            }
        }
        if let Some(ref ret_ty) = method.return_ty {
            check_type_caps_and_legality(
                ret_ty,
                parsed_attrs.required_caps,
                method.span,
                self.diags,
            );
            if type_expr_contains_ident(ret_ty, "RayQuery") {
                self.diags.push(Diagnostic::error(
                    "E0214",
                    "`RayQuery` is forbidden as a function return type",
                    method.span,
                ));
            }
        }
    }

    fn lower_function(
        &mut self,
        func: &FnDecl,
        self_ty: Option<TypeId>,
        impl_generics: &IndexSet<SmolStr>,
    ) {
        let tier_range = parse_tier_range(&func.attrs, self.diags);

        // Spec §3.4: Early Tier Pruning strips excluded functions before cap/reflection emission.
        if let Some(active_tier) = self.active_tier {
            let active = active_tier.as_u8();
            if !(tier_range.0..=tier_range.1).contains(&active) {
                self.module.pruned_functions += 1;
                return;
            }
        }

        validate_generic_bounds(&func.generics, self.trait_names, self.diags);

        let mut fn_generics = impl_generics.clone();
        fn_generics.extend(func.generics.iter().map(|g| g.name.name.clone()));

        let mut parsed_attrs = ParsedFnAttrs::default();
        for attr in &func.attrs {
            parse_fn_attribute(attr, func, &mut parsed_attrs, self.diags);
        }

        // Check required caps against active device caps.
        let missing_caps = parsed_attrs.required_caps.difference(self.active_caps);
        if !missing_caps.is_empty() {
            self.diags.push(Diagnostic::error(
                "E0236",
                format!(
                    "function `{}` requires hardware capabilities `{missing_caps:?}` not enabled in active device profile",
                    func.name.name
                ),
                func.span,
            ));
        }

        // Lower parameters and check per-parameter capability and SPIR-V legality.
        let mut params = Vec::with_capacity(func.params.len());
        let mut param_push_constant_struct: Option<HirStructId> = None;
        let mut uses_bda = false;
        let mut root_scope: IndexMap<SmolStr, Option<TypeId>> = IndexMap::new();

        for param in &func.params {
            let mut builtin: Option<SmolStr> = None;
            let mut param_is_pc = false;
            let mut pack_resource_slot: Option<u32> = None;

            for attr in &param.attrs {
                for entry in &attr.entries {
                    match entry.name.name.as_str() {
                        "builtin" => {
                            if let Some(Expr::Path { path, .. }) = entry.args.first() {
                                builtin = Some(SmolStr::new(path.joined()));
                            }
                        }
                        "push_constant" => param_is_pc = true,
                        "pack_resource" => {
                            if let Some(slot) = parse_u32_lit(entry.args.first())
                                && slot < PACK_RESOURCE_SLOTS
                            {
                                pack_resource_slot = Some(slot);
                            } else {
                                self.diags.push(Diagnostic::error(
                                    "E0262",
                                    format!(
                                        "`#[pack_resource(slot)]` requires an integer slot within `0..={}`",
                                        PACK_RESOURCE_SLOTS - 1
                                    ),
                                    entry.span,
                                ));
                            }
                        }
                        _ => {}
                    }
                }
            }

            let resolved_ty = if param.ty.is_none() && param.name.name == "self" {
                self_ty
            } else {
                param.ty.as_ref().and_then(|ty_expr| {
                    check_type_caps_and_legality(
                        ty_expr,
                        parsed_attrs.required_caps,
                        param.span,
                        self.diags,
                    );
                    if type_expr_contains_ident(ty_expr, "RayQuery") {
                        self.diags.push(Diagnostic::error(
                            "E0213",
                            "`RayQuery` is forbidden as a function parameter",
                            param.span,
                        ));
                    }
                    if type_expr_contains_bda(ty_expr) {
                        uses_bda = true;
                    }
                    if is_self_type_expr(ty_expr) {
                        self_ty
                    } else {
                        let mut type_ctx = self.type_ctx();
                        lower_type_expr(ty_expr, &mut type_ctx, &fn_generics)
                    }
                })
            };

            if let Some(ty_id) = resolved_ty {
                if let Type::Struct { name, .. } = self.module.types.get(ty_id) {
                    let struct_name = name.clone();
                    if let Some(&sid) = self.struct_ids.get(struct_name.as_str())
                        && (param_is_pc || self.module.structs[sid].is_push_constant)
                    {
                        param_push_constant_struct = Some(sid);
                        param_is_pc = true;
                    }
                }
                if param_is_pc {
                    self.check_push_constant_struct_bool(ty_id, param.span);
                }
                if self.module.types.contains_pointer(ty_id) {
                    uses_bda = true;
                }
                if self.module.types.contains_ray_query(ty_id)
                    && param
                        .ty
                        .as_ref()
                        .is_none_or(|t| !type_expr_contains_ident(t, "RayQuery"))
                {
                    self.diags.push(Diagnostic::error(
                        "E0213",
                        "`RayQuery` is forbidden as a function parameter",
                        param.span,
                    ));
                }
                if self.module.types.requires_float16(ty_id)
                    && !parsed_attrs.required_caps.contains(HardwareCaps::FLOAT16)
                    && param.ty.as_ref().is_none_or(|t| !type_expr_uses_f16(t))
                {
                    self.diags.push(
                        Diagnostic::error(
                            "E0232",
                            "`f16` types require `#[cap(float16)]` on the enclosing function",
                            param.span,
                        )
                        .with_note(
                            "Hyaloclast does not enable shader_float16 unconditionally in device_merger.rs",
                        ),
                    );
                }
                self.check_buffer_struct_bool(ty_id, param.span);
            }

            root_scope.insert(param.name.name.clone(), resolved_ty);
            params.push(HirParam {
                name: param.name.name.clone(),
                mode: param.mode,
                ty: resolved_ty,
                builtin,
                is_push_constant: param_is_pc,
                pack_resource_slot,
                span: param.span,
            });
        }

        let return_ty = func.return_ty.as_ref().and_then(|ty_expr| {
            check_type_caps_and_legality(
                ty_expr,
                parsed_attrs.required_caps,
                func.span,
                self.diags,
            );
            if type_expr_contains_ident(ty_expr, "RayQuery") {
                self.diags.push(Diagnostic::error(
                    "E0214",
                    "`RayQuery` is forbidden as a function return type",
                    func.span,
                ));
            }
            if type_expr_contains_bda(ty_expr) {
                uses_bda = true;
            }
            if is_self_type_expr(ty_expr) {
                self_ty
            } else {
                let mut type_ctx = self.type_ctx();
                lower_type_expr(ty_expr, &mut type_ctx, &fn_generics)
            }
        });

        if let Some(ty_id) = return_ty {
            self.check_buffer_struct_bool(ty_id, func.span);
            if self.module.types.contains_ray_query(ty_id)
                && func
                    .return_ty
                    .as_ref()
                    .is_none_or(|t| !type_expr_contains_ident(t, "RayQuery"))
            {
                self.diags.push(Diagnostic::error(
                    "E0214",
                    "`RayQuery` is forbidden as a function return type",
                    func.span,
                ));
            }
            if self.module.types.requires_float16(ty_id)
                && !parsed_attrs.required_caps.contains(HardwareCaps::FLOAT16)
                && func
                    .return_ty
                    .as_ref()
                    .is_none_or(|t| !type_expr_uses_f16(t))
            {
                self.diags.push(
                    Diagnostic::error(
                        "E0232",
                        "`f16` types require `#[cap(float16)]` on the enclosing function",
                        func.span,
                    )
                    .with_note(
                        "Hyaloclast does not enable shader_float16 unconditionally in device_merger.rs",
                    ),
                );
            }
        }

        // Scan function body for legality, local types, mc::* global references, debug channels,
        // BDA, subgroups, params, specs, and Set 3 pack resources.
        let mut scan = BodyScan::default();
        let mut local_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
        if let Some(ref body) = func.body {
            let is_compute = matches!(parsed_attrs.stage, Some(ShaderStage::Compute));
            let is_non_compute_stage = matches!(
                parsed_attrs.stage,
                Some(ShaderStage::Vertex | ShaderStage::Fragment)
            );
            let mut scanner = BodyScanner {
                types: &mut self.module.types,
                structs: &mut self.module.structs,
                struct_types: self.struct_types,
                struct_ids: self.struct_ids,
                const_types: self.const_types,
                workgroup_var_types: self.workgroup_var_types,
                fn_return_types: self.fn_return_types,
                generic_params: &fn_generics,
                mc_imports: self.mc_imports,
                is_compute,
                is_non_compute_stage,
                caps: parsed_attrs.required_caps,
                scopes: vec![root_scope],
                local_types: &mut local_types,
                scan: &mut scan,
                diags: self.diags,
                bool_checked_structs: self.bool_checked_structs,
            };
            scanner.scan_block(body);
        }
        uses_bda |= scan.has(FLAG_BDA);

        // Build single-source-of-truth reflection for entry points (and record BDA/subgroup flags for all fns).
        let is_entry = parsed_attrs.stage.is_some() || parsed_attrs.hook.is_some();
        let reflection = if is_entry {
            self.build_entry_reflection(
                &params,
                param_push_constant_struct,
                &scan,
                uses_bda,
                func.span,
            )
        } else {
            EntryReflection {
                uses_bda,
                uses_subgroups: scan.has(FLAG_SUBGROUPS),
                ..EntryReflection::default()
            }
        };

        self.module.functions.alloc(HirFunction {
            name: func.name.name.clone(),
            stage: parsed_attrs.stage,
            workgroup_size: parsed_attrs.workgroup_size,
            hook: parsed_attrs.hook,
            tier_range,
            required_caps: parsed_attrs.required_caps,
            params,
            return_ty,
            local_types,
            body_ast: func.body.clone(),
            reflection,
            span: func.span,
        });
    }

    fn check_buffer_struct_bool(&mut self, ty_id: TypeId, span: Span) {
        check_buffer_struct_bool_inner(
            &self.module.types,
            self.bool_checked_structs,
            self.diags,
            ty_id,
            span,
        );
    }

    fn check_push_constant_struct_bool(&mut self, ty_id: TypeId, span: Span) {
        let Type::Struct { name, .. } = self.module.types.get(ty_id) else {
            return;
        };
        let struct_name = name.clone();
        if self.bool_checked_structs.contains(struct_name.as_str()) {
            return;
        }
        if self.module.types.contains_bool(ty_id) {
            self.bool_checked_structs.insert(struct_name.clone());
            self.diags.push(
                Diagnostic::error(
                    "E0223",
                    format!(
                        "`bool` is forbidden inside interface (`Block`) push-constant struct `{struct_name}`"
                    ),
                    span,
                )
                .with_note(
                    "Vulkan SPIR-V forbids OpTypeBool inside Block-decorated push-constant, UBO, and SSBO structs; use `u32` instead",
                ),
            );
        }
    }

    fn resolve_spec_id(&mut self, name: &SmolStr) -> u32 {
        if let Some(schema) = self.schema
            && let Some(&spec_id) = schema.spec_ids.get(name)
        {
            return spec_id;
        }
        if let Some(&spec_id) = self.fallback_spec_ids.get(name) {
            return spec_id;
        }
        let id = self.next_fallback_spec_id;
        self.next_fallback_spec_id += 1;
        self.fallback_spec_ids.insert(name.clone(), id);
        id
    }

    fn resolve_param_layout(&mut self, name: &SmolStr) -> (u32, u32) {
        if let Some(schema) = self.schema
            && let Some(&layout) = schema.params.get(name)
        {
            return layout;
        }
        if let Some(&layout) = self.fallback_param_layouts.get(name) {
            return layout;
        }
        let offset = self.next_fallback_param_offset;
        let size = 4u32;
        self.next_fallback_param_offset += size;
        self.fallback_param_layouts
            .insert(name.clone(), (offset, size));
        (offset, size)
    }

    fn build_entry_reflection(
        &mut self,
        params: &[HirParam],
        param_push_constant_struct: Option<HirStructId>,
        scan: &BodyScan,
        uses_bda: bool,
        fn_span: Span,
    ) -> EntryReflection {
        let mut descriptors = Vec::new();
        let mut used_set3_slots: IndexSet<u32> = IndexSet::new();

        // Set 0, binding 0: WorldUBO is part of the engine ABI for every entry point.
        descriptors.push(DescriptorBinding {
            name: SmolStr::new_static("world"),
            set: DESCRIPTOR_SET_PER_PASS,
            binding: 0,
            kind: DescriptorKind::UniformBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
            count: None,
        });

        // Set 0, bindings 1..=14 (or Set 3 for `#[pack_resource(slot)]`): entry-point resource parameters.
        let mut next_pass_binding = 1u32;
        for param in params {
            if param.builtin.is_some() || param.is_push_constant {
                continue;
            }
            let Some(ty_id) = param.ty else {
                continue;
            };
            if let Some(classified) =
                classify_resource_type(self.module.types.get(ty_id), param.mode)
            {
                // `world` is reserved for the ABI WorldUBO at 0:0. Reflection names are the
                // driver's lookup key, so a second descriptor with this name would shadow the
                // UBO and redirect `mc::param` BDA loads to the wrong descriptor.
                if param.name == "world" {
                    self.diags.push(Diagnostic::error(
                        "E0264",
                        format!(
                            "resource parameter `{}` uses the ABI-reserved descriptor name `world`",
                            param.name
                        ),
                        param.span,
                    ));
                    continue;
                }
                if let Some(slot) = param.pack_resource_slot {
                    if used_set3_slots.insert(slot) {
                        descriptors.push(DescriptorBinding {
                            name: param.name.clone(),
                            set: DESCRIPTOR_SET_PACK_RESOURCES,
                            binding: slot,
                            kind: classified.kind,
                            dim: classified.dim,
                            sample_scalar: classified.sample_scalar,
                            image_format: classified.image_format,
                            access: classified.access,
                            is_unsized: classified.is_unsized,
                            count: None,
                        });
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0263",
                            format!("duplicate `#[pack_resource({slot})]` binding slot in Set 3"),
                            param.span,
                        ));
                    }
                } else if next_pass_binding > MAX_PER_PASS_BINDING {
                    self.diags.push(Diagnostic::error(
                        "E0260",
                        format!(
                            "per-pass resource parameter `{}` exceeds Set 0 binding limit (`1..={MAX_PER_PASS_BINDING}`)",
                            param.name
                        ),
                        param.span,
                    ));
                } else {
                    descriptors.push(DescriptorBinding {
                        name: param.name.clone(),
                        set: DESCRIPTOR_SET_PER_PASS,
                        binding: next_pass_binding,
                        kind: classified.kind,
                        dim: classified.dim,
                        sample_scalar: classified.sample_scalar,
                        image_format: classified.image_format,
                        access: classified.access,
                        is_unsized: classified.is_unsized,
                        count: None,
                    });
                    next_pass_binding += 1;
                }
            }
        }

        // Set 0 debug channels allocated immediately after pass resource bindings.
        let mut debug_channels = Vec::with_capacity(scan.debug_channels.len());
        for channel in &scan.debug_channels {
            if next_pass_binding > MAX_PER_PASS_BINDING {
                self.diags.push(Diagnostic::error(
                    "E0261",
                    format!(
                        "debug channel `{channel}` exceeds Set 0 binding limit (`1..={MAX_PER_PASS_BINDING}`)"
                    ),
                    fn_span,
                ));
                break;
            }
            debug_channels.push(DebugChannelBinding {
                channel_name: channel.clone(),
                set: DESCRIPTOR_SET_PER_PASS,
                binding: next_pass_binding,
            });
            next_pass_binding += 1;
        }

        // Set 1: Scene globals (0 = tlas, 1 = instances, 3 = lights).
        if scan.has(FLAG_TLAS) {
            descriptors.push(DescriptorBinding {
                name: SmolStr::new_static("tlas"),
                set: DESCRIPTOR_SET_SCENE_GLOBALS,
                binding: 0,
                kind: DescriptorKind::AccelerationStructure,
                dim: None,
                sample_scalar: None,
                image_format: None,
                access: DescriptorAccess::ReadOnly,
                is_unsized: false,
                count: None,
            });
        }
        if scan.has(FLAG_INSTANCES) {
            descriptors.push(DescriptorBinding {
                name: SmolStr::new_static("instances"),
                set: DESCRIPTOR_SET_SCENE_GLOBALS,
                binding: 1,
                kind: DescriptorKind::StorageBuffer,
                dim: None,
                sample_scalar: None,
                image_format: None,
                access: DescriptorAccess::ReadOnly,
                is_unsized: false,
                count: None,
            });
        }
        if scan.has(FLAG_LIGHTS) {
            descriptors.push(DescriptorBinding {
                name: SmolStr::new_static("lights"),
                set: DESCRIPTOR_SET_SCENE_GLOBALS,
                binding: 3,
                kind: DescriptorKind::StorageBuffer,
                dim: None,
                sample_scalar: None,
                image_format: None,
                access: DescriptorAccess::ReadOnly,
                is_unsized: false,
                count: None,
            });
        }

        // Set 2: engine-compatible bindless array (binding 0, 4096 combined image samplers).
        // Hyaloclast declares this slot as `COMBINED_IMAGE_SAMPLER[4096]`; emitting an unsized
        // sampled-image array here would make the engine's rspirv reflection reject the layout.
        if scan.has(FLAG_BINDLESS) {
            descriptors.push(DescriptorBinding {
                name: SmolStr::new_static("bindless"),
                set: DESCRIPTOR_SET_BINDLESS,
                binding: 0,
                kind: DescriptorKind::CombinedImageSampler,
                dim: Some(TextureDim::Dim2D),
                sample_scalar: Some(SampleScalar::F32),
                image_format: None,
                access: DescriptorAccess::ReadOnly,
                is_unsized: false,
                count: Some(BINDLESS_IMAGE_COUNT),
            });
        }

        // Set 3: Pack texture assets referenced via `mc::resources::*` (`bindings 0..=15`).
        // Resolve explicit `PackSchema::resource_slots` assignments first so deterministic
        // sequential fallback never collides with a schema-reserved slot.
        let reserved_schema_slots: IndexSet<u32> = self.schema.map_or_else(IndexSet::new, |s| {
            s.resource_slots.values().copied().collect()
        });
        let mut fallback_pack_resources: Vec<&SmolStr> = Vec::new();

        for res_name in &scan.pack_resources {
            if res_name == "world" {
                self.diags.push(Diagnostic::error(
                    "E0264",
                    format!(
                        "pack resource `mc::resources::{res_name}` uses the ABI-reserved descriptor name `world`"
                    ),
                    fn_span,
                ));
                continue;
            }
            if let Some(schema) = self.schema
                && let Some(&slot) = schema.resource_slots.get(res_name)
            {
                if slot >= PACK_RESOURCE_SLOTS {
                    self.diags.push(Diagnostic::error(
                        "E0262",
                        format!(
                            "pack resource `mc::resources::{res_name}` schema slot `{slot}` exceeds Set 3 slot limit (`0..={}`)",
                            PACK_RESOURCE_SLOTS - 1
                        ),
                        fn_span,
                    ));
                } else if used_set3_slots.insert(slot) {
                    descriptors.push(DescriptorBinding {
                        name: res_name.clone(),
                        set: DESCRIPTOR_SET_PACK_RESOURCES,
                        binding: slot,
                        kind: DescriptorKind::SampledImage,
                        dim: Some(TextureDim::Dim2D),
                        sample_scalar: Some(SampleScalar::F32),
                        image_format: None,
                        access: DescriptorAccess::ReadOnly,
                        is_unsized: false,
                        count: None,
                    });
                } else {
                    self.diags.push(Diagnostic::error(
                        "E0263",
                        format!(
                            "duplicate Set 3 binding slot `{slot}` for pack resource `mc::resources::{res_name}`"
                        ),
                        fn_span,
                    ));
                }
            } else {
                fallback_pack_resources.push(res_name);
            }
        }

        for res_name in fallback_pack_resources {
            let next_slot = (0..PACK_RESOURCE_SLOTS)
                .find(|s| !used_set3_slots.contains(s) && !reserved_schema_slots.contains(s));
            if let Some(slot) = next_slot {
                used_set3_slots.insert(slot);
                descriptors.push(DescriptorBinding {
                    name: res_name.clone(),
                    set: DESCRIPTOR_SET_PACK_RESOURCES,
                    binding: slot,
                    kind: DescriptorKind::SampledImage,
                    dim: Some(TextureDim::Dim2D),
                    sample_scalar: Some(SampleScalar::F32),
                    image_format: None,
                    access: DescriptorAccess::ReadOnly,
                    is_unsized: false,
                    count: None,
                });
            } else {
                self.diags.push(Diagnostic::error(
                    "E0262",
                    format!(
                        "pack resource `mc::resources::{res_name}` exceeds Set 3 slot limit (`0..={}`)",
                        PACK_RESOURCE_SLOTS - 1
                    ),
                    fn_span,
                ));
                break;
            }
        }

        descriptors.sort_by_key(|d| (d.set, d.binding));

        // Push constants: prefer explicit entry-point parameter struct, otherwise unique module-level `#[push_constant]`.
        let pc_struct_id = param_push_constant_struct.or_else(|| {
            (self.module_push_constants.len() == 1).then(|| self.module_push_constants[0])
        });
        let mut push_constants = Vec::new();
        let mut push_constant_size = 0u32;
        if let Some(sid) = pc_struct_id
            && let Some(ref layout) = self.module.structs[sid].computed_layout
        {
            push_constant_size = layout.size;
            push_constants = layout
                .field_offsets
                .iter()
                .map(|(name, byte_offset, byte_size)| PushConstantField {
                    name: name.clone(),
                    byte_offset: *byte_offset,
                    byte_size: *byte_size,
                })
                .collect();
        }

        let used_specs = scan
            .used_specs
            .iter()
            .map(|name| SpecConstantUsage {
                name: name.clone(),
                spec_id: self.resolve_spec_id(name),
            })
            .collect();

        let used_params = scan
            .used_params
            .iter()
            .map(|name| {
                let (byte_offset, byte_size) = self.resolve_param_layout(name);
                PackParamBinding {
                    name: name.clone(),
                    byte_offset,
                    byte_size,
                }
            })
            .collect();

        EntryReflection {
            descriptors,
            push_constants,
            push_constant_size,
            used_specs,
            used_params,
            debug_channels,
            uses_bda,
            uses_subgroups: scan.has(FLAG_SUBGROUPS),
        }
    }
}

fn validate_generic_bounds(
    generics: &[GenericParam],
    trait_names: &IndexSet<SmolStr>,
    diags: &mut DiagnosticBag,
) {
    for param in generics {
        if let Some(ref bound) = param.bound {
            let bound_name = bound.joined();
            if !trait_names.contains(bound_name.as_str()) {
                diags.push(Diagnostic::error(
                    "E0271",
                    format!(
                        "unknown trait bound `{bound_name}` on generic parameter `{}`",
                        param.name.name
                    ),
                    bound.span,
                ));
            }
        }
    }
}

fn parse_cap_args(args: &[Expr], out_caps: &mut HardwareCaps, diags: &mut DiagnosticBag) {
    for arg in args {
        if let Expr::Path { path, .. } = arg {
            let name = path.joined();
            if let Some(cap) = HardwareCaps::from_cap_name(&name) {
                *out_caps |= cap;
            } else {
                diags.push(Diagnostic::error(
                    "E0230",
                    format!("unknown hardware capability `{name}`"),
                    arg.span(),
                ));
            }
        }
    }
}

fn collect_mc_imports(unit: &SourceUnit) -> IndexSet<SmolStr> {
    let mut imported = IndexSet::new();
    for imp in &unit.imports {
        let prefix = imp.prefix.joined();
        if prefix == "mc" {
            match &imp.tree {
                ImportTree::Group(idents) => {
                    for ident in idents {
                        imported.insert(ident.name.clone());
                    }
                }
                ImportTree::Glob => {
                    for name in ["world", "tlas", "instances", "lights", "bindless"] {
                        imported.insert(SmolStr::new_static(name));
                    }
                }
                ImportTree::Single => {}
            }
        } else if let Some(rest) = prefix.strip_prefix("mc::")
            && matches!(imp.tree, ImportTree::Single)
        {
            imported.insert(SmolStr::new(rest));
        }
    }
    imported
}

fn parse_tier_range(attrs: &[Attribute], diags: &mut DiagnosticBag) -> (u8, u8) {
    let mut range = (0u8, MAX_TIER);
    for attr in attrs {
        for entry in &attr.entries {
            if entry.name.name != "tier" {
                continue;
            }
            let parsed = match entry.args.first() {
                Some(Expr::Binary {
                    op: BinaryOp::RangeInclusive,
                    lhs,
                    rhs,
                    ..
                }) => {
                    let lo = parse_u8_lit(Some(lhs));
                    let hi = parse_u8_lit(Some(rhs));
                    match (lo, hi) {
                        (Some(a), Some(b)) if (a..=MAX_TIER).contains(&b) => Some((a, b)),
                        _ => None,
                    }
                }
                Some(Expr::Binary {
                    op: BinaryOp::Range,
                    lhs,
                    rhs,
                    ..
                }) => {
                    let lo = parse_u8_lit(Some(lhs));
                    let hi = parse_u8_lit(Some(rhs));
                    match (lo, hi) {
                        (Some(a), Some(b))
                            if b.checked_sub(1)
                                .is_some_and(|hi_inc| (a..=MAX_TIER).contains(&hi_inc)) =>
                        {
                            Some((a, b - 1))
                        }
                        _ => None,
                    }
                }
                Some(expr @ Expr::IntLit(..)) => parse_u8_lit(Some(expr))
                    .filter(|&v| v <= MAX_TIER)
                    .map(|v| (v, v)),
                _ => None,
            };

            if let Some(valid) = parsed {
                range = valid;
            } else {
                diags.push(Diagnostic::error(
                    "E0235",
                    format!(
                        "invalid `#[tier(...)]` range; expected `min..=max`, `min..max`, or single tier within `0..={MAX_TIER}`"
                    ),
                    entry.span,
                ));
            }
        }
    }
    range
}

fn parse_fn_attribute(
    attr: &Attribute,
    func: &FnDecl,
    out: &mut ParsedFnAttrs,
    diags: &mut DiagnosticBag,
) {
    for entry in &attr.entries {
        match entry.name.name.as_str() {
            "cap" => parse_cap_args(&entry.args, &mut out.required_caps, diags),
            "maximally_reconverges" => {
                out.required_caps |= HardwareCaps::MAXIMAL_RECONVERGENCE;
            }
            "quad_derivatives" => {
                out.required_caps |= HardwareCaps::QUAD_CONTROL;
            }
            "stage" if attr.is_at_decorator => {
                if let Some(arg) = entry.args.first() {
                    match arg {
                        Expr::Call { callee, args, span } => {
                            if let Expr::Path { path, .. } = callee.as_ref()
                                && path.joined() == "compute"
                            {
                                let x = parse_u32_lit(args.first().map(|a| &a.value));
                                let y = parse_u32_lit(args.get(1).map(|a| &a.value));
                                let z = parse_u32_lit(args.get(2).map(|a| &a.value));
                                if let (Some(dx), Some(dy), Some(dz)) = (x, y, z)
                                    && dx > 0
                                    && dy > 0
                                    && dz > 0
                                {
                                    out.stage = Some(ShaderStage::Compute);
                                    out.workgroup_size = Some([dx, dy, dz]);
                                } else {
                                    diags.push(Diagnostic::error(
                                        "E0234",
                                        "`@stage(compute(x, y, z))` requires three positive integer workgroup dimensions",
                                        *span,
                                    ));
                                }
                            }
                        }
                        Expr::Path { path, .. } => match path.joined().as_str() {
                            "vertex" => {
                                out.stage = Some(ShaderStage::Vertex);
                                out.workgroup_size = None;
                            }
                            "fragment" => {
                                out.stage = Some(ShaderStage::Fragment);
                                out.workgroup_size = None;
                            }
                            other => diags.push(Diagnostic::error(
                                "E0231",
                                format!("unsupported shader stage `{other}` in VSL 2.0"),
                                arg.span(),
                            )),
                        },
                        _ => {}
                    }
                }
            }
            "hook" if attr.is_at_decorator => {
                if let Some(Expr::Path { path, .. }) = entry.args.first() {
                    let name = path.joined();
                    if let Some(hook_stage) = HookStage::from_name(&name) {
                        if hook_stage.uses_alpha_over_blending() {
                            let returns_vec4 = func.return_ty.as_ref().is_some_and(|t| {
                                matches!(t, TypeExpr::Named { path, .. } if path.joined() == "vec4")
                            });
                            if !returns_vec4 {
                                diags.push(Diagnostic::error(
                                    "E0240",
                                    "`@hook(clouds)` functions must return `vec4` for alpha-over compositing",
                                    func.span,
                                ));
                            }
                        }
                        out.hook = Some(hook_stage);
                    } else {
                        diags.push(Diagnostic::error(
                            "E0241",
                            format!("unknown `@hook` stage `{name}`"),
                            entry.span,
                        ));
                    }
                }
            }
            _ => {}
        }
    }
}

struct ClassifiedResource {
    kind: DescriptorKind,
    dim: Option<TextureDim>,
    sample_scalar: Option<SampleScalar>,
    image_format: Option<ImageFormat>,
    access: DescriptorAccess,
    is_unsized: bool,
}

const fn scalar_to_sample_scalar(scalar: ScalarKind) -> Option<SampleScalar> {
    match scalar {
        ScalarKind::F32 | ScalarKind::F16 => Some(SampleScalar::F32),
        ScalarKind::I32 | ScalarKind::I64 => Some(SampleScalar::I32),
        ScalarKind::U32 | ScalarKind::U64 => Some(SampleScalar::U32),
        ScalarKind::Bool => None,
    }
}

fn classify_resource_type(ty: &Type, mode: ParamMode) -> Option<ClassifiedResource> {
    match ty {
        Type::Texture2D {
            sample_scalar,
            unsized_array,
        } => Some(ClassifiedResource {
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: scalar_to_sample_scalar(*sample_scalar),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: *unsized_array,
        }),
        Type::Texture2DArray { sample_scalar } => Some(ClassifiedResource {
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim2DArray),
            sample_scalar: scalar_to_sample_scalar(*sample_scalar),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
        }),
        Type::Texture3D { sample_scalar } => Some(ClassifiedResource {
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::Dim3D),
            sample_scalar: scalar_to_sample_scalar(*sample_scalar),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
        }),
        Type::TextureCube { sample_scalar } => Some(ClassifiedResource {
            kind: DescriptorKind::SampledImage,
            dim: Some(TextureDim::DimCube),
            sample_scalar: scalar_to_sample_scalar(*sample_scalar),
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
        }),
        Type::RwImage2D(fmt) => Some(ClassifiedResource {
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: None,
            image_format: Some(*fmt),
            access: rw_image_access(mode),
            is_unsized: false,
        }),
        Type::RwImage2DArray(fmt) => Some(ClassifiedResource {
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2DArray),
            sample_scalar: None,
            image_format: Some(*fmt),
            access: rw_image_access(mode),
            is_unsized: false,
        }),
        Type::RwImage3D(fmt) => Some(ClassifiedResource {
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim3D),
            sample_scalar: None,
            image_format: Some(*fmt),
            access: rw_image_access(mode),
            is_unsized: false,
        }),
        Type::Sampler | Type::SamplerShadow => Some(ClassifiedResource {
            kind: DescriptorKind::Sampler,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
        }),
        Type::Buffer { .. } => Some(ClassifiedResource {
            kind: DescriptorKind::StorageBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadOnly,
            is_unsized: false,
        }),
        Type::RwBuffer { .. } => Some(ClassifiedResource {
            kind: DescriptorKind::StorageBuffer,
            dim: None,
            sample_scalar: None,
            image_format: None,
            access: DescriptorAccess::ReadWrite,
            is_unsized: false,
        }),
        _ => None,
    }
}

const fn rw_image_access(mode: ParamMode) -> DescriptorAccess {
    if matches!(mode, ParamMode::Out) {
        DescriptorAccess::WriteOnly
    } else {
        DescriptorAccess::ReadWrite
    }
}

#[derive(Default)]
struct BodyScan {
    flags: u8,
    debug_channels: IndexSet<SmolStr>,
    used_specs: IndexSet<SmolStr>,
    used_params: IndexSet<SmolStr>,
    pack_resources: IndexSet<SmolStr>,
}

impl BodyScan {
    const fn has(&self, flag: u8) -> bool {
        (self.flags & flag) != 0
    }

    fn set(&mut self, flag: u8) {
        self.flags |= flag;
    }
}

struct BodyScanner<'a> {
    types: &'a mut TypeArena,
    structs: &'a mut Arena<HirStruct>,
    struct_types: &'a mut IndexMap<SmolStr, TypeId>,
    struct_ids: &'a mut IndexMap<SmolStr, HirStructId>,
    const_types: &'a IndexMap<SmolStr, TypeId>,
    workgroup_var_types: &'a IndexMap<SmolStr, TypeId>,
    fn_return_types: &'a IndexMap<SmolStr, TypeId>,
    generic_params: &'a IndexSet<SmolStr>,
    mc_imports: &'a IndexSet<SmolStr>,
    is_compute: bool,
    is_non_compute_stage: bool,
    caps: HardwareCaps,
    scopes: Vec<IndexMap<SmolStr, Option<TypeId>>>,
    local_types: &'a mut IndexMap<SmolStr, TypeId>,
    scan: &'a mut BodyScan,
    diags: &'a mut DiagnosticBag,
    bool_checked_structs: &'a mut IndexSet<SmolStr>,
}

impl BodyScanner<'_> {
    fn is_shadowed(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|s| s.contains_key(name))
    }

    fn lookup_var_type(&self, name: &str) -> Option<TypeId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&ty) = scope.get(name) {
                return ty;
            }
        }
        self.workgroup_var_types
            .get(name)
            .copied()
            .or_else(|| self.const_types.get(name).copied())
    }

    fn insert_local(&mut self, name: SmolStr, ty: Option<TypeId>) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.clone(), ty);
        }
        if let Some(ty_id) = ty {
            self.local_types.insert(name, ty_id);
        }
    }

    fn scan_block(&mut self, block: &Block) -> Option<TypeId> {
        self.scopes.push(IndexMap::new());
        for stmt in &block.stmts {
            self.scan_stmt(stmt);
        }
        let tail_ty = block
            .tail_expr
            .as_deref()
            .and_then(|tail| self.scan_expr(tail));
        self.scopes.pop();
        tail_ty
    }

    fn scan_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                name,
                ty,
                init,
                span,
                ..
            } => {
                let mut resolved_ty = None;
                let mut flagged_logical_ptr = false;

                if let Some(ty_expr) = ty {
                    check_type_caps_and_legality(ty_expr, self.caps, *span, self.diags);
                    if is_logical_pointer_type(ty_expr) {
                        self.diags.push(
                            Diagnostic::error(
                                "E0215",
                                format!(
                                    "local logical pointer variable `{}` is forbidden in SPIR-V",
                                    name.name
                                ),
                                *span,
                            )
                            .with_note(
                                "Logical pointers cannot be stored in local variables; use physical BDA pointers (`ptr<T, align = N>`) or pass `&var` directly to intrinsics",
                            ),
                        );
                        flagged_logical_ptr = true;
                    }
                    if type_expr_contains_bda(ty_expr) {
                        self.scan.set(FLAG_BDA);
                    }
                    let mut type_ctx = TypeLowerCtx::full(
                        self.types,
                        self.structs,
                        self.struct_types,
                        self.struct_ids,
                    );
                    resolved_ty = lower_type_expr(ty_expr, &mut type_ctx, self.generic_params);
                    if let Some(ty_id) = resolved_ty {
                        check_buffer_struct_bool_inner(
                            self.types,
                            self.bool_checked_structs,
                            self.diags,
                            ty_id,
                            *span,
                        );
                    }
                    if let Some(ty_id) = resolved_ty
                        && self.types.requires_float16(ty_id)
                        && !self.caps.contains(HardwareCaps::FLOAT16)
                        && !type_expr_uses_f16(ty_expr)
                    {
                        self.diags.push(Diagnostic::error(
                            "E0232",
                            "`f16` types require `#[cap(float16)]` on the enclosing function",
                            *span,
                        ));
                    }
                }

                let init_ty = init.as_ref().and_then(|expr| {
                    if !flagged_logical_ptr && expr_yields_addr_of(expr) {
                        self.diags.push(
                            Diagnostic::error(
                                "E0215",
                                format!(
                                    "storing `&...` into local pointer variable `{}` is forbidden in SPIR-V",
                                    name.name
                                ),
                                *span,
                            )
                            .with_note(
                                "Logical pointers cannot be stored in local variables; pass `&var` directly to intrinsics",
                            ),
                        );
                    }
                    let inferred = self.scan_expr(expr);
                    if let Some(ty_id) = inferred
                        && self.types.contains_ray_query(ty_id)
                    {
                        self.diags.push(Diagnostic::error(
                            "E0217",
                            "`RayQuery` cannot be assigned or copied into another variable",
                            *span,
                        ));
                    }
                    inferred
                });

                self.insert_local(name.name.clone(), resolved_ty.or(init_ty));
            }
            Stmt::Return { value, span } => {
                if let Some(expr) = value {
                    let ty = self.scan_expr(expr);
                    if let Some(ty_id) = ty
                        && self.types.contains_ray_query(ty_id)
                    {
                        self.diags.push(Diagnostic::error(
                            "E0214",
                            "`RayQuery` cannot be returned from a function",
                            *span,
                        ));
                    }
                }
            }
            Stmt::For {
                binding,
                iter,
                body,
                ..
            } => {
                let iter_ty = self.scan_expr(iter);
                self.scopes.push(IndexMap::new());
                self.insert_local(binding.name.clone(), iter_ty);
                self.scan_block(body);
                self.scopes.pop();
            }
            Stmt::While { cond, body, .. } => {
                self.scan_expr(cond);
                self.scan_block(body);
            }
            Stmt::Loop { body, .. } => {
                self.scan_block(body);
            }
            Stmt::Switch {
                selector,
                cases,
                default,
                ..
            } => {
                self.scan_expr(selector);
                for case in cases {
                    self.scan_expr(&case.value);
                    self.scan_block(&case.body);
                }
                if let Some(def_block) = default {
                    self.scan_block(def_block);
                }
            }
            Stmt::Expr { expr, .. } => {
                self.scan_expr(expr);
            }
            Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }

    fn scan_expr(&mut self, expr: &Expr) -> Option<TypeId> {
        match expr {
            Expr::IntLit(raw, _) => {
                let scalar = if raw.ends_with("u64") {
                    ScalarKind::U64
                } else if raw.ends_with("i64") {
                    ScalarKind::I64
                } else if raw.ends_with("u32") || raw.ends_with('u') {
                    ScalarKind::U32
                } else {
                    ScalarKind::I32
                };
                Some(self.types.alloc(Type::Scalar(scalar)))
            }
            Expr::FloatLit(raw, span) => {
                if raw.ends_with("f16") {
                    if !self.caps.contains(HardwareCaps::FLOAT16) {
                        self.diags.push(Diagnostic::error(
                            "E0232",
                            "`f16` literals require `#[cap(float16)]` on the enclosing function",
                            *span,
                        ));
                    }
                    Some(self.types.alloc(Type::Scalar(ScalarKind::F16)))
                } else {
                    Some(self.types.alloc(Type::Scalar(ScalarKind::F32)))
                }
            }
            Expr::BoolLit(_, _) => Some(self.types.alloc(Type::Scalar(ScalarKind::Bool))),
            Expr::StringLit(_, _) | Expr::NoneLit(_) => None,
            Expr::Path {
                path, turbofish, ..
            } => {
                let joined = path.joined();
                match joined.as_str() {
                    "mc::world" => self.scan.set(FLAG_WORLD),
                    "world" if self.mc_imports.contains("world") && !self.is_shadowed("world") => {
                        self.scan.set(FLAG_WORLD);
                    }
                    "mc::tlas" => self.scan.set(FLAG_TLAS),
                    "tlas" if self.mc_imports.contains("tlas") && !self.is_shadowed("tlas") => {
                        self.scan.set(FLAG_TLAS);
                    }
                    "mc::instances" => self.scan.set(FLAG_INSTANCES),
                    "instances"
                        if self.mc_imports.contains("instances")
                            && !self.is_shadowed("instances") =>
                    {
                        self.scan.set(FLAG_INSTANCES);
                    }
                    "mc::lights" => self.scan.set(FLAG_LIGHTS),
                    "lights"
                        if self.mc_imports.contains("lights") && !self.is_shadowed("lights") =>
                    {
                        self.scan.set(FLAG_LIGHTS);
                    }
                    "mc::bindless" => self.scan.set(FLAG_BINDLESS),
                    "bindless"
                        if self.mc_imports.contains("bindless")
                            && !self.is_shadowed("bindless") =>
                    {
                        self.scan.set(FLAG_BINDLESS);
                    }
                    _ => {}
                }

                if let Some(res_name) = joined.strip_prefix("mc::resources::")
                    && !res_name.is_empty()
                {
                    self.scan.pack_resources.insert(SmolStr::new(res_name));
                }

                if let Some(spec_name) = joined
                    .strip_prefix("spec::")
                    .or_else(|| joined.strip_prefix("mc::spec::"))
                {
                    self.scan.used_specs.insert(SmolStr::new(spec_name));
                }

                if let Some(last) = path.segments.last()
                    && last.name.starts_with("subgroup_")
                {
                    self.scan.set(FLAG_SUBGROUPS);
                }

                for arg in turbofish {
                    if let TypeArg::Type(ty_expr) = arg
                        && type_expr_contains_bda(ty_expr)
                    {
                        self.scan.set(FLAG_BDA);
                    }
                }

                if path.segments.len() == 1 {
                    let ident = path.segments[0].name.as_str();
                    if self.is_non_compute_stage
                        && !self.is_shadowed(ident)
                        && self.workgroup_var_types.contains_key(ident)
                    {
                        self.diags.push(Diagnostic::error(
                            "E0251",
                            format!("`workgroup var` `{ident}` is only valid in compute shaders"),
                            path.span,
                        ));
                    }
                    self.lookup_var_type(ident)
                } else {
                    None
                }
            }
            Expr::Call { callee, args, span } => {
                let mut callee_path = None;
                let mut callee_name = None;
                if let Expr::Path { path, .. } = callee.as_ref() {
                    let joined = path.joined();
                    if joined == "mc::debug_write" {
                        if self.is_compute {
                            if let Some(first_arg) = args.first()
                                && let Expr::StringLit(channel, _) = &first_arg.value
                            {
                                self.scan.debug_channels.insert(channel.clone());
                            }
                        } else {
                            self.diags.push(Diagnostic::error(
                                "E0250",
                                "`mc::debug_write` is only valid inside `@stage(compute(...))` entry points",
                                *span,
                            ));
                        }
                    } else if joined == "mc::param" {
                        if let Some(name) = extract_name_arg(args.first().map(|a| &a.value)) {
                            self.scan.used_params.insert(name);
                        }
                    } else if joined == "mc::spec"
                        && let Some(name) = extract_name_arg(args.first().map(|a| &a.value))
                    {
                        self.scan.used_specs.insert(name);
                    }

                    if let Some(last) = path.segments.last() {
                        if last.name.starts_with("subgroup_") {
                            self.scan.set(FLAG_SUBGROUPS);
                        }
                        callee_name = Some(last.name.clone());
                    }
                    callee_path = Some(joined);
                }

                self.scan_expr(callee);
                let mut arg_types = Vec::with_capacity(args.len());
                for arg in args {
                    arg_types.push(self.scan_expr(&arg.value));
                }

                if let Some(ref fn_name) = callee_name {
                    if matches!(fn_name.as_str(), "f16" | "f16vec2" | "f16vec3" | "f16vec4")
                        && !self.caps.contains(HardwareCaps::FLOAT16)
                    {
                        self.diags.push(
                            Diagnostic::error(
                                "E0232",
                                format!(
                                    "`{fn_name}(...)` constructor requires `#[cap(float16)]` on the enclosing function"
                                ),
                                *span,
                            )
                            .with_note(
                                "Hyaloclast does not enable shader_float16 unconditionally in device_merger.rs",
                            ),
                        );
                    }
                    if fn_name.starts_with("atomic_") {
                        self.check_atomic_caps(fn_name.as_str(), &arg_types, *span);
                    }
                }

                // Infer constructor/cast or declared module function return types.
                callee_name
                    .as_deref()
                    .and_then(|name| infer_builtin_constructor_type(name, self.types))
                    .or_else(|| {
                        callee_path
                            .as_deref()
                            .and_then(|path| self.fn_return_types.get(path).copied())
                    })
                    .or_else(|| {
                        callee_name
                            .as_deref()
                            .and_then(|name| self.fn_return_types.get(name).copied())
                    })
            }
            Expr::MethodCall {
                receiver,
                method,
                turbofish,
                args,
                ..
            } => {
                for arg in turbofish {
                    if let TypeArg::Type(ty_expr) = arg
                        && type_expr_contains_bda(ty_expr)
                    {
                        self.scan.set(FLAG_BDA);
                    }
                }
                let recv_ty = self.scan_expr(receiver);
                for arg in args {
                    self.scan_expr(&arg.value);
                }
                if method.name == "offset" {
                    recv_ty
                } else {
                    recv_ty.and_then(|ty_id| {
                        self.resolve_method_return_type(ty_id, method.name.as_str())
                    })
                }
            }
            Expr::Field {
                receiver, field, ..
            } => {
                if let Expr::Path { path, .. } = receiver.as_ref() {
                    let joined = path.joined();
                    if joined == "pack::params"
                        || joined == "mc::params"
                        || (joined == "params" && !self.is_shadowed("params"))
                    {
                        self.scan.used_params.insert(field.name.clone());
                    } else if joined == "mc::spec"
                        || (joined == "spec" && !self.is_shadowed("spec"))
                    {
                        self.scan.used_specs.insert(field.name.clone());
                    }
                }
                let recv_ty = self.scan_expr(receiver)?;
                self.resolve_field_type(recv_ty, field.name.as_str())
            }
            Expr::StructLit { path, fields, .. } => {
                for (_, value) in fields {
                    self.scan_expr(value);
                }
                self.struct_types.get(path.joined().as_str()).copied()
            }
            Expr::Unary { op, operand, .. } => {
                let inner = self.scan_expr(operand);
                match op {
                    UnaryOp::Not => Some(self.types.alloc(Type::Scalar(ScalarKind::Bool))),
                    UnaryOp::AddrOf | UnaryOp::Neg | UnaryOp::BitNot => inner,
                }
            }
            Expr::Binary { op, lhs, rhs, span } => {
                let lhs_ty = self.scan_expr(lhs);
                let rhs_ty = self.scan_expr(rhs);
                if matches!(op, BinaryOp::Assign) {
                    if expr_yields_addr_of(rhs) {
                        self.diags.push(Diagnostic::error(
                            "E0215",
                            "assigning a logical pointer (`&...`) to a variable is forbidden in SPIR-V",
                            *span,
                        ));
                    }
                    if lhs_ty.is_some_and(|t| self.types.contains_ray_query(t))
                        || rhs_ty.is_some_and(|t| self.types.contains_ray_query(t))
                    {
                        self.diags.push(Diagnostic::error(
                            "E0217",
                            "`RayQuery` cannot be assigned or copied",
                            *span,
                        ));
                    }
                }
                match op {
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::And
                    | BinaryOp::Or => Some(self.types.alloc(Type::Scalar(ScalarKind::Bool))),
                    _ => lhs_ty.or(rhs_ty),
                }
            }
            Expr::Index {
                receiver, index, ..
            } => {
                let recv_ty = self.scan_expr(receiver);
                self.scan_expr(index);
                recv_ty.and_then(|ty_id| match self.types.get(ty_id) {
                    Type::Array { element, .. }
                    | Type::Buffer { element }
                    | Type::RwBuffer { element } => Some(*element),
                    _ => None,
                })
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => {
                self.scan_expr(cond);
                let then_ty = self.scan_block(then_branch);
                let else_ty = else_branch.as_deref().and_then(|eb| self.scan_expr(eb));
                if else_branch.is_some()
                    && (then_ty.is_some_and(|t| self.types.contains_ray_query(t))
                        || else_ty.is_some_and(|t| self.types.contains_ray_query(t)))
                {
                    self.diags.push(
                        Diagnostic::error(
                            "E0216",
                            "`RayQuery` is forbidden in `if`-expression (`OpPhi`) merges",
                            *span,
                        )
                        .with_note(
                            "SPV_KHR_ray_query forbids RayQueryKHR values in OpPhi instructions",
                        ),
                    );
                }
                then_ty.or(else_ty)
            }
            Expr::Block(b) => self.scan_block(b),
            Expr::ArrayLit { elements, .. } => {
                let mut elem_ty = None;
                for el in elements {
                    let t = self.scan_expr(el);
                    if elem_ty.is_none() {
                        elem_ty = t;
                    }
                }
                elem_ty.and_then(|e| {
                    u32::try_from(elements.len())
                        .ok()
                        .map(|len| self.types.alloc(Type::Array { element: e, len }))
                })
            }
        }
    }

    fn resolve_method_return_type(&self, recv_ty: TypeId, method_name: &str) -> Option<TypeId> {
        let base_ty = match self.types.get(recv_ty) {
            Type::Pointer { pointee, .. } => *pointee,
            _ => recv_ty,
        };
        let Type::Struct { name, .. } = self.types.get(base_ty) else {
            return None;
        };
        let qualified = format!("{name}::{method_name}");
        self.fn_return_types.get(qualified.as_str()).copied()
    }

    fn resolve_field_type(&mut self, recv_ty: TypeId, field_name: &str) -> Option<TypeId> {
        let base_ty = match self.types.get(recv_ty) {
            Type::Pointer { pointee, .. } => *pointee,
            _ => recv_ty,
        };
        match self.types.get(base_ty).clone() {
            Type::Struct { fields, .. } => fields
                .iter()
                .find_map(|(name, ty_id)| (name == field_name).then_some(*ty_id)),
            Type::Vector { scalar, .. } => {
                let swizzle_len = field_name.len();
                if swizzle_len == 1 {
                    Some(self.types.alloc(Type::Scalar(scalar)))
                } else if (2..=4).contains(&swizzle_len) {
                    u8::try_from(swizzle_len)
                        .ok()
                        .map(|len| self.types.alloc(Type::Vector { scalar, len }))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn check_atomic_caps(&mut self, fn_name: &str, arg_types: &[Option<TypeId>], span: Span) {
        let mut is_int64 = false;
        let mut is_float = fn_name.starts_with("atomic_f");

        for ty_id in arg_types.iter().flatten() {
            let scalar = match self.types.get(*ty_id) {
                Type::Pointer { pointee, .. } => match self.types.get(*pointee) {
                    Type::Scalar(s) => Some(*s),
                    _ => None,
                },
                Type::Scalar(s) => Some(*s),
                _ => None,
            };
            match scalar {
                Some(ScalarKind::I64 | ScalarKind::U64) => is_int64 = true,
                Some(ScalarKind::F32 | ScalarKind::F16) => is_float = true,
                _ => {}
            }
        }

        if is_int64 && !self.caps.contains(HardwareCaps::INT64_ATOMICS) {
            self.diags.push(Diagnostic::error(
                "E0237",
                format!(
                    "64-bit atomic `{fn_name}` requires `#[cap(int64_atomics)]` on the enclosing function"
                ),
                span,
            ));
        }

        if is_float && !self.caps.contains(HardwareCaps::FLOAT_ATOMICS) {
            self.diags.push(Diagnostic::error(
                "E0238",
                format!(
                    "floating-point atomic `{fn_name}` requires `#[cap(float_atomics)]` on the enclosing function"
                ),
                span,
            ));
        }
    }
}

fn extract_name_arg(expr: Option<&Expr>) -> Option<SmolStr> {
    match expr? {
        Expr::Path { path, .. } => Some(SmolStr::new(path.joined())),
        Expr::StringLit(s, _) => Some(s.clone()),
        _ => None,
    }
}

fn infer_builtin_constructor_type(name: &str, types: &mut TypeArena) -> Option<TypeId> {
    let ty = match name {
        "bool" => Type::Scalar(ScalarKind::Bool),
        "i32" => Type::Scalar(ScalarKind::I32),
        "u32" => Type::Scalar(ScalarKind::U32),
        "i64" => Type::Scalar(ScalarKind::I64),
        "u64" => Type::Scalar(ScalarKind::U64),
        "f32" => Type::Scalar(ScalarKind::F32),
        "f16" => Type::Scalar(ScalarKind::F16),
        "vec2" => Type::Vector {
            scalar: ScalarKind::F32,
            len: 2,
        },
        "vec3" => Type::Vector {
            scalar: ScalarKind::F32,
            len: 3,
        },
        "vec4" => Type::Vector {
            scalar: ScalarKind::F32,
            len: 4,
        },
        "ivec2" => Type::Vector {
            scalar: ScalarKind::I32,
            len: 2,
        },
        "ivec3" => Type::Vector {
            scalar: ScalarKind::I32,
            len: 3,
        },
        "ivec4" => Type::Vector {
            scalar: ScalarKind::I32,
            len: 4,
        },
        "uvec2" => Type::Vector {
            scalar: ScalarKind::U32,
            len: 2,
        },
        "uvec3" => Type::Vector {
            scalar: ScalarKind::U32,
            len: 3,
        },
        "uvec4" => Type::Vector {
            scalar: ScalarKind::U32,
            len: 4,
        },
        "f16vec2" => Type::Vector {
            scalar: ScalarKind::F16,
            len: 2,
        },
        "f16vec3" => Type::Vector {
            scalar: ScalarKind::F16,
            len: 3,
        },
        "f16vec4" => Type::Vector {
            scalar: ScalarKind::F16,
            len: 4,
        },
        "mat2" => Type::Matrix {
            scalar: ScalarKind::F32,
            cols: 2,
            rows: 2,
        },
        "mat3" => Type::Matrix {
            scalar: ScalarKind::F32,
            cols: 3,
            rows: 3,
        },
        "mat4" => Type::Matrix {
            scalar: ScalarKind::F32,
            cols: 4,
            rows: 4,
        },
        _ => return None,
    };
    Some(types.alloc(ty))
}

fn is_logical_pointer_type(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Named { path, args, .. } => {
            let name = path.joined();
            if name == "ptr" || name == "ptr_mut" {
                // Physical BDA pointers explicitly carry `align = N`; unqualified `ptr<T>`
                // on a local variable represents an illegal SPIR-V logical pointer variable.
                let has_align = args.iter().any(
                    |arg| matches!(arg, TypeArg::NamedExpr { name, .. } if name.name == "align"),
                );
                !has_align
            } else {
                false
            }
        }
        TypeExpr::Array { element, .. } => is_logical_pointer_type(element),
    }
}

fn expr_yields_addr_of(expr: &Expr) -> bool {
    match expr {
        Expr::Unary {
            op: UnaryOp::AddrOf,
            ..
        } => true,
        Expr::Block(b) => b.tail_expr.as_deref().is_some_and(expr_yields_addr_of),
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            then_branch
                .tail_expr
                .as_deref()
                .is_some_and(expr_yields_addr_of)
                || else_branch.as_deref().is_some_and(expr_yields_addr_of)
        }
        _ => false,
    }
}

fn check_type_caps_and_legality(
    ty: &TypeExpr,
    caps: HardwareCaps,
    span: Span,
    diags: &mut DiagnosticBag,
) {
    if type_expr_uses_f16(ty) && !caps.contains(HardwareCaps::FLOAT16) {
        diags.push(
            Diagnostic::error(
                "E0232",
                "`f16` types require `#[cap(float16)]` on the enclosing function",
                span,
            )
            .with_note(
                "Hyaloclast does not enable shader_float16 unconditionally in device_merger.rs",
            ),
        );
    }

    if type_expr_contains_ident(ty, "RayQuery") {
        if type_expr_is_ray_query_array(ty) {
            diags.push(Diagnostic::error(
                "E0211",
                "`RayQuery` arrays are forbidden by SPV_KHR_ray_query",
                span,
            ));
        }
        if !caps.contains(HardwareCaps::RAY_QUERY) {
            diags.push(Diagnostic::error(
                "E0233",
                "`RayQuery` requires `#[cap(ray_query)]` on the enclosing function",
                span,
            ));
        }
    }

    check_bda_pointer_alignment(ty, diags);
}

fn check_bda_pointer_alignment(ty: &TypeExpr, diags: &mut DiagnosticBag) {
    match ty {
        TypeExpr::Array { element, .. } => check_bda_pointer_alignment(element, diags),
        TypeExpr::Named { path, args, .. } => {
            let name = path.joined();
            if name == "ptr" || name == "ptr_mut" {
                for arg in args {
                    if let TypeArg::NamedExpr {
                        name: arg_name,
                        value,
                        span: arg_span,
                    } = arg
                        && arg_name.name == "align"
                    {
                        let valid = parse_u32_lit(Some(value))
                            .is_some_and(|a| a > 0 && a.is_power_of_two());
                        if !valid {
                            diags.push(Diagnostic::error(
                                "E0218",
                                "BDA pointer `align` must be a positive power of two",
                                *arg_span,
                            ));
                        }
                    }
                }
            }
            for arg in args {
                if let TypeArg::Type(inner) = arg {
                    check_bda_pointer_alignment(inner, diags);
                }
            }
        }
    }
}

fn type_expr_uses_f16(ty: &TypeExpr) -> bool {
    type_expr_contains_ident(ty, "f16")
        || type_expr_contains_ident(ty, "f16vec2")
        || type_expr_contains_ident(ty, "f16vec3")
        || type_expr_contains_ident(ty, "f16vec4")
}

fn type_expr_is_ray_query_array(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Array { element, .. } => type_expr_contains_ident(element, "RayQuery"),
        TypeExpr::Named {
            path,
            unsized_array,
            args,
            ..
        } => {
            (*unsized_array && path.segments.iter().any(|s| s.name == "RayQuery"))
                || args.iter().any(|arg| match arg {
                    TypeArg::Type(inner) => type_expr_is_ray_query_array(inner),
                    TypeArg::NamedExpr { .. } => false,
                })
        }
    }
}

fn type_expr_contains_ident(ty: &TypeExpr, target: &str) -> bool {
    match ty {
        TypeExpr::Named { path, args, .. } => {
            if path.segments.iter().any(|s| s.name == target) {
                return true;
            }
            args.iter().any(|arg| match arg {
                TypeArg::Type(inner) => type_expr_contains_ident(inner, target),
                TypeArg::NamedExpr { .. } => false,
            })
        }
        TypeExpr::Array { element, .. } => type_expr_contains_ident(element, target),
    }
}

fn type_expr_contains_bda(ty: &TypeExpr) -> bool {
    type_expr_contains_ident(ty, "ptr") || type_expr_contains_ident(ty, "ptr_mut")
}

fn is_self_type_expr(ty: &TypeExpr) -> bool {
    matches!(
        ty,
        TypeExpr::Named {
            path,
            args,
            unsized_array: false,
            ..
        } if args.is_empty() && path.joined() == "Self"
    )
}

struct TypeLowerCtx<'a> {
    types: &'a mut TypeArena,
    structs: Option<&'a mut Arena<HirStruct>>,
    struct_types: &'a mut IndexMap<SmolStr, TypeId>,
    struct_ids: Option<&'a mut IndexMap<SmolStr, HirStructId>>,
}

impl<'a> TypeLowerCtx<'a> {
    const fn types_only(
        types: &'a mut TypeArena,
        struct_types: &'a mut IndexMap<SmolStr, TypeId>,
    ) -> Self {
        Self {
            types,
            structs: None,
            struct_types,
            struct_ids: None,
        }
    }

    const fn full(
        types: &'a mut TypeArena,
        structs: &'a mut Arena<HirStruct>,
        struct_types: &'a mut IndexMap<SmolStr, TypeId>,
        struct_ids: &'a mut IndexMap<SmolStr, HirStructId>,
    ) -> Self {
        Self {
            types,
            structs: Some(structs),
            struct_types,
            struct_ids: Some(struct_ids),
        }
    }
}

fn type_expr_has_type_args(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Named { args, .. } => args.iter().any(|arg| matches!(arg, TypeArg::Type(_))),
        TypeExpr::Array { element, .. } => type_expr_has_type_args(element),
    }
}

fn check_buffer_struct_bool_inner(
    types: &TypeArena,
    bool_checked_structs: &mut IndexSet<SmolStr>,
    diags: &mut DiagnosticBag,
    ty_id: TypeId,
    span: Span,
) {
    let elem_ty = match types.get(ty_id) {
        Type::Buffer { element }
        | Type::RwBuffer { element }
        | Type::Pointer {
            pointee: element, ..
        } => Some(*element),
        _ => None,
    };
    let Some(mut elem_id) = elem_ty else {
        return;
    };
    while let Type::Array { element, .. } = types.get(elem_id) {
        elem_id = *element;
    }
    let Type::Struct { name, .. } = types.get(elem_id) else {
        return;
    };
    let struct_name = name.clone();
    if bool_checked_structs.contains(struct_name.as_str()) {
        return;
    }
    if types.contains_bool(elem_id) {
        bool_checked_structs.insert(struct_name.clone());
        diags.push(
            Diagnostic::error(
                "E0223",
                format!(
                    "`bool` is forbidden inside interface (`Block`) struct `{struct_name}` used as a buffer element"
                ),
                span,
            )
            .with_note(
                "Vulkan SPIR-V forbids OpTypeBool inside Block-decorated push-constant, UBO, and SSBO structs; use `u32` instead",
            ),
        );
    }
}

fn contains_type_param(types: &TypeArena, id: TypeId) -> bool {
    let mut visited = Vec::new();
    contains_type_param_inner(types, id, &mut visited)
}

fn contains_type_param_inner(types: &TypeArena, id: TypeId, visited: &mut Vec<TypeId>) -> bool {
    if visited.contains(&id) {
        return false;
    }
    visited.push(id);
    let result = match types.get(id) {
        Type::TypeParam(_) => true,
        Type::Array { element, .. } | Type::Buffer { element } | Type::RwBuffer { element } => {
            contains_type_param_inner(types, *element, visited)
        }
        Type::Pointer { pointee, .. } => contains_type_param_inner(types, *pointee, visited),
        Type::Struct { fields, .. } => fields
            .iter()
            .any(|(_, field_ty)| contains_type_param_inner(types, *field_ty, visited)),
        _ => false,
    };
    visited.pop();
    result
}

const fn scalar_name(scalar: ScalarKind) -> &'static str {
    match scalar {
        ScalarKind::Bool => "bool",
        ScalarKind::I32 => "i32",
        ScalarKind::U32 => "u32",
        ScalarKind::I64 => "i64",
        ScalarKind::U64 => "u64",
        ScalarKind::F32 => "f32",
        ScalarKind::F16 => "f16",
    }
}

fn format_type_name(types: &TypeArena, id: TypeId) -> String {
    match types.get(id) {
        Type::Void => "void".to_owned(),
        Type::Scalar(s) => scalar_name(*s).to_owned(),
        Type::Vector { scalar, len } => match scalar {
            ScalarKind::F32 => format!("vec{len}"),
            ScalarKind::I32 => format!("ivec{len}"),
            ScalarKind::U32 => format!("uvec{len}"),
            ScalarKind::F16 => format!("f16vec{len}"),
            ScalarKind::Bool => format!("bvec{len}"),
            ScalarKind::I64 => format!("i64vec{len}"),
            ScalarKind::U64 => format!("u64vec{len}"),
        },
        Type::Matrix { scalar, cols, rows } => {
            let prefix = if *scalar == ScalarKind::F32 {
                ""
            } else {
                scalar_name(*scalar)
            };
            if cols == rows {
                format!("{prefix}mat{cols}")
            } else {
                format!("{prefix}mat{cols}x{rows}")
            }
        }
        Type::Array { element, len } => format!("[{}; {len}]", format_type_name(types, *element)),
        Type::Pointer {
            pointee,
            mutable,
            align,
        } => {
            let kind = if *mutable { "ptr_mut" } else { "ptr" };
            format!(
                "{kind}<{}, align = {align}>",
                format_type_name(types, *pointee)
            )
        }
        Type::Struct { name, .. } => name.to_string(),
        Type::Texture2D {
            sample_scalar,
            unsized_array,
        } => {
            if *unsized_array {
                format!("Texture2D<{}>[]", scalar_name(*sample_scalar))
            } else {
                format!("Texture2D<{}>", scalar_name(*sample_scalar))
            }
        }
        Type::Texture2DArray { sample_scalar } => {
            format!("Texture2DArray<{}>", scalar_name(*sample_scalar))
        }
        Type::Texture3D { sample_scalar } => {
            format!("Texture3D<{}>", scalar_name(*sample_scalar))
        }
        Type::TextureCube { sample_scalar } => {
            format!("TextureCube<{}>", scalar_name(*sample_scalar))
        }
        Type::RwImage2D(fmt) => format!("RWImage2D<{fmt:?}>"),
        Type::RwImage2DArray(fmt) => format!("RWImage2DArray<{fmt:?}>"),
        Type::RwImage3D(fmt) => format!("RWImage3D<{fmt:?}>"),
        Type::Sampler => "Sampler".to_owned(),
        Type::SamplerShadow => "SamplerShadow".to_owned(),
        Type::Buffer { element } => format!("Buffer<{}>", format_type_name(types, *element)),
        Type::RwBuffer { element } => format!("RWBuffer<{}>", format_type_name(types, *element)),
        Type::AccelStruct => "AccelStruct".to_owned(),
        Type::RayQuery => "RayQuery".to_owned(),
        Type::TypeParam(p) => p.to_string(),
    }
}

fn substitute_type(types: &mut TypeArena, id: TypeId, subst: &IndexMap<SmolStr, TypeId>) -> TypeId {
    if !contains_type_param(types, id) {
        return id;
    }
    match types.get(id).clone() {
        Type::TypeParam(param) => subst.get(&param).copied().unwrap_or(id),
        Type::Array { element, len } => {
            let new_elem = substitute_type(types, element, subst);
            if new_elem == element {
                id
            } else {
                types.alloc(Type::Array {
                    element: new_elem,
                    len,
                })
            }
        }
        Type::Pointer {
            pointee,
            mutable,
            align,
        } => {
            let new_pointee = substitute_type(types, pointee, subst);
            if new_pointee == pointee {
                id
            } else {
                types.alloc(Type::Pointer {
                    pointee: new_pointee,
                    mutable,
                    align,
                })
            }
        }
        Type::Buffer { element } => {
            let new_elem = substitute_type(types, element, subst);
            if new_elem == element {
                id
            } else {
                types.alloc(Type::Buffer { element: new_elem })
            }
        }
        Type::RwBuffer { element } => {
            let new_elem = substitute_type(types, element, subst);
            if new_elem == element {
                id
            } else {
                types.alloc(Type::RwBuffer { element: new_elem })
            }
        }
        Type::Struct {
            name,
            layout,
            fields,
        } => {
            let mut changed = false;
            let mut new_fields = Vec::with_capacity(fields.len());
            for (field_name, field_ty) in fields {
                let sub_ty = substitute_type(types, field_ty, subst);
                if sub_ty != field_ty {
                    changed = true;
                }
                new_fields.push((field_name, sub_ty));
            }
            if changed {
                types.alloc(Type::Struct {
                    name,
                    layout,
                    fields: new_fields,
                })
            } else {
                id
            }
        }
        _ => id,
    }
}

fn try_monomorphize_struct(
    base_name: &str,
    args: &[TypeArg],
    ctx: &mut TypeLowerCtx<'_>,
    generic_params: &IndexSet<SmolStr>,
) -> Option<TypeId> {
    let type_args: Vec<&TypeExpr> = args
        .iter()
        .filter_map(|arg| match arg {
            TypeArg::Type(inner) => Some(inner),
            TypeArg::NamedExpr { .. } => None,
        })
        .collect();
    if type_args.is_empty() {
        return None;
    }

    let (base_generic_params, base_layout, base_is_pc, base_caps, base_fields, base_span) = {
        let structs = ctx.structs.as_deref()?;
        let struct_ids = ctx.struct_ids.as_deref()?;
        let &base_sid = struct_ids.get(base_name)?;
        let base_struct = &structs[base_sid];
        if base_struct.generic_params.is_empty()
            || base_struct.generic_params.len() != type_args.len()
        {
            return None;
        }
        (
            base_struct.generic_params.clone(),
            base_struct.layout,
            base_struct.is_push_constant,
            base_struct.required_caps,
            base_struct.fields.clone(),
            base_struct.span,
        )
    };

    let mut lowered_args = Vec::with_capacity(type_args.len());
    for arg_expr in type_args {
        let arg_ty = lower_type_expr(arg_expr, ctx, generic_params)?;
        if contains_type_param(ctx.types, arg_ty) {
            return None;
        }
        lowered_args.push(arg_ty);
    }

    let arg_names: Vec<String> = lowered_args
        .iter()
        .map(|&arg_ty| format_type_name(ctx.types, arg_ty))
        .collect();
    let mono_name = SmolStr::from(format!("{base_name}<{}>", arg_names.join(", ")));

    if let Some(&existing_ty) = ctx.struct_types.get(mono_name.as_str()) {
        return Some(existing_ty);
    }

    let mut subst: IndexMap<SmolStr, TypeId> = IndexMap::with_capacity(base_generic_params.len());
    for (param_name, arg_ty) in base_generic_params.into_iter().zip(lowered_args) {
        subst.insert(param_name, arg_ty);
    }

    let mut hir_fields = Vec::with_capacity(base_fields.len());
    let mut type_fields = Vec::with_capacity(base_fields.len());
    for field in base_fields {
        let concrete_ty = substitute_type(ctx.types, field.ty, &subst);
        type_fields.push((field.name.clone(), concrete_ty));
        hir_fields.push(HirStructField {
            name: field.name,
            ty: concrete_ty,
            span: field.span,
        });
    }

    let computed_layout = ctx.types.compute_struct_layout(&type_fields, base_layout);
    let mono_type_id = ctx.types.alloc(Type::Struct {
        name: mono_name.clone(),
        layout: base_layout,
        fields: type_fields,
    });

    if let (Some(structs), Some(struct_ids)) =
        (ctx.structs.as_deref_mut(), ctx.struct_ids.as_deref_mut())
    {
        let mono_sid = structs.alloc(HirStruct {
            name: mono_name.clone(),
            generic_params: Vec::new(),
            layout: base_layout,
            is_push_constant: base_is_pc,
            required_caps: base_caps,
            fields: hir_fields,
            computed_layout,
            span: base_span,
        });
        struct_ids.insert(mono_name.clone(), mono_sid);
    }
    ctx.struct_types.insert(mono_name, mono_type_id);
    Some(mono_type_id)
}

fn lower_type_expr(
    ty: &TypeExpr,
    ctx: &mut TypeLowerCtx<'_>,
    generic_params: &IndexSet<SmolStr>,
) -> Option<TypeId> {
    match ty {
        TypeExpr::Named {
            path,
            args,
            unsized_array,
            ..
        } => {
            let name = path.joined();
            if args.is_empty() && !*unsized_array && generic_params.contains(name.as_str()) {
                return Some(ctx.types.alloc(Type::TypeParam(SmolStr::new(name))));
            }
            if let Some(&id) = ctx.struct_types.get(name.as_str()) {
                if !args.is_empty()
                    && let Some(mono_id) = try_monomorphize_struct(&name, args, ctx, generic_params)
                {
                    return Some(mono_id);
                }
                return Some(id);
            }
            let resolved = match name.as_str() {
                "bool" => Type::Scalar(ScalarKind::Bool),
                "i32" => Type::Scalar(ScalarKind::I32),
                "u32" => Type::Scalar(ScalarKind::U32),
                "i64" => Type::Scalar(ScalarKind::I64),
                "u64" => Type::Scalar(ScalarKind::U64),
                "f32" => Type::Scalar(ScalarKind::F32),
                "f16" => Type::Scalar(ScalarKind::F16),
                "vec2" => Type::Vector {
                    scalar: ScalarKind::F32,
                    len: 2,
                },
                "vec3" => Type::Vector {
                    scalar: ScalarKind::F32,
                    len: 3,
                },
                "vec4" => Type::Vector {
                    scalar: ScalarKind::F32,
                    len: 4,
                },
                "ivec2" => Type::Vector {
                    scalar: ScalarKind::I32,
                    len: 2,
                },
                "ivec3" => Type::Vector {
                    scalar: ScalarKind::I32,
                    len: 3,
                },
                "ivec4" => Type::Vector {
                    scalar: ScalarKind::I32,
                    len: 4,
                },
                "uvec2" => Type::Vector {
                    scalar: ScalarKind::U32,
                    len: 2,
                },
                "uvec3" => Type::Vector {
                    scalar: ScalarKind::U32,
                    len: 3,
                },
                "uvec4" => Type::Vector {
                    scalar: ScalarKind::U32,
                    len: 4,
                },
                "f16vec2" => Type::Vector {
                    scalar: ScalarKind::F16,
                    len: 2,
                },
                "f16vec3" => Type::Vector {
                    scalar: ScalarKind::F16,
                    len: 3,
                },
                "f16vec4" => Type::Vector {
                    scalar: ScalarKind::F16,
                    len: 4,
                },
                "mat2" => Type::Matrix {
                    scalar: ScalarKind::F32,
                    cols: 2,
                    rows: 2,
                },
                "mat3" => Type::Matrix {
                    scalar: ScalarKind::F32,
                    cols: 3,
                    rows: 3,
                },
                "mat4" => Type::Matrix {
                    scalar: ScalarKind::F32,
                    cols: 4,
                    rows: 4,
                },
                "mat3x4" => Type::Matrix {
                    scalar: ScalarKind::F32,
                    cols: 3,
                    rows: 4,
                },
                "mat4x3" => Type::Matrix {
                    scalar: ScalarKind::F32,
                    cols: 4,
                    rows: 3,
                },
                "ptr" | "ptr_mut" => {
                    let pointee = args
                        .iter()
                        .find_map(|arg| match arg {
                            TypeArg::Type(inner) => lower_type_expr(inner, ctx, generic_params),
                            TypeArg::NamedExpr { .. } => None,
                        })
                        .unwrap_or_else(|| ctx.types.alloc(Type::Void));
                    let align = args
                        .iter()
                        .find_map(|arg| match arg {
                            TypeArg::NamedExpr { name, value, .. } if name.name == "align" => {
                                parse_u32_lit(Some(value))
                            }
                            _ => None,
                        })
                        .unwrap_or(4);
                    Type::Pointer {
                        pointee,
                        mutable: name == "ptr_mut",
                        align,
                    }
                }
                "Texture2D" => Type::Texture2D {
                    sample_scalar: first_type_arg_scalar(args).unwrap_or(ScalarKind::F32),
                    unsized_array: *unsized_array,
                },
                "Texture2DArray" => Type::Texture2DArray {
                    sample_scalar: first_type_arg_scalar(args).unwrap_or(ScalarKind::F32),
                },
                "Texture3D" => Type::Texture3D {
                    sample_scalar: first_type_arg_scalar(args).unwrap_or(ScalarKind::F32),
                },
                "TextureCube" => Type::TextureCube {
                    sample_scalar: first_type_arg_scalar(args).unwrap_or(ScalarKind::F32),
                },
                "RWImage2D" => Type::RwImage2D(first_type_arg_image_format(args)?),
                "RWImage2DArray" => Type::RwImage2DArray(first_type_arg_image_format(args)?),
                "RWImage3D" => Type::RwImage3D(first_type_arg_image_format(args)?),
                "Sampler" => Type::Sampler,
                "SamplerShadow" => Type::SamplerShadow,
                "Buffer" => {
                    let element = args
                        .iter()
                        .find_map(|arg| match arg {
                            TypeArg::Type(inner) => lower_type_expr(inner, ctx, generic_params),
                            TypeArg::NamedExpr { .. } => None,
                        })
                        .unwrap_or_else(|| ctx.types.alloc(Type::Void));
                    Type::Buffer { element }
                }
                "RWBuffer" => {
                    let element = args
                        .iter()
                        .find_map(|arg| match arg {
                            TypeArg::Type(inner) => lower_type_expr(inner, ctx, generic_params),
                            TypeArg::NamedExpr { .. } => None,
                        })
                        .unwrap_or_else(|| ctx.types.alloc(Type::Void));
                    Type::RwBuffer { element }
                }
                "AccelerationStructure" | "AccelStruct" => Type::AccelStruct,
                "RayQuery" => Type::RayQuery,
                _ => return None,
            };
            Some(ctx.types.alloc(resolved))
        }
        TypeExpr::Array { element, len, .. } => {
            let elem_id = lower_type_expr(element, ctx, generic_params)?;
            let count = parse_u32_lit(Some(len))?;
            Some(ctx.types.alloc(Type::Array {
                element: elem_id,
                len: count,
            }))
        }
    }
}

fn first_type_arg_scalar(args: &[TypeArg]) -> Option<ScalarKind> {
    let TypeArg::Type(TypeExpr::Named { path, .. }) = args.first()? else {
        return None;
    };
    match path.joined().as_str() {
        "bool" => Some(ScalarKind::Bool),
        "i32" => Some(ScalarKind::I32),
        "u32" => Some(ScalarKind::U32),
        "i64" => Some(ScalarKind::I64),
        "u64" => Some(ScalarKind::U64),
        "f32" => Some(ScalarKind::F32),
        "f16" => Some(ScalarKind::F16),
        _ => None,
    }
}

fn first_type_arg_image_format(args: &[TypeArg]) -> Option<ImageFormat> {
    let TypeArg::Type(TypeExpr::Named { path, .. }) = args.first()? else {
        return None;
    };
    ImageFormat::from_name(&path.joined())
}

fn parse_u32_lit(expr: Option<&Expr>) -> Option<u32> {
    let Expr::IntLit(raw, _) = expr? else {
        return None;
    };
    let clean = raw
        .trim_end_matches("u64")
        .trim_end_matches("i64")
        .trim_end_matches("u32")
        .trim_end_matches("i32")
        .trim_end_matches(['u', 'i'])
        .replace('_', "");
    if let Some(hex) = clean
        .strip_prefix("0x")
        .or_else(|| clean.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).ok()
    } else if let Some(bin) = clean
        .strip_prefix("0b")
        .or_else(|| clean.strip_prefix("0B"))
    {
        u32::from_str_radix(bin, 2).ok()
    } else if let Some(oct) = clean
        .strip_prefix("0o")
        .or_else(|| clean.strip_prefix("0O"))
    {
        u32::from_str_radix(oct, 8).ok()
    } else {
        clean.parse::<u32>().ok()
    }
}

fn parse_u8_lit(expr: Option<&Expr>) -> Option<u8> {
    parse_u32_lit(expr).and_then(|v| u8::try_from(v).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vsl_diagnostics::FileId;
    use vsl_syntax::parse_source;

    #[test]
    fn reflects_sunlit_vanilla_reflections() {
        let src = include_str!("../../../examples/sunlit_vanilla/reflections.vsl");
        let parsed = parse_source(FileId(0), src);
        assert!(
            !parsed.has_errors(),
            "parse errors: {:?}",
            parsed.diagnostics
        );

        let (hir, diags) = lower_unit(
            &parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::RAY_QUERY,
        );
        assert!(
            !diags.has_errors(),
            "lowering errors: {:?}",
            diags.as_slice()
        );
        assert_eq!(hir.module_name, "pack::reflections");
        assert_eq!(hir.pruned_functions, 0);

        let func = hir
            .find_function("trace_glossy_reflections")
            .expect("trace_glossy_reflections should be present at Tier 2");
        assert_eq!(func.stage, Some(ShaderStage::Compute));
        assert_eq!(func.workgroup_size, Some([8, 8, 1]));
        assert_eq!(func.tier_range, (2, 3));
        assert_eq!(func.required_caps, HardwareCaps::RAY_QUERY);
        assert!(func.local_types.contains_key("rq"));
        assert!(func.local_types.contains_key("coord"));

        let refl = &func.reflection;
        assert_eq!(refl.descriptors.len(), 7);
        assert_eq!(refl.descriptors[0].name, "world");
        assert_eq!(refl.descriptors[0].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[0].binding, 0);
        assert_eq!(refl.descriptors[0].kind, DescriptorKind::UniformBuffer);

        // Set 0: pass resources (bindings 1..=3) with dim & sample_scalar populated.
        assert_eq!(refl.descriptors[1].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[1].binding, 1);
        assert_eq!(refl.descriptors[1].name, "gbuffer_normal");
        assert_eq!(refl.descriptors[1].kind, DescriptorKind::SampledImage);
        assert_eq!(refl.descriptors[1].dim, Some(TextureDim::Dim2D));
        assert_eq!(refl.descriptors[1].sample_scalar, Some(SampleScalar::F32));

        assert_eq!(refl.descriptors[2].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[2].binding, 2);
        assert_eq!(refl.descriptors[2].name, "gbuffer_depth");
        assert_eq!(refl.descriptors[2].kind, DescriptorKind::SampledImage);
        assert_eq!(refl.descriptors[2].dim, Some(TextureDim::Dim2D));
        assert_eq!(refl.descriptors[2].sample_scalar, Some(SampleScalar::F32));

        assert_eq!(refl.descriptors[3].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[3].binding, 3);
        assert_eq!(refl.descriptors[3].name, "out_refl");
        assert_eq!(refl.descriptors[3].kind, DescriptorKind::StorageImage);
        assert_eq!(refl.descriptors[3].dim, Some(TextureDim::Dim2D));
        assert_eq!(refl.descriptors[3].sample_scalar, None);
        assert_eq!(refl.descriptors[3].image_format, Some(ImageFormat::Rgba16f));
        assert_eq!(refl.descriptors[3].access, DescriptorAccess::ReadWrite);

        // Set 1: tlas (binding 0) + instances (binding 1).
        assert_eq!(refl.descriptors[4].set, DESCRIPTOR_SET_SCENE_GLOBALS);
        assert_eq!(refl.descriptors[4].binding, 0);
        assert_eq!(
            refl.descriptors[4].kind,
            DescriptorKind::AccelerationStructure
        );

        assert_eq!(refl.descriptors[5].set, DESCRIPTOR_SET_SCENE_GLOBALS);
        assert_eq!(refl.descriptors[5].binding, 1);
        assert_eq!(refl.descriptors[5].kind, DescriptorKind::StorageBuffer);
        assert_eq!(refl.descriptors[5].access, DescriptorAccess::ReadOnly);

        // Set 2: engine-compatible bindless combined image sampler array (binding 0).
        assert_eq!(refl.descriptors[6].set, DESCRIPTOR_SET_BINDLESS);
        assert_eq!(refl.descriptors[6].binding, 0);
        assert_eq!(
            refl.descriptors[6].kind,
            DescriptorKind::CombinedImageSampler
        );
        assert_eq!(refl.descriptors[6].dim, Some(TextureDim::Dim2D));
        assert_eq!(refl.descriptors[6].sample_scalar, Some(SampleScalar::F32));
        assert!(!refl.descriptors[6].is_unsized);
        assert_eq!(refl.descriptors[6].count, Some(BINDLESS_IMAGE_COUNT));

        // Debug channel allocated on Set 0 immediately after pass resources (binding 4).
        assert_eq!(refl.debug_channels.len(), 1);
        assert_eq!(refl.debug_channels[0].channel_name, "custom_reflections");
        assert_eq!(refl.debug_channels[0].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.debug_channels[0].binding, 4);
    }

    #[test]
    fn prunes_tier2_reflections_at_tier0_without_cap_errors() {
        let src = include_str!("../../../examples/sunlit_vanilla/reflections.vsl");
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());

        // Tier 0 with empty caps must prune `#[tier(2..=3)]` before checking `#[cap(ray_query)]`.
        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        assert!(
            !diags.has_errors(),
            "pruned fn should not emit cap errors: {:?}",
            diags.as_slice()
        );
        assert_eq!(hir.pruned_functions, 1);
        assert!(hir.find_function("trace_glossy_reflections").is_none());
    }

    #[test]
    fn reflects_sunlit_vanilla_tiled_lighting() {
        let src = include_str!("../../../examples/sunlit_vanilla/tiled_lighting.vsl");
        let parsed = parse_source(FileId(0), src);
        assert!(
            !parsed.has_errors(),
            "parse errors: {:?}",
            parsed.diagnostics
        );

        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        assert!(
            !diags.has_errors(),
            "lowering errors: {:?}",
            diags.as_slice()
        );
        assert_eq!(hir.workgroup_vars.len(), 2);

        let func = hir
            .find_function("cull_tile_lights")
            .expect("cull_tile_lights should be lowered");
        assert_eq!(func.stage, Some(ShaderStage::Compute));
        assert_eq!(func.workgroup_size, Some([8, 8, 1]));

        let refl = &func.reflection;
        assert_eq!(refl.push_constant_size, 16);
        assert_eq!(refl.push_constants.len(), 4);
        assert_eq!(refl.push_constants[0].name, "step_size");
        assert_eq!(refl.push_constants[0].byte_offset, 0);
        assert_eq!(refl.push_constants[3].name, "phi_depth");
        assert_eq!(refl.push_constants[3].byte_offset, 12);

        // Set 0: gbuffer_depth (1), out_lit (2); Set 1: lights (3).
        assert_eq!(refl.descriptors.len(), 4);
        assert_eq!(refl.descriptors[0].name, "world");
        assert_eq!(refl.descriptors[0].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[0].binding, 0);
        assert_eq!(refl.descriptors[1].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[1].binding, 1);
        assert_eq!(refl.descriptors[1].name, "gbuffer_depth");

        assert_eq!(refl.descriptors[2].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[2].binding, 2);
        assert_eq!(refl.descriptors[2].name, "out_lit");
        assert_eq!(refl.descriptors[2].image_format, Some(ImageFormat::Rgba16f));

        assert_eq!(refl.descriptors[3].set, DESCRIPTOR_SET_SCENE_GLOBALS);
        assert_eq!(refl.descriptors[3].binding, 3);
        assert_eq!(refl.descriptors[3].name, "lights");
        assert_eq!(refl.descriptors[3].kind, DescriptorKind::StorageBuffer);
        assert_eq!(refl.descriptors[3].access, DescriptorAccess::ReadOnly);
    }

    #[test]
    fn lowers_sunlit_vanilla_gi_cache_and_restir() {
        let gi_src = include_str!("../../../examples/sunlit_vanilla/gi_cache.vsl");
        let gi_parsed = parse_source(FileId(0), gi_src);
        assert!(!gi_parsed.has_errors());
        let (_, gi_diags) = lower_unit(
            &gi_parsed.unit,
            RenderTier::Tier1ComputeGi,
            HardwareCaps::INT64_ATOMICS,
        );
        assert!(
            !gi_diags.has_errors(),
            "gi_cache errors: {:?}",
            gi_diags.as_slice()
        );

        let restir_src = include_str!("../../../examples/sunlit_vanilla/restir.vsl");
        let restir_parsed = parse_source(FileId(0), restir_src);
        assert!(!restir_parsed.has_errors());
        let (_, restir_diags) = lower_unit(
            &restir_parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::MAXIMAL_RECONVERGENCE,
        );
        assert!(
            !restir_diags.has_errors(),
            "restir errors: {:?}",
            restir_diags.as_slice()
        );
    }

    #[test]
    fn detects_world_ubo_bda_and_subgroups() {
        let src = r"
            module pack::test_features;

            #[cap(maximal_reconvergence)]
            @stage(compute(8, 8, 1))
            pub fn inspect_features(
                data: ptr_mut<u32, align = 4>,
                buf: Buffer<f32>,
            ) {
                let cam = mc::world.camera_pos;
                let mask = subgroup_ballot(cam.x > 0.0);
                data.offset(0i64).store(subgroup_ballot_bit_count(mask));
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());

        let (hir, diags) = lower_unit(
            &parsed.unit,
            RenderTier::Tier1ComputeGi,
            HardwareCaps::MAXIMAL_RECONVERGENCE,
        );
        assert!(!diags.has_errors(), "errors: {:?}", diags.as_slice());

        let func = hir.find_function("inspect_features").expect("fn present");
        let refl = &func.reflection;
        assert!(refl.uses_bda);
        assert!(refl.uses_subgroups);
        assert_eq!(refl.descriptors.len(), 2);
        assert_eq!(refl.descriptors[0].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[0].binding, 0);
        assert_eq!(refl.descriptors[0].kind, DescriptorKind::UniformBuffer);
        assert_eq!(refl.descriptors[1].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(refl.descriptors[1].binding, 1);
        assert_eq!(refl.descriptors[1].kind, DescriptorKind::StorageBuffer);
        assert_eq!(refl.descriptors[1].access, DescriptorAccess::ReadOnly);
    }

    #[test]
    fn enforces_int64_and_float_atomic_caps() {
        let src_i64 = r"
            pub const EMPTY_KEY: u64 = 0u64;
            pub fn bad_i64_atomic(keys: ptr_mut<u64, align = 8>, h: u64) {
                let k = keys.offset(0i64);
                let _p = atomic_cas(k, EMPTY_KEY, h);
            }
        ";
        let parsed = parse_source(FileId(0), src_i64);
        assert!(!parsed.has_errors());
        let (_, diags) = lower_unit(
            &parsed.unit,
            RenderTier::Tier3PathTracing,
            HardwareCaps::all(),
        );
        assert!(diags.as_slice().iter().any(|d| d.code == "E0237"));

        let src_f32_atomic = r"
            pub fn bad_f32_atomic(acc: ptr_mut<f32, align = 4>) {
                let _v = atomic_add(acc, 1.0);
            }
        ";
        let parsed_f = parse_source(FileId(0), src_f32_atomic);
        assert!(!parsed_f.has_errors());
        let (_, diags_f) = lower_unit(
            &parsed_f.unit,
            RenderTier::Tier3PathTracing,
            HardwareCaps::all(),
        );
        assert!(diags_f.as_slice().iter().any(|d| d.code == "E0238"));
    }

    #[test]
    fn forbids_local_logical_pointer_variables() {
        let src = r"
            pub fn bad_ptrs() {
                let x: u32 = 1u;
                let p = &x;
                let q: ptr<u32>;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        let e0215_count = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0215")
            .count();
        assert_eq!(e0215_count, 2);
    }

    #[test]
    fn forbids_rayquery_in_phi_assign_param_return_and_array() {
        let src = r"
            #[cap(ray_query)]
            pub fn bad_rq(p: RayQuery, cond: bool) -> RayQuery {
                let mut a: RayQuery;
                let mut b: RayQuery;
                let arr: [RayQuery; 2];
                let _m = if cond { a } else { b };
                a = b;
                return a;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = lower_unit(
            &parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::RAY_QUERY,
        );
        let codes: Vec<&str> = diags.as_slice().iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E0211"), "expected array ban: {codes:?}");
        assert!(codes.contains(&"E0213"), "expected param ban: {codes:?}");
        assert!(codes.contains(&"E0214"), "expected return ban: {codes:?}");
        assert!(codes.contains(&"E0216"), "expected phi ban: {codes:?}");
        assert!(codes.contains(&"E0217"), "expected assign ban: {codes:?}");
    }

    #[test]
    fn checks_f16_in_structs_and_workgroup_vars() {
        let src = r"
            workgroup var s_half: f16vec4;
            struct HalfData {
                val: f16,
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::all());
        let e0232_count = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0232")
            .count();
        assert_eq!(e0232_count, 2);

        let gated = r"
            #[cap(float16)]
            workgroup var s_half: f16vec4;
            #[cap(float16)]
            struct HalfData {
                val: f16,
            }
        ";
        let parsed_ok = parse_source(FileId(0), gated);
        assert!(!parsed_ok.has_errors());
        let (_, diags_ok) = lower_unit(
            &parsed_ok.unit,
            RenderTier::Tier0Raster,
            HardwareCaps::FLOAT16,
        );
        assert!(!diags_ok.has_errors(), "{:?}", diags_ok.as_slice());
    }

    #[test]
    fn diagnoses_multiple_push_constants_and_bool_in_interface_blocks() {
        let src = r"
            #[push_constant, layout(scalar)]
            struct PushA {
                flag: bool,
            }
            #[push_constant, layout(scalar)]
            struct PushB {
                val: u32,
            }
            #[layout(std140)]
            struct UboBlock {
                enabled: bool,
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        let codes: Vec<&str> = diags.as_slice().iter().map(|d| d.code.as_str()).collect();
        assert!(
            codes.contains(&"E0222"),
            "expected duplicate PC error: {codes:?}"
        );
        assert_eq!(
            codes.iter().filter(|&&c| c == "E0223").count(),
            2,
            "expected two bool-in-block errors: {codes:?}"
        );
    }

    #[test]
    fn supports_forward_struct_references_in_two_pass_lowering() {
        let src = r"
            #[layout(std140)]
            struct Outer {
                a: f32,
                inner: Inner,
            }
            #[layout(std430)]
            struct Inner {
                x: f32,
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        let outer = hir.find_struct("Outer").expect("Outer struct");
        let layout = outer.computed_layout.as_ref().expect("Outer layout");
        assert_eq!(layout.field_offsets[0], (SmolStr::new_static("a"), 0, 4));
        assert_eq!(
            layout.field_offsets[1],
            (SmolStr::new_static("inner"), 16, 16)
        );
        assert_eq!(layout.size, 32);
    }

    #[test]
    fn respects_mc_imports_and_local_shadowing() {
        let src = r"
            import mc::{world};

            @stage(compute(8, 8, 1))
            pub fn shadowed_test(world: f32) {
                let x = world;
                let tlas = 1u;
                let y = tlas;
                let z = bindless;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        assert!(!diags.has_errors());

        let func = hir.find_function("shadowed_test").expect("fn");
        // The scalar `world` parameter shadows the import but is not a resource, so it must
        // not add a descriptor alongside the ABI-mandated WorldUBO.
        assert_eq!(
            func.reflection.descriptors.len(),
            1,
            "only the ABI-mandated WorldUBO may be reflected: {:?}",
            func.reflection.descriptors
        );
        assert_eq!(func.reflection.descriptors[0].name, "world");
        assert_eq!(func.reflection.descriptors[0].set, DESCRIPTOR_SET_PER_PASS);
        assert_eq!(func.reflection.descriptors[0].binding, 0);
    }

    #[test]
    fn rejects_resource_param_named_world() {
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn bad_world_name(world: Texture2D<f32>) {
                let _v = world.sample(vec2(0.0, 0.0));
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());

        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        assert!(
            diags.as_slice().iter().any(|d| d.code == "E0264"),
            "expected E0264, got {:?}",
            diags.as_slice()
        );
        let func = hir.find_function("bad_world_name").expect("fn");
        assert_eq!(
            func.reflection.descriptors.len(),
            1,
            "the reserved name must be skipped, leaving only the WorldUBO: {:?}",
            func.reflection.descriptors
        );
        assert_eq!(func.reflection.descriptors[0].name, "world");
        assert_eq!(func.reflection.descriptors[0].binding, 0);
    }

    #[test]
    fn reflects_pack_schema_params_specs_and_set3_resources() {
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn pass_a(
                #[pack_resource(2)] blue_noise: Texture2D<f32>,
                vol: Texture3D<u32>,
                cube: TextureCube<i32>,
                arr: Texture2DArray<f32>,
                rw3d: RWImage3D<rgba16f>,
            ) {
                let a = mc::param(sun_tint);
                let b = params.sun_intensity;
                let c = pack::params.extra_param;
                let s = spec::pt_max_bounces;
                let r = mc::resources::lut_tex;
            }

            @stage(compute(8, 8, 1))
            pub fn pass_b() {
                let p = mc::params.sun_intensity;
                let s2 = mc::spec::other_spec;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let mut schema = PackSchema::default();
        schema
            .params
            .insert(SmolStr::new_static("sun_intensity"), (0, 4));
        schema
            .params
            .insert(SmolStr::new_static("sun_tint"), (4, 12));
        schema
            .spec_ids
            .insert(SmolStr::new_static("pt_max_bounces"), 7);

        let (hir, diags) = lower_unit_with_schema(
            &parsed.unit,
            RenderTier::Tier3PathTracing,
            HardwareCaps::all(),
            Some(&schema),
        );
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        let fa = hir.find_function("pass_a").expect("pass_a");
        assert_eq!(
            fa.reflection.used_params,
            vec![
                PackParamBinding {
                    name: SmolStr::new_static("sun_tint"),
                    byte_offset: 4,
                    byte_size: 12,
                },
                PackParamBinding {
                    name: SmolStr::new_static("sun_intensity"),
                    byte_offset: 0,
                    byte_size: 4,
                },
                PackParamBinding {
                    name: SmolStr::new_static("extra_param"),
                    byte_offset: 16,
                    byte_size: 4,
                },
            ]
        );
        assert_eq!(
            fa.reflection.used_specs,
            vec![SpecConstantUsage {
                name: SmolStr::new_static("pt_max_bounces"),
                spec_id: 7,
            }]
        );

        // Verify Set 0 texture dimensions/scalars and Set 3 pack resources (slot 0 for lut_tex, slot 2 for blue_noise).
        let set3: Vec<&DescriptorBinding> = fa
            .reflection
            .descriptors
            .iter()
            .filter(|d| d.set == DESCRIPTOR_SET_PACK_RESOURCES)
            .collect();
        assert_eq!(set3.len(), 2);
        assert_eq!(set3[0].binding, 0);
        assert_eq!(set3[0].name, "lut_tex");
        assert_eq!(set3[1].binding, 2);
        assert_eq!(set3[1].name, "blue_noise");

        let vol_desc = fa
            .reflection
            .descriptors
            .iter()
            .find(|d| d.name == "vol")
            .expect("vol");
        assert_eq!(vol_desc.dim, Some(TextureDim::Dim3D));
        assert_eq!(vol_desc.sample_scalar, Some(SampleScalar::U32));

        let cube_desc = fa
            .reflection
            .descriptors
            .iter()
            .find(|d| d.name == "cube")
            .expect("cube");
        assert_eq!(cube_desc.dim, Some(TextureDim::DimCube));
        assert_eq!(cube_desc.sample_scalar, Some(SampleScalar::I32));

        let arr_desc = fa
            .reflection
            .descriptors
            .iter()
            .find(|d| d.name == "arr")
            .expect("arr");
        assert_eq!(arr_desc.dim, Some(TextureDim::Dim2DArray));
        assert_eq!(arr_desc.sample_scalar, Some(SampleScalar::F32));

        let rw3d_desc = fa
            .reflection
            .descriptors
            .iter()
            .find(|d| d.name == "rw3d")
            .expect("rw3d");
        assert_eq!(rw3d_desc.dim, Some(TextureDim::Dim3D));
        assert_eq!(rw3d_desc.sample_scalar, None);

        // Verify pass_b gets non-colliding module-wide fallback spec_id (8, not 0).
        let fb = hir.find_function("pass_b").expect("pass_b");
        assert_eq!(
            fb.reflection.used_specs,
            vec![SpecConstantUsage {
                name: SmolStr::new_static("other_spec"),
                spec_id: 8,
            }]
        );
    }

    #[test]
    fn validates_trait_bounds_and_impl_targets() {
        let src = r"
            struct S<T: MissingTrait> {
                x: f32,
            }
            impl MissingTrait for MissingStruct {
                pub fn foo() {}
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        let codes: Vec<&str> = diags.as_slice().iter().map(|d| d.code.as_str()).collect();
        assert!(
            codes.contains(&"E0271"),
            "expected unknown bound: {codes:?}"
        );
        assert!(
            codes.contains(&"E0272"),
            "expected unknown trait: {codes:?}"
        );
        assert!(
            codes.contains(&"E0273"),
            "expected unknown target: {codes:?}"
        );
    }

    #[test]
    fn enforces_atomic_caps_through_fn_return_types() {
        let src = r"
            struct PtrHelper {
                raw_u64: ptr_mut<u64, align = 8>,
                raw_f32: ptr_mut<f32, align = 4>,
            }

            pub fn bad_atomics(
                base_u64: ptr_mut<u64, align = 8>,
                base_f32: ptr_mut<f32, align = 4>,
                helper: PtrHelper,
                a: u32,
                b: u32,
            ) {
                let _r1 = atomic_add(get_u64_ptr(base_u64), 1u64);
                let _r2 = atomic_cas(get_u64_ptr(base_u64), a, b);
                let _r3 = atomic_add(get_f32_ptr(base_f32), 1.0);
                let _r4 = atomic_add(helper.get_f32_ptr(), a);
                let _r5 = atomic_add(PtrHelper::get_u64_ptr(helper), a);
            }

            pub fn get_u64_ptr(p: ptr_mut<u64, align = 8>) -> ptr_mut<u64, align = 8> {
                return p;
            }

            pub fn get_f32_ptr(p: ptr_mut<f32, align = 4>) -> ptr_mut<f32, align = 4> {
                return p;
            }

            impl PtrHelper {
                pub fn get_u64_ptr(self) -> ptr_mut<u64, align = 8> {
                    return self.raw_u64;
                }

                pub fn get_f32_ptr(self) -> ptr_mut<f32, align = 4> {
                    return self.raw_f32;
                }
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (_, diags) = lower_unit(
            &parsed.unit,
            RenderTier::Tier3PathTracing,
            HardwareCaps::all(),
        );
        let e0237_count = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0237")
            .count();
        let e0238_count = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0238")
            .count();
        assert_eq!(
            e0237_count,
            3,
            "expected 3 E0237 diagnostics: {:?}",
            diags.as_slice()
        );
        assert_eq!(
            e0238_count,
            2,
            "expected 2 E0238 diagnostics: {:?}",
            diags.as_slice()
        );

        let gated_src = r"
            #[cap(int64_atomics, float_atomics)]
            pub fn ok_atomics(
                base_u64: ptr_mut<u64, align = 8>,
                base_f32: ptr_mut<f32, align = 4>,
            ) {
                let _r1 = atomic_add(get_u64_ptr(base_u64), 1u64);
                let _r2 = atomic_add(get_f32_ptr(base_f32), 1.0);
            }

            pub fn get_u64_ptr(p: ptr_mut<u64, align = 8>) -> ptr_mut<u64, align = 8> {
                return p;
            }

            pub fn get_f32_ptr(p: ptr_mut<f32, align = 4>) -> ptr_mut<f32, align = 4> {
                return p;
            }
        ";
        let parsed_ok = parse_source(FileId(0), gated_src);
        assert!(!parsed_ok.has_errors(), "{:?}", parsed_ok.diagnostics);
        let (_, diags_ok) = lower_unit(
            &parsed_ok.unit,
            RenderTier::Tier3PathTracing,
            HardwareCaps::INT64_ATOMICS | HardwareCaps::FLOAT_ATOMICS,
        );
        assert!(!diags_ok.has_errors(), "{:?}", diags_ok.as_slice());
    }

    #[test]
    fn enforces_float16_cap_on_f16_constructor_calls() {
        let src = r"
            pub fn unannotated_f16_ctors() {
                let a = f16(1.0);
                let b = f16vec2(1.0, 2.0);
                let c = f16vec3(1.0, 2.0, 3.0);
                let d = f16vec4(1.0, 2.0, 3.0, 4.0);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::all());
        let e0232_count = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0232")
            .count();
        assert_eq!(
            e0232_count,
            4,
            "expected 4 E0232 diagnostics: {:?}",
            diags.as_slice()
        );

        let func = hir
            .find_function("unannotated_f16_ctors")
            .expect("fn present");
        let a_ty = func.local_types.get("a").copied().expect("local a type");
        assert_eq!(*hir.types.get(a_ty), Type::Scalar(ScalarKind::F16));
        let c_ty = func.local_types.get("c").copied().expect("local c type");
        assert_eq!(
            *hir.types.get(c_ty),
            Type::Vector {
                scalar: ScalarKind::F16,
                len: 3,
            }
        );

        let gated_src = r"
            #[cap(float16)]
            pub fn gated_f16_ctors() {
                let _a = f16(1.0);
                let _b = f16vec3(1.0, 2.0, 3.0);
            }
        ";
        let parsed_ok = parse_source(FileId(0), gated_src);
        assert!(!parsed_ok.has_errors(), "{:?}", parsed_ok.diagnostics);
        let (_, diags_ok) = lower_unit(
            &parsed_ok.unit,
            RenderTier::Tier0Raster,
            HardwareCaps::FLOAT16,
        );
        assert!(!diags_ok.has_errors(), "{:?}", diags_ok.as_slice());
    }

    #[test]
    fn uses_pack_schema_resource_slots_with_deterministic_fallback() {
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn sample_pack_resources() {
                let _a = mc::resources::unlisted_a;
                let _b = mc::resources::blue_noise;
                let _c = mc::resources::brdf_lut;
                let _d = mc::resources::unlisted_b;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let mut schema = PackSchema::default();
        schema
            .resource_slots
            .insert(SmolStr::new_static("blue_noise"), 5);
        schema
            .resource_slots
            .insert(SmolStr::new_static("brdf_lut"), 0);

        let (hir, diags) = lower_unit_with_schema(
            &parsed.unit,
            RenderTier::Tier0Raster,
            HardwareCaps::empty(),
            Some(&schema),
        );
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        let func = hir
            .find_function("sample_pack_resources")
            .expect("fn present");
        let set3: Vec<(u32, &str)> = func
            .reflection
            .descriptors
            .iter()
            .filter(|d| d.set == DESCRIPTOR_SET_PACK_RESOURCES)
            .map(|d| (d.binding, d.name.as_str()))
            .collect();
        assert_eq!(
            set3,
            vec![
                (0, "brdf_lut"),
                (1, "unlisted_a"),
                (2, "unlisted_b"),
                (5, "blue_noise"),
            ]
        );
    }

    #[test]
    fn preserves_generic_struct_params_and_fields_for_reservoir() {
        let src = include_str!("../../../examples/sunlit_vanilla/restir.vsl");
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (hir, diags) = lower_unit(
            &parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::MAXIMAL_RECONVERGENCE,
        );
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        let reservoir = hir.find_struct("Reservoir").expect("Reservoir struct");
        assert_eq!(reservoir.generic_params, vec![SmolStr::new_static("S")]);
        assert!(reservoir.computed_layout.is_none());

        let field_names: Vec<&str> = reservoir.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(field_names, vec!["sample", "w_sum", "m", "w"]);
        assert_eq!(
            *hir.types.get(reservoir.fields[0].ty),
            Type::TypeParam(SmolStr::new_static("S"))
        );
        assert_eq!(
            *hir.types.get(reservoir.fields[1].ty),
            Type::Scalar(ScalarKind::F32)
        );
        assert_eq!(
            *hir.types.get(reservoir.fields[2].ty),
            Type::Scalar(ScalarKind::U32)
        );
        assert_eq!(
            *hir.types.get(reservoir.fields[3].ty),
            Type::Scalar(ScalarKind::F32)
        );
    }

    #[test]
    fn supports_binary_and_octal_literals_in_workgroup_and_array_lengths() {
        let src = r"
            workgroup var scratch: [u32; 0b1000];

            @stage(compute(0b1000, 0o10, 0b1))
            pub fn radix_compute() {
                let _arr: [f32; 0o20];
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        let func = hir.find_function("radix_compute").expect("fn present");
        assert_eq!(func.workgroup_size, Some([8, 8, 1]));

        assert_eq!(hir.workgroup_vars.len(), 1);
        assert!(matches!(
            hir.types.get(hir.workgroup_vars[0].ty),
            Type::Array { len: 8, .. }
        ));

        let arr_ty = func
            .local_types
            .get("_arr")
            .copied()
            .expect("local _arr type");
        assert!(matches!(hir.types.get(arr_ty), Type::Array { len: 16, .. }));
    }

    #[test]
    fn monomorphizes_generic_struct_layout_and_rejects_bool_in_buffer() {
        let src = r"
            struct HitSample {
                pos: vec3,
                dist: f32,
            }

            struct HasBool {
                valid: bool,
                value: f32,
            }

            #[layout(std430)]
            struct Reservoir<S> {
                sample: S,
                w_sum: f32,
                m: u32,
                w: f32,
            }

            @stage(compute(8, 8, 1))
            pub fn resample(
                in_res: Buffer<Reservoir<HitSample>>,
                out_res: RWBuffer<Reservoir<HitSample>>,
                bad_res: Buffer<Reservoir<HasBool>>,
            ) {}
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());

        // Concrete monomorphized `Reservoir<HitSample>` is registered once with exact std430 layout.
        let mono_count = hir
            .structs
            .iter()
            .filter(|(_, s)| s.name == "Reservoir<HitSample>")
            .count();
        assert_eq!(mono_count, 1, "expected monomorphized struct to be reused");

        let mono = hir
            .find_struct("Reservoir<HitSample>")
            .expect("monomorphized Reservoir<HitSample>");
        assert_eq!(mono.generic_params, Vec::<SmolStr>::new());
        let layout = mono
            .computed_layout
            .as_ref()
            .expect("concrete StructLayout");
        assert_eq!(layout.align, 16);
        assert_eq!(layout.size, 32);
        assert_eq!(
            layout.field_offsets[0],
            (SmolStr::new_static("sample"), 0, 16)
        );
        assert_eq!(
            layout.field_offsets[1],
            (SmolStr::new_static("w_sum"), 16, 4)
        );
        assert_eq!(layout.field_offsets[2], (SmolStr::new_static("m"), 20, 4));
        assert_eq!(layout.field_offsets[3], (SmolStr::new_static("w"), 24, 4));

        // `Buffer<Reservoir<HasBool>>` must trigger interface-block `bool` rejection (`E0223`).
        let bool_errors: Vec<_> = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0223")
            .collect();
        assert_eq!(
            bool_errors.len(),
            1,
            "expected E0223 for Buffer<Reservoir<HasBool>>: {:?}",
            diags.as_slice()
        );
    }

    #[test]
    fn type_arena_interns_non_struct_types() {
        let mut arena = TypeArena::new();
        let a = arena.alloc(Type::Scalar(ScalarKind::F32));
        let b = arena.alloc(Type::Scalar(ScalarKind::F32));
        assert_eq!(a, b);
        let va = arena.alloc(Type::Vector {
            scalar: ScalarKind::F32,
            len: 4,
        });
        let vb = arena.alloc(Type::Vector {
            scalar: ScalarKind::F32,
            len: 4,
        });
        assert_eq!(va, vb);
    }

    #[test]
    fn validates_duplicates_recursive_structs_ptr_align_and_workgroup_stage() {
        let src = r"
            struct Dup { a: u32, a: f32 }
            struct Dup { b: u32 }
            struct CycA { b: CycB }
            struct CycB { a: CycA }

            workgroup var s_shared: u32;

            @stage(fragment)
            pub fn bad_fs(p: ptr<u32, align = 12>) {
                let x = s_shared;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (_, diags) = lower_unit(&parsed.unit, RenderTier::Tier2RayQuery, HardwareCaps::all());
        let codes: Vec<&str> = diags.as_slice().iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E0202"), "expected E0202: {codes:?}");
        assert!(codes.contains(&"E0203"), "expected E0203: {codes:?}");
        assert!(codes.contains(&"E0204"), "expected E0204: {codes:?}");
        assert!(codes.contains(&"E0218"), "expected E0218: {codes:?}");
        assert!(codes.contains(&"E0251"), "expected E0251: {codes:?}");
    }
}
