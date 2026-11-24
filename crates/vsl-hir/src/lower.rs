//! AST-to-HIR lowering, early tier/capability pruning (Spec §3.4), compile-time SPIR-V
//! legality validation, and Hyaloclast 4-set Vulkan ABI reflection computation.

use crate::hir::{
    EntryReflection, HirBlock, HirCallArg, HirConst, HirExpr, HirExprKind, HirFnId, HirFunction,
    HirImpl, HirModule, HirParam, HirStmt, HirStruct, HirStructField, HirStructId, HirSwitchCase,
    HirTrait, HirTraitMethod, HirWorkgroupVar, SymbolId, SymbolKind,
};
use crate::ty::{LayoutKind, ScalarKind, Type, TypeArena, TypeId};
use indexmap::{IndexMap, IndexSet};
use la_arena::Arena;
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, DiagnosticBag, Span};
use vsl_mc::{
    BINDLESS_IMAGE_COUNT, BuiltinFieldType, BuiltinStructDesc, DESCRIPTOR_SET_BINDLESS,
    DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS, DESCRIPTOR_SET_SCENE_GLOBALS,
    DebugChannelBinding, DescriptorAccess, DescriptorBinding, DescriptorKind,
    EMISSIVE_LIGHT_LAYOUT, EMISSIVE_LIGHT_SSBO_LAYOUT, HardwareCaps, HookStage,
    INSTANCE_META_LAYOUT, INSTANCE_META_SSBO_LAYOUT, ImageFormat, MAX_PUSH_CONSTANT_BYTES,
    PACK_RESOURCE_SLOTS, PackParamBinding, PackParamType, PushConstantField,
    RESTIR_RESERVOIR_LAYOUT, RESTIR_RESERVOIR_SSBO_LAYOUT, RenderTier, SHARC_CONFIG_UBO_LAYOUT,
    SampleScalar, ShaderStage, SpecConstantDefault, SpecConstantUsage, TextureDim,
    WORLD_UBO_LAYOUT,
};
use vsl_syntax::ast::{
    Attribute, BinaryOp, Block, CallArg, Expr, FnDecl, GenericParam, ImportTree, Item, ParamMode,
    SourceUnit, Stmt, StructDecl, TypeArg, TypeExpr, UnaryOp,
};

/// Maximum Set 0 binding index reserved for per-pass resources (`bindings 1..=14`).
const MAX_PER_PASS_BINDING: u32 = 14;

/// Maximum valid rendering tier index (`Tier3PathTracing = 3`).
const MAX_TIER: u8 = 3;

const FLAG_WORLD: u16 = 1 << 0;
const FLAG_TLAS: u16 = 1 << 1;
const FLAG_INSTANCES: u16 = 1 << 2;
const FLAG_LIGHTS: u16 = 1 << 3;
const FLAG_BINDLESS: u16 = 1 << 4;
const FLAG_BDA: u16 = 1 << 5;
const FLAG_SUBGROUPS: u16 = 1 << 6;
const FLAG_SHARC: u16 = 1 << 7;
const FLAG_RESERVOIRS: u16 = 1 << 8;

/// Pack-wide parameter layout, specialization-constant IDs, and Set 3 resource slot table
/// supplied by `pack.vsl`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PackSchema {
    /// Maps pack parameter name -> `(byte_offset, byte_size)` in `PackParamBuffer`.
    pub params: IndexMap<SmolStr, (u32, u32)>,
    /// Maps pack parameter name -> concrete shader value type.
    pub param_types: IndexMap<SmolStr, PackParamType>,
    /// Maps specialization constant name -> pack-global Vulkan `SpecId`.
    pub spec_ids: IndexMap<SmolStr, u32>,
    /// Maps specialization constant name -> typed specialization default metadata.
    pub spec_defaults: IndexMap<SmolStr, SpecConstantDefault>,
    /// Maps Set 3 pack resource texture name -> pack-global Set 3 binding slot (`0..=15`).
    pub resource_slots: IndexMap<SmolStr, u32>,
}

fn builtin_field_type_to_id(
    field_ty: BuiltinFieldType,
    types: &mut TypeArena,
    builtin_structs: &IndexMap<SmolStr, TypeId>,
) -> TypeId {
    let ty = match field_ty {
        BuiltinFieldType::Bool => Type::Scalar(ScalarKind::Bool),
        BuiltinFieldType::I32 => Type::Scalar(ScalarKind::I32),
        BuiltinFieldType::U32 => Type::Scalar(ScalarKind::U32),
        BuiltinFieldType::I64 => Type::Scalar(ScalarKind::I64),
        BuiltinFieldType::U64 => Type::Scalar(ScalarKind::U64),
        BuiltinFieldType::F32 => Type::Scalar(ScalarKind::F32),
        BuiltinFieldType::Vec2 => Type::Vector {
            scalar: ScalarKind::F32,
            len: 2,
        },
        BuiltinFieldType::Vec3 => Type::Vector {
            scalar: ScalarKind::F32,
            len: 3,
        },
        BuiltinFieldType::Vec4 => Type::Vector {
            scalar: ScalarKind::F32,
            len: 4,
        },
        BuiltinFieldType::UVec2 => Type::Vector {
            scalar: ScalarKind::U32,
            len: 2,
        },
        BuiltinFieldType::UVec3 => Type::Vector {
            scalar: ScalarKind::U32,
            len: 3,
        },
        BuiltinFieldType::UVec4 => Type::Vector {
            scalar: ScalarKind::U32,
            len: 4,
        },
        BuiltinFieldType::Mat4 => Type::Matrix {
            scalar: ScalarKind::F32,
            cols: 4,
            rows: 4,
        },
        BuiltinFieldType::RuntimeArrayStruct(elem_name) => {
            let element = builtin_structs
                .get(elem_name)
                .copied()
                .unwrap_or_else(|| types.alloc(Type::Void));
            Type::Buffer { element }
        }
    };
    types.alloc(ty)
}

fn register_builtin_mc_structs(types: &mut TypeArena) -> IndexMap<SmolStr, TypeId> {
    const LAYOUTS: &[(&BuiltinStructDesc, LayoutKind)] = &[
        (&WORLD_UBO_LAYOUT, LayoutKind::Std140),
        (&INSTANCE_META_LAYOUT, LayoutKind::Std430),
        (&INSTANCE_META_SSBO_LAYOUT, LayoutKind::Std430),
        (&SHARC_CONFIG_UBO_LAYOUT, LayoutKind::Std140),
        (&EMISSIVE_LIGHT_LAYOUT, LayoutKind::Std430),
        (&EMISSIVE_LIGHT_SSBO_LAYOUT, LayoutKind::Std430),
        (&RESTIR_RESERVOIR_LAYOUT, LayoutKind::Std430),
        (&RESTIR_RESERVOIR_SSBO_LAYOUT, LayoutKind::Std430),
    ];

    let mut map = IndexMap::with_capacity(LAYOUTS.len());
    for &(desc, layout_kind) in LAYOUTS {
        let mut fields = Vec::with_capacity(desc.fields.len());
        for field in desc.fields {
            let field_ty = builtin_field_type_to_id(field.ty, types, &map);
            fields.push((SmolStr::new_static(field.name), field_ty));
        }
        let ty_id = types.alloc(Type::Struct {
            name: SmolStr::new_static(desc.name),
            layout: layout_kind,
            fields,
        });
        map.insert(SmolStr::new_static(desc.name), ty_id);
    }
    map
}

fn pack_param_type_to_id(param_ty: PackParamType, types: &mut TypeArena) -> TypeId {
    let ty = match param_ty {
        PackParamType::Bool => Type::Scalar(ScalarKind::Bool),
        PackParamType::I32 | PackParamType::Enum => Type::Scalar(ScalarKind::I32),
        PackParamType::U32 => Type::Scalar(ScalarKind::U32),
        PackParamType::F32 => Type::Scalar(ScalarKind::F32),
        PackParamType::Vec3 => Type::Vector {
            scalar: ScalarKind::F32,
            len: 3,
        },
    };
    types.alloc(ty)
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
    let builtin_struct_types = register_builtin_mc_structs(&mut module.types);
    let mut struct_generic_args: IndexMap<SmolStr, (SmolStr, Vec<TypeId>)> = IndexMap::new();
    let empty_subst: IndexMap<SmolStr, TypeId> = IndexMap::new();

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

            let mut type_ctx = TypeLowerCtx::types_only(
                &mut module.types,
                &mut struct_types,
                &builtin_struct_types,
            );
            if let Some(ty_id) =
                lower_type_expr(&field.ty, &mut type_ctx, &generic_names, &empty_subst)
            {
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
                &builtin_struct_types,
                &mut struct_generic_args,
            );
            if let Some(ty_id) =
                lower_type_expr(&field.ty, &mut type_ctx, &generic_names, &empty_subst)
            {
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
    let empty_bounds: IndexMap<SmolStr, SmolStr> = IndexMap::new();
    let mut const_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
    let mut const_symbols: IndexMap<SmolStr, SymbolId> = IndexMap::new();
    let mut workgroup_var_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
    let mut workgroup_var_symbols: IndexMap<SmolStr, SymbolId> = IndexMap::new();

    for item in &unit.items {
        match item {
            Item::Const(decl) => {
                let mut type_ctx = TypeLowerCtx::full(
                    &mut module.types,
                    &mut module.structs,
                    &mut struct_types,
                    &mut struct_ids,
                    &builtin_struct_types,
                    &mut struct_generic_args,
                );
                if let Some(ty_id) =
                    lower_type_expr(&decl.ty, &mut type_ctx, &empty_generics, &empty_subst)
                {
                    let sym_id = module.alloc_symbol(
                        decl.name.name.clone(),
                        SymbolKind::Const,
                        ty_id,
                        false,
                        decl.span,
                    );
                    const_types.insert(decl.name.name.clone(), ty_id);
                    const_symbols.insert(decl.name.name.clone(), sym_id);
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
                        &builtin_struct_types,
                        &mut struct_generic_args,
                    );
                    lower_type_expr(&var.ty, &mut type_ctx, &empty_generics, &empty_subst)
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

                let sym_id = module.alloc_symbol(
                    var.name.name.clone(),
                    SymbolKind::WorkgroupVar,
                    ty,
                    true,
                    var.span,
                );
                workgroup_var_types.insert(var.name.name.clone(), ty);
                workgroup_var_symbols.insert(var.name.name.clone(), sym_id);
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

    // Pass 4.5: Pre-collect declared signatures, return types, and generic templates for all
    // module functions, traits, and `impl` blocks so call-site typechecking and monomorphization
    // resolve regardless of declaration order.
    let mut fn_return_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
    let mut fn_sigs: IndexMap<SmolStr, FnSig> = IndexMap::new();
    let mut trait_method_sigs: IndexMap<SmolStr, IndexMap<SmolStr, FnSig>> = IndexMap::new();
    let mut generic_fn_templates: IndexMap<SmolStr, GenericFnTemplate> = IndexMap::new();
    let mut generic_impl_templates: Vec<GenericImplTemplate> = Vec::new();

    for item in &unit.items {
        match item {
            Item::Trait(decl) => {
                let mut methods_map = IndexMap::new();
                for method in &decl.methods {
                    let method_generics: IndexSet<SmolStr> = method
                        .generics
                        .iter()
                        .map(|g| g.name.name.clone())
                        .collect();
                    let mut sig_params = Vec::with_capacity(method.params.len());
                    for param in &method.params {
                        let ty = param.ty.as_ref().and_then(|ty_expr| {
                            let mut type_ctx = TypeLowerCtx::full(
                                &mut module.types,
                                &mut module.structs,
                                &mut struct_types,
                                &mut struct_ids,
                                &builtin_struct_types,
                                &mut struct_generic_args,
                            );
                            lower_type_expr(ty_expr, &mut type_ctx, &method_generics, &empty_subst)
                        });
                        sig_params.push((param.name.name.clone(), param.mode, ty));
                    }
                    let ret_ty = method.return_ty.as_ref().and_then(|ty_expr| {
                        let mut type_ctx = TypeLowerCtx::full(
                            &mut module.types,
                            &mut module.structs,
                            &mut struct_types,
                            &mut struct_ids,
                            &builtin_struct_types,
                            &mut struct_generic_args,
                        );
                        lower_type_expr(ty_expr, &mut type_ctx, &method_generics, &empty_subst)
                    });
                    methods_map.insert(
                        method.name.name.clone(),
                        FnSig {
                            params: sig_params,
                            return_ty: ret_ty,
                            fn_id: None,
                        },
                    );
                }
                trait_method_sigs.insert(decl.name.name.clone(), methods_map);
            }
            Item::Function(func) => {
                let fn_generics: IndexSet<SmolStr> =
                    func.generics.iter().map(|g| g.name.name.clone()).collect();
                if !func.generics.is_empty() {
                    let gp = func
                        .generics
                        .iter()
                        .map(|g| {
                            (
                                g.name.name.clone(),
                                g.bound.as_ref().map(|b| SmolStr::new(b.joined())),
                            )
                        })
                        .collect();
                    generic_fn_templates.insert(
                        func.name.name.clone(),
                        GenericFnTemplate {
                            decl: func.clone(),
                            generic_params: gp,
                        },
                    );
                }
                let mut sig_params = Vec::with_capacity(func.params.len());
                for param in &func.params {
                    let ty = param.ty.as_ref().and_then(|ty_expr| {
                        let mut type_ctx = TypeLowerCtx::full(
                            &mut module.types,
                            &mut module.structs,
                            &mut struct_types,
                            &mut struct_ids,
                            &builtin_struct_types,
                            &mut struct_generic_args,
                        );
                        lower_type_expr(ty_expr, &mut type_ctx, &fn_generics, &empty_subst)
                    });
                    sig_params.push((param.name.name.clone(), param.mode, ty));
                }
                let ret_ty = func.return_ty.as_ref().and_then(|ty_expr| {
                    let mut type_ctx = TypeLowerCtx::full(
                        &mut module.types,
                        &mut module.structs,
                        &mut struct_types,
                        &mut struct_ids,
                        &builtin_struct_types,
                        &mut struct_generic_args,
                    );
                    lower_type_expr(ty_expr, &mut type_ctx, &fn_generics, &empty_subst)
                });
                if let Some(ty_id) = ret_ty {
                    fn_return_types.insert(func.name.name.clone(), ty_id);
                }
                fn_sigs.insert(
                    func.name.name.clone(),
                    FnSig {
                        params: sig_params,
                        return_ty: ret_ty,
                        fn_id: None,
                    },
                );
            }
            Item::Impl(imp) => {
                let target_name = match &imp.target {
                    TypeExpr::Named { path, .. } => Some(path.joined()),
                    TypeExpr::Array { .. } => None,
                };
                if !imp.generics.is_empty()
                    && let Some(ref target) = target_name
                {
                    let gp = imp
                        .generics
                        .iter()
                        .map(|g| {
                            (
                                g.name.name.clone(),
                                g.bound.as_ref().map(|b| SmolStr::new(b.joined())),
                            )
                        })
                        .collect();
                    let mut methods_map = IndexMap::new();
                    for method in &imp.methods {
                        methods_map.insert(method.name.name.clone(), method.clone());
                    }
                    generic_impl_templates.push(GenericImplTemplate {
                        generic_params: gp,
                        target_struct: SmolStr::new(target),
                        methods: methods_map,
                    });
                }
                let impl_self_ty = target_name
                    .as_deref()
                    .and_then(|name| struct_types.get(name).copied())
                    .or_else(|| {
                        let mut type_ctx = TypeLowerCtx::full(
                            &mut module.types,
                            &mut module.structs,
                            &mut struct_types,
                            &mut struct_ids,
                            &builtin_struct_types,
                            &mut struct_generic_args,
                        );
                        lower_type_expr(&imp.target, &mut type_ctx, &empty_generics, &empty_subst)
                    });
                for method in &imp.methods {
                    let mut method_generics: IndexSet<SmolStr> =
                        imp.generics.iter().map(|g| g.name.name.clone()).collect();
                    method_generics.extend(method.generics.iter().map(|g| g.name.name.clone()));

                    let mut sig_params = Vec::with_capacity(method.params.len());
                    for param in &method.params {
                        let ty = if param.ty.is_none() && param.name.name == "self" {
                            impl_self_ty
                        } else {
                            param.ty.as_ref().and_then(|ty_expr| {
                                if is_self_type_expr(ty_expr) {
                                    impl_self_ty
                                } else {
                                    let mut type_ctx = TypeLowerCtx::full(
                                        &mut module.types,
                                        &mut module.structs,
                                        &mut struct_types,
                                        &mut struct_ids,
                                        &builtin_struct_types,
                                        &mut struct_generic_args,
                                    );
                                    lower_type_expr(
                                        ty_expr,
                                        &mut type_ctx,
                                        &method_generics,
                                        &empty_subst,
                                    )
                                }
                            })
                        };
                        sig_params.push((param.name.name.clone(), param.mode, ty));
                    }

                    let resolved_ret = method.return_ty.as_ref().and_then(|ret_ty| {
                        if is_self_type_expr(ret_ty) {
                            impl_self_ty
                        } else {
                            let mut type_ctx = TypeLowerCtx::full(
                                &mut module.types,
                                &mut module.structs,
                                &mut struct_types,
                                &mut struct_ids,
                                &builtin_struct_types,
                                &mut struct_generic_args,
                            );
                            lower_type_expr(ret_ty, &mut type_ctx, &method_generics, &empty_subst)
                        }
                    });
                    if let Some(ref target) = target_name {
                        let qualified = SmolStr::from(format!("{target}::{}", method.name.name));
                        if let Some(ty_id) = resolved_ret {
                            fn_return_types.insert(qualified.clone(), ty_id);
                        }
                        fn_sigs.insert(
                            qualified,
                            FnSig {
                                params: sig_params,
                                return_ty: resolved_ret,
                                fn_id: None,
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }

    // Pass 5: Validate traits, lower const initializers, lower impl blocks (concrete trait impls
    // first so monomorphized generic methods can resolve trait method targets regardless of order),
    // and lower top-level functions.
    let mut ctx = LowerCtx {
        module: &mut module,
        diags: &mut diags,
        struct_types: &mut struct_types,
        struct_ids: &mut struct_ids,
        builtin_struct_types: &builtin_struct_types,
        struct_generic_args: &mut struct_generic_args,
        trait_names: &trait_names,
        trait_method_sigs: &trait_method_sigs,
        const_types: &const_types,
        const_symbols: &const_symbols,
        workgroup_var_types: &workgroup_var_types,
        workgroup_var_symbols: &workgroup_var_symbols,
        fn_return_types: &mut fn_return_types,
        fn_sigs: &mut fn_sigs,
        fn_symbols: IndexMap::new(),
        mc_descriptor_symbols: IndexMap::new(),
        trait_impls: IndexMap::new(),
        inherent_impls: IndexMap::new(),
        generic_fn_templates: &generic_fn_templates,
        generic_impl_templates: &generic_impl_templates,
        monomorphized_fns: IndexMap::new(),
        monomorphizing_in_progress: IndexSet::new(),
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

    // 5a: Traits and module constants.
    for item in &unit.items {
        match item {
            Item::Trait(decl) => {
                let mut hir_methods = Vec::with_capacity(decl.methods.len());
                for method in &decl.methods {
                    ctx.validate_trait_method(method);
                    hir_methods.push(ctx.lower_trait_method(method));
                }
                ctx.module.traits.push(HirTrait {
                    name: decl.name.name.clone(),
                    methods: hir_methods,
                    span: decl.span,
                });
            }
            Item::Const(decl) => {
                let declared_ty = ctx
                    .const_types
                    .get(&decl.name.name)
                    .copied()
                    .unwrap_or_else(|| ctx.module.types.alloc(Type::Void));
                let mut dummy_locals = IndexMap::new();
                let mut dummy_scan = BodyScan::default();
                let value_expr = {
                    let mut scanner = BodyScanner {
                        ctx: &mut ctx,
                        generic_params: &empty_generics,
                        generic_bounds: &empty_bounds,
                        type_subst: &empty_subst,
                        expected_return_ty: None,
                        is_compute: false,
                        is_non_compute_stage: false,
                        caps: HardwareCaps::empty(),
                        scopes: vec![IndexMap::new()],
                        local_types: &mut dummy_locals,
                        scan: &mut dummy_scan,
                    };
                    scanner.lower_expr(&decl.value, false).0
                };
                ctx.module.consts.push(HirConst {
                    name: decl.name.name.clone(),
                    ty: declared_ty,
                    value_expr,
                    is_pub: decl.is_pub,
                    span: decl.span,
                });
            }
            _ => {}
        }
    }

    // 5b: Concrete `impl` blocks (trait impls first, then inherent impls) so trait method lookup
    // tables are populated before generic impls and functions trigger monomorphization.
    for pass in 0..3 {
        for item in &unit.items {
            if let Item::Impl(imp) = item {
                let is_concrete_trait = imp.generics.is_empty() && imp.trait_path.is_some();
                let is_concrete_inherent = imp.generics.is_empty() && imp.trait_path.is_none();
                let is_generic = !imp.generics.is_empty();
                let run_now = match pass {
                    0 => is_concrete_trait,
                    1 => is_concrete_inherent,
                    _ => is_generic,
                };
                if !run_now {
                    continue;
                }

                validate_generic_bounds(&imp.generics, ctx.trait_names, ctx.diags);
                let trait_name = imp.trait_path.as_ref().map(|trait_path| {
                    let name = trait_path.joined();
                    if !ctx.trait_names.contains(name.as_str()) {
                        ctx.diags.push(Diagnostic::error(
                            "E0272",
                            format!("unknown trait `{name}` in `impl` block"),
                            trait_path.span,
                        ));
                    }
                    SmolStr::new(name)
                });

                let impl_generics: IndexSet<SmolStr> =
                    imp.generics.iter().map(|g| g.name.name.clone()).collect();
                let impl_bounds: IndexMap<SmolStr, SmolStr> = imp
                    .generics
                    .iter()
                    .filter_map(|g| {
                        g.bound
                            .as_ref()
                            .map(|b| (g.name.name.clone(), SmolStr::new(b.joined())))
                    })
                    .collect();
                let self_ty = ctx.validate_impl_target(&imp.target, imp.span);
                let target_struct = match &imp.target {
                    TypeExpr::Named { path, .. } => SmolStr::new(path.joined()),
                    TypeExpr::Array { .. } => SmolStr::new_static("array"),
                };
                let mut method_ids = Vec::with_capacity(imp.methods.len());
                for method in &imp.methods {
                    if let Some(fn_id) = ctx.lower_function_inner(
                        method,
                        self_ty,
                        &impl_generics,
                        &impl_bounds,
                        &empty_subst,
                        None,
                    ) {
                        method_ids.push(fn_id);
                        let qualified =
                            SmolStr::from(format!("{target_struct}::{}", method.name.name));
                        if let Some(sig) = ctx.fn_sigs.get_mut(&qualified) {
                            sig.fn_id = Some(fn_id);
                        }
                        if imp.generics.is_empty() {
                            if let Some(ref tr) = trait_name {
                                ctx.trait_impls
                                    .entry((tr.clone(), target_struct.clone()))
                                    .or_default()
                                    .insert(method.name.name.clone(), fn_id);
                            } else {
                                ctx.inherent_impls
                                    .entry(target_struct.clone())
                                    .or_default()
                                    .insert(method.name.name.clone(), fn_id);
                            }
                        }
                    }
                }
                let hir_generics = imp
                    .generics
                    .iter()
                    .map(|g| {
                        (
                            g.name.name.clone(),
                            g.bound.as_ref().map(|b| SmolStr::new(b.joined())),
                        )
                    })
                    .collect();
                ctx.module.impls.push(HirImpl {
                    generic_params: hir_generics,
                    trait_name,
                    target_struct,
                    methods: method_ids,
                    span: imp.span,
                });
            }
        }
    }

    // 5c: Top-level functions.
    for item in &unit.items {
        if let Item::Function(func) = item
            && let Some(fn_id) = ctx.lower_function_inner(
                func,
                None,
                &empty_generics,
                &empty_bounds,
                &empty_subst,
                None,
            )
            && let Some(sig) = ctx.fn_sigs.get_mut(&func.name.name)
        {
            sig.fn_id = Some(fn_id);
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

#[derive(Clone)]
struct FnSig {
    params: Vec<(SmolStr, ParamMode, Option<TypeId>)>,
    return_ty: Option<TypeId>,
    fn_id: Option<HirFnId>,
}

#[derive(Clone)]
struct GenericFnTemplate {
    decl: FnDecl,
    generic_params: Vec<(SmolStr, Option<SmolStr>)>,
}

#[derive(Clone)]
struct GenericImplTemplate {
    generic_params: Vec<(SmolStr, Option<SmolStr>)>,
    target_struct: SmolStr,
    methods: IndexMap<SmolStr, FnDecl>,
}

struct LowerCtx<'a> {
    module: &'a mut HirModule,
    diags: &'a mut DiagnosticBag,
    struct_types: &'a mut IndexMap<SmolStr, TypeId>,
    struct_ids: &'a mut IndexMap<SmolStr, HirStructId>,
    builtin_struct_types: &'a IndexMap<SmolStr, TypeId>,
    struct_generic_args: &'a mut IndexMap<SmolStr, (SmolStr, Vec<TypeId>)>,
    trait_names: &'a IndexSet<SmolStr>,
    trait_method_sigs: &'a IndexMap<SmolStr, IndexMap<SmolStr, FnSig>>,
    const_types: &'a IndexMap<SmolStr, TypeId>,
    const_symbols: &'a IndexMap<SmolStr, SymbolId>,
    workgroup_var_types: &'a IndexMap<SmolStr, TypeId>,
    workgroup_var_symbols: &'a IndexMap<SmolStr, SymbolId>,
    fn_return_types: &'a mut IndexMap<SmolStr, TypeId>,
    fn_sigs: &'a mut IndexMap<SmolStr, FnSig>,
    fn_symbols: IndexMap<SmolStr, SymbolId>,
    mc_descriptor_symbols: IndexMap<SmolStr, SymbolId>,
    trait_impls: IndexMap<(SmolStr, SmolStr), IndexMap<SmolStr, HirFnId>>,
    inherent_impls: IndexMap<SmolStr, IndexMap<SmolStr, HirFnId>>,
    generic_fn_templates: &'a IndexMap<SmolStr, GenericFnTemplate>,
    generic_impl_templates: &'a [GenericImplTemplate],
    monomorphized_fns: IndexMap<SmolStr, HirFnId>,
    monomorphizing_in_progress: IndexSet<SmolStr>,
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
            self.builtin_struct_types,
            self.struct_generic_args,
        )
    }

    fn get_or_alloc_mc_symbol(
        &mut self,
        name: &str,
        ty: TypeId,
        is_mut: bool,
        span: Span,
    ) -> SymbolId {
        if let Some(&id) = self.mc_descriptor_symbols.get(name) {
            return id;
        }
        let id =
            self.module
                .alloc_symbol(SmolStr::new(name), SymbolKind::Descriptor, ty, is_mut, span);
        self.mc_descriptor_symbols.insert(SmolStr::new(name), id);
        id
    }

    fn type_implements_trait(&self, ty_id: TypeId, trait_name: &str) -> bool {
        let Type::Struct { name, .. } = self.module.types.get(ty_id) else {
            return false;
        };
        self.trait_impls
            .contains_key(&(SmolStr::new(trait_name), name.clone()))
    }

    fn find_concrete_method(&self, struct_name: &str, method_name: &str) -> Option<HirFnId> {
        if let Some(methods) = self.inherent_impls.get(struct_name)
            && let Some(&fn_id) = methods.get(method_name)
        {
            return Some(fn_id);
        }
        for ((_, target), methods) in &self.trait_impls {
            if target == struct_name
                && let Some(&fn_id) = methods.get(method_name)
            {
                return Some(fn_id);
            }
        }
        None
    }

    /// Instantiates a generic function template with concrete type arguments, checking trait
    /// bounds and lowering the specialized body with substituted `TypeId`s.
    fn monomorphize_function(
        &mut self,
        fn_name: &str,
        type_args: &[TypeId],
        call_span: Span,
    ) -> Option<HirFnId> {
        let template = self.generic_fn_templates.get(fn_name)?.clone();
        if template.generic_params.len() != type_args.len() {
            return None;
        }
        if type_args
            .iter()
            .any(|&arg_ty| contains_type_param(&self.module.types, arg_ty))
        {
            return None;
        }

        for ((param_name, bound_opt), &arg_ty) in template.generic_params.iter().zip(type_args) {
            if let Some(bound_trait) = bound_opt
                && !self.type_implements_trait(arg_ty, bound_trait.as_str())
            {
                let arg_name = format_type_name(&self.module.types, arg_ty);
                self.diags.push(Diagnostic::error(
                    "E0274",
                    format!(
                        "type `{arg_name}` does not implement trait `{bound_trait}` required by generic parameter `{param_name}` of `{fn_name}`"
                    ),
                    call_span,
                ));
            }
        }

        let arg_names: Vec<String> = type_args
            .iter()
            .map(|&ty| format_type_name(&self.module.types, ty))
            .collect();
        let mangled = SmolStr::from(format!("{fn_name}<{}>", arg_names.join(", ")));

        if let Some(&existing) = self.monomorphized_fns.get(&mangled) {
            return Some(existing);
        }
        if !self.monomorphizing_in_progress.insert(mangled.clone()) {
            return None;
        }

        let mut subst = IndexMap::with_capacity(template.generic_params.len());
        for ((param_name, _), &arg_ty) in template.generic_params.iter().zip(type_args) {
            subst.insert(param_name.clone(), arg_ty);
        }

        let empty_generics = IndexSet::new();
        let empty_bounds = IndexMap::new();
        let fn_id = self.lower_function_inner(
            &template.decl,
            None,
            &empty_generics,
            &empty_bounds,
            &subst,
            Some(mangled.clone()),
        );
        self.monomorphizing_in_progress.swap_remove(&mangled);

        if let Some(id) = fn_id {
            let sig = {
                let func = &self.module.functions[id];
                FnSig {
                    params: func
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.mode, p.ty))
                        .collect(),
                    return_ty: func.return_ty,
                    fn_id: Some(id),
                }
            };
            if let Some(ret) = sig.return_ty {
                self.fn_return_types.insert(mangled.clone(), ret);
            }
            self.fn_sigs.insert(mangled.clone(), sig);
            self.monomorphized_fns.insert(mangled, id);
        }
        fn_id
    }

    /// Instantiates a generic `impl<T: Trait> Struct<T>` method for a concrete monomorphized
    /// struct type (such as `Reservoir<MySample>`), resolving trait method calls inside the body
    /// to concrete `impl Trait for ConcreteType` methods.
    fn monomorphize_impl_method(
        &mut self,
        concrete_self_ty: TypeId,
        method_name: &str,
        call_span: Span,
    ) -> Option<HirFnId> {
        let Type::Struct { name, .. } = self.module.types.get(concrete_self_ty) else {
            return None;
        };
        let concrete_struct_name = name.clone();
        let (base_struct_name, type_args) = self
            .struct_generic_args
            .get(&concrete_struct_name)
            .cloned()?;

        let template = self
            .generic_impl_templates
            .iter()
            .find(|t| t.target_struct == base_struct_name && t.methods.contains_key(method_name))?
            .clone();
        if template.generic_params.len() != type_args.len() {
            return None;
        }
        let method_decl = template.methods.get(method_name)?.clone();

        for ((param_name, bound_opt), &arg_ty) in template.generic_params.iter().zip(&type_args) {
            if let Some(bound_trait) = bound_opt
                && !self.type_implements_trait(arg_ty, bound_trait.as_str())
            {
                let arg_name = format_type_name(&self.module.types, arg_ty);
                self.diags.push(Diagnostic::error(
                    "E0274",
                    format!(
                        "type `{arg_name}` does not implement trait `{bound_trait}` required by generic parameter `{param_name}` of `{base_struct_name}`"
                    ),
                    call_span,
                ));
            }
        }

        let mangled = SmolStr::from(format!("{concrete_struct_name}::{method_name}"));
        if let Some(&existing) = self.monomorphized_fns.get(&mangled) {
            return Some(existing);
        }
        if !self.monomorphizing_in_progress.insert(mangled.clone()) {
            return None;
        }

        let mut subst = IndexMap::with_capacity(template.generic_params.len());
        for ((param_name, _), &arg_ty) in template.generic_params.iter().zip(&type_args) {
            subst.insert(param_name.clone(), arg_ty);
        }

        let empty_generics = IndexSet::new();
        let empty_bounds = IndexMap::new();
        let fn_id = self.lower_function_inner(
            &method_decl,
            Some(concrete_self_ty),
            &empty_generics,
            &empty_bounds,
            &subst,
            Some(mangled.clone()),
        );
        self.monomorphizing_in_progress.swap_remove(&mangled);

        if let Some(id) = fn_id {
            let sig = {
                let func = &self.module.functions[id];
                FnSig {
                    params: func
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.mode, p.ty))
                        .collect(),
                    return_ty: func.return_ty,
                    fn_id: Some(id),
                }
            };
            if let Some(ret) = sig.return_ty {
                self.fn_return_types.insert(mangled.clone(), ret);
            }
            self.fn_sigs.insert(mangled.clone(), sig);
            self.inherent_impls
                .entry(concrete_struct_name)
                .or_default()
                .insert(SmolStr::new(method_name), id);
            self.monomorphized_fns.insert(mangled, id);
        }
        fn_id
    }

    fn validate_impl_target(&mut self, target: &TypeExpr, span: Span) -> Option<TypeId> {
        let empty_generics = IndexSet::new();
        let empty_subst = IndexMap::new();
        match target {
            TypeExpr::Named { path, .. } => {
                let name = path.joined();
                if let Some(&id) = self.struct_types.get(name.as_str()) {
                    return Some(id);
                }
                let mut type_ctx = self.type_ctx();
                if let Some(id) =
                    lower_type_expr(target, &mut type_ctx, &empty_generics, &empty_subst)
                {
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
                lower_type_expr(target, &mut type_ctx, &empty_generics, &empty_subst)
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

    fn lower_trait_method(&mut self, method: &FnDecl) -> HirTraitMethod {
        let method_generics: IndexSet<SmolStr> = method
            .generics
            .iter()
            .map(|g| g.name.name.clone())
            .collect();
        let empty_subst = IndexMap::new();
        let mut params = Vec::with_capacity(method.params.len());
        for param in &method.params {
            let ty = param.ty.as_ref().and_then(|ty_expr| {
                let mut type_ctx = self.type_ctx();
                lower_type_expr(ty_expr, &mut type_ctx, &method_generics, &empty_subst)
            });
            params.push(HirParam {
                name: param.name.name.clone(),
                mode: param.mode,
                ty,
                builtin: None,
                is_push_constant: false,
                pack_resource_slot: None,
                span: param.span,
            });
        }
        let return_ty = method.return_ty.as_ref().and_then(|ty_expr| {
            let mut type_ctx = self.type_ctx();
            lower_type_expr(ty_expr, &mut type_ctx, &method_generics, &empty_subst)
        });
        HirTraitMethod {
            name: method.name.name.clone(),
            params,
            return_ty,
            span: method.span,
        }
    }

    fn lower_function_inner(
        &mut self,
        func: &FnDecl,
        self_ty: Option<TypeId>,
        impl_generics: &IndexSet<SmolStr>,
        impl_bounds: &IndexMap<SmolStr, SmolStr>,
        type_subst: &IndexMap<SmolStr, TypeId>,
        override_name: Option<SmolStr>,
    ) -> Option<HirFnId> {
        let tier_range = parse_tier_range(&func.attrs, self.diags);

        // Spec §3.4: Early Tier Pruning strips excluded functions before cap/reflection emission.
        if let Some(active_tier) = self.active_tier {
            let active = active_tier.as_u8();
            if !(tier_range.0..=tier_range.1).contains(&active) {
                self.module.pruned_functions += 1;
                return None;
            }
        }

        validate_generic_bounds(&func.generics, self.trait_names, self.diags);

        let mut fn_generics = impl_generics.clone();
        let mut fn_bounds = impl_bounds.clone();
        if type_subst.is_empty() {
            for g in &func.generics {
                fn_generics.insert(g.name.name.clone());
                if let Some(ref bound) = g.bound {
                    fn_bounds.insert(g.name.name.clone(), SmolStr::new(bound.joined()));
                }
            }
        } else {
            for g in &func.generics {
                if !type_subst.contains_key(&g.name.name) {
                    fn_generics.insert(g.name.name.clone());
                    if let Some(ref bound) = g.bound {
                        fn_bounds.insert(g.name.name.clone(), SmolStr::new(bound.joined()));
                    }
                }
            }
        }

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
        let mut root_scope: IndexMap<SmolStr, ScopedVar> = IndexMap::new();

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
                        lower_type_expr(ty_expr, &mut type_ctx, &fn_generics, type_subst)
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

            let is_mut = matches!(
                param.mode,
                ParamMode::Mut | ParamMode::InOut | ParamMode::Out
            );
            let sym_ty = resolved_ty.unwrap_or_else(|| self.module.types.alloc(Type::Void));
            let sym_id = self.module.alloc_symbol(
                param.name.name.clone(),
                SymbolKind::Param,
                sym_ty,
                is_mut,
                param.span,
            );
            root_scope.insert(
                param.name.name.clone(),
                ScopedVar {
                    ty: resolved_ty,
                    is_mut,
                    symbol: sym_id,
                },
            );
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
                lower_type_expr(ty_expr, &mut type_ctx, &fn_generics, type_subst)
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

        // Lower function body into typed `HirBlock` while scanning for SPIR-V legality,
        // local types, mc::* global references, debug channels, BDA, subgroups, params, specs,
        // and Set 3 pack resources.
        let mut scan = BodyScan::default();
        let mut local_types: IndexMap<SmolStr, TypeId> = IndexMap::new();
        let body_hir = func.body.as_ref().map(|body| {
            let is_compute = matches!(parsed_attrs.stage, Some(ShaderStage::Compute));
            let is_non_compute_stage = matches!(
                parsed_attrs.stage,
                Some(ShaderStage::Vertex | ShaderStage::Fragment)
            );
            let mut scanner = BodyScanner {
                ctx: self,
                generic_params: &fn_generics,
                generic_bounds: &fn_bounds,
                type_subst,
                expected_return_ty: return_ty,
                is_compute,
                is_non_compute_stage,
                caps: parsed_attrs.required_caps,
                scopes: vec![root_scope],
                local_types: &mut local_types,
                scan: &mut scan,
            };
            scanner.lower_block(body).0
        });
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

        let fn_name = override_name.unwrap_or_else(|| func.name.name.clone());
        let hir_generic_params = if type_subst.is_empty() {
            func.generics
                .iter()
                .map(|g| {
                    (
                        g.name.name.clone(),
                        g.bound.as_ref().map(|b| SmolStr::new(b.joined())),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        let fn_ret_sym_ty = return_ty.unwrap_or_else(|| self.module.types.alloc(Type::Void));
        let fn_sym_id = self.module.alloc_symbol(
            fn_name.clone(),
            SymbolKind::Function,
            fn_ret_sym_ty,
            false,
            func.span,
        );
        self.fn_symbols.insert(fn_name.clone(), fn_sym_id);

        let fn_id = self.module.functions.alloc(HirFunction {
            name: fn_name,
            generic_params: hir_generic_params,
            stage: parsed_attrs.stage,
            workgroup_size: parsed_attrs.workgroup_size,
            hook: parsed_attrs.hook,
            tier_range,
            required_caps: parsed_attrs.required_caps,
            params,
            return_ty,
            local_types,
            body_hir,
            body_ast: func.body.clone(),
            is_pub: func.is_pub,
            reflection,
            span: func.span,
        });
        Some(fn_id)
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

        // Set 1: Scene globals (0 = tlas, 1 = instances, 2 = sharc, 3 = lights, 4 = reservoirs).
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
        if scan.has(FLAG_SHARC) {
            descriptors.push(DescriptorBinding {
                name: SmolStr::new_static("sharc"),
                set: DESCRIPTOR_SET_SCENE_GLOBALS,
                binding: 2,
                kind: DescriptorKind::UniformBuffer,
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
        if scan.has(FLAG_RESERVOIRS) {
            descriptors.push(DescriptorBinding {
                name: SmolStr::new_static("reservoirs"),
                set: DESCRIPTOR_SET_SCENE_GLOBALS,
                binding: 4,
                kind: DescriptorKind::StorageBuffer,
                dim: None,
                sample_scalar: None,
                image_format: None,
                access: DescriptorAccess::ReadWrite,
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
                    for name in [
                        "world",
                        "tlas",
                        "instances",
                        "sharc",
                        "lights",
                        "reservoirs",
                        "bindless",
                    ] {
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
    flags: u16,
    debug_channels: IndexSet<SmolStr>,
    used_specs: IndexSet<SmolStr>,
    used_params: IndexSet<SmolStr>,
    pack_resources: IndexSet<SmolStr>,
}

impl BodyScan {
    const fn has(&self, flag: u16) -> bool {
        (self.flags & flag) != 0
    }

    fn set(&mut self, flag: u16) {
        self.flags |= flag;
    }
}

#[derive(Clone, Copy)]
struct ScopedVar {
    ty: Option<TypeId>,
    is_mut: bool,
    symbol: SymbolId,
}

enum SwizzleError {
    InvalidLength,
    InvalidComponent,
    MixedSets,
    OutOfRange { component: char, vec_len: u8 },
    DuplicateWriteComponent(char),
}

fn parse_vector_swizzle(
    swizzle: &str,
    vec_len: u8,
    is_write: bool,
) -> Result<Vec<u8>, SwizzleError> {
    if swizzle.is_empty() || swizzle.len() > 4 {
        return Err(SwizzleError::InvalidLength);
    }
    let mut family: Option<u8> = None;
    let mut components = Vec::with_capacity(swizzle.len());
    let mut seen_mask = 0u8;

    for ch in swizzle.chars() {
        let (ch_family, idx) = match ch {
            'x' => (0u8, 0u8),
            'y' => (0, 1),
            'z' => (0, 2),
            'w' => (0, 3),
            'r' => (1, 0),
            'g' => (1, 1),
            'b' => (1, 2),
            'a' => (1, 3),
            's' => (2, 0),
            't' => (2, 1),
            'p' => (2, 2),
            'q' => (2, 3),
            _ => return Err(SwizzleError::InvalidComponent),
        };
        if let Some(expected_family) = family {
            if ch_family != expected_family {
                return Err(SwizzleError::MixedSets);
            }
        } else {
            family = Some(ch_family);
        }
        if idx >= vec_len {
            return Err(SwizzleError::OutOfRange {
                component: ch,
                vec_len,
            });
        }
        if is_write {
            let bit = 1u8 << idx;
            if (seen_mask & bit) != 0 {
                return Err(SwizzleError::DuplicateWriteComponent(ch));
            }
            seen_mask |= bit;
        }
        components.push(idx);
    }

    Ok(components)
}

const fn is_numeric_scalar(s: ScalarKind) -> bool {
    matches!(
        s,
        ScalarKind::I32
            | ScalarKind::U32
            | ScalarKind::I64
            | ScalarKind::U64
            | ScalarKind::F32
            | ScalarKind::F16
    )
}

const fn is_integer_scalar(s: ScalarKind) -> bool {
    matches!(
        s,
        ScalarKind::I32 | ScalarKind::U32 | ScalarKind::I64 | ScalarKind::U64
    )
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

const fn compound_base_op(op: BinaryOp) -> Option<BinaryOp> {
    match op {
        BinaryOp::AddAssign => Some(BinaryOp::Add),
        BinaryOp::SubAssign => Some(BinaryOp::Sub),
        BinaryOp::MulAssign => Some(BinaryOp::Mul),
        BinaryOp::DivAssign => Some(BinaryOp::Div),
        BinaryOp::RemAssign => Some(BinaryOp::Rem),
        BinaryOp::BitAndAssign => Some(BinaryOp::BitAnd),
        BinaryOp::BitOrAssign => Some(BinaryOp::BitOr),
        BinaryOp::BitXorAssign => Some(BinaryOp::BitXor),
        _ => None,
    }
}

fn unify_type_params(
    types: &TypeArena,
    pattern: TypeId,
    concrete: TypeId,
    out: &mut IndexMap<SmolStr, TypeId>,
) {
    match (types.get(pattern), types.get(concrete)) {
        (Type::TypeParam(name), _) => {
            if !matches!(types.get(concrete), Type::Void | Type::TypeParam(_)) {
                out.entry(name.clone()).or_insert(concrete);
            }
        }
        (Type::Array { element: pe, .. }, Type::Array { element: ce, .. })
        | (Type::Buffer { element: pe }, Type::Buffer { element: ce })
        | (Type::RwBuffer { element: pe }, Type::RwBuffer { element: ce })
        | (Type::Pointer { pointee: pe, .. }, Type::Pointer { pointee: ce, .. }) => {
            unify_type_params(types, *pe, *ce, out);
        }
        (Type::Struct { fields: pf, .. }, Type::Struct { fields: cf, .. })
            if pf.len() == cf.len() =>
        {
            for ((_, pt), (_, ct)) in pf.iter().zip(cf) {
                unify_type_params(types, *pt, *ct, out);
            }
        }
        _ => {}
    }
}

struct BodyScanner<'a, 'b> {
    ctx: &'b mut LowerCtx<'a>,
    generic_params: &'b IndexSet<SmolStr>,
    generic_bounds: &'b IndexMap<SmolStr, SmolStr>,
    type_subst: &'b IndexMap<SmolStr, TypeId>,
    expected_return_ty: Option<TypeId>,
    is_compute: bool,
    is_non_compute_stage: bool,
    caps: HardwareCaps,
    scopes: Vec<IndexMap<SmolStr, ScopedVar>>,
    local_types: &'b mut IndexMap<SmolStr, TypeId>,
    scan: &'b mut BodyScan,
}

impl BodyScanner<'_, '_> {
    fn void_ty(&mut self) -> TypeId {
        self.ctx.module.types.alloc(Type::Void)
    }

    fn is_concrete_value_ty(&self, ty_id: TypeId) -> bool {
        !matches!(
            self.ctx.module.types.get(ty_id),
            Type::Void | Type::TypeParam(_)
        )
    }

    fn is_shadowed(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|s| s.contains_key(name))
    }

    fn lookup_scoped(&self, name: &str) -> Option<ScopedVar> {
        for scope in self.scopes.iter().rev() {
            if let Some(&var) = scope.get(name) {
                return Some(var);
            }
        }
        None
    }

    fn lookup_var(&self, name: &str) -> (Option<TypeId>, Option<SymbolId>) {
        if let Some(var) = self.lookup_scoped(name) {
            return (var.ty, Some(var.symbol));
        }
        if let Some(&ty) = self.ctx.workgroup_var_types.get(name) {
            return (Some(ty), self.ctx.workgroup_var_symbols.get(name).copied());
        }
        if let Some(&ty) = self.ctx.const_types.get(name) {
            return (Some(ty), self.ctx.const_symbols.get(name).copied());
        }
        (None, self.ctx.fn_symbols.get(name).copied())
    }

    fn insert_local(
        &mut self,
        name: SmolStr,
        ty: Option<TypeId>,
        is_mut: bool,
        span: Span,
    ) -> SymbolId {
        let sym_ty = ty.unwrap_or_else(|| self.void_ty());
        let sym_id =
            self.ctx
                .module
                .alloc_symbol(name.clone(), SymbolKind::Local, sym_ty, is_mut, span);
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(
                name.clone(),
                ScopedVar {
                    ty,
                    is_mut,
                    symbol: sym_id,
                },
            );
        }
        if let Some(ty_id) = ty {
            self.local_types.insert(name, ty_id);
        }
        sym_id
    }

    fn is_lvalue_mutable(&self, expr: &HirExpr) -> bool {
        match &expr.kind {
            HirExprKind::Path { path, .. } => {
                if let Some(var) = self.lookup_scoped(path.as_str()) {
                    return var.is_mut;
                }
                if self.ctx.workgroup_var_types.contains_key(path.as_str()) {
                    return true;
                }
                if self.ctx.const_types.contains_key(path.as_str()) {
                    return false;
                }
                if path == "mc::reservoirs"
                    || (path == "reservoirs" && self.ctx.mc_imports.contains("reservoirs"))
                {
                    return true;
                }
                if path.starts_with("mc::") {
                    return false;
                }
                // Unknown/unresolved identifier: avoid false-positive mutability errors.
                true
            }
            HirExprKind::Field { receiver, .. } => match self.ctx.module.types.get(receiver.ty) {
                Type::Pointer { mutable, .. } => *mutable,
                _ => self.is_lvalue_mutable(receiver),
            },
            HirExprKind::Swizzle { base, .. } => self.is_lvalue_mutable(base),
            HirExprKind::Index { receiver, .. } => match self.ctx.module.types.get(receiver.ty) {
                Type::RwBuffer { .. } => true,
                Type::Buffer { .. } => false,
                Type::Pointer { mutable, .. } => *mutable,
                _ => self.is_lvalue_mutable(receiver),
            },
            _ => false,
        }
    }

    fn check_binary_types(
        &mut self,
        op: BinaryOp,
        lhs_ty: Option<TypeId>,
        rhs_ty: Option<TypeId>,
        span: Span,
    ) -> Option<TypeId> {
        let bool_ty = self.ctx.module.types.alloc(Type::Scalar(ScalarKind::Bool));

        if matches!(op, BinaryOp::Assign) {
            if let (Some(lt), Some(rt)) = (lhs_ty, rhs_ty)
                && self.is_concrete_value_ty(lt)
                && self.is_concrete_value_ty(rt)
                && lt != rt
            {
                let lhs_name = format_type_name(&self.ctx.module.types, lt);
                let rhs_name = format_type_name(&self.ctx.module.types, rt);
                self.ctx.diags.push(Diagnostic::error(
                    "E0270",
                    format!("cannot assign value of type `{rhs_name}` to `{lhs_name}`"),
                    span,
                ));
            }
            return None;
        }

        if let Some(base_op) = compound_base_op(op) {
            let res = self.check_binary_types(base_op, lhs_ty, rhs_ty, span);
            if let (Some(lt), Some(res_ty)) = (lhs_ty, res)
                && self.is_concrete_value_ty(lt)
                && self.is_concrete_value_ty(res_ty)
                && lt != res_ty
            {
                let lhs_name = format_type_name(&self.ctx.module.types, lt);
                let res_name = format_type_name(&self.ctx.module.types, res_ty);
                self.ctx.diags.push(Diagnostic::error(
                    "E0270",
                    format!(
                        "compound assignment result `{res_name}` is incompatible with `{lhs_name}`"
                    ),
                    span,
                ));
            }
            return None;
        }

        let (Some(lt), Some(rt)) = (lhs_ty, rhs_ty) else {
            return match op {
                BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::And
                | BinaryOp::Or => Some(bool_ty),
                _ => lhs_ty.or(rhs_ty),
            };
        };

        let lhs_t = self.ctx.module.types.get(lt).clone();
        let rhs_t = self.ctx.module.types.get(rt).clone();

        if matches!(lhs_t, Type::Void | Type::TypeParam(_))
            || matches!(rhs_t, Type::Void | Type::TypeParam(_))
        {
            return match op {
                BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::And
                | BinaryOp::Or => Some(bool_ty),
                _ => Some(lt),
            };
        }

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                match (&lhs_t, &rhs_t) {
                    (Type::Scalar(a), Type::Scalar(b)) if a == b && is_numeric_scalar(*a) => {
                        Some(lt)
                    }
                    (
                        Type::Vector {
                            scalar: sa,
                            len: la,
                        },
                        Type::Vector {
                            scalar: sb,
                            len: lb,
                        },
                    ) if sa == sb && la == lb && is_numeric_scalar(*sa) => Some(lt),
                    (Type::Vector { scalar: sa, .. }, Type::Scalar(sb))
                        if sa == sb && is_numeric_scalar(*sa) =>
                    {
                        Some(lt)
                    }
                    (Type::Scalar(sa), Type::Vector { scalar: sb, .. })
                        if sa == sb && is_numeric_scalar(*sa) =>
                    {
                        Some(rt)
                    }
                    (
                        Type::Matrix {
                            scalar: sa,
                            cols: ca,
                            rows: ra,
                        },
                        Type::Matrix {
                            scalar: sb,
                            cols: cb,
                            rows: rb,
                        },
                    ) if sa == sb && is_numeric_scalar(*sa) => {
                        if matches!(op, BinaryOp::Mul) && ca == rb {
                            Some(self.ctx.module.types.alloc(Type::Matrix {
                                scalar: *sa,
                                cols: *cb,
                                rows: *ra,
                            }))
                        } else if matches!(op, BinaryOp::Add | BinaryOp::Sub)
                            && ca == cb
                            && ra == rb
                        {
                            Some(lt)
                        } else {
                            let ln = format_type_name(&self.ctx.module.types, lt);
                            let rn = format_type_name(&self.ctx.module.types, rt);
                            self.ctx.diags.push(Diagnostic::error(
                                "E0270",
                                format!("incompatible matrix operands `{ln}` and `{rn}`"),
                                span,
                            ));
                            Some(lt)
                        }
                    }
                    (
                        Type::Matrix {
                            scalar: sa,
                            cols,
                            rows,
                        },
                        Type::Vector { scalar: sb, len },
                    ) if matches!(op, BinaryOp::Mul)
                        && sa == sb
                        && cols == len
                        && is_numeric_scalar(*sa) =>
                    {
                        Some(self.ctx.module.types.alloc(Type::Vector {
                            scalar: *sa,
                            len: *rows,
                        }))
                    }
                    (
                        Type::Vector { scalar: sa, len },
                        Type::Matrix {
                            scalar: sb,
                            cols,
                            rows,
                        },
                    ) if matches!(op, BinaryOp::Mul)
                        && sa == sb
                        && len == rows
                        && is_numeric_scalar(*sa) =>
                    {
                        Some(self.ctx.module.types.alloc(Type::Vector {
                            scalar: *sa,
                            len: *cols,
                        }))
                    }
                    (Type::Matrix { scalar: sa, .. }, Type::Scalar(sb))
                        if matches!(op, BinaryOp::Mul | BinaryOp::Div)
                            && sa == sb
                            && is_numeric_scalar(*sa) =>
                    {
                        Some(lt)
                    }
                    (Type::Scalar(sa), Type::Matrix { scalar: sb, .. })
                        if matches!(op, BinaryOp::Mul) && sa == sb && is_numeric_scalar(*sa) =>
                    {
                        Some(rt)
                    }
                    _ => {
                        let ln = format_type_name(&self.ctx.module.types, lt);
                        let rn = format_type_name(&self.ctx.module.types, rt);
                        self.ctx.diags.push(Diagnostic::error(
                            "E0270",
                            format!("incompatible operand types `{ln}` and `{rn}` for arithmetic operator `{op:?}`"),
                            span,
                        ));
                        Some(lt)
                    }
                }
            }
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => match (&lhs_t, &rhs_t) {
                (Type::Scalar(a), Type::Scalar(b)) if a == b && is_integer_scalar(*a) => Some(lt),
                (
                    Type::Vector {
                        scalar: sa,
                        len: la,
                    },
                    Type::Vector {
                        scalar: sb,
                        len: lb,
                    },
                ) if sa == sb && la == lb && is_integer_scalar(*sa) => Some(lt),
                (Type::Vector { scalar: sa, .. }, Type::Scalar(sb))
                    if sa == sb && is_integer_scalar(*sa) =>
                {
                    Some(lt)
                }
                (Type::Scalar(sa), Type::Vector { scalar: sb, .. })
                    if sa == sb && is_integer_scalar(*sa) =>
                {
                    Some(rt)
                }
                _ => {
                    let ln = format_type_name(&self.ctx.module.types, lt);
                    let rn = format_type_name(&self.ctx.module.types, rt);
                    self.ctx.diags.push(Diagnostic::error(
                        "E0270",
                        format!("incompatible operand types `{ln}` and `{rn}` for bitwise operator `{op:?}`"),
                        span,
                    ));
                    Some(lt)
                }
            },
            BinaryOp::Shl | BinaryOp::Shr => match (&lhs_t, &rhs_t) {
                (Type::Scalar(a), Type::Scalar(b))
                    if is_integer_scalar(*a) && is_integer_scalar(*b) =>
                {
                    Some(lt)
                }
                (
                    Type::Vector {
                        scalar: sa,
                        len: la,
                    },
                    Type::Vector {
                        scalar: sb,
                        len: lb,
                    },
                ) if is_integer_scalar(*sa) && is_integer_scalar(*sb) && la == lb => Some(lt),
                (Type::Vector { scalar: sa, .. }, Type::Scalar(sb))
                    if is_integer_scalar(*sa) && is_integer_scalar(*sb) =>
                {
                    Some(lt)
                }
                _ => {
                    let ln = format_type_name(&self.ctx.module.types, lt);
                    let rn = format_type_name(&self.ctx.module.types, rt);
                    self.ctx.diags.push(Diagnostic::error(
                        "E0270",
                        format!("incompatible operand types `{ln}` and `{rn}` for shift operator `{op:?}`"),
                        span,
                    ));
                    Some(lt)
                }
            },
            BinaryOp::And | BinaryOp::Or => {
                if lhs_t != Type::Scalar(ScalarKind::Bool)
                    || rhs_t != Type::Scalar(ScalarKind::Bool)
                {
                    let ln = format_type_name(&self.ctx.module.types, lt);
                    let rn = format_type_name(&self.ctx.module.types, rt);
                    self.ctx.diags.push(Diagnostic::error(
                        "E0270",
                        format!("logical operator `{op:?}` requires `bool` operands, found `{ln}` and `{rn}`"),
                        span,
                    ));
                }
                Some(bool_ty)
            }
            BinaryOp::Eq | BinaryOp::Ne => {
                if lt != rt {
                    let ln = format_type_name(&self.ctx.module.types, lt);
                    let rn = format_type_name(&self.ctx.module.types, rt);
                    self.ctx.diags.push(Diagnostic::error(
                        "E0270",
                        format!("cannot compare `{ln}` with `{rn}`"),
                        span,
                    ));
                }
                Some(bool_ty)
            }
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => match (&lhs_t, &rhs_t) {
                (Type::Scalar(a), Type::Scalar(b)) if a == b && is_numeric_scalar(*a) => {
                    Some(bool_ty)
                }
                (
                    Type::Vector {
                        scalar: sa,
                        len: la,
                    },
                    Type::Vector {
                        scalar: sb,
                        len: lb,
                    },
                ) if sa == sb && la == lb && is_numeric_scalar(*sa) => {
                    Some(self.ctx.module.types.alloc(Type::Vector {
                        scalar: ScalarKind::Bool,
                        len: *la,
                    }))
                }
                _ => {
                    let ln = format_type_name(&self.ctx.module.types, lt);
                    let rn = format_type_name(&self.ctx.module.types, rt);
                    self.ctx.diags.push(Diagnostic::error(
                        "E0270",
                        format!("cannot order-compare `{ln}` and `{rn}`"),
                        span,
                    ));
                    Some(bool_ty)
                }
            },
            BinaryOp::Range | BinaryOp::RangeInclusive => match (&lhs_t, &rhs_t) {
                (Type::Scalar(a), Type::Scalar(b)) if a == b && is_integer_scalar(*a) => Some(lt),
                _ => {
                    let ln = format_type_name(&self.ctx.module.types, lt);
                    let rn = format_type_name(&self.ctx.module.types, rt);
                    self.ctx.diags.push(Diagnostic::error(
                        "E0270",
                        format!(
                            "range bounds must have matching integer types, found `{ln}` and `{rn}`"
                        ),
                        span,
                    ));
                    Some(lt)
                }
            },
            BinaryOp::Assign
            | BinaryOp::AddAssign
            | BinaryOp::SubAssign
            | BinaryOp::MulAssign
            | BinaryOp::DivAssign
            | BinaryOp::RemAssign
            | BinaryOp::BitAndAssign
            | BinaryOp::BitOrAssign
            | BinaryOp::BitXorAssign => None,
        }
    }

    fn lower_block(&mut self, block: &Block) -> (HirBlock, Option<TypeId>) {
        self.scopes.push(IndexMap::new());
        let mut stmts = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            stmts.push(self.lower_stmt(stmt));
        }
        let (tail_expr, tail_ty) = if let Some(tail) = block.tail_expr.as_deref() {
            let (hir_tail, ty_opt) = self.lower_expr(tail, false);
            (Some(Box::new(hir_tail)), ty_opt)
        } else {
            (None, None)
        };
        self.scopes.pop();
        let ty = tail_ty.unwrap_or_else(|| self.void_ty());
        (
            HirBlock {
                stmts,
                tail_expr,
                ty,
                span: block.span,
            },
            tail_ty,
        )
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> HirStmt {
        match stmt {
            Stmt::Let {
                is_mut,
                name,
                ty,
                init,
                span,
            } => {
                let mut resolved_ty = None;
                let mut flagged_logical_ptr = false;

                if let Some(ty_expr) = ty {
                    check_type_caps_and_legality(ty_expr, self.caps, *span, self.ctx.diags);
                    if is_logical_pointer_type(ty_expr) {
                        self.ctx.diags.push(
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
                    let mut type_ctx = self.ctx.type_ctx();
                    resolved_ty = lower_type_expr(
                        ty_expr,
                        &mut type_ctx,
                        self.generic_params,
                        self.type_subst,
                    );
                    if let Some(ty_id) = resolved_ty {
                        self.ctx.check_buffer_struct_bool(ty_id, *span);
                    }
                    if let Some(ty_id) = resolved_ty
                        && self.ctx.module.types.requires_float16(ty_id)
                        && !self.caps.contains(HardwareCaps::FLOAT16)
                        && !type_expr_uses_f16(ty_expr)
                    {
                        self.ctx.diags.push(Diagnostic::error(
                            "E0232",
                            "`f16` types require `#[cap(float16)]` on the enclosing function",
                            *span,
                        ));
                    }
                }

                let mut lowered_init = None;
                let mut init_ty = None;
                if let Some(expr) = init {
                    if !flagged_logical_ptr && expr_yields_addr_of(expr) {
                        self.ctx.diags.push(
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
                    let (hir_init, inferred) = self.lower_expr(expr, false);
                    if let Some(ty_id) = inferred
                        && self.ctx.module.types.contains_ray_query(ty_id)
                    {
                        self.ctx.diags.push(Diagnostic::error(
                            "E0217",
                            "`RayQuery` cannot be assigned or copied into another variable",
                            *span,
                        ));
                    }
                    if let (Some(ann_ty), Some(inf_ty)) = (resolved_ty, inferred)
                        && self.is_concrete_value_ty(ann_ty)
                        && self.is_concrete_value_ty(inf_ty)
                        && ann_ty != inf_ty
                    {
                        let ann_name = format_type_name(&self.ctx.module.types, ann_ty);
                        let inf_name = format_type_name(&self.ctx.module.types, inf_ty);
                        self.ctx.diags.push(Diagnostic::error(
                            "E0270",
                            format!(
                                "mismatched initializer type for `{}`: expected `{ann_name}`, found `{inf_name}`",
                                name.name
                            ),
                            *span,
                        ));
                    }
                    init_ty = inferred;
                    lowered_init = Some(hir_init);
                }

                let final_ty_opt = resolved_ty.or(init_ty);
                let sym_id = self.insert_local(name.name.clone(), final_ty_opt, *is_mut, *span);
                let final_ty = final_ty_opt.unwrap_or_else(|| self.void_ty());
                HirStmt::Let {
                    symbol: sym_id,
                    is_mut: *is_mut,
                    name: name.name.clone(),
                    ty: final_ty,
                    init: lowered_init,
                    span: *span,
                }
            }
            Stmt::Return { value, span } => {
                let (hir_val, val_ty) = if let Some(expr) = value {
                    let (e, ty) = self.lower_expr(expr, false);
                    if let Some(ty_id) = ty
                        && self.ctx.module.types.contains_ray_query(ty_id)
                    {
                        self.ctx.diags.push(Diagnostic::error(
                            "E0214",
                            "`RayQuery` cannot be returned from a function",
                            *span,
                        ));
                    }
                    (Some(e), ty)
                } else {
                    (None, None)
                };

                match (self.expected_return_ty, val_ty) {
                    (Some(exp_ty), Some(actual_ty))
                        if self.is_concrete_value_ty(exp_ty)
                            && self.is_concrete_value_ty(actual_ty)
                            && exp_ty != actual_ty =>
                    {
                        let exp_name = format_type_name(&self.ctx.module.types, exp_ty);
                        let act_name = format_type_name(&self.ctx.module.types, actual_ty);
                        self.ctx.diags.push(Diagnostic::error(
                            "E0275",
                            format!(
                                "return type mismatch: expected `{exp_name}`, found `{act_name}`"
                            ),
                            *span,
                        ));
                    }
                    (Some(exp_ty), None)
                        if value.is_none() && self.is_concrete_value_ty(exp_ty) =>
                    {
                        let exp_name = format_type_name(&self.ctx.module.types, exp_ty);
                        self.ctx.diags.push(Diagnostic::error(
                            "E0275",
                            format!("missing return value; expected `{exp_name}`"),
                            *span,
                        ));
                    }
                    (None, Some(actual_ty)) if self.is_concrete_value_ty(actual_ty) => {
                        let act_name = format_type_name(&self.ctx.module.types, actual_ty);
                        self.ctx.diags.push(Diagnostic::error(
                            "E0275",
                            format!(
                                "cannot return a value of type `{act_name}` from a void function"
                            ),
                            *span,
                        ));
                    }
                    _ => {}
                }

                HirStmt::Return {
                    value: hir_val,
                    span: *span,
                }
            }
            Stmt::For {
                binding,
                iter,
                body,
                span,
                ..
            } => {
                let (hir_iter, iter_ty) = self.lower_expr(iter, false);
                self.scopes.push(IndexMap::new());
                let binding_symbol =
                    self.insert_local(binding.name.clone(), iter_ty, false, binding.span);
                let (hir_body, _) = self.lower_block(body);
                self.scopes.pop();
                let binding_ty = iter_ty.unwrap_or_else(|| self.void_ty());
                HirStmt::For {
                    binding_symbol,
                    binding: binding.name.clone(),
                    binding_ty,
                    iter: hir_iter,
                    body: hir_body,
                    span: *span,
                }
            }
            Stmt::While {
                cond, body, span, ..
            } => {
                let (hir_cond, _) = self.lower_expr(cond, false);
                let (hir_body, _) = self.lower_block(body);
                HirStmt::While {
                    cond: hir_cond,
                    body: hir_body,
                    span: *span,
                }
            }
            Stmt::Loop { body, span, .. } => {
                let (hir_body, _) = self.lower_block(body);
                HirStmt::Loop {
                    body: hir_body,
                    span: *span,
                }
            }
            Stmt::Switch {
                selector,
                cases,
                default,
                span,
            } => {
                let (hir_sel, _) = self.lower_expr(selector, false);
                let mut hir_cases = Vec::with_capacity(cases.len());
                for case in cases {
                    let (val, _) = self.lower_expr(&case.value, false);
                    let (b, _) = self.lower_block(&case.body);
                    hir_cases.push(HirSwitchCase {
                        value: val,
                        body: b,
                        span: case.span,
                    });
                }
                let hir_default = default.as_ref().map(|d| self.lower_block(d).0);
                HirStmt::Switch {
                    selector: hir_sel,
                    cases: hir_cases,
                    default: hir_default,
                    span: *span,
                }
            }
            Stmt::Expr {
                expr,
                has_semi,
                span,
            } => {
                let (hir_expr, _) = self.lower_expr(expr, false);
                HirStmt::Expr {
                    expr: hir_expr,
                    has_semi: *has_semi,
                    span: *span,
                }
            }
            Stmt::Break(span) => HirStmt::Break(*span),
            Stmt::Continue(span) => HirStmt::Continue(*span),
        }
    }

    fn lower_call_args(&mut self, args: &[CallArg]) -> (Vec<HirCallArg>, Vec<Option<TypeId>>) {
        let mut hir_args = Vec::with_capacity(args.len());
        let mut arg_types = Vec::with_capacity(args.len());
        for arg in args {
            let (val, ty) = self.lower_expr(&arg.value, false);
            arg_types.push(ty);
            hir_args.push(HirCallArg {
                name: arg.name.as_ref().map(|n| n.name.clone()),
                value: val,
                span: arg.span,
            });
        }
        (hir_args, arg_types)
    }

    fn check_call_against_sig(
        &mut self,
        callee_label: &str,
        sig_params: &[(SmolStr, ParamMode, Option<TypeId>)],
        hir_args: &[HirCallArg],
        arg_types: &[Option<TypeId>],
        call_span: Span,
    ) {
        if hir_args.len() != sig_params.len() {
            self.ctx.diags.push(Diagnostic::error(
                "E0274",
                format!(
                    "`{callee_label}` expects {} argument(s), but {} were supplied",
                    sig_params.len(),
                    hir_args.len()
                ),
                call_span,
            ));
        }

        for ((param_name, mode, expected_ty_opt), (hir_arg, actual_ty_opt)) in
            sig_params.iter().zip(hir_args.iter().zip(arg_types.iter()))
        {
            if matches!(mode, ParamMode::InOut | ParamMode::Out)
                && !self.is_lvalue_mutable(&hir_arg.value)
            {
                let mode_str = if matches!(mode, ParamMode::InOut) {
                    "inout"
                } else {
                    "out"
                };
                self.ctx.diags.push(Diagnostic::error(
                    "E0272",
                    format!(
                        "`{mode_str}` parameter `{param_name}` of `{callee_label}` requires a mutable lvalue argument"
                    ),
                    hir_arg.span,
                ));
            }
            if let (Some(exp_ty), Some(act_ty)) = (*expected_ty_opt, *actual_ty_opt)
                && self.is_concrete_value_ty(exp_ty)
                && self.is_concrete_value_ty(act_ty)
                && exp_ty != act_ty
            {
                let exp_name = format_type_name(&self.ctx.module.types, exp_ty);
                let act_name = format_type_name(&self.ctx.module.types, act_ty);
                self.ctx.diags.push(Diagnostic::error(
                    "E0274",
                    format!(
                        "argument type mismatch for parameter `{param_name}` of `{callee_label}`: expected `{exp_name}`, found `{act_name}`"
                    ),
                    hir_arg.span,
                ));
            }
        }
    }

    fn lower_expr(&mut self, expr: &Expr, is_write_target: bool) -> (HirExpr, Option<TypeId>) {
        match expr {
            Expr::IntLit(raw, span) => {
                let scalar = if raw.ends_with("u64") {
                    ScalarKind::U64
                } else if raw.ends_with("i64") {
                    ScalarKind::I64
                } else if raw.ends_with("u32") || raw.ends_with('u') {
                    ScalarKind::U32
                } else {
                    ScalarKind::I32
                };
                let ty = self.ctx.module.types.alloc(Type::Scalar(scalar));
                (
                    HirExpr {
                        kind: HirExprKind::IntLit(raw.clone()),
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    Some(ty),
                )
            }
            Expr::FloatLit(raw, span) => {
                let scalar = if raw.ends_with("f16") {
                    if !self.caps.contains(HardwareCaps::FLOAT16) {
                        self.ctx.diags.push(Diagnostic::error(
                            "E0232",
                            "`f16` literals require `#[cap(float16)]` on the enclosing function",
                            *span,
                        ));
                    }
                    ScalarKind::F16
                } else {
                    ScalarKind::F32
                };
                let ty = self.ctx.module.types.alloc(Type::Scalar(scalar));
                (
                    HirExpr {
                        kind: HirExprKind::FloatLit(raw.clone()),
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    Some(ty),
                )
            }
            Expr::BoolLit(val, span) => {
                let ty = self.ctx.module.types.alloc(Type::Scalar(ScalarKind::Bool));
                (
                    HirExpr {
                        kind: HirExprKind::BoolLit(*val),
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    Some(ty),
                )
            }
            Expr::StringLit(val, span) => {
                let ty = self.void_ty();
                (
                    HirExpr {
                        kind: HirExprKind::StringLit(val.clone()),
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    None,
                )
            }
            Expr::NoneLit(span) => {
                let ty = self.void_ty();
                (
                    HirExpr {
                        kind: HirExprKind::NoneLit,
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    None,
                )
            }
            Expr::Path {
                path,
                turbofish,
                span,
            } => {
                let joined = path.joined();
                let mut resolved_ty: Option<TypeId> = None;
                let mut resolved_sym: Option<SymbolId> = None;

                match joined.as_str() {
                    "mc::world" => {
                        self.scan.set(FLAG_WORLD);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("WorldUBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::world",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "world"
                        if self.ctx.mc_imports.contains("world") && !self.is_shadowed("world") =>
                    {
                        self.scan.set(FLAG_WORLD);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("WorldUBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::world",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "mc::tlas" => {
                        self.scan.set(FLAG_TLAS);
                        let ty = self.ctx.module.types.alloc(Type::AccelStruct);
                        resolved_ty = Some(ty);
                        resolved_sym = Some(
                            self.ctx
                                .get_or_alloc_mc_symbol("mc::tlas", ty, false, *span),
                        );
                    }
                    "tlas" if self.ctx.mc_imports.contains("tlas") && !self.is_shadowed("tlas") => {
                        self.scan.set(FLAG_TLAS);
                        let ty = self.ctx.module.types.alloc(Type::AccelStruct);
                        resolved_ty = Some(ty);
                        resolved_sym = Some(
                            self.ctx
                                .get_or_alloc_mc_symbol("mc::tlas", ty, false, *span),
                        );
                    }
                    "mc::instances" => {
                        self.scan.set(FLAG_INSTANCES);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("InstanceMetaSSBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::instances",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "instances"
                        if self.ctx.mc_imports.contains("instances")
                            && !self.is_shadowed("instances") =>
                    {
                        self.scan.set(FLAG_INSTANCES);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("InstanceMetaSSBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::instances",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "mc::sharc" => {
                        self.scan.set(FLAG_SHARC);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("SharcConfigUBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::sharc",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "sharc"
                        if self.ctx.mc_imports.contains("sharc") && !self.is_shadowed("sharc") =>
                    {
                        self.scan.set(FLAG_SHARC);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("SharcConfigUBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::sharc",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "mc::lights" => {
                        self.scan.set(FLAG_LIGHTS);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("EmissiveLightSSBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::lights",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "lights"
                        if self.ctx.mc_imports.contains("lights")
                            && !self.is_shadowed("lights") =>
                    {
                        self.scan.set(FLAG_LIGHTS);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("EmissiveLightSSBO") {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::lights",
                                ty,
                                false,
                                *span,
                            ));
                        }
                    }
                    "mc::reservoirs" => {
                        self.scan.set(FLAG_RESERVOIRS);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("RestirReservoirSSBO")
                        {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::reservoirs",
                                ty,
                                true,
                                *span,
                            ));
                        }
                    }
                    "reservoirs"
                        if self.ctx.mc_imports.contains("reservoirs")
                            && !self.is_shadowed("reservoirs") =>
                    {
                        self.scan.set(FLAG_RESERVOIRS);
                        if let Some(&ty) = self.ctx.builtin_struct_types.get("RestirReservoirSSBO")
                        {
                            resolved_ty = Some(ty);
                            resolved_sym = Some(self.ctx.get_or_alloc_mc_symbol(
                                "mc::reservoirs",
                                ty,
                                true,
                                *span,
                            ));
                        }
                    }
                    "mc::bindless" => {
                        self.scan.set(FLAG_BINDLESS);
                        let ty = self.ctx.module.types.alloc(Type::Texture2D {
                            sample_scalar: ScalarKind::F32,
                            unsized_array: true,
                        });
                        resolved_ty = Some(ty);
                        resolved_sym =
                            Some(
                                self.ctx
                                    .get_or_alloc_mc_symbol("mc::bindless", ty, false, *span),
                            );
                    }
                    "bindless"
                        if self.ctx.mc_imports.contains("bindless")
                            && !self.is_shadowed("bindless") =>
                    {
                        self.scan.set(FLAG_BINDLESS);
                        let ty = self.ctx.module.types.alloc(Type::Texture2D {
                            sample_scalar: ScalarKind::F32,
                            unsized_array: true,
                        });
                        resolved_ty = Some(ty);
                        resolved_sym =
                            Some(
                                self.ctx
                                    .get_or_alloc_mc_symbol("mc::bindless", ty, false, *span),
                            );
                    }
                    _ => {}
                }

                if let Some(res_name) = joined.strip_prefix("mc::resources::")
                    && !res_name.is_empty()
                {
                    self.scan.pack_resources.insert(SmolStr::new(res_name));
                    resolved_ty = Some(self.ctx.module.types.alloc(Type::Texture2D {
                        sample_scalar: ScalarKind::F32,
                        unsized_array: false,
                    }));
                }

                if let Some(spec_name) = joined
                    .strip_prefix("spec::")
                    .or_else(|| joined.strip_prefix("mc::spec::"))
                {
                    self.scan.used_specs.insert(SmolStr::new(spec_name));
                    if let Some(schema) = self.ctx.schema
                        && let Some(spec_def) = schema.spec_defaults.get(spec_name)
                    {
                        resolved_ty = Some(pack_param_type_to_id(
                            spec_def.ty,
                            &mut self.ctx.module.types,
                        ));
                    }
                }

                if joined.starts_with("mc::RayFlags::") || joined.starts_with("mc::HitType::") {
                    resolved_ty = Some(self.ctx.module.types.alloc(Type::Scalar(ScalarKind::U32)));
                }

                if let Some(last) = path.segments.last()
                    && last.name.starts_with("subgroup_")
                {
                    self.scan.set(FLAG_SUBGROUPS);
                }

                let mut lowered_turbofish = Vec::new();
                for arg in turbofish {
                    if let TypeArg::Type(ty_expr) = arg {
                        if type_expr_contains_bda(ty_expr) {
                            self.scan.set(FLAG_BDA);
                        }
                        let mut type_ctx = self.ctx.type_ctx();
                        if let Some(tid) = lower_type_expr(
                            ty_expr,
                            &mut type_ctx,
                            self.generic_params,
                            self.type_subst,
                        ) {
                            lowered_turbofish.push(tid);
                        }
                    }
                }

                if resolved_ty.is_none() && path.segments.len() == 1 {
                    let ident = path.segments[0].name.as_str();
                    if self.is_non_compute_stage
                        && !self.is_shadowed(ident)
                        && self.ctx.workgroup_var_types.contains_key(ident)
                    {
                        self.ctx.diags.push(Diagnostic::error(
                            "E0251",
                            format!("`workgroup var` `{ident}` is only valid in compute shaders"),
                            path.span,
                        ));
                    }
                    let (vty, vsym) = self.lookup_var(ident);
                    resolved_ty = vty;
                    resolved_sym = vsym;
                }

                let ty = resolved_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::Path {
                            path: SmolStr::new(joined),
                            turbofish: lowered_turbofish,
                        },
                        ty,
                        symbol: resolved_sym,
                        span: *span,
                    },
                    resolved_ty,
                )
            }
            Expr::Call { callee, args, span } => {
                let mut callee_path = None;
                let mut callee_name = None;
                let mut mc_param_ty: Option<TypeId> = None;
                let mut mc_spec_ty: Option<TypeId> = None;

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
                            self.ctx.diags.push(Diagnostic::error(
                                "E0250",
                                "`mc::debug_write` is only valid inside `@stage(compute(...))` entry points",
                                *span,
                            ));
                        }
                    } else if joined == "mc::param" {
                        if let Some(name) = extract_name_arg(args.first().map(|a| &a.value)) {
                            self.scan.used_params.insert(name.clone());
                            if let Some(schema) = self.ctx.schema
                                && let Some(&pt) = schema.param_types.get(&name)
                            {
                                mc_param_ty =
                                    Some(pack_param_type_to_id(pt, &mut self.ctx.module.types));
                            }
                        }
                    } else if joined == "mc::spec"
                        && let Some(name) = extract_name_arg(args.first().map(|a| &a.value))
                    {
                        self.scan.used_specs.insert(name.clone());
                        if let Some(schema) = self.ctx.schema
                            && let Some(spec_def) = schema.spec_defaults.get(&name)
                        {
                            mc_spec_ty = Some(pack_param_type_to_id(
                                spec_def.ty,
                                &mut self.ctx.module.types,
                            ));
                        }
                    }

                    if let Some(last) = path.segments.last() {
                        if last.name.starts_with("subgroup_") {
                            self.scan.set(FLAG_SUBGROUPS);
                        }
                        callee_name = Some(last.name.clone());
                    }
                    callee_path = Some(joined);
                }

                let (hir_callee, _) = self.lower_expr(callee, false);
                let (hir_args, arg_types) = self.lower_call_args(args);

                if let Some(ref fn_name) = callee_name {
                    if matches!(fn_name.as_str(), "f16" | "f16vec2" | "f16vec3" | "f16vec4")
                        && !self.caps.contains(HardwareCaps::FLOAT16)
                    {
                        self.ctx.diags.push(
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

                // Attempt generic function monomorphization if callee is a generic function template.
                let mut resolved_fn: Option<HirFnId> = None;
                let mut mono_sig: Option<FnSig> = None;
                if let Some(ref cp) = callee_path
                    && let Some(template) = self.ctx.generic_fn_templates.get(cp.as_str()).cloned()
                {
                    let explicit_targs = match &hir_callee.kind {
                        HirExprKind::Path { turbofish, .. } if !turbofish.is_empty() => {
                            Some(turbofish.clone())
                        }
                        _ => None,
                    };
                    let inferred_targs = explicit_targs.or_else(|| {
                        let base_sig = self.ctx.fn_sigs.get(cp.as_str())?.clone();
                        let mut inferred_map: IndexMap<SmolStr, TypeId> = IndexMap::new();
                        for ((_, _, param_ty_opt), arg_ty_opt) in
                            base_sig.params.iter().zip(arg_types.iter())
                        {
                            if let (Some(pt), Some(at)) = (*param_ty_opt, *arg_ty_opt) {
                                unify_type_params(
                                    &self.ctx.module.types,
                                    pt,
                                    at,
                                    &mut inferred_map,
                                );
                            }
                        }
                        let mut ordered = Vec::with_capacity(template.generic_params.len());
                        for (gname, _) in &template.generic_params {
                            ordered.push(*inferred_map.get(gname)?);
                        }
                        Some(ordered)
                    });
                    if let Some(targs) = inferred_targs
                        && let Some(mono_id) =
                            self.ctx.monomorphize_function(cp.as_str(), &targs, *span)
                    {
                        resolved_fn = Some(mono_id);
                        let func = &self.ctx.module.functions[mono_id];
                        mono_sig = Some(FnSig {
                            params: func
                                .params
                                .iter()
                                .map(|p| (p.name.clone(), p.mode, p.ty))
                                .collect(),
                            return_ty: func.return_ty,
                            fn_id: Some(mono_id),
                        });
                    }
                }

                // Typecheck arguments against declared signature (if this is a declared user fn).
                let is_builtin_ctor = callee_name
                    .as_deref()
                    .and_then(|n| infer_builtin_constructor_type(n, &mut self.ctx.module.types))
                    .is_some();

                let mut ret_ty: Option<TypeId> = mc_param_ty.or(mc_spec_ty);
                if let Some(sig) = mono_sig {
                    let label = callee_path.as_deref().unwrap_or("<fn>");
                    self.check_call_against_sig(label, &sig.params, &hir_args, &arg_types, *span);
                    ret_ty = sig.return_ty;
                } else if !is_builtin_ctor {
                    let found_sig = callee_path
                        .as_deref()
                        .and_then(|p| self.ctx.fn_sigs.get(p).cloned())
                        .or_else(|| {
                            callee_name
                                .as_deref()
                                .and_then(|n| self.ctx.fn_sigs.get(n).cloned())
                        });
                    if let Some(sig) = found_sig {
                        resolved_fn = sig.fn_id;
                        let label = callee_path
                            .as_deref()
                            .or(callee_name.as_deref())
                            .unwrap_or("<fn>");
                        self.check_call_against_sig(
                            label,
                            &sig.params,
                            &hir_args,
                            &arg_types,
                            *span,
                        );
                        ret_ty = sig.return_ty;
                    }
                }

                if ret_ty.is_none() {
                    ret_ty = callee_name
                        .as_deref()
                        .and_then(|name| {
                            infer_builtin_constructor_type(name, &mut self.ctx.module.types)
                        })
                        .or_else(|| {
                            callee_path.as_deref().and_then(|p| {
                                infer_builtin_call_return_type(
                                    p,
                                    &arg_types,
                                    &mut self.ctx.module.types,
                                )
                            })
                        })
                        .or_else(|| {
                            callee_name.as_deref().and_then(|n| {
                                infer_builtin_call_return_type(
                                    n,
                                    &arg_types,
                                    &mut self.ctx.module.types,
                                )
                            })
                        })
                        .or_else(|| {
                            callee_path
                                .as_deref()
                                .and_then(|p| self.ctx.fn_return_types.get(p).copied())
                        })
                        .or_else(|| {
                            callee_name
                                .as_deref()
                                .and_then(|n| self.ctx.fn_return_types.get(n).copied())
                        });
                }

                let ty = ret_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::Call {
                            callee: Box::new(hir_callee),
                            resolved_fn,
                            args: hir_args,
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    ret_ty,
                )
            }
            Expr::MethodCall {
                receiver,
                method,
                turbofish,
                args,
                span,
            } => {
                let mut lowered_turbofish = Vec::new();
                for arg in turbofish {
                    if let TypeArg::Type(ty_expr) = arg {
                        if type_expr_contains_bda(ty_expr) {
                            self.scan.set(FLAG_BDA);
                        }
                        let mut type_ctx = self.ctx.type_ctx();
                        if let Some(tid) = lower_type_expr(
                            ty_expr,
                            &mut type_ctx,
                            self.generic_params,
                            self.type_subst,
                        ) {
                            lowered_turbofish.push(tid);
                        }
                    }
                }
                let (hir_recv, recv_ty) = self.lower_expr(receiver, false);
                let (hir_args, arg_types) = self.lower_call_args(args);

                let mut resolved_fn: Option<HirFnId> = None;
                let mut ret_ty: Option<TypeId> = None;

                if method.name == "offset" || method.name == "step_by" {
                    ret_ty = recv_ty;
                } else if let Some(rty) = recv_ty {
                    let base_ty = match self.ctx.module.types.get(rty) {
                        Type::Pointer { pointee, .. } => *pointee,
                        _ => rty,
                    };
                    match self.ctx.module.types.get(base_ty).clone() {
                        Type::Struct { name, .. } => {
                            // 1. Check concrete inherent/trait impl methods first.
                            let concrete_id = self
                                .ctx
                                .find_concrete_method(name.as_str(), method.name.as_str())
                                .or_else(|| {
                                    // 2. If `name` is a monomorphized generic struct (e.g.
                                    // `Reservoir<MySample>`), instantiate the `impl<T: Trait>` method!
                                    self.ctx.monomorphize_impl_method(
                                        base_ty,
                                        method.name.as_str(),
                                        *span,
                                    )
                                });

                            if let Some(fn_id) = concrete_id {
                                resolved_fn = Some(fn_id);
                                let (fn_params, fn_ret) = {
                                    let f = &self.ctx.module.functions[fn_id];
                                    let ps: Vec<(SmolStr, ParamMode, Option<TypeId>)> = f
                                        .params
                                        .iter()
                                        .map(|p| (p.name.clone(), p.mode, p.ty))
                                        .collect();
                                    (ps, f.return_ty)
                                };
                                let label = format!("{name}::{}", method.name);
                                if let Some((first_name, first_mode, _)) = fn_params.first()
                                    && first_name == "self"
                                {
                                    if matches!(first_mode, ParamMode::InOut | ParamMode::Out)
                                        && !self.is_lvalue_mutable(&hir_recv)
                                    {
                                        self.ctx.diags.push(Diagnostic::error(
                                            "E0272",
                                            format!(
                                                "method `{label}` requires a mutable receiver for `self`"
                                            ),
                                            hir_recv.span,
                                        ));
                                    }
                                    self.check_call_against_sig(
                                        &label,
                                        &fn_params[1..],
                                        &hir_args,
                                        &arg_types,
                                        *span,
                                    );
                                } else {
                                    self.check_call_against_sig(
                                        &label, &fn_params, &hir_args, &arg_types, *span,
                                    );
                                }
                                ret_ty = fn_ret;
                            } else {
                                let qualified = format!("{name}::{}", method.name);
                                if let Some(sig) = self.ctx.fn_sigs.get(qualified.as_str()).cloned()
                                {
                                    resolved_fn = sig.fn_id;
                                    if let Some((first_name, first_mode, _)) = sig.params.first()
                                        && first_name == "self"
                                    {
                                        if matches!(first_mode, ParamMode::InOut | ParamMode::Out)
                                            && !self.is_lvalue_mutable(&hir_recv)
                                        {
                                            self.ctx.diags.push(Diagnostic::error(
                                                "E0272",
                                                format!(
                                                    "method `{qualified}` requires a mutable receiver for `self`"
                                                ),
                                                hir_recv.span,
                                            ));
                                        }
                                        self.check_call_against_sig(
                                            &qualified,
                                            &sig.params[1..],
                                            &hir_args,
                                            &arg_types,
                                            *span,
                                        );
                                    } else {
                                        self.check_call_against_sig(
                                            &qualified,
                                            &sig.params,
                                            &hir_args,
                                            &arg_types,
                                            *span,
                                        );
                                    }
                                    ret_ty = sig.return_ty;
                                } else {
                                    ret_ty =
                                        self.ctx.fn_return_types.get(qualified.as_str()).copied();
                                }
                            }
                        }
                        Type::TypeParam(param_name) => {
                            // Resolve trait-bounded method calls (e.g. `self.sample.target_pdf(...)`).
                            if let Some(bound_trait) = self.generic_bounds.get(&param_name)
                                && let Some(methods) = self.ctx.trait_method_sigs.get(bound_trait)
                                && let Some(sig) = methods.get(method.name.as_str()).cloned()
                            {
                                let label = format!("{bound_trait}::{}", method.name);
                                let check_params =
                                    if sig.params.first().is_some_and(|(n, _, _)| n == "self") {
                                        &sig.params[1..]
                                    } else {
                                        &sig.params[..]
                                    };
                                self.check_call_against_sig(
                                    &label,
                                    check_params,
                                    &hir_args,
                                    &arg_types,
                                    *span,
                                );
                                ret_ty = sig.return_ty;
                            }
                        }
                        Type::RayQuery => {
                            ret_ty = infer_rayquery_method_return(
                                method.name.as_str(),
                                &mut self.ctx.module.types,
                            );
                        }
                        Type::Texture2D { sample_scalar, .. }
                        | Type::Texture2DArray { sample_scalar }
                        | Type::Texture3D { sample_scalar }
                        | Type::TextureCube { sample_scalar } => {
                            if matches!(
                                method.name.as_str(),
                                "sample" | "sample_lod" | "sample_grad" | "fetch" | "load"
                            ) {
                                ret_ty = Some(self.ctx.module.types.alloc(Type::Vector {
                                    scalar: sample_scalar,
                                    len: 4,
                                }));
                            }
                        }
                        _ => {
                            if let Type::Pointer { pointee, .. } = self.ctx.module.types.get(rty)
                                && method.name == "load"
                            {
                                ret_ty = Some(*pointee);
                            }
                        }
                    }
                }

                let ty = ret_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::MethodCall {
                            receiver: Box::new(hir_recv),
                            method: method.name.clone(),
                            resolved_fn,
                            turbofish: lowered_turbofish,
                            args: hir_args,
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    ret_ty,
                )
            }
            Expr::Field {
                receiver,
                field,
                span,
            } => {
                let mut pack_field_ty: Option<TypeId> = None;
                if let Expr::Path { path, .. } = receiver.as_ref() {
                    let joined = path.joined();
                    if joined == "pack::params"
                        || joined == "mc::params"
                        || (joined == "params" && !self.is_shadowed("params"))
                    {
                        self.scan.used_params.insert(field.name.clone());
                        if let Some(schema) = self.ctx.schema
                            && let Some(&pt) = schema.param_types.get(&field.name)
                        {
                            pack_field_ty =
                                Some(pack_param_type_to_id(pt, &mut self.ctx.module.types));
                        }
                    } else if joined == "mc::spec"
                        || (joined == "spec" && !self.is_shadowed("spec"))
                    {
                        self.scan.used_specs.insert(field.name.clone());
                        if let Some(schema) = self.ctx.schema
                            && let Some(spec_def) = schema.spec_defaults.get(&field.name)
                        {
                            pack_field_ty = Some(pack_param_type_to_id(
                                spec_def.ty,
                                &mut self.ctx.module.types,
                            ));
                        }
                    }
                }

                let (hir_recv, recv_ty) = self.lower_expr(receiver, false);

                if let Some(rty) = recv_ty {
                    let base_ty = match self.ctx.module.types.get(rty) {
                        Type::Pointer { pointee, .. } => *pointee,
                        _ => rty,
                    };
                    match self.ctx.module.types.get(base_ty).clone() {
                        Type::Vector { scalar, len } => {
                            match parse_vector_swizzle(field.name.as_str(), len, is_write_target) {
                                Ok(components) => {
                                    let swizzle_ty = if components.len() == 1 {
                                        self.ctx.module.types.alloc(Type::Scalar(scalar))
                                    } else {
                                        self.ctx.module.types.alloc(Type::Vector {
                                            scalar,
                                            len: components.len() as u8,
                                        })
                                    };
                                    return (
                                        HirExpr {
                                            kind: HirExprKind::Swizzle {
                                                base: Box::new(hir_recv),
                                                components,
                                                is_write: is_write_target,
                                            },
                                            ty: swizzle_ty,
                                            symbol: None,
                                            span: *span,
                                        },
                                        Some(swizzle_ty),
                                    );
                                }
                                Err(err) => {
                                    let msg = match err {
                                        SwizzleError::InvalidLength => format!(
                                            "invalid vector swizzle `.{}`: length must be 1..=4",
                                            field.name
                                        ),
                                        SwizzleError::InvalidComponent => format!(
                                            "invalid vector swizzle `.{}`: components must be from `xyzw`, `rgba`, or `stpq`",
                                            field.name
                                        ),
                                        SwizzleError::MixedSets => format!(
                                            "invalid vector swizzle `.{}`: cannot mix component sets (`xyzw`/`rgba`/`stpq`)",
                                            field.name
                                        ),
                                        SwizzleError::OutOfRange { component, vec_len } => {
                                            format!(
                                                "swizzle component `{component}` is out of bounds for {vec_len}-component vector"
                                            )
                                        }
                                        SwizzleError::DuplicateWriteComponent(ch) => format!(
                                            "duplicate component `{ch}` in write swizzle `.{}`",
                                            field.name
                                        ),
                                    };
                                    self.ctx.diags.push(Diagnostic::error("E0273", msg, *span));
                                    let void = self.void_ty();
                                    return (
                                        HirExpr {
                                            kind: HirExprKind::Field {
                                                receiver: Box::new(hir_recv),
                                                field: field.name.clone(),
                                                field_index: None,
                                            },
                                            ty: void,
                                            symbol: None,
                                            span: *span,
                                        },
                                        None,
                                    );
                                }
                            }
                        }
                        Type::Struct { fields, .. } => {
                            let found = fields
                                .iter()
                                .enumerate()
                                .find(|(_, (name, _))| *name == field.name);
                            let (field_ty_opt, field_index) = match found {
                                Some((idx, (_, ty_id))) => (Some(*ty_id), Some(idx as u32)),
                                None => (pack_field_ty, None),
                            };
                            let ty = field_ty_opt.unwrap_or_else(|| self.void_ty());
                            return (
                                HirExpr {
                                    kind: HirExprKind::Field {
                                        receiver: Box::new(hir_recv),
                                        field: field.name.clone(),
                                        field_index,
                                    },
                                    ty,
                                    symbol: None,
                                    span: *span,
                                },
                                field_ty_opt,
                            );
                        }
                        _ => {}
                    }
                }

                let ty = pack_field_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::Field {
                            receiver: Box::new(hir_recv),
                            field: field.name.clone(),
                            field_index: None,
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    pack_field_ty,
                )
            }
            Expr::StructLit { path, fields, span } => {
                let mut hir_fields = Vec::with_capacity(fields.len());
                for (fname, fval) in fields {
                    let (v, _) = self.lower_expr(fval, false);
                    hir_fields.push((fname.name.clone(), v));
                }
                let struct_name = SmolStr::new(path.joined());
                let resolved_ty = self
                    .ctx
                    .struct_types
                    .get(struct_name.as_str())
                    .copied()
                    .or_else(|| {
                        self.ctx
                            .builtin_struct_types
                            .get(struct_name.as_str())
                            .copied()
                    });
                let ty = resolved_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::StructLit {
                            struct_name,
                            fields: hir_fields,
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    resolved_ty,
                )
            }
            Expr::Unary { op, operand, span } => {
                let (hir_op, inner_ty) = self.lower_expr(operand, false);
                let result_ty = match op {
                    UnaryOp::Not => {
                        if let Some(it) = inner_ty
                            && self.is_concrete_value_ty(it)
                        {
                            match self.ctx.module.types.get(it) {
                                Type::Scalar(ScalarKind::Bool)
                                | Type::Vector {
                                    scalar: ScalarKind::Bool,
                                    ..
                                } => {}
                                _ => {
                                    let tname = format_type_name(&self.ctx.module.types, it);
                                    self.ctx.diags.push(Diagnostic::error(
                                        "E0270",
                                        format!(
                                            "logical `!` requires `bool` operand, found `{tname}`"
                                        ),
                                        *span,
                                    ));
                                }
                            }
                        }
                        Some(self.ctx.module.types.alloc(Type::Scalar(ScalarKind::Bool)))
                    }
                    UnaryOp::Neg => {
                        if let Some(it) = inner_ty
                            && self.is_concrete_value_ty(it)
                        {
                            let valid = matches!(
                                self.ctx.module.types.get(it),
                                Type::Scalar(s)
                                | Type::Vector { scalar: s, .. }
                                | Type::Matrix { scalar: s, .. } if is_numeric_scalar(*s)
                            );
                            if !valid {
                                let tname = format_type_name(&self.ctx.module.types, it);
                                self.ctx.diags.push(Diagnostic::error(
                                    "E0270",
                                    format!("unary `-` requires numeric operand, found `{tname}`"),
                                    *span,
                                ));
                            }
                        }
                        inner_ty
                    }
                    UnaryOp::BitNot => {
                        if let Some(it) = inner_ty
                            && self.is_concrete_value_ty(it)
                        {
                            let valid = matches!(
                                self.ctx.module.types.get(it),
                                Type::Scalar(s) | Type::Vector { scalar: s, .. } if is_integer_scalar(*s)
                            );
                            if !valid {
                                let tname = format_type_name(&self.ctx.module.types, it);
                                self.ctx.diags.push(Diagnostic::error(
                                    "E0270",
                                    format!(
                                        "bitwise `~` requires integer operand, found `{tname}`"
                                    ),
                                    *span,
                                ));
                            }
                        }
                        inner_ty
                    }
                    UnaryOp::AddrOf => inner_ty,
                };
                let ty = result_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::Unary {
                            op: *op,
                            operand: Box::new(hir_op),
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    result_ty,
                )
            }
            Expr::Binary { op, lhs, rhs, span } => {
                let is_assign = is_assignment_op(*op);
                let (hir_lhs, lhs_ty) = self.lower_expr(lhs, is_assign);
                let (hir_rhs, rhs_ty) = self.lower_expr(rhs, false);

                if is_assign && !self.is_lvalue_mutable(&hir_lhs) {
                    self.ctx.diags.push(Diagnostic::error(
                        "E0271",
                        "cannot assign to immutable variable or non-mutable lvalue",
                        hir_lhs.span,
                    ));
                }

                if matches!(op, BinaryOp::Assign) {
                    if expr_yields_addr_of(rhs) {
                        self.ctx.diags.push(Diagnostic::error(
                            "E0215",
                            "assigning a logical pointer (`&...`) to a variable is forbidden in SPIR-V",
                            *span,
                        ));
                    }
                    if lhs_ty.is_some_and(|t| self.ctx.module.types.contains_ray_query(t))
                        || rhs_ty.is_some_and(|t| self.ctx.module.types.contains_ray_query(t))
                    {
                        self.ctx.diags.push(Diagnostic::error(
                            "E0217",
                            "`RayQuery` cannot be assigned or copied",
                            *span,
                        ));
                    }
                }

                let res_ty = self.check_binary_types(*op, lhs_ty, rhs_ty, *span);
                let ty = res_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::Binary {
                            op: *op,
                            lhs: Box::new(hir_lhs),
                            rhs: Box::new(hir_rhs),
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    res_ty,
                )
            }
            Expr::Index {
                receiver,
                index,
                span,
            } => {
                let (hir_recv, recv_ty) = self.lower_expr(receiver, false);
                let (hir_idx, _) = self.lower_expr(index, false);
                let elem_ty =
                    recv_ty.and_then(|ty_id| match self.ctx.module.types.get(ty_id).clone() {
                        Type::Array { element, .. }
                        | Type::Buffer { element }
                        | Type::RwBuffer { element } => Some(element),
                        Type::Texture2D {
                            sample_scalar,
                            unsized_array: true,
                        } => Some(self.ctx.module.types.alloc(Type::Texture2D {
                            sample_scalar,
                            unsized_array: false,
                        })),
                        Type::Vector { scalar, .. } => {
                            Some(self.ctx.module.types.alloc(Type::Scalar(scalar)))
                        }
                        Type::Matrix { scalar, rows, .. } => Some(
                            self.ctx
                                .module
                                .types
                                .alloc(Type::Vector { scalar, len: rows }),
                        ),
                        Type::Struct { fields, .. }
                            if fields.len() == 1 && fields[0].0 == "items" =>
                        {
                            match self.ctx.module.types.get(fields[0].1) {
                                Type::Buffer { element } | Type::RwBuffer { element } => {
                                    Some(*element)
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    });
                let ty = elem_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::Index {
                            receiver: Box::new(hir_recv),
                            index: Box::new(hir_idx),
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    elem_ty,
                )
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => {
                let (hir_cond, _) = self.lower_expr(cond, false);
                let (hir_then, then_ty) = self.lower_block(then_branch);
                let (hir_else, else_ty) = if let Some(eb) = else_branch.as_deref() {
                    let (e, ty) = self.lower_expr(eb, false);
                    (Some(Box::new(e)), ty)
                } else {
                    (None, None)
                };
                if else_branch.is_some()
                    && (then_ty.is_some_and(|t| self.ctx.module.types.contains_ray_query(t))
                        || else_ty.is_some_and(|t| self.ctx.module.types.contains_ray_query(t)))
                {
                    self.ctx.diags.push(
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
                let res_ty = then_ty.or(else_ty);
                let ty = res_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::If {
                            cond: Box::new(hir_cond),
                            then_branch: hir_then,
                            else_branch: hir_else,
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    res_ty,
                )
            }
            Expr::Block(b) => {
                let (hir_block, block_ty) = self.lower_block(b);
                let ty = hir_block.ty;
                let span = hir_block.span;
                (
                    HirExpr {
                        kind: HirExprKind::Block(hir_block),
                        ty,
                        symbol: None,
                        span,
                    },
                    block_ty,
                )
            }
            Expr::ArrayLit { elements, span } => {
                let mut hir_elems = Vec::with_capacity(elements.len());
                let mut elem_ty = None;
                for el in elements {
                    let (he, t) = self.lower_expr(el, false);
                    if elem_ty.is_none() {
                        elem_ty = t;
                    }
                    hir_elems.push(he);
                }
                let arr_ty = elem_ty.and_then(|e| {
                    u32::try_from(elements.len())
                        .ok()
                        .map(|len| self.ctx.module.types.alloc(Type::Array { element: e, len }))
                });
                let ty = arr_ty.unwrap_or_else(|| self.void_ty());
                (
                    HirExpr {
                        kind: HirExprKind::ArrayLit {
                            elements: hir_elems,
                        },
                        ty,
                        symbol: None,
                        span: *span,
                    },
                    arr_ty,
                )
            }
        }
    }

    fn check_atomic_caps(&mut self, fn_name: &str, arg_types: &[Option<TypeId>], span: Span) {
        let mut is_int64 = false;
        let mut is_float = fn_name.starts_with("atomic_f");

        for ty_id in arg_types.iter().flatten() {
            let scalar = match self.ctx.module.types.get(*ty_id) {
                Type::Pointer { pointee, .. } => match self.ctx.module.types.get(*pointee) {
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
            self.ctx.diags.push(Diagnostic::error(
                "E0237",
                format!(
                    "64-bit atomic `{fn_name}` requires `#[cap(int64_atomics)]` on the enclosing function"
                ),
                span,
            ));
        }

        if is_float && !self.caps.contains(HardwareCaps::FLOAT_ATOMICS) {
            self.ctx.diags.push(Diagnostic::error(
                "E0238",
                format!(
                    "floating-point atomic `{fn_name}` requires `#[cap(float_atomics)]` on the enclosing function"
                ),
                span,
            ));
        }
    }
}

fn infer_rayquery_method_return(method: &str, types: &mut TypeArena) -> Option<TypeId> {
    let ty = match method {
        "proceed" => Type::Scalar(ScalarKind::Bool),
        "committed_type"
        | "committed_instance_custom_index"
        | "committed_instance_id"
        | "committed_primitive_index"
        | "committed_geometry_index" => Type::Scalar(ScalarKind::U32),
        "committed_t" => Type::Scalar(ScalarKind::F32),
        "committed_barycentrics" => Type::Vector {
            scalar: ScalarKind::F32,
            len: 2,
        },
        _ => return None,
    };
    Some(types.alloc(ty))
}

fn infer_builtin_call_return_type(
    name: &str,
    arg_types: &[Option<TypeId>],
    types: &mut TypeArena,
) -> Option<TypeId> {
    match name {
        "subgroup_ballot" => Some(types.alloc(Type::Vector {
            scalar: ScalarKind::U32,
            len: 4,
        })),
        "subgroup_ballot_bit_count" | "subgroup_ballot_exclusive_bit_count" => {
            Some(types.alloc(Type::Scalar(ScalarKind::U32)))
        }
        "subgroup_broadcast_first"
        | "subgroup_broadcast"
        | "nonuniform"
        | "min"
        | "max"
        | "clamp"
        | "mix"
        | "abs"
        | "floor"
        | "ceil"
        | "fract"
        | "sqrt"
        | "inversesqrt"
        | "sin"
        | "cos"
        | "tan"
        | "exp"
        | "exp2"
        | "log"
        | "log2"
        | "pow"
        | "normalize"
        | "reflect"
        | "refract"
        | "cross"
        | "fma"
        | "smoothstep"
        | "step"
        | "saturate" => arg_types.first().copied().flatten(),
        "dot" | "length" | "distance" => Some(types.alloc(Type::Scalar(ScalarKind::F32))),
        "mc::reconstruct_world_pos" | "mc::view_dir" | "mc::sample_sky" => {
            Some(types.alloc(Type::Vector {
                scalar: ScalarKind::F32,
                len: 3,
            }))
        }
        _ if name.starts_with("atomic_") => {
            let first = arg_types.first().copied().flatten()?;
            match types.get(first) {
                Type::Pointer { pointee, .. } => Some(*pointee),
                _ => Some(first),
            }
        }
        _ => None,
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
    builtin_struct_types: &'a IndexMap<SmolStr, TypeId>,
    struct_generic_args: Option<&'a mut IndexMap<SmolStr, (SmolStr, Vec<TypeId>)>>,
}

impl<'a> TypeLowerCtx<'a> {
    const fn types_only(
        types: &'a mut TypeArena,
        struct_types: &'a mut IndexMap<SmolStr, TypeId>,
        builtin_struct_types: &'a IndexMap<SmolStr, TypeId>,
    ) -> Self {
        Self {
            types,
            structs: None,
            struct_types,
            struct_ids: None,
            builtin_struct_types,
            struct_generic_args: None,
        }
    }

    const fn full(
        types: &'a mut TypeArena,
        structs: &'a mut Arena<HirStruct>,
        struct_types: &'a mut IndexMap<SmolStr, TypeId>,
        struct_ids: &'a mut IndexMap<SmolStr, HirStructId>,
        builtin_struct_types: &'a IndexMap<SmolStr, TypeId>,
        struct_generic_args: &'a mut IndexMap<SmolStr, (SmolStr, Vec<TypeId>)>,
    ) -> Self {
        Self {
            types,
            structs: Some(structs),
            struct_types,
            struct_ids: Some(struct_ids),
            builtin_struct_types,
            struct_generic_args: Some(struct_generic_args),
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
    type_subst: &IndexMap<SmolStr, TypeId>,
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
        let arg_ty = lower_type_expr(arg_expr, ctx, generic_params, type_subst)?;
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
    for (param_name, &arg_ty) in base_generic_params.iter().zip(&lowered_args) {
        subst.insert(param_name.clone(), arg_ty);
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
    if let Some(sga) = ctx.struct_generic_args.as_deref_mut() {
        sga.insert(mono_name.clone(), (SmolStr::new(base_name), lowered_args));
    }
    ctx.struct_types.insert(mono_name, mono_type_id);
    Some(mono_type_id)
}

fn lower_type_expr(
    ty: &TypeExpr,
    ctx: &mut TypeLowerCtx<'_>,
    generic_params: &IndexSet<SmolStr>,
    type_subst: &IndexMap<SmolStr, TypeId>,
) -> Option<TypeId> {
    match ty {
        TypeExpr::Named {
            path,
            args,
            unsized_array,
            ..
        } => {
            let name = path.joined();
            if args.is_empty() && !*unsized_array {
                if let Some(&sub_ty) = type_subst.get(name.as_str()) {
                    return Some(sub_ty);
                }
                if generic_params.contains(name.as_str()) {
                    return Some(ctx.types.alloc(Type::TypeParam(SmolStr::new(name))));
                }
            }
            if let Some(&id) = ctx.struct_types.get(name.as_str()) {
                if !args.is_empty()
                    && let Some(mono_id) =
                        try_monomorphize_struct(&name, args, ctx, generic_params, type_subst)
                {
                    return Some(mono_id);
                }
                return Some(id);
            }
            if let Some(&id) = ctx.builtin_struct_types.get(name.as_str()) {
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
                            TypeArg::Type(inner) => {
                                lower_type_expr(inner, ctx, generic_params, type_subst)
                            }
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
                            TypeArg::Type(inner) => {
                                lower_type_expr(inner, ctx, generic_params, type_subst)
                            }
                            TypeArg::NamedExpr { .. } => None,
                        })
                        .unwrap_or_else(|| ctx.types.alloc(Type::Void));
                    Type::Buffer { element }
                }
                "RWBuffer" => {
                    let element = args
                        .iter()
                        .find_map(|arg| match arg {
                            TypeArg::Type(inner) => {
                                lower_type_expr(inner, ctx, generic_params, type_subst)
                            }
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
            let elem_id = lower_type_expr(element, ctx, generic_params, type_subst)?;
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

    #[test]
    fn lowers_typed_hir_body_symbols_consts_and_swizzles() {
        let src = r"
            pub const SCALE: f32 = 2.0;

            pub fn compute_tint(v: vec4) -> vec3 {
                let mut out = v.xyz * SCALE;
                out.xy = vec2(1.0, 2.0);
                return out;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        assert_eq!(hir.consts.len(), 1);
        assert_eq!(hir.consts[0].name, "SCALE");
        assert_eq!(
            *hir.types.get(hir.consts[0].ty),
            Type::Scalar(ScalarKind::F32)
        );

        let func = hir.find_function("compute_tint").expect("compute_tint");
        assert!(func.is_pub);
        let body = func.body_hir.as_ref().expect("body_hir populated");
        assert_eq!(body.stmts.len(), 3);

        // First statement: `let mut out = v.xyz * SCALE;`
        match &body.stmts[0] {
            HirStmt::Let {
                symbol,
                is_mut,
                ty,
                init: Some(init_expr),
                ..
            } => {
                assert!(*is_mut);
                assert_eq!(hir.symbol(*symbol).kind, SymbolKind::Local);
                assert_eq!(
                    *hir.types.get(*ty),
                    Type::Vector {
                        scalar: ScalarKind::F32,
                        len: 3,
                    }
                );
                assert!(matches!(
                    &init_expr.kind,
                    HirExprKind::Binary {
                        op: BinaryOp::Mul,
                        lhs,
                        ..
                    } if matches!(
                        &lhs.kind,
                        HirExprKind::Swizzle {
                            components,
                            is_write: false,
                            ..
                        } if components == &[0, 1, 2]
                    )
                ));
            }
            other => panic!("expected HirStmt::Let, got {other:?}"),
        }

        // Second statement: `out.xy = vec2(1.0, 2.0);` -> write swizzle
        match &body.stmts[1] {
            HirStmt::Expr { expr, .. } => match &expr.kind {
                HirExprKind::Binary {
                    op: BinaryOp::Assign,
                    lhs,
                    ..
                } => match &lhs.kind {
                    HirExprKind::Swizzle {
                        components,
                        is_write,
                        ..
                    } => {
                        assert_eq!(components, &[0, 1]);
                        assert!(*is_write);
                    }
                    other => panic!("expected write swizzle, got {other:?}"),
                },
                other => panic!("expected assign expr, got {other:?}"),
            },
            other => panic!("expected HirStmt::Expr, got {other:?}"),
        }
    }

    #[test]
    fn reports_type_mismatch_mutability_inout_swizzle_call_and_return_errors() {
        let src = r"
            pub fn bump(inout x: f32, y: u32) -> f32 {
                x += 1.0;
                return x;
            }

            pub fn bad_types(in_val: f32, mut v: vec3) -> u32 {
                let imm = 1.0;
                imm = 2.0;
                in_val = 3.0;
                let _sum = 1u + 2.0;
                let _r1 = bump(imm, 1u);
                let _r2 = bump(v.x, 1.0);
                let _r3 = bump(v.x);
                v.xx = vec2(1.0, 2.0);
                let _oob = v.w;
                return 1.0;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (_, diags) = lower_unit(&parsed.unit, RenderTier::Tier0Raster, HardwareCaps::empty());
        let codes: Vec<&str> = diags.as_slice().iter().map(|d| d.code.as_str()).collect();
        assert!(
            codes.contains(&"E0270"),
            "expected E0270 binary mismatch: {codes:?}"
        );
        assert!(
            codes.iter().filter(|&&c| c == "E0271").count() >= 2,
            "expected E0271 for immutable let & ParamMode::In: {codes:?}"
        );
        assert!(
            codes.contains(&"E0272"),
            "expected E0272 inout immutable arg: {codes:?}"
        );
        assert!(
            codes.iter().filter(|&&c| c == "E0273").count() >= 2,
            "expected E0273 for write-swizzle dup & out-of-range: {codes:?}"
        );
        assert!(
            codes.iter().filter(|&&c| c == "E0274").count() >= 2,
            "expected E0274 for call arg type & arity mismatch: {codes:?}"
        );
        assert!(
            codes.contains(&"E0275"),
            "expected E0275 return mismatch: {codes:?}"
        );
    }

    #[test]
    fn monomorphizes_generic_functions_and_trait_bounded_impl_methods() {
        let src = r"
            pub trait ResampleTarget {
                fn target_pdf(self, world_pos: vec3, normal: vec3) -> f32;
            }

            pub struct MySample {
                pub pos: vec3,
                pub radiance: f32,
            }

            impl ResampleTarget for MySample {
                pub fn target_pdf(self, world_pos: vec3, normal: vec3) -> f32 {
                    return self.radiance;
                }
            }

            #[layout(std430)]
            pub struct Reservoir<S: ResampleTarget> {
                pub sample: S,
                pub w_sum: f32,
                pub m: u32,
                pub w: f32,
            }

            impl<S: ResampleTarget> Reservoir<S> {
                pub fn finalize(inout self, world_pos: vec3, normal: vec3) {
                    let p_hat = self.sample.target_pdf(world_pos, normal);
                    self.w = if p_hat > 1e-6 && self.m > 0u {
                        self.w_sum / (f32(self.m) * p_hat)
                    } else {
                        0.0
                    };
                }
            }

            pub fn eval_generic<T: ResampleTarget>(s: T, p: vec3, n: vec3) -> f32 {
                return s.target_pdf(p, n);
            }

            pub fn driver(
                inout res: Reservoir<MySample>,
                s: MySample,
                p: vec3,
                n: vec3,
            ) -> f32 {
                res.finalize(p, n);
                return eval_generic::<MySample>(s, p, n);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (hir, diags) = lower_unit(&parsed.unit, RenderTier::Tier2RayQuery, HardwareCaps::all());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        // Trait and impl metadata populated on HirModule.
        assert_eq!(hir.traits.len(), 1);
        assert_eq!(hir.traits[0].name, "ResampleTarget");
        assert_eq!(hir.impls.len(), 2);

        // Monomorphized `Reservoir<MySample>::finalize` instantiated with concrete `target_pdf` call.
        let mono_finalize = hir
            .find_function("Reservoir<MySample>::finalize")
            .expect("monomorphized Reservoir<MySample>::finalize");
        assert!(mono_finalize.generic_params.is_empty());
        let finalize_body = mono_finalize.body_hir.as_ref().expect("finalize body_hir");
        match &finalize_body.stmts[0] {
            HirStmt::Let {
                ty,
                init: Some(init_expr),
                ..
            } => {
                assert_eq!(*hir.types.get(*ty), Type::Scalar(ScalarKind::F32));
                match &init_expr.kind {
                    HirExprKind::MethodCall {
                        method,
                        resolved_fn,
                        ..
                    } => {
                        assert_eq!(method, "target_pdf");
                        let target_fn_id =
                            resolved_fn.expect("trait method target_pdf resolved to concrete impl");
                        assert_eq!(hir.functions[target_fn_id].name, "target_pdf");
                    }
                    other => panic!("expected MethodCall, got {other:?}"),
                }
            }
            other => panic!("expected Let p_hat, got {other:?}"),
        }

        // Monomorphized generic function `eval_generic<MySample>` instantiated cleanly.
        let mono_eval = hir
            .find_function("eval_generic<MySample>")
            .expect("monomorphized eval_generic<MySample>");
        assert_eq!(
            mono_eval.return_ty.map(|t| hir.types.get(t).clone()),
            Some(Type::Scalar(ScalarKind::F32))
        );
    }

    #[test]
    fn typechecks_vsl_mc_builtin_struct_fields_and_pack_schema_types() {
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn inspect_engine_buffers(
                #[builtin(global_invocation_id)] gid: uvec3,
            ) {
                let cam: vec3 = mc::world.camera_pos;
                let vp: mat4 = mc::world.inv_view_proj;
                let frame: u32 = mc::world.frame_index;
                let light_cnt: u32 = mc::lights.count;
                let light_pos: vec3 = mc::lights.items[0u].position;
                let light_flux: f32 = mc::lights.items[0u].flux;
                let inst_mat: u32 = mc::instances[0u].material_id;
                let sharc_cap: u32 = mc::sharc.capacity;
                let res_w: f32 = mc::reservoirs.items[0u].weight_sum;
                let tint: vec3 = mc::param(sun_tint);
                let bounces: i32 = spec::max_bounces;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let mut schema = PackSchema::default();
        schema
            .params
            .insert(SmolStr::new_static("sun_tint"), (0, 12));
        schema
            .param_types
            .insert(SmolStr::new_static("sun_tint"), PackParamType::Vec3);
        schema
            .spec_ids
            .insert(SmolStr::new_static("max_bounces"), 1);
        schema.spec_defaults.insert(
            SmolStr::new_static("max_bounces"),
            SpecConstantDefault {
                spec_id: 1,
                ty: PackParamType::I32,
                default_bits: 4,
            },
        );

        let (hir, diags) = lower_unit_with_schema(
            &parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::all(),
            Some(&schema),
        );
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        let func = hir
            .find_function("inspect_engine_buffers")
            .expect("inspect_engine_buffers");
        assert_eq!(
            *hir.types.get(func.local_types["cam"]),
            Type::Vector {
                scalar: ScalarKind::F32,
                len: 3,
            }
        );
        assert_eq!(
            *hir.types.get(func.local_types["light_cnt"]),
            Type::Scalar(ScalarKind::U32)
        );
        assert_eq!(
            *hir.types.get(func.local_types["light_pos"]),
            Type::Vector {
                scalar: ScalarKind::F32,
                len: 3,
            }
        );
        assert_eq!(
            *hir.types.get(func.local_types["inst_mat"]),
            Type::Scalar(ScalarKind::U32)
        );
        assert_eq!(
            *hir.types.get(func.local_types["sharc_cap"]),
            Type::Scalar(ScalarKind::U32)
        );
        assert_eq!(
            *hir.types.get(func.local_types["res_w"]),
            Type::Scalar(ScalarKind::F32)
        );
        assert_eq!(
            *hir.types.get(func.local_types["tint"]),
            Type::Vector {
                scalar: ScalarKind::F32,
                len: 3,
            }
        );
        assert_eq!(
            *hir.types.get(func.local_types["bounces"]),
            Type::Scalar(ScalarKind::I32)
        );
    }
}
