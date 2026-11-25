//! Native SPIR-V `@hook` chain dispatch synthesis and `HookStage::Clouds` alpha-over
//! compositing (Item 23).

use crate::cache::hash_pack_schema;
use crate::lower_ir::{build_unified_descriptors, lower_hook_callee_to_ir};
use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, SourceDatabase, Span};
use vsl_hir::module::{L1ModuleHash, compute_l2_artifact_hash};
use vsl_hir::{HirFunction, HirModule, lower_unit_with_schema};
use vsl_ir::{
    BuiltinKind, GlslStd450Op, InstData, InstId, IrDescriptorVar, IrFunction, IrModule,
    IrPushConstantBlock, IrStorageClass, IrType, PassPipeline, ScalarIrTy, StructuredStmt,
};
use vsl_mc::{
    CompiledShaderUnit, DESCRIPTOR_SET_PER_PASS, DescriptorAccess, DescriptorBinding,
    DescriptorKind, HardwareCaps, HookAbiRoute, HookStage, ImageFormat, PushConstantField,
    ShaderStage, TextureDim, WORLD_UBO_LAYOUT, builtin_descriptor_layout,
};
use vsl_spirv::emit_module_spirv;
use vsl_syntax::ast::Item;
use vsl_syntax::parse_source;

use crate::CompileOptions;

/// Compiles an ordered chain of native SPIR-V `@hook` functions for `stage` into a single
/// dispatch entry point `vs_hook_chain_<stage>`.
///
/// For `HookStage::Clouds` (`stage.uses_alpha_over_blending()`), emits exact front-to-back
/// premultiplied alpha-over compositing between each layer:
/// `acc.rgb += (1.0 - acc.a) * src.rgb * src.a` and
/// `acc.a   += (1.0 - acc.a) * src.a`, then writes the final `vec4` to the output target.
pub(crate) fn compile_hook_chain_sources(
    sources: &mut SourceDatabase,
    stage: HookStage,
    hook_functions: &[(&str, &str)],
    options: &CompileOptions,
) -> Result<CompiledShaderUnit, Vec<Diagnostic>> {
    if stage.abi_route() != HookAbiRoute::NativeSpirv {
        return Err(vec![Diagnostic::error(
            "E0511",
            format!(
                "hook stage `{}` uses the raster G-Buffer GLSL bridge and cannot be compiled as a native SPIR-V hook chain",
                stage.as_str()
            ),
            Span::synthetic(),
        )]);
    }
    if hook_functions.is_empty() {
        return Err(vec![Diagnostic::error(
            "E0511",
            format!(
                "cannot compile empty `@hook({})` chain: at least one hook function is required",
                stage.as_str()
            ),
            Span::synthetic(),
        )]);
    }

    let mut diagnostics = Vec::new();
    let mut lowered_hooks: Vec<(HirModule, HirFunction, Vec<Item>)> =
        Vec::with_capacity(hook_functions.len());
    let mut chain_hasher = blake3::Hasher::new();
    chain_hasher.update(b"vsl2-hook-chain-v1");
    chain_hasher.update(stage.as_str().as_bytes());

    for (idx, &(label, source)) in hook_functions.iter().enumerate() {
        chain_hasher.update(&(label.len() as u64).to_le_bytes());
        chain_hasher.update(label.as_bytes());
        chain_hasher.update(&(source.len() as u64).to_le_bytes());
        chain_hasher.update(source.as_bytes());

        let file_name = if label.ends_with(".vsl") {
            label.to_owned()
        } else {
            format!("hook_chain_{}_{idx}.vsl", stage.as_str())
        };
        let file = sources.add_file(file_name, source);
        let parsed = parse_source(file, source);
        diagnostics.extend(parsed.diagnostics);
        if diagnostics.iter().any(Diagnostic::is_error) {
            continue;
        }

        let (hir, hir_diags) = lower_unit_with_schema(
            &parsed.unit,
            options.tier,
            options.active_caps,
            options.pack_schema.as_ref(),
        );
        diagnostics.extend(hir_diags.into_vec());
        if diagnostics.iter().any(Diagnostic::is_error) {
            continue;
        }

        let matched = hir
            .functions
            .iter()
            .map(|(_, f)| f)
            .find(|f| f.hook == Some(stage) && f.name == label)
            .or_else(|| {
                hir.functions
                    .iter()
                    .map(|(_, f)| f)
                    .find(|f| f.hook == Some(stage))
            })
            .cloned();

        if let Some(hook_fn) = matched {
            lowered_hooks.push((hir, hook_fn, parsed.unit.items));
        } else {
            diagnostics.push(Diagnostic::error(
                "E0511",
                format!(
                    "no `@hook({})` function found in hook chain entry `{label}`",
                    stage.as_str()
                ),
                Span::new(file, 0, 0),
            ));
        }
    }

    if diagnostics.iter().any(Diagnostic::is_error) {
        return Err(diagnostics);
    }

    let chain_l1 = L1ModuleHash(*chain_hasher.finalize().as_bytes());
    build_hook_chain_unit(stage, &lowered_hooks, chain_l1, options, &mut diagnostics)
        .ok_or(diagnostics)
}

/// Synthesizes a native SPIR-V hook chain [`CompiledShaderUnit`] from pre-lowered
/// `(HirModule, HirFunction, Vec<Item>)` entries (shared by `compile_hook_chain` and
/// `compile_project`).
pub(crate) fn build_hook_chain_unit(
    stage: HookStage,
    lowered_hooks: &[(HirModule, HirFunction, Vec<Item>)],
    chain_l1: L1ModuleHash,
    options: &CompileOptions,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CompiledShaderUnit> {
    let entry_name = SmolStr::new(format!("vs_hook_chain_{}", stage.as_str()));
    let workgroup_size = Some([8, 8, 1]);

    // Merge descriptors, caps, specs, params, push constants, and debug channels across all
    // chained hooks, remapping colliding Set 0 bindings (1..=14) cleanly.
    let mut merged_descriptors: IndexMap<(u32, u32), DescriptorBinding> = IndexMap::new();
    let mut required_caps = HardwareCaps::empty();
    let mut used_specs = Vec::new();
    let mut seen_specs = IndexSet::new();
    let mut used_params = Vec::new();
    let mut seen_params = IndexSet::new();
    let mut debug_channels = Vec::new();
    let mut merged_push_constants: Vec<PushConstantField> = Vec::new();
    let mut merged_push_constant_size = 0u32;
    let mut uses_bda = false;
    let mut uses_subgroups = false;
    let mut uses_quad_control = false;

    // Bind Set 0 binding 0 (`WorldUBO`) so dispatch can derive stage view directions/time.
    merged_descriptors.insert(
        (DESCRIPTOR_SET_PER_PASS, 0),
        DescriptorBinding {
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
        },
    );

    for (_, hook_fn, _) in lowered_hooks {
        required_caps |= hook_fn.required_caps;
        uses_bda |= hook_fn.reflection.uses_bda;
        uses_subgroups |= hook_fn.reflection.uses_subgroups;
        if hook_fn.reflection.push_constant_size > merged_push_constant_size {
            merged_push_constant_size = hook_fn.reflection.push_constant_size;
            merged_push_constants = hook_fn.reflection.push_constants.clone();
        }
        for mut desc in build_unified_descriptors(hook_fn) {
            if desc.set == DESCRIPTOR_SET_PER_PASS && (1..=14).contains(&desc.binding) {
                if let Some(existing) = merged_descriptors.get(&(desc.set, desc.binding)) {
                    let same_binding = existing.name == desc.name
                        && existing.kind == desc.kind
                        && existing.image_format == desc.image_format
                        && existing.access == desc.access;
                    if !same_binding {
                        let Some(free_b) = (1u32..=14).find(|&b| {
                            !merged_descriptors.contains_key(&(DESCRIPTOR_SET_PER_PASS, b))
                        }) else {
                            diagnostics.push(Diagnostic::error(
                                "E0511",
                                format!(
                                    "cannot remap colliding Set 0 binding `{}` in `@hook({})` chain: bindings 1..=14 are exhausted",
                                    desc.name,
                                    stage.as_str()
                                ),
                                hook_fn.span,
                            ));
                            return None;
                        };
                        desc.binding = free_b;
                        merged_descriptors.insert((DESCRIPTOR_SET_PER_PASS, free_b), desc);
                    }
                } else {
                    merged_descriptors.insert((desc.set, desc.binding), desc);
                }
            } else {
                merged_descriptors
                    .entry((desc.set, desc.binding))
                    .or_insert(desc);
            }
        }
        for spec in &hook_fn.reflection.used_specs {
            if seen_specs.insert(spec.name.clone()) {
                used_specs.push(spec.clone());
            }
        }
        for param in &hook_fn.reflection.used_params {
            if seen_params.insert(param.name.clone()) {
                used_params.push(param.clone());
            }
        }
        for dbg in &hook_fn.reflection.debug_channels {
            if !debug_channels
                .iter()
                .any(|d: &vsl_mc::DebugChannelBinding| d.channel_name == dbg.channel_name)
            {
                debug_channels.push(dbg.clone());
            }
        }
    }

    // Reserve a free Set 0 output storage image binding (`1..=14`) for writing the final
    // composited hook chain result; never silently overwrite binding 14.
    let Some(out_binding) = (1u32..=14)
        .find(|&b| !merged_descriptors.contains_key(&(DESCRIPTOR_SET_PER_PASS, b)))
    else {
        diagnostics.push(Diagnostic::error(
            "E0511",
            format!(
                "cannot allocate Set 0 output binding for `@hook({})` chain: all bindings 1..=14 are occupied",
                stage.as_str()
            ),
            Span::synthetic(),
        ));
        return None;
    };
    let out_desc_name = SmolStr::new(format!("out_hook_{}", stage.as_str()));
    merged_descriptors.insert(
        (DESCRIPTOR_SET_PER_PASS, out_binding),
        DescriptorBinding {
            name: out_desc_name,
            set: DESCRIPTOR_SET_PER_PASS,
            binding: out_binding,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: None,
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
        },
    );

    let mut unified_descriptors: Vec<DescriptorBinding> =
        merged_descriptors.into_values().collect();
    unified_descriptors.sort_by_key(|d| (d.set, d.binding));
    let world_desc_index = unified_descriptors
        .iter()
        .position(|d| d.set == DESCRIPTOR_SET_PER_PASS && d.binding == 0)
        .unwrap_or(0) as u32;
    let out_desc_index = unified_descriptors
        .iter()
        .position(|d| d.set == DESCRIPTOR_SET_PER_PASS && d.binding == out_binding)
        .unwrap_or(0) as u32;

    // Lower each hook function into a callable non-entry `IrFunction`, mangling colliding helper
    // names per hook and rewriting `FunctionCall` targets.
    let mut callee_fns: Vec<IrFunction> = Vec::with_capacity(lowered_hooks.len());
    let mut helper_fns: Vec<IrFunction> = Vec::new();
    let mut seen_helper_names: IndexSet<SmolStr> = IndexSet::new();
    let mut merged_workgroup_vars = Vec::new();
    let mut seen_workgroup_names: IndexSet<SmolStr> = IndexSet::new();
    let mut merged_ir_push_constants: Option<IrPushConstantBlock> = None;

    for (idx, (hir, hook_fn, items)) in lowered_hooks.iter().enumerate() {
        let (mut primary, mut helpers, hook_diags) = lower_hook_callee_to_ir(
            hir,
            hook_fn,
            items,
            options.pack_schema.as_ref(),
            &unified_descriptors,
        );
        diagnostics.extend(hook_diags);
        if diagnostics.iter().any(Diagnostic::is_error) {
            return None;
        }

        // Validate hook return type against the stage output format.
        if !validate_hook_return_type(stage, &primary.return_ty, hook_fn, diagnostics) {
            return None;
        }

        let mut rename_map: IndexMap<SmolStr, SmolStr> = IndexMap::new();
        for helper in &mut helpers {
            if !seen_helper_names.insert(helper.name.clone()) {
                let mut mangled = SmolStr::new(format!("{}_hook{idx}", helper.name));
                let mut suffix = 1u32;
                while !seen_helper_names.insert(mangled.clone()) {
                    mangled = SmolStr::new(format!("{}_hook{idx}_{suffix}", helper.name));
                    suffix += 1;
                }
                rename_map.insert(helper.name.clone(), mangled.clone());
                helper.name = mangled;
            }
        }
        if !rename_map.is_empty() {
            rewrite_function_call_targets(&mut primary, &rename_map);
            for helper in &mut helpers {
                rewrite_function_call_targets(helper, &rename_map);
            }
        }

        required_caps |= primary.required_caps;
        uses_bda |= primary.uses_bda;
        uses_subgroups |= primary.uses_subgroups;
        uses_quad_control |= primary.uses_quad_control;
        for wg in &primary.workgroup_vars {
            if seen_workgroup_names.insert(wg.name.clone()) {
                merged_workgroup_vars.push(wg.clone());
            }
        }
        if let Some(ref pc) = primary.push_constants {
            let replace = merged_ir_push_constants
                .as_ref()
                .is_none_or(|cur| pc.total_size > cur.total_size);
            if replace {
                merged_ir_push_constants = Some(pc.clone());
            }
        }

        for helper in &helpers {
            required_caps |= helper.required_caps;
            uses_bda |= helper.uses_bda;
            uses_subgroups |= helper.uses_subgroups;
            uses_quad_control |= helper.uses_quad_control;
            for wg in &helper.workgroup_vars {
                if seen_workgroup_names.insert(wg.name.clone()) {
                    merged_workgroup_vars.push(wg.clone());
                }
            }
        }

        primary.name = SmolStr::new(format!("{}_{idx}", hook_fn.name));
        callee_fns.push(primary);
        helper_fns.extend(helpers);
    }

    if uses_quad_control {
        required_caps |= HardwareCaps::QUAD_CONTROL;
    }

    // Build the `vs_hook_chain_<stage>` compute entry point.
    let mut dispatch_fn = IrFunction::new(entry_name.clone());
    dispatch_fn.stage = Some(ShaderStage::Compute);
    dispatch_fn.workgroup_size = workgroup_size;
    dispatch_fn.required_caps = required_caps;
    dispatch_fn.uses_bda = uses_bda;
    dispatch_fn.uses_subgroups = uses_subgroups;
    dispatch_fn.uses_quad_control = uses_quad_control;
    dispatch_fn.workgroup_vars = merged_workgroup_vars;
    dispatch_fn.push_constants = merged_ir_push_constants;
    dispatch_fn.descriptors = unified_descriptors
        .iter()
        .map(|d| IrDescriptorVar {
            name: d.name.clone(),
            set: d.set,
            binding: d.binding,
            kind: d.kind,
            dim: d.dim,
            sample_scalar: d.sample_scalar,
            image_format: d.image_format,
            access: d.access,
            is_unsized: d.is_unsized,
            count: d.count,
            struct_layout: builtin_descriptor_layout(d.set, d.binding).and_then(|desc| {
                match IrType::from_builtin_struct(desc) {
                    IrType::Struct(layout) => Some(layout),
                    _ => None,
                }
            }),
        })
        .collect();

    let mut stage_inputs = StageDispatchInputs::new(&mut dispatch_fn, world_desc_index, out_desc_index);
    let coord = stage_inputs.coord_ivec2;

    let f32_ty = IrType::Scalar(ScalarIrTy::F32);
    let vec3_ty = IrType::Vector {
        scalar: ScalarIrTy::F32,
        len: 3,
    };
    let vec4_ty = IrType::Vector {
        scalar: ScalarIrTy::F32,
        len: 4,
    };

    if stage.uses_alpha_over_blending() {
        // Clouds front-to-back premultiplied alpha-over compositing chain:
        // acc.rgb += (1.0 - acc.a) * src.rgb * src.a
        // acc.a   += (1.0 - acc.a) * src.a
        let zero = dispatch_fn.emit_typed(f32_ty.clone(), InstData::const_f32(0.0));
        let one = dispatch_fn.emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
        let mut acc_rgb =
            dispatch_fn.emit_typed(vec3_ty.clone(), InstData::MakeVec3(zero, zero, zero));
        let mut acc_a = zero;

        for callee in &callee_fns {
            let call_args =
                build_stage_hook_args(&mut dispatch_fn, callee, None, &mut stage_inputs);
            let src = dispatch_fn.emit_typed(
                callee.return_ty.clone(),
                InstData::FunctionCall {
                    callee: callee.name.clone(),
                    args: call_args,
                },
            );
            dispatch_fn.body.push(StructuredStmt::Eval(src));
            let src_f32 = ensure_vec4_f32(&mut dispatch_fn, src);

            let src_r = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src_f32,
                    indices: vec![0],
                },
            );
            let src_g = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src_f32,
                    indices: vec![1],
                },
            );
            let src_b = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src_f32,
                    indices: vec![2],
                },
            );
            let src_a = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src_f32,
                    indices: vec![3],
                },
            );
            let src_rgb =
                dispatch_fn.emit_typed(vec3_ty.clone(), InstData::MakeVec3(src_r, src_g, src_b));
            let transmittance =
                dispatch_fn.emit_typed(f32_ty.clone(), InstData::FSub32(one, acc_a));
            let src_premul = dispatch_fn.emit_typed(
                vec3_ty.clone(),
                InstData::VectorTimesScalar {
                    vec: src_rgb,
                    scalar: src_a,
                },
            );
            let rgb_contrib = dispatch_fn.emit_typed(
                vec3_ty.clone(),
                InstData::VectorTimesScalar {
                    vec: src_premul,
                    scalar: transmittance,
                },
            );
            acc_rgb =
                dispatch_fn.emit_typed(vec3_ty.clone(), InstData::FAdd32(acc_rgb, rgb_contrib));
            let a_contrib =
                dispatch_fn.emit_typed(f32_ty.clone(), InstData::FMul32(transmittance, src_a));
            acc_a = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FAdd32(acc_a, a_contrib));
        }

        let final_r = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: acc_rgb,
                indices: vec![0],
            },
        );
        let final_g = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: acc_rgb,
                indices: vec![1],
            },
        );
        let final_b = dispatch_fn.emit_typed(
            f32_ty,
            InstData::CompositeExtract {
                composite: acc_rgb,
                indices: vec![2],
            },
        );
        let final_rgba = dispatch_fn.emit_typed(
            vec4_ty,
            InstData::MakeVec4(final_r, final_g, final_b, acc_a),
        );
        let store = dispatch_fn.emit_typed(
            IrType::Void,
            InstData::ImageStore {
                descriptor_index: out_desc_index,
                coord,
                value: final_rgba,
            },
        );
        dispatch_fn.body.push(StructuredStmt::Eval(store));
    } else {
        // Sequential state-threading chain for non-clouds native hooks.
        let mut threaded_state: Option<InstId> = None;
        for callee in &callee_fns {
            let call_args = build_stage_hook_args(
                &mut dispatch_fn,
                callee,
                threaded_state,
                &mut stage_inputs,
            );
            let ret_ty = callee.return_ty.clone();
            let res = dispatch_fn.emit_typed(
                ret_ty.clone(),
                InstData::FunctionCall {
                    callee: callee.name.clone(),
                    args: call_args,
                },
            );
            dispatch_fn.body.push(StructuredStmt::Eval(res));
            if ret_ty != IrType::Void {
                threaded_state = Some(res);
            }
        }
        if let Some(final_val) = threaded_state {
            let final_rgba = ensure_vec4_f32(&mut dispatch_fn, final_val);
            let store = dispatch_fn.emit_typed(
                IrType::Void,
                InstData::ImageStore {
                    descriptor_index: out_desc_index,
                    coord,
                    value: final_rgba,
                },
            );
            dispatch_fn.body.push(StructuredStmt::Eval(store));
        }
    }

    let mut functions = Vec::with_capacity(1 + callee_fns.len() + helper_fns.len());
    functions.push(dispatch_fn);
    functions.extend(callee_fns);
    functions.extend(helper_fns);

    let mut ir_mod = IrModule {
        name: SmolStr::new(format!("hook_chain_{}", stage.as_str())),
        functions,
    };
    PassPipeline::default().run_module(&mut ir_mod);
    let spirv = emit_module_spirv(&ir_mod);

    let pack_schema_hash = hash_pack_schema(options.pack_schema.as_ref());
    let used_spec_pairs: Vec<(u32, u32)> = used_specs
        .iter()
        .map(|s| {
            let raw = options.spec_values.get(&s.name).copied().unwrap_or(0);
            (s.spec_id, raw)
        })
        .collect();
    let cache_key = compute_l2_artifact_hash(
        chain_l1,
        entry_name.as_str(),
        options.tier,
        options.active_caps,
        &used_spec_pairs,
        &pack_schema_hash,
        &options.driver_uuid,
    )
    .0;

    Some(CompiledShaderUnit {
        entry_point: entry_name,
        stage: ShaderStage::Compute,
        spirv,
        descriptors: unified_descriptors,
        push_constants: merged_push_constants,
        push_constant_size: merged_push_constant_size,
        workgroup_size,
        required_caps,
        used_specs,
        used_params,
        debug_channels,
        cache_key,
    })
}

fn validate_hook_return_type(
    stage: HookStage,
    return_ty: &IrType,
    hook_fn: &HirFunction,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    if stage.uses_alpha_over_blending() {
        let is_vec4_float = matches!(
            return_ty,
            IrType::Vector {
                scalar: ScalarIrTy::F32 | ScalarIrTy::F16,
                len: 4
            }
        );
        if !is_vec4_float {
            diagnostics.push(Diagnostic::error(
                "E0511",
                format!(
                    "`@hook({})` function `{}` must return `vec4` for alpha-over compositing, got `{return_ty:?}`",
                    stage.as_str(),
                    hook_fn.name
                ),
                hook_fn.span,
            ));
            return false;
        }
        return true;
    }

    let is_compatible = matches!(
        return_ty,
        IrType::Void
            | IrType::Scalar(ScalarIrTy::F32 | ScalarIrTy::F16)
            | IrType::Vector {
                scalar: ScalarIrTy::F32 | ScalarIrTy::F16,
                len: 2..=4
            }
    );
    if !is_compatible {
        diagnostics.push(Diagnostic::error(
            "E0511",
            format!(
                "`@hook({})` function `{}` return type `{return_ty:?}` is incompatible with `rgba16f` stage output",
                stage.as_str(),
                hook_fn.name
            ),
            hook_fn.span,
        ));
        return false;
    }
    true
}

fn rewrite_function_call_targets(ir_fn: &mut IrFunction, rename_map: &IndexMap<SmolStr, SmolStr>) {
    for (_, inst) in ir_fn.insts.iter_mut() {
        if let InstData::FunctionCall { callee, .. } = &mut inst.data
            && let Some(new_name) = rename_map.get(callee)
        {
            *callee = new_name.clone();
        }
    }
}

/// Lazily computed dispatch inputs derived from `GlobalInvocationId`, `ImageQuerySize`, and
/// `WorldUBO` so chained hook callees receive real coordinates, UVs, view directions, and frame
/// state instead of zero-constant stubs.
struct StageDispatchInputs {
    world_desc_index: u32,
    out_desc_index: u32,
    gid_x: InstId,
    gid_uvec3: InstId,
    coord_ivec2: InstId,
    coord_uvec2: InstId,
    uv_vec2: Option<InstId>,
    view_dir_vec3: Option<InstId>,
    camera_pos_vec3: Option<InstId>,
    sun_dir_vec3: Option<InstId>,
    game_time_f32: Option<InstId>,
    rain_f32: Option<InstId>,
    frame_index_u32: Option<InstId>,
}

impl StageDispatchInputs {
    fn new(dispatch_fn: &mut IrFunction, world_desc_index: u32, out_desc_index: u32) -> Self {
        let gid_uvec3 = dispatch_fn.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 3,
            },
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: None,
            },
        );
        let gid_x = dispatch_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: Some(0),
            },
        );
        let gid_y = dispatch_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::LoadBuiltin {
                kind: BuiltinKind::GlobalInvocationId,
                component: Some(1),
            },
        );
        let coord_uvec2 = dispatch_fn.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 2,
            },
            InstData::MakeVec2(gid_x, gid_y),
        );
        let coord_x =
            dispatch_fn.emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::CastToI32(gid_x));
        let coord_y =
            dispatch_fn.emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::CastToI32(gid_y));
        let coord_ivec2 = dispatch_fn.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::I32,
                len: 2,
            },
            InstData::MakeVec2(coord_x, coord_y),
        );
        dispatch_fn.body.push(StructuredStmt::Eval(coord_ivec2));
        Self {
            world_desc_index,
            out_desc_index,
            gid_x,
            gid_uvec3,
            coord_ivec2,
            coord_uvec2,
            uv_vec2: None,
            view_dir_vec3: None,
            camera_pos_vec3: None,
            sun_dir_vec3: None,
            game_time_f32: None,
            rain_f32: None,
            frame_index_u32: None,
        }
    }

    fn uv(&mut self, dispatch_fn: &mut IrFunction) -> InstId {
        if let Some(uv) = self.uv_vec2 {
            return uv;
        }
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let u32_ty = IrType::Scalar(ScalarIrTy::U32);
        let vec2_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 2,
        };
        let half = dispatch_fn.emit_typed(f32_ty.clone(), InstData::const_f32(0.5));
        let ux = dispatch_fn.emit_typed(
            u32_ty.clone(),
            InstData::CompositeExtract {
                composite: self.coord_uvec2,
                indices: vec![0],
            },
        );
        let uy = dispatch_fn.emit_typed(
            u32_ty.clone(),
            InstData::CompositeExtract {
                composite: self.coord_uvec2,
                indices: vec![1],
            },
        );
        let fx = dispatch_fn.emit_typed(f32_ty.clone(), InstData::CastToF32(ux));
        let fy = dispatch_fn.emit_typed(f32_ty.clone(), InstData::CastToF32(uy));
        let px = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FAdd32(fx, half));
        let py = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FAdd32(fy, half));
        let pixel_center = dispatch_fn.emit_typed(vec2_ty.clone(), InstData::MakeVec2(px, py));

        let img_size = dispatch_fn.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 2,
            },
            InstData::ImageQuerySize {
                descriptor_index: self.out_desc_index,
                lod: None,
            },
        );
        let sw = dispatch_fn.emit_typed(
            u32_ty.clone(),
            InstData::CompositeExtract {
                composite: img_size,
                indices: vec![0],
            },
        );
        let sh = dispatch_fn.emit_typed(
            u32_ty,
            InstData::CompositeExtract {
                composite: img_size,
                indices: vec![1],
            },
        );
        let fsw = dispatch_fn.emit_typed(f32_ty.clone(), InstData::CastToF32(sw));
        let fsh = dispatch_fn.emit_typed(f32_ty, InstData::CastToF32(sh));
        let size_f = dispatch_fn.emit_typed(vec2_ty.clone(), InstData::MakeVec2(fsw, fsh));
        let uv = dispatch_fn.emit_typed(vec2_ty, InstData::FDiv32(pixel_center, size_f));
        self.uv_vec2 = Some(uv);
        uv
    }

    fn camera_pos(&mut self, dispatch_fn: &mut IrFunction) -> InstId {
        if let Some(pos) = self.camera_pos_vec3 {
            return pos;
        }
        let pos = load_world_ubo_field(dispatch_fn, self.world_desc_index, "camera_pos_frac");
        self.camera_pos_vec3 = Some(pos);
        pos
    }

    fn sun_dir(&mut self, dispatch_fn: &mut IrFunction) -> InstId {
        if let Some(dir) = self.sun_dir_vec3 {
            return dir;
        }
        let dir = load_world_ubo_field(dispatch_fn, self.world_desc_index, "sun_dir");
        self.sun_dir_vec3 = Some(dir);
        dir
    }

    fn game_time(&mut self, dispatch_fn: &mut IrFunction) -> InstId {
        if let Some(t) = self.game_time_f32 {
            return t;
        }
        let t = load_world_ubo_field(dispatch_fn, self.world_desc_index, "game_time_sec");
        self.game_time_f32 = Some(t);
        t
    }

    fn rain(&mut self, dispatch_fn: &mut IrFunction) -> InstId {
        if let Some(r) = self.rain_f32 {
            return r;
        }
        let r = load_world_ubo_field(dispatch_fn, self.world_desc_index, "rain");
        self.rain_f32 = Some(r);
        r
    }

    fn frame_index(&mut self, dispatch_fn: &mut IrFunction) -> InstId {
        if let Some(f) = self.frame_index_u32 {
            return f;
        }
        let f = load_world_ubo_field(dispatch_fn, self.world_desc_index, "frame_index");
        self.frame_index_u32 = Some(f);
        f
    }

    fn view_dir(&mut self, dispatch_fn: &mut IrFunction) -> InstId {
        if let Some(dir) = self.view_dir_vec3 {
            return dir;
        }
        let uv = self.uv(dispatch_fn);
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let vec3_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 3,
        };
        let vec4_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 4,
        };
        let mat4_ty = IrType::Matrix {
            scalar: ScalarIrTy::F32,
            cols: 4,
            rows: 4,
        };
        let uv_x = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: uv,
                indices: vec![0],
            },
        );
        let uv_y = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: uv,
                indices: vec![1],
            },
        );
        let two = dispatch_fn.emit_typed(f32_ty.clone(), InstData::const_f32(2.0));
        let one = dispatch_fn.emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
        let ndc_x_2 = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FMul32(uv_x, two));
        let ndc_x = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FSub32(ndc_x_2, one));
        let ndc_y_2 = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FMul32(uv_y, two));
        let ndc_y = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FSub32(ndc_y_2, one));
        let ndc =
            dispatch_fn.emit_typed(vec4_ty.clone(), InstData::MakeVec4(ndc_x, ndc_y, one, one));

        let inv_proj = load_world_ubo_field(dispatch_fn, self.world_desc_index, "inv_proj");
        let inv_view = load_world_ubo_field(dispatch_fn, self.world_desc_index, "inv_view");
        let inv_vp = dispatch_fn.emit_typed(
            mat4_ty,
            InstData::MatrixTimesMatrix {
                lhs: inv_view,
                rhs: inv_proj,
            },
        );
        let clip = dispatch_fn.emit_typed(
            vec4_ty,
            InstData::MatrixTimesVector {
                mat: inv_vp,
                vec: ndc,
            },
        );
        let cx = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![0],
            },
        );
        let cy = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![1],
            },
        );
        let cz = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![2],
            },
        );
        let cw = dispatch_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![3],
            },
        );
        let wx = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FDiv32(cx, cw));
        let wy = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FDiv32(cy, cw));
        let wz = dispatch_fn.emit_typed(f32_ty, InstData::FDiv32(cz, cw));
        let world_pos = dispatch_fn.emit_typed(vec3_ty.clone(), InstData::MakeVec3(wx, wy, wz));
        let cam_pos = self.camera_pos(dispatch_fn);
        let diff = dispatch_fn.emit_typed(vec3_ty.clone(), InstData::FSub32(world_pos, cam_pos));
        let dir = dispatch_fn.emit_typed(
            vec3_ty,
            InstData::GlslStd450 {
                op: GlslStd450Op::Normalize,
                args: vec![diff],
            },
        );
        self.view_dir_vec3 = Some(dir);
        dir
    }
}

fn load_world_ubo_field(
    dispatch_fn: &mut IrFunction,
    world_desc_index: u32,
    field_name: &str,
) -> InstId {
    let f_idx = WORLD_UBO_LAYOUT.field_index(field_name).unwrap_or(0);
    let field_ty = WORLD_UBO_LAYOUT
        .field(field_name)
        .map_or(IrType::Scalar(ScalarIrTy::F32), |f| {
            IrType::from_builtin_field_type(f.ty)
        });
    let idx_const =
        dispatch_fn.emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(f_idx));
    let chain = dispatch_fn.emit_typed(
        IrType::Pointer {
            pointee: Box::new(field_ty.clone()),
            align: 4,
            storage: IrStorageClass::Uniform,
        },
        InstData::DescriptorAccessChain {
            descriptor_index: world_desc_index,
            indices: vec![idx_const],
        },
    );
    dispatch_fn.emit_typed(field_ty, InstData::ResourceLoad { ptr: chain })
}

fn build_stage_hook_args(
    dispatch_fn: &mut IrFunction,
    callee: &IrFunction,
    threaded_state: Option<InstId>,
    inputs: &mut StageDispatchInputs,
) -> Vec<InstId> {
    let mut args = Vec::with_capacity(callee.params.len());
    for (idx, param) in callee.params.iter().enumerate() {
        if idx == 0
            && let Some(prev) = threaded_state
            && dispatch_fn.insts[prev].ir_ty == param.ty
        {
            args.push(prev);
            continue;
        }
        let lower_name = param.name.to_ascii_lowercase();
        let val = match &param.ty {
            IrType::Vector {
                scalar: ScalarIrTy::I32,
                len: 2,
            } => inputs.coord_ivec2,
            IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 2,
            } => inputs.coord_uvec2,
            IrType::Vector {
                scalar: ScalarIrTy::U32,
                len: 3,
            } => inputs.gid_uvec3,
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            } => inputs.uv(dispatch_fn),
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 3,
            } => {
                if lower_name.contains("sun") {
                    inputs.sun_dir(dispatch_fn)
                } else if lower_name.contains("pos") || lower_name.contains("origin") {
                    inputs.camera_pos(dispatch_fn)
                } else {
                    inputs.view_dir(dispatch_fn)
                }
            }
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 4,
            } => {
                let dir = inputs.view_dir(dispatch_fn);
                ensure_vec4_f32(dispatch_fn, dir)
            }
            IrType::Scalar(ScalarIrTy::F32) => {
                if lower_name.contains("rain") {
                    inputs.rain(dispatch_fn)
                } else if lower_name.contains("time") {
                    inputs.game_time(dispatch_fn)
                } else {
                    emit_default_ir_value(dispatch_fn, &param.ty)
                }
            }
            IrType::Scalar(ScalarIrTy::U32) => {
                if lower_name.contains("frame") {
                    inputs.frame_index(dispatch_fn)
                } else {
                    inputs.gid_x
                }
            }
            _ => emit_default_ir_value(dispatch_fn, &param.ty),
        };
        args.push(val);
    }
    args
}

fn ensure_vec4_f32(ir_fn: &mut IrFunction, val: InstId) -> InstId {
    let f32_ty = IrType::Scalar(ScalarIrTy::F32);
    let vec4_ty = IrType::Vector {
        scalar: ScalarIrTy::F32,
        len: 4,
    };
    let val_ty = ir_fn.insts[val].ir_ty.clone();
    match val_ty {
        IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 4,
        } => val,
        IrType::Vector { scalar, len: 4 } => {
            let comps: Vec<InstId> = (0u32..4)
                .map(|i| {
                    let c = ir_fn.emit_typed(
                        IrType::Scalar(scalar),
                        InstData::CompositeExtract {
                            composite: val,
                            indices: vec![i],
                        },
                    );
                    if scalar == ScalarIrTy::F16 {
                        ir_fn.emit_typed(
                            f32_ty.clone(),
                            InstData::FConvert {
                                value: c,
                                target: ScalarIrTy::F32,
                            },
                        )
                    } else {
                        c
                    }
                })
                .collect();
            ir_fn.emit_typed(
                vec4_ty,
                InstData::MakeVec4(comps[0], comps[1], comps[2], comps[3]),
            )
        }
        IrType::Vector { scalar, len: 3 } => {
            let one = ir_fn.emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
            let comps: Vec<InstId> = (0u32..3)
                .map(|i| {
                    let c = ir_fn.emit_typed(
                        IrType::Scalar(scalar),
                        InstData::CompositeExtract {
                            composite: val,
                            indices: vec![i],
                        },
                    );
                    if scalar == ScalarIrTy::F16 {
                        ir_fn.emit_typed(
                            f32_ty.clone(),
                            InstData::FConvert {
                                value: c,
                                target: ScalarIrTy::F32,
                            },
                        )
                    } else {
                        c
                    }
                })
                .collect();
            ir_fn.emit_typed(vec4_ty, InstData::MakeVec4(comps[0], comps[1], comps[2], one))
        }
        IrType::Vector { scalar, len: 2 } => {
            let zero = ir_fn.emit_typed(f32_ty.clone(), InstData::const_f32(0.0));
            let one = ir_fn.emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
            let x = ir_fn.emit_typed(
                IrType::Scalar(scalar),
                InstData::CompositeExtract {
                    composite: val,
                    indices: vec![0],
                },
            );
            let y = ir_fn.emit_typed(
                IrType::Scalar(scalar),
                InstData::CompositeExtract {
                    composite: val,
                    indices: vec![1],
                },
            );
            let fx = if scalar == ScalarIrTy::F16 {
                ir_fn.emit_typed(
                    f32_ty.clone(),
                    InstData::FConvert {
                        value: x,
                        target: ScalarIrTy::F32,
                    },
                )
            } else {
                x
            };
            let fy = if scalar == ScalarIrTy::F16 {
                ir_fn.emit_typed(
                    f32_ty,
                    InstData::FConvert {
                        value: y,
                        target: ScalarIrTy::F32,
                    },
                )
            } else {
                y
            };
            ir_fn.emit_typed(vec4_ty, InstData::MakeVec4(fx, fy, zero, one))
        }
        IrType::Scalar(scalar) => {
            let s = if scalar == ScalarIrTy::F16 {
                ir_fn.emit_typed(
                    f32_ty.clone(),
                    InstData::FConvert {
                        value: val,
                        target: ScalarIrTy::F32,
                    },
                )
            } else {
                val
            };
            let one = ir_fn.emit_typed(f32_ty, InstData::const_f32(1.0));
            ir_fn.emit_typed(vec4_ty, InstData::MakeVec4(s, s, s, one))
        }
        _ => val,
    }
}

fn emit_default_ir_value(ir_fn: &mut IrFunction, ty: &IrType) -> InstId {
    match ty {
        IrType::Scalar(ScalarIrTy::F32) => ir_fn.emit_typed(ty.clone(), InstData::const_f32(0.0)),
        IrType::Scalar(ScalarIrTy::F16) => ir_fn.emit_typed(ty.clone(), InstData::ConstF16(0)),
        IrType::Scalar(ScalarIrTy::I32) => ir_fn.emit_typed(ty.clone(), InstData::ConstI32(0)),
        IrType::Scalar(ScalarIrTy::U32) => ir_fn.emit_typed(ty.clone(), InstData::ConstU32(0)),
        IrType::Scalar(ScalarIrTy::I64) => ir_fn.emit_typed(ty.clone(), InstData::ConstI64(0)),
        IrType::Scalar(ScalarIrTy::U64) => ir_fn.emit_typed(ty.clone(), InstData::ConstU64(0)),
        IrType::Scalar(ScalarIrTy::Bool) => {
            ir_fn.emit_typed(ty.clone(), InstData::ConstBool(false))
        }
        IrType::Vector { scalar, len } => {
            let s = emit_default_ir_value(ir_fn, &IrType::Scalar(*scalar));
            match *len {
                2 => ir_fn.emit_typed(ty.clone(), InstData::MakeVec2(s, s)),
                3 => ir_fn.emit_typed(ty.clone(), InstData::MakeVec3(s, s, s)),
                4 => ir_fn.emit_typed(ty.clone(), InstData::MakeVec4(s, s, s, s)),
                _ => ir_fn.emit_typed(
                    ty.clone(),
                    InstData::CompositeConstruct {
                        elems: vec![s; usize::from(*len)],
                    },
                ),
            }
        }
        IrType::Matrix { scalar, cols, rows } => {
            let col_ty = IrType::Vector {
                scalar: *scalar,
                len: *rows,
            };
            let col = emit_default_ir_value(ir_fn, &col_ty);
            ir_fn.emit_typed(
                ty.clone(),
                InstData::CompositeConstruct {
                    elems: vec![col; usize::from(*cols)],
                },
            )
        }
        IrType::Array { element, len } => {
            let elem = emit_default_ir_value(ir_fn, element);
            ir_fn.emit_typed(
                ty.clone(),
                InstData::CompositeConstruct {
                    elems: vec![elem; *len as usize],
                },
            )
        }
        IrType::Struct(layout) => {
            let elems: Vec<InstId> = layout
                .fields
                .iter()
                .map(|f| emit_default_ir_value(ir_fn, &f.ty))
                .collect();
            ir_fn.emit_typed(ty.clone(), InstData::CompositeConstruct { elems })
        }
        _ => ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0)),
    }
}
