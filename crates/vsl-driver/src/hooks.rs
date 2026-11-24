//! Native SPIR-V `@hook` chain dispatch synthesis and `HookStage::Clouds` alpha-over
//! compositing (Item 23).

use crate::cache::hash_pack_schema;
use crate::lower_ir::{build_unified_descriptors, lower_hook_callee_to_ir};
use indexmap::IndexMap;
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, SourceDatabase, Span};
use vsl_hir::module::{L1ModuleHash, compute_l2_artifact_hash};
use vsl_hir::{HirFunction, HirModule, lower_unit_with_schema};
use vsl_ir::{
    BuiltinKind, InstData, InstId, IrDescriptorVar, IrFunction, IrModule, IrType, PassPipeline,
    ScalarIrTy, StructuredStmt,
};
use vsl_mc::{
    CompiledShaderUnit, DESCRIPTOR_SET_PER_PASS, DescriptorAccess, DescriptorBinding,
    DescriptorKind, HardwareCaps, HookAbiRoute, HookStage, ImageFormat, ShaderStage, TextureDim,
    builtin_descriptor_layout,
};
use vsl_spirv::emit_module_spirv;
use vsl_syntax::ast::Item;
use vsl_syntax::parse_source;

use crate::CompileOptions;

/// Compiles an ordered chain of native SPIR-V `@hook` functions for `stage` into a single
/// dispatch entry point `vs_hook_chain_<stage>`.
///
/// For `HookStage::Clouds` (`stage.uses_alpha_over_blending()`), emits exact front-to-back /
/// Porter-Duff alpha-over compositing between each layer:
/// `acc.rgb = src.rgb * src.a + acc.rgb * (1.0 - src.a)` and
/// `acc.a = src.a + acc.a * (1.0 - src.a)`, then writes the final `vec4` to the output target.
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
            });

        if let Some(hook_fn) = matched {
            lowered_hooks.push((hir, hook_fn.clone(), parsed.unit.items));
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
    build_hook_chain_unit(
        stage,
        &lowered_hooks,
        chain_l1,
        options,
        &mut diagnostics,
    )
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

    // Merge descriptors, caps, specs, params, and debug channels across all chained hooks.
    let mut merged_descriptors: IndexMap<(u32, u32), DescriptorBinding> = IndexMap::new();
    let mut required_caps = HardwareCaps::empty();
    let mut used_specs = Vec::new();
    let mut seen_specs = indexmap::IndexSet::new();
    let mut used_params = Vec::new();
    let mut seen_params = indexmap::IndexSet::new();
    let mut debug_channels = Vec::new();
    let mut uses_bda = false;
    let mut uses_subgroups = false;

    for (_, hook_fn, _) in lowered_hooks {
        required_caps |= hook_fn.required_caps;
        uses_bda |= hook_fn.reflection.uses_bda;
        uses_subgroups |= hook_fn.reflection.uses_subgroups;
        for desc in build_unified_descriptors(hook_fn) {
            merged_descriptors
                .entry((desc.set, desc.binding))
                .or_insert(desc);
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

    // Reserve a Set 0 output storage image binding (`1..=14`) for writing the final composited
    // hook chain result.
    let out_binding = (1u32..=14)
        .find(|&b| !merged_descriptors.contains_key(&(DESCRIPTOR_SET_PER_PASS, b)))
        .unwrap_or(14);
    let out_desc_name = SmolStr::new(format!("out_hook_{}", stage.as_str()));
    merged_descriptors
        .entry((DESCRIPTOR_SET_PER_PASS, out_binding))
        .or_insert_with(|| DescriptorBinding {
            name: out_desc_name.clone(),
            set: DESCRIPTOR_SET_PER_PASS,
            binding: out_binding,
            kind: DescriptorKind::StorageImage,
            dim: Some(TextureDim::Dim2D),
            sample_scalar: None,
            image_format: Some(ImageFormat::Rgba16f),
            access: DescriptorAccess::WriteOnly,
            is_unsized: false,
            count: None,
        });

    let mut unified_descriptors: Vec<DescriptorBinding> =
        merged_descriptors.into_values().collect();
    unified_descriptors.sort_by_key(|d| (d.set, d.binding));
    let out_desc_index = unified_descriptors
        .iter()
        .position(|d| d.set == DESCRIPTOR_SET_PER_PASS && d.binding == out_binding)
        .unwrap_or(0) as u32;

    // Lower each hook function into a callable non-entry `IrFunction`.
    let mut callee_fns: Vec<IrFunction> = Vec::with_capacity(lowered_hooks.len());
    let mut helper_fns: Vec<IrFunction> = Vec::new();
    let mut seen_helper_names: indexmap::IndexSet<SmolStr> = indexmap::IndexSet::new();

    for (idx, (hir, hook_fn, items)) in lowered_hooks.iter().enumerate() {
        let (mut primary, helpers, hook_diags) = lower_hook_callee_to_ir(
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
        primary.name = SmolStr::new(format!("{}_{idx}", hook_fn.name));
        callee_fns.push(primary);
        for helper in helpers {
            if seen_helper_names.insert(helper.name.clone()) {
                helper_fns.push(helper);
            }
        }
    }

    // Build the `vs_hook_chain_<stage>` compute entry point.
    let mut dispatch_fn = IrFunction::new(entry_name.clone());
    dispatch_fn.stage = Some(ShaderStage::Compute);
    dispatch_fn.workgroup_size = workgroup_size;
    dispatch_fn.required_caps = required_caps;
    dispatch_fn.uses_bda = uses_bda;
    dispatch_fn.uses_subgroups = uses_subgroups;
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
    let coord_x = dispatch_fn.emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::CastToI32(gid_x));
    let coord_y = dispatch_fn.emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::CastToI32(gid_y));
    let coord = dispatch_fn.emit_typed(
        IrType::Vector {
            scalar: ScalarIrTy::I32,
            len: 2,
        },
        InstData::MakeVec2(coord_x, coord_y),
    );
    dispatch_fn.body.push(StructuredStmt::Eval(coord));

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
        // Clouds alpha-over compositing chain:
        // acc.rgb = src.rgb * src.a + acc.rgb * (1.0 - src.a)
        // acc.a   = src.a + acc.a * (1.0 - src.a)
        let zero = dispatch_fn.emit_typed(f32_ty.clone(), InstData::const_f32(0.0));
        let one = dispatch_fn.emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
        let mut acc_rgb =
            dispatch_fn.emit_typed(vec3_ty.clone(), InstData::MakeVec3(zero, zero, zero));
        let mut acc_a = zero;

        for callee in &callee_fns {
            let call_args = build_default_hook_args(&mut dispatch_fn, callee, None);
            let src = dispatch_fn.emit_typed(
                vec4_ty.clone(),
                InstData::FunctionCall {
                    callee: callee.name.clone(),
                    args: call_args,
                },
            );
            dispatch_fn.body.push(StructuredStmt::Eval(src));

            let src_r = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src,
                    indices: vec![0],
                },
            );
            let src_g = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src,
                    indices: vec![1],
                },
            );
            let src_b = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src,
                    indices: vec![2],
                },
            );
            let src_a = dispatch_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: src,
                    indices: vec![3],
                },
            );
            let src_rgb = dispatch_fn.emit_typed(
                vec3_ty.clone(),
                InstData::MakeVec3(src_r, src_g, src_b),
            );
            let inv_a = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FSub32(one, src_a));
            let src_premul = dispatch_fn.emit_typed(
                vec3_ty.clone(),
                InstData::VectorTimesScalar {
                    vec: src_rgb,
                    scalar: src_a,
                },
            );
            let acc_scaled = dispatch_fn.emit_typed(
                vec3_ty.clone(),
                InstData::VectorTimesScalar {
                    vec: acc_rgb,
                    scalar: inv_a,
                },
            );
            acc_rgb = dispatch_fn.emit_typed(vec3_ty.clone(), InstData::FAdd32(src_premul, acc_scaled));
            let acc_a_scaled = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FMul32(acc_a, inv_a));
            acc_a = dispatch_fn.emit_typed(f32_ty.clone(), InstData::FAdd32(src_a, acc_a_scaled));
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
            let call_args = build_default_hook_args(&mut dispatch_fn, callee, threaded_state);
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
            let store = dispatch_fn.emit_typed(
                IrType::Void,
                InstData::ImageStore {
                    descriptor_index: out_desc_index,
                    coord,
                    value: final_val,
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
        push_constants: Vec::new(),
        push_constant_size: 0,
        workgroup_size,
        required_caps,
        used_specs,
        used_params,
        debug_channels,
        cache_key,
    })
}

fn build_default_hook_args(
    dispatch_fn: &mut IrFunction,
    callee: &IrFunction,
    threaded_state: Option<InstId>,
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
        args.push(emit_default_ir_value(dispatch_fn, &param.ty));
    }
    args
}

fn emit_default_ir_value(ir_fn: &mut IrFunction, ty: &IrType) -> InstId {
    match ty {
        IrType::Scalar(ScalarIrTy::F32) => ir_fn.emit_typed(ty.clone(), InstData::const_f32(0.0)),
        IrType::Scalar(ScalarIrTy::F16) => ir_fn.emit_typed(ty.clone(), InstData::ConstF16(0)),
        IrType::Scalar(ScalarIrTy::I32) => ir_fn.emit_typed(ty.clone(), InstData::ConstI32(0)),
        IrType::Scalar(ScalarIrTy::U32) => ir_fn.emit_typed(ty.clone(), InstData::ConstU32(0)),
        IrType::Scalar(ScalarIrTy::I64) => ir_fn.emit_typed(ty.clone(), InstData::ConstI64(0)),
        IrType::Scalar(ScalarIrTy::U64) => ir_fn.emit_typed(ty.clone(), InstData::ConstU64(0)),
        IrType::Scalar(ScalarIrTy::Bool) => ir_fn.emit_typed(ty.clone(), InstData::ConstBool(false)),
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

