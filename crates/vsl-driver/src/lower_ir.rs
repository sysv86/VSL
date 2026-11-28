//! Typed HIR-to-SSA IR lowering (`vsl-hir` -> `vsl-ir`) with zero stubs or type heuristics.
//!
//! Lowers `HirFunction::body_hir` directly using resolved `TypeId` -> `IrType` conversions,
//! emits `IrLocalVar` + `LocalAddr`/`LocalLoad`/`LocalStore`/`AccessChain` for mutable locals
//! and composites (promoted to SSA by `SroaPass` + `Mem2RegPass`), lowers helper functions into
//! separate `IrFunction`s with `IrParam`/`FunctionCall`, uses `vsl_mc::BuiltinStructDesc` for
//! all built-in descriptor field accesses, and implements the full `mc::*` built-in library.

use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, Span};
use vsl_hir::{
    HirBlock, HirCallArg, HirExpr, HirExprKind, HirFnId, HirFunction, HirModule, HirStmt,
    LayoutKind, PackSchema, Type, TypeId,
};
use vsl_ir::{
    AtomicOrder, AtomicScope, BuiltinKind, CoopMatrixLayout, GlslStd450Op, InstData, InstId,
    IrDescriptorVar,
    IrFunction, IrInterlockMode, SubgroupClusteredOp, IrInterpolation, IrLocalVar, IrModule, IrParam,
    IrPushConstantBlock,
    IrPushConstantField, IrStageExecMeta, IrStageIoVar, IrStageVar, IrStorageClass,
    IrStructFieldLayout, IrStructLayout, IrTessDomain, IrTessOrder, IrTessSpacing, IrTopology,
    IrType, IrWorkgroupVar, MemoryAccess, ScalarIrTy, StructuredStmt,
};
use vsl_mc::{
    BuiltinStructDesc, DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS,
    DESCRIPTOR_SET_SCENE_GLOBALS, DescriptorAccess, DescriptorBinding, DescriptorKind,
    EMISSIVE_LIGHT_LAYOUT, EMISSIVE_LIGHT_SSBO_LAYOUT, INSTANCE_META_LAYOUT,
    INSTANCE_META_SSBO_LAYOUT, ImageFormat, PackParamLayout, PackParamType,
    RESTIR_RESERVOIR_LAYOUT, RESTIR_RESERVOIR_SSBO_LAYOUT, ShaderStage, SpecConstantDefault,
    TextureDim, WORLD_UBO_LAYOUT, builtin_descriptor_layout,
};
use vsl_syntax::ast::{
    Attribute, BinaryOp, Block, Expr, FnDecl, Item, ParamMode, Stmt, TypeArg, TypeExpr, UnaryOp,
};

/// Result of lowering an entry point (plus its module helper functions) into an [`IrModule`].
pub(crate) struct LoweredEntryModule {
    pub(crate) ir_module: IrModule,
    pub(crate) unified_descriptors: Vec<DescriptorBinding>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

/// Lowers a validated entry-point [`HirFunction`] and all concrete non-entry helper functions in
/// `hir` into a typed [`IrModule`].
pub(crate) fn lower_entry_module_to_ir(
    hir: &HirModule,
    entry_fn: &HirFunction,
    module_items: &[Item],
    pack_schema: Option<&PackSchema>,
    stage: ShaderStage,
    workgroup_size: Option<[u32; 3]>,
) -> LoweredEntryModule {
    let mut diagnostics = Vec::new();
    let unified_descriptors = build_unified_descriptors_with_helpers(hir, entry_fn);

    let entry_ir_fn = lower_single_function(
        hir,
        entry_fn,
        module_items,
        pack_schema,
        &unified_descriptors,
        &entry_fn.reflection.descriptor_types,
        Some(stage),
        workgroup_size,
        stage.allows_workgroup_vars(),
        None,
        &mut diagnostics,
    );
    let inherited_push_constants = entry_ir_fn.push_constants.clone();

    let mut functions = vec![entry_ir_fn];

    // Lower every concrete non-entry helper function into a first-class `IrFunction` so
    // `FunctionCall` instructions target real `IrModule::functions` entries and `SelectiveInlinePass`
    // can inline small helpers cleanly.
    for (_, helper) in hir.functions.iter() {
        if helper.name == entry_fn.name
            || helper.stage.is_some()
            || helper.hook.is_some()
            || !helper.generic_params.is_empty()
            || helper.body_hir.is_none()
        {
            continue;
        }
        let helper_ir = lower_single_function(
            hir,
            helper,
            module_items,
            pack_schema,
            &unified_descriptors,
            &entry_fn.reflection.descriptor_types,
            None,
            None,
            stage.allows_workgroup_vars(),
            inherited_push_constants.as_ref(),
            &mut diagnostics,
        );
        functions.push(helper_ir);
    }

    LoweredEntryModule {
        ir_module: IrModule {
            name: hir.module_name.clone(),
            functions,
        },
        unified_descriptors,
        diagnostics,
    }
}

/// Collects all module helper functions transitively reachable from `root` via call/method
/// expressions in `body_hir`.
pub(crate) fn collect_reachable_helpers<'a>(
    hir: &'a HirModule,
    root: &'a HirFunction,
) -> Vec<&'a HirFunction> {
    let mut visited: IndexSet<HirFnId> = IndexSet::new();
    let mut queue: std::collections::VecDeque<&'a HirFunction> = std::collections::VecDeque::new();
    queue.push_back(root);

    let mut helpers = Vec::new();
    while let Some(cur) = queue.pop_front() {
        let Some(ref body) = cur.body_hir else {
            continue;
        };
        let mut called_ids: Vec<HirFnId> = Vec::new();
        collect_called_fns_in_block(hir, body, &mut called_ids);
        for fn_id in called_ids {
            if visited.insert(fn_id) {
                let target = &hir.functions[fn_id];
                if target.name != root.name {
                    helpers.push(target);
                    queue.push_back(target);
                }
            }
        }
    }
    helpers
}

fn collect_called_fns_in_block(hir: &HirModule, block: &HirBlock, out: &mut Vec<HirFnId>) {
    for stmt in &block.stmts {
        collect_called_fns_in_stmt(hir, stmt, out);
    }
    if let Some(ref tail) = block.tail_expr {
        collect_called_fns_in_expr(hir, tail, out);
    }
}

fn collect_called_fns_in_stmt(hir: &HirModule, stmt: &HirStmt, out: &mut Vec<HirFnId>) {
    match stmt {
        HirStmt::Let { init: Some(e), .. }
        | HirStmt::Return { value: Some(e), .. }
        | HirStmt::Expr { expr: e, .. } => collect_called_fns_in_expr(hir, e, out),
        HirStmt::For { iter, body, .. } => {
            collect_called_fns_in_expr(hir, iter, out);
            collect_called_fns_in_block(hir, body, out);
        }
        HirStmt::While { cond, body, .. } => {
            collect_called_fns_in_expr(hir, cond, out);
            collect_called_fns_in_block(hir, body, out);
        }
        HirStmt::Loop { body, .. } => collect_called_fns_in_block(hir, body, out),
        HirStmt::Switch {
            selector,
            cases,
            default,
            ..
        } => {
            collect_called_fns_in_expr(hir, selector, out);
            for c in cases {
                collect_called_fns_in_expr(hir, &c.value, out);
                collect_called_fns_in_block(hir, &c.body, out);
            }
            if let Some(d) = default {
                collect_called_fns_in_block(hir, d, out);
            }
        }
        HirStmt::Let { init: None, .. }
        | HirStmt::Return { value: None, .. }
        | HirStmt::Break(_)
        | HirStmt::Continue(_) => {}
    }
}

fn collect_called_fns_in_expr(hir: &HirModule, expr: &HirExpr, out: &mut Vec<HirFnId>) {
    match &expr.kind {
        HirExprKind::Call {
            callee,
            resolved_fn,
            args,
        } => {
            if let Some(fn_id) = *resolved_fn {
                out.push(fn_id);
            } else if let HirExprKind::Path { path, .. } = &callee.kind {
                let short = path.rsplit("::").next().unwrap_or(path.as_str());
                if let Some((fn_id, _)) = hir.functions.iter().find(|(_, f)| f.name == short) {
                    out.push(fn_id);
                }
            }
            collect_called_fns_in_expr(hir, callee, out);
            for a in args {
                collect_called_fns_in_expr(hir, &a.value, out);
            }
        }
        HirExprKind::MethodCall {
            receiver,
            resolved_fn,
            args,
            ..
        } => {
            if let Some(fn_id) = *resolved_fn {
                out.push(fn_id);
            }
            collect_called_fns_in_expr(hir, receiver, out);
            for a in args {
                collect_called_fns_in_expr(hir, &a.value, out);
            }
        }
        HirExprKind::ArrayLit { elements } => {
            for e in elements {
                collect_called_fns_in_expr(hir, e, out);
            }
        }
        HirExprKind::StructLit { fields, .. } => {
            for (_, e) in fields {
                collect_called_fns_in_expr(hir, e, out);
            }
        }
        HirExprKind::Unary { operand, .. }
        | HirExprKind::Field {
            receiver: operand, ..
        }
        | HirExprKind::Swizzle { base: operand, .. } => {
            collect_called_fns_in_expr(hir, operand, out);
        }
        HirExprKind::Binary { lhs, rhs, .. } => {
            collect_called_fns_in_expr(hir, lhs, out);
            collect_called_fns_in_expr(hir, rhs, out);
        }
        HirExprKind::Index { receiver, index } => {
            collect_called_fns_in_expr(hir, receiver, out);
            collect_called_fns_in_expr(hir, index, out);
        }
        HirExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_called_fns_in_expr(hir, cond, out);
            collect_called_fns_in_block(hir, then_branch, out);
            if let Some(eb) = else_branch {
                collect_called_fns_in_expr(hir, eb, out);
            }
        }
        HirExprKind::Block(b) => collect_called_fns_in_block(hir, b, out),
        HirExprKind::IntLit(_)
        | HirExprKind::FloatLit(_)
        | HirExprKind::BoolLit(_)
        | HirExprKind::StringLit(_)
        | HirExprKind::NoneLit
        | HirExprKind::Path { .. } => {}
    }
}

/// Combines regular reflected descriptors and debug-channel storage images into one sorted,
/// single-source-of-truth Vulkan descriptor list.
pub(crate) fn build_unified_descriptors(func: &HirFunction) -> Vec<DescriptorBinding> {
    let mut unified = func.reflection.descriptors.clone();
    for dbg in &func.reflection.debug_channels {
        let already_present = unified
            .iter()
            .any(|d| d.set == dbg.set && d.binding == dbg.binding);
        if !already_present {
            unified.push(DescriptorBinding {
                name: SmolStr::new(format!("debug_{}", dbg.channel_name)),
                set: dbg.set,
                binding: dbg.binding,
                kind: DescriptorKind::StorageImage,
                dim: Some(TextureDim::Dim2D),
                sample_scalar: None,
                image_format: Some(ImageFormat::Rgba16f),
                access: DescriptorAccess::WriteOnly,
                is_unsized: false,
                count: None,
            });
        }
    }
    unified.sort_by_key(|d| (d.set, d.binding));
    unified
}

/// Combines reflected descriptors and debug-channel storage images from `entry_fn` and all
/// transitively reachable helper functions into one sorted descriptor list.
pub(crate) fn build_unified_descriptors_with_helpers(
    hir: &HirModule,
    entry_fn: &HirFunction,
) -> Vec<DescriptorBinding> {
    let mut unified = build_unified_descriptors(entry_fn);
    for helper in collect_reachable_helpers(hir, entry_fn) {
        for desc in build_unified_descriptors(helper) {
            let already_present = unified
                .iter()
                .any(|d| d.set == desc.set && d.binding == desc.binding);
            if !already_present {
                unified.push(desc);
            }
        }
    }
    unified.sort_by_key(|d| (d.set, d.binding));
    unified
}

fn ir_type_byte_stride(ty: &IrType) -> u32 {
    match ty {
        IrType::Scalar(s) => s.byte_size(),
        IrType::Vector { scalar, len: 3 } => scalar.byte_size().saturating_mul(4),
        IrType::Vector { scalar, len } => scalar.byte_size().saturating_mul(u32::from(*len)),
        IrType::Matrix { scalar, cols, rows } => {
            let col_stride = if *rows == 3 {
                scalar.byte_size().saturating_mul(4)
            } else {
                scalar.byte_size().saturating_mul(u32::from(*rows))
            };
            col_stride.saturating_mul(u32::from(*cols))
        }
        IrType::Array { element, len } => {
            ir_type_byte_stride(element).saturating_mul((*len).max(1))
        }
        IrType::Struct(layout) => {
            let align = layout.align.max(1);
            let aligned = layout.size.div_ceil(align).saturating_mul(align);
            aligned.max(align).max(4)
        }
        IrType::Pointer {
            storage: IrStorageClass::PhysicalStorageBuffer,
            ..
        } => 8,
        IrType::Void | IrType::RuntimeArray(_) | IrType::Pointer { .. } | IrType::CoopMatrix { .. } => 4,
    }
}

fn user_descriptor_struct_layout(
    hir: &HirModule,
    name: &str,
    ty_id: TypeId,
) -> Option<IrStructLayout> {
    match hir.types.get(ty_id) {
        Type::Buffer { element } | Type::RwBuffer { element } => {
            let elem_ir = IrType::from_hir_type(&hir.types, *element);
            let elem_stride = ir_type_byte_stride(&elem_ir);
            Some(IrStructLayout {
                name: SmolStr::new(format!("{name}Buffer")),
                fields: vec![IrStructFieldLayout {
                    name: SmolStr::new_static("items"),
                    byte_offset: 0,
                    byte_size: elem_stride,
                    ty: IrType::RuntimeArray(Box::new(elem_ir)),
                }],
                size: 0,
                align: 4,
                layout: LayoutKind::Std430,
            })
        }
        Type::Struct { .. } => match IrType::from_hir_type(&hir.types, ty_id) {
            IrType::Struct(layout) => Some(layout),
            _ => None,
        },
        _ => None,
    }
}

fn descriptor_binding_to_ir(
    d: &DescriptorBinding,
    hir: &HirModule,
    descriptor_types: &IndexMap<(u32, u32), TypeId>,
) -> IrDescriptorVar {
    let struct_layout = descriptor_types
        .get(&(d.set, d.binding))
        .and_then(|&ty_id| user_descriptor_struct_layout(hir, d.name.as_str(), ty_id))
        .or_else(|| {
            builtin_descriptor_layout(d.set, d.binding).and_then(|desc| {
                match IrType::from_builtin_struct(desc) {
                    IrType::Struct(layout) => Some(layout),
                    _ => None,
                }
            })
        });
    IrDescriptorVar {
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
        struct_layout,
    }
}

/// Lowers a single `@hook` function as a callable non-entry `IrFunction` bound against
/// the chain's unified descriptor table so `OpReturnValue` and descriptor indices match.
pub(crate) fn lower_hook_callee_to_ir(
    hir: &HirModule,
    func: &HirFunction,
    module_items: &[Item],
    pack_schema: Option<&PackSchema>,
    unified_descriptors: &[DescriptorBinding],
) -> (IrFunction, Vec<IrFunction>, Vec<Diagnostic>) {
    let mut diagnostics = Vec::new();
    let primary = lower_single_function(
        hir,
        func,
        module_items,
        pack_schema,
        unified_descriptors,
        &func.reflection.descriptor_types,
        None,
        None,
        true,
        None,
        &mut diagnostics,
    );
    let inherited_push_constants = primary.push_constants.clone();
    let mut helpers = Vec::new();
    for (_, helper) in hir.functions.iter() {
        if helper.name == func.name
            || helper.stage.is_some()
            || helper.hook.is_some()
            || !helper.generic_params.is_empty()
            || helper.body_hir.is_none()
        {
            continue;
        }
        helpers.push(lower_single_function(
            hir,
            helper,
            module_items,
            pack_schema,
            unified_descriptors,
            &func.reflection.descriptor_types,
            None,
            None,
            true,
            inherited_push_constants.as_ref(),
            &mut diagnostics,
        ));
    }
    (primary, helpers, diagnostics)
}

#[allow(clippy::too_many_arguments)]
fn lower_single_function(
    hir: &HirModule,
    func: &HirFunction,
    module_items: &[Item],
    pack_schema: Option<&PackSchema>,
    unified_descriptors: &[DescriptorBinding],
    descriptor_types: &IndexMap<(u32, u32), TypeId>,
    stage: Option<ShaderStage>,
    workgroup_size: Option<[u32; 3]>,
    allow_workgroup_vars: bool,
    inherited_push_constants: Option<&IrPushConstantBlock>,
    diagnostics: &mut Vec<Diagnostic>,
) -> IrFunction {
    let mut ir_fn = IrFunction::new(func.name.clone());
    ir_fn.stage = stage;
    ir_fn.workgroup_size = workgroup_size;
    ir_fn.required_caps = func.required_caps;
    ir_fn.uses_bda = func.reflection.uses_bda;
    ir_fn.uses_subgroups = func.reflection.uses_subgroups;
    ir_fn.return_ty = func
        .return_ty
        .map_or(IrType::Void, |id| IrType::from_hir_type(&hir.types, id));

    ir_fn.descriptors = unified_descriptors
        .iter()
        .map(|d| descriptor_binding_to_ir(d, hir, descriptor_types))
        .collect();

    if allow_workgroup_vars {
        for wg in &hir.workgroup_vars {
            let ty = IrType::from_hir_type(&hir.types, wg.ty);
            ir_fn
                .workgroup_vars
                .push(IrWorkgroupVar::typed(wg.name.clone(), ty));
        }
    }

    if func.reflection.push_constant_size > 0 {
        let pc_struct_fields = find_push_constant_struct_fields(hir, func);
        ir_fn.push_constants = Some(IrPushConstantBlock {
            name: SmolStr::new(format!("{}_push", func.name)),
            fields: func
                .reflection
                .push_constants
                .iter()
                .map(|f| {
                    let scalar = pc_struct_fields
                        .get(&f.name)
                        .copied()
                        .unwrap_or(ScalarIrTy::U32);
                    IrPushConstantField {
                        name: f.name.clone(),
                        byte_offset: f.byte_offset,
                        byte_size: f.byte_size,
                        scalar,
                    }
                })
                .collect(),
            total_size: func.reflection.push_constant_size,
        });
    } else if let Some(pc) = inherited_push_constants {
        ir_fn.push_constants = Some(pc.clone());
    }

    let ast_fn = find_ast_function_any(module_items, &func.name);
    let ast_hints = collect_ast_hints(func.body_ast.as_ref());

    let mut ctx = LowerCtx::new(
        hir,
        func,
        ast_fn,
        pack_schema,
        ast_hints,
        stage,
        &mut ir_fn,
        diagnostics,
    );

    if let Some(ref body_hir) = func.body_hir {
        let mut stmts = std::mem::take(&mut ctx.entry_preamble);
        let (mut lowered, tail) = ctx.lower_block_value(body_hir);
        stmts.append(&mut lowered);
        if let Some(val) = tail {
            if stage.is_none() && ir_fn.return_ty != IrType::Void {
                stmts.push(StructuredStmt::Return(Some(val)));
            } else {
                stmts.push(StructuredStmt::Eval(val));
            }
        }
        ir_fn.body = stmts;
    } else if stage.is_some() {
        diagnostics.push(Diagnostic::error(
            "E0520",
            format!("entry point `{}` has no lowered HIR body", func.name),
            func.span,
        ));
    }

    ir_fn
}

/// Auxiliary metadata preserved from the AST body (loop `#[unroll]` attributes and atomic
/// `<Scope::*, Order::*>` turbofish arguments, which are not carried as `TypeId`s in HIR).
#[derive(Default)]
struct AstHints {
    unroll_spans: IndexSet<Span>,
    atomic_turbofish: IndexMap<Span, (AtomicScope, AtomicOrder)>,
}

fn collect_ast_hints(body: Option<&Block>) -> AstHints {
    let mut hints = AstHints::default();
    if let Some(block) = body {
        collect_ast_hints_block(block, &mut hints);
    }
    hints
}

fn has_unroll_attr(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.entries.iter().any(|e| e.name.name == "unroll"))
}

fn collect_ast_hints_block(block: &Block, hints: &mut AstHints) {
    for stmt in &block.stmts {
        collect_ast_hints_stmt(stmt, hints);
    }
    if let Some(ref tail) = block.tail_expr {
        collect_ast_hints_expr(tail, hints);
    }
}

fn collect_ast_hints_stmt(stmt: &Stmt, hints: &mut AstHints) {
    match stmt {
        Stmt::Let { init: Some(e), .. } | Stmt::Return { value: Some(e), .. } => {
            collect_ast_hints_expr(e, hints);
        }
        Stmt::For {
            attrs,
            iter,
            body,
            span,
            ..
        } => {
            if has_unroll_attr(attrs) {
                hints.unroll_spans.insert(*span);
            }
            collect_ast_hints_expr(iter, hints);
            collect_ast_hints_block(body, hints);
        }
        Stmt::While {
            attrs,
            cond,
            body,
            span,
            ..
        } => {
            if has_unroll_attr(attrs) {
                hints.unroll_spans.insert(*span);
            }
            collect_ast_hints_expr(cond, hints);
            collect_ast_hints_block(body, hints);
        }
        Stmt::Loop {
            attrs, body, span, ..
        } => {
            if has_unroll_attr(attrs) {
                hints.unroll_spans.insert(*span);
            }
            collect_ast_hints_block(body, hints);
        }
        Stmt::Switch {
            selector,
            cases,
            default,
            ..
        } => {
            collect_ast_hints_expr(selector, hints);
            for c in cases {
                collect_ast_hints_block(&c.body, hints);
            }
            if let Some(d) = default {
                collect_ast_hints_block(d, hints);
            }
        }
        Stmt::Expr { expr, .. } => collect_ast_hints_expr(expr, hints),
        Stmt::Let { init: None, .. }
        | Stmt::Return { value: None, .. }
        | Stmt::Break(_)
        | Stmt::Continue(_) => {}
    }
}

fn collect_ast_hints_expr(expr: &Expr, hints: &mut AstHints) {
    match expr {
        Expr::Path {
            turbofish, span, ..
        } if !turbofish.is_empty() => {
            hints
                .atomic_turbofish
                .insert(*span, parse_atomic_turbofish(turbofish));
        }
        Expr::Call {
            callee, args, span, ..
        } => {
            if let Expr::Path { turbofish, .. } = callee.as_ref()
                && !turbofish.is_empty()
            {
                hints
                    .atomic_turbofish
                    .insert(*span, parse_atomic_turbofish(turbofish));
            }
            collect_ast_hints_expr(callee, hints);
            for a in args {
                collect_ast_hints_expr(&a.value, hints);
            }
        }
        Expr::MethodCall {
            receiver,
            turbofish,
            args,
            span,
            ..
        } => {
            if !turbofish.is_empty() {
                hints
                    .atomic_turbofish
                    .insert(*span, parse_atomic_turbofish(turbofish));
            }
            collect_ast_hints_expr(receiver, hints);
            for a in args {
                collect_ast_hints_expr(&a.value, hints);
            }
        }
        Expr::Unary { operand, .. }
        | Expr::Field {
            receiver: operand, ..
        } => collect_ast_hints_expr(operand, hints),
        Expr::Binary { lhs, rhs, .. } => {
            collect_ast_hints_expr(lhs, hints);
            collect_ast_hints_expr(rhs, hints);
        }
        Expr::Index {
            receiver, index, ..
        } => {
            collect_ast_hints_expr(receiver, hints);
            collect_ast_hints_expr(index, hints);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_ast_hints_expr(cond, hints);
            collect_ast_hints_block(then_branch, hints);
            if let Some(eb) = else_branch {
                collect_ast_hints_expr(eb, hints);
            }
        }
        Expr::Block(b) => collect_ast_hints_block(b, hints),
        Expr::ArrayLit { elements, .. } => {
            for e in elements {
                collect_ast_hints_expr(e, hints);
            }
        }
        Expr::StructLit { fields, .. } => {
            for (_, e) in fields {
                collect_ast_hints_expr(e, hints);
            }
        }
        _ => {}
    }
}

fn find_ast_function_any<'a>(items: &'a [Item], name: &str) -> Option<&'a FnDecl> {
    for item in items {
        match item {
            Item::Function(f) if f.name.name == name => return Some(f),
            Item::Impl(imp) => {
                for m in &imp.methods {
                    if m.name.name == name {
                        return Some(m);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_stage_io_attrs(attrs: &[Attribute]) -> (Option<u32>, IrInterpolation) {
    let mut location = None;
    let mut interpolation = IrInterpolation::Smooth;
    for attr in attrs {
        for entry in &attr.entries {
            match entry.name.name.as_str() {
                "location" => {
                    if let Some(Expr::IntLit(raw, _)) = entry.args.first()
                        && let Some(val) = parse_u64_literal(raw)
                    {
                        location = Some(val as u32);
                    }
                }
                "flat" => interpolation = IrInterpolation::Flat,
                "noperspective" => interpolation = IrInterpolation::NoPerspective,
                "centroid" => interpolation = IrInterpolation::Centroid,
                _ => {}
            }
        }
    }
    (location, interpolation)
}

fn has_param_attr(attrs: &[Attribute], name: &str) -> bool {
    attrs
        .iter()
        .any(|a| a.entries.iter().any(|e| e.name.name.as_str() == name))
}

fn parse_stage_var_attr(attrs: &[Attribute]) -> Option<(IrStorageClass, Option<u32>)> {
    for attr in attrs {
        for entry in &attr.entries {
            let loc = entry.args.first().and_then(|e| {
                if let Expr::IntLit(value, _) = e {
                    value.parse::<u32>().ok()
                } else {
                    None
                }
            });
            match entry.name.name.as_str() {
                "ray_payload" => return Some((IrStorageClass::RayPayloadKhr, Some(loc.unwrap_or(0)))),
                "incoming_ray_payload" => {
                    return Some((IrStorageClass::IncomingRayPayloadKhr, Some(loc.unwrap_or(0))));
                }
                "hit_attribute" => return Some((IrStorageClass::HitAttributeKhr, None)),
                "callable_data" => {
                    return Some((IrStorageClass::CallableDataKhr, Some(loc.unwrap_or(0))));
                }
                "incoming_callable_data" => {
                    return Some((IrStorageClass::IncomingCallableDataKhr, Some(loc.unwrap_or(0))));
                }
                "shader_record" | "shader_record_buffer" => {
                    return Some((IrStorageClass::ShaderRecordBufferKhr, None));
                }
                "task_payload" => return Some((IrStorageClass::TaskPayloadWorkgroupExt, None)),
                _ => {}
            }
        }
    }
    None
}

fn parse_stage_exec_meta(attrs: &[Attribute]) -> IrStageExecMeta {
    let mut meta = IrStageExecMeta::default();
    for attr in attrs {
        for entry in &attr.entries {
            let first_u32 = entry.args.first().and_then(|e| {
                if let Expr::IntLit(value, _) = e {
                    value.parse::<u32>().ok()
                } else {
                    None
                }
            });
            let first_ident = entry.args.first().and_then(|e| {
                if let Expr::Path { path, .. } = e {
                    Some(path.joined())
                } else {
                    None
                }
            });
            match entry.name.name.as_str() {
                "max_vertices" | "output_vertices" => meta.max_vertices = first_u32,
                "max_primitives" | "output_primitives" => meta.max_primitives = first_u32,
                "invocations" => meta.invocations = first_u32,
                "output_topology" => {
                    meta.output_topology = first_ident.as_deref().and_then(|s| match s {
                        "points" => Some(IrTopology::Points),
                        "lines" => Some(IrTopology::Lines),
                        "triangles" => Some(IrTopology::Triangles),
                        "line_strip" => Some(IrTopology::LineStrip),
                        "triangle_strip" => Some(IrTopology::TriangleStrip),
                        _ => None,
                    });
                }
                "input_topology" => {
                    meta.input_topology = first_ident.as_deref().and_then(|s| match s {
                        "points" => Some(IrTopology::Points),
                        "lines" => Some(IrTopology::Lines),
                        "triangles" => Some(IrTopology::Triangles),
                        _ => None,
                    });
                }
                "tess_domain" => {
                    meta.tess_domain = first_ident.as_deref().and_then(|s| match s {
                        "triangles" => Some(IrTessDomain::Triangles),
                        "quads" => Some(IrTessDomain::Quads),
                        "isolines" => Some(IrTessDomain::Isolines),
                        _ => None,
                    });
                }
                "tess_spacing" => {
                    meta.tess_spacing = first_ident.as_deref().and_then(|s| match s {
                        "equal" => Some(IrTessSpacing::Equal),
                        "fractional_even" => Some(IrTessSpacing::FractionalEven),
                        "fractional_odd" => Some(IrTessSpacing::FractionalOdd),
                        _ => None,
                    });
                }
                "tess_order" => {
                    meta.tess_order = first_ident.as_deref().and_then(|s| match s {
                        "cw" => Some(IrTessOrder::Cw),
                        "ccw" => Some(IrTessOrder::Ccw),
                        _ => None,
                    });
                }
                "point_mode" => meta.point_mode = true,
                "early_fragment_tests" => meta.early_fragment_tests = true,
                "interlock" => {
                    meta.interlock_mode = first_ident.as_deref().and_then(|s| match s {
                        "pixel_ordered" => Some(IrInterlockMode::PixelOrdered),
                        "pixel_unordered" => Some(IrInterlockMode::PixelUnordered),
                        "sample_ordered" => Some(IrInterlockMode::SampleOrdered),
                        "sample_unordered" => Some(IrInterlockMode::SampleUnordered),
                        _ => None,
                    });
                }
                _ => {}
            }
        }
    }
    meta
}

fn parse_builtin_kind(name: &str) -> Option<BuiltinKind> {
    match name {
        "global_invocation_id" => Some(BuiltinKind::GlobalInvocationId),
        "local_invocation_id" => Some(BuiltinKind::LocalInvocationId),
        "local_invocation_index" => Some(BuiltinKind::LocalInvocationIndex),
        "workgroup_id" => Some(BuiltinKind::WorkgroupId),
        "num_workgroups" => Some(BuiltinKind::NumWorkgroups),
        "vertex_index" => Some(BuiltinKind::VertexIndex),
        "instance_index" => Some(BuiltinKind::InstanceIndex),
        "frag_coord" => Some(BuiltinKind::FragCoord),
        "subgroup_local_invocation_id" => Some(BuiltinKind::SubgroupLocalInvocationId),
        "subgroup_size" => Some(BuiltinKind::SubgroupSize),
        "position" => Some(BuiltinKind::Position),
        "frag_depth" => Some(BuiltinKind::FragDepth),
        "front_facing" => Some(BuiltinKind::FrontFacing),
        "launch_id" => Some(BuiltinKind::LaunchIdKhr),
        "launch_size" => Some(BuiltinKind::LaunchSizeKhr),
        "world_ray_origin" => Some(BuiltinKind::WorldRayOriginKhr),
        "world_ray_direction" => Some(BuiltinKind::WorldRayDirectionKhr),
        "object_ray_origin" => Some(BuiltinKind::ObjectRayOriginKhr),
        "object_ray_direction" => Some(BuiltinKind::ObjectRayDirectionKhr),
        "ray_tmin" => Some(BuiltinKind::RayTminKhr),
        "ray_tmax" => Some(BuiltinKind::RayTmaxKhr),
        "instance_custom_index" => Some(BuiltinKind::InstanceCustomIndexKhr),
        "object_to_world" => Some(BuiltinKind::ObjectToWorldKhr),
        "world_to_object" => Some(BuiltinKind::WorldToObjectKhr),
        "hit_kind" => Some(BuiltinKind::HitKindKhr),
        "incoming_ray_flags" => Some(BuiltinKind::IncomingRayFlagsKhr),
        "ray_geometry_index" => Some(BuiltinKind::RayGeometryIndexKhr),
        "primitive_point_indices" => Some(BuiltinKind::PrimitivePointIndicesExt),
        "primitive_line_indices" => Some(BuiltinKind::PrimitiveLineIndicesExt),
        "primitive_triangle_indices" => Some(BuiltinKind::PrimitiveTriangleIndicesExt),
        "cull_primitive" => Some(BuiltinKind::CullPrimitiveExt),
        "invocation_id" => Some(BuiltinKind::InvocationId),
        "patch_vertices" => Some(BuiltinKind::PatchVertices),
        "tess_level_outer" => Some(BuiltinKind::TessLevelOuter),
        "tess_level_inner" => Some(BuiltinKind::TessLevelInner),
        "tess_coord" => Some(BuiltinKind::TessCoord),
        "primitive_id" => Some(BuiltinKind::PrimitiveId),
        "layer" => Some(BuiltinKind::Layer),
        "hit_triangle_vertex_positions" => Some(BuiltinKind::HitTriangleVertexPositionsKhr),
        "sample_id" => Some(BuiltinKind::SampleId),
        "sample_position" => Some(BuiltinKind::SamplePosition),
        "sample_mask" => Some(BuiltinKind::SampleMask),
        "primitive_shading_rate" => Some(BuiltinKind::PrimitiveShadingRateKhr),
        "shading_rate" => Some(BuiltinKind::ShadingRateKhr),
        "view_index" => Some(BuiltinKind::ViewIndex),
        "clip_distance" => Some(BuiltinKind::ClipDistance),
        "cull_distance" => Some(BuiltinKind::CullDistance),
        _ => None,
    }
}

fn find_push_constant_struct_fields(
    hir: &HirModule,
    func: &HirFunction,
) -> IndexMap<SmolStr, ScalarIrTy> {
    let mut out = IndexMap::new();
    if let Some(ty_id) = func
        .params
        .iter()
        .find(|p| p.is_push_constant)
        .and_then(|p| p.ty)
        && let Type::Struct { fields, .. } = hir.types.get(ty_id)
    {
        for (name, field_ty) in fields {
            out.insert(name.clone(), type_id_to_scalar_ir(hir, *field_ty));
        }
    }
    if out.is_empty()
        && let Some((_, s)) = hir.structs.iter().find(|(_, s)| s.is_push_constant)
    {
        for field in &s.fields {
            out.insert(field.name.clone(), type_id_to_scalar_ir(hir, field.ty));
        }
    }
    out
}

fn type_id_to_scalar_ir(hir: &HirModule, ty_id: TypeId) -> ScalarIrTy {
    match hir.types.get(ty_id) {
        Type::Scalar(s) | Type::Vector { scalar: s, .. } | Type::Matrix { scalar: s, .. } => {
            ScalarIrTy::from_hir_scalar(*s)
        }
        Type::Array { element, .. } => type_id_to_scalar_ir(hir, *element),
        Type::Pointer { .. } => ScalarIrTy::U64,
        _ => ScalarIrTy::U32,
    }
}

/// Tracks a local variable holding an element of a built-in SSBO runtime array
/// (such as `let inst = mc::instances[i];`) so subsequent `.field` accesses emit exact
/// `DescriptorAccessChain` + `DescriptorLoad` instructions.
#[derive(Clone)]
struct DescriptorElemBinding {
    descriptor_index: u32,
    items_field_index: u32,
    index_val: InstId,
    elem_desc: &'static BuiltinStructDesc,
}

struct LowerCtx<'a, 'd> {
    hir: &'a HirModule,
    ir_fn: &'a mut IrFunction,
    diagnostics: &'d mut Vec<Diagnostic>,
    pack_schema: Option<&'a PackSchema>,
    ast_hints: AstHints,
    stage: Option<ShaderStage>,
    entry_preamble: Vec<StructuredStmt>,
    /// Immutable SSA locals (`name -> InstId`).
    ssa_locals: IndexMap<SmolStr, InstId>,
    /// Mutable or composite locals backed by `IrLocalVar` (`name -> (local_index, IrType)`).
    local_var_slots: IndexMap<SmolStr, (u32, IrType)>,
    ptr_param_locals: IndexMap<SmolStr, (InstId, IrType)>,
    /// Descriptor element bindings (`let inst = mc::instances[i]`).
    desc_elem_bindings: IndexMap<SmolStr, DescriptorElemBinding>,
    /// BDA pointer stride and alignment metadata (`name -> (stride_bytes, align)`).
    ptr_meta: IndexMap<SmolStr, (u64, u32)>,
    /// Descriptor name -> index in `ir_fn.descriptors`.
    descriptor_indices: IndexMap<SmolStr, u32>,
    storage_image_descriptors: IndexMap<SmolStr, u32>,
    tlas_descriptor_index: Option<u32>,
    workgroup_indices: IndexMap<SmolStr, u32>,
    builtin_params: IndexMap<SmolStr, BuiltinKind>,
    stage_io_outputs: IndexMap<SmolStr, u32>,
    stage_var_indices: IndexMap<SmolStr, u32>,
    push_constant_param: Option<SmolStr>,
    push_constant_fields: IndexMap<SmolStr, u32>,
    ray_query_handles: IndexMap<SmolStr, u32>,
    next_ray_query_id: u32,
    module_consts: IndexMap<SmolStr, HirExpr>,
    spec_defaults: IndexMap<SmolStr, SpecConstantDefault>,
    param_layouts: IndexMap<SmolStr, PackParamLayout>,
    fn_names_by_id: IndexMap<HirFnId, SmolStr>,
}

impl<'a, 'd> LowerCtx<'a, 'd> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        hir: &'a HirModule,
        func: &HirFunction,
        ast_fn: Option<&FnDecl>,
        pack_schema: Option<&'a PackSchema>,
        ast_hints: AstHints,
        stage: Option<ShaderStage>,
        ir_fn: &'a mut IrFunction,
        diagnostics: &'d mut Vec<Diagnostic>,
    ) -> Self {
        let mut entry_preamble = Vec::new();
        let mut ssa_locals = IndexMap::new();
        let mut local_var_slots = IndexMap::new();
        let mut ptr_param_locals = IndexMap::new();
        let mut ptr_meta = IndexMap::new();
        let mut descriptor_indices = IndexMap::new();
        let mut storage_image_descriptors = IndexMap::new();
        let mut tlas_descriptor_index = None;
        let mut workgroup_indices = IndexMap::new();
        let mut builtin_params = IndexMap::new();
        let mut stage_io_outputs = IndexMap::new();
        let mut stage_var_indices = IndexMap::new();
        if let Some(afn) = ast_fn {
            ir_fn.stage_exec_meta = parse_stage_exec_meta(&afn.attrs);
        }
        let mut push_constant_param = None;
        let mut push_constant_fields = IndexMap::new();

        for (idx, desc) in ir_fn.descriptors.iter().enumerate() {
            let idx_u32 = idx as u32;
            descriptor_indices.insert(desc.name.clone(), idx_u32);
            if desc.kind == DescriptorKind::StorageImage {
                storage_image_descriptors.insert(desc.name.clone(), idx_u32);
            } else if desc.kind == DescriptorKind::AccelerationStructure {
                tlas_descriptor_index = Some(idx_u32);
            }
        }

        for (idx, wg) in ir_fn.workgroup_vars.iter().enumerate() {
            workgroup_indices.insert(wg.name.clone(), idx as u32);
        }

        if let Some(ref pc_block) = ir_fn.push_constants {
            for (idx, field) in pc_block.fields.iter().enumerate() {
                push_constant_fields.insert(field.name.clone(), idx as u32);
            }
        } else {
            for (idx, field) in func.reflection.push_constants.iter().enumerate() {
                push_constant_fields.insert(field.name.clone(), idx as u32);
            }
        }

        for (param_pos, param) in func.params.iter().enumerate() {
            if let Some(ref builtin_name) = param.builtin
                && let Some(kind) = parse_builtin_kind(builtin_name.as_str())
            {
                builtin_params.insert(param.name.clone(), kind);
                if matches!(param.mode, ParamMode::Out | ParamMode::InOut) {
                    let ir_ty = param
                        .ty
                        .map_or(IrType::Void, |id| IrType::from_hir_type(&hir.types, id));
                    let slot = alloc_local_var(ir_fn, param.name.clone(), ir_ty.clone());
                    local_var_slots.insert(param.name.clone(), (slot, ir_ty));
                } else {
                    let val = ir_fn.emit(
                        param.ty,
                        InstData::LoadBuiltin {
                            kind,
                            component: None,
                        },
                    );
                    entry_preamble.push(StructuredStmt::Eval(val));
                    ssa_locals.insert(param.name.clone(), val);
                }
                continue;
            }

            if param.is_push_constant {
                push_constant_param = Some(param.name.clone());
                continue;
            }

            // RT/Task stage variables (`#[ray_payload]`, `#[hit_attribute]`, `#[callable_data]`, `#[shader_record]`, `#[task_payload]`).
            if let Some(ast_param) = ast_fn.and_then(|f| f.params.get(param_pos))
                && let Some((storage, location)) = parse_stage_var_attr(&ast_param.attrs)
            {
                let ir_ty = param
                    .ty
                    .map_or(IrType::Void, |id| IrType::from_hir_type(&hir.types, id));
                let var_index = ir_fn.stage_vars.len() as u32;
                ir_fn.stage_vars.push(IrStageVar {
                    name: param.name.clone(),
                    storage,
                    location,
                    ty: ir_ty.clone(),
                });
                stage_var_indices.insert(param.name.clone(), var_index);
                let loaded = ir_fn.emit_typed(ir_ty.clone(), InstData::LoadStageVar { var_index });
                entry_preamble.push(StructuredStmt::Eval(loaded));
                ssa_locals.insert(param.name.clone(), loaded);
                if matches!(param.mode, ParamMode::Out | ParamMode::InOut) {
                    let slot = alloc_local_var(ir_fn, param.name.clone(), ir_ty.clone());
                    if param.mode == ParamMode::InOut {
                        let addr = emit_local_addr(ir_fn, slot, &ir_ty);
                        let init_st = ir_fn.emit_typed(
                            IrType::Void,
                            InstData::LocalStore {
                                ptr: addr,
                                value: loaded,
                            },
                        );
                        entry_preamble.push(StructuredStmt::Eval(init_st));
                    }
                    local_var_slots.insert(param.name.clone(), (slot, ir_ty));
                }
                continue;
            }

            // Stage I/O parameters (`#[location(N)]`, `#[flat]`, `#[noperspective]`, `#[centroid]`, `#[per_primitive]`).
            if stage.is_some_and(ShaderStage::uses_stage_io)
                && let Some(ast_param) = ast_fn.and_then(|f| f.params.get(param_pos))
                && let (Some(location), interpolation) = parse_stage_io_attrs(&ast_param.attrs)
            {
                let ir_ty = param
                    .ty
                    .map_or(IrType::Void, |id| IrType::from_hir_type(&hir.types, id));
                let is_output = matches!(param.mode, ParamMode::Out | ParamMode::InOut);
                let per_primitive = has_param_attr(&ast_param.attrs, "per_primitive");
                let var_index = ir_fn.stage_io.len() as u32;
                ir_fn.stage_io.push(IrStageIoVar {
                    name: param.name.clone(),
                    location,
                    is_output,
                    ty: ir_ty.clone(),
                    interpolation,
                    per_primitive,
                });
                if is_output {
                    stage_io_outputs.insert(param.name.clone(), var_index);
                    let slot = alloc_local_var(ir_fn, param.name.clone(), ir_ty.clone());
                    local_var_slots.insert(param.name.clone(), (slot, ir_ty));
                } else {
                    let loaded = ir_fn.emit_typed(ir_ty, InstData::LoadStageInput { var_index });
                    entry_preamble.push(StructuredStmt::Eval(loaded));
                    ssa_locals.insert(param.name.clone(), loaded);
                }
                continue;
            }

            // Record `#[input_attachment_index(N)]` for `SubpassInput` descriptors.
            if let Some(&desc_idx) = descriptor_indices.get(&param.name) {
                if let Some(ast_param) = ast_fn.and_then(|f| f.params.get(param_pos)) {
                    for attr in &ast_param.attrs {
                        for entry in &attr.entries {
                            if entry.name.name == "input_attachment_index"
                                && let Some(Expr::IntLit(value, _)) = entry.args.first()
                                && let Ok(idx) = value.parse::<u32>()
                            {
                                ir_fn.stage_exec_meta.input_attachment_indices.push((desc_idx, idx));
                            }
                        }
                    }
                }
                continue;
            }

            // Record BDA pointer stride metadata when parameter is a physical pointer.
            if let Some(ty_id) = param.ty
                && let Type::Pointer { pointee, align, .. } = *hir.types.get(ty_id)
            {
                let stride = hir
                    .types
                    .size_and_align(pointee, LayoutKind::Std430)
                    .map_or(u64::from(align.max(1)), |(sz, _)| u64::from(sz.max(1)));
                ptr_meta.insert(param.name.clone(), (stride, align));
            }

            // Bind formal parameter via real `IrParam` + `InstData::Param { index }` (never zero-constant stubs!).
            let ir_ty = param
                .ty
                .map_or(IrType::Void, |id| IrType::from_hir_type(&hir.types, id));
            let is_inout = stage.is_none() && matches!(param.mode, ParamMode::InOut);
            let is_out = stage.is_none() && matches!(param.mode, ParamMode::Out);
            let is_mut_local = matches!(param.mode, ParamMode::Mut);
            let p_idx = ir_fn.params.len() as u32;
            ir_fn.params.push(IrParam {
                name: param.name.clone(),
                ty: ir_ty.clone(),
                is_inout,
                is_out,
            });
            if is_inout || is_out {
                let ptr_ty = IrType::Pointer {
                    pointee: Box::new(ir_ty.clone()),
                    align: 4,
                    storage: IrStorageClass::Function,
                };
                let param_ptr = ir_fn.emit_typed(ptr_ty, InstData::Param { index: p_idx });
                ptr_param_locals.insert(param.name.clone(), (param_ptr, ir_ty));
            } else {
                let param_val = ir_fn.emit_typed(ir_ty.clone(), InstData::Param { index: p_idx });
                if is_mut_local || is_composite_ir_type(&ir_ty) {
                    let slot = alloc_local_var(ir_fn, param.name.clone(), ir_ty.clone());
                    let addr = emit_local_addr(ir_fn, slot, &ir_ty);
                    let store = ir_fn.emit_typed(
                        IrType::Void,
                        InstData::LocalStore {
                            ptr: addr,
                            value: param_val,
                        },
                    );
                    entry_preamble.push(StructuredStmt::Eval(store));
                    local_var_slots.insert(param.name.clone(), (slot, ir_ty));
                } else {
                    ssa_locals.insert(param.name.clone(), param_val);
                }
            }
        }

        // Populate specialization constant defaults from `PackSchema::spec_defaults` + reflection.
        let mut spec_defaults = IndexMap::new();
        if let Some(schema) = pack_schema {
            for (name, def) in &schema.spec_defaults {
                spec_defaults.insert(name.clone(), *def);
            }
            for (name, &spec_id) in &schema.spec_ids {
                spec_defaults
                    .entry(name.clone())
                    .or_insert(SpecConstantDefault {
                        spec_id,
                        ty: PackParamType::U32,
                        default_bits: 0,
                    });
            }
        }
        for used in &func.reflection.used_specs {
            spec_defaults
                .entry(used.name.clone())
                .or_insert(SpecConstantDefault {
                    spec_id: used.spec_id,
                    ty: PackParamType::U32,
                    default_bits: 0,
                });
        }

        // Populate typed `PackParamLayout` entries from `PackSchema` + reflection.
        let mut param_layouts = IndexMap::new();
        if let Some(schema) = pack_schema {
            for (name, &(byte_offset, byte_size)) in &schema.params {
                let ty = schema
                    .param_types
                    .get(name)
                    .copied()
                    .unwrap_or(if byte_size >= 12 {
                        PackParamType::Vec3
                    } else {
                        PackParamType::F32
                    });
                param_layouts.insert(
                    name.clone(),
                    PackParamLayout {
                        byte_offset,
                        byte_size,
                        ty,
                    },
                );
            }
        }
        for used in &func.reflection.used_params {
            param_layouts.entry(used.name.clone()).or_insert_with(|| {
                let ty = pack_schema
                    .and_then(|s| s.param_types.get(&used.name).copied())
                    .unwrap_or(if used.byte_size >= 12 {
                        PackParamType::Vec3
                    } else {
                        PackParamType::F32
                    });
                PackParamLayout {
                    byte_offset: used.byte_offset,
                    byte_size: used.byte_size,
                    ty,
                }
            });
        }

        let module_consts = hir
            .consts
            .iter()
            .map(|c| (c.name.clone(), c.value_expr.clone()))
            .collect();

        let fn_names_by_id = hir
            .functions
            .iter()
            .map(|(id, f)| (id, f.name.clone()))
            .collect();

        Self {
            hir,
            ir_fn,
            diagnostics,
            pack_schema,
            ast_hints,
            stage,
            entry_preamble,
            ssa_locals,
            local_var_slots,
            ptr_param_locals,
            desc_elem_bindings: IndexMap::new(),
            ptr_meta,
            descriptor_indices,
            storage_image_descriptors,
            tlas_descriptor_index,
            workgroup_indices,
            builtin_params,
            stage_io_outputs,
            stage_var_indices,
            push_constant_param,
            push_constant_fields,
            ray_query_handles: IndexMap::new(),
            next_ray_query_id: 0,
            module_consts,
            spec_defaults,
            param_layouts,
            fn_names_by_id,
        }
    }

    fn ir_ty_of(&self, ty_id: TypeId) -> IrType {
        IrType::from_hir_type(&self.hir.types, ty_id)
    }

    fn inst_ty(&self, id: InstId) -> &IrType {
        &self.ir_fn.insts[id].ir_ty
    }

    fn emit_diag(&mut self, code: &'static str, msg: impl Into<String>, span: Span) -> InstId {
        self.diagnostics.push(Diagnostic::error(code, msg, span));
        self.ir_fn.emit_typed(IrType::Void, InstData::Nop)
    }

    fn ray_query_id_for_receiver(&mut self, receiver: &HirExpr) -> u32 {
        let key = match &receiver.kind {
            HirExprKind::Path { path, .. } => path.clone(),
            _ => SmolStr::new_static("__default_rq"),
        };
        if let Some(&id) = self.ray_query_handles.get(&key) {
            return id;
        }
        let next_id = self.next_ray_query_id;
        self.next_ray_query_id = self.next_ray_query_id.saturating_add(1);
        self.ray_query_handles.insert(key, next_id);
        next_id
    }

    fn clear_local_binding(&mut self, name: &SmolStr) {
        self.ssa_locals.shift_remove(name);
        self.local_var_slots.shift_remove(name);
        self.ptr_param_locals.shift_remove(name);
        self.desc_elem_bindings.shift_remove(name);
        self.ptr_meta.shift_remove(name);
        self.ray_query_handles.shift_remove(name);
    }

    fn record_local_ptr_meta(&mut self, name: &SmolStr, ty_id: TypeId) {
        if let Type::Pointer { pointee, align, .. } = *self.hir.types.get(ty_id) {
            let stride = self
                .hir
                .types
                .size_and_align(pointee, LayoutKind::Std430)
                .map_or(u64::from(align.max(1)), |(sz, _)| u64::from(sz.max(1)));
            self.ptr_meta.insert(name.clone(), (stride, align));
        }
    }

    fn lower_block(&mut self, block: &HirBlock) -> Vec<StructuredStmt> {
        let (mut stmts, tail) = self.lower_block_value(block);
        if let Some(val) = tail {
            stmts.push(StructuredStmt::Eval(val));
        }
        stmts
    }

    fn lower_block_value(&mut self, block: &HirBlock) -> (Vec<StructuredStmt>, Option<InstId>) {
        let saved_ssa = self.ssa_locals.clone();
        let saved_slots = self.local_var_slots.clone();
        let saved_ptr_params = self.ptr_param_locals.clone();
        let saved_desc_elems = self.desc_elem_bindings.clone();
        let saved_ptr_meta = self.ptr_meta.clone();
        let saved_rq = self.ray_query_handles.clone();

        let mut out = Vec::new();
        for stmt in &block.stmts {
            self.lower_stmt(stmt, &mut out);
        }
        let tail = block
            .tail_expr
            .as_deref()
            .map(|e| self.lower_expr(e, &mut out));

        self.ssa_locals = saved_ssa;
        self.local_var_slots = saved_slots;
        self.ptr_param_locals = saved_ptr_params;
        self.desc_elem_bindings = saved_desc_elems;
        self.ptr_meta = saved_ptr_meta;
        self.ray_query_handles = saved_rq;

        (out, tail)
    }

    fn eval_const_i64(&self, expr: &HirExpr, depth: u8) -> Option<i64> {
        if depth > 16 {
            return None;
        }
        match &expr.kind {
            HirExprKind::IntLit(raw) => parse_u64_literal(raw).map(u64::cast_signed),
            HirExprKind::Unary {
                op: UnaryOp::Neg,
                operand,
            } => self
                .eval_const_i64(operand, depth + 1)
                .map(i64::saturating_neg),
            HirExprKind::Path { path, .. } if !path.contains("::") => self
                .module_consts
                .get(path)
                .and_then(|c| self.eval_const_i64(c, depth + 1)),
            _ => None,
        }
    }

    fn lower_stmt(&mut self, stmt: &HirStmt, out: &mut Vec<StructuredStmt>) {
        match stmt {
            HirStmt::Let {
                is_mut,
                name,
                ty,
                init,
                ..
            } => {
                // Skip `RayQuery` declarations: RayQuery handles are tracked by name, never stored
                // in `IrLocalVar` or SSA values (SPV_KHR_ray_query legality).
                if matches!(self.hir.types.get(*ty), Type::RayQuery) {
                    self.clear_local_binding(name);
                    let rq_id = self.next_ray_query_id;
                    self.next_ray_query_id = self.next_ray_query_id.saturating_add(1);
                    self.ray_query_handles.insert(name.clone(), rq_id);
                    return;
                }

                // Track `let inst = mc::instances[idx]` / `mc::lights.items[i]` / `mc::reservoirs.items[i]`.
                if let Some(init_expr) = init
                    && let Some(elem_binding) = self.try_lower_descriptor_elem_index(init_expr, out)
                {
                    self.clear_local_binding(name);
                    self.desc_elem_bindings.insert(name.clone(), elem_binding);
                    return;
                }

                let ir_ty = self.ir_ty_of(*ty);
                let use_local_var = *is_mut || is_composite_ir_type(&ir_ty);

                if use_local_var {
                    let effective_ty = if ir_ty != IrType::Void {
                        ir_ty
                    } else if let Some(init_expr) = init {
                        self.ir_ty_of(init_expr.ty)
                    } else {
                        IrType::Void
                    };
                    if let Some(init_expr) = init {
                        let val = self.lower_expr(init_expr, out);
                        let slot_ty = if effective_ty != IrType::Void {
                            effective_ty
                        } else {
                            self.inst_ty(val).clone()
                        };
                        let slot = alloc_local_var(self.ir_fn, name.clone(), slot_ty.clone());
                        let addr = emit_local_addr(self.ir_fn, slot, &slot_ty);
                        let store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::LocalStore {
                                ptr: addr,
                                value: val,
                            },
                        );
                        out.push(StructuredStmt::Eval(store));
                        self.clear_local_binding(name);
                        self.record_local_ptr_meta(name, init_expr.ty);
                        self.record_local_ptr_meta(name, *ty);
                        self.local_var_slots.insert(name.clone(), (slot, slot_ty));
                    } else {
                        let slot = alloc_local_var(self.ir_fn, name.clone(), effective_ty.clone());
                        self.clear_local_binding(name);
                        self.record_local_ptr_meta(name, *ty);
                        self.local_var_slots
                            .insert(name.clone(), (slot, effective_ty));
                    }
                } else if let Some(init_expr) = init {
                    let val = self.lower_expr(init_expr, out);
                    if !is_const_inst(self.ir_fn, val) {
                        out.push(StructuredStmt::Eval(val));
                    }
                    self.clear_local_binding(name);
                    self.record_local_ptr_meta(name, init_expr.ty);
                    self.record_local_ptr_meta(name, *ty);
                    self.ssa_locals.insert(name.clone(), val);
                } else {
                    let slot = alloc_local_var(self.ir_fn, name.clone(), ir_ty.clone());
                    self.clear_local_binding(name);
                    self.record_local_ptr_meta(name, *ty);
                    self.local_var_slots.insert(name.clone(), (slot, ir_ty));
                }
            }
            HirStmt::Return { value, .. } => {
                let ret_val = value.as_ref().map(|e| self.lower_expr(e, out));
                out.push(StructuredStmt::Return(ret_val));
            }
            HirStmt::Break(_) => out.push(StructuredStmt::Break),
            HirStmt::Continue(_) => out.push(StructuredStmt::Continue),
            HirStmt::For {
                binding,
                binding_ty,
                iter,
                body,
                span,
                ..
            } => {
                self.lower_for_stmt(binding, *binding_ty, iter, body, *span, out);
            }
            HirStmt::While {
                cond, body, span, ..
            } => {
                let unroll = self.ast_hints.unroll_spans.contains(span);
                let mut loop_body = Vec::new();
                let cond_val = self.lower_expr(cond, &mut loop_body);
                loop_body.push(StructuredStmt::If {
                    cond: cond_val,
                    then_body: Vec::new(),
                    else_body: vec![StructuredStmt::Break],
                });
                let mut inner = self.lower_block(body);
                loop_body.append(&mut inner);
                out.push(StructuredStmt::Loop {
                    body: loop_body,
                    continuing: Vec::new(),
                    unroll,
                    trip_count: None,
                    induction_init: None,
                    induction_step: 0,
                });
            }
            HirStmt::Loop { body, span } => {
                let unroll = self.ast_hints.unroll_spans.contains(span);
                let loop_body = self.lower_block(body);
                out.push(StructuredStmt::Loop {
                    body: loop_body,
                    continuing: Vec::new(),
                    unroll,
                    trip_count: None,
                    induction_init: None,
                    induction_step: 0,
                });
            }
            HirStmt::Switch {
                selector,
                cases,
                default,
                ..
            } => {
                let sel_val = self.lower_expr(selector, out);
                let mut lowered_cases = Vec::with_capacity(cases.len());
                for c in cases {
                    let Some(key) = self.eval_const_i64(&c.value, 0) else {
                        self.emit_diag(
                            "E0521",
                            "switch case label must be a compile-time integer constant",
                            c.span,
                        );
                        continue;
                    };
                    let body_stmts = self.lower_block(&c.body);
                    lowered_cases.push((key, body_stmts));
                }
                let lowered_default = default
                    .as_ref()
                    .map_or_else(Vec::new, |b| self.lower_block(b));
                out.push(StructuredStmt::Switch {
                    selector: sel_val,
                    cases: lowered_cases,
                    default: lowered_default,
                });
            }
            HirStmt::Expr { expr, .. } => {
                if let HirExprKind::If {
                    cond,
                    then_branch,
                    else_branch,
                } = &expr.kind
                {
                    let _ = self.lower_if_branches(
                        cond,
                        then_branch,
                        else_branch.as_deref(),
                        expr.ty,
                        false,
                        out,
                    );
                } else {
                    let val = self.lower_expr(expr, out);
                    if !matches!(self.ir_fn.insts[val].data, InstData::Nop) {
                        out.push(StructuredStmt::Eval(val));
                    }
                }
            }
        }
    }

    fn lower_for_stmt(
        &mut self,
        binding: &SmolStr,
        binding_ty: TypeId,
        iter: &HirExpr,
        body: &HirBlock,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) {
        let unroll = self.ast_hints.unroll_spans.contains(&span);
        let (range_expr, step_expr) = unwrap_step_by_hir(iter);
        let step_const = step_expr.and_then(parse_const_i64_hir).unwrap_or(1).max(1);

        let bind_ir_ty = self.ir_ty_of(binding_ty);
        let (trip_count, induction_init) = extract_const_range_bounds_hir(
            range_expr,
            step_const,
            &bind_ir_ty,
            self.hir,
            self.ir_fn,
        );

        let saved_ssa = self.ssa_locals.clone();
        let saved_slots = self.local_var_slots.clone();
        let saved_ptr_params = self.ptr_param_locals.clone();
        let saved_desc_elems = self.desc_elem_bindings.clone();
        let saved_ptr_meta = self.ptr_meta.clone();
        let saved_rq = self.ray_query_handles.clone();

        // Always emit the canonical induction `IrLocalVar`, exit guard (`If { else: Break }`),
        // and `continuing` increment even when `#[unroll]` is present so any fallback from
        // `UnrollLoopsPass` still produces a terminating loop.
        let (mut loop_body, continuing) = if let HirExprKind::Binary {
            op: op @ (BinaryOp::Range | BinaryOp::RangeInclusive),
            lhs,
            rhs,
        } = &range_expr.kind
        {
            let raw_start = induction_init.unwrap_or_else(|| self.lower_expr(lhs, out));
            let raw_end = self.lower_expr(rhs, out);
            let is_unsigned = self.inst_ty(raw_start).is_unsigned_int()
                || self.inst_ty(raw_end).is_unsigned_int()
                || bind_ir_ty.is_unsigned_int();
            let is_64 = self.inst_ty(raw_start).is_64bit()
                || self.inst_ty(raw_end).is_64bit()
                || bind_ir_ty.is_64bit();
            let ind_scalar = match (is_64, is_unsigned) {
                (true, true) => ScalarIrTy::U64,
                (true, false) => ScalarIrTy::I64,
                (false, true) => ScalarIrTy::U32,
                (false, false) => ScalarIrTy::I32,
            };
            let ind_ty = IrType::Scalar(ind_scalar);

            let start_val = match self.inst_ty(raw_start).scalar_ty() {
                Some(from) if from != ind_scalar => {
                    coerce_scalar(self.ir_fn, raw_start, from, ind_scalar)
                }
                _ => raw_start,
            };
            let end_val = match self.inst_ty(raw_end).scalar_ty() {
                Some(from) if from != ind_scalar => {
                    coerce_scalar(self.ir_fn, raw_end, from, ind_scalar)
                }
                _ => raw_end,
            };
            let step_val = match step_expr {
                Some(se) => {
                    let raw_step = self.lower_expr(se, out);
                    match self.inst_ty(raw_step).scalar_ty() {
                        Some(from) if from != ind_scalar => {
                            coerce_scalar(self.ir_fn, raw_step, from, ind_scalar)
                        }
                        _ => raw_step,
                    }
                }
                None => match ind_scalar {
                    ScalarIrTy::U64 => self.ir_fn.emit_typed(ind_ty.clone(), InstData::ConstU64(1)),
                    ScalarIrTy::I64 => self.ir_fn.emit_typed(ind_ty.clone(), InstData::ConstI64(1)),
                    ScalarIrTy::U32 => self.ir_fn.emit_typed(ind_ty.clone(), InstData::ConstU32(1)),
                    _ => self.ir_fn.emit_typed(ind_ty.clone(), InstData::ConstI32(1)),
                },
            };
            for v in [start_val, end_val, step_val] {
                if !is_const_inst(self.ir_fn, v) {
                    out.push(StructuredStmt::Eval(v));
                }
            }

            let ind_slot = alloc_local_var(self.ir_fn, binding.clone(), ind_ty.clone());
            let ind_addr = emit_local_addr(self.ir_fn, ind_slot, &ind_ty);
            let init_store = self.ir_fn.emit_typed(
                IrType::Void,
                InstData::LocalStore {
                    ptr: ind_addr,
                    value: start_val,
                },
            );
            out.push(StructuredStmt::Eval(init_store));
            self.clear_local_binding(binding);
            self.local_var_slots
                .insert(binding.clone(), (ind_slot, ind_ty.clone()));

            let mut body_stmts = Vec::new();
            let cur_addr = emit_local_addr(self.ir_fn, ind_slot, &ind_ty);
            let cur_val = self
                .ir_fn
                .emit_typed(ind_ty.clone(), InstData::LocalLoad { ptr: cur_addr });
            let cond = if matches!(op, BinaryOp::RangeInclusive) {
                self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::ICmpLe(cur_val, end_val),
                )
            } else if is_unsigned {
                self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::UCmpLt(cur_val, end_val),
                )
            } else {
                self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::ICmpLt(cur_val, end_val),
                )
            };
            body_stmts.push(StructuredStmt::If {
                cond,
                then_body: Vec::new(),
                else_body: vec![StructuredStmt::Break],
            });
            body_stmts.extend(self.lower_block(body));

            let cont_addr = emit_local_addr(self.ir_fn, ind_slot, &ind_ty);
            let cont_val = self
                .ir_fn
                .emit_typed(ind_ty.clone(), InstData::LocalLoad { ptr: cont_addr });
            let next_val = if is_64 {
                self.ir_fn
                    .emit_typed(ind_ty.clone(), InstData::IAdd64(cont_val, step_val))
            } else {
                self.ir_fn
                    .emit_typed(ind_ty.clone(), InstData::IAdd32(cont_val, step_val))
            };
            let cont_store = self.ir_fn.emit_typed(
                IrType::Void,
                InstData::LocalStore {
                    ptr: cont_addr,
                    value: next_val,
                },
            );
            (body_stmts, vec![StructuredStmt::Eval(cont_store)])
        } else {
            let iter_val = self.lower_expr(iter, out);
            self.clear_local_binding(binding);
            self.ssa_locals.insert(binding.clone(), iter_val);
            let mut body_stmts = self.lower_block(body);
            body_stmts.push(StructuredStmt::Break);
            (body_stmts, Vec::new())
        };

        if loop_body.is_empty() {
            loop_body.push(StructuredStmt::Break);
        }

        out.push(StructuredStmt::Loop {
            body: loop_body,
            continuing,
            unroll,
            trip_count,
            induction_init,
            induction_step: step_const,
        });

        self.ssa_locals = saved_ssa;
        self.local_var_slots = saved_slots;
        self.ptr_param_locals = saved_ptr_params;
        self.desc_elem_bindings = saved_desc_elems;
        self.ptr_meta = saved_ptr_meta;
        self.ray_query_handles = saved_rq;
    }

    fn lower_if_branches(
        &mut self,
        cond: &HirExpr,
        then_branch: &HirBlock,
        else_branch: Option<&HirExpr>,
        result_ty: TypeId,
        want_value: bool,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<InstId> {
        let cond_val = self.lower_expr(cond, out);
        let res_ir_ty = self.ir_ty_of(result_ty);

        let (mut then_body, then_val) = self.lower_block_value(then_branch);
        if !want_value && let Some(v) = then_val {
            then_body.push(StructuredStmt::Eval(v));
        }

        let (mut else_body, else_val) = match else_branch {
            Some(eb) => match &eb.kind {
                HirExprKind::Block(b) => self.lower_block_value(b),
                _ => {
                    let mut stmts = Vec::new();
                    let v = self.lower_expr(eb, &mut stmts);
                    (stmts, Some(v))
                }
            },
            None => (Vec::new(), None),
        };
        if !want_value && let Some(v) = else_val {
            else_body.push(StructuredStmt::Eval(v));
        }

        let then_terminates = block_ends_with_terminator(&then_body);
        let else_terminates = block_ends_with_terminator(&else_body);

        if want_value && let (Some(on_true), Some(on_false)) = (then_val, else_val) {
            let merged_ty = if res_ir_ty != IrType::Void {
                res_ir_ty
            } else {
                self.inst_ty(on_true).clone()
            };
            let merge_slot = alloc_local_var(
                self.ir_fn,
                SmolStr::new(format!("__if_merge_{}", self.ir_fn.local_vars.len())),
                merged_ty.clone(),
            );
            if !then_terminates {
                let addr = emit_local_addr(self.ir_fn, merge_slot, &merged_ty);
                let store = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::LocalStore {
                        ptr: addr,
                        value: on_true,
                    },
                );
                then_body.push(StructuredStmt::Eval(store));
            }
            if !else_terminates {
                let addr = emit_local_addr(self.ir_fn, merge_slot, &merged_ty);
                let store = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::LocalStore {
                        ptr: addr,
                        value: on_false,
                    },
                );
                else_body.push(StructuredStmt::Eval(store));
            }
            out.push(StructuredStmt::If {
                cond: cond_val,
                then_body,
                else_body,
            });
            let load_addr = emit_local_addr(self.ir_fn, merge_slot, &merged_ty);
            let loaded = self
                .ir_fn
                .emit_typed(merged_ty, InstData::LocalLoad { ptr: load_addr });
            out.push(StructuredStmt::Eval(loaded));
            return Some(loaded);
        }

        out.push(StructuredStmt::If {
            cond: cond_val,
            then_body,
            else_body,
        });
        None
    }

    fn lower_binary_op(
        &mut self,
        op: BinaryOp,
        mut l: InstId,
        mut r: InstId,
        result_ty: &IrType,
    ) -> InstId {
        let mut lhs_ty = self.inst_ty(l).clone();
        let mut rhs_ty = self.inst_ty(r).clone();
        let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
        let is_float = lhs_ty.is_float() || rhs_ty.is_float() || result_ty.is_float();
        let is_64 = if is_shift {
            lhs_ty.is_64bit() || result_ty.is_64bit()
        } else {
            lhs_ty.is_64bit() || rhs_ty.is_64bit() || result_ty.is_64bit()
        };
        let is_unsigned = lhs_ty.is_unsigned_int()
            || (!is_shift && rhs_ty.is_unsigned_int())
            || result_ty.is_unsigned_int();
        let mut out_ty = if *result_ty != IrType::Void {
            result_ty.clone()
        } else if lhs_ty != IrType::Void {
            lhs_ty.clone()
        } else {
            rhs_ty.clone()
        };

        // Preserve 64-bit integer width on `out_ty` for 64-bit arithmetic, bitwise, and shift ops.
        if !is_float && is_64 && !out_ty.is_64bit() {
            let wide_scalar = if is_unsigned {
                ScalarIrTy::U64
            } else {
                ScalarIrTy::I64
            };
            out_ty = match out_ty {
                IrType::Vector { len, .. } => IrType::Vector {
                    scalar: wide_scalar,
                    len,
                },
                _ => IrType::Scalar(wide_scalar),
            };
        }

        // Matrix-vector, matrix-matrix, and vector-scalar multiplications.
        if matches!(op, BinaryOp::Mul | BinaryOp::MulAssign) {
            if matches!(lhs_ty, IrType::Matrix { .. }) && matches!(rhs_ty, IrType::Vector { .. }) {
                return self
                    .ir_fn
                    .emit_typed(out_ty, InstData::MatrixTimesVector { mat: l, vec: r });
            }
            if matches!(lhs_ty, IrType::Matrix { .. }) && matches!(rhs_ty, IrType::Matrix { .. }) {
                return self
                    .ir_fn
                    .emit_typed(out_ty, InstData::MatrixTimesMatrix { lhs: l, rhs: r });
            }
            if let (IrType::Vector { scalar: vs, .. }, IrType::Scalar(rs)) = (&lhs_ty, &rhs_ty)
                && is_float
            {
                let s = coerce_scalar(self.ir_fn, r, *rs, *vs);
                return self
                    .ir_fn
                    .emit_typed(lhs_ty, InstData::VectorTimesScalar { vec: l, scalar: s });
            }
            if let (IrType::Scalar(ls), IrType::Vector { scalar: vs, .. }) = (&lhs_ty, &rhs_ty)
                && is_float
            {
                let s = coerce_scalar(self.ir_fn, l, *ls, *vs);
                return self
                    .ir_fn
                    .emit_typed(rhs_ty, InstData::VectorTimesScalar { vec: r, scalar: s });
            }
        }

        // Splat scalar operands to match vector operands in elementwise binary ops (`+`, `-`, `*`,
        // `/`, `%`, bitwise, shifts), preserving `f16` vs `f32` and integer signedness/width.
        if matches!(
            op,
            BinaryOp::Add
                | BinaryOp::AddAssign
                | BinaryOp::Sub
                | BinaryOp::SubAssign
                | BinaryOp::Mul
                | BinaryOp::MulAssign
                | BinaryOp::Div
                | BinaryOp::DivAssign
                | BinaryOp::Rem
                | BinaryOp::RemAssign
                | BinaryOp::BitAnd
                | BinaryOp::BitAndAssign
                | BinaryOp::BitOr
                | BinaryOp::BitOrAssign
                | BinaryOp::BitXor
                | BinaryOp::BitXorAssign
                | BinaryOp::Shl
                | BinaryOp::Shr
        ) {
            if let (IrType::Vector { scalar, len }, IrType::Scalar(rs)) = (&lhs_ty, &rhs_ty) {
                r = splat_scalar_to_vector(self.ir_fn, r, *rs, *scalar, *len);
                rhs_ty = lhs_ty.clone();
                if !matches!(out_ty, IrType::Vector { .. }) {
                    out_ty = lhs_ty.clone();
                }
            } else if let (IrType::Scalar(ls), IrType::Vector { scalar, len }) = (&lhs_ty, &rhs_ty)
            {
                l = splat_scalar_to_vector(self.ir_fn, l, *ls, *scalar, *len);
                lhs_ty = rhs_ty.clone();
                if !matches!(out_ty, IrType::Vector { .. }) {
                    out_ty = rhs_ty.clone();
                }
            }
        }

        // Coerce scalar integer operands to 64-bit when performing non-shift 64-bit integer ops.
        if !is_float
            && is_64
            && !is_shift
            && let IrType::Scalar(target_s) = out_ty
        {
            if let IrType::Scalar(ls) = lhs_ty
                && ls != target_s
            {
                l = coerce_scalar(self.ir_fn, l, ls, target_s);
            }
            if let IrType::Scalar(rs) = rhs_ty
                && rs != target_s
            {
                r = coerce_scalar(self.ir_fn, r, rs, target_s);
            }
        }

        let bool_ty = IrType::Scalar(ScalarIrTy::Bool);
        match op {
            BinaryOp::Add | BinaryOp::AddAssign => {
                if is_float {
                    self.ir_fn.emit_typed(out_ty, InstData::FAdd32(l, r))
                } else if is_64 {
                    self.ir_fn.emit_typed(out_ty, InstData::IAdd64(l, r))
                } else {
                    self.ir_fn.emit_typed(out_ty, InstData::IAdd32(l, r))
                }
            }
            BinaryOp::Sub | BinaryOp::SubAssign => {
                if is_float {
                    self.ir_fn.emit_typed(out_ty, InstData::FSub32(l, r))
                } else {
                    self.ir_fn.emit_typed(out_ty, InstData::ISub32(l, r))
                }
            }
            BinaryOp::Mul | BinaryOp::MulAssign => {
                if is_float {
                    self.ir_fn.emit_typed(out_ty, InstData::FMul32(l, r))
                } else if is_64 {
                    self.ir_fn.emit_typed(out_ty, InstData::IMul64(l, r))
                } else {
                    self.ir_fn.emit_typed(out_ty, InstData::IMul32(l, r))
                }
            }
            BinaryOp::Div | BinaryOp::DivAssign => {
                if is_float {
                    self.ir_fn.emit_typed(out_ty, InstData::FDiv32(l, r))
                } else {
                    self.ir_fn.emit_typed(out_ty, InstData::IDiv32(l, r))
                }
            }
            BinaryOp::Rem | BinaryOp::RemAssign => {
                self.ir_fn.emit_typed(out_ty, InstData::IRem32(l, r))
            }
            BinaryOp::BitAnd | BinaryOp::BitAndAssign => {
                self.ir_fn.emit_typed(out_ty, InstData::BitAnd32(l, r))
            }
            BinaryOp::BitOr | BinaryOp::BitOrAssign => {
                self.ir_fn.emit_typed(out_ty, InstData::BitOr32(l, r))
            }
            BinaryOp::BitXor | BinaryOp::BitXorAssign => {
                self.ir_fn.emit_typed(out_ty, InstData::BitXor32(l, r))
            }
            BinaryOp::Shl => self.ir_fn.emit_typed(out_ty, InstData::Shl32(l, r)),
            BinaryOp::Shr => self.ir_fn.emit_typed(out_ty, InstData::Shr32(l, r)),
            BinaryOp::Eq => {
                if is_float {
                    self.ir_fn.emit_typed(bool_ty, InstData::FCmpEq(l, r))
                } else {
                    self.ir_fn.emit_typed(bool_ty, InstData::ICmpEq(l, r))
                }
            }
            BinaryOp::Ne => {
                if is_float {
                    self.ir_fn.emit_typed(bool_ty, InstData::FCmpNe(l, r))
                } else {
                    self.ir_fn.emit_typed(bool_ty, InstData::ICmpNe(l, r))
                }
            }
            BinaryOp::Lt => {
                if is_float {
                    self.ir_fn.emit_typed(bool_ty, InstData::FCmpLt(l, r))
                } else if is_unsigned {
                    self.ir_fn.emit_typed(bool_ty, InstData::UCmpLt(l, r))
                } else {
                    self.ir_fn.emit_typed(bool_ty, InstData::ICmpLt(l, r))
                }
            }
            BinaryOp::Le => {
                if is_float {
                    self.ir_fn.emit_typed(bool_ty, InstData::FCmpLe(l, r))
                } else {
                    self.ir_fn.emit_typed(bool_ty, InstData::ICmpLe(l, r))
                }
            }
            BinaryOp::Gt => {
                if is_float {
                    self.ir_fn.emit_typed(bool_ty, InstData::FCmpGt(l, r))
                } else {
                    self.ir_fn.emit_typed(bool_ty, InstData::ICmpGt(l, r))
                }
            }
            BinaryOp::Ge => {
                if is_float {
                    self.ir_fn.emit_typed(bool_ty, InstData::FCmpGe(l, r))
                } else {
                    self.ir_fn.emit_typed(bool_ty, InstData::ICmpGe(l, r))
                }
            }
            BinaryOp::And => self.ir_fn.emit_typed(bool_ty, InstData::LogicalAnd(l, r)),
            BinaryOp::Or => self.ir_fn.emit_typed(bool_ty, InstData::LogicalOr(l, r)),
            BinaryOp::Assign | BinaryOp::Range | BinaryOp::RangeInclusive => r,
        }
    }

    fn lower_expr(&mut self, expr: &HirExpr, out: &mut Vec<StructuredStmt>) -> InstId {
        let expr_ir_ty = self.ir_ty_of(expr.ty);
        match &expr.kind {
            HirExprKind::BoolLit(b) => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::Bool), InstData::ConstBool(*b)),
            HirExprKind::IntLit(raw) => lower_int_literal(raw, &expr_ir_ty, self.ir_fn),
            HirExprKind::FloatLit(raw) => lower_float_literal(raw, &expr_ir_ty, self.ir_fn),
            HirExprKind::StringLit(s) => self.emit_diag(
                "E0521",
                format!("unexpected string literal `{s}` in GPU shader expression"),
                expr.span,
            ),
            HirExprKind::NoneLit => self.emit_diag(
                "E0521",
                "`None` literal is not supported in GPU IR",
                expr.span,
            ),
            HirExprKind::Path { path, .. } => {
                self.lower_path_expr(path, &expr_ir_ty, expr.span, out)
            }
            HirExprKind::ArrayLit { elements } => {
                let lowered: Vec<InstId> =
                    elements.iter().map(|e| self.lower_expr(e, out)).collect();
                let arr_ty = if expr_ir_ty != IrType::Void {
                    expr_ir_ty
                } else {
                    let elem_ty = lowered
                        .first()
                        .map_or(IrType::Scalar(ScalarIrTy::F32), |&id| {
                            self.inst_ty(id).clone()
                        });
                    IrType::Array {
                        element: Box::new(elem_ty),
                        len: lowered.len() as u32,
                    }
                };
                self.ir_fn
                    .emit_typed(arr_ty, InstData::CompositeConstruct { elems: lowered })
            }
            HirExprKind::StructLit {
                struct_name,
                fields,
            } => {
                // Order field expressions to match the canonical `HirStruct` declaration order.
                let ordered_elems: Vec<InstId> = if let Some(st) = self.hir.find_struct(struct_name)
                {
                    let mut map: IndexMap<&str, InstId> = IndexMap::with_capacity(fields.len());
                    for (fname, fexpr) in fields {
                        map.insert(fname.as_str(), self.lower_expr(fexpr, out));
                    }
                    let mut elems = Vec::with_capacity(st.fields.len());
                    for sf in &st.fields {
                        if let Some(&val) = map.get(sf.name.as_str()) {
                            elems.push(val);
                        } else {
                            return self.emit_diag(
                                "E0522",
                                format!(
                                    "missing field `{}` in struct literal `{struct_name}`",
                                    sf.name
                                ),
                                expr.span,
                            );
                        }
                    }
                    elems
                } else {
                    fields
                        .iter()
                        .map(|(_, fexpr)| self.lower_expr(fexpr, out))
                        .collect()
                };
                self.ir_fn.emit_typed(
                    expr_ir_ty,
                    InstData::CompositeConstruct {
                        elems: ordered_elems,
                    },
                )
            }
            HirExprKind::Unary { op, operand } => {
                self.lower_unary_expr(*op, operand, &expr_ir_ty, out)
            }
            HirExprKind::Binary { op, lhs, rhs } => {
                if is_assign_op(*op) {
                    return self.lower_assignment(*op, lhs, rhs, out);
                }
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    let bool_ty = IrType::Scalar(ScalarIrTy::Bool);
                    let slot = alloc_local_var(
                        self.ir_fn,
                        SmolStr::new(format!("__sc_{}", self.ir_fn.local_vars.len())),
                        bool_ty.clone(),
                    );
                    let l = self.lower_expr(lhs, out);
                    let init_addr = emit_local_addr(self.ir_fn, slot, &bool_ty);
                    let init_store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::LocalStore {
                            ptr: init_addr,
                            value: l,
                        },
                    );
                    out.push(StructuredStmt::Eval(init_store));

                    let mut rhs_stmts = Vec::new();
                    let r = self.lower_expr(rhs, &mut rhs_stmts);
                    let rhs_addr = emit_local_addr(self.ir_fn, slot, &bool_ty);
                    let rhs_store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::LocalStore {
                            ptr: rhs_addr,
                            value: r,
                        },
                    );
                    rhs_stmts.push(StructuredStmt::Eval(rhs_store));

                    let (then_body, else_body) = if *op == BinaryOp::And {
                        (rhs_stmts, Vec::new())
                    } else {
                        (Vec::new(), rhs_stmts)
                    };
                    out.push(StructuredStmt::If {
                        cond: l,
                        then_body,
                        else_body,
                    });

                    let load_addr = emit_local_addr(self.ir_fn, slot, &bool_ty);
                    let res = self
                        .ir_fn
                        .emit_typed(bool_ty, InstData::LocalLoad { ptr: load_addr });
                    out.push(StructuredStmt::Eval(res));
                    return res;
                }
                let l = self.lower_expr(lhs, out);
                let r = self.lower_expr(rhs, out);
                self.lower_binary_op(*op, l, r, &expr_ir_ty)
            }
            HirExprKind::Call {
                callee,
                resolved_fn,
                args,
            } => self.lower_call_expr(callee, *resolved_fn, args, &expr_ir_ty, expr.span, out),
            HirExprKind::MethodCall {
                receiver,
                method,
                resolved_fn,
                args,
                ..
            } => self.lower_method_call_expr(
                receiver,
                method,
                *resolved_fn,
                args,
                &expr_ir_ty,
                expr.span,
                out,
            ),
            HirExprKind::Field {
                receiver,
                field,
                field_index,
            } => self.lower_field_expr(receiver, field, *field_index, &expr_ir_ty, expr.span, out),
            HirExprKind::Index { receiver, index } => {
                self.lower_index_expr(receiver, index, &expr_ir_ty, expr.span, out)
            }
            HirExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => self
                .lower_if_branches(
                    cond,
                    then_branch,
                    else_branch.as_deref(),
                    expr.ty,
                    true,
                    out,
                )
                .unwrap_or_else(|| self.ir_fn.emit_typed(IrType::Void, InstData::Nop)),
            HirExprKind::Block(b) => {
                let (mut stmts, tail) = self.lower_block_value(b);
                out.append(&mut stmts);
                tail.unwrap_or_else(|| self.ir_fn.emit_typed(IrType::Void, InstData::Nop))
            }
            HirExprKind::Swizzle {
                base, components, ..
            } => {
                let base_val = self.lower_expr(base, out);
                self.lower_swizzle_read(base_val, components, &expr_ir_ty, expr.span)
            }
        }
    }

    fn lower_path_expr(
        &mut self,
        path: &SmolStr,
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        if !path.contains("::") {
            if let Some(&(ptr, ref ty)) = self.ptr_param_locals.get(path) {
                return self
                    .ir_fn
                    .emit_typed(ty.clone(), InstData::LocalLoad { ptr });
            }
            if let Some(&(slot, ref ty)) = self.local_var_slots.get(path) {
                let ty_cloned = ty.clone();
                let addr = emit_local_addr(self.ir_fn, slot, &ty_cloned);
                return self
                    .ir_fn
                    .emit_typed(ty_cloned, InstData::LocalLoad { ptr: addr });
            }
            if let Some(&id) = self.ssa_locals.get(path) {
                return id;
            }
            if let Some(&var_index) = self.workgroup_indices.get(path) {
                let ty = self.ir_fn.workgroup_vars[var_index as usize].effective_type();
                return self.ir_fn.emit_typed(
                    ty,
                    InstData::WorkgroupLoad {
                        var_index,
                        index: None,
                    },
                );
            }
            if let Some(const_expr) = self.module_consts.get(path).cloned() {
                return self.lower_expr(&const_expr, out);
            }
        }

        if let Some(spec_name) = path
            .strip_prefix("mc::spec::")
            .or_else(|| path.strip_prefix("spec::"))
        {
            return self.emit_spec_constant(spec_name, expr_ir_ty, span);
        }

        if let Some(param_name) = path
            .strip_prefix("mc::params::")
            .or_else(|| path.strip_prefix("pack::params::"))
        {
            return self.emit_pack_param_load(param_name, expr_ir_ty, span);
        }

        match path.as_str() {
            "mc::HitType::None" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0)),
            "mc::HitType::Triangle" | "mc::RayFlags::OPAQUE" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(1)),
            "mc::HitType::AABB" | "mc::HitType::Procedural" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(2)),
            "mc::RayFlags::NONE" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0)),
            "mc::RayFlags::NO_OPAQUE" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(2)),
            "mc::RayFlags::TERMINATE_ON_FIRST_HIT" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(4)),
            "mc::RayFlags::SKIP_CLOSEST_HIT" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(8)),
            "mc::RayFlags::CULL_BACK_FACING" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(16)),
            "mc::RayFlags::CULL_FRONT_FACING" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(32)),
            "mc::RayFlags::CULL_OPAQUE" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(64)),
            "mc::RayFlags::CULL_NO_OPAQUE" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(128)),
            "mc::RayFlags::FORCE_OPACITY_MICROMAP_2_STATE"
            | "RayFlags::FORCE_OPACITY_MICROMAP_2_STATE" => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0x400)),
            _ => self.emit_diag(
                "E0522",
                format!("unresolved path `{path}` during typed IR lowering"),
                span,
            ),
        }
    }

    fn lower_unary_expr(
        &mut self,
        op: UnaryOp,
        operand: &HirExpr,
        expr_ir_ty: &IrType,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        if matches!(op, UnaryOp::AddrOf) {
            match &operand.kind {
                HirExprKind::Path { path, .. }
                    if !path.contains("::") && self.workgroup_indices.contains_key(path) =>
                {
                    let var_index = self.workgroup_indices[path];
                    return self.ir_fn.emit(
                        None,
                        InstData::WorkgroupAddr {
                            var_index,
                            index: None,
                        },
                    );
                }
                HirExprKind::Index { receiver, index } => {
                    if let HirExprKind::Path { path, .. } = &receiver.kind
                        && !path.contains("::")
                        && let Some(&var_index) = self.workgroup_indices.get(path)
                    {
                        let idx_val = self.lower_expr(index, out);
                        return self.ir_fn.emit(
                            None,
                            InstData::WorkgroupAddr {
                                var_index,
                                index: Some(idx_val),
                            },
                        );
                    }
                    // Storage buffer element address for SSBO atomics (`&buf[i]`).
                    if let HirExprKind::Path { path, .. } = &receiver.kind
                        && let Some(desc_idx) = self.resolve_descriptor_path(path)
                    {
                        let idx_val = self.lower_expr(index, out);
                        let elem_ty = self.ir_ty_of(operand.ty);
                        let ptr_ty = IrType::Pointer {
                            pointee: Box::new(elem_ty),
                            align: 4,
                            storage: IrStorageClass::StorageBuffer,
                        };
                        let zero = self
                            .ir_fn
                            .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0));
                        return self.ir_fn.emit_typed(
                            ptr_ty,
                            InstData::DescriptorAccessChain {
                                descriptor_index: desc_idx,
                                indices: vec![zero, idx_val],
                            },
                        );
                    }
                }
                HirExprKind::Field {
                    receiver,
                    field,
                    field_index,
                } => {
                    if let Some((field_ptr, _, _, _)) =
                        self.try_lower_bda_field_ptr(receiver, field, *field_index, out)
                    {
                        return field_ptr;
                    }
                    if let HirExprKind::Index {
                        receiver: buf_recv,
                        index: idx_expr,
                    } = &receiver.kind
                        && let HirExprKind::Path { path, .. } = &buf_recv.kind
                        && let Some(desc_idx) = self.resolve_descriptor_path(path)
                    {
                        let elem_ty = self.ir_ty_of(receiver.ty);
                        let f_idx = field_index.or_else(|| match &elem_ty {
                            IrType::Struct(layout) => layout
                                .fields
                                .iter()
                                .position(|f| f.name == *field)
                                .map(|i| i as u32),
                            _ => None,
                        });
                        if let Some(idx) = f_idx {
                            let idx_val = self.lower_expr(idx_expr, out);
                            let field_ty = self.ir_ty_of(operand.ty);
                            let zero = self
                                .ir_fn
                                .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0));
                            let f_const = self.ir_fn.emit_typed(
                                IrType::Scalar(ScalarIrTy::U32),
                                InstData::ConstU32(idx),
                            );
                            let ptr_ty = IrType::Pointer {
                                pointee: Box::new(field_ty),
                                align: 4,
                                storage: IrStorageClass::StorageBuffer,
                            };
                            return self.ir_fn.emit_typed(
                                ptr_ty,
                                InstData::DescriptorAccessChain {
                                    descriptor_index: desc_idx,
                                    indices: vec![zero, idx_val, f_const],
                                },
                            );
                        }
                    }
                }
                HirExprKind::Path { path, .. }
                    if !path.contains("::") && self.ptr_param_locals.contains_key(path) =>
                {
                    return self.ptr_param_locals[path].0;
                }
                HirExprKind::Path { path, .. }
                    if !path.contains("::") && self.local_var_slots.contains_key(path) =>
                {
                    let (slot, ty) = self.local_var_slots[path].clone();
                    return emit_local_addr(self.ir_fn, slot, &ty);
                }
                _ => {}
            }
        }

        let val = self.lower_expr(operand, out);
        let operand_ty = self.inst_ty(val).clone();
        let val_ty = if *expr_ir_ty != IrType::Void {
            if operand_ty.is_64bit() && !expr_ir_ty.is_64bit() {
                operand_ty
            } else {
                expr_ir_ty.clone()
            }
        } else {
            operand_ty
        };
        match op {
            UnaryOp::Neg => {
                if val_ty.is_float() {
                    let zero = emit_zero_of_type(self.ir_fn, &val_ty);
                    self.ir_fn.emit_typed(val_ty, InstData::FSub32(zero, val))
                } else {
                    let zero = emit_zero_of_type(self.ir_fn, &val_ty);
                    self.ir_fn.emit_typed(val_ty, InstData::ISub32(zero, val))
                }
            }
            UnaryOp::Not => self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::Bool), InstData::LogicalNot(val)),
            UnaryOp::BitNot => self.ir_fn.emit_typed(val_ty, InstData::BitNot32(val)),
            UnaryOp::AddrOf => val,
        }
    }

    fn flush_output_shadow_if_needed(
        &mut self,
        target: &HirExpr,
        value: InstId,
        out: &mut Vec<StructuredStmt>,
    ) {
        if let HirExprKind::Path { path, .. } = &target.kind
            && !path.contains("::")
        {
            if let Some(&var_index) = self.stage_io_outputs.get(path) {
                let store = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::StoreStageOutput { var_index, value },
                );
                out.push(StructuredStmt::Eval(store));
            }
            if let Some(&var_index) = self.stage_var_indices.get(path) {
                let store = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::StoreStageVar { var_index, value },
                );
                out.push(StructuredStmt::Eval(store));
            }
            if let Some(&kind) = self.builtin_params.get(path) {
                let store = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::StoreBuiltin { kind, value });
                out.push(StructuredStmt::Eval(store));
            }
        }
    }

    fn lower_assignment(
        &mut self,
        op: BinaryOp,
        lhs: &HirExpr,
        rhs: &HirExpr,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let rhs_val = self.lower_expr(rhs, out);
        let lhs_ir_ty = self.ir_ty_of(lhs.ty);

        match &lhs.kind {
            HirExprKind::Path { path, .. } if !path.contains("::") => {
                if let Some(&var_index) = self.workgroup_indices.get(path) {
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let wg_ty = self.ir_fn.workgroup_vars[var_index as usize].effective_type();
                        let cur = self.ir_fn.emit_typed(
                            wg_ty,
                            InstData::WorkgroupLoad {
                                var_index,
                                index: None,
                            },
                        );
                        self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                    };
                    let store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::WorkgroupStore {
                            var_index,
                            index: None,
                            value: final_val,
                        },
                    );
                    out.push(StructuredStmt::Eval(store));
                    return final_val;
                }
                let stage_out = self.stage_io_outputs.get(path).copied();
                let stage_var_out = self.stage_var_indices.get(path).copied();
                let builtin_out = self.builtin_params.get(path).copied();
                if stage_out.is_some() || stage_var_out.is_some() || builtin_out.is_some() {
                    let shadow = self.local_var_slots.get(path).cloned();
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let cur = if let Some((slot, ref ty)) = shadow {
                            let addr = emit_local_addr(self.ir_fn, slot, ty);
                            self.ir_fn
                                .emit_typed(ty.clone(), InstData::LocalLoad { ptr: addr })
                        } else if let Some(kind) = builtin_out {
                            self.ir_fn.emit(
                                Some(lhs.ty),
                                InstData::LoadBuiltin {
                                    kind,
                                    component: None,
                                },
                            )
                        } else {
                            emit_zero_of_type(self.ir_fn, &lhs_ir_ty)
                        };
                        self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                    };
                    if let Some((slot, ty)) = shadow {
                        let addr = emit_local_addr(self.ir_fn, slot, &ty);
                        let shadow_store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::LocalStore {
                                ptr: addr,
                                value: final_val,
                            },
                        );
                        out.push(StructuredStmt::Eval(shadow_store));
                    }
                    if let Some(var_index) = stage_out {
                        let store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::StoreStageOutput {
                                var_index,
                                value: final_val,
                            },
                        );
                        out.push(StructuredStmt::Eval(store));
                    }
                    if let Some(var_index) = stage_var_out {
                        let store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::StoreStageVar {
                                var_index,
                                value: final_val,
                            },
                        );
                        out.push(StructuredStmt::Eval(store));
                    }
                    if let Some(kind) = builtin_out {
                        let store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::StoreBuiltin {
                                kind,
                                value: final_val,
                            },
                        );
                        out.push(StructuredStmt::Eval(store));
                    }
                    return final_val;
                }
                if let Some((ptr, ty)) = self.ptr_param_locals.get(path).cloned() {
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let cur = self.ir_fn.emit_typed(ty, InstData::LocalLoad { ptr });
                        self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                    };
                    let store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::LocalStore {
                            ptr,
                            value: final_val,
                        },
                    );
                    out.push(StructuredStmt::Eval(store));
                    return final_val;
                }
                if let Some((slot, ty)) = self.local_var_slots.get(path).cloned() {
                    let addr = emit_local_addr(self.ir_fn, slot, &ty);
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let cur = self.ir_fn.emit_typed(ty, InstData::LocalLoad { ptr: addr });
                        self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                    };
                    let store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::LocalStore {
                            ptr: addr,
                            value: final_val,
                        },
                    );
                    out.push(StructuredStmt::Eval(store));
                    return final_val;
                }
                self.emit_diag(
                    "E0523",
                    format!("cannot assign to immutable or unknown binding `{path}`"),
                    lhs.span,
                )
            }
            HirExprKind::Index { receiver, index } => {
                let idx_val = self.lower_expr(index, out);
                if let HirExprKind::Path { path, .. } = &receiver.kind
                    && !path.contains("::")
                    && let Some(&var_index) = self.workgroup_indices.get(path)
                {
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let elem_ty =
                            match self.ir_fn.workgroup_vars[var_index as usize].effective_type() {
                                IrType::Array { element, .. } => *element,
                                other => other,
                            };
                        let cur = self.ir_fn.emit_typed(
                            elem_ty,
                            InstData::WorkgroupLoad {
                                var_index,
                                index: Some(idx_val),
                            },
                        );
                        self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                    };
                    let store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::WorkgroupStore {
                            var_index,
                            index: Some(idx_val),
                            value: final_val,
                        },
                    );
                    out.push(StructuredStmt::Eval(store));
                    return final_val;
                }
                if let Some((base_ptr, base_ty)) = self.try_lower_local_lvalue_ptr(receiver, out) {
                    let elem_ty = match base_ty {
                        IrType::Array { element, .. } => *element,
                        IrType::Vector { scalar, .. } => IrType::Scalar(scalar),
                        IrType::Matrix { scalar, rows, .. } => IrType::Vector { scalar, len: rows },
                        _ => lhs_ir_ty.clone(),
                    };
                    let elem_ptr = self.ir_fn.emit_typed(
                        IrType::Pointer {
                            pointee: Box::new(elem_ty.clone()),
                            align: 4,
                            storage: IrStorageClass::Function,
                        },
                        InstData::AccessChain {
                            base_ptr,
                            indices: vec![idx_val],
                        },
                    );
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let cur = self
                            .ir_fn
                            .emit_typed(elem_ty, InstData::LocalLoad { ptr: elem_ptr });
                        self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                    };
                    let store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::LocalStore {
                            ptr: elem_ptr,
                            value: final_val,
                        },
                    );
                    out.push(StructuredStmt::Eval(store));
                    return final_val;
                }
                if let HirExprKind::Path { path, .. } = &receiver.kind
                    && let Some(desc_idx) = self.resolve_descriptor_path(path)
                {
                    let zero = self
                        .ir_fn
                        .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0));
                    let ptr = self.ir_fn.emit_typed(
                        IrType::Pointer {
                            pointee: Box::new(lhs_ir_ty.clone()),
                            align: 4,
                            storage: IrStorageClass::StorageBuffer,
                        },
                        InstData::DescriptorAccessChain {
                            descriptor_index: desc_idx,
                            indices: vec![zero, idx_val],
                        },
                    );
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let cur = self
                            .ir_fn
                            .emit_typed(lhs_ir_ty.clone(), InstData::ResourceLoad { ptr });
                        self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                    };
                    let store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::DescriptorStore {
                            ptr,
                            value: final_val,
                        },
                    );
                    out.push(StructuredStmt::Eval(store));
                    return final_val;
                }
                self.emit_diag(
                    "E0523",
                    "unsupported indexed assignment target during IR lowering",
                    lhs.span,
                )
            }
            HirExprKind::Swizzle {
                base, components, ..
            } => self.lower_swizzle_write(base, components, op, rhs_val, &lhs_ir_ty, lhs.span, out),
            HirExprKind::Field {
                receiver,
                field,
                field_index,
            } => {
                // Vector write swizzle encoded as `Field` (e.g. `v.xy = rhs` or `v.y = rhs`).
                if let Some(comps) = parse_swizzle_str(field.as_str())
                    && matches!(self.ir_ty_of(receiver.ty), IrType::Vector { .. })
                {
                    return self.lower_swizzle_write(
                        receiver, &comps, op, rhs_val, &lhs_ir_ty, lhs.span, out,
                    );
                }

                // Local struct or `inout`/`out` parameter field write (`s.field = rhs`).
                if let Some((base_ptr, base_ty)) = self.try_lower_local_lvalue_ptr(receiver, out) {
                    let f_idx = field_index.or_else(|| match &base_ty {
                        IrType::Struct(layout) => layout
                            .fields
                            .iter()
                            .position(|f| f.name == *field)
                            .map(|i| i as u32),
                        _ => None,
                    });
                    if let Some(idx) = f_idx {
                        let idx_const = self
                            .ir_fn
                            .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(idx));
                        let field_ty = match &base_ty {
                            IrType::Struct(layout) => layout
                                .fields
                                .get(idx as usize)
                                .map_or_else(|| lhs_ir_ty.clone(), |f| f.ty.clone()),
                            _ => lhs_ir_ty.clone(),
                        };
                        let field_ptr = self.ir_fn.emit_typed(
                            IrType::Pointer {
                                pointee: Box::new(field_ty.clone()),
                                align: 4,
                                storage: IrStorageClass::Function,
                            },
                            InstData::AccessChain {
                                base_ptr,
                                indices: vec![idx_const],
                            },
                        );
                        let final_val = if op == BinaryOp::Assign {
                            rhs_val
                        } else {
                            let cur = self
                                .ir_fn
                                .emit_typed(field_ty, InstData::LocalLoad { ptr: field_ptr });
                            self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                        };
                        let store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::LocalStore {
                                ptr: field_ptr,
                                value: final_val,
                            },
                        );
                        out.push(StructuredStmt::Eval(store));
                        return final_val;
                    }
                }

                // Built-in or user SSBO element field write (`buf[i].field = rhs` or `inst.field = rhs`).
                let ssbo_elem = if let HirExprKind::Path { path, .. } = &receiver.kind
                    && !path.contains("::")
                {
                    self.desc_elem_bindings.get(path).cloned()
                } else {
                    self.try_lower_descriptor_elem_index(receiver, out)
                };
                if let Some(elem_binding) = ssbo_elem {
                    let canonical =
                        canonicalize_builtin_field_name(elem_binding.elem_desc, field.as_str());
                    if let Some(f_idx) = elem_binding.elem_desc.field_index(canonical) {
                        let field_ty = elem_binding.elem_desc.field(canonical).map_or_else(
                            || lhs_ir_ty.clone(),
                            |f| IrType::from_builtin_field_type(f.ty),
                        );
                        let items_idx = self.ir_fn.emit_typed(
                            IrType::Scalar(ScalarIrTy::U32),
                            InstData::ConstU32(elem_binding.items_field_index),
                        );
                        let field_idx_const = self
                            .ir_fn
                            .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(f_idx));
                        let ptr = self.ir_fn.emit_typed(
                            IrType::Pointer {
                                pointee: Box::new(field_ty.clone()),
                                align: 4,
                                storage: IrStorageClass::StorageBuffer,
                            },
                            InstData::DescriptorAccessChain {
                                descriptor_index: elem_binding.descriptor_index,
                                indices: vec![items_idx, elem_binding.index_val, field_idx_const],
                            },
                        );
                        let final_val = if op == BinaryOp::Assign {
                            rhs_val
                        } else {
                            let cur = self
                                .ir_fn
                                .emit_typed(field_ty, InstData::ResourceLoad { ptr });
                            self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                        };
                        let store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::DescriptorStore {
                                ptr,
                                value: final_val,
                            },
                        );
                        out.push(StructuredStmt::Eval(store));
                        return final_val;
                    }
                }
                if let HirExprKind::Index {
                    receiver: buf_recv,
                    index: idx_expr,
                } = &receiver.kind
                    && let HirExprKind::Path { path, .. } = &buf_recv.kind
                    && let Some(desc_idx) = self.resolve_descriptor_path(path)
                {
                    let elem_ty = self.ir_ty_of(receiver.ty);
                    let f_idx = field_index.or_else(|| match &elem_ty {
                        IrType::Struct(layout) => layout
                            .fields
                            .iter()
                            .position(|f| f.name == *field)
                            .map(|i| i as u32),
                        _ => None,
                    });
                    if let Some(idx) = f_idx {
                        let idx_val = self.lower_expr(idx_expr, out);
                        let zero = self
                            .ir_fn
                            .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0));
                        let f_const = self
                            .ir_fn
                            .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(idx));
                        let ptr = self.ir_fn.emit_typed(
                            IrType::Pointer {
                                pointee: Box::new(lhs_ir_ty.clone()),
                                align: 4,
                                storage: IrStorageClass::StorageBuffer,
                            },
                            InstData::DescriptorAccessChain {
                                descriptor_index: desc_idx,
                                indices: vec![zero, idx_val, f_const],
                            },
                        );
                        let final_val = if op == BinaryOp::Assign {
                            rhs_val
                        } else {
                            let cur = self
                                .ir_fn
                                .emit_typed(lhs_ir_ty.clone(), InstData::ResourceLoad { ptr });
                            self.lower_binary_op(op, cur, rhs_val, &lhs_ir_ty)
                        };
                        let store = self.ir_fn.emit_typed(
                            IrType::Void,
                            InstData::DescriptorStore {
                                ptr,
                                value: final_val,
                            },
                        );
                        out.push(StructuredStmt::Eval(store));
                        return final_val;
                    }
                }
                if let Some((field_ptr, field_ty, field_align, _)) =
                    self.try_lower_bda_field_ptr(receiver, field, *field_index, out)
                {
                    let final_val = if op == BinaryOp::Assign {
                        rhs_val
                    } else {
                        let cur = self.ir_fn.emit_typed(
                            field_ty.clone(),
                            InstData::BdaLoad {
                                ptr: field_ptr,
                                access: MemoryAccess {
                                    align: field_align,
                                    non_writable: false,
                                    restrict: false,
                                },
                            },
                        );
                        self.lower_binary_op(op, cur, rhs_val, &field_ty)
                    };
                    let store = self.ir_fn.emit_typed(
                        IrType::Void,
                        InstData::BdaStore {
                            ptr: field_ptr,
                            value: final_val,
                            access: MemoryAccess {
                                align: field_align,
                                non_writable: false,
                                restrict: false,
                            },
                        },
                    );
                    out.push(StructuredStmt::Eval(store));
                    return final_val;
                }
                self.emit_diag(
                    "E0523",
                    format!("unsupported field assignment `.{field}` during IR lowering"),
                    lhs.span,
                )
            }
            _ => self.emit_diag(
                "E0523",
                "unsupported assignment target during IR lowering",
                lhs.span,
            ),
        }
    }

    fn try_lower_local_lvalue_ptr(
        &mut self,
        expr: &HirExpr,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<(InstId, IrType)> {
        match &expr.kind {
            HirExprKind::Path { path, .. } if !path.contains("::") => {
                if let Some((ptr, ty)) = self.ptr_param_locals.get(path).cloned() {
                    return Some((ptr, ty));
                }
                let (slot, ty) = self.local_var_slots.get(path)?.clone();
                let addr = emit_local_addr(self.ir_fn, slot, &ty);
                Some((addr, ty))
            }
            HirExprKind::Field {
                receiver,
                field,
                field_index,
            } => {
                let (base_ptr, base_ty) = self.try_lower_local_lvalue_ptr(receiver, out)?;
                let (idx, field_ty) = match &base_ty {
                    IrType::Struct(layout) => {
                        let idx = field_index.or_else(|| {
                            layout
                                .fields
                                .iter()
                                .position(|f| f.name == *field)
                                .map(|i| i as u32)
                        })?;
                        let f_ty = layout.fields.get(idx as usize)?.ty.clone();
                        (idx, f_ty)
                    }
                    _ => return None,
                };
                let idx_const = self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(idx));
                let ptr = self.ir_fn.emit_typed(
                    IrType::Pointer {
                        pointee: Box::new(field_ty.clone()),
                        align: 4,
                        storage: IrStorageClass::Function,
                    },
                    InstData::AccessChain {
                        base_ptr,
                        indices: vec![idx_const],
                    },
                );
                Some((ptr, field_ty))
            }
            HirExprKind::Index { receiver, index } => {
                let (base_ptr, base_ty) = self.try_lower_local_lvalue_ptr(receiver, out)?;
                let idx_val = self.lower_expr(index, out);
                let elem_ty = match base_ty {
                    IrType::Array { element, .. } => *element,
                    _ => return None,
                };
                let ptr = self.ir_fn.emit_typed(
                    IrType::Pointer {
                        pointee: Box::new(elem_ty.clone()),
                        align: 4,
                        storage: IrStorageClass::Function,
                    },
                    InstData::AccessChain {
                        base_ptr,
                        indices: vec![idx_val],
                    },
                );
                Some((ptr, elem_ty))
            }
            _ => None,
        }
    }

    fn lower_swizzle_read(
        &mut self,
        base_val: InstId,
        components: &[u8],
        expr_ir_ty: &IrType,
        span: Span,
    ) -> InstId {
        if components.is_empty() {
            return self.emit_diag("E0524", "empty vector swizzle", span);
        }
        let scalar = self
            .inst_ty(base_val)
            .scalar_ty()
            .or_else(|| expr_ir_ty.scalar_ty())
            .unwrap_or(ScalarIrTy::F32);
        if components.len() == 1 {
            return self.ir_fn.emit_typed(
                IrType::Scalar(scalar),
                InstData::CompositeExtract {
                    composite: base_val,
                    indices: vec![u32::from(components[0])],
                },
            );
        }
        let vec_ty = IrType::Vector {
            scalar,
            len: components.len() as u8,
        };
        let comps: Vec<u32> = components.iter().map(|&c| u32::from(c)).collect();
        self.ir_fn.emit_typed(
            vec_ty,
            InstData::VectorShuffle {
                lhs: base_val,
                rhs: base_val,
                components: comps,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_swizzle_write(
        &mut self,
        base: &HirExpr,
        components: &[u8],
        op: BinaryOp,
        rhs_val: InstId,
        lhs_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let Some((base_ptr, base_ty)) = self.try_lower_local_lvalue_ptr(base, out) else {
            return self.emit_diag(
                "E0524",
                "vector write swizzle target must be a mutable local variable",
                span,
            );
        };
        let IrType::Vector { scalar, len } = base_ty else {
            return self.emit_diag(
                "E0524",
                "vector write swizzle target is not a vector type",
                span,
            );
        };
        let old_vec = self
            .ir_fn
            .emit_typed(base_ty.clone(), InstData::LocalLoad { ptr: base_ptr });

        let final_val = if op == BinaryOp::Assign {
            rhs_val
        } else {
            let old_swiz = self.lower_swizzle_read(old_vec, components, lhs_ir_ty, span);
            self.lower_binary_op(op, old_swiz, rhs_val, lhs_ir_ty)
        };

        let updated = if components.len() == 1 {
            self.ir_fn.emit_typed(
                base_ty.clone(),
                InstData::CompositeInsert {
                    composite: old_vec,
                    value: final_val,
                    indices: vec![u32::from(components[0])],
                },
            )
        } else {
            let mut acc = old_vec;
            for (src_idx, &dst_comp) in components.iter().enumerate() {
                let elem = self.ir_fn.emit_typed(
                    IrType::Scalar(scalar),
                    InstData::CompositeExtract {
                        composite: final_val,
                        indices: vec![src_idx as u32],
                    },
                );
                if dst_comp < len {
                    acc = self.ir_fn.emit_typed(
                        base_ty.clone(),
                        InstData::CompositeInsert {
                            composite: acc,
                            value: elem,
                            indices: vec![u32::from(dst_comp)],
                        },
                    );
                }
            }
            acc
        };

        let store = self.ir_fn.emit_typed(
            IrType::Void,
            InstData::LocalStore {
                ptr: base_ptr,
                value: updated,
            },
        );
        out.push(StructuredStmt::Eval(store));
        self.flush_output_shadow_if_needed(base, updated, out);
        final_val
    }

    fn emit_spec_constant(&mut self, spec_name: &str, expr_ir_ty: &IrType, span: Span) -> InstId {
        let (spec_id, scalar, default_bits) = if let Some(def) = self.spec_defaults.get(spec_name) {
            let scalar = pack_param_type_to_scalar(def.ty);
            (def.spec_id, scalar, def.default_bits)
        } else if self.pack_schema.is_some() {
            return self.emit_diag(
                "E0525",
                format!(
                    "unknown specialization constant `{spec_name}` not declared in active `pack.vsl` schema"
                ),
                span,
            );
        } else {
            let scalar = expr_ir_ty.scalar_ty().unwrap_or(ScalarIrTy::U32);
            (0, scalar, 0)
        };
        self.ir_fn.emit_typed(
            IrType::Scalar(scalar),
            InstData::SpecConstant {
                spec_id,
                scalar,
                default_bits,
            },
        )
    }

    fn emit_pack_param_load(
        &mut self,
        param_name: &str,
        expr_ir_ty: &IrType,
        span: Span,
    ) -> InstId {
        let layout = if let Some(&layout) = self.param_layouts.get(param_name) {
            layout
        } else if self.pack_schema.is_some() {
            return self.emit_diag(
                "E0525",
                format!(
                    "unknown pack parameter `{param_name}` not declared in active `pack.vsl` schema"
                ),
                span,
            );
        } else {
            let ty = match expr_ir_ty {
                IrType::Vector { len: 3, .. } => PackParamType::Vec3,
                IrType::Scalar(ScalarIrTy::Bool) => PackParamType::Bool,
                IrType::Scalar(ScalarIrTy::I32) => PackParamType::I32,
                IrType::Scalar(ScalarIrTy::U32) => PackParamType::U32,
                _ => PackParamType::F32,
            };
            PackParamLayout {
                byte_offset: 0,
                byte_size: ty.byte_size(),
                ty,
            }
        };

        let Some(world_desc) = self.find_world_descriptor() else {
            return self.emit_diag(
                "E0525",
                format!(
                    "cannot load pack parameter `{param_name}`: `WorldUBO` (Set 0 binding 0) is not bound"
                ),
                span,
            );
        };

        let base_u64 = self.emit_builtin_struct_field_load(
            world_desc,
            &WORLD_UBO_LAYOUT,
            "pack_params_bda",
            IrStorageClass::Uniform,
            span,
        );
        self.ir_fn.uses_bda = true;

        if layout.ty == PackParamType::Vec3 {
            let f32_ty = IrType::Scalar(ScalarIrTy::F32);
            let mut lanes = [base_u64; 3];
            for (i, lane) in lanes.iter_mut().enumerate() {
                let off_bytes = u64::from(layout.byte_offset) + (i as u64) * 4;
                let off_u64 = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U64),
                    InstData::ConstU64(off_bytes),
                );
                let addr = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U64),
                    InstData::IAdd64(base_u64, off_u64),
                );
                let ptr = self.ir_fn.emit_typed(
                    IrType::Pointer {
                        pointee: Box::new(f32_ty.clone()),
                        align: 4,
                        storage: IrStorageClass::PhysicalStorageBuffer,
                    },
                    InstData::ConvertUToPtr {
                        addr_u64: addr,
                        align: 4,
                    },
                );
                *lane = self.ir_fn.emit_typed(
                    f32_ty.clone(),
                    InstData::BdaLoad {
                        ptr,
                        access: MemoryAccess {
                            align: 4,
                            non_writable: true,
                            restrict: false,
                        },
                    },
                );
            }
            return self.ir_fn.emit_typed(
                IrType::Vector {
                    scalar: ScalarIrTy::F32,
                    len: 3,
                },
                InstData::MakeVec3(lanes[0], lanes[1], lanes[2]),
            );
        }

        let target_ty = IrType::from_pack_param_type(layout.ty);
        let align = layout.ty.std430_align();
        let off_u64 = self.ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U64),
            InstData::ConstU64(u64::from(layout.byte_offset)),
        );
        let addr = self.ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U64),
            InstData::IAdd64(base_u64, off_u64),
        );
        let ptr = self.ir_fn.emit_typed(
            IrType::Pointer {
                pointee: Box::new(target_ty.clone()),
                align,
                storage: IrStorageClass::PhysicalStorageBuffer,
            },
            InstData::ConvertUToPtr {
                addr_u64: addr,
                align,
            },
        );
        self.ir_fn.emit_typed(
            target_ty,
            InstData::BdaLoad {
                ptr,
                access: MemoryAccess {
                    align,
                    non_writable: true,
                    restrict: false,
                },
            },
        )
    }

    fn find_world_descriptor(&self) -> Option<u32> {
        self.ir_fn
            .descriptors
            .iter()
            .position(|d| d.set == DESCRIPTOR_SET_PER_PASS && d.binding == 0)
            .map(|i| i as u32)
    }

    /// Loads a `WorldUBO` member using the type and offset declared by the canonical engine
    /// layout, so every `mc::world.*` read is typed from the same 544-byte schema.
    fn load_world_field(&mut self, field_name: &str, span: Span) -> InstId {
        let Some(world_desc) = self.find_world_descriptor() else {
            return self.emit_diag(
                "E0525",
                format!("cannot load `WorldUBO::{field_name}`: Set 0 binding 0 is not bound"),
                span,
            );
        };
        self.emit_builtin_struct_field_load(
            world_desc,
            &WORLD_UBO_LAYOUT,
            field_name,
            IrStorageClass::Uniform,
            span,
        )
    }

    fn world_game_time_sec(&mut self, span: Span) -> InstId {
        self.load_world_field("game_time_sec", span)
    }

    fn world_rain(&mut self, span: Span) -> InstId {
        self.load_world_field("rain", span)
    }

    fn world_thunder(&mut self, span: Span) -> InstId {
        self.load_world_field("thunder", span)
    }

    /// CPU-integrated wind phase (`sum(dt * windSpeed)`) at `WorldUBO` offset 536. Unlike
    /// `game_time_sec * wind_speed`, this stays continuous when weather changes the wind speed.
    fn world_wind_phase(&mut self, span: Span) -> InstId {
        self.load_world_field("wind_phase", span)
    }

    fn world_eye_in_fluid(&mut self, span: Span) -> InstId {
        self.load_world_field("eye_in_fluid", span)
    }

    fn world_sun_dir(&mut self, span: Span) -> InstId {
        self.load_world_field("sun_dir", span)
    }

    /// Reads one `vec4` slot of the engine's `packParams[12]` array.
    fn world_pack_param_slot(&mut self, slot: u32, span: Span) -> InstId {
        if slot >= 12 {
            return self.emit_diag(
                "E0526",
                format!("`packParams[{slot}]` is out of range (expected 0..=11)"),
                span,
            );
        }
        self.load_world_field(world_pack_param_field_name(slot), span)
    }

    /// `WorldStateCollector.kt` writes a non-zero atmosphere density into `packParams[11].w`
    /// only when the biome/dimension telemetry tail is valid.
    fn lower_mc_has_biome_telemetry(&mut self, span: Span) -> InstId {
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let validity = self.world_pack_param_slot(11, span);
        let density = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: validity,
                indices: vec![3],
            },
        );
        let threshold = self.ir_fn.emit_typed(f32_ty, InstData::const_f32(0.001));
        self.ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::Bool),
            InstData::FCmpGt(density, threshold),
        )
    }

    /// `mc::dimension_id()`: mirrors `vsDimensionId()`, round-to-nearest on
    /// `packParams[10].z`, with 0 (Overworld) when telemetry is absent.
    fn lower_mc_dimension_id(&mut self, span: Span) -> InstId {
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let i32_ty = IrType::Scalar(ScalarIrTy::I32);
        let valid = self.lower_mc_has_biome_telemetry(span);
        let telemetry = self.world_pack_param_slot(10, span);
        let raw = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: telemetry,
                indices: vec![2],
            },
        );
        let half = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.5));
        let rounded = self.ir_fn.emit_typed(f32_ty, InstData::FAdd32(raw, half));
        let id = self
            .ir_fn
            .emit_typed(i32_ty.clone(), InstData::CastToI32(rounded));
        let overworld = self.ir_fn.emit_typed(i32_ty, InstData::ConstI32(0));
        self.ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::I32),
            InstData::Select {
                cond: valid,
                on_true: id,
                on_false: overworld,
            },
        )
    }

    /// `mc::biome_cloud_scale()`: mirrors `vsBiomeCloudScale()`, clamped to `0.0..=1.5`,
    /// with the neutral `1.0` fallback when telemetry is absent.
    fn lower_mc_biome_cloud_scale(&mut self, span: Span) -> InstId {
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let valid = self.lower_mc_has_biome_telemetry(span);
        let telemetry = self.world_pack_param_slot(10, span);
        let raw = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: telemetry,
                indices: vec![3],
            },
        );
        let zero = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.0));
        let one = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
        let max_scale = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(1.5));
        let clamped = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::GlslStd450 {
                op: GlslStd450Op::FClamp,
                args: vec![raw, zero, max_scale],
            },
        );
        self.ir_fn.emit_typed(
            f32_ty,
            InstData::Select {
                cond: valid,
                on_true: clamped,
                on_false: one,
            },
        )
    }

    fn resolve_descriptor_path(&self, path: &str) -> Option<u32> {
        if let Some(res_name) = path.strip_prefix("mc::resources::")
            && let Some(slot) = self
                .pack_schema
                .and_then(|s| s.resource_slots.get(res_name).copied())
            && let Some(idx) = self
                .ir_fn
                .descriptors
                .iter()
                .position(|d| d.set == DESCRIPTOR_SET_PACK_RESOURCES && d.binding == slot)
        {
            return Some(idx as u32);
        }
        let short = path.rsplit("::").next().unwrap_or(path);
        self.descriptor_indices
            .get(path)
            .or_else(|| self.descriptor_indices.get(short))
            .copied()
    }

    /// Emits `DescriptorAccessChain` + typed `DescriptorLoad` (and `CompositeConstruct` for
    /// vectors/matrices) using exact [`BuiltinStructDesc`] field index and byte offset.
    fn emit_builtin_struct_field_load(
        &mut self,
        descriptor_index: u32,
        struct_desc: &BuiltinStructDesc,
        field_name: &str,
        storage: IrStorageClass,
        span: Span,
    ) -> InstId {
        let canonical_field = canonicalize_builtin_field_name(struct_desc, field_name);
        let Some(field_idx) = struct_desc.field_index(canonical_field) else {
            return self.emit_diag(
                "E0526",
                format!(
                    "unknown field `{field_name}` on built-in buffer `{}`",
                    struct_desc.name
                ),
                span,
            );
        };
        let field_desc = &struct_desc.fields[field_idx as usize];
        let field_ir_ty = IrType::from_builtin_field_type(field_desc.ty);
        let idx_const = self.ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::ConstU32(field_idx),
        );
        let chain = self.ir_fn.emit_typed(
            IrType::Pointer {
                pointee: Box::new(field_ir_ty.clone()),
                align: 4,
                storage,
            },
            InstData::DescriptorAccessChain {
                descriptor_index,
                indices: vec![idx_const],
            },
        );
        self.ir_fn
            .emit_typed(field_ir_ty, InstData::ResourceLoad { ptr: chain })
    }

    fn try_lower_descriptor_elem_index(
        &mut self,
        expr: &HirExpr,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<DescriptorElemBinding> {
        let HirExprKind::Index { receiver, index } = &expr.kind else {
            return None;
        };
        match &receiver.kind {
            // `mc::instances[i]` / `instances[i]`
            HirExprKind::Path { path, .. } if path == "mc::instances" || path == "instances" => {
                let desc_idx = self
                    .ir_fn
                    .descriptors
                    .iter()
                    .position(|d| d.set == DESCRIPTOR_SET_SCENE_GLOBALS && d.binding == 1)?
                    as u32;
                let idx_val = self.lower_expr(index, out);
                Some(DescriptorElemBinding {
                    descriptor_index: desc_idx,
                    items_field_index: INSTANCE_META_SSBO_LAYOUT.field_index("items").unwrap_or(0),
                    index_val: idx_val,
                    elem_desc: &INSTANCE_META_LAYOUT,
                })
            }
            // `mc::lights.items[i]` or `mc::reservoirs.items[i]`
            HirExprKind::Field {
                receiver: ssbo_recv,
                field,
                ..
            } if field == "items" => {
                let HirExprKind::Path { path, .. } = &ssbo_recv.kind else {
                    return None;
                };
                if path == "mc::lights" || path == "lights" {
                    let desc_idx = self
                        .ir_fn
                        .descriptors
                        .iter()
                        .position(|d| d.set == DESCRIPTOR_SET_SCENE_GLOBALS && d.binding == 3)?
                        as u32;
                    let idx_val = self.lower_expr(index, out);
                    Some(DescriptorElemBinding {
                        descriptor_index: desc_idx,
                        items_field_index: EMISSIVE_LIGHT_SSBO_LAYOUT
                            .field_index("items")
                            .unwrap_or(4),
                        index_val: idx_val,
                        elem_desc: &EMISSIVE_LIGHT_LAYOUT,
                    })
                } else if path == "mc::reservoirs" || path == "reservoirs" {
                    let desc_idx = self
                        .ir_fn
                        .descriptors
                        .iter()
                        .position(|d| d.set == DESCRIPTOR_SET_SCENE_GLOBALS && d.binding == 4)?
                        as u32;
                    let idx_val = self.lower_expr(index, out);
                    Some(DescriptorElemBinding {
                        descriptor_index: desc_idx,
                        items_field_index: RESTIR_RESERVOIR_SSBO_LAYOUT
                            .field_index("items")
                            .unwrap_or(4),
                        index_val: idx_val,
                        elem_desc: &RESTIR_RESERVOIR_LAYOUT,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn emit_descriptor_elem_field_load(
        &mut self,
        binding: &DescriptorElemBinding,
        field_name: &str,
        span: Span,
    ) -> InstId {
        let canonical = canonicalize_builtin_field_name(binding.elem_desc, field_name);
        let Some(elem_field_idx) = binding.elem_desc.field_index(canonical) else {
            return self.emit_diag(
                "E0526",
                format!(
                    "unknown field `{field_name}` on built-in SSBO element `{}`",
                    binding.elem_desc.name
                ),
                span,
            );
        };
        let field_desc = &binding.elem_desc.fields[elem_field_idx as usize];
        let field_ir_ty = IrType::from_builtin_field_type(field_desc.ty);
        let items_idx_const = self.ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::ConstU32(binding.items_field_index),
        );
        let field_idx_const = self.ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::ConstU32(elem_field_idx),
        );
        let chain = self.ir_fn.emit_typed(
            IrType::Pointer {
                pointee: Box::new(field_ir_ty.clone()),
                align: 4,
                storage: IrStorageClass::StorageBuffer,
            },
            InstData::DescriptorAccessChain {
                descriptor_index: binding.descriptor_index,
                indices: vec![items_idx_const, binding.index_val, field_idx_const],
            },
        );
        self.ir_fn
            .emit_typed(field_ir_ty, InstData::ResourceLoad { ptr: chain })
    }

    /// Resolves a BDA pointer base expression (or nested BDA field chain) into
    /// `(bda_ptr_inst, pointee_ir_ty, align, is_mut)`.
    fn try_lower_bda_field_base(
        &mut self,
        expr: &HirExpr,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<(InstId, IrType, u32, bool)> {
        if let Type::Pointer {
            pointee,
            align,
            mutable,
        } = self.hir.types.get(expr.ty)
        {
            let (pointee_id, ptr_align, is_mut) = (*pointee, *align, *mutable);
            let base_ptr = self.lower_expr(expr, out);
            let pointee_ir_ty = self.ir_ty_of(pointee_id);
            return Some((base_ptr, pointee_ir_ty, ptr_align, is_mut));
        }
        if let HirExprKind::Field {
            receiver,
            field,
            field_index,
        } = &expr.kind
        {
            return self.try_lower_bda_field_ptr(receiver, field, *field_index, out);
        }
        None
    }

    /// Computes a BDA field pointer using 64-bit integer offset math (`ConvertPtrToU` +
    /// `IAdd64` + `ConvertUToPtr`) and returns `(field_ptr, field_ty, field_align, is_mut)`.
    fn try_lower_bda_field_ptr(
        &mut self,
        receiver: &HirExpr,
        field: &SmolStr,
        field_index: Option<u32>,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<(InstId, IrType, u32, bool)> {
        let (base_ptr, base_ir_ty, ptr_align, is_mut) =
            self.try_lower_bda_field_base(receiver, out)?;
        let IrType::Struct(layout) = base_ir_ty else {
            return None;
        };
        let f_idx = field_index
            .and_then(|idx| {
                usize::try_from(idx)
                    .ok()
                    .filter(|&i| i < layout.fields.len())
            })
            .or_else(|| layout.fields.iter().position(|f| f.name == *field))?;
        let f_layout = &layout.fields[f_idx];
        let byte_offset = f_layout.byte_offset;
        let field_ty = f_layout.ty.clone();
        let field_align = if byte_offset == 0 {
            ptr_align.max(1)
        } else {
            ptr_align.max(1).min(1u32 << byte_offset.trailing_zeros())
        };
        self.ir_fn.uses_bda = true;
        let u64_ty = IrType::Scalar(ScalarIrTy::U64);
        let base_u64 = self
            .ir_fn
            .emit_typed(u64_ty.clone(), InstData::ConvertPtrToU(base_ptr));
        let addr_u64 = if byte_offset == 0 {
            base_u64
        } else {
            let off_const = self
                .ir_fn
                .emit_typed(u64_ty.clone(), InstData::ConstU64(u64::from(byte_offset)));
            self.ir_fn
                .emit_typed(u64_ty, InstData::IAdd64(base_u64, off_const))
        };
        let ptr_ty = IrType::Pointer {
            pointee: Box::new(field_ty.clone()),
            align: field_align,
            storage: IrStorageClass::PhysicalStorageBuffer,
        };
        let field_ptr = self.ir_fn.emit_typed(
            ptr_ty,
            InstData::ConvertUToPtr {
                addr_u64,
                align: field_align,
            },
        );
        Some((field_ptr, field_ty, field_align, is_mut))
    }

    fn lower_field_expr(
        &mut self,
        receiver: &HirExpr,
        field: &SmolStr,
        field_index: Option<u32>,
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        // `mc::params.<key>` / `params.<key>` / `pack::params.<key>`
        if let HirExprKind::Path { path, .. } = &receiver.kind
            && matches!(path.as_str(), "mc::params" | "params" | "pack::params")
        {
            return self.emit_pack_param_load(field.as_str(), expr_ir_ty, span);
        }

        // Builtin parameter swizzle (`gid.xy`, `gid.x`, `frag_coord.xy`, etc.)
        if let HirExprKind::Path { path, .. } = &receiver.kind
            && !path.contains("::")
        {
            if let Some(&kind) = self.builtin_params.get(path)
                && let Some(comps) = parse_swizzle_str(field.as_str())
            {
                let scalar = match kind {
                    BuiltinKind::FragCoord
                    | BuiltinKind::Position
                    | BuiltinKind::FragDepth
                    | BuiltinKind::WorldRayOriginKhr
                    | BuiltinKind::WorldRayDirectionKhr
                    | BuiltinKind::ObjectRayOriginKhr
                    | BuiltinKind::ObjectRayDirectionKhr
                    | BuiltinKind::TessCoord
                    | BuiltinKind::SamplePosition
                    | BuiltinKind::ClipDistance
                    | BuiltinKind::CullDistance => ScalarIrTy::F32,
                    BuiltinKind::VertexIndex | BuiltinKind::InstanceIndex => ScalarIrTy::I32,
                    BuiltinKind::FrontFacing => ScalarIrTy::Bool,
                    _ => ScalarIrTy::U32,
                };
                if comps.len() == 1 {
                    return self.ir_fn.emit_typed(
                        IrType::Scalar(scalar),
                        InstData::LoadBuiltin {
                            kind,
                            component: Some(u32::from(comps[0])),
                        },
                    );
                }
                let elems: Vec<InstId> = comps
                    .iter()
                    .map(|&c| {
                        self.ir_fn.emit_typed(
                            IrType::Scalar(scalar),
                            InstData::LoadBuiltin {
                                kind,
                                component: Some(u32::from(c)),
                            },
                        )
                    })
                    .collect();
                let vec_ty = IrType::Vector {
                    scalar,
                    len: elems.len() as u8,
                };
                return emit_vec_construct(self.ir_fn, vec_ty, elems);
            }

            if self.push_constant_param.as_ref() == Some(path)
                && let Some(&f_idx) = self.push_constant_fields.get(field)
            {
                if *expr_ir_ty != IrType::Void {
                    return self.ir_fn.emit_typed(
                        expr_ir_ty.clone(),
                        InstData::LoadPushConstant { field_index: f_idx },
                    );
                }
                return self
                    .ir_fn
                    .emit(None, InstData::LoadPushConstant { field_index: f_idx });
            }

            if let Some(elem_binding) = self.desc_elem_bindings.get(path).cloned() {
                return self.emit_descriptor_elem_field_load(&elem_binding, field.as_str(), span);
            }
        }

        // Direct chained SSBO element field access (`mc::instances[i].material_id`,
        // `mc::lights.items[i].position`, `mc::reservoirs.items[i].weight_sum`).
        if let Some(elem_binding) = self.try_lower_descriptor_elem_index(receiver, out) {
            return self.emit_descriptor_elem_field_load(&elem_binding, field.as_str(), span);
        }

        // Direct user Buffer<T> / RWBuffer<T> element field load (`buf[i].field`).
        if let HirExprKind::Index {
            receiver: buf_recv,
            index: idx_expr,
        } = &receiver.kind
            && let HirExprKind::Path { path, .. } = &buf_recv.kind
            && let Some(desc_idx) = self.resolve_descriptor_path(path)
        {
            let elem_ty = self.ir_ty_of(receiver.ty);
            let f_idx = field_index.or_else(|| match &elem_ty {
                IrType::Struct(layout) => layout
                    .fields
                    .iter()
                    .position(|f| f.name == *field)
                    .map(|i| i as u32),
                _ => None,
            });
            if let Some(idx) = f_idx {
                let idx_val = self.lower_expr(idx_expr, out);
                let zero = self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0));
                let f_const = self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(idx));
                let ptr = self.ir_fn.emit_typed(
                    IrType::Pointer {
                        pointee: Box::new(expr_ir_ty.clone()),
                        align: 4,
                        storage: IrStorageClass::StorageBuffer,
                    },
                    InstData::DescriptorAccessChain {
                        descriptor_index: desc_idx,
                        indices: vec![zero, idx_val, f_const],
                    },
                );
                return self
                    .ir_fn
                    .emit_typed(expr_ir_ty.clone(), InstData::ResourceLoad { ptr });
            }
        }

        // BDA pointer struct field load (`bda_ptr.field`).
        if let Some((field_ptr, field_ty, field_align, is_mut)) =
            self.try_lower_bda_field_ptr(receiver, field, field_index, out)
        {
            return self.ir_fn.emit_typed(
                field_ty,
                InstData::BdaLoad {
                    ptr: field_ptr,
                    access: MemoryAccess {
                        align: field_align,
                        non_writable: !is_mut,
                        restrict: false,
                    },
                },
            );
        }

        // Direct built-in UBO/SSBO descriptor field access (`mc::world.*`, `mc::sharc.*`,
        // `mc::lights.count`, `mc::reservoirs.count`).
        if let HirExprKind::Path { path, .. } = &receiver.kind
            && let Some(desc_idx) = self.resolve_descriptor_path(path)
        {
            let desc = &self.ir_fn.descriptors[desc_idx as usize];
            let (set, binding, is_uniform) = (
                desc.set,
                desc.binding,
                desc.kind == DescriptorKind::UniformBuffer,
            );
            if let Some(builtin_layout) = builtin_descriptor_layout(set, binding) {
                let storage = if is_uniform {
                    IrStorageClass::Uniform
                } else {
                    IrStorageClass::StorageBuffer
                };
                if builtin_layout.name == "WorldUBO" && is_uniform {
                    return match canonicalize_builtin_field_name(builtin_layout, field.as_str()) {
                        "game_time_sec" => self.world_game_time_sec(span),
                        "rain" => self.world_rain(span),
                        "thunder" => self.world_thunder(span),
                        "wind_phase" => self.world_wind_phase(span),
                        "eye_in_fluid" => self.world_eye_in_fluid(span),
                        "sun_dir" => self.world_sun_dir(span),
                        _ => self.emit_builtin_struct_field_load(
                            desc_idx,
                            builtin_layout,
                            field.as_str(),
                            storage,
                            span,
                        ),
                    };
                }
                return self.emit_builtin_struct_field_load(
                    desc_idx,
                    builtin_layout,
                    field.as_str(),
                    storage,
                    span,
                );
            }
        }

        // General composite/vector field or swizzle access.
        let recv_val = self.lower_expr(receiver, out);
        let recv_ty = self.inst_ty(recv_val).clone();

        if matches!(recv_ty, IrType::Vector { .. })
            && let Some(comps) = parse_swizzle_str(field.as_str())
        {
            return self.lower_swizzle_read(recv_val, &comps, expr_ir_ty, span);
        }

        if let IrType::Struct(ref layout) = recv_ty {
            let f_idx = field_index.or_else(|| {
                layout
                    .fields
                    .iter()
                    .position(|f| f.name == *field)
                    .map(|i| i as u32)
            });
            if let Some(idx) = f_idx {
                let f_ty = layout
                    .fields
                    .get(idx as usize)
                    .map_or_else(|| expr_ir_ty.clone(), |f| f.ty.clone());
                return self.ir_fn.emit_typed(
                    f_ty,
                    InstData::CompositeExtract {
                        composite: recv_val,
                        indices: vec![idx],
                    },
                );
            }
        }

        if let Some(idx) = field_index {
            return self.ir_fn.emit_typed(
                expr_ir_ty.clone(),
                InstData::CompositeExtract {
                    composite: recv_val,
                    indices: vec![idx],
                },
            );
        }

        self.emit_diag(
            "E0526",
            format!("cannot resolve field `.{field}` on expression of type `{recv_ty:?}`"),
            span,
        )
    }

    fn lower_index_expr(
        &mut self,
        receiver: &HirExpr,
        index: &HirExpr,
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        // Built-in SSBO array element (`mc::instances[i]`, `mc::lights.items[i]`, `mc::reservoirs.items[i]`)
        // used directly as a value (e.g. passed to a function).
        if let Some(elem_binding) = self.try_lower_descriptor_elem_index(
            &HirExpr {
                kind: HirExprKind::Index {
                    receiver: Box::new(receiver.clone()),
                    index: Box::new(index.clone()),
                },
                ty: receiver.ty,
                symbol: None,
                span,
            },
            out,
        ) {
            let items_idx_const = self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::ConstU32(elem_binding.items_field_index),
            );
            let elem_ty = IrType::from_builtin_struct(elem_binding.elem_desc);
            return self.ir_fn.emit_typed(
                IrType::Pointer {
                    pointee: Box::new(elem_ty),
                    align: elem_binding.elem_desc.align,
                    storage: IrStorageClass::StorageBuffer,
                },
                InstData::DescriptorAccessChain {
                    descriptor_index: elem_binding.descriptor_index,
                    indices: vec![items_idx_const, elem_binding.index_val],
                },
            );
        }

        let idx_val = self.lower_expr(index, out);

        if let HirExprKind::Path { path, .. } = &receiver.kind {
            if !path.contains("::")
                && let Some(&var_index) = self.workgroup_indices.get(path)
            {
                let elem_ty = match self.ir_fn.workgroup_vars[var_index as usize].effective_type() {
                    IrType::Array { element, .. } => *element,
                    other => other,
                };
                return self.ir_fn.emit_typed(
                    elem_ty,
                    InstData::WorkgroupLoad {
                        var_index,
                        index: Some(idx_val),
                    },
                );
            }

            if let Some(desc_idx) = self.resolve_descriptor_path(path) {
                let kind = self.ir_fn.descriptors[desc_idx as usize].kind;
                if matches!(
                    kind,
                    DescriptorKind::CombinedImageSampler | DescriptorKind::SampledImage
                ) {
                    let chain = self.ir_fn.emit(
                        None,
                        InstData::ResourceAccessChain {
                            descriptor_index: desc_idx,
                            index: idx_val,
                        },
                    );
                    let load = self.ir_fn.emit(None, InstData::ResourceLoad { ptr: chain });
                    out.push(StructuredStmt::Eval(chain));
                    out.push(StructuredStmt::Eval(load));
                    return load;
                }
                // StorageBuffer indexing (`buf[i]`).
                let zero = self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::ConstU32(0));
                let ptr = self.ir_fn.emit_typed(
                    IrType::Pointer {
                        pointee: Box::new(expr_ir_ty.clone()),
                        align: 4,
                        storage: IrStorageClass::StorageBuffer,
                    },
                    InstData::DescriptorAccessChain {
                        descriptor_index: desc_idx,
                        indices: vec![zero, idx_val],
                    },
                );
                return self
                    .ir_fn
                    .emit_typed(expr_ir_ty.clone(), InstData::ResourceLoad { ptr });
            }
        }

        // Local array / composite dynamic indexing via `LocalAddr` + `AccessChain` + `LocalLoad`.
        if let Some((base_ptr, base_ty)) = self.try_lower_local_lvalue_ptr(receiver, out) {
            let elem_ty = match base_ty {
                IrType::Array { element, .. } => *element,
                IrType::Vector { scalar, .. } => IrType::Scalar(scalar),
                IrType::Matrix { scalar, rows, .. } => IrType::Vector { scalar, len: rows },
                _ => expr_ir_ty.clone(),
            };
            let elem_ptr = self.ir_fn.emit_typed(
                IrType::Pointer {
                    pointee: Box::new(elem_ty.clone()),
                    align: 4,
                    storage: IrStorageClass::Function,
                },
                InstData::AccessChain {
                    base_ptr,
                    indices: vec![idx_val],
                },
            );
            return self
                .ir_fn
                .emit_typed(elem_ty, InstData::LocalLoad { ptr: elem_ptr });
        }

        // Constant index into an SSA vector or array value.
        let recv_val = self.lower_expr(receiver, out);
        if let Some(const_idx) = parse_const_i64_hir(index).and_then(|v| u32::try_from(v).ok()) {
            let elem_ty = match self.inst_ty(recv_val) {
                IrType::Array { element, .. } => element.as_ref().clone(),
                IrType::Vector { scalar, .. } => IrType::Scalar(*scalar),
                IrType::Matrix { scalar, rows, .. } => IrType::Vector {
                    scalar: *scalar,
                    len: *rows,
                },
                _ => expr_ir_ty.clone(),
            };
            return self.ir_fn.emit_typed(
                elem_ty,
                InstData::CompositeExtract {
                    composite: recv_val,
                    indices: vec![const_idx],
                },
            );
        }

        self.emit_diag(
            "E0527",
            "dynamic indexing into a non-addressable rvalue is not supported in GPU IR",
            span,
        )
    }

    fn lower_call_expr(
        &mut self,
        callee: &HirExpr,
        resolved_fn: Option<HirFnId>,
        args: &[HirCallArg],
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        match self.lower_call_expr_inner(callee, resolved_fn, args, expr_ir_ty, span, out) {
            Ok(id) | Err(id) => id,
        }
    }

    fn lower_call_arg_for_mode(
        &mut self,
        arg_expr: &HirExpr,
        mode: Option<ParamMode>,
        tmp_prefix: &str,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        if matches!(mode, Some(ParamMode::InOut | ParamMode::Out)) {
            if let Some((ptr, _)) = self.try_lower_local_lvalue_ptr(arg_expr, out) {
                return ptr;
            }
            let val = self.lower_expr(arg_expr, out);
            let val_ty = self.inst_ty(val).clone();
            let tmp = alloc_local_var(
                self.ir_fn,
                SmolStr::new(format!("{tmp_prefix}_{}", self.ir_fn.local_vars.len())),
                val_ty.clone(),
            );
            let addr = emit_local_addr(self.ir_fn, tmp, &val_ty);
            let store = self.ir_fn.emit_typed(
                IrType::Void,
                InstData::LocalStore {
                    ptr: addr,
                    value: val,
                },
            );
            out.push(StructuredStmt::Eval(store));
            return addr;
        }
        self.lower_expr(arg_expr, out)
    }

    fn lower_call_expr_inner(
        &mut self,
        callee: &HirExpr,
        resolved_fn: Option<HirFnId>,
        args: &[HirCallArg],
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> Result<InstId, InstId> {
        // Direct call to a resolved `HirFunction` in this module -> `InstData::FunctionCall`.
        if let Some(fn_id) = resolved_fn
            && let Some(callee_name) = self.fn_names_by_id.get(&fn_id).cloned()
        {
            let param_modes: Vec<ParamMode> = self.hir.functions[fn_id]
                .params
                .iter()
                .map(|p| p.mode)
                .collect();
            let mut lowered_args = Vec::with_capacity(args.len());
            for (idx, a) in args.iter().enumerate() {
                lowered_args.push(self.lower_call_arg_for_mode(
                    &a.value,
                    param_modes.get(idx).copied(),
                    "__inout_tmp",
                    out,
                ));
            }
            let id = self.ir_fn.emit_typed(
                expr_ir_ty.clone(),
                InstData::FunctionCall {
                    callee: callee_name,
                    args: lowered_args,
                },
            );
            out.push(StructuredStmt::Eval(id));
            return Ok(id);
        }

        let HirExprKind::Path { path, .. } = &callee.kind else {
            return Ok(self.emit_diag(
                "E0520",
                "indirect or non-path function calls are not supported in VSL 2.0",
                span,
            ));
        };
        let joined = path.as_str();

        // Scalar casts and precision conversions.
        match joined {
            "u32" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::CastToU32(inner)));
            }
            "i32" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::CastToI32(inner)));
            }
            "f32" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                if self.inst_ty(inner).is_f16() {
                    return Ok(self.ir_fn.emit_typed(
                        IrType::Scalar(ScalarIrTy::F32),
                        InstData::FConvert {
                            value: inner,
                            target: ScalarIrTy::F32,
                        },
                    ));
                }
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::CastToF32(inner)));
            }
            "bool" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                return Ok(coerce_to_bool(self.ir_fn, inner));
            }
            "f16" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                let from_ty = self.inst_ty(inner).clone();
                if from_ty.is_f16() {
                    return Ok(inner);
                }
                let as_f32 = if from_ty.is_float() {
                    inner
                } else {
                    self.ir_fn
                        .emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::CastToF32(inner))
                };
                return Ok(self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::F16),
                    InstData::FConvert {
                        value: as_f32,
                        target: ScalarIrTy::F16,
                    },
                ));
            }
            "i64" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::CastToI64(inner)));
            }
            "u64" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::U64), InstData::CastToU64(inner)));
            }
            "vec2" | "ivec2" | "uvec2" | "bvec2" | "f16vec2" | "vec3" | "ivec3" | "uvec3"
            | "bvec3" | "f16vec3" | "vec4" | "ivec4" | "uvec4" | "bvec4" | "f16vec4" => {
                return Ok(self.lower_vector_constructor(joined, args, span, out));
            }
            "mat2" | "mat2x2" | "mat3" | "mat3x3" | "mat4" | "mat4x4" | "mat3x4" | "mat4x3" => {
                return Ok(self.lower_matrix_constructor(joined, args, expr_ir_ty, span, out));
            }
            "mc::spec" | "spec" => {
                let name =
                    extract_hir_name_arg(args.first()).unwrap_or_else(|| SmolStr::new_static(""));
                if name.is_empty() {
                    return Ok(self.emit_diag(
                        "E0525",
                        "`mc::spec` requires a setting name argument",
                        span,
                    ));
                }
                return Ok(self.emit_spec_constant(name.as_str(), expr_ir_ty, span));
            }
            "mc::param" | "param" | "pack::param" => {
                let name =
                    extract_hir_name_arg(args.first()).unwrap_or_else(|| SmolStr::new_static(""));
                if name.is_empty() {
                    return Ok(self.emit_diag(
                        "E0525",
                        "`mc::param` requires a parameter name argument",
                        span,
                    ));
                }
                return Ok(self.emit_pack_param_load(name.as_str(), expr_ir_ty, span));
            }
            // Built-in `mc::*` library (Item 22) — real physical IR, zero stubs.
            "mc::reconstruct_world_pos" => {
                return Ok(self.lower_mc_reconstruct_world_pos(args, span, out));
            }
            "mc::view_dir" => {
                return Ok(self.lower_mc_view_dir(args, span, out));
            }
            "mc::sample_sky" => {
                return Ok(self.lower_mc_sample_sky(args, span, out));
            }
            "mc::has_biome_telemetry" => {
                return Ok(self.lower_mc_has_biome_telemetry(span));
            }
            "mc::dimension_id" => {
                return Ok(self.lower_mc_dimension_id(span));
            }
            "mc::biome_cloud_scale" => {
                return Ok(self.lower_mc_biome_cloud_scale(span));
            }
            "mc::debug_write" | "debug_write" => {
                return Ok(self.lower_mc_debug_write(args, span, out));
            }
            "discard" | "demote" => {
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::DemoteToHelperInvocation);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "trace_ray" => {
                let tlas_desc = args
                    .first()
                    .and_then(|a| {
                        if let HirExprKind::Path { path, .. } = &a.value.kind {
                            self.descriptor_indices.get(path.as_str()).copied()
                        } else {
                            None
                        }
                    })
                    .or(self.tlas_descriptor_index)
                    .unwrap_or(0);
                let ray_flags = self.require_arg(args, 1, joined, span, out)?;
                let cull_mask = self.require_arg(args, 2, joined, span, out)?;
                let sbt_offset = self.require_arg(args, 3, joined, span, out)?;
                let sbt_stride = self.require_arg(args, 4, joined, span, out)?;
                let miss_index = self.require_arg(args, 5, joined, span, out)?;
                let origin = self.require_arg(args, 6, joined, span, out)?;
                let t_min = self.require_arg(args, 7, joined, span, out)?;
                let direction = self.require_arg(args, 8, joined, span, out)?;
                let t_max = self.require_arg(args, 9, joined, span, out)?;
                let payload_var_index = args
                    .get(10)
                    .and_then(|a| {
                        if let HirExprKind::Path { path, .. } = &a.value.kind {
                            self.stage_var_indices.get(path.as_str()).copied()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::TraceRayKhr {
                        tlas_descriptor_index: tlas_desc,
                        ray_flags,
                        cull_mask,
                        sbt_offset,
                        sbt_stride,
                        miss_index,
                        origin,
                        t_min,
                        direction,
                        t_max,
                        payload_var_index,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "ignore_intersection" => {
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::IgnoreIntersectionKhr);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "terminate_ray" => {
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::TerminateRayKhr);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "report_intersection" => {
                let hit_t = self.require_arg(args, 0, joined, span, out)?;
                let hit_kind = self.require_arg(args, 1, joined, span, out)?;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::ReportIntersectionKhr { hit_t, hit_kind },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "execute_callable" => {
                let sbt_index = self.require_arg(args, 0, joined, span, out)?;
                let callable_var_index = args
                    .get(1)
                    .and_then(|a| {
                        if let HirExprKind::Path { path, .. } = &a.value.kind {
                            self.stage_var_indices.get(path.as_str()).copied()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::ExecuteCallableKhr {
                        sbt_index,
                        callable_var_index,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "emit_mesh_tasks" => {
                let group_x = self.require_arg(args, 0, joined, span, out)?;
                let group_y = self.require_arg(args, 1, joined, span, out)?;
                let group_z = self.require_arg(args, 2, joined, span, out)?;
                let payload_var_index = args.get(3).and_then(|a| {
                    if let HirExprKind::Path { path, .. } = &a.value.kind {
                        self.stage_var_indices.get(path.as_str()).copied()
                    } else {
                        None
                    }
                });
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::EmitMeshTasksExt {
                        group_x,
                        group_y,
                        group_z,
                        payload_var_index,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "set_mesh_outputs" => {
                let vertex_count = self.require_arg(args, 0, joined, span, out)?;
                let primitive_count = self.require_arg(args, 1, joined, span, out)?;
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::SetMeshOutputsExt {
                        vertex_count,
                        primitive_count,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "emit_vertex" => {
                let id = self.ir_fn.emit_typed(IrType::Void, InstData::EmitVertex);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "end_primitive" => {
                let id = self.ir_fn.emit_typed(IrType::Void, InstData::EndPrimitive);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "reorder_thread_with_hint" => {
                let hint = self.require_arg(args, 0, joined, span, out)?;
                let bits = self.require_arg(args, 1, joined, span, out)?;
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::ReorderThreadWithHint { hint, bits });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "reorder_thread_with_hit_object" => {
                let hint = if args.len() >= 3 {
                    Some(self.require_arg(args, 1, joined, span, out)?)
                } else {
                    None
                };
                let bits = if args.len() >= 3 {
                    Some(self.require_arg(args, 2, joined, span, out)?)
                } else {
                    None
                };
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::ReorderThreadWithHitObject {
                        hit_object_id: 0,
                        hint,
                        bits,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "coop_mat_load" => {
                let ptr = self.require_arg(args, 0, joined, span, out)?;
                let stride = if args.len() >= 2 {
                    Some(self.require_arg(args, 1, joined, span, out)?)
                } else {
                    None
                };
                let id = self.ir_fn.emit_typed(
                    expr_ir_ty.clone(),
                    InstData::CoopMatrixLoad {
                        ptr,
                        layout: CoopMatrixLayout::RowMajor,
                        stride,
                    },
                );
                return Ok(id);
            }
            "coop_mat_store" => {
                let ptr = self.require_arg(args, 0, joined, span, out)?;
                let value = self.require_arg(args, 1, joined, span, out)?;
                let stride = if args.len() >= 3 {
                    Some(self.require_arg(args, 2, joined, span, out)?)
                } else {
                    None
                };
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::CoopMatrixStore {
                        ptr,
                        value,
                        layout: CoopMatrixLayout::RowMajor,
                        stride,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "coop_mat_mul_add" => {
                let a = self.require_arg(args, 0, joined, span, out)?;
                let b = self.require_arg(args, 1, joined, span, out)?;
                let c = self.require_arg(args, 2, joined, span, out)?;
                let ty = self.inst_ty(c).clone();
                let id = self.ir_fn.emit_typed(
                    ty,
                    InstData::CoopMatrixMulAdd {
                        a,
                        b,
                        c,
                        operands_mask: 0,
                    },
                );
                return Ok(id);
            }
            "coop_mat_length" => {
                let mat_ty = args.first().map_or(IrType::Void, |a| {
                    IrType::from_hir_type(&self.hir.types, a.value.ty)
                });
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U32),
                    InstData::CoopMatrixLength { mat_ty },
                );
                return Ok(id);
            }
            "begin_invocation_interlock" => {
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::BeginInvocationInterlockExt);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "end_invocation_interlock" => {
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::EndInvocationInterlockExt);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "debug_printf" | "debugPrintfEXT" | "std::debug::debug_printf" => {
                let format = match args.first().map(|a| &a.value.kind) {
                    Some(HirExprKind::StringLit(raw)) => SmolStr::new(raw.trim_matches('"')),
                    _ => SmolStr::new(""),
                };
                let mut printf_args = Vec::with_capacity(args.len().saturating_sub(1));
                for arg in args.iter().skip(1) {
                    printf_args.push(self.lower_expr(&arg.value, out));
                }
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::DebugPrintf {
                        format,
                        args: printf_args,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "nonuniform" => {
                let inner = self.require_arg(args, 0, joined, span, out)?;
                let ty = self.inst_ty(inner).clone();
                return Ok(self.ir_fn.emit_typed(ty, InstData::NonUniform(inner)));
            }
            "workgroup_barrier" | "std::sync::workgroup_barrier" => {
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::WorkgroupBarrier);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "memory_barrier" | "std::sync::memory_barrier" => {
                let (scope, order) = self
                    .ast_hints
                    .atomic_turbofish
                    .get(&span)
                    .copied()
                    .unwrap_or((AtomicScope::Device, AtomicOrder::AcqRel));
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::MemoryBarrier { scope, order });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            // Atomics (32-bit, 64-bit, float, workgroup, SSBO, BDA, image texel).
            "atomic_add" | "std::sync::atomic_add" => {
                return Ok(self.lower_atomic_binary(args, span, AtomicKind::Add, out));
            }
            "atomic_min" | "std::sync::atomic_min" => {
                return Ok(self.lower_atomic_binary(args, span, AtomicKind::Min, out));
            }
            "atomic_max" | "std::sync::atomic_max" => {
                return Ok(self.lower_atomic_binary(args, span, AtomicKind::Max, out));
            }
            "atomic_cas"
            | "atomic_compare_exchange"
            | "std::sync::atomic_cas"
            | "std::sync::atomic_compare_exchange" => {
                let (scope, order) = self
                    .ast_hints
                    .atomic_turbofish
                    .get(&span)
                    .copied()
                    .unwrap_or((AtomicScope::Device, AtomicOrder::Relaxed));
                let ptr = self.require_arg(args, 0, joined, span, out)?;
                let expected = self.require_arg(args, 1, joined, span, out)?;
                let desired = self.require_arg(args, 2, joined, span, out)?;
                let res_ty = self.inst_ty(expected).clone();
                let id = self.ir_fn.emit_typed(
                    res_ty,
                    InstData::AtomicCas {
                        ptr,
                        expected,
                        desired,
                        scope,
                        order,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            // Subgroup & Quad intrinsics.
            "subgroup_ballot" | "std::subgroup::subgroup_ballot" => {
                let pred = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Vector {
                        scalar: ScalarIrTy::U32,
                        len: 4,
                    },
                    InstData::SubgroupBallot(pred),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_ballot_bit_count" | "std::subgroup::subgroup_ballot_bit_count" => {
                let ballot = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U32),
                    InstData::SubgroupBallotBitCount(ballot),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_ballot_exclusive_bit_count"
            | "std::subgroup::subgroup_ballot_exclusive_bit_count" => {
                let ballot = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U32),
                    InstData::SubgroupBallotExclusiveBitCount(ballot),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_ballot_inclusive_bit_count"
            | "std::subgroup::subgroup_ballot_inclusive_bit_count" => {
                let ballot = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U32),
                    InstData::SubgroupBallotInclusiveBitCount(ballot),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_broadcast_first" | "std::subgroup::subgroup_broadcast_first" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupBroadcastFirst(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_add" | "std::subgroup::subgroup_add" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupAdd(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_min" | "std::subgroup::subgroup_min" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupMin(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_max" | "std::subgroup::subgroup_max" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupMax(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_exclusive_add" | "std::subgroup::subgroup_exclusive_add" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupExclusiveAdd(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_inclusive_add" | "std::subgroup::subgroup_inclusive_add" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupInclusiveAdd(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_shuffle" | "std::subgroup::subgroup_shuffle" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let lane = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(value).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupShuffle { value, lane });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_shuffle_xor" | "std::subgroup::subgroup_shuffle_xor" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let mask = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(value).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupShuffleXor { value, mask });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_any" | "std::subgroup::subgroup_any" => {
                let pred = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::SubgroupAny(pred),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_all" | "std::subgroup::subgroup_all" => {
                let pred = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::SubgroupAll(pred),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_elect" | "std::subgroup::subgroup_elect" => {
                self.ir_fn.uses_subgroups = true;
                let id = self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::Bool), InstData::SubgroupElect);
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "quad_swap_horizontal" | "std::subgroup::quad_swap_horizontal" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_quad_control = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::QuadSwapHorizontal(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "quad_swap_vertical" | "std::subgroup::quad_swap_vertical" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_quad_control = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::QuadSwapVertical(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "quad_swap_diagonal" | "std::subgroup::quad_swap_diagonal" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_quad_control = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::QuadSwapDiagonal(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "quad_broadcast" | "std::subgroup::quad_broadcast" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let lane = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_quad_control = true;
                let ty = self.inst_ty(value).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::QuadBroadcast { value, lane });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_broadcast" | "std::subgroup::subgroup_broadcast" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let lane = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(value).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupBroadcast { value, lane });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_ballot_bit_extract" | "std::subgroup::subgroup_ballot_bit_extract" => {
                let ballot = self.require_arg(args, 0, joined, span, out)?;
                let index = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::SubgroupBallotBitExtract { ballot, index },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_ballot_find_lsb" | "std::subgroup::subgroup_ballot_find_lsb" => {
                let ballot = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U32),
                    InstData::SubgroupBallotFindLsb(ballot),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_ballot_find_msb" | "std::subgroup::subgroup_ballot_find_msb" => {
                let ballot = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::U32),
                    InstData::SubgroupBallotFindMsb(ballot),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_shuffle_up" | "std::subgroup::subgroup_shuffle_up" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let delta = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(value).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupShuffleUp { value, delta });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_shuffle_down" | "std::subgroup::subgroup_shuffle_down" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let delta = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(value).clone();
                let id = self
                    .ir_fn
                    .emit_typed(ty, InstData::SubgroupShuffleDown { value, delta });
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_mul" | "std::subgroup::subgroup_mul" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupMul(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_exclusive_mul" | "std::subgroup::subgroup_exclusive_mul" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupExclusiveMul(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_inclusive_mul" | "std::subgroup::subgroup_inclusive_mul" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupInclusiveMul(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_exclusive_min" | "std::subgroup::subgroup_exclusive_min" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupExclusiveMin(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_inclusive_min" | "std::subgroup::subgroup_inclusive_min" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupInclusiveMin(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_exclusive_max" | "std::subgroup::subgroup_exclusive_max" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupExclusiveMax(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_inclusive_max" | "std::subgroup::subgroup_inclusive_max" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupInclusiveMax(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_bit_and" | "std::subgroup::subgroup_bit_and" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupBitAnd(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_bit_or" | "std::subgroup::subgroup_bit_or" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupBitOr(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_bit_xor" | "std::subgroup::subgroup_bit_xor" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let ty = self.inst_ty(val).clone();
                let id = self.ir_fn.emit_typed(ty, InstData::SubgroupBitXor(val));
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_logical_and" | "std::subgroup::subgroup_logical_and" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::SubgroupLogicalAnd(val),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_logical_or" | "std::subgroup::subgroup_logical_or" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::SubgroupLogicalOr(val),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_logical_xor" | "std::subgroup::subgroup_logical_xor" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::SubgroupLogicalXor(val),
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            "subgroup_clustered_add"
            | "std::subgroup::subgroup_clustered_add"
            | "subgroup_clustered_mul"
            | "std::subgroup::subgroup_clustered_mul"
            | "subgroup_clustered_min"
            | "std::subgroup::subgroup_clustered_min"
            | "subgroup_clustered_max"
            | "std::subgroup::subgroup_clustered_max"
            | "subgroup_clustered_and"
            | "std::subgroup::subgroup_clustered_and"
            | "subgroup_clustered_or"
            | "std::subgroup::subgroup_clustered_or"
            | "subgroup_clustered_xor"
            | "std::subgroup::subgroup_clustered_xor" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let cluster_size = self.require_arg(args, 1, joined, span, out)?;
                self.ir_fn.uses_subgroups = true;
                let op = if joined.ends_with("_add") {
                    SubgroupClusteredOp::Add
                } else if joined.ends_with("_mul") {
                    SubgroupClusteredOp::Mul
                } else if joined.ends_with("_min") {
                    SubgroupClusteredOp::Min
                } else if joined.ends_with("_max") {
                    SubgroupClusteredOp::Max
                } else if joined.ends_with("_and") {
                    SubgroupClusteredOp::BitAnd
                } else if joined.ends_with("_or") {
                    SubgroupClusteredOp::BitOr
                } else {
                    SubgroupClusteredOp::BitXor
                };
                let ty = self.inst_ty(value).clone();
                let id = self.ir_fn.emit_typed(
                    ty,
                    InstData::SubgroupClustered {
                        op,
                        value,
                        cluster_size,
                    },
                );
                out.push(StructuredStmt::Eval(id));
                return Ok(id);
            }
            // Bit, predicate, and fragment derivative intrinsics.
            "bitfield_extract" => {
                let base = self.require_arg(args, 0, joined, span, out)?;
                let offset = self.require_arg(args, 1, joined, span, out)?;
                let count = self.require_arg(args, 2, joined, span, out)?;
                let ty = self.inst_ty(base).clone();
                let signed = ty.is_signed_int();
                return Ok(self.ir_fn.emit_typed(
                    ty,
                    InstData::BitFieldExtract {
                        base,
                        offset,
                        count,
                        signed,
                    },
                ));
            }
            "bitfield_insert" => {
                let base = self.require_arg(args, 0, joined, span, out)?;
                let insert = self.require_arg(args, 1, joined, span, out)?;
                let offset = self.require_arg(args, 2, joined, span, out)?;
                let count = self.require_arg(args, 3, joined, span, out)?;
                let ty = self.inst_ty(base).clone();
                return Ok(self.ir_fn.emit_typed(
                    ty,
                    InstData::BitFieldInsert {
                        base,
                        insert,
                        offset,
                        count,
                    },
                ));
            }
            "bit_count" | "count_bits" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::U32), InstData::BitCount(val)));
            }
            "float_bits_to_uint" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let target = IrType::Scalar(ScalarIrTy::U32);
                return Ok(self
                    .ir_fn
                    .emit_typed(target.clone(), InstData::Bitcast { value, target }));
            }
            "uint_bits_to_float" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let target = IrType::Scalar(ScalarIrTy::F32);
                return Ok(self
                    .ir_fn
                    .emit_typed(target.clone(), InstData::Bitcast { value, target }));
            }
            "float_bits_to_int" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let target = IrType::Scalar(ScalarIrTy::I32);
                return Ok(self
                    .ir_fn
                    .emit_typed(target.clone(), InstData::Bitcast { value, target }));
            }
            "int_bits_to_float" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let target = IrType::Scalar(ScalarIrTy::F32);
                return Ok(self
                    .ir_fn
                    .emit_typed(target.clone(), InstData::Bitcast { value, target }));
            }
            "bitcast" => {
                let value = self.require_arg(args, 0, joined, span, out)?;
                let target = expr_ir_ty.clone();
                return Ok(self
                    .ir_fn
                    .emit_typed(target.clone(), InstData::Bitcast { value, target }));
            }
            "is_nan" | "isnan" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::Bool), InstData::IsNan(val)));
            }
            "is_inf" | "isinf" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                return Ok(self
                    .ir_fn
                    .emit_typed(IrType::Scalar(ScalarIrTy::Bool), InstData::IsInf(val)));
            }
            "dpdx" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                let ty = self.inst_ty(val).clone();
                return Ok(self.ir_fn.emit_typed(ty, InstData::DPdx(val)));
            }
            "dpdy" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                let ty = self.inst_ty(val).clone();
                return Ok(self.ir_fn.emit_typed(ty, InstData::DPdy(val)));
            }
            "fwidth" => {
                let val = self.require_arg(args, 0, joined, span, out)?;
                let ty = self.inst_ty(val).clone();
                return Ok(self.ir_fn.emit_typed(ty, InstData::Fwidth(val)));
            }
            "select" => {
                let cond = self.require_arg(args, 0, joined, span, out)?;
                let on_true = self.require_arg(args, 1, joined, span, out)?;
                let on_false = self.require_arg(args, 2, joined, span, out)?;
                let ty = self.inst_ty(on_true).clone();
                return Ok(self.ir_fn.emit_typed(
                    ty,
                    InstData::Select {
                        cond,
                        on_true,
                        on_false,
                    },
                ));
            }
            _ => {}
        }

        // GLSL.std.450 math & packing intrinsics.
        if let Some(id) = self.try_lower_glsl_std450(joined, args, expr_ir_ty, out) {
            return Ok(id);
        }

        // Check module-defined function by short name.
        let short_name = joined.rsplit("::").next().unwrap_or(joined);
        if let Some(target_fn) = self.hir.find_function(short_name) {
            let param_modes: Vec<ParamMode> = target_fn.params.iter().map(|p| p.mode).collect();
            let mut lowered_args = Vec::with_capacity(args.len());
            for (idx, a) in args.iter().enumerate() {
                lowered_args.push(self.lower_call_arg_for_mode(
                    &a.value,
                    param_modes.get(idx).copied(),
                    "__inout_tmp",
                    out,
                ));
            }
            let id = self.ir_fn.emit_typed(
                expr_ir_ty.clone(),
                InstData::FunctionCall {
                    callee: SmolStr::new(short_name),
                    args: lowered_args,
                },
            );
            out.push(StructuredStmt::Eval(id));
            return Ok(id);
        }

        // Reject unresolved qualified or unqualified calls with an explicit diagnostic rather than
        // emitting a dangling `FunctionCall`.
        Ok(self.emit_diag(
            "E0520",
            format!("unresolved function call `{joined}` during IR lowering"),
            span,
        ))
    }

    fn require_arg(
        &mut self,
        args: &[HirCallArg],
        index: usize,
        callee_name: &str,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> Result<InstId, InstId> {
        if let Some(arg) = args.get(index) {
            Ok(self.lower_expr(&arg.value, out))
        } else {
            Err(self.emit_diag(
                "E0528",
                format!("missing argument #{} in call to `{callee_name}`", index + 1),
                span,
            ))
        }
    }

    fn lower_vector_constructor(
        &mut self,
        ctor: &str,
        args: &[HirCallArg],
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let (target_scalar, target_len) = match ctor {
            "vec2" => (ScalarIrTy::F32, 2u8),
            "ivec2" => (ScalarIrTy::I32, 2u8),
            "uvec2" => (ScalarIrTy::U32, 2u8),
            "bvec2" => (ScalarIrTy::Bool, 2u8),
            "f16vec2" => (ScalarIrTy::F16, 2u8),
            "vec3" => (ScalarIrTy::F32, 3u8),
            "ivec3" => (ScalarIrTy::I32, 3u8),
            "uvec3" => (ScalarIrTy::U32, 3u8),
            "bvec3" => (ScalarIrTy::Bool, 3u8),
            "f16vec3" => (ScalarIrTy::F16, 3u8),
            "vec4" => (ScalarIrTy::F32, 4u8),
            "ivec4" => (ScalarIrTy::I32, 4u8),
            "uvec4" => (ScalarIrTy::U32, 4u8),
            "bvec4" => (ScalarIrTy::Bool, 4u8),
            "f16vec4" => (ScalarIrTy::F16, 4u8),
            _ => unreachable!(),
        };

        let mut scalars = Vec::with_capacity(usize::from(target_len));
        for arg in args {
            let val = self.lower_expr(&arg.value, out);
            match self.inst_ty(val).clone() {
                IrType::Vector { scalar, len } => {
                    for c in 0..u32::from(len) {
                        let comp = self.ir_fn.emit_typed(
                            IrType::Scalar(scalar),
                            InstData::CompositeExtract {
                                composite: val,
                                indices: vec![c],
                            },
                        );
                        scalars.push(coerce_scalar(self.ir_fn, comp, scalar, target_scalar));
                    }
                }
                IrType::Scalar(scalar) => {
                    scalars.push(coerce_scalar(self.ir_fn, val, scalar, target_scalar));
                }
                _ => {
                    scalars.push(val);
                }
            }
        }

        // Splat single-scalar constructor (e.g. `vec3(1.0)` -> `vec3(1.0, 1.0, 1.0)`).
        if scalars.len() == 1 && target_len > 1 {
            let s = scalars[0];
            while scalars.len() < usize::from(target_len) {
                scalars.push(s);
            }
        }

        if scalars.len() != usize::from(target_len) {
            return self.emit_diag(
                "E0528",
                format!(
                    "vector constructor `{ctor}` expects {target_len} scalar component(s), got {}",
                    scalars.len()
                ),
                span,
            );
        }

        let vec_ty = IrType::Vector {
            scalar: target_scalar,
            len: target_len,
        };
        emit_vec_construct(self.ir_fn, vec_ty, scalars)
    }

    fn lower_matrix_constructor(
        &mut self,
        ctor: &str,
        args: &[HirCallArg],
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let (cols, rows) = match ctor {
            "mat2" | "mat2x2" => (2u8, 2u8),
            "mat3" | "mat3x3" => (3u8, 3u8),
            "mat4" | "mat4x4" => (4u8, 4u8),
            "mat3x4" => (3u8, 4u8),
            "mat4x3" => (4u8, 3u8),
            _ => unreachable!(),
        };
        let target_scalar = if expr_ir_ty.is_f16() {
            ScalarIrTy::F16
        } else {
            ScalarIrTy::F32
        };
        let col_ty = IrType::Vector {
            scalar: target_scalar,
            len: rows,
        };
        let mat_ty = IrType::Matrix {
            scalar: target_scalar,
            cols,
            rows,
        };

        let lowered: Vec<InstId> = args
            .iter()
            .map(|a| self.lower_expr(&a.value, out))
            .collect();

        // Single scalar diagonal matrix (e.g. `mat4(1.0)`).
        if lowered.len() == 1
            && let IrType::Scalar(s) = *self.inst_ty(lowered[0])
        {
            let diag = coerce_scalar(self.ir_fn, lowered[0], s, target_scalar);
            let zero = emit_zero_of_type(self.ir_fn, &IrType::Scalar(target_scalar));
            let mut col_vecs = Vec::with_capacity(usize::from(cols));
            for c in 0..cols {
                let mut elems = Vec::with_capacity(usize::from(rows));
                for r in 0..rows {
                    elems.push(if c == r { diag } else { zero });
                }
                col_vecs.push(emit_vec_construct(self.ir_fn, col_ty.clone(), elems));
            }
            return self
                .ir_fn
                .emit_typed(mat_ty, InstData::CompositeConstruct { elems: col_vecs });
        }

        // Single matrix resize/conversion (e.g. `mat3(m4)` or `mat4(m3)`).
        if lowered.len() == 1
            && let IrType::Matrix {
                scalar: src_s,
                cols: src_c,
                rows: src_r,
            } = *self.inst_ty(lowered[0])
        {
            let zero = emit_zero_of_type(self.ir_fn, &IrType::Scalar(target_scalar));
            let one = emit_splat_f32(self.ir_fn, &IrType::Scalar(target_scalar), 1.0);
            let mut col_vecs = Vec::with_capacity(usize::from(cols));
            for c in 0..cols {
                let mut elems = Vec::with_capacity(usize::from(rows));
                for r in 0..rows {
                    if c < src_c && r < src_r {
                        let extracted = self.ir_fn.emit_typed(
                            IrType::Scalar(src_s),
                            InstData::CompositeExtract {
                                composite: lowered[0],
                                indices: vec![u32::from(c), u32::from(r)],
                            },
                        );
                        elems.push(coerce_scalar(self.ir_fn, extracted, src_s, target_scalar));
                    } else if c == r {
                        elems.push(one);
                    } else {
                        elems.push(zero);
                    }
                }
                col_vecs.push(emit_vec_construct(self.ir_fn, col_ty.clone(), elems));
            }
            return self
                .ir_fn
                .emit_typed(mat_ty, InstData::CompositeConstruct { elems: col_vecs });
        }

        // Direct column vectors matching `(cols, rows)`.
        if lowered.len() == usize::from(cols)
            && lowered.iter().all(|&id| {
                matches!(
                    self.inst_ty(id),
                    IrType::Vector { scalar, len } if *scalar == target_scalar && *len == rows
                )
            })
        {
            return self
                .ir_fn
                .emit_typed(mat_ty, InstData::CompositeConstruct { elems: lowered });
        }

        // Flatten scalars/vectors into `cols * rows` components in column-major order.
        let expected = usize::from(cols) * usize::from(rows);
        let mut scalars = Vec::with_capacity(expected);
        for val in lowered {
            match self.inst_ty(val).clone() {
                IrType::Scalar(s) => {
                    scalars.push(coerce_scalar(self.ir_fn, val, s, target_scalar));
                }
                IrType::Vector { scalar, len } => {
                    for c in 0..u32::from(len) {
                        let comp = self.ir_fn.emit_typed(
                            IrType::Scalar(scalar),
                            InstData::CompositeExtract {
                                composite: val,
                                indices: vec![c],
                            },
                        );
                        scalars.push(coerce_scalar(self.ir_fn, comp, scalar, target_scalar));
                    }
                }
                _ => scalars.push(val),
            }
        }

        if scalars.len() != expected {
            return self.emit_diag(
                "E0528",
                format!(
                    "matrix constructor `{ctor}` expects {expected} scalar component(s), got {}",
                    scalars.len()
                ),
                span,
            );
        }

        let mut col_vecs = Vec::with_capacity(usize::from(cols));
        for c in 0..usize::from(cols) {
            let start = c * usize::from(rows);
            let elems = scalars[start..start + usize::from(rows)].to_vec();
            col_vecs.push(emit_vec_construct(self.ir_fn, col_ty.clone(), elems));
        }
        self.ir_fn
            .emit_typed(mat_ty, InstData::CompositeConstruct { elems: col_vecs })
    }

    fn lower_atomic_binary(
        &mut self,
        args: &[HirCallArg],
        span: Span,
        kind: AtomicKind,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let (scope, order) = self
            .ast_hints
            .atomic_turbofish
            .get(&span)
            .copied()
            .unwrap_or((AtomicScope::Device, AtomicOrder::Relaxed));
        let Ok(ptr) = self.require_arg(args, 0, "atomic_*", span, out) else {
            return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
        };
        let Ok(value) = self.require_arg(args, 1, "atomic_*", span, out) else {
            return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
        };
        let res_ty = self.inst_ty(value).clone();
        let data = match kind {
            AtomicKind::Add => InstData::AtomicAdd {
                ptr,
                value,
                scope,
                order,
            },
            AtomicKind::Min => InstData::AtomicMin {
                ptr,
                value,
                scope,
                order,
            },
            AtomicKind::Max => InstData::AtomicMax {
                ptr,
                value,
                scope,
                order,
            },
        };
        let id = self.ir_fn.emit_typed(res_ty, data);
        out.push(StructuredStmt::Eval(id));
        id
    }

    fn try_lower_glsl_std450(
        &mut self,
        name: &str,
        args: &[HirCallArg],
        expr_ir_ty: &IrType,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<InstId> {
        let short = name.rsplit("::").next().unwrap_or(name);
        let is_known = matches!(
            short,
            "sin"
                | "cos"
                | "tan"
                | "asin"
                | "acos"
                | "atan"
                | "atan2"
                | "sqrt"
                | "inversesqrt"
                | "abs"
                | "sign"
                | "floor"
                | "ceil"
                | "trunc"
                | "round"
                | "fract"
                | "mod"
                | "exp"
                | "exp2"
                | "log"
                | "log2"
                | "pow"
                | "min"
                | "max"
                | "clamp"
                | "mix"
                | "step"
                | "smoothstep"
                | "saturate"
                | "fma"
                | "dot"
                | "length"
                | "distance"
                | "cross"
                | "normalize"
                | "reflect"
                | "pack_half_2x16"
                | "unpack_half_2x16"
                | "pack_unorm_4x8"
                | "unpack_unorm_4x8"
                | "find_lsb"
                | "find_ilsb"
                | "find_msb"
                | "find_smsb"
                | "find_umsb"
        );
        if !is_known {
            return None;
        }

        let mut lowered_args: Vec<InstId> = args
            .iter()
            .map(|a| self.lower_expr(&a.value, out))
            .collect();
        let first_ty = lowered_args
            .first()
            .map_or(IrType::Scalar(ScalarIrTy::F32), |&id| {
                self.inst_ty(id).clone()
            });
        let is_float = first_ty.is_float() || expr_ir_ty.is_float();
        let is_unsigned = first_ty.is_unsigned_int() || expr_ir_ty.is_unsigned_int();

        // Special composite math expansions: `saturate(x)` -> `FClamp(x, 0, 1)`,
        // `distance(a, b)` -> `Length(a - b)`.
        if short == "saturate" && lowered_args.len() == 1 {
            let zero = emit_zero_of_type(self.ir_fn, &first_ty);
            let one = emit_splat_f32(self.ir_fn, &first_ty, 1.0);
            return Some(self.ir_fn.emit_typed(
                first_ty,
                InstData::GlslStd450 {
                    op: GlslStd450Op::FClamp,
                    args: vec![lowered_args[0], zero, one],
                },
            ));
        }
        if short == "distance" && lowered_args.len() == 2 {
            let scalar_out = if first_ty.is_f16() {
                ScalarIrTy::F16
            } else {
                ScalarIrTy::F32
            };
            let diff = self
                .ir_fn
                .emit_typed(first_ty, InstData::FSub32(lowered_args[0], lowered_args[1]));
            return Some(self.ir_fn.emit_typed(
                IrType::Scalar(scalar_out),
                InstData::GlslStd450 {
                    op: GlslStd450Op::Length,
                    args: vec![diff],
                },
            ));
        }

        let op = match short {
            "sin" => GlslStd450Op::Sin,
            "cos" => GlslStd450Op::Cos,
            "tan" => GlslStd450Op::Tan,
            "asin" => GlslStd450Op::Asin,
            "acos" => GlslStd450Op::Acos,
            "atan" => GlslStd450Op::Atan,
            "atan2" => GlslStd450Op::Atan2,
            "sqrt" => GlslStd450Op::Sqrt,
            "inversesqrt" => GlslStd450Op::InverseSqrt,
            "sign" => GlslStd450Op::Sign,
            "floor" => GlslStd450Op::Floor,
            "ceil" => GlslStd450Op::Ceil,
            "trunc" => GlslStd450Op::Trunc,
            "round" => GlslStd450Op::Round,
            "fract" => GlslStd450Op::Fract,
            "mod" => GlslStd450Op::Mod,
            "exp" => GlslStd450Op::Exp,
            "exp2" => GlslStd450Op::Exp2,
            "log" => GlslStd450Op::Log,
            "log2" => GlslStd450Op::Log2,
            "pow" => GlslStd450Op::Pow,
            "mix" => GlslStd450Op::FMix,
            "step" => GlslStd450Op::Step,
            "smoothstep" => GlslStd450Op::SmoothStep,
            "fma" => GlslStd450Op::Fma,
            "dot" => GlslStd450Op::Dot,
            "length" => GlslStd450Op::Length,
            "cross" => GlslStd450Op::Cross,
            "normalize" => GlslStd450Op::Normalize,
            "reflect" => GlslStd450Op::Reflect,
            "pack_half_2x16" => GlslStd450Op::PackHalf2x16,
            "unpack_half_2x16" => GlslStd450Op::UnpackHalf2x16,
            "pack_unorm_4x8" => GlslStd450Op::PackUnorm4x8,
            "unpack_unorm_4x8" => GlslStd450Op::UnpackUnorm4x8,
            "find_lsb" | "find_ilsb" => GlslStd450Op::FindILsb,
            "find_smsb" => GlslStd450Op::FindSMsb,
            "find_umsb" => GlslStd450Op::FindUMsb,
            "find_msb" => {
                if is_unsigned {
                    GlslStd450Op::FindUMsb
                } else {
                    GlslStd450Op::FindSMsb
                }
            }
            "abs" => {
                if is_float {
                    GlslStd450Op::FAbs
                } else {
                    GlslStd450Op::SAbs
                }
            }
            "min" => {
                if is_float {
                    GlslStd450Op::FMin
                } else if is_unsigned {
                    GlslStd450Op::UMin
                } else {
                    GlslStd450Op::SMin
                }
            }
            "max" => {
                if is_float {
                    GlslStd450Op::FMax
                } else if is_unsigned {
                    GlslStd450Op::UMax
                } else {
                    GlslStd450Op::SMax
                }
            }
            "clamp" => {
                if is_float {
                    GlslStd450Op::FClamp
                } else if is_unsigned {
                    GlslStd450Op::UClamp
                } else {
                    GlslStd450Op::SClamp
                }
            }
            _ => unreachable!(),
        };

        // Splat scalar operands to match vector operands on elementwise multi-argument
        // `GLSL.std.450` ops (e.g., `clamp(v, 0.0, 1.0)`, `mix(a, b, t)`, `min(v, 0.0)`).
        if matches!(
            op,
            GlslStd450Op::FMin
                | GlslStd450Op::UMin
                | GlslStd450Op::SMin
                | GlslStd450Op::FMax
                | GlslStd450Op::UMax
                | GlslStd450Op::SMax
                | GlslStd450Op::FClamp
                | GlslStd450Op::UClamp
                | GlslStd450Op::SClamp
                | GlslStd450Op::FMix
                | GlslStd450Op::Step
                | GlslStd450Op::SmoothStep
                | GlslStd450Op::Fma
                | GlslStd450Op::Pow
                | GlslStd450Op::Mod
                | GlslStd450Op::Atan2
        ) {
            let target_vec = lowered_args
                .iter()
                .find_map(|&id| match *self.inst_ty(id) {
                    IrType::Vector { scalar, len } => Some((scalar, len)),
                    _ => None,
                })
                .or(match *expr_ir_ty {
                    IrType::Vector { scalar, len } => Some((scalar, len)),
                    _ => None,
                });
            if let Some((vec_scalar, vec_len)) = target_vec {
                for arg_id in &mut lowered_args {
                    if let IrType::Scalar(arg_s) = *self.inst_ty(*arg_id) {
                        *arg_id =
                            splat_scalar_to_vector(self.ir_fn, *arg_id, arg_s, vec_scalar, vec_len);
                    }
                }
            }
        }

        let result_ty = match op {
            GlslStd450Op::Dot | GlslStd450Op::Length => IrType::Scalar(if first_ty.is_f16() {
                ScalarIrTy::F16
            } else {
                ScalarIrTy::F32
            }),
            GlslStd450Op::PackHalf2x16 | GlslStd450Op::PackUnorm4x8 => {
                IrType::Scalar(ScalarIrTy::U32)
            }
            GlslStd450Op::UnpackHalf2x16 => IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            GlslStd450Op::UnpackUnorm4x8 => IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 4,
            },
            GlslStd450Op::FindILsb | GlslStd450Op::FindSMsb | GlslStd450Op::FindUMsb => {
                IrType::Scalar(ScalarIrTy::I32)
            }
            _ if *expr_ir_ty != IrType::Void => expr_ir_ty.clone(),
            _ => lowered_args
                .iter()
                .find_map(|&id| {
                    let ty = self.inst_ty(id);
                    matches!(ty, IrType::Vector { .. }).then(|| ty.clone())
                })
                .unwrap_or(first_ty),
        };

        Some(self.ir_fn.emit_typed(
            result_ty,
            InstData::GlslStd450 {
                op,
                args: lowered_args,
            },
        ))
    }

    // -------------------------------------------------------------------------
    // Built-in `mc::*` library (Item 22) — real physical IR, zero stubs.
    // -------------------------------------------------------------------------

    /// `mc::reconstruct_world_pos(uv, depth)` or `mc::reconstruct_world_pos(coord, depth_tex)`:
    /// reconstructs NDC `vec4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, depth, 1.0)`, multiplies by
    /// `world.inv_proj` (`MatrixTimesVector`), and divides `.xyz` by `.w`. The legacy
    /// `inv_view_proj` alias maps to the engine's `inv_proj` member.
    fn lower_mc_reconstruct_world_pos(
        &mut self,
        args: &[HirCallArg],
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let Some(world_desc) = self.find_world_descriptor() else {
            return self.emit_diag(
                "E0525",
                "`mc::reconstruct_world_pos` requires `WorldUBO` (Set 0 binding 0)",
                span,
            );
        };
        let Ok(first_arg) = self.require_arg(args, 0, "mc::reconstruct_world_pos", span, out)
        else {
            return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
        };
        let Some(second_expr) = args.get(1) else {
            return self.emit_diag(
                "E0528",
                "`mc::reconstruct_world_pos` expects 2 arguments `(uv, depth)` or `(coord, depth_tex)`",
                span,
            );
        };

        let f32_ty = IrType::Scalar(ScalarIrTy::F32);

        // Determine `(uv: vec2, depth: f32)` from either `(coord, depth_tex)` or `(uv, depth)`.
        let (uv, depth) = if let HirExprKind::Path { path, .. } = &second_expr.value.kind
            && let Some(depth_desc_idx) = self.resolve_descriptor_path(path)
        {
            let lod0 = self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::ConstI32(0));
            let depth_sample = self.ir_fn.emit_typed(
                IrType::Vector {
                    scalar: ScalarIrTy::F32,
                    len: 4,
                },
                InstData::ImageSample {
                    descriptor_index: depth_desc_idx,
                    sampler_descriptor_index: None,
                    image_inst: None,
                    coord: first_arg,
                    lod: lod0,
                    is_fetch: true,
                },
            );
            let depth_val = self.ir_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: depth_sample,
                    indices: vec![0],
                },
            );
            let pixel_center = self.coord_to_uv(first_arg);
            let img_size = self.ir_fn.emit_typed(
                IrType::Vector {
                    scalar: ScalarIrTy::U32,
                    len: 2,
                },
                InstData::ImageQuerySize {
                    descriptor_index: depth_desc_idx,
                    lod: Some(lod0),
                },
            );
            let sw = self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::CompositeExtract {
                    composite: img_size,
                    indices: vec![0],
                },
            );
            let sh = self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::CompositeExtract {
                    composite: img_size,
                    indices: vec![1],
                },
            );
            let fsw = coerce_scalar(self.ir_fn, sw, ScalarIrTy::U32, ScalarIrTy::F32);
            let fsh = coerce_scalar(self.ir_fn, sh, ScalarIrTy::U32, ScalarIrTy::F32);
            let vec2_ty = IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            };
            let size_f = self
                .ir_fn
                .emit_typed(vec2_ty.clone(), InstData::MakeVec2(fsw, fsh));
            let uv_val = self
                .ir_fn
                .emit_typed(vec2_ty, InstData::FDiv32(pixel_center, size_f));
            (uv_val, depth_val)
        } else {
            let depth_val = self.lower_expr(&second_expr.value, out);
            let uv_val = if self.inst_ty(first_arg).is_float() {
                first_arg
            } else {
                self.coord_to_uv(first_arg)
            };
            (uv_val, depth_val)
        };

        self.emit_ndc_to_world(uv, depth, world_desc, span)
    }

    /// Converts an integer pixel coordinate `ivec2`/`uvec2` to normalized `[0, 1]` UV.
    ///
    /// The 544-byte engine UBO has no screen-size member, so coordinates are treated as
    /// already normalized; callers holding pixel coordinates should pass them as `vec2` UVs
    /// obtained from their target dimensions.
    fn coord_to_uv(&mut self, coord: InstId) -> InstId {
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let vec2_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 2,
        };
        let coord_scalar = self.inst_ty(coord).scalar_ty().unwrap_or(ScalarIrTy::I32);
        let cx = self.ir_fn.emit_typed(
            IrType::Scalar(coord_scalar),
            InstData::CompositeExtract {
                composite: coord,
                indices: vec![0],
            },
        );
        let cy = self.ir_fn.emit_typed(
            IrType::Scalar(coord_scalar),
            InstData::CompositeExtract {
                composite: coord,
                indices: vec![1],
            },
        );
        let fx = coerce_scalar(self.ir_fn, cx, coord_scalar, ScalarIrTy::F32);
        let fy = coerce_scalar(self.ir_fn, cy, coord_scalar, ScalarIrTy::F32);
        let half = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.5));
        let px = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FAdd32(fx, half));
        let py = self.ir_fn.emit_typed(f32_ty, InstData::FAdd32(fy, half));
        self.ir_fn.emit_typed(vec2_ty, InstData::MakeVec2(px, py))
    }

    fn emit_ndc_to_world(
        &mut self,
        uv: InstId,
        depth: InstId,
        world_desc: u32,
        span: Span,
    ) -> InstId {
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let vec3_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 3,
        };
        let vec4_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 4,
        };

        let uv_x = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: uv,
                indices: vec![0],
            },
        );
        let uv_y = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: uv,
                indices: vec![1],
            },
        );
        let two = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(2.0));
        let one = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
        let ndc_x_2 = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FMul32(uv_x, two));
        let ndc_x = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FSub32(ndc_x_2, one));
        let ndc_y_2 = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FMul32(uv_y, two));
        let ndc_y = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FSub32(ndc_y_2, one));
        let ndc = self.ir_fn.emit_typed(
            vec4_ty.clone(),
            InstData::MakeVec4(ndc_x, ndc_y, depth, one),
        );

        let inv_proj = self.emit_builtin_struct_field_load(
            world_desc,
            &WORLD_UBO_LAYOUT,
            "inv_proj",
            IrStorageClass::Uniform,
            span,
        );
        let inv_view = self.emit_builtin_struct_field_load(
            world_desc,
            &WORLD_UBO_LAYOUT,
            "inv_view",
            IrStorageClass::Uniform,
            span,
        );
        let mat4_ty = IrType::Matrix {
            scalar: ScalarIrTy::F32,
            cols: 4,
            rows: 4,
        };
        let inv_vp = self.ir_fn.emit_typed(
            mat4_ty,
            InstData::MatrixTimesMatrix {
                lhs: inv_view,
                rhs: inv_proj,
            },
        );
        let clip = self.ir_fn.emit_typed(
            vec4_ty,
            InstData::MatrixTimesVector {
                mat: inv_vp,
                vec: ndc,
            },
        );
        let cx = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![0],
            },
        );
        let cy = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![1],
            },
        );
        let cz = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![2],
            },
        );
        let cw = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: clip,
                indices: vec![3],
            },
        );
        let wx = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FDiv32(cx, cw));
        let wy = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FDiv32(cy, cw));
        let wz = self.ir_fn.emit_typed(f32_ty, InstData::FDiv32(cz, cw));
        self.ir_fn
            .emit_typed(vec3_ty, InstData::MakeVec3(wx, wy, wz))
    }

    /// `mc::view_dir(world_pos)` (or `mc::view_dir(coord)`):
    /// computes `normalize(world_pos - world.camera_pos)` using `WorldUBO::camera_pos` and
    /// `GlslStd450Op::Normalize`.
    fn lower_mc_view_dir(
        &mut self,
        args: &[HirCallArg],
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let Some(world_desc) = self.find_world_descriptor() else {
            return self.emit_diag(
                "E0525",
                "`mc::view_dir` requires `WorldUBO` (Set 0 binding 0)",
                span,
            );
        };
        let Ok(arg) = self.require_arg(args, 0, "mc::view_dir", span, out) else {
            return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
        };
        let vec3_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 3,
        };
        let world_pos = if self.inst_ty(arg).vector_len() == Some(3) {
            arg
        } else {
            let uv = if self.inst_ty(arg).is_float() {
                arg
            } else {
                self.coord_to_uv(arg)
            };
            let far_depth = self
                .ir_fn
                .emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(1.0));
            self.emit_ndc_to_world(uv, far_depth, world_desc, span)
        };
        let cam_pos = self.load_world_field("camera_pos_frac", span);
        let diff = self
            .ir_fn
            .emit_typed(vec3_ty.clone(), InstData::FSub32(world_pos, cam_pos));
        self.ir_fn.emit_typed(
            vec3_ty,
            InstData::GlslStd450 {
                op: GlslStd450Op::Normalize,
                args: vec![diff],
            },
        )
    }

    /// `mc::sample_sky(dir)`: gradient sky driven by the engine's `sun_dir` and `rain`, with
    /// horizon tint from the biome fog telemetry in `packParams[11].rgb`. The 544-byte engine
    /// UBO has no `sky_*`/`sun_color` members, so the zenith is derived from the horizon and the
    /// sun tint is a neutral warm constant.
    fn lower_mc_sample_sky(
        &mut self,
        args: &[HirCallArg],
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let Ok(dir) = self.require_arg(args, 0, "mc::sample_sky", span, out) else {
            return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
        };
        let f32_ty = IrType::Scalar(ScalarIrTy::F32);
        let vec3_ty = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 3,
        };

        let fog = self.world_pack_param_slot(11, span);
        let fog_lanes = [0u32, 1, 2].map(|lane| {
            self.ir_fn.emit_typed(
                f32_ty.clone(),
                InstData::CompositeExtract {
                    composite: fog,
                    indices: vec![lane],
                },
            )
        });
        let sky_horizon = self.ir_fn.emit_typed(
            vec3_ty.clone(),
            InstData::MakeVec3(fog_lanes[0], fog_lanes[1], fog_lanes[2]),
        );
        let zenith_scale = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.6));
        let sky_zenith = self.ir_fn.emit_typed(
            vec3_ty.clone(),
            InstData::VectorTimesScalar {
                vec: sky_horizon,
                scalar: zenith_scale,
            },
        );
        let sun_dir = self.world_sun_dir(span);
        let r = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
        let g = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.96));
        let b = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.86));
        let sun_color = self
            .ir_fn
            .emit_typed(vec3_ty.clone(), InstData::MakeVec3(r, g, b));
        let rain_strength = self.world_rain(span);

        let zero = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.0));
        let one = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(1.0));
        let dir_y = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::CompositeExtract {
                composite: dir,
                indices: vec![1],
            },
        );
        let elev = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::GlslStd450 {
                op: GlslStd450Op::FClamp,
                args: vec![dir_y, zero, one],
            },
        );
        let elev_v3 = self
            .ir_fn
            .emit_typed(vec3_ty.clone(), InstData::MakeVec3(elev, elev, elev));
        let gradient = self.ir_fn.emit_typed(
            vec3_ty.clone(),
            InstData::GlslStd450 {
                op: GlslStd450Op::FMix,
                args: vec![sky_horizon, sky_zenith, elev_v3],
            },
        );

        let ndotl = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::GlslStd450 {
                op: GlslStd450Op::Dot,
                args: vec![dir, sun_dir],
            },
        );
        let sun_cos = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::GlslStd450 {
                op: GlslStd450Op::FMax,
                args: vec![ndotl, zero],
            },
        );
        let sun_exp = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(64.0));
        let sun_glow = self.ir_fn.emit_typed(
            f32_ty.clone(),
            InstData::GlslStd450 {
                op: GlslStd450Op::Pow,
                args: vec![sun_cos, sun_exp],
            },
        );
        let sun_term = self.ir_fn.emit_typed(
            vec3_ty.clone(),
            InstData::VectorTimesScalar {
                vec: sun_color,
                scalar: sun_glow,
            },
        );
        let raw_sky = self
            .ir_fn
            .emit_typed(vec3_ty.clone(), InstData::FAdd32(gradient, sun_term));

        let half = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::const_f32(0.5));
        let rain_dim = self
            .ir_fn
            .emit_typed(f32_ty.clone(), InstData::FMul32(rain_strength, half));
        let rain_atten = self
            .ir_fn
            .emit_typed(f32_ty, InstData::FSub32(one, rain_dim));
        self.ir_fn.emit_typed(
            vec3_ty,
            InstData::VectorTimesScalar {
                vec: raw_sky,
                scalar: rain_atten,
            },
        )
    }

    fn lower_mc_debug_write(
        &mut self,
        args: &[HirCallArg],
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        let Some(HirExprKind::StringLit(channel)) = args.first().map(|a| &a.value.kind) else {
            return self.emit_diag(
                "E0528",
                "`mc::debug_write` requires a string literal channel name as its first argument",
                span,
            );
        };
        let Ok(coord) = self.require_arg(args, 1, "mc::debug_write", span, out) else {
            return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
        };
        let Ok(value) = self.require_arg(args, 2, "mc::debug_write", span, out) else {
            return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
        };
        let desc_name = SmolStr::new(format!("debug_{channel}"));
        let Some(&descriptor_index) = self.storage_image_descriptors.get(&desc_name) else {
            return self.emit_diag(
                "E0525",
                format!("debug channel `{channel}` has no bound storage image descriptor"),
                span,
            );
        };
        let id = self.ir_fn.emit_typed(
            IrType::Void,
            InstData::ImageStore {
                descriptor_index,
                coord,
                value,
            },
        );
        out.push(StructuredStmt::Eval(id));
        id
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_method_call_expr(
        &mut self,
        receiver: &HirExpr,
        method: &SmolStr,
        resolved_fn: Option<HirFnId>,
        args: &[HirCallArg],
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> InstId {
        // Resolved inherent or trait `impl` method -> `InstData::FunctionCall` with receiver as arg 0.
        if let Some(fn_id) = resolved_fn
            && let Some(callee_name) = self.fn_names_by_id.get(&fn_id).cloned()
        {
            let param_modes: Vec<ParamMode> = self.hir.functions[fn_id]
                .params
                .iter()
                .map(|p| p.mode)
                .collect();
            let mut lowered_args = Vec::with_capacity(1 + args.len());
            lowered_args.push(self.lower_call_arg_for_mode(
                receiver,
                param_modes.first().copied(),
                "__self_inout_tmp",
                out,
            ));
            for (idx, arg) in args.iter().enumerate() {
                lowered_args.push(self.lower_call_arg_for_mode(
                    &arg.value,
                    param_modes.get(idx + 1).copied(),
                    "__method_inout_tmp",
                    out,
                ));
            }
            let id = self.ir_fn.emit_typed(
                expr_ir_ty.clone(),
                InstData::FunctionCall {
                    callee: callee_name,
                    args: lowered_args,
                },
            );
            out.push(StructuredStmt::Eval(id));
            return id;
        }

        // Complete `SPV_KHR_ray_query` method suite.
        if let Some(id) =
            self.try_lower_ray_query_method(receiver, method.as_str(), args, span, out)
        {
            return id;
        }

        // Texture & Storage Image methods.
        if let Some(id) =
            self.try_lower_image_method(receiver, method.as_str(), args, expr_ir_ty, span, out)
        {
            return id;
        }

        // BDA pointer methods (`.offset(i)`, `.load()`, `.store(v)`).
        let recv_val = self.lower_expr(receiver, out);
        match method.as_str() {
            "offset" => {
                let Ok(idx) = self.require_arg(args, 0, "ptr::offset", span, out) else {
                    return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
                };
                let idx_i64 = match self.inst_ty(idx).scalar_ty() {
                    Some(ScalarIrTy::I64) => idx,
                    Some(from) => coerce_scalar(self.ir_fn, idx, from, ScalarIrTy::I64),
                    None => idx,
                };
                let (stride_bytes, align) = match &receiver.kind {
                    HirExprKind::Path { path, .. } if !path.contains("::") => {
                        self.ptr_meta.get(path).copied().unwrap_or_else(|| {
                            match self.inst_ty(recv_val) {
                                IrType::Pointer { pointee, align, .. } => {
                                    (u64::from(pointee.byte_size().max(1)), *align)
                                }
                                _ => (4, 4),
                            }
                        })
                    }
                    _ => match self.inst_ty(recv_val) {
                        IrType::Pointer { pointee, align, .. } => {
                            (u64::from(pointee.byte_size().max(1)), *align)
                        }
                        _ => (4, 4),
                    },
                };
                self.ir_fn.uses_bda = true;
                let ptr_ty = self.inst_ty(recv_val).clone();
                let off = self.ir_fn.emit_typed(
                    ptr_ty,
                    InstData::BdaOffset {
                        base_ptr: recv_val,
                        index_i64: idx_i64,
                        stride_bytes,
                        align,
                    },
                );
                out.push(StructuredStmt::Eval(off));
                off
            }
            "load" => {
                let align = match self.inst_ty(recv_val) {
                    IrType::Pointer { align, .. } => *align,
                    _ => 4,
                };
                let is_mut = matches!(
                    self.hir.types.get(receiver.ty),
                    Type::Pointer { mutable: true, .. }
                );
                self.ir_fn.uses_bda = true;
                self.ir_fn.emit_typed(
                    expr_ir_ty.clone(),
                    InstData::BdaLoad {
                        ptr: recv_val,
                        access: MemoryAccess {
                            align,
                            non_writable: !is_mut,
                            restrict: false,
                        },
                    },
                )
            }
            "store" => {
                let Ok(val) = self.require_arg(args, 0, "ptr::store", span, out) else {
                    return self.ir_fn.emit_typed(IrType::Void, InstData::Nop);
                };
                let align = match self.inst_ty(recv_val) {
                    IrType::Pointer { align, .. } => *align,
                    _ => 4,
                };
                self.ir_fn.uses_bda = true;
                let id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::BdaStore {
                        ptr: recv_val,
                        value: val,
                        access: MemoryAccess {
                            align,
                            non_writable: false,
                            restrict: false,
                        },
                    },
                );
                out.push(StructuredStmt::Eval(id));
                id
            }
            _ => self.emit_diag(
                "E0529",
                format!("unsupported method call `.{method}(...)` during typed IR lowering"),
                span,
            ),
        }
    }

    fn try_lower_ray_query_method(
        &mut self,
        receiver: &HirExpr,
        method: &str,
        args: &[HirCallArg],
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<InstId> {
        let is_rq_receiver = matches!(self.hir.types.get(receiver.ty), Type::RayQuery)
            || matches!(&receiver.kind, HirExprKind::Path { path, .. } if self.ray_query_handles.contains_key(path));
        if !is_rq_receiver {
            return None;
        }

        let ray_query_id = self.ray_query_id_for_receiver(receiver);
        let id = match method {
            "initialize" => {
                if args.len() < 7 {
                    return Some(self.emit_diag(
                        "E0528",
                        "`RayQuery::initialize` requires 7 arguments",
                        span,
                    ));
                }
                let Some(tlas_descriptor_index) = match &args[0].value.kind {
                    HirExprKind::Path { path, .. } => self.resolve_descriptor_path(path),
                    _ => None,
                }
                .or(self.tlas_descriptor_index) else {
                    return Some(self.emit_diag(
                        "E0525",
                        "`RayQuery::initialize` requires a bound acceleration structure (`mc::tlas`)",
                        span,
                    ));
                };
                let flags = self.lower_expr(&args[1].value, out);
                let cull_mask = self.lower_expr(&args[2].value, out);
                let origin = self.lower_expr(&args[3].value, out);
                let t_min = self.lower_expr(&args[4].value, out);
                let dir = self.lower_expr(&args[5].value, out);
                let t_max = self.lower_expr(&args[6].value, out);
                let init_id = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::RayQueryInitialize {
                        ray_query_id,
                        tlas_descriptor_index,
                        flags,
                        cull_mask,
                        origin,
                        t_min,
                        dir,
                        t_max,
                    },
                );
                out.push(StructuredStmt::Eval(init_id));
                init_id
            }
            "proceed" => {
                let p = self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::Bool),
                    InstData::RayQueryProceed(ray_query_id),
                );
                out.push(StructuredStmt::Eval(p));
                p
            }
            "committed_type" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::RayQueryCommittedType(ray_query_id),
            ),
            "candidate_type" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::RayQueryCandidateType(ray_query_id),
            ),
            "confirm_intersection" => {
                let c = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::RayQueryConfirmIntersection(ray_query_id),
                );
                out.push(StructuredStmt::Eval(c));
                c
            }
            "terminate" => {
                let t = self
                    .ir_fn
                    .emit_typed(IrType::Void, InstData::RayQueryTerminate(ray_query_id));
                out.push(StructuredStmt::Eval(t));
                t
            }
            "generate_intersection" => {
                let Ok(t_hit) =
                    self.require_arg(args, 0, "RayQuery::generate_intersection", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let g = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::RayQueryGenerateIntersection {
                        ray_query_id,
                        t_hit,
                    },
                );
                out.push(StructuredStmt::Eval(g));
                g
            }
            "committed_t" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::F32),
                InstData::RayQueryCommittedT(ray_query_id),
            ),
            "candidate_t" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::F32),
                InstData::RayQueryCandidateT(ray_query_id),
            ),
            "committed_instance_id" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::RayQueryCommittedInstanceId(ray_query_id),
            ),
            "committed_instance_custom_index" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::RayQueryCommittedInstanceCustomIndex(ray_query_id),
            ),
            "committed_primitive_index" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::RayQueryCommittedPrimitiveIndex(ray_query_id),
            ),
            "committed_geometry_index" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::RayQueryCommittedGeometryIndex(ray_query_id),
            ),
            "committed_object_to_world" => self.ir_fn.emit_typed(
                IrType::Matrix {
                    scalar: ScalarIrTy::F32,
                    cols: 4,
                    rows: 3,
                },
                InstData::RayQueryCommittedObjectToWorld(ray_query_id),
            ),
            "committed_world_to_object" => self.ir_fn.emit_typed(
                IrType::Matrix {
                    scalar: ScalarIrTy::F32,
                    cols: 4,
                    rows: 3,
                },
                InstData::RayQueryCommittedWorldToObject(ray_query_id),
            ),
            "committed_front_face" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::Bool),
                InstData::RayQueryCommittedFrontFace(ray_query_id),
            ),
            "committed_barycentrics" => self.ir_fn.emit_typed(
                IrType::Vector {
                    scalar: ScalarIrTy::F32,
                    len: 2,
                },
                InstData::RayQueryCommittedBarycentrics(ray_query_id),
            ),
            "committed_triangle_vertex_positions" => self.ir_fn.emit_typed(
                IrType::Array {
                    element: Box::new(IrType::Vector {
                        scalar: ScalarIrTy::F32,
                        len: 3,
                    }),
                    len: 3,
                },
                InstData::RayQueryTriangleVertexPositions {
                    ray_query_id,
                    committed: true,
                },
            ),
            "candidate_triangle_vertex_positions" => self.ir_fn.emit_typed(
                IrType::Array {
                    element: Box::new(IrType::Vector {
                        scalar: ScalarIrTy::F32,
                        len: 3,
                    }),
                    len: 3,
                },
                InstData::RayQueryTriangleVertexPositions {
                    ray_query_id,
                    committed: false,
                },
            ),
            _ => return None,
        };
        Some(id)
    }

    fn resolve_image_receiver(
        &mut self,
        receiver: &HirExpr,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<(u32, Option<InstId>)> {
        match &receiver.kind {
            HirExprKind::Path { path, .. } => {
                let desc_idx = self.resolve_descriptor_path(path)?;
                Some((desc_idx, None))
            }
            HirExprKind::Index {
                receiver: arr_recv,
                index,
            } => {
                let HirExprKind::Path { path, .. } = &arr_recv.kind else {
                    return None;
                };
                let desc_idx = self.resolve_descriptor_path(path)?;
                let idx_val = self.lower_expr(index, out);
                let chain = self.ir_fn.emit(
                    None,
                    InstData::ResourceAccessChain {
                        descriptor_index: desc_idx,
                        index: idx_val,
                    },
                );
                let loaded = self.ir_fn.emit(None, InstData::ResourceLoad { ptr: chain });
                out.push(StructuredStmt::Eval(chain));
                out.push(StructuredStmt::Eval(loaded));
                Some((desc_idx, Some(loaded)))
            }
            _ => None,
        }
    }

    fn try_lower_image_method(
        &mut self,
        receiver: &HirExpr,
        method: &str,
        args: &[HirCallArg],
        expr_ir_ty: &IrType,
        span: Span,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<InstId> {
        if !matches!(
            method,
            "sample"
                | "sample_lod"
                | "sample_grad"
                | "sample_compare"
                | "sample_compare_lod"
                | "gather"
                | "fetch"
                | "store"
                | "query_size"
                | "query_levels"
                | "subpass_load"
                | "fetch_ms"
                | "texel_pointer"
        ) {
            return None;
        }

        let (descriptor_index, image_inst) = self.resolve_image_receiver(receiver, out)?;
        let vec4_f32 = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 4,
        };
        let res_ty = if *expr_ir_ty != IrType::Void {
            expr_ir_ty.clone()
        } else {
            vec4_f32.clone()
        };

        let mut sampler_descriptor_index = None;
        let arg_offset = usize::from(args.first().is_some_and(|a| {
            let is_sampler = matches!(
                self.hir.types.get(a.value.ty),
                Type::Sampler | Type::SamplerShadow
            );
            if is_sampler
                && let HirExprKind::Path { path, .. } = &a.value.kind
            {
                sampler_descriptor_index = self.resolve_descriptor_path(path);
            }
            is_sampler
        }));

        let id = match method {
            "store" => {
                if !self
                    .storage_image_descriptors
                    .values()
                    .any(|&idx| idx == descriptor_index)
                {
                    return None;
                }
                let Ok(coord) = self.require_arg(args, 0, "image::store", span, out) else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let Ok(value) = self.require_arg(args, 1, "image::store", span, out) else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let st = self.ir_fn.emit_typed(
                    IrType::Void,
                    InstData::ImageStore {
                        descriptor_index,
                        coord,
                        value,
                    },
                );
                out.push(StructuredStmt::Eval(st));
                st
            }
            "sample" => {
                let Ok(coord) = self.require_arg(args, arg_offset, "texture::sample", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                if self.stage == Some(ShaderStage::Fragment) {
                    self.ir_fn.emit_typed(
                        res_ty,
                        InstData::ImageSampleImplicitLod {
                            descriptor_index,
                            sampler_descriptor_index,
                            image_inst,
                            coord,
                        },
                    )
                } else {
                    let lod = self
                        .ir_fn
                        .emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(0.0));
                    self.ir_fn.emit_typed(
                        res_ty,
                        InstData::ImageSample {
                            descriptor_index,
                            sampler_descriptor_index,
                            image_inst,
                            coord,
                            lod,
                            is_fetch: false,
                        },
                    )
                }
            }
            "sample_lod" => {
                let Ok(coord) =
                    self.require_arg(args, arg_offset, "texture::sample_lod", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let Ok(lod) =
                    self.require_arg(args, arg_offset + 1, "texture::sample_lod", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                self.ir_fn.emit_typed(
                    res_ty,
                    InstData::ImageSample {
                        descriptor_index,
                        sampler_descriptor_index,
                        image_inst,
                        coord,
                        lod,
                        is_fetch: false,
                    },
                )
            }
            "sample_grad" => {
                let Ok(coord) =
                    self.require_arg(args, arg_offset, "texture::sample_grad", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let Ok(ddx) =
                    self.require_arg(args, arg_offset + 1, "texture::sample_grad", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let Ok(ddy) =
                    self.require_arg(args, arg_offset + 2, "texture::sample_grad", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                self.ir_fn.emit_typed(
                    res_ty,
                    InstData::ImageSampleGrad {
                        descriptor_index,
                        sampler_descriptor_index,
                        image_inst,
                        coord,
                        ddx,
                        ddy,
                    },
                )
            }
            "sample_compare" | "sample_compare_lod" => {
                let Ok(coord) =
                    self.require_arg(args, arg_offset, "texture::sample_compare", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let Ok(dref) =
                    self.require_arg(args, arg_offset + 1, "texture::sample_compare", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let lod = match args.get(arg_offset + 2) {
                    Some(a) => self.lower_expr(&a.value, out),
                    None => self
                        .ir_fn
                        .emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(0.0)),
                };
                self.ir_fn.emit_typed(
                    IrType::Scalar(ScalarIrTy::F32),
                    InstData::ImageSampleDrefExplicitLod {
                        descriptor_index,
                        sampler_descriptor_index,
                        image_inst,
                        coord,
                        dref,
                        lod,
                    },
                )
            }
            "gather" => {
                let Ok(coord) = self.require_arg(args, arg_offset, "texture::gather", span, out)
                else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let component = args
                    .get(arg_offset + 1)
                    .and_then(|a| parse_const_i64_hir(&a.value))
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
                self.ir_fn.emit_typed(
                    res_ty,
                    InstData::ImageGather {
                        descriptor_index,
                        sampler_descriptor_index,
                        image_inst,
                        coord,
                        component,
                    },
                )
            }
            "fetch" => {
                let Ok(coord) = self.require_arg(args, 0, "texture::fetch", span, out) else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let lod = match args.get(1) {
                    Some(a) => self.lower_expr(&a.value, out),
                    None => self
                        .ir_fn
                        .emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::ConstI32(0)),
                };
                self.ir_fn.emit_typed(
                    res_ty,
                    InstData::ImageSample {
                        descriptor_index,
                        sampler_descriptor_index: None,
                        image_inst,
                        coord,
                        lod,
                        is_fetch: true,
                    },
                )
            }
            "query_size" => {
                let lod = args.first().map(|a| self.lower_expr(&a.value, out));
                let size_ty = if *expr_ir_ty != IrType::Void {
                    expr_ir_ty.clone()
                } else {
                    IrType::Vector {
                        scalar: ScalarIrTy::U32,
                        len: 2,
                    }
                };
                self.ir_fn.emit_typed(
                    size_ty,
                    InstData::ImageQuerySize {
                        descriptor_index,
                        lod,
                    },
                )
            }
            "subpass_load" => {
                let sample = args.first().map(|a| self.lower_expr(&a.value, out));
                self.ir_fn.emit_typed(
                    res_ty,
                    InstData::SubpassLoad {
                        descriptor_index,
                        sample,
                    },
                )
            }
            "fetch_ms" => {
                let Some(coord) = args.first().map(|a| self.lower_expr(&a.value, out)) else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let sample = args.get(1).map_or_else(
                    || {
                        self.ir_fn
                            .emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::ConstI32(0))
                    },
                    |a| self.lower_expr(&a.value, out),
                );
                self.ir_fn.emit_typed(
                    res_ty,
                    InstData::ImageFetchMs {
                        descriptor_index,
                        coord,
                        sample,
                    },
                )
            }
            "query_levels" => self.ir_fn.emit_typed(
                IrType::Scalar(ScalarIrTy::U32),
                InstData::ImageQueryLevels { descriptor_index },
            ),
            "texel_pointer" => {
                let Ok(coord) = self.require_arg(args, 0, "image::texel_pointer", span, out) else {
                    return Some(self.ir_fn.emit_typed(IrType::Void, InstData::Nop));
                };
                let sample = match args.get(1) {
                    Some(a) => self.lower_expr(&a.value, out),
                    None => self
                        .ir_fn
                        .emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::ConstI32(0)),
                };
                let ptr_ty = IrType::Pointer {
                    pointee: Box::new(IrType::Scalar(ScalarIrTy::U32)),
                    align: 4,
                    storage: IrStorageClass::Image,
                };
                self.ir_fn.emit_typed(
                    ptr_ty,
                    InstData::ImageTexelPointer {
                        descriptor_index,
                        coord,
                        sample,
                    },
                )
            }
            _ => return None,
        };
        Some(id)
    }
}

#[derive(Clone, Copy)]
enum AtomicKind {
    Add,
    Min,
    Max,
}

fn canonicalize_builtin_field_name<'a>(desc: &BuiltinStructDesc, field_name: &'a str) -> &'a str {
    if desc.name == "InstanceMeta" && field_name == "albedo_slot" {
        return "texture_id";
    }
    if desc.name == "WorldUBO" {
        // Legacy VSL field names kept source-compatible with the 544-byte engine ABI.
        return match field_name {
            "time" | "time_sec" => "game_time_sec",
            "rain_strength" => "rain",
            "camera_pos" => "camera_pos_frac",
            "inv_view_proj" => "inv_proj",
            "view_proj" => "cur_view_proj",
            "far_plane" => "depth_far",
            other => other,
        };
    }
    field_name
}

/// `packParams` slot aliases. The engine exposes a `vec4[12]` array; built-in layout descriptors
/// model each slot as a separate `vec4` member, with slot 0 keeping the unsuffixed engine name.
const fn world_pack_param_field_name(slot: u32) -> &'static str {
    match slot {
        0 => "pack_params",
        1 => "pack_params_1",
        2 => "pack_params_2",
        3 => "pack_params_3",
        4 => "pack_params_4",
        5 => "pack_params_5",
        6 => "pack_params_6",
        7 => "pack_params_7",
        8 => "pack_params_8",
        9 => "pack_params_9",
        10 => "pack_params_10",
        11 => "pack_params_11",
        _ => "",
    }
}

const fn is_assign_op(op: BinaryOp) -> bool {
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

const fn is_composite_ir_type(ty: &IrType) -> bool {
    matches!(ty, IrType::Struct(_) | IrType::Array { .. })
}

const fn pack_param_type_to_scalar(ty: PackParamType) -> ScalarIrTy {
    match ty {
        PackParamType::Bool => ScalarIrTy::Bool,
        PackParamType::I32 | PackParamType::Enum => ScalarIrTy::I32,
        PackParamType::U32 => ScalarIrTy::U32,
        PackParamType::F32 | PackParamType::Vec3 => ScalarIrTy::F32,
    }
}

fn alloc_local_var(ir_fn: &mut IrFunction, name: SmolStr, ty: IrType) -> u32 {
    let idx = ir_fn.local_vars.len() as u32;
    ir_fn.local_vars.push(IrLocalVar { name, ty });
    idx
}

fn emit_local_addr(ir_fn: &mut IrFunction, local_index: u32, pointee_ty: &IrType) -> InstId {
    ir_fn.emit_typed(
        IrType::Pointer {
            pointee: Box::new(pointee_ty.clone()),
            align: 4,
            storage: IrStorageClass::Function,
        },
        InstData::LocalAddr { local_index },
    )
}

fn emit_vec_construct(ir_fn: &mut IrFunction, vec_ty: IrType, elems: Vec<InstId>) -> InstId {
    match elems.as_slice() {
        [x, y] => ir_fn.emit_typed(vec_ty, InstData::MakeVec2(*x, *y)),
        [x, y, z] => ir_fn.emit_typed(vec_ty, InstData::MakeVec3(*x, *y, *z)),
        [x, y, z, w] => ir_fn.emit_typed(vec_ty, InstData::MakeVec4(*x, *y, *z, *w)),
        _ => ir_fn.emit_typed(vec_ty, InstData::CompositeConstruct { elems }),
    }
}

fn coerce_to_bool(ir_fn: &mut IrFunction, val: InstId) -> InstId {
    let bool_ty = IrType::Scalar(ScalarIrTy::Bool);
    match ir_fn.insts[val].ir_ty.scalar_ty() {
        Some(ScalarIrTy::Bool) | None => val,
        Some(ScalarIrTy::F32) => {
            let zero = ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(0.0));
            ir_fn.emit_typed(bool_ty, InstData::FCmpNe(val, zero))
        }
        Some(ScalarIrTy::F16) => {
            let zero = ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::F16), InstData::ConstF16(0));
            ir_fn.emit_typed(bool_ty, InstData::FCmpNe(val, zero))
        }
        Some(int_scalar) => {
            let zero = emit_zero_of_type(ir_fn, &IrType::Scalar(int_scalar));
            ir_fn.emit_typed(bool_ty, InstData::ICmpNe(val, zero))
        }
    }
}

fn splat_scalar_to_vector(
    ir_fn: &mut IrFunction,
    val: InstId,
    from: ScalarIrTy,
    to: ScalarIrTy,
    len: u8,
) -> InstId {
    let coerced = coerce_scalar(ir_fn, val, from, to);
    let vec_ty = IrType::Vector { scalar: to, len };
    let elems = vec![coerced; usize::from(len)];
    emit_vec_construct(ir_fn, vec_ty, elems)
}

fn coerce_scalar(ir_fn: &mut IrFunction, val: InstId, from: ScalarIrTy, to: ScalarIrTy) -> InstId {
    if from == to {
        return val;
    }
    let ty = IrType::Scalar(to);
    match to {
        ScalarIrTy::U32 => ir_fn.emit_typed(ty, InstData::CastToU32(val)),
        ScalarIrTy::I32 => ir_fn.emit_typed(ty, InstData::CastToI32(val)),
        ScalarIrTy::F32 => {
            if from == ScalarIrTy::F16 {
                ir_fn.emit_typed(
                    ty,
                    InstData::FConvert {
                        value: val,
                        target: ScalarIrTy::F32,
                    },
                )
            } else {
                ir_fn.emit_typed(ty, InstData::CastToF32(val))
            }
        }
        ScalarIrTy::F16 => {
            let as_f32 = if from == ScalarIrTy::F32 {
                val
            } else {
                ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::CastToF32(val))
            };
            ir_fn.emit_typed(
                ty,
                InstData::FConvert {
                    value: as_f32,
                    target: ScalarIrTy::F16,
                },
            )
        }
        ScalarIrTy::I64 => ir_fn.emit_typed(ty, InstData::CastToI64(val)),
        ScalarIrTy::U64 => ir_fn.emit_typed(ty, InstData::CastToU64(val)),
        ScalarIrTy::Bool => coerce_to_bool(ir_fn, val),
    }
}

fn emit_zero_of_type(ir_fn: &mut IrFunction, ty: &IrType) -> InstId {
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
            let s = emit_zero_of_type(ir_fn, &IrType::Scalar(*scalar));
            let elems = vec![s; usize::from(*len)];
            emit_vec_construct(ir_fn, ty.clone(), elems)
        }
        _ => ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::I32), InstData::ConstI32(0)),
    }
}

fn emit_splat_f32(ir_fn: &mut IrFunction, ty: &IrType, value: f32) -> InstId {
    let f32_s = ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(value));
    let s = if ty.is_f16() {
        ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::F16),
            InstData::FConvert {
                value: f32_s,
                target: ScalarIrTy::F16,
            },
        )
    } else {
        f32_s
    };
    match ty {
        IrType::Vector { len, .. } => {
            let elems = vec![s; usize::from(*len)];
            emit_vec_construct(ir_fn, ty.clone(), elems)
        }
        _ => s,
    }
}

fn is_const_inst(ir_fn: &IrFunction, id: InstId) -> bool {
    matches!(
        ir_fn.insts[id].data,
        InstData::ConstBool(_)
            | InstData::ConstI32(_)
            | InstData::ConstU32(_)
            | InstData::ConstF32(_)
            | InstData::ConstF16(_)
            | InstData::ConstI64(_)
            | InstData::ConstU64(_)
    )
}

fn block_ends_with_terminator(stmts: &[StructuredStmt]) -> bool {
    matches!(
        stmts.last(),
        Some(StructuredStmt::Return(_) | StructuredStmt::Break | StructuredStmt::Continue)
    )
}

fn parse_swizzle_str(field: &str) -> Option<Vec<u8>> {
    if field.is_empty() || field.len() > 4 {
        return None;
    }
    let mut comps = Vec::with_capacity(field.len());
    for b in field.bytes() {
        let idx = match b {
            b'x' | b'r' => 0u8,
            b'y' | b'g' => 1u8,
            b'z' | b'b' => 2u8,
            b'w' | b'a' => 3u8,
            _ => return None,
        };
        comps.push(idx);
    }
    Some(comps)
}

fn extract_hir_name_arg(arg: Option<&HirCallArg>) -> Option<SmolStr> {
    match &arg?.value.kind {
        HirExprKind::StringLit(s) => Some(s.clone()),
        HirExprKind::Path { path, .. } => Some(path.clone()),
        _ => None,
    }
}

fn unwrap_step_by_hir(expr: &HirExpr) -> (&HirExpr, Option<&HirExpr>) {
    if let HirExprKind::MethodCall {
        receiver,
        method,
        args,
        ..
    } = &expr.kind
        && method == "step_by"
    {
        return (receiver.as_ref(), args.first().map(|a| &a.value));
    }
    (expr, None)
}

fn parse_atomic_turbofish(turbofish: &[TypeArg]) -> (AtomicScope, AtomicOrder) {
    let mut scope = AtomicScope::Device;
    let mut order = AtomicOrder::Relaxed;
    for arg in turbofish {
        if let TypeArg::Type(TypeExpr::Named { path, .. }) = arg {
            let joined = path.joined();
            match joined.as_str() {
                "Scope::Device" | "Device" => scope = AtomicScope::Device,
                "Scope::Workgroup" | "Workgroup" => scope = AtomicScope::Workgroup,
                "Scope::Subgroup" | "Subgroup" => scope = AtomicScope::Subgroup,
                "Order::Relaxed" | "Relaxed" => order = AtomicOrder::Relaxed,
                "Order::Acquire" | "Acquire" => order = AtomicOrder::Acquire,
                "Order::Release" | "Release" => order = AtomicOrder::Release,
                "Order::AcqRel" | "AcqRel" => order = AtomicOrder::AcqRel,
                _ => {}
            }
        }
    }
    (scope, order)
}

fn parse_u64_literal(raw: &str) -> Option<u64> {
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
        u64::from_str_radix(hex, 16).ok()
    } else if let Some(bin) = clean
        .strip_prefix("0b")
        .or_else(|| clean.strip_prefix("0B"))
    {
        u64::from_str_radix(bin, 2).ok()
    } else if let Some(oct) = clean
        .strip_prefix("0o")
        .or_else(|| clean.strip_prefix("0O"))
    {
        u64::from_str_radix(oct, 8).ok()
    } else {
        clean.parse::<u64>().ok()
    }
}

fn lower_int_literal(raw: &str, expr_ir_ty: &IrType, ir_fn: &mut IrFunction) -> InstId {
    let is_u64 = raw.ends_with("u64") || matches!(expr_ir_ty, IrType::Scalar(ScalarIrTy::U64));
    let is_i64 = raw.ends_with("i64") || matches!(expr_ir_ty, IrType::Scalar(ScalarIrTy::I64));
    let is_u32 = raw.ends_with("u32")
        || (raw.ends_with('u') && !raw.ends_with("u64"))
        || matches!(expr_ir_ty, IrType::Scalar(ScalarIrTy::U32));
    let parsed = parse_u64_literal(raw).unwrap_or(0);

    if is_u64 {
        ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::U64), InstData::ConstU64(parsed))
    } else if is_i64 {
        ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::I64),
            InstData::ConstI64(parsed.cast_signed()),
        )
    } else if is_u32 {
        ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::ConstU32(parsed as u32),
        )
    } else {
        ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::I32),
            InstData::ConstI32((parsed as u32).cast_signed()),
        )
    }
}

fn lower_float_literal(raw: &str, expr_ir_ty: &IrType, ir_fn: &mut IrFunction) -> InstId {
    let is_f16 = raw.ends_with("f16") || matches!(expr_ir_ty, IrType::Scalar(ScalarIrTy::F16));
    let val = raw
        .trim_end_matches("f32")
        .trim_end_matches("f16")
        .trim_end_matches('f')
        .replace('_', "")
        .parse::<f32>()
        .unwrap_or(0.0);
    if is_f16 {
        let f32_id = ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(val));
        ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::F16),
            InstData::FConvert {
                value: f32_id,
                target: ScalarIrTy::F16,
            },
        )
    } else {
        ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::F32), InstData::const_f32(val))
    }
}

fn parse_const_i64_hir(expr: &HirExpr) -> Option<i64> {
    match &expr.kind {
        HirExprKind::IntLit(raw) => parse_u64_literal(raw).map(u64::cast_signed),
        HirExprKind::Unary {
            op: UnaryOp::Neg,
            operand,
        } => parse_const_i64_hir(operand).map(i64::saturating_neg),
        _ => None,
    }
}

fn int_expr_scalar_hint(expr: &HirExpr, hir: &HirModule) -> Option<ScalarIrTy> {
    if let Some(s) = IrType::from_hir_type(&hir.types, expr.ty).scalar_ty()
        && matches!(
            s,
            ScalarIrTy::I32 | ScalarIrTy::U32 | ScalarIrTy::I64 | ScalarIrTy::U64
        )
    {
        return Some(s);
    }
    if let HirExprKind::IntLit(raw) = &expr.kind {
        if raw.ends_with("u64") {
            return Some(ScalarIrTy::U64);
        }
        if raw.ends_with("i64") {
            return Some(ScalarIrTy::I64);
        }
        if raw.ends_with("u32") || (raw.ends_with('u') && !raw.ends_with("u64")) {
            return Some(ScalarIrTy::U32);
        }
    }
    None
}

fn extract_const_range_bounds_hir(
    expr: &HirExpr,
    step: i64,
    bind_ir_ty: &IrType,
    hir: &HirModule,
    ir_fn: &mut IrFunction,
) -> (Option<u32>, Option<InstId>) {
    let HirExprKind::Binary {
        op: op @ (BinaryOp::Range | BinaryOp::RangeInclusive),
        lhs,
        rhs,
    } = &expr.kind
    else {
        return (None, None);
    };
    let (Some(start), Some(end)) = (parse_const_i64_hir(lhs), parse_const_i64_hir(rhs)) else {
        return (None, None);
    };
    let span_len = if matches!(op, BinaryOp::RangeInclusive) {
        end.saturating_sub(start).saturating_add(1).max(0)
    } else {
        end.saturating_sub(start).max(0)
    };
    let safe_step = step.max(1);
    let count = if span_len == 0 {
        0
    } else {
        (span_len - 1) / safe_step + 1
    };
    let trip = u32::try_from(count).ok();

    let hints = [
        bind_ir_ty.scalar_ty(),
        int_expr_scalar_hint(lhs, hir),
        int_expr_scalar_hint(rhs, hir),
    ];
    let is_unsigned = hints
        .iter()
        .any(|h| matches!(h, Some(ScalarIrTy::U32 | ScalarIrTy::U64)));
    let is_64 = hints
        .iter()
        .any(|h| matches!(h, Some(ScalarIrTy::I64 | ScalarIrTy::U64)));

    let init = match (is_64, is_unsigned) {
        (true, true) => ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U64),
            InstData::ConstU64(start.cast_unsigned()),
        ),
        (true, false) => {
            ir_fn.emit_typed(IrType::Scalar(ScalarIrTy::I64), InstData::ConstI64(start))
        }
        (false, true) => ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::ConstU32(start.cast_unsigned() as u32),
        ),
        (false, false) => ir_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::I32),
            InstData::ConstI32(start as i32),
        ),
    };
    (trip, Some(init))
}

#[cfg(test)]
mod tests {
    use vsl_mc::RenderTier;

    use crate::{CompileOptions, VslCompiler};

    #[test]
    fn lowers_engine_world_ubo_fields_and_biome_intrinsics() {
        let src = r"
            module pack::world_abi;

            import mc::{world};

            @stage(compute(8, 8, 1))
            pub fn probe(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let t: f32 = mc::world.game_time_sec;
                let legacy_time: f32 = mc::world.time;
                let rain_now: f32 = mc::world.rain;
                let legacy_rain: f32 = mc::world.rain_strength;
                let storm: f32 = mc::world.thunder;
                let wind_phase: f32 = mc::world.wind_phase;
                let fluid: i32 = mc::world.eye_in_fluid;
                let sun: vec3 = mc::world.sun_dir;
                let cam: vec3 = mc::world.camera_pos;
                let legacy_vp: mat4 = mc::world.view_proj;
                let legacy_inv_vp: mat4 = mc::world.inv_view_proj;
                let prev_vp: mat4 = mc::world.prev_view_proj;
                let slot0: vec4 = mc::world.pack_params;
                let telemetry: vec4 = mc::world.pack_params_10;
                let dim: i32 = mc::dimension_id();
                let clouds: f32 = mc::biome_cloud_scale();
                let valid: bool = mc::has_biome_telemetry();
                let validity: f32 = if valid { 1.0 } else { 0.0 };
                let value = t + legacy_time + rain_now + legacy_rain + storm + wind_phase + f32(fluid)
                    + sun.x + cam.x + legacy_vp[0u].x + legacy_inv_vp[0u].x + prev_vp[0u].x
                    + slot0.x + telemetry.z + f32(dim) + clouds + validity;
                out_img.store(ivec2(gid.xy), vec4(value, 0.0, 0.0, 1.0));
            }
        ";
        let mut compiler = VslCompiler::new();
        let opts = CompileOptions::for_tier(RenderTier::Tier2RayQuery);
        let output = compiler.compile_shader_source("world_abi.vsl", src, &opts);
        assert!(
            !output.has_errors(),
            "diagnostics: {:?}",
            output.diagnostics.as_slice()
        );
        assert_eq!(output.units.len(), 1, "expected one compiled entry point");
    }
}

#[cfg(test)]
mod image_param_inline_tests {
    use crate::{CompileOptions, VslCompiler};
    use std::process::Command;
    use vsl_mc::RenderTier;

    /// Runs `spirv-val --target-env vulkan1.3` when available, mirroring the workspace tests.
    fn validate_spirv(label: &str, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let path =
            std::env::temp_dir().join(format!("vsl_img_param_{label}_{}.spv", std::process::id()));
        std::fs::write(&path, &bytes).expect("write temporary SPIR-V");
        let output = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&path)
            .output();
        let _ = std::fs::remove_file(&path);
        match output {
            Ok(out) => assert!(
                out.status.success(),
                "spirv-val failed for `{label}`:\n{}",
                String::from_utf8_lossy(&out.stderr)
            ),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => panic!("failed to invoke spirv-val: {err}"),
        }
    }

    const IMAGE_PARAM_SRC: &str = r"
        module pack::image_param_inline;

        import mc::world;

        fn sample_depth(depth_tex: Texture2D<f32>, coord: ivec2) -> f32 {
            depth_tex.fetch(coord, 0).r
        }

        fn shade_depth(depth_tex: Texture2D<f32>, coord: ivec2) -> vec4<f32> {
            let d = sample_depth(depth_tex, coord);
            return vec4(d, d, d, 1.0);
        }

        fn write_color(dst_img: RWImage2D<rgba16f>, coord: ivec2, color: vec4<f32>) {
            dst_img.store(coord, color);
        }

        @stage(compute(8, 8, 1))
        pub fn main_cs(
            #[builtin(global_invocation_id)] gid: uvec3,
            gbuffer_depth: Texture2D<f32>,
            out_img: RWImage2D<rgba16f>,
        ) {
            let coord = ivec2(gid.xy);
            let color = shade_depth(gbuffer_depth, coord);
            write_color(out_img, coord, color);
        }
    ";

    #[test]
    fn helpers_taking_texture2d_and_rwimage2d_compile_to_valid_spirv() {
        let mut compiler = VslCompiler::new();
        let out = compiler.compile_shader_source(
            "image_param_inline.vsl",
            IMAGE_PARAM_SRC,
            &CompileOptions::for_tier(RenderTier::Tier2RayQuery),
        );
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        let unit = &out.units[0];
        assert_eq!(unit.entry_point, "main_cs");
        assert_eq!(unit.spirv[0], 0x0723_0203);

        // Descriptor substitution must not add extra bindings: world + depth + output only.
        assert_eq!(unit.descriptors.len(), 3);
        let names: Vec<_> = unit.descriptors.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"world"));
        assert!(names.contains(&"gbuffer_depth"));
        assert!(names.contains(&"out_img"));

        validate_spirv("image_param_inline", &unit.spirv);
    }

    #[test]
    fn cyclic_image_param_helper_keeps_original_lowering_behavior() {
        let src = r"
            module pack::img_param_cycle;

            import mc::world;

            fn recurse(tex: Texture2D<f32>, coord: ivec2, n: i32) -> f32 {
                if n > 0 {
                    return recurse(tex, coord, n - 1);
                }
                tex.fetch(coord, 0).r
            }

            @stage(compute(8, 8, 1))
            pub fn main_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                depth: Texture2D<f32>,
            ) {
                let v = recurse(depth, ivec2(gid.xy), 2);
            }
        ";
        let mut compiler = VslCompiler::new();
        let out = compiler.compile_shader_source(
            "img_param_cycle.vsl",
            src,
            &CompileOptions::for_tier(RenderTier::Tier2RayQuery),
        );
        // The cycle cannot be inlined; lowering must report diagnostics rather than emit a
        // call to a helper whose body was dropped.
        assert!(out.has_errors());
        assert!(out.units.is_empty());
    }
}
