//! Concrete free-function specializations for the AST-based GLSL emitter.

use super::*;

const MAX_SPECIALIZATIONS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct FreeGenericKey {
    name: SmolStr,
    types: Vec<String>,
}

impl FreeGenericKey {
    fn symbol(&self) -> String {
        // Length-prefix and escape each component to make specialization names injective.
        let mut symbol = String::from("vsl_mono");
        for part in std::iter::once(self.name.as_str()).chain(self.types.iter().map(String::as_str))
        {
            let _ = write!(symbol, "_{}_", part.len());
            for byte in part.bytes() {
                if byte.is_ascii_alphanumeric() {
                    symbol.push(char::from(byte));
                } else {
                    let _ = write!(symbol, "_{byte:02x}");
                }
            }
        }
        symbol
    }
}

#[derive(Clone, Default)]
pub(super) struct FreeGenerics {
    pub templates: IndexMap<SmolStr, FnDecl>,
    pending: IndexSet<FreeGenericKey>,
    current: Option<FreeGenericKey>,
    edges: IndexMap<FreeGenericKey, IndexSet<FreeGenericKey>>,
    pub trait_impls: IndexSet<(String, String)>,
}

impl FreeGenerics {
    pub(super) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

fn scalar_or_vector_or_matrix(ty: &str) -> bool {
    matches!(
        ty,
        "bool" | "float" | "double" | "int" | "uint" | "int64_t" | "uint64_t" | "float16_t"
    ) || [
        "vec", "ivec", "uvec", "bvec", "dvec", "f16vec", "i64vec", "u64vec",
    ]
    .iter()
    .any(|prefix| {
        ty.strip_prefix(prefix)
            .is_some_and(|n| matches!(n, "2" | "3" | "4"))
    }) || ["mat", "dmat"].iter().any(|prefix| {
        ty.strip_prefix(prefix).is_some_and(|dims| {
            dims.split_once('x')
                .map_or(matches!(dims, "2" | "3" | "4"), |(a, b)| {
                    matches!(a, "2" | "3" | "4") && matches!(b, "2" | "3" | "4")
                })
        })
    })
}

fn value_type_supported(ty: &str, ctx: &EmitCtx) -> bool {
    let (base, mut suffix) = ty
        .split_once('[')
        .map_or((ty, ""), |(base, _)| (base, &ty[base.len()..]));
    if !scalar_or_vector_or_matrix(base) && !ctx.struct_fields.contains_key(base) {
        return false;
    }
    while !suffix.is_empty() {
        let Some((dimension, rest)) = suffix.strip_prefix('[').and_then(|s| s.split_once(']'))
        else {
            return false;
        };
        if !dimension.parse::<u64>().is_ok_and(|n| n > 0) {
            return false;
        }
        suffix = rest;
    }
    true
}

fn full_type(ty: &TypeExpr, subst: &IndexMap<SmolStr, String>) -> String {
    let (base, suffix) = type_to_glsl_with_subst(ty, subst);
    format!("{base}{suffix}")
}

fn direct_type_parameter<'a>(ty: &'a TypeExpr, f: &FnDecl) -> Option<&'a SmolStr> {
    let TypeExpr::Named {
        path,
        args,
        unsized_array: false,
        ..
    } = ty
    else {
        return None;
    };
    let name = &path.segments.first()?.name;
    (path.segments.len() == 1 && args.is_empty() && f.generics.iter().any(|g| &g.name.name == name))
        .then_some(name)
}

fn bind_parameter(
    subst: &mut IndexMap<SmolStr, String>,
    name: &SmolStr,
    actual: &str,
) -> Result<(), String> {
    if let Some(previous) = subst.get(name) {
        if previous != actual {
            return Err(format!(
                "conflicting types for generic parameter {name}: {previous} and {actual}"
            ));
        }
    } else {
        subst.insert(name.clone(), actual.to_owned());
    }
    Ok(())
}

fn infer_parameter(
    pattern: &TypeExpr,
    actual: &str,
    f: &FnDecl,
    subst: &mut IndexMap<SmolStr, String>,
    ctx: &EmitCtx,
) -> Result<(), String> {
    if let Some(name) = direct_type_parameter(pattern, f) {
        bind_parameter(subst, name, actual)?;
    } else if let TypeExpr::Array { element, len, .. } = pattern {
        // Fixed-size array patterns can infer their element type without guessing dimensions.
        let Some((base, suffix)) = actual.split_once('[') else {
            return Err("expected fixed array argument".to_owned());
        };
        let Some((dimension, remaining)) = suffix.split_once(']') else {
            return Err("invalid array type".to_owned());
        };
        if dimension != emit_expr_const(len, &EmitCtx::new(EmitOptions::default())) {
            return Err("generic array length mismatch".to_owned());
        }
        infer_parameter(element, &format!("{base}{remaining}"), f, subst, ctx)?;
    } else if let TypeExpr::Named {
        path,
        args,
        unsized_array: false,
        ..
    } = pattern
        && !args.is_empty()
        && let Some(key) = ctx.generic_struct_keys.get(actual)
    {
        if key.struct_name != path.joined() || key.type_args.len() != args.len() {
            return Err("generic struct argument does not match parameter type".to_owned());
        }
        for (arg, concrete) in args.iter().zip(&key.type_args) {
            let TypeArg::Type(ty) = arg else {
                return Err("unsupported generic struct type argument".to_owned());
            };
            infer_parameter(ty, concrete, f, subst, ctx)?;
        }
    }
    Ok(())
}

fn explicit_type(ty: &TypeExpr, ctx: &EmitCtx) -> Result<String, String> {
    match ty {
        TypeExpr::Named {
            args,
            unsized_array,
            ..
        } => {
            if *unsized_array
                || args
                    .iter()
                    .any(|arg| matches!(arg, TypeArg::NamedExpr { .. }))
            {
                return Err("unsupported generic type argument".to_owned());
            }
            for arg in args {
                if let TypeArg::Type(inner) = arg {
                    let inner = explicit_type(inner, ctx)?;
                    if inner.contains('[') {
                        return Err(
                            "array-valued generic struct type arguments are unsupported".to_owned()
                        );
                    }
                }
            }
        }
        TypeExpr::Array { element, .. } => {
            explicit_type(element, ctx)?;
        }
    }
    let concrete = full_type(ty, &ctx.generic_subst);
    if value_type_supported(&concrete, ctx) {
        Ok(concrete)
    } else {
        Err(format!(
            "unsupported or unresolved generic type argument {concrete}"
        ))
    }
}

pub(super) fn resolve_generic_call(
    f: &FnDecl,
    type_args: &[TypeArg],
    args: &[CallArg],
    ctx: &EmitCtx,
) -> Result<(FreeGenericKey, String), String> {
    let name = &f.name.name;
    if args.len() != f.params.len() {
        return Err(format!(
            "generic function {name} expects {} arguments, found {}",
            f.params.len(),
            args.len()
        ));
    }
    if args
        .iter()
        .zip(&f.params)
        .any(|(arg, p)| arg.name.as_ref().is_some_and(|n| n.name != p.name.name))
    {
        // Reordering expressions could change side effects. Do not silently drop named argument order.
        return Err(format!(
            "generic function {name} requires arguments in declaration order"
        ));
    }
    let mut subst = IndexMap::new();
    if !type_args.is_empty() {
        if type_args.len() != f.generics.len() {
            return Err(format!(
                "generic function {name} expects {} type arguments, found {}",
                f.generics.len(),
                type_args.len()
            ));
        }
        for (g, arg) in f.generics.iter().zip(type_args) {
            let TypeArg::Type(ty) = arg else {
                return Err(format!("generic function {name} requires type arguments"));
            };
            subst.insert(g.name.name.clone(), explicit_type(ty, ctx)?);
        }
    }
    let mut actual_types = Vec::with_capacity(args.len());
    for (param, arg) in f.params.iter().zip(args) {
        let Some(pattern) = &param.ty else {
            return Err(format!("generic function {name} has an untyped parameter"));
        };
        let actual = known_expr_type(&arg.value, ctx);
        if type_args.is_empty()
            && let Some(ref actual) = actual
        {
            infer_parameter(pattern, actual, f, &mut subst, ctx)?;
        }
        actual_types.push(actual);
    }
    let mut types = Vec::with_capacity(f.generics.len());
    for g in &f.generics {
        let Some(ty) = subst.get(&g.name.name) else {
            return Err(format!(
                "cannot infer {} for generic function {name}; supply explicit type arguments",
                g.name.name
            ));
        };
        if !value_type_supported(ty, ctx) {
            return Err(format!(
                "generic function {name} does not support type {ty}"
            ));
        }
        if let Some(bound) = &g.bound
            && !ctx
                .free_generics
                .trait_impls
                .contains(&(bound.joined(), ty.clone()))
        {
            return Err(format!(
                "generic function {name}: {ty} does not have a concrete GLSL implementation of {}",
                bound.joined()
            ));
        }
        types.push(ty.clone());
    }
    for (param, actual) in f.params.iter().zip(actual_types) {
        let expected = full_type(param.ty.as_ref().expect("checked parameter"), &subst);
        if !value_type_supported(&expected, ctx) {
            return Err(format!(
                "generic function {name} has unsupported concrete parameter type {expected}"
            ));
        }
        match actual {
            Some(actual) if actual == expected => {}
            Some(actual) => {
                return Err(format!(
                    "generic function {name}: parameter {} expects {expected}, got {actual}",
                    param.name.name
                ));
            }
            None => {
                return Err(format!(
                    "cannot determine type of argument {} to generic function {name}",
                    param.name.name
                ));
            }
        }
    }
    let ret = fn_return_glsl_type_with_subst(f, &subst);
    if ret != "void" && !value_type_supported(&ret, ctx) {
        return Err(format!(
            "generic function {name} has unsupported concrete return type {ret}"
        ));
    }
    Ok((
        FreeGenericKey {
            name: name.clone(),
            types,
        },
        ret,
    ))
}

// This deliberately avoids infer_glsl_type's historical unknown-expression -> float fallback.
// Inferred generic substitutions must never be derived from that fallback.
pub(super) fn known_expr_type(expr: &Expr, ctx: &EmitCtx) -> Option<String> {
    match expr {
        Expr::IntLit(..) | Expr::FloatLit(..) | Expr::BoolLit(..) => {
            Some(infer_glsl_type(expr, ctx))
        }
        Expr::Path { path, .. } => {
            let name = path.joined().replace("::", "_");
            if ctx.uncertain_locals.contains(name.as_str()) {
                return None;
            }
            ctx.local_types.get(name.as_str()).cloned()
        }
        Expr::StructLit { .. } => {
            let ty = infer_glsl_type(expr, ctx);
            ctx.struct_fields.contains_key(ty.as_str()).then_some(ty)
        }
        Expr::ArrayLit { elements, .. } => {
            let ty = known_expr_type(elements.first()?, ctx)?;
            if !elements
                .iter()
                .all(|e| known_expr_type(e, ctx).as_deref() == Some(&ty))
            {
                return None;
            }
            let (base, rest) = ty
                .split_once('[')
                .map_or((ty.as_str(), ""), |(base, _)| (base, &ty[base.len()..]));
            Some(format!("{base}[{}]{rest}", elements.len()))
        }
        Expr::Field {
            receiver, field, ..
        } => {
            let recv = known_expr_type(receiver, ctx)?;
            if let Some(swizzle) = swizzle_result_type(&recv, &field.name) {
                return Some(swizzle);
            }
            let mut ty = ctx
                .struct_fields
                .get(recv.as_str())?
                .get(&field.name)?
                .clone();
            if let Some(dims) = ctx
                .struct_field_dims
                .get(recv.as_str())
                .and_then(|fields| fields.get(&field.name))
            {
                for dim in dims {
                    let _ = write!(ty, "[{dim}]");
                }
            }
            Some(ty)
        }
        Expr::Index { receiver, .. } => {
            let recv = known_expr_type(receiver, ctx)?;
            if recv.contains('[')
                || swizzle_result_type(&recv, "x").is_some()
                || recv.starts_with("mat")
            {
                Some(infer_glsl_type(expr, ctx))
            } else {
                None
            }
        }
        Expr::Unary { op, operand, .. } => {
            let ty = known_expr_type(operand, ctx)?;
            match op {
                UnaryOp::Not if ty == "bool" => Some(ty),
                UnaryOp::Neg | UnaryOp::BitNot => Some(ty),
                _ => None,
            }
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let l = known_expr_type(lhs, ctx)?;
            let r = known_expr_type(rhs, ctx)?;
            if l != r
                || is_assignment_op(*op)
                || matches!(op, BinaryOp::Range | BinaryOp::RangeInclusive)
            {
                return None;
            }
            Some(infer_glsl_type(expr, ctx))
        }
        Expr::Call { callee, args, .. } => {
            let Expr::Path {
                path, turbofish, ..
            } = callee.as_ref()
            else {
                return None;
            };
            let name = path.joined();
            if let Some(f) = ctx.free_generics.templates.get(name.as_str()) {
                return resolve_generic_call(f, turbofish, args, ctx)
                    .ok()
                    .map(|(_, ret)| ret);
            }
            let mapped = map_call_name(&name);
            if scalar_or_vector_or_matrix(&mapped)
                || ctx.struct_fields.contains_key(mapped.as_str())
            {
                return Some(mapped);
            }
            if matches!(name.as_str(), "f16" | "f64") {
                return Some(if name == "f16" { "float16_t" } else { "double" }.to_owned());
            }
            if let Some(ret) = ctx.fn_returns.get(mapped.as_str()) {
                return Some(ret.clone());
            }
            match name.as_str() {
                "abs" | "min" | "max" | "clamp" | "mix" | "normalize" | "reflect" | "cross" => {
                    known_expr_type(&args.first()?.value, ctx)
                }
                "dot" | "length" | "distance" => {
                    let ty = known_expr_type(&args.first()?.value, ctx)?;
                    swizzle_result_type(&ty, "x")
                        .or_else(|| matches!(ty.as_str(), "float" | "double").then_some(ty))
                }
                _ => None,
            }
        }
        Expr::MethodCall {
            receiver, method, ..
        } => {
            let recv = known_expr_type(receiver, ctx)?;
            if recv == "rayQueryEXT" {
                return ray_query_method_type(&method.name);
            }
            ctx.fn_returns
                .get(format!("{recv}_{}", method.name).as_str())
                .cloned()
        }
        Expr::If {
            then_branch,
            else_branch: Some(other),
            ..
        } if then_branch.stmts.is_empty() => {
            let a = known_expr_type(then_branch.tail_expr.as_deref()?, ctx)?;
            (known_expr_type(other, ctx)? == a).then_some(a)
        }
        Expr::Block(block) if block.stmts.is_empty() => {
            known_expr_type(block.tail_expr.as_deref()?, ctx)
        }
        _ => None,
    }
}

pub(super) fn emit_generic_call(
    name: &str,
    type_args: &[TypeArg],
    args: &[CallArg],
    indent: usize,
    ctx: &mut EmitCtx,
    pre: &mut String,
) -> Option<String> {
    let f = ctx.free_generics.templates.get(name)?;
    let key = match resolve_generic_call(f, type_args, args, ctx) {
        Ok((key, _)) => key,
        Err(error) => {
            ctx.errors.push(error);
            return Some("vsl_invalid_generic_call()".to_owned());
        }
    };
    let symbol = key.symbol();
    if ctx.fn_returns.contains_key(symbol.as_str())
        || ctx.struct_fields.contains_key(symbol.as_str())
        || ctx.free_generics.templates.contains_key(symbol.as_str())
        || ctx.local_types.contains_key(symbol.as_str())
        || ctx.entry_bindings.contains_key(symbol.as_str())
    {
        ctx.errors.push(format!(
            "generic GLSL specialization name collides with {symbol}"
        ));
        return Some("vsl_invalid_generic_call()".to_owned());
    }
    if !ctx.free_generics.pending.contains(&key)
        && ctx.free_generics.pending.len() >= MAX_SPECIALIZATIONS
    {
        ctx.errors.push(format!(
            "generic GLSL specialization limit ({MAX_SPECIALIZATIONS}) exceeded"
        ));
        return Some("vsl_invalid_generic_call()".to_owned());
    }
    if let Some(current) = ctx.free_generics.current.clone() {
        ctx.free_generics
            .edges
            .entry(current)
            .or_default()
            .insert(key.clone());
    }
    ctx.free_generics.pending.insert(key);
    let args = args
        .iter()
        .map(|arg| emit_expr_with_hoist(&arg.value, indent, ctx, pre))
        .collect::<Vec<_>>();
    Some(format!("{symbol}({})", args.join(", ")))
}

fn specialization_cycle(
    key: &FreeGenericKey,
    edges: &IndexMap<FreeGenericKey, IndexSet<FreeGenericKey>>,
    active: &mut IndexSet<FreeGenericKey>,
    done: &mut IndexSet<FreeGenericKey>,
) -> bool {
    if done.contains(key) {
        return false;
    }
    if !active.insert(key.clone()) {
        return true;
    }
    if edges.get(key).is_some_and(|targets| {
        targets
            .iter()
            .any(|target| specialization_cycle(target, edges, active, done))
    }) {
        return true;
    }
    active.shift_remove(key);
    done.insert(key.clone());
    false
}

// All concrete prototypes precede ordinary helpers, generic helpers and entry bodies.
// A growing worklist discovers nested instantiations as their concrete bodies are emitted.
pub(super) fn emit_free_specializations(ctx: &mut EmitCtx, scan: &FeatureScan) -> (String, String) {
    let mut prototypes = String::new();
    let mut bodies = String::new();
    let mut cursor = 0;
    while let Some(key) = ctx.free_generics.pending.get_index(cursor).cloned() {
        cursor += 1;
        let mut f = ctx.free_generics.templates[&key.name].clone();
        if f.body.is_none() {
            ctx.errors
                .push(format!("generic function {} has no body", key.name));
            continue;
        }
        let subst = f
            .generics
            .iter()
            .zip(&key.types)
            .map(|(g, ty)| (g.name.name.clone(), ty.clone()))
            .collect();
        let mut specialized_scan = FeatureScan::default();
        scan_fn_into_with_subst(&f, &subst, ctx.opts, &mut specialized_scan);
        if !specialized_scan.extensions.is_subset(&scan.extensions) {
            ctx.errors.push(format!(
                "generic function {} requires an unsupported specialized GLSL extension",
                key.name
            ));
            continue;
        }
        if !specialized_scan.bda_refs.is_subset(&scan.bda_refs) {
            ctx.errors.push(format!(
                "generic function {} requires an unsupported specialized BDA declaration",
                key.name
            ));
            continue;
        }
        let saved_subst = std::mem::replace(&mut ctx.generic_subst, subst);
        let symbol = key.symbol();
        f.name.name = SmolStr::new(&symbol);
        f.generics.clear();
        let ret = fn_return_glsl_type_with_subst(&f, &ctx.generic_subst);
        let params = format_param_list_with_self(&f.params, None, &ctx.generic_subst);
        let _ = writeln!(prototypes, "{ret} {symbol}({params});");
        ctx.free_generics.current = Some(key);
        emit_fn_decl_with_ctx(&f, None, ctx, &mut bodies);
        ctx.free_generics.current = None;
        ctx.generic_subst = saved_subst;
        bodies.push('\n');
    }
    let mut done = IndexSet::new();
    if ctx.free_generics.pending.iter().any(|key| {
        specialization_cycle(
            key,
            &ctx.free_generics.edges,
            &mut IndexSet::new(),
            &mut done,
        )
    }) {
        ctx.errors
            .push("recursive generic GLSL specialization is unsupported".to_owned());
    }
    (prototypes, bodies)
}
