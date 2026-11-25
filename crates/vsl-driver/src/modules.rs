//! Per-module export table and visibility-checked symbol import resolution for multi-file
//! `compile_project` builds.

use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use std::collections::VecDeque;
use vsl_diagnostics::{Diagnostic, Span};
use vsl_hir::module::ModuleGraph;
use vsl_syntax::ast::{
    Block, ConstDecl, Expr, FnDecl, ImplBlock, ImportDecl, ImportTree, Item, SourceUnit, Stmt,
    StructDecl, TraitDecl, TypeArg, TypeExpr,
};

/// Exported `pub` symbols, private/upstream dependency items, and associated `impl` blocks
/// collected from a single project module.
#[derive(Debug, Default, Clone)]
pub(crate) struct ModuleExports {
    pub(crate) structs: IndexMap<SmolStr, StructDecl>,
    pub(crate) consts: IndexMap<SmolStr, ConstDecl>,
    pub(crate) functions: IndexMap<SmolStr, FnDecl>,
    pub(crate) traits: IndexMap<SmolStr, TraitDecl>,
    pub(crate) impls: Vec<ImplBlock>,
    /// All top-level symbol names declared in this module (including non-`pub`) so explicit
    /// imports of private symbols produce a clear `E0502` visibility error.
    pub(crate) all_declared: IndexSet<SmolStr>,
    /// Complete symbol lookup table for transitive dependency closure when another module imports
    /// from this module: includes this module's own `pub` + private items as well as upstream
    /// items imported into this module.
    pub(crate) all_items: IndexMap<SmolStr, Item>,
}

impl ModuleExports {
    /// Collects all `pub` items (`Struct`, `Const`, non-entry `Function`, `Trait`), private
    /// dependency items, and `Impl` blocks from a parsed [`SourceUnit`].
    pub(crate) fn from_unit(unit: &SourceUnit) -> Self {
        let mut exports = Self::default();
        for item in &unit.items {
            match item {
                Item::Struct(s) => {
                    exports.all_declared.insert(s.name.name.clone());
                    exports.all_items.insert(s.name.name.clone(), item.clone());
                    if s.is_pub {
                        exports.structs.insert(s.name.name.clone(), s.clone());
                    }
                }
                Item::Const(c) => {
                    exports.all_declared.insert(c.name.name.clone());
                    exports.all_items.insert(c.name.name.clone(), item.clone());
                    if c.is_pub {
                        exports.consts.insert(c.name.name.clone(), c.clone());
                    }
                }
                Item::Trait(t) => {
                    exports.all_declared.insert(t.name.name.clone());
                    exports.all_items.insert(t.name.name.clone(), item.clone());
                    if t.is_pub {
                        exports.traits.insert(t.name.name.clone(), t.clone());
                    }
                }
                Item::Function(f) => {
                    exports.all_declared.insert(f.name.name.clone());
                    let is_entry_or_hook = f.attrs.iter().any(|a| {
                        a.entries
                            .iter()
                            .any(|e| e.name.name == "stage" || e.name.name == "hook")
                    });
                    if !is_entry_or_hook {
                        exports.all_items.insert(f.name.name.clone(), item.clone());
                        if f.is_pub {
                            exports.functions.insert(f.name.name.clone(), f.clone());
                        }
                    }
                }
                Item::Impl(imp) => {
                    exports.impls.push(imp.clone());
                }
                _ => {}
            }
        }
        exports
    }

    /// Records resolved upstream items (from modules this module imports) into the transitive
    /// dependency pool without exposing them as this module's direct `pub` exports.
    pub(crate) fn record_upstream_items(&mut self, upstream_items: &[Item]) {
        for item in upstream_items {
            match item {
                Item::Struct(s) => {
                    self.all_items
                        .entry(s.name.name.clone())
                        .or_insert_with(|| item.clone());
                }
                Item::Const(c) => {
                    self.all_items
                        .entry(c.name.name.clone())
                        .or_insert_with(|| item.clone());
                }
                Item::Trait(t) => {
                    self.all_items
                        .entry(t.name.name.clone())
                        .or_insert_with(|| item.clone());
                }
                Item::Function(f) => {
                    self.all_items
                        .entry(f.name.name.clone())
                        .or_insert_with(|| item.clone());
                }
                Item::Impl(imp) => {
                    self.impls.push(imp.clone());
                }
                _ => {}
            }
        }
    }

    fn get_pub_item(&self, name: &str) -> Option<Item> {
        if let Some(s) = self.structs.get(name) {
            return Some(Item::Struct(s.clone()));
        }
        if let Some(c) = self.consts.get(name) {
            return Some(Item::Const(c.clone()));
        }
        if let Some(t) = self.traits.get(name) {
            return Some(Item::Trait(t.clone()));
        }
        if let Some(f) = self.functions.get(name) {
            return Some(Item::Function(f.clone()));
        }
        None
    }

    fn all_pub_items(&self) -> Vec<(SmolStr, Item)> {
        let mut out = Vec::with_capacity(
            self.structs.len() + self.consts.len() + self.traits.len() + self.functions.len(),
        );
        for (name, s) in &self.structs {
            out.push((name.clone(), Item::Struct(s.clone())));
        }
        for (name, t) in &self.traits {
            out.push((name.clone(), Item::Trait(t.clone())));
        }
        for (name, c) in &self.consts {
            out.push((name.clone(), Item::Const(c.clone())));
        }
        for (name, f) in &self.functions {
            out.push((name.clone(), Item::Function(f.clone())));
        }
        out
    }
}

/// Resolves the target project module path for an [`ImportDecl`] when registering edges in
/// [`ModuleGraph`].
///
/// For `import pack::lib::helper;` (`ImportTree::Single`), if `pack::lib::helper` is not itself
/// a registered module name and `pack::lib` is, returns `pack::lib`.
pub(crate) fn module_graph_dependency_path(
    imp: &ImportDecl,
    known_modules: &IndexSet<SmolStr>,
) -> SmolStr {
    let full = SmolStr::new(imp.prefix.joined());
    if imp.tree == ImportTree::Single
        && !known_modules.contains(&full)
        && imp.prefix.segments.len() >= 2
    {
        let parent = imp.prefix.segments[..imp.prefix.segments.len() - 1]
            .iter()
            .map(|seg| seg.name.as_str())
            .collect::<Vec<_>>()
            .join("::");
        let parent_smol = SmolStr::new(parent);
        if known_modules.contains(&parent_smol) {
            return parent_smol;
        }
    }
    full
}

fn impl_target_name(target: &TypeExpr) -> Option<SmolStr> {
    match target {
        TypeExpr::Named { path, .. } => path.segments.last().map(|seg| seg.name.clone()),
        TypeExpr::Array { .. } => None,
    }
}

fn collect_local_declarations(unit: &SourceUnit) -> IndexMap<SmolStr, Span> {
    let mut locals = IndexMap::new();
    for item in &unit.items {
        match item {
            Item::Struct(s) => {
                locals.insert(s.name.name.clone(), s.span);
            }
            Item::Const(c) => {
                locals.insert(c.name.name.clone(), c.span);
            }
            Item::Trait(t) => {
                locals.insert(t.name.name.clone(), t.span);
            }
            Item::Function(f) => {
                locals.insert(f.name.name.clone(), f.span);
            }
            _ => {}
        }
    }
    locals
}

struct ImportResolutionState<'a> {
    local_decls: &'a IndexMap<SmolStr, Span>,
    imported_sources: IndexMap<SmolStr, SmolStr>,
    emitted_names: IndexSet<SmolStr>,
    imported_items: Vec<Item>,
    imported_types_or_traits: IndexSet<SmolStr>,
    seen_impl_keys: IndexSet<(Option<SmolStr>, Option<SmolStr>, usize)>,
}

/// Resolves all project imports for `unit` against `exports_by_module`, respecting
/// `ImportTree::Single`, `ImportTree::Group`, and `ImportTree::Glob`, and pulls in the full
/// transitive closure of private helpers, structs, constants, sibling `pub` items, `impl` blocks,
/// and upstream imported items referenced by each imported symbol.
///
/// Emits:
/// - `E0502` when an explicitly imported symbol does not exist or is not marked `pub`.
/// - `E0503` when an imported symbol collides with a local declaration or another import.
pub(crate) fn resolve_module_imports(
    unit: &SourceUnit,
    exports_by_module: &IndexMap<SmolStr, ModuleExports>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<Item> {
    let local_decls = collect_local_declarations(unit);
    let mut state = ImportResolutionState {
        local_decls: &local_decls,
        imported_sources: IndexMap::new(),
        emitted_names: IndexSet::new(),
        imported_items: Vec::new(),
        imported_types_or_traits: IndexSet::new(),
        seen_impl_keys: IndexSet::new(),
    };

    for imp in &unit.imports {
        let full_prefix = SmolStr::new(imp.prefix.joined());
        if ModuleGraph::is_virtual_module(full_prefix.as_str()) {
            continue;
        }

        match &imp.tree {
            ImportTree::Single => {
                if let Some(exports) = exports_by_module.get(&full_prefix) {
                    // Module-level `import pack::lib;` imports all `pub` items from that module.
                    import_all_pub_from_module(
                        &full_prefix,
                        exports,
                        imp.span,
                        &mut state,
                        diagnostics,
                    );
                } else if imp.prefix.segments.len() >= 2 {
                    let parent = SmolStr::new(
                        imp.prefix.segments[..imp.prefix.segments.len() - 1]
                            .iter()
                            .map(|seg| seg.name.as_str())
                            .collect::<Vec<_>>()
                            .join("::"),
                    );
                    if ModuleGraph::is_virtual_module(parent.as_str()) {
                        continue;
                    }
                    let sym_ident = &imp.prefix.segments[imp.prefix.segments.len() - 1];
                    let Some(exports) = exports_by_module.get(&parent) else {
                        diagnostics.push(Diagnostic::error(
                            "E0502",
                            format!(
                                "cannot import `{}`: module `{parent}` not found in project",
                                sym_ident.name
                            ),
                            sym_ident.span,
                        ));
                        continue;
                    };
                    import_named_symbol(
                        &parent,
                        exports,
                        &sym_ident.name,
                        sym_ident.span,
                        &mut state,
                        diagnostics,
                    );
                } else {
                    diagnostics.push(Diagnostic::error(
                        "E0502",
                        format!("cannot resolve import `{full_prefix}`: module not found"),
                        imp.span,
                    ));
                }
            }
            ImportTree::Group(idents) => {
                let Some(exports) = exports_by_module.get(&full_prefix) else {
                    diagnostics.push(Diagnostic::error(
                        "E0502",
                        format!("cannot import from `{full_prefix}`: module not found in project"),
                        imp.span,
                    ));
                    continue;
                };
                for ident in idents {
                    import_named_symbol(
                        &full_prefix,
                        exports,
                        &ident.name,
                        ident.span,
                        &mut state,
                        diagnostics,
                    );
                }
            }
            ImportTree::Glob => {
                let Some(exports) = exports_by_module.get(&full_prefix) else {
                    diagnostics.push(Diagnostic::error(
                        "E0502",
                        format!(
                            "cannot glob-import `{full_prefix}::*`: module not found in project"
                        ),
                        imp.span,
                    ));
                    continue;
                };
                import_all_pub_from_module(
                    &full_prefix,
                    exports,
                    imp.span,
                    &mut state,
                    diagnostics,
                );
            }
        }
    }

    state.imported_items
}

fn import_named_symbol(
    source_mod: &SmolStr,
    exports: &ModuleExports,
    sym_name: &SmolStr,
    span: Span,
    state: &mut ImportResolutionState<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(item) = exports.get_pub_item(sym_name.as_str()) else {
        let msg = if exports.all_declared.contains(sym_name) {
            format!("symbol `{sym_name}` in module `{source_mod}` is private (missing `pub`)")
        } else {
            format!("symbol `{sym_name}` not found in module `{source_mod}`")
        };
        diagnostics.push(Diagnostic::error("E0502", msg, span));
        return;
    };

    if state.local_decls.contains_key(sym_name) {
        diagnostics.push(Diagnostic::error(
            "E0503",
            format!(
                "imported symbol `{sym_name}` from `{source_mod}` collides with a local declaration"
            ),
            span,
        ));
        return;
    }

    if let Some(prev_mod) = state.imported_sources.get(sym_name) {
        if prev_mod != source_mod {
            diagnostics.push(Diagnostic::error(
                "E0503",
                format!(
                    "name collision: symbol `{sym_name}` is imported from both `{prev_mod}` and `{source_mod}`"
                ),
                span,
            ));
        }
        return;
    }

    state
        .imported_sources
        .insert(sym_name.clone(), source_mod.clone());
    if matches!(item, Item::Struct(_) | Item::Trait(_)) {
        state.imported_types_or_traits.insert(sym_name.clone());
    }

    let mut worklist = VecDeque::new();
    if state.emitted_names.insert(sym_name.clone()) {
        worklist.push_back(item.clone());
        state.imported_items.push(item);
    }

    expand_transitive_closure(exports, worklist, state);
}

fn import_all_pub_from_module(
    source_mod: &SmolStr,
    exports: &ModuleExports,
    span: Span,
    state: &mut ImportResolutionState<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut worklist = VecDeque::new();
    for (sym_name, item) in exports.all_pub_items() {
        if state.local_decls.contains_key(&sym_name) {
            diagnostics.push(Diagnostic::error(
                "E0503",
                format!(
                    "imported symbol `{sym_name}` from `{source_mod}` collides with a local declaration"
                ),
                span,
            ));
            continue;
        }
        if let Some(prev_mod) = state.imported_sources.get(&sym_name) {
            if prev_mod != source_mod {
                diagnostics.push(Diagnostic::error(
                    "E0503",
                    format!(
                        "name collision: symbol `{sym_name}` is imported from both `{prev_mod}` and `{source_mod}`"
                    ),
                    span,
                ));
            }
            continue;
        }
        state
            .imported_sources
            .insert(sym_name.clone(), source_mod.clone());
        if matches!(item, Item::Struct(_) | Item::Trait(_)) {
            state.imported_types_or_traits.insert(sym_name.clone());
        }
        if state.emitted_names.insert(sym_name) {
            worklist.push_back(item.clone());
            state.imported_items.push(item);
        }
    }

    expand_transitive_closure(exports, worklist, state);
}

fn expand_transitive_closure(
    exports: &ModuleExports,
    mut worklist: VecDeque<Item>,
    state: &mut ImportResolutionState<'_>,
) {
    attach_matching_impls(exports, &mut worklist, state);

    while let Some(item) = worklist.pop_front() {
        let mut refs = IndexSet::new();
        collect_referenced_idents_in_item(&item, &mut refs);

        let mut added_type_or_trait = false;
        for dep_name in refs {
            if state.local_decls.contains_key(&dep_name) || state.emitted_names.contains(&dep_name)
            {
                continue;
            }
            if let Some(dep_item) = exports.all_items.get(&dep_name).cloned() {
                state.emitted_names.insert(dep_name.clone());
                if matches!(dep_item, Item::Struct(_) | Item::Trait(_)) {
                    state.imported_types_or_traits.insert(dep_name);
                    added_type_or_trait = true;
                }
                worklist.push_back(dep_item.clone());
                state.imported_items.push(dep_item);
            }
        }

        if added_type_or_trait {
            attach_matching_impls(exports, &mut worklist, state);
        }
    }
}

fn attach_matching_impls(
    exports: &ModuleExports,
    worklist: &mut VecDeque<Item>,
    state: &mut ImportResolutionState<'_>,
) {
    for imp_block in &exports.impls {
        let target_name = impl_target_name(&imp_block.target);
        let trait_name = imp_block
            .trait_path
            .as_ref()
            .and_then(|p| p.segments.last())
            .map(|s| s.name.clone());
        let matches_target = target_name
            .as_ref()
            .is_some_and(|t| state.imported_types_or_traits.contains(t));
        let matches_trait = trait_name
            .as_ref()
            .is_some_and(|t| state.imported_types_or_traits.contains(t));
        if matches_target || matches_trait {
            let key = (trait_name, target_name, imp_block.methods.len());
            if state.seen_impl_keys.insert(key) {
                let item = Item::Impl(imp_block.clone());
                worklist.push_back(item.clone());
                state.imported_items.push(item);
            }
        }
    }
}

fn collect_referenced_idents_in_item(item: &Item, out: &mut IndexSet<SmolStr>) {
    match item {
        Item::Struct(s) => {
            for g in &s.generics {
                if let Some(ref b) = g.bound
                    && let Some(last) = b.segments.last()
                {
                    out.insert(last.name.clone());
                }
            }
            for f in &s.fields {
                collect_referenced_idents_in_type(&f.ty, out);
            }
        }
        Item::Const(c) => {
            collect_referenced_idents_in_type(&c.ty, out);
            collect_referenced_idents_in_expr(&c.value, out);
        }
        Item::Trait(t) => {
            for m in &t.methods {
                collect_referenced_idents_in_fn(m, out);
            }
        }
        Item::Impl(imp) => {
            for g in &imp.generics {
                if let Some(ref b) = g.bound
                    && let Some(last) = b.segments.last()
                {
                    out.insert(last.name.clone());
                }
            }
            if let Some(ref tp) = imp.trait_path
                && let Some(last) = tp.segments.last()
            {
                out.insert(last.name.clone());
            }
            collect_referenced_idents_in_type(&imp.target, out);
            for m in &imp.methods {
                collect_referenced_idents_in_fn(m, out);
            }
        }
        Item::Function(f) => collect_referenced_idents_in_fn(f, out),
        Item::WorkgroupVar(w) => collect_referenced_idents_in_type(&w.ty, out),
        _ => {}
    }
}

fn collect_referenced_idents_in_fn(f: &FnDecl, out: &mut IndexSet<SmolStr>) {
    for g in &f.generics {
        if let Some(ref b) = g.bound
            && let Some(last) = b.segments.last()
        {
            out.insert(last.name.clone());
        }
    }
    for p in &f.params {
        if let Some(ref ty) = p.ty {
            collect_referenced_idents_in_type(ty, out);
        }
    }
    if let Some(ref ret) = f.return_ty {
        collect_referenced_idents_in_type(ret, out);
    }
    if let Some(ref body) = f.body {
        collect_referenced_idents_in_block(body, out);
    }
}

fn collect_referenced_idents_in_type(ty: &TypeExpr, out: &mut IndexSet<SmolStr>) {
    match ty {
        TypeExpr::Named { path, args, .. } => {
            if let Some(last) = path.segments.last() {
                out.insert(last.name.clone());
            }
            for arg in args {
                match arg {
                    TypeArg::Type(inner) => collect_referenced_idents_in_type(inner, out),
                    TypeArg::NamedExpr { value, .. } => {
                        collect_referenced_idents_in_expr(value, out);
                    }
                }
            }
        }
        TypeExpr::Array { element, len, .. } => {
            collect_referenced_idents_in_type(element, out);
            collect_referenced_idents_in_expr(len, out);
        }
    }
}

fn collect_referenced_idents_in_block(block: &Block, out: &mut IndexSet<SmolStr>) {
    for stmt in &block.stmts {
        collect_referenced_idents_in_stmt(stmt, out);
    }
    if let Some(ref tail) = block.tail_expr {
        collect_referenced_idents_in_expr(tail, out);
    }
}

fn collect_referenced_idents_in_stmt(stmt: &Stmt, out: &mut IndexSet<SmolStr>) {
    match stmt {
        Stmt::Let { ty, init, .. } => {
            if let Some(t) = ty {
                collect_referenced_idents_in_type(t, out);
            }
            if let Some(e) = init {
                collect_referenced_idents_in_expr(e, out);
            }
        }
        Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
            collect_referenced_idents_in_expr(e, out);
        }
        Stmt::For { iter, body, .. } => {
            collect_referenced_idents_in_expr(iter, out);
            collect_referenced_idents_in_block(body, out);
        }
        Stmt::While { cond, body, .. } => {
            collect_referenced_idents_in_expr(cond, out);
            collect_referenced_idents_in_block(body, out);
        }
        Stmt::Loop { body, .. } => {
            collect_referenced_idents_in_block(body, out);
        }
        Stmt::Switch {
            selector,
            cases,
            default,
            ..
        } => {
            collect_referenced_idents_in_expr(selector, out);
            for c in cases {
                collect_referenced_idents_in_expr(&c.value, out);
                collect_referenced_idents_in_block(&c.body, out);
            }
            if let Some(d) = default {
                collect_referenced_idents_in_block(d, out);
            }
        }
        Stmt::Return { value: None, .. } | Stmt::Break(_) | Stmt::Continue(_) => {}
    }
}

fn collect_referenced_idents_in_expr(expr: &Expr, out: &mut IndexSet<SmolStr>) {
    match expr {
        Expr::Path {
            path, turbofish, ..
        } => {
            if let Some(last) = path.segments.last() {
                out.insert(last.name.clone());
            }
            for arg in turbofish {
                match arg {
                    TypeArg::Type(t) => collect_referenced_idents_in_type(t, out),
                    TypeArg::NamedExpr { value, .. } => {
                        collect_referenced_idents_in_expr(value, out);
                    }
                }
            }
        }
        Expr::StructLit { path, fields, .. } => {
            if let Some(last) = path.segments.last() {
                out.insert(last.name.clone());
            }
            for (_, fexpr) in fields {
                collect_referenced_idents_in_expr(fexpr, out);
            }
        }
        Expr::ArrayLit { elements, .. } => {
            for e in elements {
                collect_referenced_idents_in_expr(e, out);
            }
        }
        Expr::Unary { operand, .. }
        | Expr::Field {
            receiver: operand, ..
        } => collect_referenced_idents_in_expr(operand, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_referenced_idents_in_expr(lhs, out);
            collect_referenced_idents_in_expr(rhs, out);
        }
        Expr::Call { callee, args, .. } => {
            collect_referenced_idents_in_expr(callee, out);
            for a in args {
                collect_referenced_idents_in_expr(&a.value, out);
            }
        }
        Expr::MethodCall {
            receiver,
            turbofish,
            args,
            ..
        } => {
            collect_referenced_idents_in_expr(receiver, out);
            for arg in turbofish {
                match arg {
                    TypeArg::Type(t) => collect_referenced_idents_in_type(t, out),
                    TypeArg::NamedExpr { value, .. } => {
                        collect_referenced_idents_in_expr(value, out);
                    }
                }
            }
            for a in args {
                collect_referenced_idents_in_expr(&a.value, out);
            }
        }
        Expr::Index {
            receiver, index, ..
        } => {
            collect_referenced_idents_in_expr(receiver, out);
            collect_referenced_idents_in_expr(index, out);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_referenced_idents_in_expr(cond, out);
            collect_referenced_idents_in_block(then_branch, out);
            if let Some(eb) = else_branch {
                collect_referenced_idents_in_expr(eb, out);
            }
        }
        Expr::Block(b) => collect_referenced_idents_in_block(b, out),
        Expr::IntLit(_, _)
        | Expr::FloatLit(_, _)
        | Expr::BoolLit(_, _)
        | Expr::StringLit(_, _)
        | Expr::NoneLit(_) => {}
    }
}
