//! Per-module export table and visibility-checked symbol import resolution for multi-file
//! `compile_project` builds.

use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, Span};
use vsl_hir::module::ModuleGraph;
use vsl_syntax::ast::{
    ConstDecl, FnDecl, ImplBlock, ImportDecl, ImportTree, Item, SourceUnit, StructDecl, TraitDecl,
    TypeExpr,
};

/// Exported `pub` symbols and associated `impl` blocks collected from a single project module.
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
}

impl ModuleExports {
    /// Collects all `pub` items (`Struct`, `Const`, non-entry `Function`, `Trait`) and `Impl`
    /// blocks from a parsed [`SourceUnit`].
    pub(crate) fn from_unit(unit: &SourceUnit) -> Self {
        let mut exports = Self::default();
        for item in &unit.items {
            match item {
                Item::Struct(s) => {
                    exports.all_declared.insert(s.name.name.clone());
                    if s.is_pub {
                        exports.structs.insert(s.name.name.clone(), s.clone());
                    }
                }
                Item::Const(c) => {
                    exports.all_declared.insert(c.name.name.clone());
                    if c.is_pub {
                        exports.consts.insert(c.name.name.clone(), c.clone());
                    }
                }
                Item::Trait(t) => {
                    exports.all_declared.insert(t.name.name.clone());
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
                    if f.is_pub && !is_entry_or_hook {
                        exports.functions.insert(f.name.name.clone(), f.clone());
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

/// Resolves all project imports for `unit` against `exports_by_module`, respecting
/// `ImportTree::Single`, `ImportTree::Group`, and `ImportTree::Glob`.
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
    let mut imported_sources: IndexMap<SmolStr, SmolStr> = IndexMap::new();
    let mut imported_items: Vec<Item> = Vec::new();
    let mut imported_types_or_traits: IndexSet<SmolStr> = IndexSet::new();
    let mut seen_impl_keys: IndexSet<(Option<SmolStr>, Option<SmolStr>, usize)> = IndexSet::new();

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
                        &local_decls,
                        &mut imported_sources,
                        &mut imported_items,
                        &mut imported_types_or_traits,
                        &mut seen_impl_keys,
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
                        &local_decls,
                        &mut imported_sources,
                        &mut imported_items,
                        &mut imported_types_or_traits,
                        &mut seen_impl_keys,
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
                        &local_decls,
                        &mut imported_sources,
                        &mut imported_items,
                        &mut imported_types_or_traits,
                        &mut seen_impl_keys,
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
                    &local_decls,
                    &mut imported_sources,
                    &mut imported_items,
                    &mut imported_types_or_traits,
                    &mut seen_impl_keys,
                    diagnostics,
                );
            }
        }
    }

    imported_items
}

#[allow(clippy::too_many_arguments)]
fn import_named_symbol(
    source_mod: &SmolStr,
    exports: &ModuleExports,
    sym_name: &SmolStr,
    span: Span,
    local_decls: &IndexMap<SmolStr, Span>,
    imported_sources: &mut IndexMap<SmolStr, SmolStr>,
    imported_items: &mut Vec<Item>,
    imported_types_or_traits: &mut IndexSet<SmolStr>,
    seen_impl_keys: &mut IndexSet<(Option<SmolStr>, Option<SmolStr>, usize)>,
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

    if local_decls.contains_key(sym_name) {
        diagnostics.push(Diagnostic::error(
            "E0503",
            format!(
                "imported symbol `{sym_name}` from `{source_mod}` collides with a local declaration"
            ),
            span,
        ));
        return;
    }

    if let Some(prev_mod) = imported_sources.get(sym_name) {
        diagnostics.push(Diagnostic::error(
            "E0503",
            format!(
                "name collision: symbol `{sym_name}` is imported from both `{prev_mod}` and `{source_mod}`"
            ),
            span,
        ));
        return;
    }

    imported_sources.insert(sym_name.clone(), source_mod.clone());
    if matches!(item, Item::Struct(_) | Item::Trait(_)) {
        imported_types_or_traits.insert(sym_name.clone());
    }
    imported_items.push(item);

    attach_matching_impls(
        exports,
        imported_types_or_traits,
        seen_impl_keys,
        imported_items,
    );
}

#[allow(clippy::too_many_arguments)]
fn import_all_pub_from_module(
    source_mod: &SmolStr,
    exports: &ModuleExports,
    span: Span,
    local_decls: &IndexMap<SmolStr, Span>,
    imported_sources: &mut IndexMap<SmolStr, SmolStr>,
    imported_items: &mut Vec<Item>,
    imported_types_or_traits: &mut IndexSet<SmolStr>,
    seen_impl_keys: &mut IndexSet<(Option<SmolStr>, Option<SmolStr>, usize)>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (sym_name, item) in exports.all_pub_items() {
        if local_decls.contains_key(&sym_name) {
            diagnostics.push(Diagnostic::error(
                "E0503",
                format!(
                    "imported symbol `{sym_name}` from `{source_mod}` collides with a local declaration"
                ),
                span,
            ));
            continue;
        }
        if let Some(prev_mod) = imported_sources.get(&sym_name) {
            diagnostics.push(Diagnostic::error(
                "E0503",
                format!(
                    "name collision: symbol `{sym_name}` is imported from both `{prev_mod}` and `{source_mod}`"
                ),
                span,
            ));
            continue;
        }
        imported_sources.insert(sym_name.clone(), source_mod.clone());
        if matches!(item, Item::Struct(_) | Item::Trait(_)) {
            imported_types_or_traits.insert(sym_name);
        }
        imported_items.push(item);
    }

    attach_matching_impls(
        exports,
        imported_types_or_traits,
        seen_impl_keys,
        imported_items,
    );
}

fn attach_matching_impls(
    exports: &ModuleExports,
    imported_types_or_traits: &IndexSet<SmolStr>,
    seen_impl_keys: &mut IndexSet<(Option<SmolStr>, Option<SmolStr>, usize)>,
    imported_items: &mut Vec<Item>,
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
            .is_some_and(|t| imported_types_or_traits.contains(t));
        let matches_trait = trait_name
            .as_ref()
            .is_some_and(|t| imported_types_or_traits.contains(t));
        if matches_target || matches_trait {
            let key = (trait_name, target_name, imp_block.methods.len());
            if seen_impl_keys.insert(key) {
                imported_items.push(Item::Impl(imp_block.clone()));
            }
        }
    }
}
