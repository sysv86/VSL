use crate::ir::{PassBindingRef, PassRef, PipelineIr, PipelineMode, SchedulePlan};
use indexmap::map::Entry;
use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vsl_mc::{CANONICAL_PASSES, CanonicalPass};

/// Returns `true` when a pass binding key represents a write output target.
pub(crate) fn is_write_binding_key(key: &str) -> bool {
    key == "out" || key.starts_with("out_")
}

fn is_reachable(adj: &[Vec<usize>], start: usize, target: usize) -> bool {
    let mut visited = vec![false; adj.len()];
    let mut stack = vec![start];
    while let Some(u) = stack.pop() {
        if u == target {
            return true;
        }
        if visited[u] {
            continue;
        }
        visited[u] = true;
        for &next in &adj[u] {
            if !visited[next] {
                stack.push(next);
            }
        }
    }
    false
}

/// Validates pass declarations and computes a deterministic topological execution schedule.
///
/// Validation invariants:
/// - Every pass has a non-empty `entry_fn`.
/// - No duplicate pass names (including collisions with active canonical passes in `Extend` mode).
/// - No two custom passes `.replaces(...)` the same [`CanonicalPass`].
/// - Every `dispatch_target` exists in `ir.targets`.
/// - Every `CustomTarget` referenced in `bindings` exists in `ir.targets`, and `history: true`
///   bindings reference a target declared with `temporal == true`.
/// - Every `PassRef` in `after` / `before` resolves to an active canonical or declared custom pass.
/// - Write-to-read edges between custom passes sharing a `CustomTarget` (`history: false`) are
///   inferred automatically from `is_write` and `out`/`out_` naming.
/// - Replacing custom passes remain anchored to the replaced canonical pass's unreplaced
///   predecessor/successor whenever the corresponding `.after(...)` or `.before(...)` constraint
///   is omitted.
/// - The combined pass constraint graph is acyclic.
pub fn validate_and_schedule(ir: &PipelineIr) -> Result<SchedulePlan, Vec<String>> {
    let mut errors = Vec::new();

    // (a) Non-empty entry_fn, duplicate custom pass names, duplicate `.replaces(...)`,
    //     dispatch targets, and custom target binding checks.
    let mut seen_custom_names: IndexSet<&str> = IndexSet::new();
    let mut replaced_by: IndexMap<CanonicalPass, usize> = IndexMap::new();

    for (idx, pass) in ir.passes.iter().enumerate() {
        if pass.entry_fn.trim().is_empty() {
            errors.push(format!("pass `{}` has an empty `entry_fn`", pass.name));
        }

        if !seen_custom_names.insert(pass.name.as_str()) {
            errors.push(format!("duplicate pass name `{}`", pass.name));
        }

        if let Some(canonical) = pass.replaces {
            match replaced_by.entry(canonical) {
                Entry::Occupied(occ) => {
                    let first_idx = *occ.get();
                    errors.push(format!(
                        "canonical pass `{}` is replaced by multiple custom passes: `{}` and `{}`",
                        canonical.as_str(),
                        ir.passes[first_idx].name,
                        pass.name
                    ));
                }
                Entry::Vacant(vac) => {
                    vac.insert(idx);
                }
            }
        }

        if let Some(ref target) = pass.dispatch_target
            && !ir.targets.contains_key(target)
        {
            errors.push(format!(
                "pass `{}` references undeclared dispatch target `{target}`",
                pass.name
            ));
        }

        for (bind_key, binding) in &pass.bindings {
            if let PassBindingRef::CustomTarget { name, history, .. } = binding {
                if let Some(target_decl) = ir.targets.get(name) {
                    if *history && !target_decl.temporal {
                        errors.push(format!(
                            "pass `{}` binding `{bind_key}` references history of non-temporal target `{name}`",
                            pass.name
                        ));
                    }
                } else {
                    errors.push(format!(
                        "pass `{}` binding `{bind_key}` references undeclared target `{name}`",
                        pass.name
                    ));
                }
            }
        }
    }

    // In `Extend` mode, ensure custom pass names do not collide with unreplaced canonical passes.
    if ir.mode == PipelineMode::Extend {
        for &canonical in CANONICAL_PASSES {
            if !replaced_by.contains_key(&canonical)
                && seen_custom_names.contains(canonical.as_str())
            {
                errors.push(format!(
                    "custom pass name `{}` collides with active canonical pass `{}`",
                    canonical.as_str(),
                    canonical.as_str()
                ));
            }
        }
    }

    // Validate all `.after(...)` and `.before(...)` pass references before building edges.
    for pass in &ir.passes {
        for dep in pass.after.iter().chain(pass.before.iter()) {
            match dep {
                PassRef::Custom(custom_name) => {
                    if !seen_custom_names.contains(custom_name.as_str()) {
                        errors.push(format!(
                            "pass `{}` references undeclared custom pass `{custom_name}`",
                            pass.name
                        ));
                    }
                }
                PassRef::Canonical(cp) => {
                    if ir.mode == PipelineMode::Custom && !replaced_by.contains_key(cp) {
                        errors.push(format!(
                            "pass `{}` references canonical pass `{}` which is inactive in custom pipeline mode",
                            pass.name,
                            cp.as_str()
                        ));
                    }
                }
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Build deterministic node table.
    // In `Extend` mode, replacing custom passes occupy the canonical slot of the pass they replace
    // so deterministic tie-breaking places them naturally at the replaced stage.
    let mut node_names: Vec<SmolStr> = Vec::new();
    let mut canonical_to_node: IndexMap<CanonicalPass, usize> = IndexMap::new();
    let mut custom_name_to_node: IndexMap<SmolStr, usize> = IndexMap::new();
    let mut custom_to_node: Vec<usize> = vec![usize::MAX; ir.passes.len()];
    let mut active_canonical_nodes: Vec<usize> = Vec::new();

    if ir.mode == PipelineMode::Extend {
        for &canonical in CANONICAL_PASSES {
            let node_idx = node_names.len();
            if let Some(&custom_idx) = replaced_by.get(&canonical) {
                let name = ir.passes[custom_idx].name.clone();
                node_names.push(name.clone());
                custom_to_node[custom_idx] = node_idx;
                custom_name_to_node.insert(name, node_idx);
            } else {
                node_names.push(SmolStr::new_static(canonical.as_str()));
                active_canonical_nodes.push(node_idx);
            }
            canonical_to_node.insert(canonical, node_idx);
        }
    }

    for (custom_idx, pass) in ir.passes.iter().enumerate() {
        if custom_to_node[custom_idx] == usize::MAX {
            let node_idx = node_names.len();
            node_names.push(pass.name.clone());
            custom_to_node[custom_idx] = node_idx;
            custom_name_to_node.insert(pass.name.clone(), node_idx);
            if let Some(canonical) = pass.replaces {
                canonical_to_node.insert(canonical, node_idx);
            }
        }
    }

    let node_count = node_names.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    let mut in_degree: Vec<usize> = vec![0; node_count];
    let mut seen_edges: IndexSet<(usize, usize)> = IndexSet::new();

    let mut add_edge = |from: usize, to: usize| {
        if seen_edges.insert((from, to)) {
            adj[from].push(to);
            in_degree[to] += 1;
        }
    };

    // Preserve canonical execution order across all non-replaced canonical passes in `Extend` mode.
    if ir.mode == PipelineMode::Extend {
        for pair in active_canonical_nodes.windows(2) {
            add_edge(pair[0], pair[1]);
        }

        // Anchor each replacing custom pass to its nearest non-replaced canonical predecessor
        // (when `after` is omitted) and nearest non-replaced canonical successor (when `before`
        // is omitted) so partial constraints never let a replacement pass drift to the end.
        for (pos, &canonical) in CANONICAL_PASSES.iter().enumerate() {
            if let Some(&custom_idx) = replaced_by.get(&canonical) {
                let pass = &ir.passes[custom_idx];
                let node = custom_to_node[custom_idx];
                if pass.after.is_empty()
                    && let Some(&prev_cp) = CANONICAL_PASSES[..pos]
                        .iter()
                        .rev()
                        .find(|cp| !replaced_by.contains_key(*cp))
                    && let Some(&prev_node) = canonical_to_node.get(&prev_cp)
                {
                    add_edge(prev_node, node);
                }
                if pass.before.is_empty()
                    && let Some(&next_cp) = CANONICAL_PASSES[pos + 1..]
                        .iter()
                        .find(|cp| !replaced_by.contains_key(*cp))
                    && let Some(&next_node) = canonical_to_node.get(&next_cp)
                {
                    add_edge(node, next_node);
                }
            }
        }
    }

    // Apply explicit `after` and `before` constraints from custom passes.
    let resolve_pass_ref = |pref: &PassRef| -> Option<usize> {
        match pref {
            PassRef::Canonical(cp) => canonical_to_node.get(cp).copied(),
            PassRef::Custom(name) => custom_name_to_node.get(name).copied(),
        }
    };

    for (custom_idx, pass) in ir.passes.iter().enumerate() {
        let node = custom_to_node[custom_idx];
        for after_ref in &pass.after {
            if let Some(target_node) = resolve_pass_ref(after_ref) {
                add_edge(target_node, node);
            }
        }
        for before_ref in &pass.before {
            if let Some(target_node) = resolve_pass_ref(before_ref) {
                add_edge(node, target_node);
            }
        }
    }

    // Infer write -> read dependency edges between custom passes sharing `CustomTarget` bindings.
    let mut target_writers: IndexMap<&str, Vec<usize>> = IndexMap::new();
    let mut target_readers: IndexMap<&str, Vec<usize>> = IndexMap::new();
    for (custom_idx, pass) in ir.passes.iter().enumerate() {
        let node = custom_to_node[custom_idx];
        for (bind_key, binding) in &pass.bindings {
            if let PassBindingRef::CustomTarget {
                name,
                history: false,
                is_write,
            } = binding
            {
                if *is_write || is_write_binding_key(bind_key.as_str()) {
                    target_writers.entry(name.as_str()).or_default().push(node);
                } else {
                    target_readers.entry(name.as_str()).or_default().push(node);
                }
            }
        }
    }
    for (target_name, writers) in &target_writers {
        if let Some(readers) = target_readers.get(target_name) {
            for &writer_node in writers {
                for &reader_node in readers {
                    if writer_node != reader_node
                        && !is_reachable(&adj, reader_node, writer_node)
                        && seen_edges.insert((writer_node, reader_node))
                    {
                        adj[writer_node].push(reader_node);
                        in_degree[reader_node] += 1;
                    }
                }
            }
        }
    }

    // Deterministic Kahn's topological sort using a min-heap ordered by node index.
    let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    for (idx, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(Reverse(idx));
        }
    }

    let mut ordered_passes = Vec::with_capacity(node_count);
    while let Some(Reverse(u)) = ready.pop() {
        ordered_passes.push(node_names[u].clone());
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                ready.push(Reverse(v));
            }
        }
    }

    if ordered_passes.len() != node_count {
        let cycle_nodes: Vec<&str> = in_degree
            .iter()
            .enumerate()
            .filter(|&(_, &deg)| deg > 0)
            .map(|(idx, _)| node_names[idx].as_str())
            .collect();
        return Err(vec![format!(
            "dependency cycle detected in render-graph pass schedule involving: {}",
            cycle_nodes.join(", ")
        )]);
    }

    let replaced_canonical: Vec<CanonicalPass> = CANONICAL_PASSES
        .iter()
        .copied()
        .filter(|cp| replaced_by.contains_key(cp))
        .collect();

    Ok(SchedulePlan {
        ordered_passes,
        replaced_canonical,
    })
}
