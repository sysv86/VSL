use crate::ir::{
    PassBindingRef, PassDispatch, PassKind, PassNode, PassRef, PipelineIr, PipelineMode,
    SchedulePlan,
};
use indexmap::map::Entry;
use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vsl_mc::{
    BuiltinResourceId, CANONICAL_PASSES, CanonicalPass, CompiledShaderUnit,
    DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS, DescriptorAccess, DescriptorBinding,
    DescriptorKind, TextureDim,
};

/// Highest Set 0 binding index reserved for per-pass targets (`bindings 1..=14`).
const MAX_PASS_TARGET_BINDING: u32 = 14;

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
/// - Every non-`Transfer` pass has a non-empty `entry_fn` (`PassKind::Transfer` passes may leave
///   `entry_fn` empty for built-in transfer/blit/mip operations).
/// - No duplicate pass names (including collisions with active canonical passes in `Extend` mode).
/// - No two custom passes `.replaces(...)` the same [`CanonicalPass`].
/// - Every `dispatch_target` / `PassDispatch::ForTarget` target exists in `ir.targets`,
///   `PassDispatch::Explicit([x, y, z])` has all dimensions `> 0`, and `PassDispatch::Indirect`
///   has a non-empty buffer/target name.
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

    // (a) Non-empty entry_fn (except Transfer passes), duplicate custom pass names, duplicate
    //     `.replaces(...)`, dispatch configuration, and custom target binding checks.
    let mut seen_custom_names: IndexSet<&str> = IndexSet::new();
    let mut replaced_by: IndexMap<CanonicalPass, usize> = IndexMap::new();

    for (idx, pass) in ir.passes.iter().enumerate() {
        if pass.kind != PassKind::Transfer && pass.entry_fn.trim().is_empty() {
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

        let mut checked_for_target: Option<&str> = None;
        if let Some(ref dispatch) = pass.dispatch {
            match dispatch {
                PassDispatch::ForTarget(target) => {
                    checked_for_target = Some(target.as_str());
                    if !ir.targets.contains_key(target) {
                        errors.push(format!(
                            "pass `{}` references undeclared dispatch target `{target}`",
                            pass.name
                        ));
                    }
                }
                PassDispatch::Explicit([x, y, z]) => {
                    if *x == 0 || *y == 0 || *z == 0 {
                        errors.push(format!(
                            "pass `{}` has invalid explicit dispatch dimensions [{x}, {y}, {z}]: all dimensions must be > 0",
                            pass.name
                        ));
                    }
                }
                PassDispatch::Indirect(buf) => {
                    if buf.trim().is_empty() {
                        errors.push(format!(
                            "pass `{}` has an empty indirect dispatch buffer/target name",
                            pass.name
                        ));
                    }
                }
            }
        }

        if let Some(ref target) = pass.dispatch_target
            && checked_for_target != Some(target.as_str())
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

fn is_pass_local_descriptor(desc: &DescriptorBinding, unit: &CompiledShaderUnit) -> bool {
    let is_debug_channel = unit
        .debug_channels
        .iter()
        .any(|dc| dc.set == desc.set && dc.binding == desc.binding);
    if is_debug_channel {
        return false;
    }
    (desc.set == DESCRIPTOR_SET_PER_PASS && (1..=MAX_PASS_TARGET_BINDING).contains(&desc.binding))
        || desc.set == DESCRIPTOR_SET_PACK_RESOURCES
}

fn builtin_resource_compatible_kind(resource: BuiltinResourceId, kind: DescriptorKind) -> bool {
    match resource {
        BuiltinResourceId::ResTlas => kind == DescriptorKind::AccelerationStructure,
        BuiltinResourceId::ResSharcHashTable
        | BuiltinResourceId::ResEmissiveLights
        | BuiltinResourceId::ResRestirReservoirs => kind == DescriptorKind::StorageBuffer,
        _ => matches!(
            kind,
            DescriptorKind::SampledImage
                | DescriptorKind::StorageImage
                | DescriptorKind::CombinedImageSampler
        ),
    }
}

/// Cross-validates a [`PassNode`]'s declared `bindings` against the pass-local descriptors
/// reflected in a compiled shader unit (`Set 0` bindings `1..=14` and `Set 3` pack resources).
///
/// Returns a list of human-readable diagnostic messages; an empty vector indicates full ABI
/// compatibility.
pub fn validate_pass_bindings_against_shader(
    pass: &PassNode,
    unit: &CompiledShaderUnit,
    pipeline: &PipelineIr,
) -> Vec<String> {
    let mut errors = Vec::new();

    let mut pass_local_descs: IndexMap<&str, &DescriptorBinding> = IndexMap::new();
    for desc in &unit.descriptors {
        if is_pass_local_descriptor(desc, unit) {
            pass_local_descs.insert(desc.name.as_str(), desc);
        }
    }

    // Every shader-declared Set 0 pass target (`binding >= 1`) must be satisfied by `pass.bindings`.
    for desc in pass_local_descs.values() {
        if desc.set == DESCRIPTOR_SET_PER_PASS
            && desc.binding >= 1
            && !pass.bindings.contains_key(desc.name.as_str())
        {
            errors.push(format!(
                "pass `{}` is missing binding `{}` required by shader `{}` (set {}, binding {})",
                pass.name, desc.name, unit.entry_point, desc.set, desc.binding
            ));
        }
    }

    // Validate every binding declared on the pass against the reflected descriptor table.
    for (bind_key, binding_ref) in &pass.bindings {
        let Some(&desc) = pass_local_descs.get(bind_key.as_str()) else {
            if let Some(other) = unit.descriptors.iter().find(|d| d.name == *bind_key) {
                errors.push(format!(
                    "pass `{}` binding `{bind_key}` targets engine-managed descriptor (set {}, binding {}) in shader `{}`",
                    pass.name, other.set, other.binding, unit.entry_point
                ));
            } else {
                errors.push(format!(
                    "pass `{}` declares unknown binding `{bind_key}` not present in shader `{}`",
                    pass.name, unit.entry_point
                ));
            }
            continue;
        };

        match binding_ref {
            PassBindingRef::ResourceTexture { slot } => {
                if desc.set != DESCRIPTOR_SET_PACK_RESOURCES {
                    errors.push(format!(
                        "pass `{}` resource texture binding `{bind_key}` (slot {slot}) must target Set {DESCRIPTOR_SET_PACK_RESOURCES}, but shader `{}` declares set {}",
                        pass.name, unit.entry_point, desc.set
                    ));
                }
                if !matches!(
                    desc.kind,
                    DescriptorKind::SampledImage | DescriptorKind::CombinedImageSampler
                ) {
                    errors.push(format!(
                        "pass `{}` resource texture binding `{bind_key}` (slot {slot}) requires a sampled image descriptor, but shader `{}` declares `{:?}`",
                        pass.name, unit.entry_point, desc.kind
                    ));
                }
                if *slot != desc.binding {
                    errors.push(format!(
                        "pass `{}` resource texture binding `{bind_key}` uses slot {slot}, but shader `{}` declares Set {} binding {}",
                        pass.name, unit.entry_point, desc.set, desc.binding
                    ));
                }
            }
            PassBindingRef::BuiltinTarget { resource } => {
                if desc.set != DESCRIPTOR_SET_PER_PASS {
                    errors.push(format!(
                        "pass `{}` builtin target binding `{bind_key}` ({resource:?}) must target Set {DESCRIPTOR_SET_PER_PASS}, but shader `{}` declares set {}",
                        pass.name, unit.entry_point, desc.set
                    ));
                }
                if !builtin_resource_compatible_kind(*resource, desc.kind) {
                    errors.push(format!(
                        "pass `{}` builtin target binding `{bind_key}` ({resource:?}) is incompatible with descriptor kind `{:?}` in shader `{}`",
                        pass.name, desc.kind, unit.entry_point
                    ));
                }
            }
            PassBindingRef::CustomTarget {
                name, is_write, ..
            } => {
                if desc.set != DESCRIPTOR_SET_PER_PASS {
                    errors.push(format!(
                        "pass `{}` custom target binding `{bind_key}` (target `{name}`) must target Set {DESCRIPTOR_SET_PER_PASS}, but shader `{}` declares set {}",
                        pass.name, unit.entry_point, desc.set
                    ));
                }
                if !matches!(
                    desc.kind,
                    DescriptorKind::SampledImage
                        | DescriptorKind::StorageImage
                        | DescriptorKind::CombinedImageSampler
                ) {
                    errors.push(format!(
                        "pass `{}` custom target binding `{bind_key}` (target `{name}`) requires an image descriptor, but shader `{}` declares `{:?}`",
                        pass.name, unit.entry_point, desc.kind
                    ));
                }

                if *is_write && desc.access == DescriptorAccess::ReadOnly {
                    errors.push(format!(
                        "pass `{}` write binding `{bind_key}` (target `{name}`) cannot bind to `ReadOnly` descriptor in shader `{}`",
                        pass.name, unit.entry_point
                    ));
                }
                if !*is_write && desc.access == DescriptorAccess::WriteOnly {
                    errors.push(format!(
                        "pass `{}` read binding `{bind_key}` (target `{name}`) cannot bind to `WriteOnly` descriptor in shader `{}`",
                        pass.name, unit.entry_point
                    ));
                }

                if let Some(target) = pipeline.targets.get(name) {
                    if desc.kind == DescriptorKind::StorageImage
                        && let Some(shader_fmt) = desc.image_format
                        && target.format != shader_fmt
                    {
                        errors.push(format!(
                            "pass `{}` binding `{bind_key}` target `{name}` has format `{}`, but shader `{}` storage image expects `{}`",
                            pass.name,
                            target.format.as_str(),
                            unit.entry_point,
                            shader_fmt.as_str()
                        ));
                    }

                    if let Some(dim) = desc.dim {
                        if target.layers > 1 {
                            if !matches!(dim, TextureDim::Dim2DArray | TextureDim::Dim3D) {
                                errors.push(format!(
                                    "pass `{}` binding `{bind_key}` multi-layer target `{name}` ({} layers) requires `Dim2DArray` or `Dim3D`, but shader `{}` declares `{dim:?}`",
                                    pass.name, target.layers, unit.entry_point
                                ));
                            }
                        } else if dim != TextureDim::Dim2D {
                            errors.push(format!(
                                "pass `{}` binding `{bind_key}` single-layer target `{name}` requires `Dim2D`, but shader `{}` declares `{dim:?}`",
                                pass.name, unit.entry_point
                            ));
                        }
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

    errors
}
