use crate::ir::{
    BarrierKind, PassBarrier, PassBindingRef, PassDispatch, PassKind, PassNode, PassRef,
    PipelineIr, PipelineMode, SchedulePlan,
};
use indexmap::map::Entry;
use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vsl_mc::{
    BuiltinResourceId, CANONICAL_PASSES, CanonicalPass, CompiledShaderUnit,
    DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS, DescriptorAccess, DescriptorBinding,
    DescriptorKind, PACK_RESOURCE_SLOTS, TextureDim,
};

/// Highest Set 0 binding index reserved for per-pass targets (`bindings 1..=14`).
const MAX_PASS_TARGET_BINDING: u32 = 14;

/// Returns `true` when a pass binding key represents a write output target.
pub(crate) fn is_write_binding_key(key: &str) -> bool {
    key == "out" || key.starts_with("out_")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ResourceKey<'a> {
    Target(&'a str),
    Buffer(&'a str),
}

impl<'a> ResourceKey<'a> {
    fn name(self) -> &'a str {
        match self {
            Self::Target(name) | Self::Buffer(name) => name,
        }
    }
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

/// Resolves `PackMetadata.extends` inheritance chains by loading parent packs via `load_parent`
/// and merging child overrides deterministically on top of base packs.
///
/// Merge semantics (child overrides base):
/// - `metadata`: child's identity and tier range are preserved; `capabilities` are unioned
///   (base order then child additions); `extends` is cleared (`None`).
/// - `mode`: child's [`PipelineMode`] takes precedence.
/// - `settings`, `targets`, `buffers`: base entries are kept, with child entries overriding
///   matching keys or appending new ones.
/// - `presets`, `hooks`, `passes`: matched by preset name, [`vsl_mc::HookStage`], or pass name
///   (child overrides matching entries in place; new entries are appended).
/// - `resource_textures`: merged by resource name (or path when unnamed) with sequential Set 3
///   slot assignment (`0..PACK_RESOURCE_SLOTS`) and automatic remapping of child pass bindings.
///
/// Detects inheritance cycles (including self-inheritance), missing parent packs, and Set 3 slot
/// overflow.
pub fn resolve_pack_inheritance<F>(
    child: &PipelineIr,
    mut load_parent: F,
) -> Result<PipelineIr, Vec<String>>
where
    F: FnMut(&str) -> Result<PipelineIr, String>,
{
    if child.metadata.extends.is_none() {
        return Ok(child.clone());
    }

    let mut chain: Vec<PipelineIr> = vec![child.clone()];
    let mut visited: IndexSet<SmolStr> = IndexSet::new();
    if !child.metadata.id.is_empty() {
        visited.insert(child.metadata.id.clone());
    }

    while let Some(parent_id) = chain.last().and_then(|p| p.metadata.extends.clone()) {
        if parent_id.trim().is_empty() {
            return Err(vec!["pack `extends` cannot be empty".to_owned()]);
        }
        if !visited.insert(parent_id.clone()) {
            return Err(vec![format!(
                "pack inheritance cycle detected involving `{parent_id}`"
            )]);
        }
        let parent_ir = load_parent(parent_id.as_str())
            .map_err(|err| vec![format!("failed to load parent pack `{parent_id}`: {err}")])?;
        if !parent_ir.metadata.id.is_empty()
            && parent_ir.metadata.id != parent_id
            && !visited.insert(parent_ir.metadata.id.clone())
        {
            return Err(vec![format!(
                "pack inheritance cycle detected involving `{}`",
                parent_ir.metadata.id
            )]);
        }
        chain.push(parent_ir);
    }

    let mut iter = chain.into_iter().rev();
    let Some(mut merged) = iter.next() else {
        return Ok(child.clone());
    };
    for overlay in iter {
        merged = merge_pack_ir(merged, overlay);
    }
    merged.metadata.extends = None;

    if u32::try_from(merged.resource_textures.len()).unwrap_or(u32::MAX) > PACK_RESOURCE_SLOTS {
        return Err(vec![format!(
            "inherited pack texture assets ({}) exceed Set 3 limit of {PACK_RESOURCE_SLOTS} slots",
            merged.resource_textures.len()
        )]);
    }

    Ok(merged)
}

fn merge_pack_ir(mut base: PipelineIr, child: PipelineIr) -> PipelineIr {
    let mut merged_caps = base.metadata.capabilities;
    for cap in &child.metadata.capabilities {
        if !merged_caps.contains(cap) {
            merged_caps.push(cap.clone());
        }
    }
    base.metadata = child.metadata;
    base.metadata.capabilities = merged_caps;
    base.metadata.extends = None;
    base.mode = child.mode;

    for (key, setting) in child.settings {
        base.settings.insert(key, setting);
    }

    for preset in child.presets {
        if let Some(existing) = base.presets.iter_mut().find(|p| p.name == preset.name) {
            *existing = preset;
        } else {
            base.presets.push(preset);
        }
    }

    for hook in child.hooks {
        if let Some(existing) = base.hooks.iter_mut().find(|h| h.stage == hook.stage) {
            *existing = hook;
        } else {
            base.hooks.push(hook);
        }
    }

    let mut child_slot_remap: IndexMap<u32, u32> = IndexMap::new();
    for mut tex in child.resource_textures {
        let old_slot = tex.slot;
        let match_idx = base.resource_textures.iter().position(|existing| {
            if !existing.name.is_empty() && !tex.name.is_empty() {
                existing.name == tex.name
            } else {
                existing.path == tex.path
            }
        });
        if let Some(idx) = match_idx {
            let slot = u32::try_from(idx).unwrap_or(u32::MAX);
            tex.slot = slot;
            base.resource_textures[idx] = tex;
            child_slot_remap.insert(old_slot, slot);
        } else {
            let slot = u32::try_from(base.resource_textures.len()).unwrap_or(u32::MAX);
            tex.slot = slot;
            base.resource_textures.push(tex);
            child_slot_remap.insert(old_slot, slot);
        }
    }

    for (key, target) in child.targets {
        base.targets.insert(key, target);
    }

    for (key, buffer) in child.buffers {
        base.buffers.insert(key, buffer);
    }

    for mut pass in child.passes {
        for binding in pass.bindings.values_mut() {
            if let PassBindingRef::ResourceTexture { slot } = binding
                && let Some(&remapped) = child_slot_remap.get(slot)
            {
                *slot = remapped;
            }
        }
        if let Some(existing) = base.passes.iter_mut().find(|p| p.name == pass.name) {
            *existing = pass;
        } else {
            base.passes.push(pass);
        }
    }

    base
}

/// Validates pass declarations and computes a deterministic topological execution schedule
/// together with concrete RAW/WAW/WAR/IndirectDispatch synchronization barriers.
///
/// Validation invariants:
/// - Every non-`Transfer` pass has a non-empty `entry_fn` (`PassKind::Transfer` passes may leave
///   `entry_fn` empty for built-in transfer/blit/mip operations).
/// - No duplicate pass names (including collisions with active canonical passes in `Extend` mode).
/// - No two custom passes `.replaces(...)` the same [`CanonicalPass`].
/// - Every `dispatch_target` / `PassDispatch::ForTarget` target exists in `ir.targets`,
///   `PassDispatch::Explicit([x, y, z])` has all dimensions `> 0`, and `PassDispatch::Indirect`
///   has a non-empty buffer/target name (and resolves to `ir.buffers` when custom buffers are
///   declared).
/// - Every `CustomTarget` referenced in `bindings` exists in `ir.targets`, `history: true`
///   bindings reference a target declared with `temporal == true`, and every `CustomBuffer`
///   exists in `ir.buffers`.
/// - Every `PassRef` in `after` / `before` resolves to an active canonical or declared custom pass.
/// - Two writers to the same `CustomTarget` or `CustomBuffer` without a transitive ordering are
///   rejected as a write-after-write (WAW) hazard.
/// - Write-to-read edges between custom passes sharing a `CustomTarget` (`history: false`),
///   `CustomBuffer`, or `PassDispatch::Indirect` buffer are inferred automatically.
/// - Replacing custom passes remain anchored to their adjacent canonical predecessor/successor
///   slots whenever the corresponding `.after(...)` or `.before(...)` constraint is omitted.
/// - The combined pass constraint graph is acyclic.
pub fn validate_and_schedule(ir: &PipelineIr) -> Result<SchedulePlan, Vec<String>> {
    let mut errors = Vec::new();

    // (a) Non-empty entry_fn (except Transfer passes), duplicate custom pass names, duplicate
    //     `.replaces(...)`, dispatch configuration, and custom target/buffer binding checks.
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
                    } else if !ir.buffers.is_empty() && !ir.buffers.contains_key(buf) {
                        errors.push(format!(
                            "pass `{}` references undeclared indirect dispatch buffer `{buf}`",
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
            match binding {
                PassBindingRef::CustomTarget { name, history, .. } => {
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
                PassBindingRef::CustomBuffer { name, .. } => {
                    if !ir.buffers.contains_key(name) {
                        errors.push(format!(
                            "pass `{}` binding `{bind_key}` references undeclared buffer `{name}`",
                            pass.name
                        ));
                    }
                }
                PassBindingRef::BuiltinTarget { .. } | PassBindingRef::ResourceTexture { .. } => {}
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

        // Anchor each replacing custom pass to its immediately preceding canonical slot (when
        // `after` is omitted) and immediately succeeding canonical slot (when `before` is omitted),
        // preserving relative order even when adjacent canonical passes are both replaced.
        for (pos, &canonical) in CANONICAL_PASSES.iter().enumerate() {
            if let Some(&custom_idx) = replaced_by.get(&canonical) {
                let pass = &ir.passes[custom_idx];
                let node = custom_to_node[custom_idx];
                if pass.after.is_empty()
                    && pos > 0
                    && let Some(&prev_node) = canonical_to_node.get(&CANONICAL_PASSES[pos - 1])
                {
                    add_edge(prev_node, node);
                }
                if pass.before.is_empty()
                    && pos + 1 < CANONICAL_PASSES.len()
                    && let Some(&next_node) = canonical_to_node.get(&CANONICAL_PASSES[pos + 1])
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

    // Collect per-resource writers, readers, and indirect-dispatch readers for targets & buffers.
    let mut resource_writers: IndexMap<ResourceKey<'_>, Vec<usize>> = IndexMap::new();
    let mut resource_readers: IndexMap<ResourceKey<'_>, Vec<usize>> = IndexMap::new();
    let mut indirect_readers: IndexMap<ResourceKey<'_>, Vec<usize>> = IndexMap::new();

    for (custom_idx, pass) in ir.passes.iter().enumerate() {
        let node = custom_to_node[custom_idx];
        for (bind_key, binding) in &pass.bindings {
            match binding {
                PassBindingRef::CustomTarget {
                    name,
                    history: false,
                    is_write,
                } => {
                    let key = ResourceKey::Target(name.as_str());
                    if *is_write || is_write_binding_key(bind_key.as_str()) {
                        let list = resource_writers.entry(key).or_default();
                        if !list.contains(&node) {
                            list.push(node);
                        }
                    } else {
                        let list = resource_readers.entry(key).or_default();
                        if !list.contains(&node) {
                            list.push(node);
                        }
                    }
                }
                PassBindingRef::CustomBuffer { name, is_write } => {
                    let key = ResourceKey::Buffer(name.as_str());
                    if *is_write || is_write_binding_key(bind_key.as_str()) {
                        let list = resource_writers.entry(key).or_default();
                        if !list.contains(&node) {
                            list.push(node);
                        }
                    } else {
                        let list = resource_readers.entry(key).or_default();
                        if !list.contains(&node) {
                            list.push(node);
                        }
                    }
                }
                PassBindingRef::CustomTarget { history: true, .. }
                | PassBindingRef::BuiltinTarget { .. }
                | PassBindingRef::ResourceTexture { .. } => {}
            }
        }

        if let Some(PassDispatch::Indirect(buf_name)) = &pass.dispatch {
            let key = if ir.buffers.contains_key(buf_name) {
                ResourceKey::Buffer(buf_name.as_str())
            } else if ir.targets.contains_key(buf_name) {
                ResourceKey::Target(buf_name.as_str())
            } else {
                ResourceKey::Buffer(buf_name.as_str())
            };
            let list = indirect_readers.entry(key).or_default();
            if !list.contains(&node) {
                list.push(node);
            }
        }
    }

    // Detect ambiguous multi-writer WAW hazards before inferring RAW edges.
    for (&res_key, writers) in &resource_writers {
        if writers.len() > 1 {
            for (i, &w1) in writers.iter().enumerate() {
                for &w2 in &writers[i + 1..] {
                    if w1 != w2 && !is_reachable(&adj, w1, w2) && !is_reachable(&adj, w2, w1) {
                        errors.push(format!(
                            "write-after-write (WAW) hazard on resource `{}`: passes `{}` and `{}` both write `{}` without explicit ordering",
                            res_key.name(),
                            node_names[w1],
                            node_names[w2],
                            res_key.name()
                        ));
                    }
                }
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Infer write -> read (RAW) and write -> indirect-dispatch dependency edges.
    for (&res_key, writers) in &resource_writers {
        if let Some(readers) = resource_readers.get(&res_key) {
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
        if let Some(ind_readers) = indirect_readers.get(&res_key) {
            for &writer_node in writers {
                for &ind_node in ind_readers {
                    if writer_node != ind_node
                        && !is_reachable(&adj, ind_node, writer_node)
                        && seen_edges.insert((writer_node, ind_node))
                    {
                        adj[writer_node].push(ind_node);
                        in_degree[ind_node] += 1;
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

    let mut ordered_nodes = Vec::with_capacity(node_count);
    let mut ordered_passes = Vec::with_capacity(node_count);
    while let Some(Reverse(u)) = ready.pop() {
        ordered_nodes.push(u);
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

    // Build position lookup and emit concrete RAW, WAW, WAR, and IndirectDispatch barriers.
    let mut node_pos = vec![0usize; node_count];
    for (pos, &node_idx) in ordered_nodes.iter().enumerate() {
        node_pos[node_idx] = pos;
    }

    let mut barrier_set: IndexSet<PassBarrier> = IndexSet::new();
    for (&res_key, writers) in &resource_writers {
        let res_name = SmolStr::new(res_key.name());
        let mut sorted_writers = writers.clone();
        sorted_writers.sort_by_key(|&n| node_pos[n]);

        let mut sorted_readers = resource_readers.get(&res_key).cloned().unwrap_or_default();
        sorted_readers.sort_by_key(|&n| node_pos[n]);

        let mut sorted_indirect = indirect_readers.get(&res_key).cloned().unwrap_or_default();
        sorted_indirect.sort_by_key(|&n| node_pos[n]);

        // RAW: latest preceding writer -> each reader.
        for &r in &sorted_readers {
            if let Some(&w) = sorted_writers
                .iter()
                .rev()
                .find(|&&w| w != r && node_pos[w] < node_pos[r])
            {
                barrier_set.insert(PassBarrier {
                    src_pass: node_names[w].clone(),
                    dst_pass: node_names[r].clone(),
                    resource: res_name.clone(),
                    kind: BarrierKind::Raw,
                });
            }
        }

        // IndirectDispatch: latest preceding writer -> each indirect-dispatch consumer.
        for &ind in &sorted_indirect {
            if let Some(&w) = sorted_writers
                .iter()
                .rev()
                .find(|&&w| w != ind && node_pos[w] < node_pos[ind])
            {
                barrier_set.insert(PassBarrier {
                    src_pass: node_names[w].clone(),
                    dst_pass: node_names[ind].clone(),
                    resource: res_name.clone(),
                    kind: BarrierKind::IndirectDispatch,
                });
            }
        }

        // WAW between consecutive writers and WAR between preceding readers and a later writer.
        for (idx, &w) in sorted_writers.iter().enumerate() {
            let prev_w_pos = if idx > 0 {
                let prev_w = sorted_writers[idx - 1];
                barrier_set.insert(PassBarrier {
                    src_pass: node_names[prev_w].clone(),
                    dst_pass: node_names[w].clone(),
                    resource: res_name.clone(),
                    kind: BarrierKind::Waw,
                });
                Some(node_pos[prev_w])
            } else {
                None
            };

            for &r in &sorted_readers {
                if r != w && node_pos[r] < node_pos[w] && prev_w_pos.is_none_or(|p| node_pos[r] > p)
                {
                    barrier_set.insert(PassBarrier {
                        src_pass: node_names[r].clone(),
                        dst_pass: node_names[w].clone(),
                        resource: res_name.clone(),
                        kind: BarrierKind::War,
                    });
                }
            }
            for &ind in &sorted_indirect {
                if ind != w
                    && node_pos[ind] < node_pos[w]
                    && prev_w_pos.is_none_or(|p| node_pos[ind] > p)
                {
                    barrier_set.insert(PassBarrier {
                        src_pass: node_names[ind].clone(),
                        dst_pass: node_names[w].clone(),
                        resource: res_name.clone(),
                        kind: BarrierKind::War,
                    });
                }
            }
        }
    }

    let replaced_canonical: Vec<CanonicalPass> = CANONICAL_PASSES
        .iter()
        .copied()
        .filter(|cp| replaced_by.contains_key(cp))
        .collect();

    Ok(SchedulePlan {
        ordered_passes,
        replaced_canonical,
        barriers: barrier_set.into_iter().collect(),
    })
}

fn is_debug_channel_descriptor(desc: &DescriptorBinding, unit: &CompiledShaderUnit) -> bool {
    unit.debug_channels
        .iter()
        .any(|dc| dc.set == desc.set && dc.binding == desc.binding)
}

fn is_pass_local_descriptor(desc: &DescriptorBinding, unit: &CompiledShaderUnit) -> bool {
    if is_debug_channel_descriptor(desc, unit) {
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

fn is_readonly_builtin_resource(resource: BuiltinResourceId) -> bool {
    matches!(
        resource,
        BuiltinResourceId::ResTlas
            | BuiltinResourceId::ResShadowCsm
            | BuiltinResourceId::ResGbufferAlbedo
            | BuiltinResourceId::ResGbufferNormal
            | BuiltinResourceId::ResGbufferMaterial
            | BuiltinResourceId::ResGbufferVelocity
            | BuiltinResourceId::ResLinearDepth
            | BuiltinResourceId::ResMcMainDepth
            | BuiltinResourceId::ResPrevHudlessColor
            | BuiltinResourceId::ResSsgiHistory
            | BuiltinResourceId::ResSvgfHistoryRadiance
            | BuiltinResourceId::ResSvgfHistoryMoments
            | BuiltinResourceId::ResTsrHistory
            | BuiltinResourceId::ResEmissiveLights
            | BuiltinResourceId::ResVolumetricFogHistory
            | BuiltinResourceId::ResDhAlbedo
            | BuiltinResourceId::ResDhNormal
            | BuiltinResourceId::ResDhMaterial
            | BuiltinResourceId::ResDhDepth
    )
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
        if !is_debug_channel_descriptor(desc, unit)
            && desc.set == DESCRIPTOR_SET_PER_PASS
            && desc.binding > MAX_PASS_TARGET_BINDING
        {
            errors.push(format!(
                "pass `{}` shader `{}` declares Set {} binding {} (`{}`), exceeding the maximum per-pass binding {MAX_PASS_TARGET_BINDING}",
                pass.name, unit.entry_point, desc.set, desc.binding, desc.name
            ));
            continue;
        }
        if is_pass_local_descriptor(desc, unit)
            && let Some(prev) = pass_local_descs.insert(desc.name.as_str(), desc)
        {
            errors.push(format!(
                "pass `{}` shader `{}` declares duplicate pass-local descriptor name `{}` (set {} binding {} and set {} binding {})",
                pass.name,
                unit.entry_point,
                desc.name,
                prev.set,
                prev.binding,
                desc.set,
                desc.binding
            ));
        }
    }

    // Every shader-declared Set 0 pass target (`binding >= 1`) and Set 3 pack resource must be
    // satisfied by `pass.bindings`.
    for desc in pass_local_descs.values() {
        let is_required = (desc.set == DESCRIPTOR_SET_PER_PASS && desc.binding >= 1)
            || desc.set == DESCRIPTOR_SET_PACK_RESOURCES;
        if is_required && !pass.bindings.contains_key(desc.name.as_str()) {
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
                if !pipeline.resource_textures.iter().any(|r| r.slot == *slot) {
                    errors.push(format!(
                        "pass `{}` resource texture binding `{bind_key}` references undeclared Set {DESCRIPTOR_SET_PACK_RESOURCES} slot {slot}",
                        pass.name
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
                if is_readonly_builtin_resource(*resource)
                    && matches!(
                        desc.access,
                        DescriptorAccess::WriteOnly | DescriptorAccess::ReadWrite
                    )
                {
                    errors.push(format!(
                        "pass `{}` read-only builtin target `{bind_key}` ({resource:?}) cannot bind to `{:?}` descriptor in shader `{}`",
                        pass.name, desc.access, unit.entry_point
                    ));
                }
            }
            PassBindingRef::CustomBuffer { name, is_write } => {
                if desc.set != DESCRIPTOR_SET_PER_PASS {
                    errors.push(format!(
                        "pass `{}` custom buffer binding `{bind_key}` (buffer `{name}`) must target Set {DESCRIPTOR_SET_PER_PASS}, but shader `{}` declares set {}",
                        pass.name, unit.entry_point, desc.set
                    ));
                }
                if desc.kind != DescriptorKind::StorageBuffer {
                    errors.push(format!(
                        "pass `{}` custom buffer binding `{bind_key}` (buffer `{name}`) requires a `StorageBuffer` descriptor, but shader `{}` declares `{:?}`",
                        pass.name, unit.entry_point, desc.kind
                    ));
                }
                if *is_write && desc.access == DescriptorAccess::ReadOnly {
                    errors.push(format!(
                        "pass `{}` write buffer binding `{bind_key}` (buffer `{name}`) cannot bind to `ReadOnly` descriptor in shader `{}`",
                        pass.name, unit.entry_point
                    ));
                }
                if !*is_write && desc.access == DescriptorAccess::WriteOnly {
                    errors.push(format!(
                        "pass `{}` read buffer binding `{bind_key}` (buffer `{name}`) cannot bind to `WriteOnly` descriptor in shader `{}`",
                        pass.name, unit.entry_point
                    ));
                }
                if !pipeline.buffers.contains_key(name) {
                    errors.push(format!(
                        "pass `{}` binding `{bind_key}` references undeclared buffer `{name}`",
                        pass.name
                    ));
                }
            }
            PassBindingRef::CustomTarget {
                name,
                history,
                is_write,
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

                if *history
                    && (*is_write
                        || matches!(
                            desc.access,
                            DescriptorAccess::WriteOnly | DescriptorAccess::ReadWrite
                        ))
                {
                    errors.push(format!(
                        "pass `{}` history target binding `{bind_key}` (target `{name}`) is read-only and cannot bind to `{:?}` descriptor in shader `{}`",
                        pass.name, desc.access, unit.entry_point
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
