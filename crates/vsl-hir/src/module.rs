use indexmap::IndexMap;
use smol_str::SmolStr;
use thiserror::Error;
use vsl_mc::{HardwareCaps, RenderTier};

/// Dense index for a resolved module in [`ModuleGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub u32);

/// L1 cache key: BLAKE3 hash of module source text plus imported interface hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L1ModuleHash(pub [u8; 32]);

/// L2 cache key: BLAKE3 hash of `(entry_point_hir_hash, tier, caps, used_specs, driver_uuid)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L2ArtifactHash(pub [u8; 32]);

#[derive(Debug, Error)]
pub enum ModuleGraphError {
    #[error("cyclic module dependency detected involving `{0}`")]
    CyclicDependency(SmolStr),
    #[error("unknown imported module `{0}`")]
    UnknownModule(SmolStr),
}

#[derive(Debug, Clone)]
struct ModuleNode {
    id: ModuleId,
    name: SmolStr,
    source_hash: [u8; 32],
    imports: Vec<SmolStr>,
}

/// Content-hashed module dependency graph with virtual `std::*` / `mc::*` recognition.
#[derive(Debug, Default)]
pub struct ModuleGraph {
    modules: IndexMap<SmolStr, ModuleNode>,
}

impl ModuleGraph {
    pub fn new() -> Self {
        Self {
            modules: IndexMap::new(),
        }
    }

    /// Registers a module with its raw source text and imported module prefixes.
    pub fn register_module(
        &mut self,
        name: impl Into<SmolStr>,
        source: &str,
        imports: Vec<SmolStr>,
    ) -> ModuleId {
        let name = name.into();
        let id = ModuleId(self.modules.len() as u32);
        let source_hash = *blake3::hash(source.as_bytes()).as_bytes();
        self.modules.insert(
            name.clone(),
            ModuleNode {
                id,
                name,
                source_hash,
                imports,
            },
        );
        id
    }

    /// Returns `true` for built-in virtual namespaces (`std::*` and `mc::*`).
    pub fn is_virtual_module(path: &str) -> bool {
        path == "std" || path.starts_with("std::") || path == "mc" || path.starts_with("mc::")
    }

    /// Returns a topological compilation order (dependencies before dependents) or an error on cycles.
    pub fn topological_order(&self) -> Result<Vec<SmolStr>, ModuleGraphError> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum VisitState {
            Visiting,
            Visited,
        }

        fn dfs(
            name: &str,
            graph: &ModuleGraph,
            state: &mut IndexMap<SmolStr, VisitState>,
            out: &mut Vec<SmolStr>,
        ) -> Result<(), ModuleGraphError> {
            if ModuleGraph::is_virtual_module(name) {
                return Ok(());
            }
            if let Some(&st) = state.get(name) {
                return match st {
                    VisitState::Visiting => {
                        Err(ModuleGraphError::CyclicDependency(SmolStr::new(name)))
                    }
                    VisitState::Visited => Ok(()),
                };
            }
            let Some(node) = graph.modules.get(name) else {
                return Err(ModuleGraphError::UnknownModule(SmolStr::new(name)));
            };
            state.insert(node.name.clone(), VisitState::Visiting);
            for dep in &node.imports {
                dfs(dep.as_str(), graph, state, out)?;
            }
            state.insert(node.name.clone(), VisitState::Visited);
            out.push(node.name.clone());
            Ok(())
        }

        let mut state = IndexMap::new();
        let mut order = Vec::with_capacity(self.modules.len());
        for name in self.modules.keys() {
            dfs(name.as_str(), self, &mut state, &mut order)?;
        }
        Ok(order)
    }

    /// Computes the L1 HIR cache key for every module in topological order.
    pub fn compute_l1_hashes(&self) -> Result<IndexMap<SmolStr, L1ModuleHash>, ModuleGraphError> {
        let order = self.topological_order()?;
        let mut hashes: IndexMap<SmolStr, L1ModuleHash> = IndexMap::with_capacity(order.len());

        for name in order {
            let node = &self.modules[&name];
            let mut l1_hasher = blake3::Hasher::new();
            l1_hasher.update(b"vsl2-l1-hir-v1:");
            l1_hasher.update(node.name.as_bytes());
            l1_hasher.update(&node.source_hash);
            for dep in &node.imports {
                l1_hasher.update(dep.as_bytes());
                if let Some(dep_hash) = hashes.get(dep) {
                    l1_hasher.update(&dep_hash.0);
                }
            }
            hashes.insert(name, L1ModuleHash(*l1_hasher.finalize().as_bytes()));
        }

        Ok(hashes)
    }

    pub fn module_id(&self, name: &str) -> Option<ModuleId> {
        self.modules.get(name).map(|n| n.id)
    }
}

/// Computes the deterministic L2 SPIR-V artifact cache key per Spec §8.
///
/// Includes `pack_schema_hash` so changes to `pack.vsl` parameter BDA offsets or Set 3
/// resource slots always invalidate cached SPIR-V artifacts.
pub fn compute_l2_artifact_hash(
    entry_point_hir_hash: L1ModuleHash,
    entry_point_name: &str,
    active_tier: RenderTier,
    active_caps: HardwareCaps,
    used_spec_values: &[(u32, u32)],
    pack_schema_hash: &[u8; 32],
    driver_uuid: &[u8; 16],
) -> L2ArtifactHash {
    let mut l1_hasher = blake3::Hasher::new();
    l1_hasher.update(b"vsl2-l2-spv-v2:");
    l1_hasher.update(&entry_point_hir_hash.0);
    l1_hasher.update(entry_point_name.as_bytes());
    l1_hasher.update(&[active_tier.as_u8()]);
    l1_hasher.update(&active_caps.bits().to_le_bytes());
    for (spec_id, raw_val) in used_spec_values {
        l1_hasher.update(&spec_id.to_le_bytes());
        l1_hasher.update(&raw_val.to_le_bytes());
    }
    l1_hasher.update(pack_schema_hash);
    l1_hasher.update(driver_uuid);
    L2ArtifactHash(*l1_hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_module_cycles_and_computes_l1_hashes() {
        let mut g = ModuleGraph::new();
        g.register_module(
            "pack::a",
            "module pack::a;",
            vec![SmolStr::new_static("std::sync")],
        );
        g.register_module(
            "pack::b",
            "module pack::b;",
            vec![SmolStr::new_static("pack::a")],
        );
        let hashes = g.compute_l1_hashes().expect("acyclic");
        assert_eq!(hashes.len(), 2);

        let mut cyclic = ModuleGraph::new();
        cyclic.register_module("pack::x", "", vec![SmolStr::new_static("pack::y")]);
        cyclic.register_module("pack::y", "", vec![SmolStr::new_static("pack::x")]);
        assert!(matches!(
            cyclic.topological_order(),
            Err(ModuleGraphError::CyclicDependency(_))
        ));
    }
}
