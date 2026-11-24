//! Deterministic BLAKE3 schema hashing and OS-independent L2 disk artifact cache
//! for `CompiledShaderUnit` objects.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use vsl_hir::PackSchema;
use vsl_hir::module::L1ModuleHash;
use vsl_mc::{CompiledShaderUnit, HardwareCaps, PackParamType, RenderTier};

/// Composite L1 cache key incorporating source/import hash, target tier, active hardware caps,
/// and `pack.vsl` schema hash so HIR caching is sound across configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct L1CacheKey {
    pub(crate) hash: L1ModuleHash,
    pub(crate) tier: RenderTier,
    pub(crate) active_caps: HardwareCaps,
    pub(crate) pack_schema_hash: [u8; 32],
}

const fn pack_param_type_tag(ty: PackParamType) -> u8 {
    match ty {
        PackParamType::Bool => 0,
        PackParamType::I32 => 1,
        PackParamType::U32 => 2,
        PackParamType::F32 => 3,
        PackParamType::Vec3 => 4,
        PackParamType::Enum => 5,
    }
}

/// Computes a deterministic BLAKE3 digest covering every field of an optional [`PackSchema`].
pub(crate) fn hash_pack_schema(schema: Option<&PackSchema>) -> [u8; 32] {
    let Some(schema) = schema else {
        return [0u8; 32];
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vsl2-pack-schema-v2");

    let mut params: Vec<_> = schema.params.iter().collect();
    params.sort_by_key(|(a, _)| *a);
    hasher.update(&(params.len() as u64).to_le_bytes());
    for (name, (off, sz)) in params {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&off.to_le_bytes());
        hasher.update(&sz.to_le_bytes());
    }

    let mut param_types: Vec<_> = schema.param_types.iter().collect();
    param_types.sort_by_key(|(a, _)| *a);
    hasher.update(&(param_types.len() as u64).to_le_bytes());
    for (name, ty) in param_types {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&[pack_param_type_tag(*ty)]);
    }

    let mut specs: Vec<_> = schema.spec_ids.iter().collect();
    specs.sort_by_key(|(a, _)| *a);
    hasher.update(&(specs.len() as u64).to_le_bytes());
    for (name, id) in specs {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&id.to_le_bytes());
    }

    let mut spec_defaults: Vec<_> = schema.spec_defaults.iter().collect();
    spec_defaults.sort_by_key(|(a, _)| *a);
    hasher.update(&(spec_defaults.len() as u64).to_le_bytes());
    for (name, def) in spec_defaults {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&def.spec_id.to_le_bytes());
        hasher.update(&[pack_param_type_tag(def.ty)]);
        hasher.update(&def.default_bits.to_le_bytes());
    }

    let mut slots: Vec<_> = schema.resource_slots.iter().collect();
    slots.sort_by_key(|(a, _)| *a);
    hasher.update(&(slots.len() as u64).to_le_bytes());
    for (name, slot) in slots {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&slot.to_le_bytes());
    }

    *hasher.finalize().as_bytes()
}

/// Encodes a 32-byte L2 artifact hash as lowercase hexadecimal.
pub(crate) fn l2_cache_hex(key: &[u8; 32]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in key {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Returns the deterministic disk cache path `<cache_dir>/<hex>.vsl-l2.json`.
pub(crate) fn l2_cache_file_path(dir: &Path, key: &[u8; 32]) -> PathBuf {
    dir.join(format!("{}.vsl-l2.json", l2_cache_hex(key)))
}

/// Attempts to load and validate a cached [`CompiledShaderUnit`] from disk.
pub(crate) fn load_l2_from_disk(dir: &Path, key: &[u8; 32]) -> Option<CompiledShaderUnit> {
    let path = l2_cache_file_path(dir, key);
    let bytes = fs::read(&path).ok()?;
    let unit: CompiledShaderUnit = serde_json::from_slice(&bytes).ok()?;
    (unit.cache_key == *key).then_some(unit)
}

/// Persists a [`CompiledShaderUnit`] to disk using an atomic write-and-rename sequence.
pub(crate) fn save_l2_to_disk(dir: &Path, key: &[u8; 32], unit: &CompiledShaderUnit) {
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let Ok(bytes) = serde_json::to_vec(unit) else {
        return;
    };
    let final_path = l2_cache_file_path(dir, key);
    let hex = l2_cache_hex(key);
    let tmp_path = dir.join(format!("{hex}.vsl-l2.json.tmp.{}", std::process::id()));
    if fs::write(&tmp_path, &bytes).is_err() {
        let _ = fs::remove_file(&tmp_path);
        return;
    }
    if fs::rename(&tmp_path, &final_path).is_err() {
        let _ = fs::remove_file(&final_path);
        if fs::rename(&tmp_path, &final_path).is_err() {
            let _ = fs::remove_file(&tmp_path);
        }
    }
}
