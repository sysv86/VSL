use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use vsl_mc::{BuiltinResourceId, CanonicalPass, HookStage, ImageFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PipelineMode {
    #[default]
    Extend,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackMetadata {
    pub id: SmolStr,
    pub name: SmolStr,
    pub version: SmolStr,
    pub author: SmolStr,
    pub description: SmolStr,
    pub extends: Option<SmolStr>,
    pub tier_min: u8,
    pub tier_max: u8,
    pub capabilities: Vec<SmolStr>,
}

impl Default for PackMetadata {
    fn default() -> Self {
        Self {
            id: SmolStr::default(),
            name: SmolStr::default(),
            version: SmolStr::default(),
            author: SmolStr::default(),
            description: SmolStr::default(),
            extends: None,
            tier_min: 0,
            tier_max: 3,
            capabilities: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum SettingValueIr {
    Bool(bool),
    Int(i64),
    Float(f32),
    ColorRgb([f32; 3]),
    EnumVariant(SmolStr),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SettingIr {
    pub name: SmolStr,
    pub category: SmolStr,
    pub is_spec: bool,
    pub label: SmolStr,
    #[serde(default)]
    pub description: SmolStr,
    pub default_value: SettingValueIr,
    pub range: Option<(f64, f64)>,
    pub step: Option<f64>,
    pub tiers: (u8, u8),
    pub enum_variants: Vec<SmolStr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetIr {
    pub name: SmolStr,
    pub label: SmolStr,
    #[serde(default)]
    pub description: SmolStr,
    pub order: i32,
    pub tiers: (u8, u8),
    pub overrides: IndexMap<SmolStr, SettingValueIr>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookRegistration {
    pub stage: HookStage,
    pub target_fn: SmolStr,
}

/// Set 3 pack texture asset (`bindings 0..=15`, `PACK_RESOURCE_SLOTS = 16`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTextureDecl {
    pub slot: u32,
    #[serde(default)]
    pub name: SmolStr,
    pub path: SmolStr,
    pub format: ImageFormat,
    pub filter: SmolStr,
    pub wrap: SmolStr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetDecl {
    pub name: SmolStr,
    pub format: ImageFormat,
    pub scale: f32,
    pub layers: u32,
    pub mip_levels: u32,
    pub temporal: bool,
    pub clear_color: Option<[f32; 4]>,
}

/// Pack-declared custom GPU storage buffer (`SSBO` / indirect dispatch arguments buffer).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferDecl {
    pub name: SmolStr,
    #[serde(default)]
    pub element_type: SmolStr,
    #[serde(default)]
    pub count: Option<u64>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub indirect: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassKind {
    Compute,
    Raster,
    Transfer,
}

/// Dispatch sizing mode for a compute or transfer render-graph pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassDispatch {
    ForTarget(SmolStr),
    Explicit([u32; 3]),
    Indirect(SmolStr),
}

/// Reference to either a canonical Hyaloclast render-graph pass or a pack-defined custom pass.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum PassRef {
    Canonical(CanonicalPass),
    Custom(SmolStr),
}

impl PassRef {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Canonical(cp) => cp.as_str(),
            Self::Custom(name) => name.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PassBindingRef {
    CustomTarget {
        name: SmolStr,
        history: bool,
        #[serde(default)]
        is_write: bool,
    },
    CustomBuffer {
        name: SmolStr,
        #[serde(default)]
        is_write: bool,
    },
    BuiltinTarget {
        resource: BuiltinResourceId,
    },
    ResourceTexture {
        slot: u32,
    },
}

/// Advisory pass metadata carried from v1 manifests. Tier gating is enforced by the engine:
/// `tiers` is forwarded to engine pipeline construction and is not checked by the deterministic
/// `pack.vsl` evaluator or the schedule validator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PassMeta {
    pub reads: Vec<SmolStr>,
    pub writes: Vec<SmolStr>,
    pub scale: Option<f32>,
    pub tiers: Option<(u8, u8)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PassNode {
    pub name: SmolStr,
    pub kind: PassKind,
    pub entry_fn: SmolStr,
    pub replaces: Option<CanonicalPass>,
    pub after: Vec<PassRef>,
    pub before: Vec<PassRef>,
    pub bindings: IndexMap<SmolStr, PassBindingRef>,
    #[serde(default)]
    pub meta: PassMeta,
    pub dispatch_target: Option<SmolStr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch: Option<PassDispatch>,
}

/// Synchronization hazard kind represented by a [`PassBarrier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarrierKind {
    Raw,
    Waw,
    War,
    IndirectDispatch,
}

/// Synchronization barrier between two scheduled passes sharing a target or buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PassBarrier {
    pub src_pass: SmolStr,
    pub dst_pass: SmolStr,
    pub resource: SmolStr,
    pub kind: BarrierKind,
}

/// Deterministic topological pass execution schedule produced from a [`PipelineIr`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SchedulePlan {
    pub ordered_passes: Vec<SmolStr>,
    pub replaced_canonical: Vec<CanonicalPass>,
    #[serde(default)]
    pub barriers: Vec<PassBarrier>,
}

/// Validated render-graph IR emitted by `pack.vsl` evaluation or `pack.toml` desugaring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PipelineIr {
    pub metadata: PackMetadata,
    pub mode: PipelineMode,
    pub settings: IndexMap<SmolStr, SettingIr>,
    pub presets: Vec<PresetIr>,
    pub hooks: Vec<HookRegistration>,
    pub resource_textures: Vec<ResourceTextureDecl>,
    pub targets: IndexMap<SmolStr, TargetDecl>,
    #[serde(default)]
    pub buffers: IndexMap<SmolStr, BufferDecl>,
    pub passes: Vec<PassNode>,
}
