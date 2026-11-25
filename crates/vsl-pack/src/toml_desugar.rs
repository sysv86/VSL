use crate::ir::{
    BufferDecl, HookRegistration, PackMetadata, PassBindingRef, PassDispatch, PassKind, PassMeta,
    PassNode, PassRef, PipelineIr, PipelineMode, PresetIr, ResourceTextureDecl, SettingIr,
    SettingValueIr, TargetDecl,
};
use crate::schedule::is_write_binding_key;
use indexmap::IndexMap;
use serde::Deserialize;
use smol_str::SmolStr;
use thiserror::Error;
use vsl_mc::{BuiltinResourceId, CanonicalPass, HookStage, ImageFormat, PACK_RESOURCE_SLOTS};

#[derive(Debug, Error)]
pub enum TomlDesugarError {
    #[error("invalid pack.toml syntax: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("unknown hook stage `{0}` in pack.toml")]
    UnknownHook(String),
    #[error("unknown canonical pass `{0}` in pack.toml")]
    UnknownCanonicalPass(String),
    #[error("unknown image format `{0}` in pack.toml")]
    UnknownImageFormat(String),
    #[error("invalid setting value for `{0}` in pack.toml")]
    InvalidSettingValue(String),
    #[error("invalid range {min}..={max} for setting `{key}` in pack.toml: lower bound exceeds upper bound")]
    InvalidSettingRange { key: String, min: f64, max: f64 },
    #[error(
        "setting `{key}` default value {value} is outside declared range {min}..={max} in pack.toml"
    )]
    SettingValueOutOfRange {
        key: String,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error("preset `{preset}` references unknown setting key `{key}` in pack.toml")]
    UnknownPresetSetting { preset: String, key: String },
    #[error(
        "preset `{preset}` assigns out-of-range value {value} to setting `{key}` (expected {min}..={max}) in pack.toml"
    )]
    PresetValueOutOfRange {
        preset: String,
        key: String,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error("pack texture assets exceed Set 3 limit of {PACK_RESOURCE_SLOTS} slots in pack.toml")]
    TooManyResources,
    #[error("unknown resource texture reference `{0}` in pack.toml")]
    UnknownResource(String),
}

#[derive(Debug, Deserialize)]
struct LegacyPackToml {
    pack: LegacyPackHeader,
    #[serde(default)]
    pipeline: LegacyPipeline,
    #[serde(default)]
    settings: IndexMap<String, LegacySetting>,
    #[serde(default)]
    presets: IndexMap<String, LegacyPreset>,
    #[serde(default)]
    resources: LegacyResources,
    #[serde(default)]
    capabilities: Option<toml::Value>,
    #[serde(default)]
    hooks: IndexMap<String, String>,
    #[serde(default)]
    targets: IndexMap<String, LegacyTarget>,
    #[serde(default)]
    buffers: IndexMap<String, LegacyBuffer>,
    #[serde(default)]
    passes: Vec<LegacyPass>,
}

#[derive(Debug, Deserialize)]
struct LegacyPackHeader {
    id: String,
    name: String,
    #[serde(default = "default_ver")]
    version: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    extends: Option<String>,
    #[serde(default)]
    tiers: Option<[u8; 2]>,
    #[serde(default)]
    capabilities: Vec<String>,
}

fn default_ver() -> String {
    "1.0.0".to_owned()
}

#[derive(Debug, Default, Deserialize)]
struct LegacyPipeline {
    #[serde(default)]
    mode: PipelineMode,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LegacyResources {
    Seq(Vec<LegacyResource>),
    Map(IndexMap<String, LegacyResource>),
}

#[derive(Debug, Deserialize)]
struct LegacySetting {
    #[serde(default = "default_category")]
    category: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    spec: bool,
    #[serde(default)]
    label: Option<String>,
    default: toml::Value,
    #[serde(default)]
    range: Option<[toml::Value; 2]>,
    #[serde(default)]
    step: Option<toml::Value>,
    #[serde(default)]
    tiers: Option<[u8; 2]>,
    #[serde(default)]
    min: Option<toml::Value>,
    #[serde(default)]
    max: Option<toml::Value>,
    #[serde(default)]
    options: Vec<String>,
    #[serde(default)]
    runtime: Option<bool>,
    #[serde(default)]
    min_tier: Option<u8>,
    #[serde(default)]
    max_tier: Option<u8>,
    #[serde(default)]
    variants: Vec<String>,
}

fn default_category() -> String {
    "General".to_owned()
}

#[derive(Debug, Deserialize)]
struct LegacyPreset {
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    description: String,
    #[serde(default)]
    order: Option<i32>,
    #[serde(default)]
    tiers: Option<[u8; 2]>,
    #[serde(default, alias = "values")]
    overrides: IndexMap<String, toml::Value>,
}

#[derive(Debug, Deserialize)]
struct LegacyResource {
    #[serde(default)]
    name: Option<SmolStr>,
    path: String,
    #[serde(default = "default_resource_fmt")]
    format: String,
    #[serde(default = "default_filter")]
    filter: String,
    #[serde(default = "default_wrap")]
    wrap: String,
}

fn default_resource_fmt() -> String {
    "rgba8".to_owned()
}

fn default_filter() -> String {
    "linear".to_owned()
}

fn default_wrap() -> String {
    "clamp".to_owned()
}

#[derive(Debug, Deserialize)]
struct LegacyTarget {
    #[serde(default = "default_fmt")]
    format: String,
    #[serde(default = "default_one_f32")]
    scale: f32,
    #[serde(default = "default_one_u32", alias = "layers")]
    array_layers: u32,
    #[serde(default = "default_one_u32")]
    mip_levels: u32,
    #[serde(default)]
    temporal: bool,
    #[serde(default)]
    clear_color: Option<[f32; 4]>,
}

#[derive(Debug, Deserialize)]
struct LegacyBuffer {
    #[serde(default)]
    element_type: String,
    #[serde(default)]
    count: Option<u64>,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    indirect: bool,
}

fn default_fmt() -> String {
    "rgba16f".to_owned()
}

const fn default_one_f32() -> f32 {
    1.0
}

const fn default_one_u32() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct LegacyPass {
    #[serde(alias = "id")]
    name: String,
    #[serde(default = "default_compute")]
    kind: PassKind,
    #[serde(default)]
    shader: String,
    #[serde(default)]
    replaces: Option<String>,
    #[serde(default)]
    after: Vec<String>,
    #[serde(default)]
    before: Vec<String>,
    #[serde(default)]
    reads: Vec<SmolStr>,
    #[serde(default)]
    writes: Vec<SmolStr>,
    #[serde(default)]
    scale: Option<f32>,
    #[serde(default)]
    min_tier: Option<u8>,
    #[serde(default)]
    max_tier: Option<u8>,
    #[serde(default)]
    bindings: IndexMap<String, String>,
    #[serde(default, alias = "dispatch_target")]
    dispatch: Option<String>,
    #[serde(default)]
    dispatch_xyz: Option<[u32; 3]>,
    #[serde(default)]
    dispatch_indirect: Option<String>,
}

const fn default_compute() -> PassKind {
    PassKind::Compute
}

#[allow(clippy::cast_precision_loss)]
fn toml_value_to_setting(
    key: &str,
    value: &toml::Value,
) -> Result<SettingValueIr, TomlDesugarError> {
    match value {
        toml::Value::Boolean(b) => Ok(SettingValueIr::Bool(*b)),
        toml::Value::Integer(i) => Ok(SettingValueIr::Int(*i)),
        toml::Value::Float(f) => Ok(SettingValueIr::Float(*f as f32)),
        toml::Value::String(s) => Ok(SettingValueIr::EnumVariant(SmolStr::new(s))),
        toml::Value::Array(arr) if arr.len() == 3 => {
            let mut rgb = [0.0f32; 3];
            for (slot, item) in rgb.iter_mut().zip(arr.iter()) {
                match item {
                    toml::Value::Float(f) => *slot = *f as f32,
                    toml::Value::Integer(i) => *slot = *i as f32,
                    _ => return Err(TomlDesugarError::InvalidSettingValue(key.to_owned())),
                }
            }
            Ok(SettingValueIr::ColorRgb(rgb))
        }
        _ => Err(TomlDesugarError::InvalidSettingValue(key.to_owned())),
    }
}

#[allow(clippy::cast_precision_loss)]
fn toml_numeric_f64(key: &str, value: &toml::Value) -> Result<(f64, bool), TomlDesugarError> {
    match value {
        toml::Value::Float(f) => Ok((*f, true)),
        toml::Value::Integer(i) => Ok((*i as f64, false)),
        _ => Err(TomlDesugarError::InvalidSettingValue(key.to_owned())),
    }
}

#[allow(clippy::cast_precision_loss)]
fn setting_numeric_f64(val: &SettingValueIr) -> Option<f64> {
    match val {
        SettingValueIr::Int(i) => Some(*i as f64),
        SettingValueIr::Float(f) => Some(f64::from(*f)),
        _ => None,
    }
}

fn parse_image_format(name: &str) -> Result<ImageFormat, TomlDesugarError> {
    ImageFormat::from_name(name)
        .or_else(|| match name.trim().to_ascii_lowercase().as_str() {
            "rgba8_unorm" => Some(ImageFormat::Rgba8),
            "r8_unorm" => Some(ImageFormat::R8),
            "rg8_unorm" => Some(ImageFormat::Rg8),
            "rgba16f" => Some(ImageFormat::Rgba16f),
            "r16f" => Some(ImageFormat::R16f),
            "rg16f" => Some(ImageFormat::Rg16f),
            "r32f" => Some(ImageFormat::R32f),
            "rgba32f" => Some(ImageFormat::Rgba32f),
            "r32uint" => Some(ImageFormat::R32ui),
            "r32sint" => Some(ImageFormat::R32i),
            _ => None,
        })
        .ok_or_else(|| TomlDesugarError::UnknownImageFormat(name.to_owned()))
}

fn parse_pass_ref(name: &str) -> PassRef {
    CanonicalPass::from_name(name)
        .map_or_else(|| PassRef::Custom(SmolStr::new(name)), PassRef::Canonical)
}

fn parse_binding_ref(
    bind_key: &str,
    raw: &str,
    resources: &[ResourceTextureDecl],
    buffers: &IndexMap<SmolStr, BufferDecl>,
) -> Result<PassBindingRef, TomlDesugarError> {
    if let Some(rest) = raw.strip_prefix("resource:") {
        if let Ok(slot) = rest.parse::<u32>() {
            if resources.iter().any(|r| r.slot == slot) {
                return Ok(PassBindingRef::ResourceTexture { slot });
            }
            return Err(TomlDesugarError::UnknownResource(raw.to_owned()));
        }
        if let Some(found) = resources
            .iter()
            .find(|r| r.name.as_str() == rest || r.path == rest)
        {
            return Ok(PassBindingRef::ResourceTexture { slot: found.slot });
        }
        return Err(TomlDesugarError::UnknownResource(raw.to_owned()));
    }

    if let Some(buf_name) = raw.strip_prefix("buffer:") {
        return Ok(PassBindingRef::CustomBuffer {
            name: SmolStr::new(buf_name),
            is_write: is_write_binding_key(bind_key),
        });
    }

    let builtin_candidate = raw.strip_prefix("mc::targets::").unwrap_or(raw);
    if let Some(resource) = BuiltinResourceId::from_name(builtin_candidate) {
        return Ok(PassBindingRef::BuiltinTarget { resource });
    }

    if let Some(target_name) = raw.strip_suffix(".history") {
        return Ok(PassBindingRef::CustomTarget {
            name: SmolStr::new(target_name),
            history: true,
            is_write: false,
        });
    }

    if buffers.contains_key(raw) {
        return Ok(PassBindingRef::CustomBuffer {
            name: SmolStr::new(raw),
            is_write: is_write_binding_key(bind_key),
        });
    }

    Ok(PassBindingRef::CustomTarget {
        name: SmolStr::new(raw),
        history: false,
        is_write: is_write_binding_key(bind_key),
    })
}

impl Default for LegacyResources {
    fn default() -> Self {
        Self::Seq(Vec::new())
    }
}

/// Desugars a legacy `pack.toml` v2 manifest into VSL 2.0's unified [`PipelineIr`].
#[allow(clippy::cast_precision_loss)]
pub fn desugar_pack_toml(source: &str) -> Result<PipelineIr, TomlDesugarError> {
    let raw: LegacyPackToml = toml::from_str(source)?;
    let pack_tiers_absent = raw.pack.tiers.is_none();
    let (mut tier_min, mut tier_max) = raw.pack.tiers.map_or((0, 3), |[lo, hi]| (lo, hi));
    let mut capabilities: Vec<SmolStr> = raw
        .pack
        .capabilities
        .into_iter()
        .map(SmolStr::new)
        .collect();
    if let Some(toml::Value::Table(table)) = &raw.capabilities {
        for (key, value) in table {
            if key == "tiers" {
                if let toml::Value::Array(tiers) = value {
                    let lo = tiers.first().and_then(toml::Value::as_integer);
                    let hi = tiers.last().and_then(toml::Value::as_integer);
                    if let (Some(lo), Some(hi)) = (lo, hi)
                        && pack_tiers_absent
                    {
                        tier_min = u8::try_from(lo).unwrap_or(0);
                        tier_max = u8::try_from(hi).unwrap_or(3);
                    }
                }
            } else if matches!(value, toml::Value::Boolean(true)) {
                capabilities.push(SmolStr::new(key));
            }
        }
    }
    let mut ir = PipelineIr {
        metadata: PackMetadata {
            id: SmolStr::new(raw.pack.id),
            name: SmolStr::new(raw.pack.name),
            version: SmolStr::new(raw.pack.version),
            author: SmolStr::new(raw.pack.author),
            description: SmolStr::new(raw.pack.description),
            extends: raw.pack.extends.map(SmolStr::new),
            tier_min,
            tier_max,
            capabilities,
        },
        mode: raw.pipeline.mode,
        ..PipelineIr::default()
    };

    for (key, setting) in raw.settings {
        // Handle explicit `[lo, hi]` ranges and one-sided `min`/`max` v1 bounds without clamping
        // missing endpoints to hardcoded tier numbers.
        let (range, range_is_float) = if let Some([lo_v, hi_v]) = setting.range {
            let (lo, lo_float) = toml_numeric_f64(&key, &lo_v)?;
            let (hi, hi_float) = toml_numeric_f64(&key, &hi_v)?;
            if lo > hi {
                return Err(TomlDesugarError::InvalidSettingRange {
                    key,
                    min: lo,
                    max: hi,
                });
            }
            (Some((lo, hi)), lo_float || hi_float)
        } else if setting.min.is_some() || setting.max.is_some() {
            let (lo, lo_float) = if let Some(ref min_v) = setting.min {
                toml_numeric_f64(&key, min_v)?
            } else {
                (f64::MIN, false)
            };
            let (hi, hi_float) = if let Some(ref max_v) = setting.max {
                toml_numeric_f64(&key, max_v)?
            } else {
                (f64::MAX, false)
            };
            if lo > hi {
                return Err(TomlDesugarError::InvalidSettingRange {
                    key,
                    min: lo,
                    max: hi,
                });
            }
            (Some((lo, hi)), lo_float || hi_float)
        } else {
            (None, false)
        };
        let (step, step_is_float) = if let Some(step_v) = setting.step {
            let (s, s_float) = toml_numeric_f64(&key, &step_v)?;
            (Some(s), s_float)
        } else {
            (None, false)
        };
        let mut default_value = toml_value_to_setting(&key, &setting.default)?;
        if (range_is_float || step_is_float)
            && let SettingValueIr::Int(i) = default_value
        {
            default_value = SettingValueIr::Float(i as f32);
        }
        if let Some((lo, hi)) = range
            && let Some(num) = setting_numeric_f64(&default_value)
            && !(lo..=hi).contains(&num)
        {
            return Err(TomlDesugarError::SettingValueOutOfRange {
                key,
                value: num,
                min: lo,
                max: hi,
            });
        }
        let variant_names = if setting.variants.is_empty() {
            setting.options
        } else {
            setting.variants
        };
        let enum_variants: Vec<SmolStr> = variant_names.into_iter().map(SmolStr::new).collect();
        if !enum_variants.is_empty() {
            match &default_value {
                SettingValueIr::EnumVariant(v) if enum_variants.contains(v) => {}
                _ => return Err(TomlDesugarError::InvalidSettingValue(key)),
            }
        }
        let name = SmolStr::new(&key);
        let label = setting.label.map_or_else(|| name.clone(), SmolStr::new);
        ir.settings.insert(
            name.clone(),
            SettingIr {
                name,
                category: SmolStr::new(setting.category),
                is_spec: setting.spec || setting.runtime == Some(false),
                label,
                description: SmolStr::new(setting.description),
                default_value,
                range,
                step,
                tiers: setting.tiers.map_or(
                    (setting.min_tier.unwrap_or(0), setting.max_tier.unwrap_or(3)),
                    |[lo, hi]| (lo, hi),
                ),
                enum_variants,
            },
        );
    }

    for (idx, (name, preset)) in raw.presets.into_iter().enumerate() {
        let preset_name = SmolStr::new(&name);
        let label = preset
            .label
            .map_or_else(|| preset_name.clone(), SmolStr::new);
        let order = preset
            .order
            .unwrap_or_else(|| i32::try_from(idx).unwrap_or(i32::MAX));
        let mut overrides = IndexMap::with_capacity(preset.overrides.len());
        for (k, v) in preset.overrides {
            let Some(setting) = ir.settings.get(k.as_str()) else {
                return Err(TomlDesugarError::UnknownPresetSetting {
                    preset: name.clone(),
                    key: k,
                });
            };
            let parsed_val = toml_value_to_setting(&k, &v)?;
            let val = match (&setting.default_value, parsed_val) {
                (SettingValueIr::Float(_), SettingValueIr::Int(i)) => {
                    SettingValueIr::Float(i as f32)
                }
                (SettingValueIr::Bool(_), v @ SettingValueIr::Bool(_))
                | (SettingValueIr::Int(_), v @ SettingValueIr::Int(_))
                | (SettingValueIr::Float(_), v @ SettingValueIr::Float(_))
                | (SettingValueIr::ColorRgb(_), v @ SettingValueIr::ColorRgb(_)) => v,
                (SettingValueIr::EnumVariant(_), SettingValueIr::EnumVariant(variant)) => {
                    if !setting.enum_variants.is_empty()
                        && !setting.enum_variants.contains(&variant)
                    {
                        return Err(TomlDesugarError::InvalidSettingValue(k));
                    }
                    SettingValueIr::EnumVariant(variant)
                }
                _ => return Err(TomlDesugarError::InvalidSettingValue(k)),
            };
            if let Some((lo, hi)) = setting.range
                && let Some(num) = setting_numeric_f64(&val)
                && !(lo..=hi).contains(&num)
            {
                return Err(TomlDesugarError::PresetValueOutOfRange {
                    preset: name.clone(),
                    key: k,
                    value: num,
                    min: lo,
                    max: hi,
                });
            }
            overrides.insert(SmolStr::new(k), val);
        }
        ir.presets.push(PresetIr {
            name: preset_name,
            label,
            description: SmolStr::new(preset.description),
            order,
            tiers: preset.tiers.map_or((0, 3), |[lo, hi]| (lo, hi)),
            overrides,
        });
    }

    let resources = match raw.resources {
        LegacyResources::Seq(list) => list,
        LegacyResources::Map(map) => map
            .into_iter()
            .map(|(key, mut resource)| {
                resource.name.get_or_insert(SmolStr::new(&key));
                resource
            })
            .collect(),
    };
    for resource in resources {
        let slot = u32::try_from(ir.resource_textures.len()).unwrap_or(u32::MAX);
        if slot >= PACK_RESOURCE_SLOTS {
            return Err(TomlDesugarError::TooManyResources);
        }
        let format = parse_image_format(&resource.format)?;
        let name = resource.name.unwrap_or_else(|| {
            SmolStr::new(
                resource
                    .path
                    .rsplit(['/', '\\'])
                    .next()
                    .unwrap_or(resource.path.as_str()),
            )
        });
        ir.resource_textures.push(ResourceTextureDecl {
            slot,
            name,
            path: SmolStr::new(resource.path),
            format,
            filter: SmolStr::new(resource.filter),
            wrap: SmolStr::new(resource.wrap),
        });
    }

    for (stage_name, target_fn) in raw.hooks {
        let stage =
            HookStage::from_name(&stage_name).ok_or(TomlDesugarError::UnknownHook(stage_name))?;
        ir.hooks.push(HookRegistration {
            stage,
            target_fn: SmolStr::new(target_fn),
        });
    }

    for (name, target) in raw.targets {
        let format = parse_image_format(&target.format)?;
        let key = SmolStr::new(name);
        ir.targets.insert(
            key.clone(),
            TargetDecl {
                name: key,
                format,
                scale: target.scale,
                layers: target.array_layers,
                mip_levels: target.mip_levels,
                temporal: target.temporal,
                clear_color: target.clear_color,
            },
        );
    }

    for (name, buf) in raw.buffers {
        let key = SmolStr::new(name);
        ir.buffers.insert(
            key.clone(),
            BufferDecl {
                name: key,
                element_type: SmolStr::new(buf.element_type),
                count: buf.count,
                size_bytes: buf.size_bytes,
                indirect: buf.indirect,
            },
        );
    }

    for pass in raw.passes {
        let replaces = match pass.replaces {
            Some(ref name) => Some(
                CanonicalPass::from_name(name)
                    .ok_or_else(|| TomlDesugarError::UnknownCanonicalPass(name.clone()))?,
            ),
            None => None,
        };
        let after: Vec<PassRef> = pass.after.iter().map(|a| parse_pass_ref(a)).collect();
        let before: Vec<PassRef> = pass.before.iter().map(|b| parse_pass_ref(b)).collect();
        let mut bindings = IndexMap::with_capacity(pass.bindings.len());
        for (bind_key, bind_val) in pass.bindings {
            let binding_ref =
                parse_binding_ref(&bind_key, &bind_val, &ir.resource_textures, &ir.buffers)?;
            bindings.insert(SmolStr::new(bind_key), binding_ref);
        }
        let (dispatch_target, dispatch) = if let Some(xyz) = pass.dispatch_xyz {
            (None, Some(PassDispatch::Explicit(xyz)))
        } else if let Some(indirect) = pass.dispatch_indirect {
            (None, Some(PassDispatch::Indirect(SmolStr::new(indirect))))
        } else if let Some(target) = pass.dispatch {
            let target_smol = SmolStr::new(target);
            (
                Some(target_smol.clone()),
                Some(PassDispatch::ForTarget(target_smol)),
            )
        } else {
            (None, None)
        };
        ir.passes.push(PassNode {
            name: SmolStr::new(pass.name),
            kind: pass.kind,
            entry_fn: SmolStr::new(pass.shader),
            replaces,
            after,
            before,
            bindings,
            meta: PassMeta {
                reads: pass.reads,
                writes: pass.writes,
                scale: pass.scale,
                tiers: if pass.min_tier.is_none() && pass.max_tier.is_none() {
                    None
                } else {
                    Some((pass.min_tier.unwrap_or(0), pass.max_tier.unwrap_or(3)))
                },
            },
            dispatch_target,
            dispatch,
        });
    }

    Ok(ir)
}
