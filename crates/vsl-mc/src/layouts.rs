//! Canonical GPU memory layouts for Hyaloclast Set 0 / Set 1 built-in buffers
//! and typed `PackParamBuffer` schema descriptors shared across VSL 2.0 crates.

use crate::{DESCRIPTOR_SET_PER_PASS, DESCRIPTOR_SET_SCENE_GLOBALS};
use serde::{Deserialize, Serialize};

/// Scalar, vector, matrix, or runtime-array element type inside a built-in engine buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFieldType {
    Bool,
    I32,
    U32,
    I64,
    U64,
    F32,
    Vec2,
    Vec3,
    Vec4,
    UVec2,
    UVec3,
    UVec4,
    Mat4,
    /// Trailing `std430` runtime array of a named built-in element struct.
    RuntimeArrayStruct(&'static str),
}

/// Single member description inside a built-in engine UBO or SSBO struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinFieldDesc {
    pub name: &'static str,
    pub byte_offset: u32,
    pub byte_size: u32,
    pub ty: BuiltinFieldType,
}

/// Complete static layout description for a built-in engine UBO/SSBO or element struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinStructDesc {
    pub name: &'static str,
    pub set: Option<u32>,
    pub binding: Option<u32>,
    /// Fixed byte size for UBOs and array element structs; header size for SSBOs with a runtime array.
    pub total_size: u32,
    pub align: u32,
    pub fields: &'static [BuiltinFieldDesc],
}

impl BuiltinStructDesc {
    /// Looks up a field descriptor by member name.
    pub fn field(&self, name: &str) -> Option<&'static BuiltinFieldDesc> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Returns the zero-based struct member index for `OpAccessChain`.
    pub fn field_index(&self, name: &str) -> Option<u32> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .map(|idx| idx as u32)
    }
}

/// Number of `vec4` slots in the engine's `packParams[12]` array.
const WORLD_PACK_PARAM_SLOT_COUNT: u32 = 12;
/// Byte stride between consecutive `packParams` slots (`vec4` under `std140`).
const WORLD_PACK_PARAM_SLOT_STRIDE: u32 = 16;
/// Byte offset of `packParams[0]` in the 544-byte `WorldUBO`.
const WORLD_PACK_PARAMS_OFFSET: u32 = 320;
/// Canonical field names for the 12 `packParams` slots. Slot 0 keeps the unsuffixed engine name
/// so `mc::world.pack_params` reads the first `vec4`; later slots use a numeric suffix because
/// built-in layout descriptors carry scalar/vector/matrix types but no fixed-size array type.
const WORLD_PACK_PARAM_FIELD_NAMES: [&str; WORLD_PACK_PARAM_SLOT_COUNT as usize] = [
    "pack_params",
    "pack_params_1",
    "pack_params_2",
    "pack_params_3",
    "pack_params_4",
    "pack_params_5",
    "pack_params_6",
    "pack_params_7",
    "pack_params_8",
    "pack_params_9",
    "pack_params_10",
    "pack_params_11",
];

const fn world_pack_param_field(slot: usize) -> BuiltinFieldDesc {
    BuiltinFieldDesc {
        name: WORLD_PACK_PARAM_FIELD_NAMES[slot],
        byte_offset: WORLD_PACK_PARAMS_OFFSET + slot as u32 * WORLD_PACK_PARAM_SLOT_STRIDE,
        byte_size: WORLD_PACK_PARAM_SLOT_STRIDE,
        ty: BuiltinFieldType::Vec4,
    }
}

/// Set 0, binding 0: per-pass `WorldUBO` (`std140`, 544 bytes).
///
/// Field names and offsets mirror `WorldUboData` in `vulkanshaders-core` and the GLSL
/// `WorldUBO` block in the engine's `common.vsl`. The first 528 bytes are the v1 `WorldUBO` ABI;
/// the remaining 16 bytes carry the VSL pack-parameter BDA (528..535), the CPU-integrated wind
/// phase (536..539), and `std140` padding (540..543).
pub const WORLD_UBO_FIELDS: &[BuiltinFieldDesc] = &[
    BuiltinFieldDesc {
        name: "cur_view_proj",
        byte_offset: 0,
        byte_size: 64,
        ty: BuiltinFieldType::Mat4,
    },
    BuiltinFieldDesc {
        name: "prev_view_proj",
        byte_offset: 64,
        byte_size: 64,
        ty: BuiltinFieldType::Mat4,
    },
    BuiltinFieldDesc {
        name: "inv_proj",
        byte_offset: 128,
        byte_size: 64,
        ty: BuiltinFieldType::Mat4,
    },
    BuiltinFieldDesc {
        name: "inv_view",
        byte_offset: 192,
        byte_size: 64,
        ty: BuiltinFieldType::Mat4,
    },
    BuiltinFieldDesc {
        name: "camera_pos_frac",
        byte_offset: 256,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "frame_index",
        byte_offset: 268,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "camera_delta",
        byte_offset: 272,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "game_time_sec",
        byte_offset: 284,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "sun_dir",
        byte_offset: 288,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "rain",
        byte_offset: 300,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "thunder",
        byte_offset: 304,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "eye_in_fluid",
        byte_offset: 308,
        byte_size: 4,
        ty: BuiltinFieldType::I32,
    },
    BuiltinFieldDesc {
        name: "depth_far",
        byte_offset: 312,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "held_light",
        byte_offset: 316,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    world_pack_param_field(0),
    world_pack_param_field(1),
    world_pack_param_field(2),
    world_pack_param_field(3),
    world_pack_param_field(4),
    world_pack_param_field(5),
    world_pack_param_field(6),
    world_pack_param_field(7),
    world_pack_param_field(8),
    world_pack_param_field(9),
    world_pack_param_field(10),
    world_pack_param_field(11),
    BuiltinFieldDesc {
        name: "held_light_position",
        byte_offset: 512,
        byte_size: 16,
        ty: BuiltinFieldType::Vec4,
    },
    BuiltinFieldDesc {
        name: "pack_params_bda",
        byte_offset: 528,
        byte_size: 8,
        ty: BuiltinFieldType::U64,
    },
    BuiltinFieldDesc {
        name: "wind_phase",
        byte_offset: 536,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "pack_params_bda_pad",
        byte_offset: 540,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
];

pub const WORLD_UBO_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "WorldUBO",
    set: Some(DESCRIPTOR_SET_PER_PASS),
    binding: Some(0),
    total_size: 544,
    align: 16,
    fields: WORLD_UBO_FIELDS,
};

/// Element layout for Set 1, binding 1: `InstanceMeta` (`std430`, 144 bytes).
pub const INSTANCE_META_FIELDS: &[BuiltinFieldDesc] = &[
    BuiltinFieldDesc {
        name: "model_to_world",
        byte_offset: 0,
        byte_size: 64,
        ty: BuiltinFieldType::Mat4,
    },
    BuiltinFieldDesc {
        name: "world_to_model",
        byte_offset: 64,
        byte_size: 64,
        ty: BuiltinFieldType::Mat4,
    },
    BuiltinFieldDesc {
        name: "material_id",
        byte_offset: 128,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "texture_id",
        byte_offset: 132,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "flags",
        byte_offset: 136,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "emissive_index",
        byte_offset: 140,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
];

pub const INSTANCE_META_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "InstanceMeta",
    set: None,
    binding: None,
    total_size: 144,
    align: 16,
    fields: INSTANCE_META_FIELDS,
};

/// Set 1, binding 1: `InstanceMetaSSBO` (`std430` runtime array of `InstanceMeta`).
pub const INSTANCE_META_SSBO_FIELDS: &[BuiltinFieldDesc] = &[BuiltinFieldDesc {
    name: "items",
    byte_offset: 0,
    byte_size: 144,
    ty: BuiltinFieldType::RuntimeArrayStruct("InstanceMeta"),
}];

pub const INSTANCE_META_SSBO_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "InstanceMetaSSBO",
    set: Some(DESCRIPTOR_SET_SCENE_GLOBALS),
    binding: Some(1),
    total_size: 0,
    align: 16,
    fields: INSTANCE_META_SSBO_FIELDS,
};

/// Set 1, binding 2: `SharcConfigUBO` (`std140`, 48 bytes).
pub const SHARC_CONFIG_UBO_FIELDS: &[BuiltinFieldDesc] = &[
    BuiltinFieldDesc {
        name: "hash_table_bda",
        byte_offset: 0,
        byte_size: 8,
        ty: BuiltinFieldType::U64,
    },
    BuiltinFieldDesc {
        name: "capacity",
        byte_offset: 8,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "grid_scale",
        byte_offset: 12,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "camera_pos",
        byte_offset: 16,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "accumulation_limit",
        byte_offset: 28,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "frame_index",
        byte_offset: 32,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "stale_frame_threshold",
        byte_offset: 36,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "radiance_scale",
        byte_offset: 40,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "flags",
        byte_offset: 44,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
];

pub const SHARC_CONFIG_UBO_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "SharcConfigUBO",
    set: Some(DESCRIPTOR_SET_SCENE_GLOBALS),
    binding: Some(2),
    total_size: 48,
    align: 16,
    fields: SHARC_CONFIG_UBO_FIELDS,
};

/// Element layout for Set 1, binding 3: `EmissiveLight` (`std430`, 48 bytes).
pub const EMISSIVE_LIGHT_FIELDS: &[BuiltinFieldDesc] = &[
    BuiltinFieldDesc {
        name: "position",
        byte_offset: 0,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "radius",
        byte_offset: 12,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "radiance",
        byte_offset: 16,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "flux",
        byte_offset: 28,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "normal",
        byte_offset: 32,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "flags",
        byte_offset: 44,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
];

pub const EMISSIVE_LIGHT_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "EmissiveLight",
    set: None,
    binding: None,
    total_size: 48,
    align: 16,
    fields: EMISSIVE_LIGHT_FIELDS,
};

/// Set 1, binding 3: `EmissiveLightSSBO` (`std430` header + runtime array of `EmissiveLight`).
pub const EMISSIVE_LIGHT_SSBO_FIELDS: &[BuiltinFieldDesc] = &[
    BuiltinFieldDesc {
        name: "count",
        byte_offset: 0,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "_pad0",
        byte_offset: 4,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "_pad1",
        byte_offset: 8,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "_pad2",
        byte_offset: 12,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "items",
        byte_offset: 16,
        byte_size: 48,
        ty: BuiltinFieldType::RuntimeArrayStruct("EmissiveLight"),
    },
];

pub const EMISSIVE_LIGHT_SSBO_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "EmissiveLightSSBO",
    set: Some(DESCRIPTOR_SET_SCENE_GLOBALS),
    binding: Some(3),
    total_size: 16,
    align: 16,
    fields: EMISSIVE_LIGHT_SSBO_FIELDS,
};

/// Element layout for Set 1, binding 4: `RestirReservoir` (`std430`, 48 bytes).
pub const RESTIR_RESERVOIR_FIELDS: &[BuiltinFieldDesc] = &[
    BuiltinFieldDesc {
        name: "sample_pos",
        byte_offset: 0,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "weight_sum",
        byte_offset: 12,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "sample_radiance",
        byte_offset: 16,
        byte_size: 12,
        ty: BuiltinFieldType::Vec3,
    },
    BuiltinFieldDesc {
        name: "target_pdf",
        byte_offset: 28,
        byte_size: 4,
        ty: BuiltinFieldType::F32,
    },
    BuiltinFieldDesc {
        name: "sample_count",
        byte_offset: 32,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "light_index",
        byte_offset: 36,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "age",
        byte_offset: 40,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "flags",
        byte_offset: 44,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
];

pub const RESTIR_RESERVOIR_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "RestirReservoir",
    set: None,
    binding: None,
    total_size: 48,
    align: 16,
    fields: RESTIR_RESERVOIR_FIELDS,
};

/// Set 1, binding 4: `RestirReservoirSSBO` (`std430` header + runtime array of `RestirReservoir`).
pub const RESTIR_RESERVOIR_SSBO_FIELDS: &[BuiltinFieldDesc] = &[
    BuiltinFieldDesc {
        name: "count",
        byte_offset: 0,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "_pad0",
        byte_offset: 4,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "_pad1",
        byte_offset: 8,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "_pad2",
        byte_offset: 12,
        byte_size: 4,
        ty: BuiltinFieldType::U32,
    },
    BuiltinFieldDesc {
        name: "items",
        byte_offset: 16,
        byte_size: 48,
        ty: BuiltinFieldType::RuntimeArrayStruct("RestirReservoir"),
    },
];

pub const RESTIR_RESERVOIR_SSBO_LAYOUT: BuiltinStructDesc = BuiltinStructDesc {
    name: "RestirReservoirSSBO",
    set: Some(DESCRIPTOR_SET_SCENE_GLOBALS),
    binding: Some(4),
    total_size: 16,
    align: 16,
    fields: RESTIR_RESERVOIR_SSBO_FIELDS,
};

/// Looks up the built-in buffer struct layout for a fixed Hyaloclast `(set, binding)` slot.
pub fn builtin_descriptor_layout(set: u32, binding: u32) -> Option<&'static BuiltinStructDesc> {
    match (set, binding) {
        (DESCRIPTOR_SET_PER_PASS, 0) => Some(&WORLD_UBO_LAYOUT),
        (DESCRIPTOR_SET_SCENE_GLOBALS, 1) => Some(&INSTANCE_META_SSBO_LAYOUT),
        (DESCRIPTOR_SET_SCENE_GLOBALS, 2) => Some(&SHARC_CONFIG_UBO_LAYOUT),
        (DESCRIPTOR_SET_SCENE_GLOBALS, 3) => Some(&EMISSIVE_LIGHT_SSBO_LAYOUT),
        (DESCRIPTOR_SET_SCENE_GLOBALS, 4) => Some(&RESTIR_RESERVOIR_SSBO_LAYOUT),
        _ => None,
    }
}

/// Looks up a built-in struct layout by canonical name (`WorldUBO`, `InstanceMeta`, etc.).
pub fn builtin_struct_layout_by_name(name: &str) -> Option<&'static BuiltinStructDesc> {
    match name {
        "WorldUBO" => Some(&WORLD_UBO_LAYOUT),
        "InstanceMeta" => Some(&INSTANCE_META_LAYOUT),
        "InstanceMetaSSBO" => Some(&INSTANCE_META_SSBO_LAYOUT),
        "SharcConfigUBO" => Some(&SHARC_CONFIG_UBO_LAYOUT),
        "EmissiveLight" => Some(&EMISSIVE_LIGHT_LAYOUT),
        "EmissiveLightSSBO" => Some(&EMISSIVE_LIGHT_SSBO_LAYOUT),
        "RestirReservoir" => Some(&RESTIR_RESERVOIR_LAYOUT),
        "RestirReservoirSSBO" => Some(&RESTIR_RESERVOIR_SSBO_LAYOUT),
        _ => None,
    }
}

/// Concrete shader value type of a `pack.vsl` setting stored in `PackParamBuffer` or a `SpecConstant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackParamType {
    Bool,
    I32,
    U32,
    F32,
    Vec3,
    Enum,
}

impl PackParamType {
    /// Byte size in the `std430` `PackParamBuffer`.
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::Vec3 => 12,
            Self::Bool | Self::I32 | Self::U32 | Self::F32 | Self::Enum => 4,
        }
    }

    /// Required `std430` alignment in `PackParamBuffer`.
    pub const fn std430_align(self) -> u32 {
        match self {
            Self::Vec3 => 16,
            Self::Bool | Self::I32 | Self::U32 | Self::F32 | Self::Enum => 4,
        }
    }

    /// Canonical GLSL 460 type name used by `vsl-glsl`'s `VsPackParams` bridge.
    pub const fn glsl_type_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I32 | Self::Enum => "int",
            Self::U32 => "uint",
            Self::F32 => "float",
            Self::Vec3 => "vec3",
        }
    }
}

/// Complete layout entry for a single parameter inside `PackParamBuffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PackParamLayout {
    pub byte_offset: u32,
    pub byte_size: u32,
    pub ty: PackParamType,
}

/// Specialization constant slot metadata with its typed default bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpecConstantDefault {
    pub spec_id: u32,
    pub ty: PackParamType,
    pub default_bits: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_ubo_offsets_are_monotonic_and_aligned() {
        let mut cursor = 0u32;
        for field in WORLD_UBO_FIELDS {
            assert!(
                field.byte_offset >= cursor,
                "field {} overlaps previous member",
                field.name
            );
            assert_eq!(field.byte_offset % 4, 0);
            cursor = field.byte_offset + field.byte_size;
        }
        assert_eq!(cursor, WORLD_UBO_LAYOUT.total_size);
        assert_eq!(WORLD_UBO_LAYOUT.total_size, 544);
        assert_eq!(WORLD_UBO_LAYOUT.align, 16);
        for (name, offset) in [
            ("cur_view_proj", 0),
            ("prev_view_proj", 64),
            ("inv_proj", 128),
            ("inv_view", 192),
            ("camera_pos_frac", 256),
            ("frame_index", 268),
            ("camera_delta", 272),
            ("game_time_sec", 284),
            ("sun_dir", 288),
            ("rain", 300),
            ("thunder", 304),
            ("eye_in_fluid", 308),
            ("depth_far", 312),
            ("held_light", 316),
            ("pack_params", 320),
            ("held_light_position", 512),
            ("pack_params_bda", 528),
            ("wind_phase", 536),
            ("pack_params_bda_pad", 540),
        ] {
            assert_eq!(
                WORLD_UBO_LAYOUT.field(name).unwrap().byte_offset,
                offset,
                "unexpected offset for `{name}`"
            );
        }
        for (slot, name) in WORLD_PACK_PARAM_FIELD_NAMES.iter().enumerate() {
            assert_eq!(
                WORLD_UBO_LAYOUT.field(name).unwrap().byte_offset,
                WORLD_PACK_PARAMS_OFFSET + slot as u32 * WORLD_PACK_PARAM_SLOT_STRIDE
            );
        }
        assert_eq!(WORLD_PACK_PARAM_FIELD_NAMES.len(), 12);
    }

    #[test]
    fn scene_global_layouts_resolve_by_set_and_binding() {
        assert_eq!(builtin_descriptor_layout(0, 0), Some(&WORLD_UBO_LAYOUT));
        assert_eq!(
            builtin_descriptor_layout(1, 1),
            Some(&INSTANCE_META_SSBO_LAYOUT)
        );
        assert_eq!(
            builtin_descriptor_layout(1, 2),
            Some(&SHARC_CONFIG_UBO_LAYOUT)
        );
        assert_eq!(
            builtin_descriptor_layout(1, 3),
            Some(&EMISSIVE_LIGHT_SSBO_LAYOUT)
        );
        assert_eq!(
            builtin_descriptor_layout(1, 4),
            Some(&RESTIR_RESERVOIR_SSBO_LAYOUT)
        );
        assert_eq!(INSTANCE_META_LAYOUT.total_size, 144);
        assert_eq!(EMISSIVE_LIGHT_LAYOUT.total_size, 48);
        assert_eq!(RESTIR_RESERVOIR_LAYOUT.total_size, 48);
    }
}
