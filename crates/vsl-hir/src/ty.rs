use indexmap::IndexMap;
use la_arena::{Arena, Idx};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use vsl_mc::ImageFormat;

pub type TypeId = Idx<Type>;

/// Memory layout rule for structs, push constants, UBOs, and BDA buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutKind {
    Scalar,
    Std430,
    Std140,
}

impl LayoutKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "scalar" => Some(Self::Scalar),
            "std430" => Some(Self::Std430),
            "std140" => Some(Self::Std140),
            _ => None,
        }
    }
}

/// Scalar primitive types supported by VSL 2.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarKind {
    Bool,
    I32,
    U32,
    I64,
    U64,
    F32,
    /// Requires `#[cap(float16)]`.
    F16,
}

impl ScalarKind {
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::F16 => 2,
            Self::Bool | Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }

    pub const fn requires_float16(self) -> bool {
        matches!(self, Self::F16)
    }
}

/// Canonical resolved VSL 2.0 type stored in [`TypeArena`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    Scalar(ScalarKind),
    Vector {
        scalar: ScalarKind,
        len: u8,
    },
    Matrix {
        scalar: ScalarKind,
        cols: u8,
        rows: u8,
    },
    Array {
        element: TypeId,
        len: u32,
    },
    /// Physical storage buffer pointer (`ptr<T, align = N>` or `ptr_mut<T, align = N>`).
    Pointer {
        pointee: TypeId,
        mutable: bool,
        align: u32,
    },
    Struct {
        name: SmolStr,
        layout: LayoutKind,
        fields: Vec<(SmolStr, TypeId)>,
    },
    Texture2D {
        sample_scalar: ScalarKind,
        unsized_array: bool,
    },
    Texture2DArray {
        sample_scalar: ScalarKind,
    },
    Texture3D {
        sample_scalar: ScalarKind,
    },
    TextureCube {
        sample_scalar: ScalarKind,
    },
    RwImage2D(ImageFormat),
    RwImage2DArray(ImageFormat),
    RwImage3D(ImageFormat),
    Sampler,
    SamplerShadow,
    /// Read-only storage buffer resource (`Buffer<T>`).
    Buffer {
        element: TypeId,
    },
    /// Read-write storage buffer resource (`RWBuffer<T>`).
    RwBuffer {
        element: TypeId,
    },
    AccelStruct,
    /// Non-copyable local handle; prohibited inside structs, arrays, or `OpPhi`.
    RayQuery,
    /// Opaque SER hit object handle (`OpTypeHitObjectNV` / `OpTypeHitObjectEXT`).
    HitObject,
    /// Uninstantiated generic type parameter (for example `S` in `struct Reservoir<S>`).
    TypeParam(SmolStr),
}

/// Computed byte layout for a struct under `scalar`, `std430`, or `std140`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    pub field_offsets: Vec<(SmolStr, u32, u32)>,
    pub size: u32,
    pub align: u32,
}

/// Dense arena owning all interned types for a compilation session.
#[derive(Debug, Default, Clone)]
pub struct TypeArena {
    arena: Arena<Type>,
    intern: IndexMap<Type, TypeId>,
}

impl TypeArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&mut self, ty: Type) -> TypeId {
        // Structs remain nominal because fields are populated in a second pass and carry
        // distinct layout/interface decorations.
        if matches!(ty, Type::Struct { .. }) {
            return self.arena.alloc(ty);
        }
        if let Some(&existing) = self.intern.get(&ty) {
            return existing;
        }
        let id = self.arena.alloc(ty.clone());
        self.intern.insert(ty, id);
        id
    }

    /// Returns `true` if `id` contains an infinite by-value cycle (recursive struct).
    pub fn has_layout_cycle(&self, id: TypeId) -> bool {
        let mut stack = Vec::new();
        self.has_layout_cycle_inner(id, &mut stack)
    }

    fn has_layout_cycle_inner(&self, id: TypeId, stack: &mut Vec<TypeId>) -> bool {
        if stack.contains(&id) {
            return true;
        }
        stack.push(id);
        let cyclic = match &self.arena[id] {
            Type::Array { element, .. } => self.has_layout_cycle_inner(*element, stack),
            Type::Struct { fields, .. } => fields
                .iter()
                .any(|(_, field_ty)| self.has_layout_cycle_inner(*field_ty, stack)),
            _ => false,
        };
        stack.pop();
        cyclic
    }

    pub fn get(&self, id: TypeId) -> &Type {
        &self.arena[id]
    }

    /// Populates the resolved fields of a pre-registered `Type::Struct` during two-pass lowering.
    pub fn set_struct_fields(&mut self, id: TypeId, new_fields: Vec<(SmolStr, TypeId)>) {
        if let Type::Struct { fields, .. } = &mut self.arena[id] {
            *fields = new_fields;
        }
    }

    /// Returns `true` if `id` is `RayQuery` or transitively contains one.
    pub fn contains_ray_query(&self, id: TypeId) -> bool {
        let mut visited = Vec::new();
        self.contains_ray_query_inner(id, &mut visited)
    }

    fn contains_ray_query_inner(&self, id: TypeId, visited: &mut Vec<TypeId>) -> bool {
        if visited.contains(&id) {
            return false;
        }
        visited.push(id);
        let result = match &self.arena[id] {
            Type::RayQuery => true,
            Type::Array { element, .. } | Type::Buffer { element } | Type::RwBuffer { element } => {
                self.contains_ray_query_inner(*element, visited)
            }
            Type::Pointer { pointee, .. } => self.contains_ray_query_inner(*pointee, visited),
            Type::Struct { fields, .. } => fields
                .iter()
                .any(|(_, field_ty)| self.contains_ray_query_inner(*field_ty, visited)),
            _ => false,
        };
        visited.pop();
        result
    }

    /// Returns `true` if `id` uses `f16` scalars/vectors and therefore requires `#[cap(float16)]`.
    pub fn requires_float16(&self, id: TypeId) -> bool {
        let mut visited = Vec::new();
        self.requires_float16_inner(id, &mut visited)
    }

    fn requires_float16_inner(&self, id: TypeId, visited: &mut Vec<TypeId>) -> bool {
        if visited.contains(&id) {
            return false;
        }
        visited.push(id);
        let result = match &self.arena[id] {
            Type::Scalar(s) | Type::Vector { scalar: s, .. } | Type::Matrix { scalar: s, .. } => {
                s.requires_float16()
            }
            Type::Array { element, .. } | Type::Buffer { element } | Type::RwBuffer { element } => {
                self.requires_float16_inner(*element, visited)
            }
            Type::Pointer { pointee, .. } => self.requires_float16_inner(*pointee, visited),
            Type::Struct { fields, .. } => fields
                .iter()
                .any(|(_, field_ty)| self.requires_float16_inner(*field_ty, visited)),
            _ => false,
        };
        visited.pop();
        result
    }

    /// Returns `true` if `id` is `bool` or transitively contains a `bool` scalar/vector.
    pub fn contains_bool(&self, id: TypeId) -> bool {
        let mut visited = Vec::new();
        self.contains_bool_inner(id, &mut visited)
    }

    fn contains_bool_inner(&self, id: TypeId, visited: &mut Vec<TypeId>) -> bool {
        if visited.contains(&id) {
            return false;
        }
        visited.push(id);
        let result = match &self.arena[id] {
            Type::Scalar(ScalarKind::Bool)
            | Type::Vector {
                scalar: ScalarKind::Bool,
                ..
            } => true,
            Type::Array { element, .. }
            | Type::Buffer { element }
            | Type::RwBuffer { element }
            | Type::Pointer {
                pointee: element, ..
            } => self.contains_bool_inner(*element, visited),
            Type::Struct { fields, .. } => fields
                .iter()
                .any(|(_, field_ty)| self.contains_bool_inner(*field_ty, visited)),
            _ => false,
        };
        visited.pop();
        result
    }

    /// Returns `true` if `id` is a physical storage buffer pointer or transitively contains one.
    pub fn contains_pointer(&self, id: TypeId) -> bool {
        let mut visited = Vec::new();
        self.contains_pointer_inner(id, &mut visited)
    }

    fn contains_pointer_inner(&self, id: TypeId, visited: &mut Vec<TypeId>) -> bool {
        if visited.contains(&id) {
            return false;
        }
        visited.push(id);
        let result = match &self.arena[id] {
            Type::Pointer { .. } => true,
            Type::Array { element, .. } | Type::Buffer { element } | Type::RwBuffer { element } => {
                self.contains_pointer_inner(*element, visited)
            }
            Type::Struct { fields, .. } => fields
                .iter()
                .any(|(_, field_ty)| self.contains_pointer_inner(*field_ty, visited)),
            _ => false,
        };
        visited.pop();
        result
    }

    /// Computes `(size, alignment)` in bytes under Vulkan `scalar`, `std430`, or `std140` rules.
    pub fn size_and_align(&self, id: TypeId, layout: LayoutKind) -> Option<(u32, u32)> {
        let mut visited = Vec::new();
        self.size_and_align_inner(id, layout, &mut visited)
    }

    fn size_and_align_inner(
        &self,
        id: TypeId,
        layout: LayoutKind,
        visited: &mut Vec<TypeId>,
    ) -> Option<(u32, u32)> {
        if visited.contains(&id) {
            return None;
        }
        visited.push(id);
        let result = match &self.arena[id] {
            Type::Scalar(s) => {
                let sz = s.byte_size();
                Some((sz, sz))
            }
            Type::Vector { scalar, len } => {
                let elem = scalar.byte_size();
                let count = u32::from(*len);
                let align = match layout {
                    LayoutKind::Scalar => elem,
                    LayoutKind::Std430 | LayoutKind::Std140 => {
                        if count == 3 {
                            elem * 4
                        } else {
                            elem * count
                        }
                    }
                };
                Some((elem * count, align))
            }
            Type::Matrix { scalar, cols, rows } => {
                let elem = scalar.byte_size();
                let col_align = match layout {
                    LayoutKind::Scalar => elem,
                    LayoutKind::Std430 => {
                        let r = u32::from(*rows);
                        if r == 3 { elem * 4 } else { elem * r }
                    }
                    LayoutKind::Std140 => {
                        let r = u32::from(*rows);
                        let base = if r == 3 { elem * 4 } else { elem * r };
                        base.max(16)
                    }
                };
                let stride = round_up(elem * u32::from(*rows), col_align)?;
                Some((stride.checked_mul(u32::from(*cols))?, col_align))
            }
            Type::Pointer { .. } => Some((8, 8)),
            Type::Array { element, len } => {
                let (elem_sz, elem_align) = self.size_and_align_inner(*element, layout, visited)?;
                let align = match layout {
                    LayoutKind::Std140 => elem_align.max(16),
                    LayoutKind::Scalar | LayoutKind::Std430 => elem_align,
                };
                let stride = round_up(elem_sz, align)?;
                Some((stride.checked_mul(*len)?, align))
            }
            Type::Struct {
                fields,
                layout: struct_layout,
                ..
            } => {
                // Propagate the stricter enclosing layout (Scalar < Std430 < Std140) so nested
                // structs inside std140 UBOs obey Vulkan std140 alignment and array-stride rules.
                let effective = stricter_layout(layout, *struct_layout);
                let computed = self.compute_struct_layout_inner(fields, effective, visited)?;
                if matches!(effective, LayoutKind::Std140) {
                    // Vulkan/GLSL std140 rounds nested struct base alignment up to a multiple of 16 bytes.
                    let align = round_up(computed.align.max(16), 16)?;
                    let size = round_up(computed.size, align)?;
                    Some((size, align))
                } else {
                    Some((computed.size, computed.align))
                }
            }
            _ => None,
        };
        visited.pop();
        result
    }

    /// Computes exact member byte offsets, total size, and alignment for a struct.
    pub fn compute_struct_layout(
        &self,
        fields: &[(SmolStr, TypeId)],
        layout: LayoutKind,
    ) -> Option<StructLayout> {
        let mut visited = Vec::new();
        self.compute_struct_layout_inner(fields, layout, &mut visited)
    }

    fn compute_struct_layout_inner(
        &self,
        fields: &[(SmolStr, TypeId)],
        layout: LayoutKind,
        visited: &mut Vec<TypeId>,
    ) -> Option<StructLayout> {
        let mut offset = 0u32;
        let mut max_align = if matches!(layout, LayoutKind::Std140) {
            16
        } else {
            1
        };
        let mut field_offsets = Vec::with_capacity(fields.len());

        for (name, ty_id) in fields {
            let (sz, align) = self.size_and_align_inner(*ty_id, layout, visited)?;
            offset = round_up(offset, align)?;
            field_offsets.push((name.clone(), offset, sz));
            offset = offset.checked_add(sz)?;
            max_align = max_align.max(align);
        }

        let size = round_up(offset, max_align)?;
        Some(StructLayout {
            field_offsets,
            size,
            align: max_align,
        })
    }
}

const fn stricter_layout(enclosing: LayoutKind, inner: LayoutKind) -> LayoutKind {
    match (enclosing, inner) {
        (LayoutKind::Std140, _) | (_, LayoutKind::Std140) => LayoutKind::Std140,
        (LayoutKind::Std430, _) | (_, LayoutKind::Std430) => LayoutKind::Std430,
        (LayoutKind::Scalar, LayoutKind::Scalar) => LayoutKind::Scalar,
    }
}

const fn round_up(value: u32, align: u32) -> Option<u32> {
    if align <= 1 {
        return Some(value);
    }
    let rem = value % align;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(align - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_scalar_and_std430_layouts() {
        let mut arena = TypeArena::new();
        let i32_ty = arena.alloc(Type::Scalar(ScalarKind::I32));
        let f32_ty = arena.alloc(Type::Scalar(ScalarKind::F32));
        let vec3_ty = arena.alloc(Type::Vector {
            scalar: ScalarKind::F32,
            len: 3,
        });

        // Scalar layout packs vec3 on 4-byte alignment (16 bytes total).
        let scalar = arena
            .compute_struct_layout(
                &[
                    (SmolStr::new_static("step_size"), i32_ty),
                    (SmolStr::new_static("rgb"), vec3_ty),
                ],
                LayoutKind::Scalar,
            )
            .expect("scalar layout");
        assert_eq!(scalar.field_offsets[0].1, 0);
        assert_eq!(scalar.field_offsets[1].1, 4);
        assert_eq!(scalar.size, 16);

        // std430 aligns vec3 to 16 bytes (32 bytes total).
        let std430 = arena
            .compute_struct_layout(
                &[
                    (SmolStr::new_static("a"), f32_ty),
                    (SmolStr::new_static("rgb"), vec3_ty),
                ],
                LayoutKind::Std430,
            )
            .expect("std430 layout");
        assert_eq!(std430.field_offsets[1].1, 16);
        assert_eq!(std430.size, 32);
    }

    #[test]
    fn rounds_std140_nested_struct_alignment_to_16_bytes() {
        let mut arena = TypeArena::new();
        let f32_ty = arena.alloc(Type::Scalar(ScalarKind::F32));
        let inner_ty = arena.alloc(Type::Struct {
            name: SmolStr::new_static("Inner"),
            layout: LayoutKind::Std430,
            fields: vec![(SmolStr::new_static("x"), f32_ty)],
        });

        let outer = arena
            .compute_struct_layout(
                &[
                    (SmolStr::new_static("a"), f32_ty),
                    (SmolStr::new_static("inner"), inner_ty),
                ],
                LayoutKind::Std140,
            )
            .expect("std140 nested struct layout");

        assert_eq!(outer.field_offsets[0], (SmolStr::new_static("a"), 0, 4));
        assert_eq!(
            outer.field_offsets[1],
            (SmolStr::new_static("inner"), 16, 16)
        );
        assert_eq!(outer.align, 16);
        assert_eq!(outer.size, 32);
    }

    #[test]
    fn round_up_overflow_returns_none_instead_of_wrapping() {
        assert_eq!(round_up(u32::MAX, 16), None);
        assert_eq!(round_up(u32::MAX - 3, 16), None);
        assert_eq!(round_up(16, 16), Some(16));

        let mut arena = TypeArena::new();
        let f32_ty = arena.alloc(Type::Scalar(ScalarKind::F32));
        // 4 * 0x3FFF_FFFF = 0xFFFF_FFFC, which overflows when rounded up to 16-byte std140 alignment.
        let huge_arr = arena.alloc(Type::Array {
            element: f32_ty,
            len: 0x3FFF_FFFF,
        });
        assert_eq!(
            arena.compute_struct_layout(
                &[
                    (SmolStr::new_static("a"), huge_arr),
                    (SmolStr::new_static("b"), f32_ty),
                ],
                LayoutKind::Std430,
            ),
            None
        );
    }

    #[test]
    fn contains_ray_query_and_bool_traverse_pointers_and_buffers() {
        let mut arena = TypeArena::new();
        let rq_ty = arena.alloc(Type::RayQuery);
        let rq_ptr = arena.alloc(Type::Pointer {
            pointee: rq_ty,
            mutable: false,
            align: 8,
        });
        assert!(arena.contains_ray_query(rq_ptr));

        let bool_ty = arena.alloc(Type::Scalar(ScalarKind::Bool));
        let bvec4_ty = arena.alloc(Type::Vector {
            scalar: ScalarKind::Bool,
            len: 4,
        });
        let buf_bool = arena.alloc(Type::Buffer { element: bool_ty });
        let rwbuf_bool = arena.alloc(Type::RwBuffer { element: bool_ty });
        let ptr_bvec4 = arena.alloc(Type::Pointer {
            pointee: bvec4_ty,
            mutable: true,
            align: 16,
        });
        assert!(arena.contains_bool(buf_bool));
        assert!(arena.contains_bool(rwbuf_bool));
        assert!(arena.contains_bool(ptr_bvec4));
    }

    #[test]
    fn std140_enclosing_layout_propagates_to_nested_structs() {
        let mut arena = TypeArena::new();
        let f32_ty = arena.alloc(Type::Scalar(ScalarKind::F32));
        let arr_f32_2 = arena.alloc(Type::Array {
            element: f32_ty,
            len: 2,
        });
        let inner_scalar = arena.alloc(Type::Struct {
            name: SmolStr::new_static("InnerScalar"),
            layout: LayoutKind::Scalar,
            fields: vec![(SmolStr::new_static("vals"), arr_f32_2)],
        });

        // Under std140, `[f32; 2]` inside `InnerScalar` must use 16-byte array stride (32 bytes total).
        let outer = arena
            .compute_struct_layout(
                &[(SmolStr::new_static("inner"), inner_scalar)],
                LayoutKind::Std140,
            )
            .expect("std140 nested scalar struct layout");
        assert_eq!(
            outer.field_offsets[0],
            (SmolStr::new_static("inner"), 0, 32)
        );
        assert_eq!(outer.align, 16);
        assert_eq!(outer.size, 32);
    }
}
