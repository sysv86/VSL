//! Checked memory layout for explicitly laid out ray shader records.
//!
//! Keep this calculation shared by SPIR-V decorations and host reflection.

use crate::{IrStorageClass, IrStructFieldLayout, IrStructLayout, IrType};
use vsl_hir::LayoutKind;

const fn stricter(a: LayoutKind, b: LayoutKind) -> LayoutKind {
    match (a, b) {
        (LayoutKind::Std140, _) | (_, LayoutKind::Std140) => LayoutKind::Std140,
        (LayoutKind::Std430, _) | (_, LayoutKind::Std430) => LayoutKind::Std430,
        _ => LayoutKind::Scalar,
    }
}

fn round_up(size: u32, align: u32) -> Option<u32> {
    size.div_ceil(align.max(1)).checked_mul(align.max(1))
}

fn vector_align(bytes: u32, len: u8, layout: LayoutKind) -> u32 {
    if layout == LayoutKind::Scalar {
        bytes
    } else {
        bytes * if len == 3 { 4 } else { u32::from(len) }
    }
}

impl IrType {
    /// Fixed-size value's occupied size and base alignment under a memory layout.
    #[must_use]
    pub fn explicit_size_align(&self, layout: LayoutKind) -> Option<(u32, u32)> {
        match self {
            Self::Scalar(scalar) => Some((scalar.byte_size(), scalar.byte_size())),
            Self::Vector { scalar, len } if (2..=4).contains(len) => {
                let bytes = scalar.byte_size();
                Some((bytes * u32::from(*len), vector_align(bytes, *len, layout)))
            }
            Self::Matrix { scalar, cols, rows }
                if (2..=4).contains(cols) && (2..=4).contains(rows) =>
            {
                let bytes = scalar.byte_size();
                let mut align = vector_align(bytes, *rows, layout);
                if layout == LayoutKind::Std140 {
                    align = align.max(16);
                }
                Some((
                    round_up(bytes * u32::from(*rows), align)?.checked_mul(u32::from(*cols))?,
                    align,
                ))
            }
            Self::Array { element, len } if *len != 0 => {
                let (size, mut align) = element.explicit_size_align(layout)?;
                if layout == LayoutKind::Std140 {
                    align = align.max(16);
                }
                Some((round_up(size, align)?.checked_mul(*len)?, align))
            }
            Self::Struct(record) => {
                let record = relayout_struct(record, layout)?;
                Some((record.size, record.align))
            }
            Self::Pointer {
                storage: IrStorageClass::PhysicalStorageBuffer,
                ..
            } => Some((8, 8)),
            _ => None,
        }
    }

    /// Stride of a fixed-size array element in the surrounding block.
    #[must_use]
    pub fn explicit_array_stride(&self, layout: LayoutKind) -> Option<u32> {
        let (size, mut align) = self.explicit_size_align(layout)?;
        if layout == LayoutKind::Std140 {
            align = align.max(16);
        }
        round_up(size, align)
    }

    /// Column-major matrix stride, including arrays of matrices.
    #[must_use]
    pub fn explicit_matrix_stride(&self, layout: LayoutKind) -> Option<u32> {
        match self {
            Self::Matrix { scalar, rows, .. } => {
                let bytes = scalar.byte_size();
                let mut align = vector_align(bytes, *rows, layout);
                if layout == LayoutKind::Std140 {
                    align = align.max(16);
                }
                round_up(bytes * u32::from(*rows), align)
            }
            Self::Array { element, .. } | Self::RuntimeArray(element) => {
                element.explicit_matrix_stride(layout)
            }
            _ => None,
        }
    }

    /// Recomputes nested struct offsets for the enclosing block's layout rule.
    /// A pointer's pointee is a separate buffer object and is not relaid out.
    #[must_use]
    pub fn with_explicit_layout(&self, layout: LayoutKind) -> Option<Self> {
        match self {
            Self::Struct(record) => Some(Self::Struct(relayout_struct(record, layout)?)),
            Self::Array { element, len } if *len != 0 => {
                let element = Box::new(element.with_explicit_layout(layout)?);
                element.explicit_array_stride(layout)?.checked_mul(*len)?;
                Some(Self::Array { element, len: *len })
            }
            other => {
                other.explicit_size_align(layout)?;
                Some(other.clone())
            }
        }
    }
}

fn relayout_struct(record: &IrStructLayout, enclosing: LayoutKind) -> Option<IrStructLayout> {
    if record.fields.is_empty() {
        return None;
    }
    let layout = stricter(record.layout, enclosing);
    let mut cursor = 0u32;
    let mut align = if layout == LayoutKind::Std140 { 16 } else { 1 };
    let mut fields = Vec::with_capacity(record.fields.len());
    for field in &record.fields {
        let ty = field.ty.with_explicit_layout(layout)?;
        let (size, member_align) = ty.explicit_size_align(layout)?;
        let offset = round_up(cursor, member_align)?;
        cursor = offset.checked_add(size)?;
        align = align.max(member_align);
        fields.push(IrStructFieldLayout {
            name: field.name.clone(),
            byte_offset: offset,
            byte_size: size,
            ty,
        });
    }
    Some(IrStructLayout {
        name: record.name.clone(),
        fields,
        size: round_up(cursor, align)?,
        align,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScalarIrTy;

    #[test]
    fn fixed_array_and_matrix_strides_follow_enclosing_layout() {
        let scalar = IrType::Scalar(ScalarIrTy::F32);
        let vec3 = IrType::Vector {
            scalar: ScalarIrTy::F32,
            len: 3,
        };
        let matrix = IrType::Matrix {
            scalar: ScalarIrTy::F32,
            cols: 2,
            rows: 3,
        };
        assert_eq!(scalar.explicit_array_stride(LayoutKind::Std140), Some(16));
        assert_eq!(vec3.explicit_array_stride(LayoutKind::Scalar), Some(12));
        assert_eq!(vec3.explicit_array_stride(LayoutKind::Std430), Some(16));
        assert_eq!(matrix.explicit_matrix_stride(LayoutKind::Scalar), Some(12));
        assert_eq!(matrix.explicit_matrix_stride(LayoutKind::Std430), Some(16));
        assert_eq!(
            matrix.explicit_size_align(LayoutKind::Std140),
            Some((32, 16))
        );
        let small_matrix = IrType::Matrix {
            scalar: ScalarIrTy::F32,
            cols: 2,
            rows: 2,
        };
        assert_eq!(
            small_matrix.explicit_size_align(LayoutKind::Std430),
            Some((16, 8))
        );
        assert_eq!(
            small_matrix.explicit_size_align(LayoutKind::Std140),
            Some((32, 16))
        );
    }

    #[test]
    fn nested_struct_inherits_stricter_layout_and_preserves_field_order() {
        let field = |name: &str, ty| IrStructFieldLayout {
            name: name.into(),
            byte_offset: 0,
            byte_size: 0,
            ty,
        };
        let inner = IrType::Struct(IrStructLayout {
            name: "Inner".into(),
            fields: vec![field(
                "data",
                IrType::Array {
                    element: Box::new(IrType::Scalar(ScalarIrTy::F32)),
                    len: 2,
                },
            )],
            size: 0,
            align: 1,
            layout: LayoutKind::Scalar,
        });
        let outer = IrType::Struct(IrStructLayout {
            name: "Outer".into(),
            fields: vec![
                field("tag", IrType::Scalar(ScalarIrTy::U32)),
                field("inner", inner),
            ],
            size: 0,
            align: 1,
            layout: LayoutKind::Std140,
        });
        let IrType::Struct(record) = outer.with_explicit_layout(LayoutKind::Std430).unwrap() else {
            panic!("expected struct");
        };
        assert_eq!((record.size, record.align), (48, 16));
        assert_eq!(
            (record.fields[1].byte_offset, record.fields[1].byte_size),
            (16, 32)
        );
        let IrType::Struct(inner) = &record.fields[1].ty else {
            panic!("expected nested struct");
        };
        assert_eq!(inner.layout, LayoutKind::Std140);
        assert_eq!(inner.fields[0].byte_size, 32);
    }

    #[test]
    fn invalid_or_overflowing_records_have_no_layout() {
        let scalar = IrType::Scalar(ScalarIrTy::U64);
        let overflow = IrType::Array {
            element: Box::new(scalar.clone()),
            len: u32::MAX,
        };
        let zero = IrType::Array {
            element: Box::new(scalar),
            len: 0,
        };
        assert!(overflow.with_explicit_layout(LayoutKind::Std430).is_none());
        assert!(zero.with_explicit_layout(LayoutKind::Std430).is_none());
        assert!(
            IrType::Void
                .with_explicit_layout(LayoutKind::Std430)
                .is_none()
        );
    }
}
