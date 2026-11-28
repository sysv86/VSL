//! Reflection derived from exactly the IR variables and layout rules used for emission.

use vsl_ir::{IrFunction, IrStorageClass, IrStructLayout, IrType, LayoutKind, ScalarIrTy};
use vsl_mc::{
    RayInterfaceKind, RayInterfaceMember, RayInterfaceType, RayInterfaceVariable, RayScalar,
    ShaderRecordField, ShaderRecordLayout,
};

fn scalar(scalar: ScalarIrTy) -> RayScalar {
    match scalar {
        ScalarIrTy::Bool => RayScalar::Bool,
        ScalarIrTy::I32 => RayScalar::I32,
        ScalarIrTy::U32 => RayScalar::U32,
        ScalarIrTy::F16 => RayScalar::F16,
        ScalarIrTy::F32 => RayScalar::F32,
        ScalarIrTy::I64 => RayScalar::I64,
        ScalarIrTy::U64 => RayScalar::U64,
    }
}

fn value_type(ty: &IrType) -> RayInterfaceType {
    match ty {
        IrType::Scalar(s) => RayInterfaceType::Scalar { scalar: scalar(*s) },
        IrType::Vector { scalar: s, len } => RayInterfaceType::Vector {
            scalar: scalar(*s),
            len: *len,
        },
        IrType::Matrix {
            scalar: s,
            cols,
            rows,
        } => RayInterfaceType::Matrix {
            scalar: scalar(*s),
            columns: *cols,
            rows: *rows,
        },
        IrType::Array { element, len } => RayInterfaceType::Array {
            element: Box::new(value_type(element)),
            len: *len,
        },
        IrType::Struct(layout) => RayInterfaceType::Struct {
            name: layout.name.clone(),
            fields: layout
                .fields
                .iter()
                .map(|field| RayInterfaceMember {
                    name: field.name.clone(),
                    ty: value_type(&field.ty),
                })
                .collect(),
        },
        IrType::Pointer {
            pointee,
            storage: IrStorageClass::PhysicalStorageBuffer,
            ..
        } => RayInterfaceType::DeviceAddress {
            pointee: Box::new(value_type(pointee)),
        },
        _ => RayInterfaceType::Unsupported,
    }
}

fn array_strides(mut ty: &IrType, layout: LayoutKind) -> Vec<u32> {
    let mut strides = Vec::new();
    while let IrType::Array { element, .. } = ty {
        if let Some(stride) = element.explicit_array_stride(layout) {
            strides.push(stride);
        }
        ty = element;
    }
    strides
}

fn nested_layout(ty: &IrType) -> Option<Box<ShaderRecordLayout>> {
    match ty {
        IrType::Struct(layout) => Some(Box::new(record_layout(layout))),
        IrType::Array { element, .. } => nested_layout(element),
        _ => None,
    }
}

fn record_layout(layout: &IrStructLayout) -> ShaderRecordLayout {
    ShaderRecordLayout {
        byte_size: layout.size,
        alignment: layout.align,
        fields: layout
            .fields
            .iter()
            .map(|field| {
                let strides = array_strides(&field.ty, layout.layout);
                ShaderRecordField {
                    name: field.name.clone(),
                    byte_offset: field.byte_offset,
                    byte_size: field.byte_size,
                    array_stride: strides.first().copied(),
                    array_strides: strides,
                    matrix_stride: field.ty.explicit_matrix_stride(layout.layout),
                    nested: nested_layout(&field.ty),
                }
            })
            .collect(),
    }
}

/// Reflects only the entry point's RT variables; task payloads remain a separate ABI.
#[must_use]
pub fn reflect_ray_interfaces(function: &IrFunction) -> Vec<RayInterfaceVariable> {
    function
        .stage_vars
        .iter()
        .filter_map(|variable| {
            let kind = match variable.storage {
                IrStorageClass::RayPayloadKhr => RayInterfaceKind::RayPayload,
                IrStorageClass::IncomingRayPayloadKhr => RayInterfaceKind::IncomingRayPayload,
                IrStorageClass::CallableDataKhr => RayInterfaceKind::CallableData,
                IrStorageClass::IncomingCallableDataKhr => RayInterfaceKind::IncomingCallableData,
                IrStorageClass::HitAttributeKhr => RayInterfaceKind::HitAttribute,
                IrStorageClass::ShaderRecordBufferKhr => RayInterfaceKind::ShaderRecord,
                _ => return None,
            };
            let record_layout = match (&variable.ty, kind) {
                (IrType::Struct(layout), RayInterfaceKind::ShaderRecord) => {
                    Some(record_layout(layout))
                }
                _ => None,
            };
            Some(RayInterfaceVariable {
                name: variable.name.clone(),
                kind,
                location: variable.location,
                ty: value_type(&variable.ty),
                record_layout,
            })
        })
        .collect()
}
