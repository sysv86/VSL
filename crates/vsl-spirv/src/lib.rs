//! Flat-arena direct SPIR-V 1.6 word emitter using official `spirv` 0.4 binary constants.

mod emitter;
mod rt_reflection;

pub use emitter::{SPIRV_VERSION_1_6, SpirvEmitter, emit_module_spirv};
pub use rt_reflection::reflect_ray_interfaces;
