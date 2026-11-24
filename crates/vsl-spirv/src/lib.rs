//! Flat-arena direct SPIR-V 1.6 word emitter using official `spirv` 0.4 binary constants.

mod emitter;

pub use emitter::{SPIRV_VERSION_1_6, SpirvEmitter, emit_module_spirv};
