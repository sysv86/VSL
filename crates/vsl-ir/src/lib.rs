//! Structured SSA intermediate representation and deterministic legalization passes.

pub mod passes;
pub mod ssa;

pub use passes::{
    ConstFoldPass, DeadCodeEliminationPass, IrPass, LowerBdaOffsetsPass, MAX_UNROLL_ITERATIONS,
    PassPipeline, PropagateNonUniformPass, SimplifyAndGvnPass, UnrollLoopsPass,
};
pub use ssa::{
    AtomicOrder, AtomicScope, BuiltinKind, GlslStd450Op, Inst, InstData, InstId, IrDescriptorVar,
    IrFunction, IrModule, IrPushConstantBlock, IrPushConstantField, IrWorkgroupVar, MemoryAccess,
    ScalarIrTy, StructuredStmt,
};
