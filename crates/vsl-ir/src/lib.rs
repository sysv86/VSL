//! Structured SSA intermediate representation and deterministic legalization passes.

pub mod passes;
pub mod ssa;

pub use passes::{
    ConstFoldPass, CopyPropPass, DeadCodeEliminationPass, DeadStoreElimPass, IrPass, LicmPass,
    LowerBdaOffsetsPass, MAX_INLINE_BODY_INSTS, MAX_UNROLL_BODY_INSTS, MAX_UNROLL_ITERATIONS,
    Mem2RegPass, PassPipeline, PropagateNonUniformPass, SelectiveInlinePass, SimplifyAndGvnPass,
    SroaPass, UnrollLoopsPass,
};
pub use ssa::{
    AtomicOrder, AtomicScope, BuiltinKind, CoopMatrixLayout, CoopMatrixScalar, CoopMatrixUse,
    GlslStd450Op, Inst, InstData, InstId, IrDescriptorVar, IrFunction, IrInterlockMode,
    IrInterpolation, IrLocalVar, IrModule, IrParam, IrPushConstantBlock,
    IrPushConstantField, IrStageExecMeta, IrStageIoVar, IrStageVar, IrStorageClass,
    IrStructFieldLayout, IrStructLayout, IrTessDomain, IrTessOrder, IrTessSpacing, IrTopology,
    IrType, IrWorkgroupVar, MemoryAccess, ScalarIrTy, SpecConstantOpKind, StructuredStmt,
};
