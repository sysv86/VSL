use crate::ssa::{
    BuiltinKind, InstData, InstId, IrFunction, IrLocalVar, IrModule, IrStorageClass, IrType,
    ScalarIrTy, StructuredStmt,
};
use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;

/// Upper bound on loop iterations that [`UnrollLoopsPass`] will expand to prevent code-size blowup.
pub const MAX_UNROLL_ITERATIONS: u32 = 64;

/// Upper bound on loop body instructions that [`UnrollLoopsPass`] will expand.
pub const MAX_UNROLL_BODY_INSTS: usize = 128;

/// Upper bound on helper function instructions that [`SelectiveInlinePass`] will inline.
pub const MAX_INLINE_BODY_INSTS: usize = 64;

/// Deterministic transformation or legalization pass over an [`IrFunction`].
pub trait IrPass {
    fn name(&self) -> &'static str;
    fn run(&self, func: &mut IrFunction);
}

/// Folds constant integer, finite `f32`, bitwise, logical, cast, comparison, `Select`, and
/// constant `StructuredStmt::If` operations into simplified SSA instructions and control flow
/// until a fixed point is reached.
pub struct ConstFoldPass;

impl IrPass for ConstFoldPass {
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut changed = true;
        while changed {
            changed = false;
            let mut select_remap: IndexMap<InstId, InstId> = IndexMap::new();
            let ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
            for id in ids {
                let inst_data = func.insts[id].data.clone();
                let folded = match inst_data {
                    InstData::IAdd32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x.wrapping_add(*y)))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x.wrapping_add(*y)))
                        }
                        _ => fold_vec_int_binop(func, a, b, i32::wrapping_add, u32::wrapping_add),
                    },
                    InstData::ISub32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x.wrapping_sub(*y)))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x.wrapping_sub(*y)))
                        }
                        _ => fold_vec_int_binop(func, a, b, i32::wrapping_sub, u32::wrapping_sub),
                    },
                    InstData::IMul32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x.wrapping_mul(*y)))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x.wrapping_mul(*y)))
                        }
                        _ => fold_vec_int_binop(func, a, b, i32::wrapping_mul, u32::wrapping_mul),
                    },
                    InstData::IDiv32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            x.checked_div(*y).map(InstData::ConstI32)
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            x.checked_div(*y).map(InstData::ConstU32)
                        }
                        _ => None,
                    },
                    InstData::IRem32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            x.checked_rem(*y).map(InstData::ConstI32)
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            x.checked_rem(*y).map(InstData::ConstU32)
                        }
                        _ => None,
                    },
                    InstData::FAdd32(a, b) => fold_float_or_vec_binop(func, a, b, |x, y| x + y),
                    InstData::FSub32(a, b) => fold_float_or_vec_binop(func, a, b, |x, y| x - y),
                    InstData::FMul32(a, b) => fold_float_or_vec_binop(func, a, b, |x, y| x * y),
                    InstData::FDiv32(a, b) => fold_float_or_vec_binop(func, a, b, |x, y| x / y),
                    #[allow(clippy::float_cmp)]
                    InstData::FCmpEq(a, b) => fold_f32_cmp(func, a, b, |x, y| x == y),
                    #[allow(clippy::float_cmp)]
                    InstData::FCmpNe(a, b) => fold_f32_cmp(func, a, b, |x, y| x != y),
                    InstData::FCmpLt(a, b) => fold_f32_cmp(func, a, b, |x, y| x < y),
                    InstData::FCmpLe(a, b) => fold_f32_cmp(func, a, b, |x, y| x <= y),
                    InstData::FCmpGt(a, b) => fold_f32_cmp(func, a, b, |x, y| x > y),
                    InstData::FCmpGe(a, b) => fold_f32_cmp(func, a, b, |x, y| x >= y),
                    InstData::ICmpEq(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstBool(x == y))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstBool(x == y))
                        }
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstBool(x == y))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstBool(x == y))
                        }
                        _ => None,
                    },
                    InstData::ICmpNe(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstBool(x != y))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstBool(x != y))
                        }
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstBool(x != y))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstBool(x != y))
                        }
                        _ => None,
                    },
                    InstData::ICmpLt(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstBool(x < y))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstBool(x < y))
                        }
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstBool(x < y))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstBool(x < y))
                        }
                        _ => None,
                    },
                    InstData::ICmpLe(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstBool(x <= y))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstBool(x <= y))
                        }
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstBool(x <= y))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstBool(x <= y))
                        }
                        _ => None,
                    },
                    InstData::ICmpGt(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstBool(x > y))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstBool(x > y))
                        }
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstBool(x > y))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstBool(x > y))
                        }
                        _ => None,
                    },
                    InstData::ICmpGe(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstBool(x >= y))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstBool(x >= y))
                        }
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstBool(x >= y))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstBool(x >= y))
                        }
                        _ => None,
                    },
                    InstData::UCmpLt(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstBool(x < y))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstBool(x < y))
                        }
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstBool(x.cast_unsigned() < y.cast_unsigned()))
                        }
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstBool(x.cast_unsigned() < y.cast_unsigned()))
                        }
                        _ => None,
                    },
                    InstData::LogicalAnd(a, b) => {
                        if let (InstData::ConstBool(x), InstData::ConstBool(y)) =
                            (&func.insts[a].data, &func.insts[b].data)
                        {
                            Some(InstData::ConstBool(*x && *y))
                        } else {
                            None
                        }
                    }
                    InstData::LogicalOr(a, b) => {
                        if let (InstData::ConstBool(x), InstData::ConstBool(y)) =
                            (&func.insts[a].data, &func.insts[b].data)
                        {
                            Some(InstData::ConstBool(*x || *y))
                        } else {
                            None
                        }
                    }
                    InstData::LogicalNot(a) => {
                        if let InstData::ConstBool(x) = func.insts[a].data {
                            Some(InstData::ConstBool(!x))
                        } else {
                            None
                        }
                    }
                    InstData::Select {
                        cond,
                        on_true,
                        on_false,
                    } => {
                        if let InstData::ConstBool(flag) = func.insts[cond].data {
                            let chosen = if flag { on_true } else { on_false };
                            if is_const_inst(&func.insts[chosen].data) {
                                Some(func.insts[chosen].data.clone())
                            } else {
                                select_remap.insert(id, chosen);
                                None
                            }
                        } else {
                            None
                        }
                    }
                    InstData::BitAnd32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x & y))
                        }
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x & y))
                        }
                        _ => None,
                    },
                    InstData::BitOr32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x | y))
                        }
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x | y))
                        }
                        _ => None,
                    },
                    InstData::BitXor32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x ^ y))
                        }
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x ^ y))
                        }
                        _ => None,
                    },
                    InstData::BitNot32(a) => match func.insts[a].data {
                        InstData::ConstU32(x) => Some(InstData::ConstU32(!x)),
                        InstData::ConstI32(x) => Some(InstData::ConstI32(!x)),
                        _ => None,
                    },
                    InstData::Shl32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstU32(x), InstData::ConstU32(y)) if *y < 32 => {
                            Some(InstData::ConstU32(x.wrapping_shl(*y)))
                        }
                        (InstData::ConstI32(x), InstData::ConstI32(y)) if (0..32).contains(y) => {
                            Some(InstData::ConstI32(x.wrapping_shl(y.cast_unsigned())))
                        }
                        (InstData::ConstI32(x), InstData::ConstU32(y)) if *y < 32 => {
                            Some(InstData::ConstI32(x.wrapping_shl(*y)))
                        }
                        _ => None,
                    },
                    InstData::Shr32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstU32(x), InstData::ConstU32(y)) if *y < 32 => {
                            Some(InstData::ConstU32(x.wrapping_shr(*y)))
                        }
                        (InstData::ConstI32(x), InstData::ConstI32(y)) if (0..32).contains(y) => {
                            Some(InstData::ConstI32(x.wrapping_shr(y.cast_unsigned())))
                        }
                        (InstData::ConstI32(x), InstData::ConstU32(y)) if *y < 32 => {
                            Some(InstData::ConstI32(x.wrapping_shr(*y)))
                        }
                        _ => None,
                    },
                    InstData::CastI64ToU64(a) => match func.insts[a].data {
                        InstData::ConstI64(x) => Some(InstData::ConstU64(x.cast_unsigned())),
                        InstData::ConstU64(x) => Some(InstData::ConstU64(x)),
                        _ => None,
                    },
                    InstData::CastToU32(a) => match func.insts[a].data {
                        InstData::ConstU32(x) => Some(InstData::ConstU32(x)),
                        InstData::ConstI32(x) => Some(InstData::ConstU32(x.cast_unsigned())),
                        InstData::ConstU64(x) => Some(InstData::ConstU32(x as u32)),
                        InstData::ConstI64(x) => Some(InstData::ConstU32(x.cast_unsigned() as u32)),
                        InstData::ConstBool(b) => Some(InstData::ConstU32(u32::from(b))),
                        InstData::ConstF32(bits) => {
                            let f = f32::from_bits(bits);
                            (f.is_finite() && (0.0..4_294_967_296.0).contains(&f)).then(|| {
                                #[allow(clippy::cast_sign_loss)]
                                let truncated = f as u32;
                                InstData::ConstU32(truncated)
                            })
                        }
                        _ => None,
                    },
                    InstData::IAdd64(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstI64(x.wrapping_add(*y)))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstU64(x.wrapping_add(*y)))
                        }
                        _ => None,
                    },
                    InstData::IMul64(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI64(x), InstData::ConstI64(y)) => {
                            Some(InstData::ConstI64(x.wrapping_mul(*y)))
                        }
                        (InstData::ConstU64(x), InstData::ConstU64(y)) => {
                            Some(InstData::ConstU64(x.wrapping_mul(*y)))
                        }
                        _ => None,
                    },
                    InstData::FConvert { value, target } => match (&func.insts[value].data, target)
                    {
                        (InstData::ConstF32(bits), ScalarIrTy::F16) => {
                            Some(InstData::ConstF16(f32_to_f16_bits(f32::from_bits(*bits))))
                        }
                        (InstData::ConstF16(bits), ScalarIrTy::F32) => {
                            Some(InstData::ConstF32(f16_bits_to_f32(*bits).to_bits()))
                        }
                        (InstData::ConstF32(bits), ScalarIrTy::F32) => {
                            Some(InstData::ConstF32(*bits))
                        }
                        (InstData::ConstF16(bits), ScalarIrTy::F16) => {
                            Some(InstData::ConstF16(*bits))
                        }
                        _ => None,
                    },
                    InstData::VectorTimesScalar { vec, scalar } => {
                        fold_vec_times_scalar(func, vec, scalar)
                    }
                    InstData::ExtractComponent { vec, index } => {
                        if let Some(elems) = extract_composite_elems(func, vec)
                            && let Some(&elem) = elems.get(index as usize)
                        {
                            if is_const_inst(&func.insts[elem].data) {
                                Some(func.insts[elem].data.clone())
                            } else {
                                select_remap.insert(id, elem);
                                None
                            }
                        } else {
                            None
                        }
                    }
                    InstData::CompositeExtract { composite, indices } => {
                        if let Some(elem) = resolve_composite_extract(func, composite, &indices) {
                            if is_const_inst(&func.insts[elem].data) {
                                Some(func.insts[elem].data.clone())
                            } else {
                                select_remap.insert(id, elem);
                                None
                            }
                        } else {
                            None
                        }
                    }
                    InstData::VectorShuffle {
                        lhs,
                        rhs,
                        components,
                    } => fold_vector_shuffle(func, id, lhs, rhs, &components, &mut select_remap),
                    InstData::ConstBool(_)
                    | InstData::ConstI32(_)
                    | InstData::ConstU32(_)
                    | InstData::ConstF32(_)
                    | InstData::ConstF16(_)
                    | InstData::ConstI64(_)
                    | InstData::ConstU64(_)
                    | InstData::Phi { .. }
                    | InstData::LocalAddr { .. }
                    | InstData::LocalLoad { .. }
                    | InstData::LocalStore { .. }
                    | InstData::AccessChain { .. }
                    | InstData::DescriptorAccessChain { .. }
                    | InstData::DescriptorStore { .. }
                    | InstData::Param { .. }
                    | InstData::FunctionCall { .. }
                    | InstData::BdaOffset { .. }
                    | InstData::ConvertPtrToU(_)
                    | InstData::ConvertUToPtr { .. }
                    | InstData::BdaLoad { .. }
                    | InstData::BdaStore { .. }
                    | InstData::ImageSample { .. }
                    | InstData::ImageSampleImplicitLod { .. }
                    | InstData::ImageSampleGrad { .. }
                    | InstData::ImageSampleDrefExplicitLod { .. }
                    | InstData::ImageGather { .. }
                    | InstData::ImageQuerySize { .. }
                    | InstData::ImageQueryLevels { .. }
                    | InstData::ImageTexelPointer { .. }
                    | InstData::ImageStore { .. }
                    | InstData::WorkgroupAddr { .. }
                    | InstData::WorkgroupLoad { .. }
                    | InstData::WorkgroupStore { .. }
                    | InstData::NonUniform(_)
                    | InstData::ResourceAccessChain { .. }
                    | InstData::ResourceLoad { .. }
                    | InstData::AtomicAdd { .. }
                    | InstData::AtomicMin { .. }
                    | InstData::AtomicMax { .. }
                    | InstData::AtomicCas { .. }
                    | InstData::WorkgroupBarrier
                    | InstData::MemoryBarrier { .. }
                    | InstData::SubgroupBallot(_)
                    | InstData::SubgroupBallotBitCount(_)
                    | InstData::SubgroupBallotExclusiveBitCount(_)
                    | InstData::SubgroupBallotInclusiveBitCount(_)
                    | InstData::SubgroupBroadcastFirst(_)
                    | InstData::SubgroupAdd(_)
                    | InstData::SubgroupMin(_)
                    | InstData::SubgroupMax(_)
                    | InstData::SubgroupExclusiveAdd(_)
                    | InstData::SubgroupInclusiveAdd(_)
                    | InstData::SubgroupShuffle { .. }
                    | InstData::SubgroupShuffleXor { .. }
                    | InstData::SubgroupAny(_)
                    | InstData::SubgroupAll(_)
                    | InstData::SubgroupElect
                    | InstData::QuadSwapHorizontal(_)
                    | InstData::QuadSwapVertical(_)
                    | InstData::QuadSwapDiagonal(_)
                    | InstData::BitFieldExtract { .. }
                    | InstData::BitFieldInsert { .. }
                    | InstData::BitCount(_)
                    | InstData::Bitcast { .. }
                    | InstData::IsNan(_)
                    | InstData::IsInf(_)
                    | InstData::DPdx(_)
                    | InstData::DPdy(_)
                    | InstData::Fwidth(_)
                    | InstData::SpecConstant { .. }
                    | InstData::SpecConstantOp { .. }
                    | InstData::DescriptorLoad { .. }
                    | InstData::GlslStd450 { .. }
                    | InstData::MatrixTimesVector { .. }
                    | InstData::MatrixTimesMatrix { .. }
                    | InstData::CompositeConstruct { .. }
                    | InstData::CompositeInsert { .. }
                    | InstData::CastToI32(_)
                    | InstData::CastToF32(_)
                    | InstData::CastToI64(_)
                    | InstData::CastToU64(_)
                    | InstData::MakeVec2(_, _)
                    | InstData::MakeVec3(_, _, _)
                    | InstData::MakeVec4(_, _, _, _)
                    | InstData::LoadBuiltin { .. }
                    | InstData::LoadStageInput { .. }
                    | InstData::StoreStageOutput { .. }
                    | InstData::StoreBuiltin { .. }
                    | InstData::DemoteToHelperInvocation
                    | InstData::LoadPushConstant { .. }
                    | InstData::RayQueryInitialize { .. }
                    | InstData::RayQueryProceed(_)
                    | InstData::RayQueryCommittedType(_)
                    | InstData::RayQueryCandidateType(_)
                    | InstData::RayQueryConfirmIntersection(_)
                    | InstData::RayQueryTerminate(_)
                    | InstData::RayQueryGenerateIntersection { .. }
                    | InstData::RayQueryCommittedT(_)
                    | InstData::RayQueryCandidateT(_)
                    | InstData::RayQueryCommittedInstanceId(_)
                    | InstData::RayQueryCommittedInstanceCustomIndex(_)
                    | InstData::RayQueryCommittedPrimitiveIndex(_)
                    | InstData::RayQueryCommittedGeometryIndex(_)
                    | InstData::RayQueryCommittedObjectToWorld(_)
                    | InstData::RayQueryCommittedWorldToObject(_)
                    | InstData::RayQueryCommittedFrontFace(_)
                    | InstData::RayQueryCommittedBarycentrics(_)
                    | InstData::TraceRayKhr { .. }
                    | InstData::IgnoreIntersectionKhr
                    | InstData::TerminateRayKhr
                    | InstData::ReportIntersectionKhr { .. }
                    | InstData::ExecuteCallableKhr { .. }
                    | InstData::EmitMeshTasksExt { .. }
                    | InstData::SetMeshOutputsExt { .. }
                    | InstData::EmitVertex
                    | InstData::EndPrimitive
                    | InstData::LoadStageVar { .. }
                    | InstData::StoreStageVar { .. }
                    | InstData::HitObjectTraceRay { .. }
                    | InstData::HitObjectRecordEmpty(_)
                    | InstData::HitObjectIsHit(_)
                    | InstData::HitObjectGetRayTMax(_)
                    | InstData::HitObjectGetSbtRecordIndex(_)
                    | InstData::HitObjectExecuteShader { .. }
                    | InstData::ReorderThreadWithHint { .. }
                    | InstData::ReorderThreadWithHitObject { .. }
                    | InstData::RayQueryTriangleVertexPositions { .. }
                    | InstData::Nop => None,
                };
                if let Some(new_data) = folded {
                    func.insts[id].data = new_data;
                    changed = true;
                }
            }

            if !select_remap.is_empty() {
                let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
                for id in all_ids {
                    if select_remap.contains_key(&id) {
                        func.insts[id].data = InstData::Nop;
                        func.insts[id].non_uniform = false;
                    } else {
                        let remapped = remap_inst_data(&func.insts[id].data, &select_remap);
                        func.insts[id].data = remapped;
                    }
                }
                let body = std::mem::take(&mut func.body);
                func.body = remap_stmts_ids(body, &select_remap);
                changed = true;
            }

            let mut folded_if = false;
            let mut phi_remap: IndexMap<InstId, InstId> = IndexMap::new();
            let body = std::mem::take(&mut func.body);
            func.body = fold_const_ifs(func, body, &mut folded_if, &mut phi_remap);
            if !phi_remap.is_empty() {
                let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
                for id in all_ids {
                    if phi_remap.contains_key(&id) {
                        func.insts[id].data = InstData::Nop;
                        func.insts[id].non_uniform = false;
                    } else {
                        let remapped = remap_inst_data(&func.insts[id].data, &phi_remap);
                        func.insts[id].data = remapped;
                    }
                }
                let body = std::mem::take(&mut func.body);
                func.body = remap_stmts_ids(body, &phi_remap);
                changed = true;
            }
            if folded_if {
                changed = true;
            }
        }
    }
}

/// Converts an IEEE-754 binary16 (`f16`) bit pattern to `f32` in pure Rust.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x03FF);
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let f32_exp = (e + 127).cast_unsigned() << 23;
            sign | f32_exp | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        let f32_exp = (exp + 112) << 23;
        sign | f32_exp | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Converts an `f32` value to an IEEE-754 binary16 (`f16`) bit pattern using round-to-nearest-even.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF).cast_signed();
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        if mant == 0 {
            return sign | 0x7C00;
        }
        let nan_mant = ((mant >> 13) as u16).max(1);
        return sign | 0x7C00 | nan_mant;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    if unbiased < -25 {
        return sign;
    }
    if unbiased == -25 {
        // Halfway to the smallest f16 subnormal (2^-24): round-to-nearest-even rounds
        // values strictly above 2^-25 (mant != 0) up to 0x0001 and exact 2^-25 to 0.
        return if mant != 0 { sign | 0x0001 } else { sign };
    }
    if unbiased < -14 {
        let shift = (-14 - unbiased).cast_unsigned();
        let full_mant = mant | 0x0080_0000;
        let shifted = full_mant >> (shift + 13);
        let round_bit = (full_mant >> (shift + 12)) & 1;
        let sticky = u32::from((full_mant & ((1u32 << (shift + 12)) - 1)) != 0);
        let rounded = shifted + u32::from(round_bit != 0 && (sticky != 0 || (shifted & 1) != 0));
        return sign | (rounded as u16);
    }

    let f16_exp = (unbiased + 15).cast_unsigned() << 10;
    let shifted_mant = mant >> 13;
    let round_bit = (mant >> 12) & 1;
    let sticky = u32::from((mant & 0x0FFF) != 0);
    let mut combined = f16_exp | shifted_mant;
    if round_bit != 0 && (sticky != 0 || (shifted_mant & 1) != 0) {
        combined += 1;
    }
    sign | (combined as u16)
}

const fn is_f16_finite(bits: u16) -> bool {
    (bits & 0x7C00) != 0x7C00
}

fn fold_f32_cmp(
    func: &IrFunction,
    a: InstId,
    b: InstId,
    op: impl FnOnce(f32, f32) -> bool,
) -> Option<InstData> {
    match (&func.insts[a].data, &func.insts[b].data) {
        (InstData::ConstF32(xb), InstData::ConstF32(yb)) => {
            let x = f32::from_bits(*xb);
            let y = f32::from_bits(*yb);
            (x.is_finite() && y.is_finite()).then(|| InstData::ConstBool(op(x, y)))
        }
        (InstData::ConstF16(xb), InstData::ConstF16(yb)) => {
            if !(is_f16_finite(*xb) && is_f16_finite(*yb)) {
                return None;
            }
            let x = f16_bits_to_f32(*xb);
            let y = f16_bits_to_f32(*yb);
            (x.is_finite() && y.is_finite()).then(|| InstData::ConstBool(op(x, y)))
        }
        _ => None,
    }
}

fn extract_composite_elems(func: &IrFunction, id: InstId) -> Option<Vec<InstId>> {
    match &func.insts[id].data {
        InstData::MakeVec2(a, b) => Some(vec![*a, *b]),
        InstData::MakeVec3(a, b, c) => Some(vec![*a, *b, *c]),
        InstData::MakeVec4(a, b, c, d) => Some(vec![*a, *b, *c, *d]),
        InstData::CompositeConstruct { elems } => Some(elems.clone()),
        _ => None,
    }
}

fn resolve_composite_extract(
    func: &IrFunction,
    mut current: InstId,
    indices: &[u32],
) -> Option<InstId> {
    if indices.is_empty() {
        return None;
    }
    for &idx in indices {
        let elems = extract_composite_elems(func, current)?;
        current = *elems.get(idx as usize)?;
    }
    Some(current)
}

fn build_vec_inst_data(elems: Vec<InstId>) -> InstData {
    match elems.as_slice() {
        [a, b] => InstData::MakeVec2(*a, *b),
        [a, b, c] => InstData::MakeVec3(*a, *b, *c),
        [a, b, c, d] => InstData::MakeVec4(*a, *b, *c, *d),
        _ => InstData::CompositeConstruct { elems },
    }
}

fn fold_float_or_vec_binop(
    func: &mut IrFunction,
    a: InstId,
    b: InstId,
    op: impl Fn(f32, f32) -> f32,
) -> Option<InstData> {
    match (&func.insts[a].data, &func.insts[b].data) {
        (InstData::ConstF32(x_bits), InstData::ConstF32(y_bits)) => {
            let x = f32::from_bits(*x_bits);
            let y = f32::from_bits(*y_bits);
            let res = op(x, y);
            (x.is_finite() && y.is_finite() && res.is_finite()).then(|| InstData::const_f32(res))
        }
        (InstData::ConstF16(x_bits), InstData::ConstF16(y_bits)) => {
            if !(is_f16_finite(*x_bits) && is_f16_finite(*y_bits)) {
                return None;
            }
            let x = f16_bits_to_f32(*x_bits);
            let y = f16_bits_to_f32(*y_bits);
            let res = op(x, y);
            if !(x.is_finite() && y.is_finite() && res.is_finite()) {
                return None;
            }
            let res_bits = f32_to_f16_bits(res);
            is_f16_finite(res_bits).then_some(InstData::ConstF16(res_bits))
        }
        _ => {
            let lhs_elems = extract_composite_elems(func, a)?;
            let rhs_elems = extract_composite_elems(func, b)?;
            if lhs_elems.is_empty() || lhs_elems.len() != rhs_elems.len() {
                return None;
            }
            let mut folded_lanes = Vec::with_capacity(lhs_elems.len());
            for (&la, &rb) in lhs_elems.iter().zip(rhs_elems.iter()) {
                let lane_data = match (&func.insts[la].data, &func.insts[rb].data) {
                    (InstData::ConstF32(xb), InstData::ConstF32(yb)) => {
                        let x = f32::from_bits(*xb);
                        let y = f32::from_bits(*yb);
                        let res = op(x, y);
                        if !(x.is_finite() && y.is_finite() && res.is_finite()) {
                            return None;
                        }
                        InstData::const_f32(res)
                    }
                    (InstData::ConstF16(xb), InstData::ConstF16(yb)) => {
                        if !(is_f16_finite(*xb) && is_f16_finite(*yb)) {
                            return None;
                        }
                        let x = f16_bits_to_f32(*xb);
                        let y = f16_bits_to_f32(*yb);
                        let res = op(x, y);
                        if !(x.is_finite() && y.is_finite() && res.is_finite()) {
                            return None;
                        }
                        let res_bits = f32_to_f16_bits(res);
                        if !is_f16_finite(res_bits) {
                            return None;
                        }
                        InstData::ConstF16(res_bits)
                    }
                    _ => return None,
                };
                folded_lanes.push(lane_data);
            }
            let lane_ids: Vec<InstId> = folded_lanes
                .into_iter()
                .map(|d| func.emit(None, d))
                .collect();
            Some(build_vec_inst_data(lane_ids))
        }
    }
}

fn fold_vec_int_binop(
    func: &mut IrFunction,
    a: InstId,
    b: InstId,
    op_i: impl Fn(i32, i32) -> i32,
    op_u: impl Fn(u32, u32) -> u32,
) -> Option<InstData> {
    let lhs_elems = extract_composite_elems(func, a)?;
    let rhs_elems = extract_composite_elems(func, b)?;
    if lhs_elems.is_empty() || lhs_elems.len() != rhs_elems.len() {
        return None;
    }
    let mut folded_lanes = Vec::with_capacity(lhs_elems.len());
    for (&la, &rb) in lhs_elems.iter().zip(rhs_elems.iter()) {
        let lane_data = match (&func.insts[la].data, &func.insts[rb].data) {
            (InstData::ConstI32(x), InstData::ConstI32(y)) => InstData::ConstI32(op_i(*x, *y)),
            (InstData::ConstU32(x), InstData::ConstU32(y)) => InstData::ConstU32(op_u(*x, *y)),
            _ => return None,
        };
        folded_lanes.push(lane_data);
    }
    let lane_ids: Vec<InstId> = folded_lanes
        .into_iter()
        .map(|d| func.emit(None, d))
        .collect();
    Some(build_vec_inst_data(lane_ids))
}

fn fold_vec_times_scalar(func: &mut IrFunction, vec: InstId, scalar: InstId) -> Option<InstData> {
    let elems = extract_composite_elems(func, vec)?;
    if elems.is_empty() {
        return None;
    }
    let mut folded_lanes = Vec::with_capacity(elems.len());
    match func.insts[scalar].data {
        InstData::ConstF32(sb) => {
            let s = f32::from_bits(sb);
            if !s.is_finite() {
                return None;
            }
            for &e in &elems {
                let InstData::ConstF32(eb) = func.insts[e].data else {
                    return None;
                };
                let x = f32::from_bits(eb);
                let res = x * s;
                if !(x.is_finite() && res.is_finite()) {
                    return None;
                }
                folded_lanes.push(InstData::const_f32(res));
            }
        }
        InstData::ConstF16(sb) => {
            if !is_f16_finite(sb) {
                return None;
            }
            let s = f16_bits_to_f32(sb);
            if !s.is_finite() {
                return None;
            }
            for &e in &elems {
                let InstData::ConstF16(eb) = func.insts[e].data else {
                    return None;
                };
                if !is_f16_finite(eb) {
                    return None;
                }
                let x = f16_bits_to_f32(eb);
                let res = x * s;
                if !(x.is_finite() && res.is_finite()) {
                    return None;
                }
                let res_bits = f32_to_f16_bits(res);
                if !is_f16_finite(res_bits) {
                    return None;
                }
                folded_lanes.push(InstData::ConstF16(res_bits));
            }
        }
        _ => return None,
    }
    let lane_ids: Vec<InstId> = folded_lanes
        .into_iter()
        .map(|d| func.emit(None, d))
        .collect();
    Some(build_vec_inst_data(lane_ids))
}

fn fold_vector_shuffle(
    func: &IrFunction,
    id: InstId,
    lhs: InstId,
    rhs: InstId,
    components: &[u32],
    select_remap: &mut IndexMap<InstId, InstId>,
) -> Option<InstData> {
    let lhs_elems = extract_composite_elems(func, lhs)?;
    let rhs_elems = if lhs == rhs {
        lhs_elems.clone()
    } else {
        extract_composite_elems(func, rhs)?
    };
    let mut selected = Vec::with_capacity(components.len());
    for &c in components {
        let idx = c as usize;
        if idx < lhs_elems.len() {
            selected.push(lhs_elems[idx]);
        } else if let Some(&elem) = rhs_elems.get(idx - lhs_elems.len()) {
            selected.push(elem);
        } else {
            return None;
        }
    }
    if selected.len() == 1 {
        let only = selected[0];
        if is_const_inst(&func.insts[only].data) {
            return Some(func.insts[only].data.clone());
        }
        select_remap.insert(id, only);
        return None;
    }
    Some(build_vec_inst_data(selected))
}

const fn is_const_inst(data: &InstData) -> bool {
    matches!(
        data,
        InstData::ConstBool(_)
            | InstData::ConstI32(_)
            | InstData::ConstU32(_)
            | InstData::ConstF32(_)
            | InstData::ConstF16(_)
            | InstData::ConstI64(_)
            | InstData::ConstU64(_)
    )
}

fn remap_stmts_ids(
    stmts: Vec<StructuredStmt>,
    remap: &IndexMap<InstId, InstId>,
) -> Vec<StructuredStmt> {
    stmts
        .into_iter()
        .map(|stmt| match stmt {
            StructuredStmt::Eval(id) => StructuredStmt::Eval(remap_id(id, remap)),
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => StructuredStmt::If {
                cond: remap_id(cond, remap),
                then_body: remap_stmts_ids(then_body, remap),
                else_body: remap_stmts_ids(else_body, remap),
            },
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => StructuredStmt::Switch {
                selector: remap_id(selector, remap),
                cases: cases
                    .into_iter()
                    .map(|(k, b)| (k, remap_stmts_ids(b, remap)))
                    .collect(),
                default: remap_stmts_ids(default, remap),
            },
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => StructuredStmt::Loop {
                body: remap_stmts_ids(body, remap),
                continuing: remap_stmts_ids(continuing, remap),
                unroll,
                trip_count,
                induction_init: induction_init.map(|id| remap_id(id, remap)),
                induction_step,
            },
            StructuredStmt::Return(val) => {
                StructuredStmt::Return(val.map(|id| remap_id(id, remap)))
            }
            other @ (StructuredStmt::Break | StructuredStmt::Continue) => other,
        })
        .collect()
}

fn phi_selected_for_const_if(func: &IrFunction, phi_id: InstId, taken_tag: u32) -> Option<InstId> {
    let InstData::Phi { incoming } = &func.insts[phi_id].data else {
        return None;
    };
    if incoming.is_empty() || incoming.len() > 2 || !incoming.iter().all(|&(_, tag)| tag <= 1) {
        return None;
    }
    if incoming
        .iter()
        .any(|&(v, _)| v == phi_id || can_reach_inst(func, v, phi_id))
    {
        return None;
    }
    incoming
        .iter()
        .find(|&&(_, tag)| tag == taken_tag)
        .map(|&(selected, _)| selected)
}

fn fold_const_ifs(
    func: &IrFunction,
    stmts: Vec<StructuredStmt>,
    folded_any: &mut bool,
    phi_remap: &mut IndexMap<InstId, InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut iter = stmts.into_iter();
    while let Some(stmt) = iter.next() {
        match stmt {
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                if let InstData::ConstBool(flag) = func.insts[cond].data {
                    *folded_any = true;
                    let taken_tag = u32::from(!flag);
                    let taken = if flag { then_body } else { else_body };
                    out.extend(fold_const_ifs(func, taken, folded_any, phi_remap));

                    // Only resolve merge-block `Eval(Phi)` statements immediately following
                    // the folded `If`, stopping at the first non-`Phi` statement so unrelated
                    // prior or later runtime `If`/`Loop` `Phi`s are never clobbered.
                    while let Some(StructuredStmt::Eval(phi_id)) = iter.as_slice().first() {
                        let phi_id = *phi_id;
                        if let Some(selected) = phi_selected_for_const_if(func, phi_id, taken_tag) {
                            let _ = iter.next();
                            phi_remap.insert(phi_id, selected);
                        } else {
                            break;
                        }
                    }
                } else {
                    out.push(StructuredStmt::If {
                        cond,
                        then_body: fold_const_ifs(func, then_body, folded_any, phi_remap),
                        else_body: fold_const_ifs(func, else_body, folded_any, phi_remap),
                    });
                }
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: cases
                        .into_iter()
                        .map(|(k, b)| (k, fold_const_ifs(func, b, folded_any, phi_remap)))
                        .collect(),
                    default: fold_const_ifs(func, default, folded_any, phi_remap),
                });
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                out.push(StructuredStmt::Loop {
                    body: fold_const_ifs(func, body, folded_any, phi_remap),
                    continuing: fold_const_ifs(func, continuing, folded_any, phi_remap),
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });
            }
            other => out.push(other),
        }
    }
    out
}

/// Unrolls `StructuredStmt::Loop { unroll: true, trip_count: Some(n), .. }` when `n <= 64`,
/// replacing the loop with `n` cloned iterations with freshly allocated SSA instruction IDs
/// for instructions defined inside the loop and stepped induction variables, while leaving
/// external instructions defined before the loop untouched.
pub struct UnrollLoopsPass;

impl IrPass for UnrollLoopsPass {
    fn name(&self) -> &'static str {
        "unroll-loops"
    }

    fn run(&self, func: &mut IrFunction) {
        let body = std::mem::take(&mut func.body);
        let ancestor_outside = IndexSet::new();
        let mut post_remap = IndexMap::new();
        let new_body = unroll_stmts(func, body, &ancestor_outside, &mut post_remap);
        func.body = if post_remap.is_empty() {
            new_body
        } else {
            remap_stmts_ids(new_body, &post_remap)
        };
    }
}

fn collect_stmt_direct_refs(stmts: &[StructuredStmt], out: &mut IndexSet<InstId>) {
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) | StructuredStmt::Return(Some(id)) => {
                out.insert(*id);
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                out.insert(*cond);
                collect_stmt_direct_refs(then_body, out);
                collect_stmt_direct_refs(else_body, out);
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                out.insert(*selector);
                for (_, body) in cases {
                    collect_stmt_direct_refs(body, out);
                }
                collect_stmt_direct_refs(default, out);
            }
            StructuredStmt::Loop {
                body,
                continuing,
                induction_init,
                ..
            } => {
                if let Some(ind_id) = induction_init {
                    out.insert(*ind_id);
                }
                collect_stmt_direct_refs(body, out);
                collect_stmt_direct_refs(continuing, out);
            }
            StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(None) => {}
        }
    }
}

fn collect_stmt_transitive_refs(
    func: &IrFunction,
    stmts: &[StructuredStmt],
    out: &mut IndexSet<InstId>,
) {
    let mut worklist: Vec<InstId> = Vec::new();
    let mut direct = IndexSet::new();
    collect_stmt_direct_refs(stmts, &mut direct);
    for id in direct {
        if out.insert(id) {
            worklist.push(id);
        }
    }
    while let Some(id) = worklist.pop() {
        for op in func.insts[id].data.operands() {
            if out.insert(op) {
                worklist.push(op);
            }
        }
    }
}

fn collect_eval_ids(stmts: &[StructuredStmt], out: &mut IndexSet<InstId>) {
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                out.insert(*id);
            }
            StructuredStmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_eval_ids(then_body, out);
                collect_eval_ids(else_body, out);
            }
            StructuredStmt::Switch { cases, default, .. } => {
                for (_, body) in cases {
                    collect_eval_ids(body, out);
                }
                collect_eval_ids(default, out);
            }
            StructuredStmt::Loop {
                body, continuing, ..
            } => {
                collect_eval_ids(body, out);
                collect_eval_ids(continuing, out);
            }
            StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(_) => {}
        }
    }
}

/// Computes the set of instructions defined inside the loop template.
///
/// Includes instructions explicitly evaluated via `StructuredStmt::Eval` inside the loop,
/// instructions that transitively depend on `induction_init` or a loop-defined instruction,
/// and pure helper instructions exclusively referenced inside the loop. External
/// `BdaLoad`/`ResourceLoad`/`WorkgroupLoad`/`Phi` instructions defined before the loop
/// are never added to `defined_in_loop`.
fn compute_defined_in_loop(
    func: &IrFunction,
    template: &[StructuredStmt],
    outside_refs: &IndexSet<InstId>,
    induction_init: Option<InstId>,
) -> IndexSet<InstId> {
    let mut direct_evals: IndexSet<InstId> = IndexSet::new();
    collect_eval_ids(template, &mut direct_evals);

    let mut roots: IndexSet<InstId> = IndexSet::new();
    collect_stmt_direct_refs(template, &mut roots);

    let mut postorder: Vec<InstId> = Vec::new();
    let mut visited: IndexSet<InstId> = IndexSet::new();
    for root in roots {
        collect_postorder(func, root, &mut visited, &mut postorder);
    }

    // Track memory domains mutated inside the loop so aliasing loads/samples are cloned
    // per iteration rather than reusing a single stale SSA definition across iterations.
    let mut written_locals: IndexSet<u32> = IndexSet::new();
    let mut has_unknown_local_store = false;
    let mut has_bda_store = false;
    let mut has_workgroup_store = false;
    let mut has_descriptor_store = false;
    for &id in &postorder {
        match &func.insts[id].data {
            InstData::LocalStore { ptr, .. } => {
                if let Some(slot) = resolve_local_slot(func, *ptr) {
                    written_locals.insert(slot);
                } else {
                    has_unknown_local_store = true;
                }
            }
            InstData::BdaStore { .. } => has_bda_store = true,
            InstData::WorkgroupStore { .. } => has_workgroup_store = true,
            InstData::DescriptorStore { .. } | InstData::ImageStore { .. } => {
                has_descriptor_store = true;
            }
            InstData::AtomicAdd { .. }
            | InstData::AtomicMin { .. }
            | InstData::AtomicMax { .. }
            | InstData::AtomicCas { .. }
            | InstData::FunctionCall { .. } => {
                has_unknown_local_store = true;
                has_bda_store = true;
                has_workgroup_store = true;
                has_descriptor_store = true;
            }
            _ => {}
        }
    }

    let mut defined_in_loop: IndexSet<InstId> = IndexSet::new();
    let mut loop_varying: IndexSet<InstId> = IndexSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for &id in &postorder {
            if Some(id) == induction_init || defined_in_loop.contains(&id) {
                continue;
            }
            if direct_evals.contains(&id) {
                defined_in_loop.insert(id);
                loop_varying.insert(id);
                changed = true;
                continue;
            }
            let data = &func.insts[id].data;
            let depends_on_loop = data
                .operands()
                .into_iter()
                .any(|op| Some(op) == induction_init || loop_varying.contains(&op));
            if depends_on_loop {
                defined_in_loop.insert(id);
                loop_varying.insert(id);
                changed = true;
                continue;
            }
            let is_external_sensitive = matches!(
                data,
                InstData::BdaLoad { .. }
                    | InstData::LocalLoad { .. }
                    | InstData::DescriptorLoad { .. }
                    | InstData::ResourceLoad { .. }
                    | InstData::WorkgroupLoad { .. }
                    | InstData::ImageSample { .. }
                    | InstData::ImageSampleImplicitLod { .. }
                    | InstData::ImageSampleGrad { .. }
                    | InstData::ImageSampleDrefExplicitLod { .. }
                    | InstData::ImageGather { .. }
                    | InstData::Phi { .. }
            );
            if is_external_sensitive {
                let aliases_loop_store = !outside_refs.contains(&id)
                    && match data {
                        InstData::LocalLoad { ptr } => {
                            has_unknown_local_store
                                || resolve_local_slot(func, *ptr)
                                    .is_some_and(|slot| written_locals.contains(&slot))
                        }
                        InstData::BdaLoad { .. } => has_bda_store,
                        InstData::WorkgroupLoad { .. } => has_workgroup_store,
                        InstData::DescriptorLoad { .. }
                        | InstData::ResourceLoad { .. }
                        | InstData::ImageSample { .. }
                        | InstData::ImageSampleImplicitLod { .. }
                        | InstData::ImageSampleGrad { .. }
                        | InstData::ImageSampleDrefExplicitLod { .. }
                        | InstData::ImageGather { .. } => has_descriptor_store,
                        _ => false,
                    };
                if aliases_loop_store {
                    defined_in_loop.insert(id);
                    loop_varying.insert(id);
                    changed = true;
                }
                continue;
            }
            if !outside_refs.contains(&id) && !func.insts[id].has_side_effects() {
                defined_in_loop.insert(id);
                changed = true;
            }
        }
    }

    defined_in_loop
}

fn collect_postorder(
    func: &IrFunction,
    id: InstId,
    visited: &mut IndexSet<InstId>,
    postorder: &mut Vec<InstId>,
) {
    if !visited.insert(id) {
        return;
    }
    for op in func.insts[id].data.operands() {
        collect_postorder(func, op, visited, postorder);
    }
    postorder.push(id);
}

fn build_outside_refs(
    func: &IrFunction,
    ancestor_outside: &IndexSet<InstId>,
    sibling_refs: &[IndexSet<InstId>],
    skip_idx: usize,
    extra: &[&[StructuredStmt]],
    extra_ids: &[InstId],
) -> IndexSet<InstId> {
    let mut refs = ancestor_outside.clone();
    for (other_idx, s_refs) in sibling_refs.iter().enumerate() {
        if other_idx != skip_idx {
            refs.extend(s_refs.iter().copied());
        }
    }
    for slice in extra {
        collect_stmt_transitive_refs(func, slice, &mut refs);
    }
    for &id in extra_ids {
        let mut wl = vec![id];
        if refs.insert(id) {
            while let Some(cur) = wl.pop() {
                for op in func.insts[cur].data.operands() {
                    if refs.insert(op) {
                        wl.push(op);
                    }
                }
            }
        }
    }
    refs
}

fn can_reach_inst(func: &IrFunction, start: InstId, target: InstId) -> bool {
    let mut visited: IndexSet<InstId> = IndexSet::new();
    let mut stack = vec![start];
    while let Some(cur) = stack.pop() {
        if cur == target {
            return true;
        }
        if !visited.insert(cur) {
            continue;
        }
        for op in func.insts[cur].data.operands() {
            stack.push(op);
        }
    }
    false
}

/// Identifies loop-carried [`InstData::Phi`] instructions whose back-edge operand is produced
/// inside the loop body, returning `(phi_id, init_val, back_val)` triples and ensuring all
/// instructions on the loop-carried dependency chain are included in `defined_in_loop`.
fn find_loop_carried_phis(
    func: &IrFunction,
    template: &[StructuredStmt],
    defined_in_loop: &mut IndexSet<InstId>,
    induction_init: Option<InstId>,
) -> Vec<(InstId, InstId, InstId)> {
    let mut template_refs: IndexSet<InstId> = IndexSet::new();
    collect_stmt_transitive_refs(func, template, &mut template_refs);

    let mut carried: Vec<(InstId, InstId, InstId)> = Vec::new();
    let mut carried_ids: IndexSet<InstId> = IndexSet::new();

    for &id in &template_refs {
        let InstData::Phi { incoming } = &func.insts[id].data else {
            continue;
        };
        if incoming.len() < 2 {
            continue;
        }
        let back_val = incoming
            .iter()
            .map(|&(v, _)| v)
            .find(|&v| v != id && (defined_in_loop.contains(&v) || can_reach_inst(func, v, id)));
        let Some(back_val) = back_val else {
            continue;
        };
        let init_val = incoming.iter().map(|&(v, _)| v).find(|&v| {
            v != id
                && v != back_val
                && !defined_in_loop.contains(&v)
                && !can_reach_inst(func, v, id)
        });
        if let Some(init_val) = init_val {
            carried.push((id, init_val, back_val));
            carried_ids.insert(id);
            defined_in_loop.shift_remove(&id);
        }
    }

    if !carried_ids.is_empty() {
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &template_refs {
                if Some(id) == induction_init
                    || carried_ids.contains(&id)
                    || defined_in_loop.contains(&id)
                    || matches!(func.insts[id].data, InstData::Phi { .. })
                {
                    continue;
                }
                let depends = func.insts[id]
                    .data
                    .operands()
                    .into_iter()
                    .any(|op| carried_ids.contains(&op) || defined_in_loop.contains(&op));
                if depends {
                    defined_in_loop.insert(id);
                    changed = true;
                }
            }
        }
    }

    carried
}

fn unroll_stmts(
    func: &mut IrFunction,
    stmts: Vec<StructuredStmt>,
    ancestor_outside: &IndexSet<InstId>,
    post_remap: &mut IndexMap<InstId, InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    let has_control_flow = stmts.iter().any(|s| {
        matches!(
            s,
            StructuredStmt::If { .. } | StructuredStmt::Switch { .. } | StructuredStmt::Loop { .. }
        )
    });

    let sibling_refs: Vec<IndexSet<InstId>> = if has_control_flow {
        stmts
            .iter()
            .map(|s| {
                let mut refs = IndexSet::new();
                collect_stmt_transitive_refs(func, std::slice::from_ref(s), &mut refs);
                refs
            })
            .collect()
    } else {
        Vec::new()
    };

    for (idx, stmt) in stmts.into_iter().enumerate() {
        match stmt {
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let then_outside = build_outside_refs(
                    func,
                    ancestor_outside,
                    &sibling_refs,
                    idx,
                    &[&else_body],
                    &[cond],
                );
                let else_outside = build_outside_refs(
                    func,
                    ancestor_outside,
                    &sibling_refs,
                    idx,
                    &[&then_body],
                    &[cond],
                );
                out.push(StructuredStmt::If {
                    cond,
                    then_body: unroll_stmts(func, then_body, &then_outside, post_remap),
                    else_body: unroll_stmts(func, else_body, &else_outside, post_remap),
                });
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                let mut case_outsides = Vec::with_capacity(cases.len());
                for case_idx in 0..cases.len() {
                    let mut case_outside = build_outside_refs(
                        func,
                        ancestor_outside,
                        &sibling_refs,
                        idx,
                        &[&default],
                        &[selector],
                    );
                    for (other_case_idx, (_, other_b)) in cases.iter().enumerate() {
                        if other_case_idx != case_idx {
                            collect_stmt_transitive_refs(func, other_b, &mut case_outside);
                        }
                    }
                    case_outsides.push(case_outside);
                }
                let mut default_outside = build_outside_refs(
                    func,
                    ancestor_outside,
                    &sibling_refs,
                    idx,
                    &[],
                    &[selector],
                );
                for (_, case_b) in &cases {
                    collect_stmt_transitive_refs(func, case_b, &mut default_outside);
                }
                let new_cases = cases
                    .into_iter()
                    .zip(case_outsides)
                    .map(|((k, b), outside)| (k, unroll_stmts(func, b, &outside, post_remap)))
                    .collect();
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: new_cases,
                    default: unroll_stmts(func, default, &default_outside, post_remap),
                });
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                let extra_ind: Vec<InstId> = induction_init.into_iter().collect();
                let loop_outside =
                    build_outside_refs(func, ancestor_outside, &sibling_refs, idx, &[], &extra_ind);
                let mut body_outside = loop_outside.clone();
                collect_stmt_transitive_refs(func, &continuing, &mut body_outside);
                let mut cont_outside = loop_outside.clone();
                collect_stmt_transitive_refs(func, &body, &mut cont_outside);

                let body = unroll_stmts(func, body, &body_outside, post_remap);
                let continuing = unroll_stmts(func, continuing, &cont_outside, post_remap);
                let body_inst_count = count_stmts_insts(&body) + count_stmts_insts(&continuing);
                if unroll
                    && let Some(trips) = trip_count
                    && trips <= MAX_UNROLL_ITERATIONS
                    && body_inst_count <= MAX_UNROLL_BODY_INSTS
                    && !contains_top_level_break_or_continue(&body)
                    && !contains_top_level_break_or_continue(&continuing)
                {
                    let mut template = Vec::with_capacity(body.len() + continuing.len());
                    template.extend(body);
                    template.extend(continuing);

                    let mut defined_in_loop =
                        compute_defined_in_loop(func, &template, &loop_outside, induction_init);
                    let loop_carried_phis = find_loop_carried_phis(
                        func,
                        &template,
                        &mut defined_in_loop,
                        induction_init,
                    );

                    if loop_carried_phis.is_empty() {
                        let empty_phis: IndexSet<InstId> = IndexSet::new();
                        for iter_idx in 0..trips {
                            if iter_idx == 0 {
                                out.extend(template.iter().cloned());
                            } else {
                                let mut remap: IndexMap<InstId, InstId> = IndexMap::new();
                                if let Some(ind_id) = induction_init {
                                    let stepped = emit_stepped_induction(
                                        func,
                                        ind_id,
                                        iter_idx,
                                        induction_step,
                                    );
                                    remap.insert(ind_id, stepped);
                                }
                                let cloned = clone_stmts_with_remap(
                                    func,
                                    &template,
                                    &defined_in_loop,
                                    &empty_phis,
                                    &mut remap,
                                );
                                out.extend(cloned);
                            }
                        }
                    } else {
                        let eliminated_phis: IndexSet<InstId> = loop_carried_phis
                            .iter()
                            .map(|&(phi_id, _, _)| phi_id)
                            .collect();
                        let mut current_phi_vals: IndexMap<InstId, InstId> = loop_carried_phis
                            .iter()
                            .map(|&(phi_id, init_val, _)| (phi_id, init_val))
                            .collect();
                        for iter_idx in 0..trips {
                            let mut remap = current_phi_vals.clone();
                            if iter_idx > 0
                                && let Some(ind_id) = induction_init
                            {
                                let stepped =
                                    emit_stepped_induction(func, ind_id, iter_idx, induction_step);
                                remap.insert(ind_id, stepped);
                            }
                            let cloned = clone_stmts_with_remap(
                                func,
                                &template,
                                &defined_in_loop,
                                &eliminated_phis,
                                &mut remap,
                            );
                            for &(phi_id, _, back_val) in &loop_carried_phis {
                                let next_val = clone_inst_in_loop(
                                    func,
                                    back_val,
                                    &defined_in_loop,
                                    &mut remap,
                                );
                                current_phi_vals.insert(phi_id, next_val);
                            }
                            out.extend(cloned);
                        }
                        let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
                        for id in all_ids {
                            if current_phi_vals.contains_key(&id) {
                                func.insts[id].data = InstData::Nop;
                                func.insts[id].non_uniform = false;
                            } else {
                                let remapped =
                                    remap_inst_data(&func.insts[id].data, &current_phi_vals);
                                func.insts[id].data = remapped;
                            }
                        }
                        post_remap.extend(current_phi_vals);
                    }
                } else {
                    out.push(StructuredStmt::Loop {
                        body,
                        continuing,
                        unroll,
                        trip_count,
                        induction_init,
                        induction_step,
                    });
                }
            }
            other => out.push(other),
        }
    }
    if !post_remap.is_empty() {
        out = remap_stmts_ids(out, post_remap);
    }
    out
}

fn emit_stepped_induction(
    func: &mut IrFunction,
    ind_id: InstId,
    iter_idx: u32,
    step: i64,
) -> InstId {
    let delta = i64::from(iter_idx).wrapping_mul(step);
    let ty = func.insts[ind_id].ty;
    match func.insts[ind_id].data {
        InstData::ConstI32(init) => {
            func.emit(ty, InstData::ConstI32(init.wrapping_add(delta as i32)))
        }
        InstData::ConstU32(init) => func.emit(
            ty,
            InstData::ConstU32(init.wrapping_add(delta.cast_unsigned() as u32)),
        ),
        InstData::ConstI64(init) => func.emit(ty, InstData::ConstI64(init.wrapping_add(delta))),
        InstData::ConstU64(init) => func.emit(
            ty,
            InstData::ConstU64(init.wrapping_add(delta.cast_unsigned())),
        ),
        _ => match func.insts[ind_id].ir_ty.scalar_ty() {
            Some(ScalarIrTy::U32) => {
                let delta_const = func.emit(None, InstData::ConstU32(delta.cast_unsigned() as u32));
                func.emit(ty, InstData::IAdd32(ind_id, delta_const))
            }
            Some(ScalarIrTy::U64) => {
                let delta_const = func.emit(None, InstData::ConstU64(delta.cast_unsigned()));
                func.emit(ty, InstData::IAdd64(ind_id, delta_const))
            }
            Some(ScalarIrTy::I64) => {
                let delta_const = func.emit(None, InstData::ConstI64(delta));
                func.emit(ty, InstData::IAdd64(ind_id, delta_const))
            }
            _ => {
                let delta_const = func.emit(None, InstData::ConstI32(delta as i32));
                func.emit(ty, InstData::IAdd32(ind_id, delta_const))
            }
        },
    }
}

fn count_stmts_insts(stmts: &[StructuredStmt]) -> usize {
    let mut total = 0usize;
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(_) => total += 1,
            StructuredStmt::If {
                then_body,
                else_body,
                ..
            } => {
                total += 1 + count_stmts_insts(then_body) + count_stmts_insts(else_body);
            }
            StructuredStmt::Switch { cases, default, .. } => {
                total += 1 + count_stmts_insts(default);
                for (_, b) in cases {
                    total += count_stmts_insts(b);
                }
            }
            StructuredStmt::Loop {
                body, continuing, ..
            } => {
                total += 1 + count_stmts_insts(body) + count_stmts_insts(continuing);
            }
            StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(_) => {
                total += 1;
            }
        }
    }
    total
}

/// Returns `true` if `stmts` contains a `Break` or `Continue` targeting the immediately
/// enclosing loop (nested loops handle their own `Break`/`Continue` internally).
fn contains_top_level_break_or_continue(stmts: &[StructuredStmt]) -> bool {
    for stmt in stmts {
        match stmt {
            StructuredStmt::Break | StructuredStmt::Continue => return true,
            StructuredStmt::If {
                then_body,
                else_body,
                ..
            } => {
                if contains_top_level_break_or_continue(then_body)
                    || contains_top_level_break_or_continue(else_body)
                {
                    return true;
                }
            }
            StructuredStmt::Switch { cases, default, .. } => {
                if cases
                    .iter()
                    .any(|(_, b)| contains_top_level_break_or_continue(b))
                    || contains_top_level_break_or_continue(default)
                {
                    return true;
                }
            }
            StructuredStmt::Loop { .. } | StructuredStmt::Eval(_) | StructuredStmt::Return(_) => {}
        }
    }
    false
}

/// Recursively clones an instruction only if it belongs to `defined_in_loop`, preserving
/// external instructions defined before the loop while remapping induction variables.
fn clone_inst_in_loop(
    func: &mut IrFunction,
    id: InstId,
    defined_in_loop: &IndexSet<InstId>,
    remap: &mut IndexMap<InstId, InstId>,
) -> InstId {
    if let Some(&mapped) = remap.get(&id) {
        return mapped;
    }
    if !defined_in_loop.contains(&id) {
        return id;
    }
    let original = func.insts[id].clone();
    // Pre-allocate a placeholder ID in `remap` before recursing so cyclic `Phi` chains
    // inside nested non-unrolled loops terminate cleanly instead of overflowing the stack.
    let new_id = func.emit_typed(original.ir_ty.clone(), InstData::Nop);
    func.insts[new_id].ty = original.ty;
    func.insts[new_id].non_uniform = original.non_uniform;
    remap.insert(id, new_id);
    for op in original.data.operands() {
        let _ = clone_inst_in_loop(func, op, defined_in_loop, remap);
    }
    let remapped_data = remap_inst_data(&original.data, remap);
    func.insts[new_id].data = remapped_data;
    new_id
}

/// Clones structured statements for an unrolled loop iteration, allocating fresh SSA
/// instruction IDs only for instructions in `defined_in_loop` and dropping `Eval`s of
/// eliminated loop-carried `Phi`s.
fn clone_stmts_with_remap(
    func: &mut IrFunction,
    stmts: &[StructuredStmt],
    defined_in_loop: &IndexSet<InstId>,
    eliminated_phis: &IndexSet<InstId>,
    remap: &mut IndexMap<InstId, InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                if eliminated_phis.contains(id) {
                    continue;
                }
                let new_id = clone_inst_in_loop(func, *id, defined_in_loop, remap);
                out.push(StructuredStmt::Eval(new_id));
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond_id = clone_inst_in_loop(func, *cond, defined_in_loop, remap);
                out.push(StructuredStmt::If {
                    cond: cond_id,
                    then_body: clone_stmts_with_remap(
                        func,
                        then_body,
                        defined_in_loop,
                        eliminated_phis,
                        remap,
                    ),
                    else_body: clone_stmts_with_remap(
                        func,
                        else_body,
                        defined_in_loop,
                        eliminated_phis,
                        remap,
                    ),
                });
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                let sel_id = clone_inst_in_loop(func, *selector, defined_in_loop, remap);
                out.push(StructuredStmt::Switch {
                    selector: sel_id,
                    cases: cases
                        .iter()
                        .map(|(k, b)| {
                            (
                                *k,
                                clone_stmts_with_remap(
                                    func,
                                    b,
                                    defined_in_loop,
                                    eliminated_phis,
                                    remap,
                                ),
                            )
                        })
                        .collect(),
                    default: clone_stmts_with_remap(
                        func,
                        default,
                        defined_in_loop,
                        eliminated_phis,
                        remap,
                    ),
                });
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                let mapped_init =
                    induction_init.map(|id| clone_inst_in_loop(func, id, defined_in_loop, remap));
                out.push(StructuredStmt::Loop {
                    body: clone_stmts_with_remap(
                        func,
                        body,
                        defined_in_loop,
                        eliminated_phis,
                        remap,
                    ),
                    continuing: clone_stmts_with_remap(
                        func,
                        continuing,
                        defined_in_loop,
                        eliminated_phis,
                        remap,
                    ),
                    unroll: *unroll,
                    trip_count: *trip_count,
                    induction_init: mapped_init,
                    induction_step: *induction_step,
                });
            }
            StructuredStmt::Return(val) => {
                let mapped_val = val.map(|id| clone_inst_in_loop(func, id, defined_in_loop, remap));
                out.push(StructuredStmt::Return(mapped_val));
            }
            StructuredStmt::Break => out.push(StructuredStmt::Break),
            StructuredStmt::Continue => out.push(StructuredStmt::Continue),
        }
    }
    out
}

fn remap_id(mut id: InstId, remap: &IndexMap<InstId, InstId>) -> InstId {
    for _ in 0..remap.len() {
        let Some(&next) = remap.get(&id) else {
            break;
        };
        if next == id {
            break;
        }
        id = next;
    }
    id
}

fn remap_inst_data(data: &InstData, remap: &IndexMap<InstId, InstId>) -> InstData {
    match data {
        InstData::ConstBool(v) => InstData::ConstBool(*v),
        InstData::ConstI32(v) => InstData::ConstI32(*v),
        InstData::ConstU32(v) => InstData::ConstU32(*v),
        InstData::ConstF32(v) => InstData::ConstF32(*v),
        InstData::ConstI64(v) => InstData::ConstI64(*v),
        InstData::ConstU64(v) => InstData::ConstU64(*v),
        InstData::IAdd32(a, b) => InstData::IAdd32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::ISub32(a, b) => InstData::ISub32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::IMul32(a, b) => InstData::IMul32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::IDiv32(a, b) => InstData::IDiv32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::IRem32(a, b) => InstData::IRem32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FAdd32(a, b) => InstData::FAdd32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FSub32(a, b) => InstData::FSub32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FMul32(a, b) => InstData::FMul32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FDiv32(a, b) => InstData::FDiv32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::ICmpEq(a, b) => InstData::ICmpEq(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::ICmpNe(a, b) => InstData::ICmpNe(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::ICmpLt(a, b) => InstData::ICmpLt(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::ICmpLe(a, b) => InstData::ICmpLe(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::ICmpGt(a, b) => InstData::ICmpGt(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::ICmpGe(a, b) => InstData::ICmpGe(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::UCmpLt(a, b) => InstData::UCmpLt(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FCmpEq(a, b) => InstData::FCmpEq(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FCmpNe(a, b) => InstData::FCmpNe(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FCmpLt(a, b) => InstData::FCmpLt(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FCmpLe(a, b) => InstData::FCmpLe(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FCmpGt(a, b) => InstData::FCmpGt(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::FCmpGe(a, b) => InstData::FCmpGe(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::LogicalAnd(a, b) => {
            InstData::LogicalAnd(remap_id(*a, remap), remap_id(*b, remap))
        }
        InstData::LogicalOr(a, b) => InstData::LogicalOr(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::LogicalNot(v) => InstData::LogicalNot(remap_id(*v, remap)),
        InstData::Select {
            cond,
            on_true,
            on_false,
        } => InstData::Select {
            cond: remap_id(*cond, remap),
            on_true: remap_id(*on_true, remap),
            on_false: remap_id(*on_false, remap),
        },
        InstData::BitAnd32(a, b) => InstData::BitAnd32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::BitOr32(a, b) => InstData::BitOr32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::BitXor32(a, b) => InstData::BitXor32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::BitNot32(v) => InstData::BitNot32(remap_id(*v, remap)),
        InstData::Shl32(a, b) => InstData::Shl32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::Shr32(a, b) => InstData::Shr32(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::CastI64ToU64(v) => InstData::CastI64ToU64(remap_id(*v, remap)),
        InstData::CastToU32(v) => InstData::CastToU32(remap_id(*v, remap)),
        InstData::CastToI32(v) => InstData::CastToI32(remap_id(*v, remap)),
        InstData::CastToF32(v) => InstData::CastToF32(remap_id(*v, remap)),
        InstData::CastToI64(v) => InstData::CastToI64(remap_id(*v, remap)),
        InstData::CastToU64(v) => InstData::CastToU64(remap_id(*v, remap)),
        InstData::SpecConstant {
            spec_id,
            scalar,
            default_bits,
        } => InstData::SpecConstant {
            spec_id: *spec_id,
            scalar: *scalar,
            default_bits: *default_bits,
        },
        InstData::GlslStd450 { op, args } => InstData::GlslStd450 {
            op: *op,
            args: args.iter().map(|&a| remap_id(a, remap)).collect(),
        },
        InstData::Phi { incoming } => InstData::Phi {
            incoming: incoming
                .iter()
                .map(|&(v, pred)| (remap_id(v, remap), pred))
                .collect(),
        },
        InstData::BdaOffset {
            base_ptr,
            index_i64,
            stride_bytes,
            align,
        } => InstData::BdaOffset {
            base_ptr: remap_id(*base_ptr, remap),
            index_i64: remap_id(*index_i64, remap),
            stride_bytes: *stride_bytes,
            align: *align,
        },
        InstData::ConvertPtrToU(v) => InstData::ConvertPtrToU(remap_id(*v, remap)),
        InstData::ConvertUToPtr { addr_u64, align } => InstData::ConvertUToPtr {
            addr_u64: remap_id(*addr_u64, remap),
            align: *align,
        },
        InstData::IMul64(a, b) => InstData::IMul64(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::IAdd64(a, b) => InstData::IAdd64(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::BdaLoad { ptr, access } => InstData::BdaLoad {
            ptr: remap_id(*ptr, remap),
            access: *access,
        },
        InstData::BdaStore { ptr, value, access } => InstData::BdaStore {
            ptr: remap_id(*ptr, remap),
            value: remap_id(*value, remap),
            access: *access,
        },
        InstData::ImageSample {
            descriptor_index,
            image_inst,
            coord,
            lod,
            is_fetch,
        } => InstData::ImageSample {
            descriptor_index: *descriptor_index,
            image_inst: image_inst.map(|i| remap_id(i, remap)),
            coord: remap_id(*coord, remap),
            lod: remap_id(*lod, remap),
            is_fetch: *is_fetch,
        },
        InstData::ImageStore {
            descriptor_index,
            coord,
            value,
        } => InstData::ImageStore {
            descriptor_index: *descriptor_index,
            coord: remap_id(*coord, remap),
            value: remap_id(*value, remap),
        },
        InstData::WorkgroupAddr { var_index, index } => InstData::WorkgroupAddr {
            var_index: *var_index,
            index: index.map(|i| remap_id(i, remap)),
        },
        InstData::WorkgroupLoad { var_index, index } => InstData::WorkgroupLoad {
            var_index: *var_index,
            index: index.map(|i| remap_id(i, remap)),
        },
        InstData::WorkgroupStore {
            var_index,
            index,
            value,
        } => InstData::WorkgroupStore {
            var_index: *var_index,
            index: index.map(|i| remap_id(i, remap)),
            value: remap_id(*value, remap),
        },
        InstData::NonUniform(v) => InstData::NonUniform(remap_id(*v, remap)),
        InstData::ResourceAccessChain {
            descriptor_index,
            index,
        } => InstData::ResourceAccessChain {
            descriptor_index: *descriptor_index,
            index: remap_id(*index, remap),
        },
        InstData::ResourceLoad { ptr } => InstData::ResourceLoad {
            ptr: remap_id(*ptr, remap),
        },
        InstData::AtomicAdd {
            ptr,
            value,
            scope,
            order,
        } => InstData::AtomicAdd {
            ptr: remap_id(*ptr, remap),
            value: remap_id(*value, remap),
            scope: *scope,
            order: *order,
        },
        InstData::AtomicMin {
            ptr,
            value,
            scope,
            order,
        } => InstData::AtomicMin {
            ptr: remap_id(*ptr, remap),
            value: remap_id(*value, remap),
            scope: *scope,
            order: *order,
        },
        InstData::AtomicMax {
            ptr,
            value,
            scope,
            order,
        } => InstData::AtomicMax {
            ptr: remap_id(*ptr, remap),
            value: remap_id(*value, remap),
            scope: *scope,
            order: *order,
        },
        InstData::AtomicCas {
            ptr,
            expected,
            desired,
            scope,
            order,
        } => InstData::AtomicCas {
            ptr: remap_id(*ptr, remap),
            expected: remap_id(*expected, remap),
            desired: remap_id(*desired, remap),
            scope: *scope,
            order: *order,
        },
        InstData::WorkgroupBarrier => InstData::WorkgroupBarrier,
        InstData::SubgroupBallot(v) => InstData::SubgroupBallot(remap_id(*v, remap)),
        InstData::SubgroupBallotBitCount(v) => {
            InstData::SubgroupBallotBitCount(remap_id(*v, remap))
        }
        InstData::SubgroupBroadcastFirst(v) => {
            InstData::SubgroupBroadcastFirst(remap_id(*v, remap))
        }
        InstData::DescriptorLoad {
            descriptor_index,
            word_index,
            scalar,
        } => InstData::DescriptorLoad {
            descriptor_index: *descriptor_index,
            word_index: *word_index,
            scalar: *scalar,
        },
        InstData::LoadBuiltin { kind, component } => InstData::LoadBuiltin {
            kind: *kind,
            component: *component,
        },
        InstData::LoadPushConstant { field_index } => InstData::LoadPushConstant {
            field_index: *field_index,
        },
        InstData::RayQueryInitialize {
            ray_query_id,
            tlas_descriptor_index,
            flags,
            cull_mask,
            origin,
            t_min,
            dir,
            t_max,
        } => InstData::RayQueryInitialize {
            ray_query_id: *ray_query_id,
            tlas_descriptor_index: *tlas_descriptor_index,
            flags: remap_id(*flags, remap),
            cull_mask: remap_id(*cull_mask, remap),
            origin: remap_id(*origin, remap),
            t_min: remap_id(*t_min, remap),
            dir: remap_id(*dir, remap),
            t_max: remap_id(*t_max, remap),
        },
        InstData::RayQueryProceed(rq) => InstData::RayQueryProceed(*rq),
        InstData::RayQueryCommittedType(rq) => InstData::RayQueryCommittedType(*rq),
        InstData::RayQueryCommittedInstanceCustomIndex(rq) => {
            InstData::RayQueryCommittedInstanceCustomIndex(*rq)
        }
        InstData::RayQueryCommittedBarycentrics(rq) => InstData::RayQueryCommittedBarycentrics(*rq),
        InstData::MakeVec2(a, b) => InstData::MakeVec2(remap_id(*a, remap), remap_id(*b, remap)),
        InstData::MakeVec3(a, b, c) => InstData::MakeVec3(
            remap_id(*a, remap),
            remap_id(*b, remap),
            remap_id(*c, remap),
        ),
        InstData::MakeVec4(a, b, c, d) => InstData::MakeVec4(
            remap_id(*a, remap),
            remap_id(*b, remap),
            remap_id(*c, remap),
            remap_id(*d, remap),
        ),
        InstData::ConstF16(bits) => InstData::ConstF16(*bits),
        InstData::FConvert { value, target } => InstData::FConvert {
            value: remap_id(*value, remap),
            target: *target,
        },
        InstData::SpecConstantOp { op, operands } => InstData::SpecConstantOp {
            op: *op,
            operands: operands.iter().map(|&o| remap_id(o, remap)).collect(),
        },
        InstData::VectorTimesScalar { vec, scalar } => InstData::VectorTimesScalar {
            vec: remap_id(*vec, remap),
            scalar: remap_id(*scalar, remap),
        },
        InstData::MatrixTimesVector { mat, vec } => InstData::MatrixTimesVector {
            mat: remap_id(*mat, remap),
            vec: remap_id(*vec, remap),
        },
        InstData::MatrixTimesMatrix { lhs, rhs } => InstData::MatrixTimesMatrix {
            lhs: remap_id(*lhs, remap),
            rhs: remap_id(*rhs, remap),
        },
        InstData::VectorShuffle {
            lhs,
            rhs,
            components,
        } => InstData::VectorShuffle {
            lhs: remap_id(*lhs, remap),
            rhs: remap_id(*rhs, remap),
            components: components.clone(),
        },
        InstData::CompositeConstruct { elems } => InstData::CompositeConstruct {
            elems: elems.iter().map(|&e| remap_id(e, remap)).collect(),
        },
        InstData::CompositeExtract { composite, indices } => InstData::CompositeExtract {
            composite: remap_id(*composite, remap),
            indices: indices.clone(),
        },
        InstData::CompositeInsert {
            composite,
            value,
            indices,
        } => InstData::CompositeInsert {
            composite: remap_id(*composite, remap),
            value: remap_id(*value, remap),
            indices: indices.clone(),
        },
        InstData::LocalAddr { local_index } => InstData::LocalAddr {
            local_index: *local_index,
        },
        InstData::LocalLoad { ptr } => InstData::LocalLoad {
            ptr: remap_id(*ptr, remap),
        },
        InstData::LocalStore { ptr, value } => InstData::LocalStore {
            ptr: remap_id(*ptr, remap),
            value: remap_id(*value, remap),
        },
        InstData::AccessChain { base_ptr, indices } => InstData::AccessChain {
            base_ptr: remap_id(*base_ptr, remap),
            indices: indices.iter().map(|&i| remap_id(i, remap)).collect(),
        },
        InstData::DescriptorAccessChain {
            descriptor_index,
            indices,
        } => InstData::DescriptorAccessChain {
            descriptor_index: *descriptor_index,
            indices: indices.iter().map(|&i| remap_id(i, remap)).collect(),
        },
        InstData::DescriptorStore { ptr, value } => InstData::DescriptorStore {
            ptr: remap_id(*ptr, remap),
            value: remap_id(*value, remap),
        },
        InstData::Param { index } => InstData::Param { index: *index },
        InstData::FunctionCall { callee, args } => InstData::FunctionCall {
            callee: callee.clone(),
            args: args.iter().map(|&a| remap_id(a, remap)).collect(),
        },
        InstData::ImageSampleImplicitLod {
            descriptor_index,
            image_inst,
            coord,
        } => InstData::ImageSampleImplicitLod {
            descriptor_index: *descriptor_index,
            image_inst: image_inst.map(|i| remap_id(i, remap)),
            coord: remap_id(*coord, remap),
        },
        InstData::ImageSampleGrad {
            descriptor_index,
            image_inst,
            coord,
            ddx,
            ddy,
        } => InstData::ImageSampleGrad {
            descriptor_index: *descriptor_index,
            image_inst: image_inst.map(|i| remap_id(i, remap)),
            coord: remap_id(*coord, remap),
            ddx: remap_id(*ddx, remap),
            ddy: remap_id(*ddy, remap),
        },
        InstData::ImageSampleDrefExplicitLod {
            descriptor_index,
            image_inst,
            coord,
            dref,
            lod,
        } => InstData::ImageSampleDrefExplicitLod {
            descriptor_index: *descriptor_index,
            image_inst: image_inst.map(|i| remap_id(i, remap)),
            coord: remap_id(*coord, remap),
            dref: remap_id(*dref, remap),
            lod: remap_id(*lod, remap),
        },
        InstData::ImageGather {
            descriptor_index,
            image_inst,
            coord,
            component,
        } => InstData::ImageGather {
            descriptor_index: *descriptor_index,
            image_inst: image_inst.map(|i| remap_id(i, remap)),
            coord: remap_id(*coord, remap),
            component: *component,
        },
        InstData::ImageQuerySize {
            descriptor_index,
            lod,
        } => InstData::ImageQuerySize {
            descriptor_index: *descriptor_index,
            lod: lod.map(|i| remap_id(i, remap)),
        },
        InstData::ImageQueryLevels { descriptor_index } => InstData::ImageQueryLevels {
            descriptor_index: *descriptor_index,
        },
        InstData::ImageTexelPointer {
            descriptor_index,
            coord,
            sample,
        } => InstData::ImageTexelPointer {
            descriptor_index: *descriptor_index,
            coord: remap_id(*coord, remap),
            sample: remap_id(*sample, remap),
        },
        InstData::MemoryBarrier { scope, order } => InstData::MemoryBarrier {
            scope: *scope,
            order: *order,
        },
        InstData::SubgroupBallotExclusiveBitCount(v) => {
            InstData::SubgroupBallotExclusiveBitCount(remap_id(*v, remap))
        }
        InstData::SubgroupBallotInclusiveBitCount(v) => {
            InstData::SubgroupBallotInclusiveBitCount(remap_id(*v, remap))
        }
        InstData::SubgroupAdd(v) => InstData::SubgroupAdd(remap_id(*v, remap)),
        InstData::SubgroupMin(v) => InstData::SubgroupMin(remap_id(*v, remap)),
        InstData::SubgroupMax(v) => InstData::SubgroupMax(remap_id(*v, remap)),
        InstData::SubgroupExclusiveAdd(v) => InstData::SubgroupExclusiveAdd(remap_id(*v, remap)),
        InstData::SubgroupInclusiveAdd(v) => InstData::SubgroupInclusiveAdd(remap_id(*v, remap)),
        InstData::SubgroupShuffle { value, lane } => InstData::SubgroupShuffle {
            value: remap_id(*value, remap),
            lane: remap_id(*lane, remap),
        },
        InstData::SubgroupShuffleXor { value, mask } => InstData::SubgroupShuffleXor {
            value: remap_id(*value, remap),
            mask: remap_id(*mask, remap),
        },
        InstData::SubgroupAny(v) => InstData::SubgroupAny(remap_id(*v, remap)),
        InstData::SubgroupAll(v) => InstData::SubgroupAll(remap_id(*v, remap)),
        InstData::SubgroupElect => InstData::SubgroupElect,
        InstData::QuadSwapHorizontal(v) => InstData::QuadSwapHorizontal(remap_id(*v, remap)),
        InstData::QuadSwapVertical(v) => InstData::QuadSwapVertical(remap_id(*v, remap)),
        InstData::QuadSwapDiagonal(v) => InstData::QuadSwapDiagonal(remap_id(*v, remap)),
        InstData::BitFieldExtract {
            base,
            offset,
            count,
            signed,
        } => InstData::BitFieldExtract {
            base: remap_id(*base, remap),
            offset: remap_id(*offset, remap),
            count: remap_id(*count, remap),
            signed: *signed,
        },
        InstData::BitFieldInsert {
            base,
            insert,
            offset,
            count,
        } => InstData::BitFieldInsert {
            base: remap_id(*base, remap),
            insert: remap_id(*insert, remap),
            offset: remap_id(*offset, remap),
            count: remap_id(*count, remap),
        },
        InstData::BitCount(v) => InstData::BitCount(remap_id(*v, remap)),
        InstData::Bitcast { value, target } => InstData::Bitcast {
            value: remap_id(*value, remap),
            target: target.clone(),
        },
        InstData::IsNan(v) => InstData::IsNan(remap_id(*v, remap)),
        InstData::IsInf(v) => InstData::IsInf(remap_id(*v, remap)),
        InstData::DPdx(v) => InstData::DPdx(remap_id(*v, remap)),
        InstData::DPdy(v) => InstData::DPdy(remap_id(*v, remap)),
        InstData::Fwidth(v) => InstData::Fwidth(remap_id(*v, remap)),
        InstData::LoadStageInput { var_index } => InstData::LoadStageInput {
            var_index: *var_index,
        },
        InstData::StoreStageOutput { var_index, value } => InstData::StoreStageOutput {
            var_index: *var_index,
            value: remap_id(*value, remap),
        },
        InstData::StoreBuiltin { kind, value } => InstData::StoreBuiltin {
            kind: *kind,
            value: remap_id(*value, remap),
        },
        InstData::DemoteToHelperInvocation => InstData::DemoteToHelperInvocation,
        InstData::RayQueryCandidateType(rq) => InstData::RayQueryCandidateType(*rq),
        InstData::RayQueryConfirmIntersection(rq) => InstData::RayQueryConfirmIntersection(*rq),
        InstData::RayQueryTerminate(rq) => InstData::RayQueryTerminate(*rq),
        InstData::RayQueryGenerateIntersection {
            ray_query_id,
            t_hit,
        } => InstData::RayQueryGenerateIntersection {
            ray_query_id: *ray_query_id,
            t_hit: remap_id(*t_hit, remap),
        },
        InstData::RayQueryCommittedT(rq) => InstData::RayQueryCommittedT(*rq),
        InstData::RayQueryCandidateT(rq) => InstData::RayQueryCandidateT(*rq),
        InstData::RayQueryCommittedInstanceId(rq) => InstData::RayQueryCommittedInstanceId(*rq),
        InstData::RayQueryCommittedPrimitiveIndex(rq) => {
            InstData::RayQueryCommittedPrimitiveIndex(*rq)
        }
        InstData::RayQueryCommittedGeometryIndex(rq) => {
            InstData::RayQueryCommittedGeometryIndex(*rq)
        }
        InstData::RayQueryCommittedObjectToWorld(rq) => {
            InstData::RayQueryCommittedObjectToWorld(*rq)
        }
        InstData::RayQueryCommittedWorldToObject(rq) => {
            InstData::RayQueryCommittedWorldToObject(*rq)
        }
        InstData::RayQueryCommittedFrontFace(rq) => InstData::RayQueryCommittedFrontFace(*rq),
        InstData::ExtractComponent { vec, index } => InstData::ExtractComponent {
            vec: remap_id(*vec, remap),
            index: *index,
        },
        InstData::TraceRayKhr {
            tlas_descriptor_index,
            ray_flags,
            cull_mask,
            sbt_offset,
            sbt_stride,
            miss_index,
            origin,
            t_min,
            direction,
            t_max,
            payload_var_index,
        } => InstData::TraceRayKhr {
            tlas_descriptor_index: *tlas_descriptor_index,
            ray_flags: remap_id(*ray_flags, remap),
            cull_mask: remap_id(*cull_mask, remap),
            sbt_offset: remap_id(*sbt_offset, remap),
            sbt_stride: remap_id(*sbt_stride, remap),
            miss_index: remap_id(*miss_index, remap),
            origin: remap_id(*origin, remap),
            t_min: remap_id(*t_min, remap),
            direction: remap_id(*direction, remap),
            t_max: remap_id(*t_max, remap),
            payload_var_index: *payload_var_index,
        },
        InstData::IgnoreIntersectionKhr => InstData::IgnoreIntersectionKhr,
        InstData::TerminateRayKhr => InstData::TerminateRayKhr,
        InstData::ReportIntersectionKhr { hit_t, hit_kind } => InstData::ReportIntersectionKhr {
            hit_t: remap_id(*hit_t, remap),
            hit_kind: remap_id(*hit_kind, remap),
        },
        InstData::ExecuteCallableKhr {
            sbt_index,
            callable_var_index,
        } => InstData::ExecuteCallableKhr {
            sbt_index: remap_id(*sbt_index, remap),
            callable_var_index: *callable_var_index,
        },
        InstData::EmitMeshTasksExt {
            group_x,
            group_y,
            group_z,
            payload_var_index,
        } => InstData::EmitMeshTasksExt {
            group_x: remap_id(*group_x, remap),
            group_y: remap_id(*group_y, remap),
            group_z: remap_id(*group_z, remap),
            payload_var_index: *payload_var_index,
        },
        InstData::SetMeshOutputsExt {
            vertex_count,
            primitive_count,
        } => InstData::SetMeshOutputsExt {
            vertex_count: remap_id(*vertex_count, remap),
            primitive_count: remap_id(*primitive_count, remap),
        },
        InstData::EmitVertex => InstData::EmitVertex,
        InstData::EndPrimitive => InstData::EndPrimitive,
        InstData::LoadStageVar { var_index } => InstData::LoadStageVar {
            var_index: *var_index,
        },
        InstData::StoreStageVar { var_index, value } => InstData::StoreStageVar {
            var_index: *var_index,
            value: remap_id(*value, remap),
        },
        InstData::HitObjectTraceRay {
            hit_object_id,
            tlas_descriptor_index,
            ray_flags,
            cull_mask,
            sbt_offset,
            sbt_stride,
            miss_index,
            origin,
            t_min,
            direction,
            t_max,
            payload_var_index,
        } => InstData::HitObjectTraceRay {
            hit_object_id: *hit_object_id,
            tlas_descriptor_index: *tlas_descriptor_index,
            ray_flags: remap_id(*ray_flags, remap),
            cull_mask: remap_id(*cull_mask, remap),
            sbt_offset: remap_id(*sbt_offset, remap),
            sbt_stride: remap_id(*sbt_stride, remap),
            miss_index: remap_id(*miss_index, remap),
            origin: remap_id(*origin, remap),
            t_min: remap_id(*t_min, remap),
            direction: remap_id(*direction, remap),
            t_max: remap_id(*t_max, remap),
            payload_var_index: *payload_var_index,
        },
        InstData::HitObjectRecordEmpty(id) => InstData::HitObjectRecordEmpty(*id),
        InstData::HitObjectIsHit(id) => InstData::HitObjectIsHit(*id),
        InstData::HitObjectGetRayTMax(id) => InstData::HitObjectGetRayTMax(*id),
        InstData::HitObjectGetSbtRecordIndex(id) => InstData::HitObjectGetSbtRecordIndex(*id),
        InstData::HitObjectExecuteShader {
            hit_object_id,
            payload_var_index,
        } => InstData::HitObjectExecuteShader {
            hit_object_id: *hit_object_id,
            payload_var_index: *payload_var_index,
        },
        InstData::ReorderThreadWithHint { hint, bits } => InstData::ReorderThreadWithHint {
            hint: remap_id(*hint, remap),
            bits: remap_id(*bits, remap),
        },
        InstData::ReorderThreadWithHitObject {
            hit_object_id,
            hint,
            bits,
        } => InstData::ReorderThreadWithHitObject {
            hit_object_id: *hit_object_id,
            hint: hint.map(|h| remap_id(h, remap)),
            bits: bits.map(|b| remap_id(b, remap)),
        },
        InstData::RayQueryTriangleVertexPositions {
            ray_query_id,
            committed,
        } => InstData::RayQueryTriangleVertexPositions {
            ray_query_id: *ray_query_id,
            committed: *committed,
        },
        InstData::Nop => InstData::Nop,
    }
}

/// Lowers reachable `BdaOffset` instructions (standalone statements and transitive operands
/// reachable from `func.body`) into explicit unsigned 64-bit integer arithmetic
/// (`OpConvertPtrToU` + `CastI64ToU64` + `OpIMul` + `OpIAdd` + `OpConvertUToPtr`) so VSL 2.0
/// never emits `OpPtrAccessChain` (avoiding NVIDIA's +4 GB negative-offset driver bug).
/// Records the least-common-ancestor (LCA) structured scope path for every instruction
/// reachable from `stmts` so dominance-sensitive passes (`LowerBdaOffsetsPass` and
/// `SimplifyAndGvnPass`) never emit definitions or GVN leaders into non-dominating branches.
fn record_inst_tree_lca(
    func: &IrFunction,
    id: InstId,
    scope_path: &[usize],
    lca_paths: &mut IndexMap<InstId, Vec<usize>>,
    stmt_visited: &mut IndexSet<InstId>,
) {
    if !stmt_visited.insert(id) {
        return;
    }
    if let Some(existing) = lca_paths.get_mut(&id) {
        let common = existing
            .iter()
            .zip(scope_path.iter())
            .take_while(|(a, b)| a == b)
            .count();
        existing.truncate(common);
    } else {
        lca_paths.insert(id, scope_path.to_vec());
    }
    if let InstData::Phi { incoming } = &func.insts[id].data {
        // Incoming operands are consumed at the merge scope; clamp their LCA to at most
        // `scope_path` without recursing into predecessor-only subtrees.
        for &(v, _) in incoming {
            if let Some(existing) = lca_paths.get_mut(&v) {
                let common = existing
                    .iter()
                    .zip(scope_path.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                existing.truncate(common);
            } else {
                lca_paths.insert(v, scope_path.to_vec());
            }
        }
        return;
    }
    for op in func.insts[id].data.operands() {
        record_inst_tree_lca(func, op, scope_path, lca_paths, stmt_visited);
    }
}

fn collect_stmts_lca_paths(
    func: &IrFunction,
    stmts: &[StructuredStmt],
    scope_path: &mut Vec<usize>,
    lca_paths: &mut IndexMap<InstId, Vec<usize>>,
) {
    for (stmt_idx, stmt) in stmts.iter().enumerate() {
        match stmt {
            StructuredStmt::Eval(id) | StructuredStmt::Return(Some(id)) => {
                let mut visited = IndexSet::new();
                record_inst_tree_lca(func, *id, scope_path, lca_paths, &mut visited);
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let mut visited = IndexSet::new();
                record_inst_tree_lca(func, *cond, scope_path, lca_paths, &mut visited);
                scope_path.push(stmt_idx * 2);
                collect_stmts_lca_paths(func, then_body, scope_path, lca_paths);
                scope_path.pop();
                scope_path.push(stmt_idx * 2 + 1);
                collect_stmts_lca_paths(func, else_body, scope_path, lca_paths);
                scope_path.pop();
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                let mut visited = IndexSet::new();
                record_inst_tree_lca(func, *selector, scope_path, lca_paths, &mut visited);
                for (case_idx, (_, case_body)) in cases.iter().enumerate() {
                    scope_path.push(stmt_idx.saturating_mul(1024) + case_idx);
                    collect_stmts_lca_paths(func, case_body, scope_path, lca_paths);
                    scope_path.pop();
                }
                scope_path.push(stmt_idx.saturating_mul(1024) + cases.len());
                collect_stmts_lca_paths(func, default, scope_path, lca_paths);
                scope_path.pop();
            }
            StructuredStmt::Loop {
                body,
                continuing,
                induction_init,
                ..
            } => {
                if let Some(ind_id) = induction_init {
                    let mut visited = IndexSet::new();
                    record_inst_tree_lca(func, *ind_id, scope_path, lca_paths, &mut visited);
                }
                scope_path.push(stmt_idx * 2);
                collect_stmts_lca_paths(func, body, scope_path, lca_paths);
                scope_path.pop();
                scope_path.push(stmt_idx * 2 + 1);
                collect_stmts_lca_paths(func, continuing, scope_path, lca_paths);
                scope_path.pop();
            }
            StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(None) => {}
        }
    }
}

pub struct LowerBdaOffsetsPass;

impl IrPass for LowerBdaOffsetsPass {
    fn name(&self) -> &'static str {
        "lower-bda-offsets"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut lca_paths: IndexMap<InstId, Vec<usize>> = IndexMap::new();
        let mut scope_path: Vec<usize> = Vec::new();
        collect_stmts_lca_paths(func, &func.body, &mut scope_path, &mut lca_paths);

        let mut bda_ctx = BdaLowerCtx {
            visited: IndexSet::new(),
            lca_paths,
            deferred_evals: IndexMap::new(),
            lowered_any: false,
        };
        let body = std::mem::take(&mut func.body);
        func.body = rewrite_stmts_bda(func, body, &mut scope_path, &mut bda_ctx);

        if bda_ctx.lowered_any {
            func.uses_bda = true;
        }
    }
}

struct BdaLowerCtx {
    visited: IndexSet<InstId>,
    lca_paths: IndexMap<InstId, Vec<usize>>,
    deferred_evals: IndexMap<Vec<usize>, Vec<StructuredStmt>>,
    lowered_any: bool,
}

fn lower_bda_in_inst(
    func: &mut IrFunction,
    id: InstId,
    is_root_eval: bool,
    scope_path: &[usize],
    ctx: &mut BdaLowerCtx,
    prefix_evals: &mut Vec<StructuredStmt>,
) {
    if !ctx.visited.insert(id) {
        return;
    }
    // Never recurse into `Phi` incoming operands: predecessor values belong to their
    // originating branches, not the merge block.
    if matches!(func.insts[id].data, InstData::Phi { .. }) {
        return;
    }

    let operands = func.insts[id].data.operands();
    for op in operands {
        lower_bda_in_inst(func, op, false, scope_path, ctx, prefix_evals);
    }

    if let InstData::BdaOffset {
        base_ptr,
        index_i64,
        stride_bytes,
        align,
    } = func.insts[id].data
    {
        let clamped_align = if stride_bytes == 0 {
            align
        } else {
            align.min(1u32 << stride_bytes.trailing_zeros().min(31))
        };
        let ty = func.insts[id].ty;
        let base_u64 = func.emit(None, InstData::ConvertPtrToU(base_ptr));
        let index_u64 = func.emit(None, InstData::CastI64ToU64(index_i64));
        let stride_const = func.emit(None, InstData::ConstU64(stride_bytes));
        let byte_off = func.emit(None, InstData::IMul64(index_u64, stride_const));
        let target_u64 = func.emit(None, InstData::IAdd64(base_u64, byte_off));
        func.insts[id].data = InstData::ConvertUToPtr {
            addr_u64: target_u64,
            align: clamped_align,
        };
        func.insts[id].ty = ty;
        if let IrType::Pointer {
            align: ty_align, ..
        } = &mut func.insts[id].ir_ty
        {
            *ty_align = clamped_align;
        }

        let target_path = ctx
            .lca_paths
            .get(&id)
            .filter(|lca| lca.len() < scope_path.len())
            .cloned();
        let emit_self = !is_root_eval || target_path.is_some();
        let eval_sink = if let Some(ancestor_path) = target_path {
            ctx.deferred_evals.entry(ancestor_path).or_default()
        } else {
            prefix_evals
        };
        eval_sink.push(StructuredStmt::Eval(base_u64));
        eval_sink.push(StructuredStmt::Eval(index_u64));
        eval_sink.push(StructuredStmt::Eval(stride_const));
        eval_sink.push(StructuredStmt::Eval(byte_off));
        eval_sink.push(StructuredStmt::Eval(target_u64));
        if emit_self {
            eval_sink.push(StructuredStmt::Eval(id));
        }
        ctx.lowered_any = true;
    }
}

fn rewrite_stmts_bda(
    func: &mut IrFunction,
    stmts: Vec<StructuredStmt>,
    scope_path: &mut Vec<usize>,
    ctx: &mut BdaLowerCtx,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for (stmt_idx, stmt) in stmts.into_iter().enumerate() {
        match stmt {
            StructuredStmt::Eval(id) => {
                lower_bda_in_inst(func, id, true, scope_path, ctx, &mut out);
                out.push(StructuredStmt::Eval(id));
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                lower_bda_in_inst(func, cond, false, scope_path, ctx, &mut out);
                scope_path.push(stmt_idx * 2);
                let then_body = rewrite_stmts_bda(func, then_body, scope_path, ctx);
                scope_path.pop();
                scope_path.push(stmt_idx * 2 + 1);
                let else_body = rewrite_stmts_bda(func, else_body, scope_path, ctx);
                scope_path.pop();
                if let Some(hoisted) = ctx.deferred_evals.shift_remove(scope_path) {
                    out.extend(hoisted);
                }
                out.push(StructuredStmt::If {
                    cond,
                    then_body,
                    else_body,
                });
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                lower_bda_in_inst(func, selector, false, scope_path, ctx, &mut out);
                let case_count = cases.len();
                let mut new_cases = Vec::with_capacity(case_count);
                for (case_idx, (k, b)) in cases.into_iter().enumerate() {
                    scope_path.push(stmt_idx.saturating_mul(1024) + case_idx);
                    let new_b = rewrite_stmts_bda(func, b, scope_path, ctx);
                    scope_path.pop();
                    new_cases.push((k, new_b));
                }
                scope_path.push(stmt_idx.saturating_mul(1024) + case_count);
                let default = rewrite_stmts_bda(func, default, scope_path, ctx);
                scope_path.pop();
                if let Some(hoisted) = ctx.deferred_evals.shift_remove(scope_path) {
                    out.extend(hoisted);
                }
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: new_cases,
                    default,
                });
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                if let Some(ind_id) = induction_init {
                    lower_bda_in_inst(func, ind_id, false, scope_path, ctx, &mut out);
                }
                scope_path.push(stmt_idx * 2);
                let body = rewrite_stmts_bda(func, body, scope_path, ctx);
                scope_path.pop();
                scope_path.push(stmt_idx * 2 + 1);
                let continuing = rewrite_stmts_bda(func, continuing, scope_path, ctx);
                scope_path.pop();
                if let Some(hoisted) = ctx.deferred_evals.shift_remove(scope_path) {
                    out.extend(hoisted);
                }
                out.push(StructuredStmt::Loop {
                    body,
                    continuing,
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });
            }
            StructuredStmt::Return(opt_val) => {
                if let Some(val_id) = opt_val {
                    lower_bda_in_inst(func, val_id, false, scope_path, ctx, &mut out);
                }
                out.push(StructuredStmt::Return(opt_val));
            }
            other @ (StructuredStmt::Break | StructuredStmt::Continue) => out.push(other),
        }
    }
    out
}

/// Propagates `NonUniform` taint across `NonUniform(idx)` -> `ResourceAccessChain` -> `ResourceLoad`
/// (and through `Select`/`Phi` merges) per `SPV_EXT_descriptor_indexing` (prevents RADV bindless corruption).
pub struct PropagateNonUniformPass;

fn resolve_local_slot(func: &IrFunction, mut ptr: InstId) -> Option<u32> {
    let mut visited = IndexSet::new();
    while visited.insert(ptr) {
        match func.insts[ptr].data {
            InstData::LocalAddr { local_index } => return Some(local_index),
            InstData::AccessChain { base_ptr, .. } => ptr = base_ptr,
            _ => return None,
        }
    }
    None
}

impl IrPass for PropagateNonUniformPass {
    fn name(&self) -> &'static str {
        "propagate-nonuniform"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut tainted_locals: IndexSet<u32> = IndexSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            let ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
            for &id in &ids {
                if let InstData::LocalStore { ptr, value } = func.insts[id].data
                    && (func.insts[value].non_uniform || func.insts[ptr].non_uniform)
                    && let Some(slot) = resolve_local_slot(func, ptr)
                    && tainted_locals.insert(slot)
                {
                    changed = true;
                }
            }
            for id in ids {
                if func.insts[id].non_uniform {
                    continue;
                }
                let tainted = match &func.insts[id].data {
                    InstData::NonUniform(_) => true,
                    InstData::CastI64ToU64(v)
                    | InstData::CastToU32(v)
                    | InstData::CastToI32(v)
                    | InstData::CastToF32(v)
                    | InstData::CastToI64(v)
                    | InstData::CastToU64(v)
                    | InstData::BitNot32(v)
                    | InstData::LogicalNot(v)
                    | InstData::ExtractComponent { vec: v, .. } => func.insts[*v].non_uniform,
                    InstData::IAdd32(a, b)
                    | InstData::ISub32(a, b)
                    | InstData::IMul32(a, b)
                    | InstData::IDiv32(a, b)
                    | InstData::IRem32(a, b)
                    | InstData::BitAnd32(a, b)
                    | InstData::BitOr32(a, b)
                    | InstData::BitXor32(a, b)
                    | InstData::Shl32(a, b)
                    | InstData::Shr32(a, b)
                    | InstData::IAdd64(a, b)
                    | InstData::IMul64(a, b)
                    | InstData::ICmpEq(a, b)
                    | InstData::ICmpNe(a, b)
                    | InstData::ICmpLt(a, b)
                    | InstData::ICmpLe(a, b)
                    | InstData::ICmpGt(a, b)
                    | InstData::ICmpGe(a, b)
                    | InstData::UCmpLt(a, b)
                    | InstData::LogicalAnd(a, b)
                    | InstData::LogicalOr(a, b)
                    | InstData::MakeVec2(a, b)
                    | InstData::VectorTimesScalar { vec: a, scalar: b }
                    | InstData::MatrixTimesVector { mat: a, vec: b }
                    | InstData::MatrixTimesMatrix { lhs: a, rhs: b }
                    | InstData::VectorShuffle { lhs: a, rhs: b, .. }
                    | InstData::CompositeInsert {
                        composite: a,
                        value: b,
                        ..
                    } => func.insts[*a].non_uniform || func.insts[*b].non_uniform,
                    InstData::MakeVec3(a, b, c)
                    | InstData::BitFieldExtract {
                        base: a,
                        offset: b,
                        count: c,
                        ..
                    }
                    | InstData::Select {
                        cond: a,
                        on_true: b,
                        on_false: c,
                    } => {
                        func.insts[*a].non_uniform
                            || func.insts[*b].non_uniform
                            || func.insts[*c].non_uniform
                    }
                    InstData::MakeVec4(a, b, c, d)
                    | InstData::BitFieldInsert {
                        base: a,
                        insert: b,
                        offset: c,
                        count: d,
                    } => {
                        func.insts[*a].non_uniform
                            || func.insts[*b].non_uniform
                            || func.insts[*c].non_uniform
                            || func.insts[*d].non_uniform
                    }
                    InstData::GlslStd450 { args, .. } => {
                        args.iter().any(|&a| func.insts[a].non_uniform)
                    }
                    InstData::ResourceAccessChain { index, .. } => func.insts[*index].non_uniform,
                    InstData::ResourceLoad { ptr } => func.insts[*ptr].non_uniform,
                    InstData::ImageSample { image_inst, .. }
                    | InstData::ImageSampleImplicitLod { image_inst, .. }
                    | InstData::ImageSampleGrad { image_inst, .. }
                    | InstData::ImageSampleDrefExplicitLod { image_inst, .. }
                    | InstData::ImageGather { image_inst, .. } => {
                        image_inst.is_some_and(|img| func.insts[img].non_uniform)
                    }
                    InstData::Phi { incoming } => {
                        incoming.iter().any(|&(v, _)| func.insts[v].non_uniform)
                    }
                    InstData::CompositeConstruct { elems } => {
                        elems.iter().any(|&e| func.insts[e].non_uniform)
                    }
                    InstData::AccessChain { base_ptr, indices } => {
                        func.insts[*base_ptr].non_uniform
                            || indices.iter().any(|&i| func.insts[i].non_uniform)
                    }
                    InstData::DescriptorAccessChain { indices, .. } => {
                        indices.iter().any(|&i| func.insts[i].non_uniform)
                    }
                    InstData::LocalLoad { ptr } => {
                        func.insts[*ptr].non_uniform
                            || resolve_local_slot(func, *ptr)
                                .is_some_and(|slot| tainted_locals.contains(&slot))
                    }
                    InstData::FConvert { value: v, .. }
                    | InstData::Bitcast { value: v, .. }
                    | InstData::BitCount(v)
                    | InstData::CompositeExtract { composite: v, .. }
                    | InstData::DPdx(v)
                    | InstData::DPdy(v)
                    | InstData::Fwidth(v) => func.insts[*v].non_uniform,
                    InstData::ConstBool(_)
                    | InstData::ConstI32(_)
                    | InstData::ConstU32(_)
                    | InstData::ConstF32(_)
                    | InstData::ConstF16(_)
                    | InstData::ConstI64(_)
                    | InstData::ConstU64(_)
                    | InstData::FAdd32(_, _)
                    | InstData::FSub32(_, _)
                    | InstData::FMul32(_, _)
                    | InstData::FDiv32(_, _)
                    | InstData::FCmpEq(_, _)
                    | InstData::FCmpNe(_, _)
                    | InstData::FCmpLt(_, _)
                    | InstData::FCmpLe(_, _)
                    | InstData::FCmpGt(_, _)
                    | InstData::FCmpGe(_, _)
                    | InstData::LocalAddr { .. }
                    | InstData::LocalStore { .. }
                    | InstData::DescriptorStore { .. }
                    | InstData::Param { .. }
                    | InstData::FunctionCall { .. }
                    | InstData::BdaOffset { .. }
                    | InstData::ConvertPtrToU(_)
                    | InstData::ConvertUToPtr { .. }
                    | InstData::BdaLoad { .. }
                    | InstData::BdaStore { .. }
                    | InstData::ImageQuerySize { .. }
                    | InstData::ImageQueryLevels { .. }
                    | InstData::ImageTexelPointer { .. }
                    | InstData::ImageStore { .. }
                    | InstData::WorkgroupAddr { .. }
                    | InstData::WorkgroupLoad { .. }
                    | InstData::WorkgroupStore { .. }
                    | InstData::AtomicAdd { .. }
                    | InstData::AtomicMin { .. }
                    | InstData::AtomicMax { .. }
                    | InstData::AtomicCas { .. }
                    | InstData::WorkgroupBarrier
                    | InstData::MemoryBarrier { .. }
                    | InstData::SubgroupBallot(_)
                    | InstData::SubgroupBallotBitCount(_)
                    | InstData::SubgroupBallotExclusiveBitCount(_)
                    | InstData::SubgroupBallotInclusiveBitCount(_)
                    | InstData::SubgroupBroadcastFirst(_)
                    | InstData::SubgroupAdd(_)
                    | InstData::SubgroupMin(_)
                    | InstData::SubgroupMax(_)
                    | InstData::SubgroupExclusiveAdd(_)
                    | InstData::SubgroupInclusiveAdd(_)
                    | InstData::SubgroupShuffle { .. }
                    | InstData::SubgroupShuffleXor { .. }
                    | InstData::SubgroupAny(_)
                    | InstData::SubgroupAll(_)
                    | InstData::SubgroupElect
                    | InstData::QuadSwapHorizontal(_)
                    | InstData::QuadSwapVertical(_)
                    | InstData::QuadSwapDiagonal(_)
                    | InstData::IsNan(_)
                    | InstData::IsInf(_)
                    | InstData::SpecConstant { .. }
                    | InstData::SpecConstantOp { .. }
                    | InstData::DescriptorLoad { .. }
                    | InstData::LoadBuiltin { .. }
                    | InstData::LoadStageInput { .. }
                    | InstData::StoreStageOutput { .. }
                    | InstData::StoreBuiltin { .. }
                    | InstData::DemoteToHelperInvocation
                    | InstData::LoadPushConstant { .. }
                    | InstData::RayQueryInitialize { .. }
                    | InstData::RayQueryProceed(_)
                    | InstData::RayQueryCommittedType(_)
                    | InstData::RayQueryCandidateType(_)
                    | InstData::RayQueryConfirmIntersection(_)
                    | InstData::RayQueryTerminate(_)
                    | InstData::RayQueryGenerateIntersection { .. }
                    | InstData::RayQueryCommittedT(_)
                    | InstData::RayQueryCandidateT(_)
                    | InstData::RayQueryCommittedInstanceId(_)
                    | InstData::RayQueryCommittedInstanceCustomIndex(_)
                    | InstData::RayQueryCommittedPrimitiveIndex(_)
                    | InstData::RayQueryCommittedGeometryIndex(_)
                    | InstData::RayQueryCommittedObjectToWorld(_)
                    | InstData::RayQueryCommittedWorldToObject(_)
                    | InstData::RayQueryCommittedFrontFace(_)
                    | InstData::RayQueryCommittedBarycentrics(_)
                    | InstData::TraceRayKhr { .. }
                    | InstData::IgnoreIntersectionKhr
                    | InstData::TerminateRayKhr
                    | InstData::ReportIntersectionKhr { .. }
                    | InstData::ExecuteCallableKhr { .. }
                    | InstData::EmitMeshTasksExt { .. }
                    | InstData::SetMeshOutputsExt { .. }
                    | InstData::EmitVertex
                    | InstData::EndPrimitive
                    | InstData::LoadStageVar { .. }
                    | InstData::StoreStageVar { .. }
                    | InstData::HitObjectTraceRay { .. }
                    | InstData::HitObjectRecordEmpty(_)
                    | InstData::HitObjectIsHit(_)
                    | InstData::HitObjectGetRayTMax(_)
                    | InstData::HitObjectGetSbtRecordIndex(_)
                    | InstData::HitObjectExecuteShader { .. }
                    | InstData::ReorderThreadWithHint { .. }
                    | InstData::ReorderThreadWithHitObject { .. }
                    | InstData::RayQueryTriangleVertexPositions { .. }
                    | InstData::Nop => false,
                };
                if tainted {
                    func.insts[id].non_uniform = true;
                    changed = true;
                }
            }
        }
    }
}

/// Canonical value-numbering key for pure, dominance-safe SSA instructions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GvnKey {
    ConstBool(bool),
    ConstI32(i32),
    ConstU32(u32),
    ConstF32(u32),
    ConstI64(i64),
    ConstU64(u64),
    CommBinOp(u8, InstId, InstId),
    OrdBinOp(u8, InstId, InstId),
    UnaryOp(u8, InstId),
    Select(InstId, InstId, InstId),
    ConvertUToPtr(InstId, u32, IrType),
    LoadBuiltin(crate::ssa::BuiltinKind, Option<u32>),
    LoadPushConstant(u32),
    MakeVec2(InstId, InstId),
}

fn canonical_pair(a: InstId, b: InstId) -> (InstId, InstId) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Performs algebraic simplification, trivial-`Phi` elimination, and dominance-scoped
/// Global Value Numbering (CSE) so VSL 2.0 produces compact SPIR-V without runtime `spirv-opt`.
pub struct SimplifyAndGvnPass;

impl IrPass for SimplifyAndGvnPass {
    fn name(&self) -> &'static str {
        "simplify-and-gvn"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut remap: IndexMap<InstId, InstId> = IndexMap::new();
        let mut scopes: Vec<IndexMap<GvnKey, InstId>> = vec![IndexMap::new()];

        // 1) Algebraic identities & trivial Phi reduction across the instruction arena.
        let ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
        for id in ids {
            let remapped = remap_inst_data(&func.insts[id].data, &remap);
            func.insts[id].data = remapped;

            if let Some(replacement) = simplify_inst(func, id) {
                let target = remap_id(replacement, &remap);
                remap.insert(id, target);
                func.insts[id].data = InstData::Nop;
                func.insts[id].non_uniform = false;
            }
        }

        if !remap.is_empty() {
            let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
            for id in all_ids {
                if !remap.contains_key(&id) {
                    let updated = remap_inst_data(&func.insts[id].data, &remap);
                    func.insts[id].data = updated;
                }
            }
            let body = std::mem::take(&mut func.body);
            func.body = remap_stmts_ids(body, &remap);
        }

        // 2) Dominance-scoped GVN / CSE over the structured control-flow tree.
        let mut lca_paths: IndexMap<InstId, Vec<usize>> = IndexMap::new();
        let mut scope_path: Vec<usize> = Vec::new();
        collect_stmts_lca_paths(func, &func.body, &mut scope_path, &mut lca_paths);

        let mut gvn_remap: IndexMap<InstId, InstId> = IndexMap::new();
        let mut visited: IndexSet<InstId> = IndexSet::new();
        let body = std::mem::take(&mut func.body);
        let new_body = gvn_stmts(
            func,
            body,
            &mut scopes,
            &lca_paths,
            &mut gvn_remap,
            &mut visited,
        );
        func.body = if gvn_remap.is_empty() {
            new_body
        } else {
            let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
            for id in all_ids {
                if gvn_remap.contains_key(&id) {
                    func.insts[id].data = InstData::Nop;
                    func.insts[id].non_uniform = false;
                } else {
                    let updated = remap_inst_data(&func.insts[id].data, &gvn_remap);
                    func.insts[id].data = updated;
                }
            }
            remap_stmts_ids(new_body, &gvn_remap)
        };
    }
}

fn is_zero_32(func: &IrFunction, id: InstId) -> bool {
    matches!(
        func.insts[id].data,
        InstData::ConstI32(0) | InstData::ConstU32(0)
    )
}

fn is_one_32(func: &IrFunction, id: InstId) -> bool {
    matches!(
        func.insts[id].data,
        InstData::ConstI32(1) | InstData::ConstU32(1)
    )
}

fn is_zero_64(func: &IrFunction, id: InstId) -> bool {
    matches!(
        func.insts[id].data,
        InstData::ConstI64(0) | InstData::ConstU64(0)
    )
}

fn is_one_64(func: &IrFunction, id: InstId) -> bool {
    matches!(
        func.insts[id].data,
        InstData::ConstI64(1) | InstData::ConstU64(1)
    )
}

fn simplify_inst(func: &IrFunction, id: InstId) -> Option<InstId> {
    match &func.insts[id].data {
        // 32-bit integer algebraic identities
        InstData::IAdd32(a, b) => {
            if is_zero_32(func, *a) {
                Some(*b)
            } else if is_zero_32(func, *b) {
                Some(*a)
            } else {
                None
            }
        }
        InstData::ISub32(a, b) if is_zero_32(func, *b) => Some(*a),
        InstData::IMul32(a, b) => {
            let res_ty = &func.insts[id].ir_ty;
            if is_one_32(func, *a) {
                Some(*b)
            } else if is_one_32(func, *b)
                || (is_zero_32(func, *a) && &func.insts[*a].ir_ty == res_ty)
            {
                Some(*a)
            } else if is_zero_32(func, *b) && &func.insts[*b].ir_ty == res_ty {
                Some(*b)
            } else {
                None
            }
        }
        InstData::IDiv32(a, b) if is_one_32(func, *b) => Some(*a),
        InstData::BitAnd32(a, b) => {
            let res_ty = &func.insts[id].ir_ty;
            if a == b || (is_zero_32(func, *a) && &func.insts[*a].ir_ty == res_ty) {
                Some(*a)
            } else if is_zero_32(func, *b) && &func.insts[*b].ir_ty == res_ty {
                Some(*b)
            } else {
                None
            }
        }
        InstData::BitOr32(a, b) => {
            if a == b || is_zero_32(func, *b) {
                Some(*a)
            } else if is_zero_32(func, *a) {
                Some(*b)
            } else {
                None
            }
        }
        InstData::BitXor32(a, b) => {
            if is_zero_32(func, *b) {
                Some(*a)
            } else if is_zero_32(func, *a) {
                Some(*b)
            } else {
                None
            }
        }
        InstData::Shl32(a, b) | InstData::Shr32(a, b) if is_zero_32(func, *b) => Some(*a),

        // 64-bit integer & BDA pointer round-trip identities (eliminates 0-offset BDA chains)
        InstData::IAdd64(a, b) => {
            if is_zero_64(func, *a) {
                Some(*b)
            } else if is_zero_64(func, *b) {
                Some(*a)
            } else {
                None
            }
        }
        InstData::IMul64(a, b) => {
            let res_ty = &func.insts[id].ir_ty;
            if is_one_64(func, *a) {
                Some(*b)
            } else if is_one_64(func, *b)
                || (is_zero_64(func, *a) && &func.insts[*a].ir_ty == res_ty)
            {
                Some(*a)
            } else if is_zero_64(func, *b) && &func.insts[*b].ir_ty == res_ty {
                Some(*b)
            } else {
                None
            }
        }
        InstData::ConvertPtrToU(p) => {
            if let InstData::ConvertUToPtr { addr_u64, .. } = func.insts[*p].data {
                Some(addr_u64)
            } else {
                None
            }
        }
        InstData::ConvertUToPtr { addr_u64, .. } => {
            if let InstData::ConvertPtrToU(orig_ptr) = func.insts[*addr_u64].data
                && func.insts[orig_ptr].ir_ty == func.insts[id].ir_ty
            {
                Some(orig_ptr)
            } else {
                None
            }
        }

        // Boolean, Select, and trivial Phi simplifications
        InstData::LogicalAnd(a, b) => match (&func.insts[*a].data, &func.insts[*b].data) {
            (InstData::ConstBool(true), _) => Some(*b),
            (_, InstData::ConstBool(true)) | (InstData::ConstBool(false), _) => Some(*a),
            (_, InstData::ConstBool(false)) => Some(*b),
            _ if a == b => Some(*a),
            _ => None,
        },
        InstData::LogicalOr(a, b) => match (&func.insts[*a].data, &func.insts[*b].data) {
            (InstData::ConstBool(false), _) => Some(*b),
            (_, InstData::ConstBool(false)) | (InstData::ConstBool(true), _) => Some(*a),
            (_, InstData::ConstBool(true)) => Some(*b),
            _ if a == b => Some(*a),
            _ => None,
        },
        InstData::LogicalNot(a) => {
            if let InstData::LogicalNot(inner) = func.insts[*a].data {
                Some(inner)
            } else {
                None
            }
        }
        InstData::Select {
            on_true, on_false, ..
        } if on_true == on_false => Some(*on_true),
        InstData::Phi { incoming } => {
            let mut unique: Option<InstId> = None;
            for &(v, _) in incoming {
                if v == id {
                    continue;
                }
                match unique {
                    None => unique = Some(v),
                    Some(prev) if prev == v => {}
                    Some(_) => return None,
                }
            }
            unique
        }
        _ => None,
    }
}

fn gvn_key_for(data: &InstData, ir_ty: &IrType) -> Option<GvnKey> {
    match *data {
        InstData::ConstBool(v) => Some(GvnKey::ConstBool(v)),
        InstData::ConstI32(v) => Some(GvnKey::ConstI32(v)),
        InstData::ConstU32(v) => Some(GvnKey::ConstU32(v)),
        InstData::ConstF32(v) => Some(GvnKey::ConstF32(v)),
        InstData::ConstI64(v) => Some(GvnKey::ConstI64(v)),
        InstData::ConstU64(v) => Some(GvnKey::ConstU64(v)),
        InstData::IAdd32(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(0, x, y))
        }
        InstData::IMul32(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(1, x, y))
        }
        InstData::FAdd32(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(2, x, y))
        }
        InstData::FMul32(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(3, x, y))
        }
        InstData::ICmpEq(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(4, x, y))
        }
        InstData::ICmpNe(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(5, x, y))
        }
        InstData::LogicalAnd(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(6, x, y))
        }
        InstData::LogicalOr(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(7, x, y))
        }
        InstData::BitAnd32(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(8, x, y))
        }
        InstData::BitOr32(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(9, x, y))
        }
        InstData::BitXor32(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(10, x, y))
        }
        InstData::IAdd64(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(11, x, y))
        }
        InstData::IMul64(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(12, x, y))
        }
        InstData::FCmpEq(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(13, x, y))
        }
        InstData::FCmpNe(a, b) => {
            let (x, y) = canonical_pair(a, b);
            Some(GvnKey::CommBinOp(14, x, y))
        }
        InstData::FCmpLt(a, b) => Some(GvnKey::OrdBinOp(12, a, b)),
        InstData::FCmpLe(a, b) => Some(GvnKey::OrdBinOp(13, a, b)),
        InstData::FCmpGt(a, b) => Some(GvnKey::OrdBinOp(14, a, b)),
        InstData::FCmpGe(a, b) => Some(GvnKey::OrdBinOp(15, a, b)),
        InstData::ISub32(a, b) => Some(GvnKey::OrdBinOp(0, a, b)),
        InstData::IDiv32(a, b) => Some(GvnKey::OrdBinOp(1, a, b)),
        InstData::IRem32(a, b) => Some(GvnKey::OrdBinOp(2, a, b)),
        InstData::FSub32(a, b) => Some(GvnKey::OrdBinOp(3, a, b)),
        InstData::FDiv32(a, b) => Some(GvnKey::OrdBinOp(4, a, b)),
        InstData::ICmpLt(a, b) => Some(GvnKey::OrdBinOp(5, a, b)),
        InstData::ICmpLe(a, b) => Some(GvnKey::OrdBinOp(6, a, b)),
        InstData::ICmpGt(a, b) => Some(GvnKey::OrdBinOp(7, a, b)),
        InstData::ICmpGe(a, b) => Some(GvnKey::OrdBinOp(8, a, b)),
        InstData::UCmpLt(a, b) => Some(GvnKey::OrdBinOp(9, a, b)),
        InstData::Shl32(a, b) => Some(GvnKey::OrdBinOp(10, a, b)),
        InstData::Shr32(a, b) => Some(GvnKey::OrdBinOp(11, a, b)),
        InstData::LogicalNot(v) => Some(GvnKey::UnaryOp(0, v)),
        InstData::BitNot32(v) => Some(GvnKey::UnaryOp(1, v)),
        InstData::CastI64ToU64(v) => Some(GvnKey::UnaryOp(2, v)),
        InstData::CastToU32(v) => Some(GvnKey::UnaryOp(3, v)),
        InstData::ConvertPtrToU(v) => Some(GvnKey::UnaryOp(4, v)),
        InstData::Select {
            cond,
            on_true,
            on_false,
        } => Some(GvnKey::Select(cond, on_true, on_false)),
        InstData::ConvertUToPtr { addr_u64, align } => {
            Some(GvnKey::ConvertUToPtr(addr_u64, align, ir_ty.clone()))
        }
        InstData::LoadBuiltin { kind, component } => Some(GvnKey::LoadBuiltin(kind, component)),
        InstData::LoadPushConstant { field_index } => Some(GvnKey::LoadPushConstant(field_index)),
        InstData::MakeVec2(a, b) => Some(GvnKey::MakeVec2(a, b)),
        _ => None,
    }
}

fn lookup_gvn(scopes: &[IndexMap<GvnKey, InstId>], key: &GvnKey) -> Option<(InstId, usize)> {
    for (depth, scope) in scopes.iter().enumerate().rev() {
        if let Some(&id) = scope.get(key) {
            return Some((id, depth));
        }
    }
    None
}

fn gvn_visit_inst(
    func: &mut IrFunction,
    id: InstId,
    scopes: &mut [IndexMap<GvnKey, InstId>],
    lca_paths: &IndexMap<InstId, Vec<usize>>,
    remap: &mut IndexMap<InstId, InstId>,
    visited: &mut IndexSet<InstId>,
) -> InstId {
    if let Some(&existing) = remap.get(&id) {
        return existing;
    }
    if !visited.insert(id) {
        return id;
    }
    // Never recurse into `Phi` incoming operands in the merge block's GVN scope: incoming
    // values may be defined in non-dominating predecessor branches and must not enter the
    // merge block's value table.
    if matches!(func.insts[id].data, InstData::Phi { .. }) {
        let remapped = remap_inst_data(&func.insts[id].data, remap);
        func.insts[id].data = remapped;
        return id;
    }
    let operands = func.insts[id].data.operands();
    for op in operands {
        let _ = gvn_visit_inst(func, op, scopes, lca_paths, remap, visited);
    }
    let remapped = remap_inst_data(&func.insts[id].data, remap);
    func.insts[id].data = remapped;

    // Never deduplicate NonUniform-tainted nodes with uniform nodes.
    if func.insts[id].non_uniform {
        return id;
    }
    let current_depth = scopes.len().saturating_sub(1);
    let id_lca_depth = lca_paths.get(&id).map_or(0, Vec::len);
    if let Some(key) = gvn_key_for(&func.insts[id].data, &func.insts[id].ir_ty) {
        if let Some((leader, leader_depth)) = lookup_gvn(scopes, &key) {
            // Only replace `id` globally when `leader` lives in a scope that dominates
            // every use of `id` (never across sibling branches).
            if leader_depth <= id_lca_depth {
                remap.insert(id, leader);
                return leader;
            }
            return id;
        }
        // Only register `id` as a leader in the current scope when all uses of `id` are
        // dominated by this scope (so sibling-shared nodes do not leak into child scopes).
        if id_lca_depth >= current_depth
            && let Some(current) = scopes.last_mut()
        {
            current.insert(key, id);
        }
    }
    id
}

fn gvn_stmts(
    func: &mut IrFunction,
    stmts: Vec<StructuredStmt>,
    scopes: &mut Vec<IndexMap<GvnKey, InstId>>,
    lca_paths: &IndexMap<InstId, Vec<usize>>,
    remap: &mut IndexMap<InstId, InstId>,
    visited: &mut IndexSet<InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                let leader = gvn_visit_inst(func, id, scopes, lca_paths, remap, visited);
                out.push(StructuredStmt::Eval(leader));
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond = gvn_visit_inst(func, cond, scopes, lca_paths, remap, visited);
                scopes.push(IndexMap::new());
                let then_body = gvn_stmts(func, then_body, scopes, lca_paths, remap, visited);
                scopes.pop();
                scopes.push(IndexMap::new());
                let else_body = gvn_stmts(func, else_body, scopes, lca_paths, remap, visited);
                scopes.pop();
                out.push(StructuredStmt::If {
                    cond,
                    then_body,
                    else_body,
                });
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                let selector = gvn_visit_inst(func, selector, scopes, lca_paths, remap, visited);
                let mut new_cases = Vec::with_capacity(cases.len());
                for (k, body) in cases {
                    scopes.push(IndexMap::new());
                    let b = gvn_stmts(func, body, scopes, lca_paths, remap, visited);
                    scopes.pop();
                    new_cases.push((k, b));
                }
                scopes.push(IndexMap::new());
                let default = gvn_stmts(func, default, scopes, lca_paths, remap, visited);
                scopes.pop();
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: new_cases,
                    default,
                });
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                let induction_init = induction_init
                    .map(|id| gvn_visit_inst(func, id, scopes, lca_paths, remap, visited));
                scopes.push(IndexMap::new());
                let body = gvn_stmts(func, body, scopes, lca_paths, remap, visited);
                scopes.pop();
                // Scope `continuing` separately so instructions emitted after a conditional
                // `Continue` in `body` never leak into `continuing`'s GVN table.
                scopes.push(IndexMap::new());
                let continuing = gvn_stmts(func, continuing, scopes, lca_paths, remap, visited);
                scopes.pop();
                out.push(StructuredStmt::Loop {
                    body,
                    continuing,
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });
            }
            StructuredStmt::Return(opt) => {
                let opt = opt.map(|id| gvn_visit_inst(func, id, scopes, lca_paths, remap, visited));
                out.push(StructuredStmt::Return(opt));
            }
            other @ (StructuredStmt::Break | StructuredStmt::Continue) => out.push(other),
        }
    }
    out
}

/// Removes dead pure SSA instructions whose results are never referenced, clears their
/// arena slots to `InstData::Nop`, and recomputes `uses_bda` and `uses_subgroups` strictly
/// from remaining live instructions so downstream SPIR-V emission never emits capabilities
/// or extensions for eliminated instructions.
pub struct DeadCodeEliminationPass;

impl IrPass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut live: IndexSet<InstId> = IndexSet::new();
        let mut worklist: Vec<InstId> = Vec::new();

        loop {
            let prev_len = live.len();
            let _ = collect_roots(&func.body, func, &mut live, &mut worklist);
            while let Some(id) = worklist.pop() {
                for operand in func.insts[id].data.operands() {
                    if live.insert(operand) {
                        worklist.push(operand);
                    }
                }
            }
            if live.len() == prev_len {
                break;
            }
        }

        let body = std::mem::take(&mut func.body);
        func.body = prune_stmts(func, body, &live);

        let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
        for id in all_ids {
            if !live.contains(&id) {
                func.insts[id].data = InstData::Nop;
                func.insts[id].non_uniform = false;
            }
        }

        func.uses_bda = func.insts.iter().any(|(_, inst)| {
            matches!(
                inst.data,
                InstData::BdaOffset { .. }
                    | InstData::ConvertPtrToU(_)
                    | InstData::ConvertUToPtr { .. }
                    | InstData::BdaLoad { .. }
                    | InstData::BdaStore { .. }
            )
        });
        func.uses_subgroups = func.insts.iter().any(|(_, inst)| {
            matches!(
                inst.data,
                InstData::SubgroupBallot(_)
                    | InstData::SubgroupBallotBitCount(_)
                    | InstData::SubgroupBallotExclusiveBitCount(_)
                    | InstData::SubgroupBallotInclusiveBitCount(_)
                    | InstData::SubgroupBroadcastFirst(_)
                    | InstData::SubgroupAdd(_)
                    | InstData::SubgroupMin(_)
                    | InstData::SubgroupMax(_)
                    | InstData::SubgroupExclusiveAdd(_)
                    | InstData::SubgroupInclusiveAdd(_)
                    | InstData::SubgroupShuffle { .. }
                    | InstData::SubgroupShuffleXor { .. }
                    | InstData::SubgroupAny(_)
                    | InstData::SubgroupAll(_)
                    | InstData::SubgroupElect
                    | InstData::LoadBuiltin {
                        kind: BuiltinKind::SubgroupLocalInvocationId | BuiltinKind::SubgroupSize,
                        ..
                    }
            )
        });
        func.uses_quad_control = func.insts.iter().any(|(_, inst)| {
            matches!(
                inst.data,
                InstData::QuadSwapHorizontal(_)
                    | InstData::QuadSwapVertical(_)
                    | InstData::QuadSwapDiagonal(_)
            )
        });
    }
}

fn has_live_merge_phi(
    func: &IrFunction,
    following: &[StructuredStmt],
    live: &IndexSet<InstId>,
) -> bool {
    for stmt in following {
        if let StructuredStmt::Eval(id) = stmt
            && matches!(func.insts[*id].data, InstData::Phi { .. })
        {
            if live.contains(id) {
                return true;
            }
        } else {
            break;
        }
    }
    false
}

fn collect_roots(
    stmts: &[StructuredStmt],
    func: &IrFunction,
    live: &mut IndexSet<InstId>,
    worklist: &mut Vec<InstId>,
) -> bool {
    let mut has_live = false;
    for (idx, stmt) in stmts.iter().enumerate() {
        let following = &stmts[idx + 1..];
        match stmt {
            StructuredStmt::Eval(id) => {
                if func.insts[*id].has_side_effects() || live.contains(id) {
                    has_live = true;
                    if live.insert(*id) {
                        worklist.push(*id);
                    }
                }
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let then_live = collect_roots(then_body, func, live, worklist);
                let else_live = collect_roots(else_body, func, live, worklist);
                if then_live || else_live || has_live_merge_phi(func, following, live) {
                    has_live = true;
                    if live.insert(*cond) {
                        worklist.push(*cond);
                    }
                }
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                let mut any_live = false;
                for (_, body) in cases {
                    any_live |= collect_roots(body, func, live, worklist);
                }
                any_live |= collect_roots(default, func, live, worklist);
                if any_live || has_live_merge_phi(func, following, live) {
                    has_live = true;
                    if live.insert(*selector) {
                        worklist.push(*selector);
                    }
                }
            }
            StructuredStmt::Loop {
                body,
                continuing,
                induction_init,
                ..
            } => {
                if let Some(ind_id) = induction_init
                    && live.insert(*ind_id)
                {
                    worklist.push(*ind_id);
                }
                let _ = collect_roots(body, func, live, worklist);
                let _ = collect_roots(continuing, func, live, worklist);
                has_live = true;
            }
            StructuredStmt::Return(Some(id)) => {
                has_live = true;
                if live.insert(*id) {
                    worklist.push(*id);
                }
                break;
            }
            StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(None) => {
                has_live = true;
                break;
            }
        }
    }
    has_live
}

fn prune_stmts(
    func: &IrFunction,
    stmts: Vec<StructuredStmt>,
    live: &IndexSet<InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut iter = stmts.into_iter();
    while let Some(stmt) = iter.next() {
        match stmt {
            StructuredStmt::Eval(id) => {
                if live.contains(&id) {
                    out.push(StructuredStmt::Eval(id));
                }
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let then_body = prune_stmts(func, then_body, live);
                let else_body = prune_stmts(func, else_body, live);
                if !then_body.is_empty()
                    || !else_body.is_empty()
                    || has_live_merge_phi(func, iter.as_slice(), live)
                {
                    out.push(StructuredStmt::If {
                        cond,
                        then_body,
                        else_body,
                    });
                }
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                let cases: Vec<(i64, Vec<StructuredStmt>)> = cases
                    .into_iter()
                    .map(|(k, b)| (k, prune_stmts(func, b, live)))
                    .collect();
                let default = prune_stmts(func, default, live);
                if cases.iter().any(|(_, b)| !b.is_empty())
                    || !default.is_empty()
                    || has_live_merge_phi(func, iter.as_slice(), live)
                {
                    out.push(StructuredStmt::Switch {
                        selector,
                        cases,
                        default,
                    });
                }
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => out.push(StructuredStmt::Loop {
                body: prune_stmts(func, body, live),
                continuing: prune_stmts(func, continuing, live),
                unroll,
                trip_count,
                induction_init,
                induction_step,
            }),
            term @ (StructuredStmt::Return(_)
            | StructuredStmt::Break
            | StructuredStmt::Continue) => {
                out.push(term);
                break;
            }
        }
    }
    out
}

/// Splits non-dynamically-indexed local struct and small-array (`len <= 8`) [`IrLocalVar`]s into
/// individual scalar/vector [`IrLocalVar`]s, rewriting `AccessChain` + `LocalLoad`/`LocalStore`
/// to direct `LocalLoad`/`LocalStore` so [`Mem2RegPass`] can promote them into SSA registers.
pub struct SroaPass;

impl IrPass for SroaPass {
    fn name(&self) -> &'static str {
        "sroa"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut changed = true;
        while changed {
            changed = sroa_step(func);
        }
    }
}

fn const_non_neg_index(func: &IrFunction, id: InstId) -> Option<u32> {
    match func.insts[id].data {
        InstData::ConstU32(k) => Some(k),
        InstData::ConstI32(k) if k >= 0 => Some(k.cast_unsigned()),
        _ => None,
    }
}

fn sroa_step(func: &mut IrFunction) -> bool {
    let mut addr_to_local: IndexMap<InstId, u32> = IndexMap::new();
    for (id, inst) in func.insts.iter() {
        if let InstData::LocalAddr { local_index } = inst.data {
            addr_to_local.insert(id, local_index);
        }
    }
    if addr_to_local.is_empty() {
        return false;
    }

    // Identify aggregate locals eligible for scalar replacement.
    let mut eligible: IndexMap<u32, Vec<(SmolStr, IrType)>> = IndexMap::new();
    for (idx, var) in func.local_vars.iter().enumerate() {
        let local_idx = idx as u32;
        match &var.ty {
            IrType::Struct(layout) if !layout.fields.is_empty() => {
                let parts = layout
                    .fields
                    .iter()
                    .map(|f| {
                        (
                            SmolStr::new(format!("{}.{}", var.name, f.name)),
                            f.ty.clone(),
                        )
                    })
                    .collect();
                eligible.insert(local_idx, parts);
            }
            IrType::Array { element, len } if *len > 0 && *len <= 8 => {
                let parts = (0..*len)
                    .map(|i| {
                        (
                            SmolStr::new(format!("{}[{i}]", var.name)),
                            element.as_ref().clone(),
                        )
                    })
                    .collect();
                eligible.insert(local_idx, parts);
            }
            _ => {}
        }
    }
    if eligible.is_empty() {
        return false;
    }

    // Disqualify any local whose address is used outside a constant-indexed `AccessChain`,
    // or whose `LocalAddr` is never accessed at all via `AccessChain`.
    let mut has_access_chain: IndexSet<u32> = IndexSet::new();
    let mut disqualified: IndexSet<u32> = IndexSet::new();
    for (_, inst) in func.insts.iter() {
        match &inst.data {
            InstData::Nop | InstData::LocalAddr { .. } => {}
            InstData::AccessChain { base_ptr, indices } => {
                if let Some(&local_idx) = addr_to_local.get(base_ptr)
                    && let Some(parts) = eligible.get(&local_idx)
                {
                    if let Some(&first_idx) = indices.first()
                        && let Some(k) = const_non_neg_index(func, first_idx)
                        && (k as usize) < parts.len()
                    {
                        has_access_chain.insert(local_idx);
                    } else {
                        disqualified.insert(local_idx);
                    }
                }
                for &idx_id in indices {
                    if let Some(&local_idx) = addr_to_local.get(&idx_id) {
                        disqualified.insert(local_idx);
                    }
                }
            }
            other => {
                for op in other.operands() {
                    if let Some(&local_idx) = addr_to_local.get(&op) {
                        disqualified.insert(local_idx);
                    }
                }
            }
        }
    }

    // Also disqualify any local whose `LocalAddr` is returned directly.
    let mut ret_ids = IndexSet::new();
    collect_return_ids(&func.body, &mut ret_ids);
    for id in ret_ids {
        if let Some(&local_idx) = addr_to_local.get(&id) {
            disqualified.insert(local_idx);
        }
    }

    let mut split_map: IndexMap<u32, Vec<(u32, IrType)>> = IndexMap::new();
    for (local_idx, parts) in eligible {
        if disqualified.contains(&local_idx) || !has_access_chain.contains(&local_idx) {
            continue;
        }
        let mut sub_locals = Vec::with_capacity(parts.len());
        for (name, ty) in parts {
            let new_idx = func.local_vars.len() as u32;
            func.local_vars.push(IrLocalVar {
                name,
                ty: ty.clone(),
            });
            sub_locals.push((new_idx, ty));
        }
        split_map.insert(local_idx, sub_locals);
    }

    if split_map.is_empty() {
        return false;
    }

    let inst_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
    for id in inst_ids {
        let InstData::AccessChain { base_ptr, indices } = func.insts[id].data.clone() else {
            continue;
        };
        let Some(&local_idx) = addr_to_local.get(&base_ptr) else {
            continue;
        };
        let Some(sub_locals) = split_map.get(&local_idx) else {
            continue;
        };
        let Some(k) = indices.first().and_then(|&i| const_non_neg_index(func, i)) else {
            continue;
        };
        let (sub_local_idx, sub_ty) = &sub_locals[k as usize];
        let ptr_ty = IrType::Pointer {
            pointee: Box::new(sub_ty.clone()),
            align: 4,
            storage: IrStorageClass::Function,
        };
        if indices.len() == 1 {
            func.insts[id].data = InstData::LocalAddr {
                local_index: *sub_local_idx,
            };
            func.insts[id].ir_ty = ptr_ty;
        } else {
            let new_base = func.emit_typed(
                ptr_ty,
                InstData::LocalAddr {
                    local_index: *sub_local_idx,
                },
            );
            func.insts[id].data = InstData::AccessChain {
                base_ptr: new_base,
                indices: indices[1..].to_vec(),
            };
        }
    }

    true
}

fn collect_return_ids(stmts: &[StructuredStmt], out: &mut IndexSet<InstId>) {
    for stmt in stmts {
        match stmt {
            StructuredStmt::Return(Some(id)) => {
                out.insert(*id);
            }
            StructuredStmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_return_ids(then_body, out);
                collect_return_ids(else_body, out);
            }
            StructuredStmt::Switch { cases, default, .. } => {
                for (_, b) in cases {
                    collect_return_ids(b, out);
                }
                collect_return_ids(default, out);
            }
            StructuredStmt::Loop {
                body, continuing, ..
            } => {
                collect_return_ids(body, out);
                collect_return_ids(continuing, out);
            }
            StructuredStmt::Eval(_)
            | StructuredStmt::Break
            | StructuredStmt::Continue
            | StructuredStmt::Return(None) => {}
        }
    }
}

/// Promotes non-address-taken scalar and vector [`IrLocalVar`]s into SSA values across
/// structured `If`, `Switch`, and `Loop` control flow, merging mutations before `Continue`
/// into the loop continuing/header `Phi` and before `Break` into post-loop `Phi`s.
pub struct Mem2RegPass;

impl IrPass for Mem2RegPass {
    fn name(&self) -> &'static str {
        "mem2reg"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut addr_to_local: IndexMap<InstId, u32> = IndexMap::new();
        for (id, inst) in func.insts.iter() {
            if let InstData::LocalAddr { local_index } = inst.data {
                addr_to_local.insert(id, local_index);
            }
        }
        if addr_to_local.is_empty() {
            return;
        }

        let mut promotable: IndexSet<u32> = IndexSet::new();
        for (idx, var) in func.local_vars.iter().enumerate() {
            if matches!(var.ty, IrType::Scalar(_) | IrType::Vector { .. }) {
                promotable.insert(idx as u32);
            }
        }

        // Disqualify any local whose address escapes into `AccessChain`, `FunctionCall`,
        // atomics, `Phi`, `Select`, or as a stored/returned value.
        for (_, inst) in func.insts.iter() {
            match &inst.data {
                InstData::Nop | InstData::LocalAddr { .. } | InstData::LocalLoad { .. } => {}
                InstData::LocalStore { value, .. } => {
                    if let Some(&local_idx) = addr_to_local.get(value) {
                        promotable.shift_remove(&local_idx);
                    }
                }
                other => {
                    for op in other.operands() {
                        if let Some(&local_idx) = addr_to_local.get(&op) {
                            promotable.shift_remove(&local_idx);
                        }
                    }
                }
            }
        }
        let mut ret_ids = IndexSet::new();
        collect_return_ids(&func.body, &mut ret_ids);
        for id in ret_ids {
            if let Some(&local_idx) = addr_to_local.get(&id) {
                promotable.shift_remove(&local_idx);
            }
        }

        // Restrict to locals actually accessed by at least one `LocalLoad` or `LocalStore`.
        let mut accessed: IndexSet<u32> = IndexSet::new();
        for (_, inst) in func.insts.iter() {
            match inst.data {
                InstData::LocalLoad { ptr } | InstData::LocalStore { ptr, .. } => {
                    if let Some(&local_idx) = addr_to_local.get(&ptr)
                        && promotable.contains(&local_idx)
                    {
                        accessed.insert(local_idx);
                    }
                }
                _ => {}
            }
        }
        promotable.retain(|idx| accessed.contains(idx));
        if promotable.is_empty() {
            return;
        }

        let mut env: IndexMap<u32, InstId> = IndexMap::new();
        for &local_idx in &promotable {
            let ty = func.local_vars[local_idx as usize].ty.clone();
            let zero = emit_zero_const(func, &ty);
            env.insert(local_idx, zero);
        }

        let mut state = Mem2RegState {
            addr_to_local,
            promotable,
            remap: IndexMap::new(),
            removed_insts: IndexSet::new(),
        };

        let body = std::mem::take(&mut func.body);
        let mut root_cf = ControlFlowMergeCtx {
            continue_envs: None,
            break_envs: None,
        };
        let (new_body, _) = mem2reg_stmts(func, body, &mut env, &mut state, &mut root_cf);

        let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
        for id in all_ids {
            if state.removed_insts.contains(&id) || state.remap.contains_key(&id) {
                func.insts[id].data = InstData::Nop;
                func.insts[id].non_uniform = false;
            } else if !state.remap.is_empty() {
                let updated = remap_inst_data(&func.insts[id].data, &state.remap);
                func.insts[id].data = updated;
            }
        }
        func.body = if state.remap.is_empty() {
            new_body
        } else {
            remap_stmts_ids(new_body, &state.remap)
        };
    }
}

struct Mem2RegState {
    addr_to_local: IndexMap<InstId, u32>,
    promotable: IndexSet<u32>,
    remap: IndexMap<InstId, InstId>,
    removed_insts: IndexSet<InstId>,
}

#[derive(Default)]
struct LoopMergeState {
    continue_envs: Vec<IndexMap<u32, InstId>>,
    break_envs: Vec<IndexMap<u32, InstId>>,
}

struct ControlFlowMergeCtx<'a> {
    continue_envs: Option<&'a mut Vec<IndexMap<u32, InstId>>>,
    break_envs: Option<&'a mut Vec<IndexMap<u32, InstId>>>,
}

fn emit_zero_const(func: &mut IrFunction, ty: &IrType) -> InstId {
    match ty {
        IrType::Scalar(ScalarIrTy::Bool) => func.emit_typed(ty.clone(), InstData::ConstBool(false)),
        IrType::Scalar(ScalarIrTy::I32) => func.emit_typed(ty.clone(), InstData::ConstI32(0)),
        IrType::Scalar(ScalarIrTy::U32) => func.emit_typed(ty.clone(), InstData::ConstU32(0)),
        IrType::Scalar(ScalarIrTy::F32) => func.emit_typed(ty.clone(), InstData::const_f32(0.0)),
        IrType::Scalar(ScalarIrTy::F16) => func.emit_typed(ty.clone(), InstData::ConstF16(0)),
        IrType::Scalar(ScalarIrTy::I64) => func.emit_typed(ty.clone(), InstData::ConstI64(0)),
        IrType::Scalar(ScalarIrTy::U64) => func.emit_typed(ty.clone(), InstData::ConstU64(0)),
        IrType::Vector { scalar, len } => {
            let elem = emit_zero_const(func, &IrType::Scalar(*scalar));
            let elems = vec![elem; usize::from(*len)];
            func.emit_typed(ty.clone(), build_vec_inst_data(elems))
        }
        _ => func.emit(None, InstData::ConstU32(0)),
    }
}

fn mem2reg_resolve_inst(
    func: &mut IrFunction,
    id: InstId,
    env: &IndexMap<u32, InstId>,
    state: &mut Mem2RegState,
    visited: &mut IndexSet<InstId>,
) -> InstId {
    if let Some(&mapped) = state.remap.get(&id) {
        return remap_id(mapped, &state.remap);
    }
    if !visited.insert(id) {
        return id;
    }
    if let InstData::LocalLoad { ptr } = func.insts[id].data
        && let Some(&local_idx) = state.addr_to_local.get(&ptr)
        && state.promotable.contains(&local_idx)
    {
        let val = env[&local_idx];
        state.remap.insert(id, val);
        state.removed_insts.insert(id);
        return val;
    }
    if matches!(func.insts[id].data, InstData::Phi { .. }) {
        return id;
    }
    let operands = func.insts[id].data.operands();
    for op in operands {
        let _ = mem2reg_resolve_inst(func, op, env, state, visited);
    }
    if !state.remap.is_empty() {
        let updated = remap_inst_data(&func.insts[id].data, &state.remap);
        func.insts[id].data = updated;
    }
    id
}

fn collect_written_promotable_locals(
    func: &IrFunction,
    stmts: &[StructuredStmt],
    state: &Mem2RegState,
    out: &mut IndexSet<u32>,
) {
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                if let InstData::LocalStore { ptr, .. } = func.insts[*id].data
                    && let Some(&local_idx) = state.addr_to_local.get(&ptr)
                    && state.promotable.contains(&local_idx)
                {
                    out.insert(local_idx);
                }
            }
            StructuredStmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_written_promotable_locals(func, then_body, state, out);
                collect_written_promotable_locals(func, else_body, state, out);
            }
            StructuredStmt::Switch { cases, default, .. } => {
                for (_, b) in cases {
                    collect_written_promotable_locals(func, b, state, out);
                }
                collect_written_promotable_locals(func, default, state, out);
            }
            StructuredStmt::Loop {
                body, continuing, ..
            } => {
                collect_written_promotable_locals(func, body, state, out);
                collect_written_promotable_locals(func, continuing, state, out);
            }
            StructuredStmt::Break | StructuredStmt::Continue | StructuredStmt::Return(_) => {}
        }
    }
}

fn merge_envs_into_phis(
    func: &mut IrFunction,
    locals: &IndexSet<u32>,
    branch_envs: &[(IndexMap<u32, InstId>, u32)],
    out_env: &mut IndexMap<u32, InstId>,
    emit_stmts: &mut Vec<StructuredStmt>,
) {
    for &local_idx in locals {
        let first_val = branch_envs[0].0[&local_idx];
        let all_same = branch_envs
            .iter()
            .all(|(b_env, _)| b_env[&local_idx] == first_val);
        if all_same {
            out_env.insert(local_idx, first_val);
        } else {
            let incoming: Vec<(InstId, u32)> = branch_envs
                .iter()
                .map(|(b_env, tag)| (b_env[&local_idx], *tag))
                .collect();
            let ty = func.local_vars[local_idx as usize].ty.clone();
            let phi = func.emit_typed(ty, InstData::Phi { incoming });
            emit_stmts.push(StructuredStmt::Eval(phi));
            out_env.insert(local_idx, phi);
        }
    }
}

fn mem2reg_stmts(
    func: &mut IrFunction,
    stmts: Vec<StructuredStmt>,
    env: &mut IndexMap<u32, InstId>,
    state: &mut Mem2RegState,
    cf_ctx: &mut ControlFlowMergeCtx<'_>,
) -> (Vec<StructuredStmt>, bool) {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                if let InstData::LocalStore { ptr, value } = func.insts[id].data
                    && let Some(&local_idx) = state.addr_to_local.get(&ptr)
                    && state.promotable.contains(&local_idx)
                {
                    let mut visited = IndexSet::new();
                    let resolved_val = mem2reg_resolve_inst(func, value, env, state, &mut visited);
                    env.insert(local_idx, resolved_val);
                    state.removed_insts.insert(id);
                    continue;
                }
                if let InstData::LocalLoad { ptr } = func.insts[id].data
                    && let Some(&local_idx) = state.addr_to_local.get(&ptr)
                    && state.promotable.contains(&local_idx)
                {
                    let val = env[&local_idx];
                    state.remap.insert(id, val);
                    state.removed_insts.insert(id);
                    continue;
                }
                let mut visited = IndexSet::new();
                let resolved = mem2reg_resolve_inst(func, id, env, state, &mut visited);
                out.push(StructuredStmt::Eval(resolved));
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let mut visited = IndexSet::new();
                let cond = mem2reg_resolve_inst(func, cond, env, state, &mut visited);
                let pre_env = env.clone();

                let mut then_env = pre_env.clone();
                let (new_then, then_term) =
                    mem2reg_stmts(func, then_body, &mut then_env, state, cf_ctx);

                let mut else_env = pre_env;
                let (new_else, else_term) =
                    mem2reg_stmts(func, else_body, &mut else_env, state, cf_ctx);

                out.push(StructuredStmt::If {
                    cond,
                    then_body: new_then,
                    else_body: new_else,
                });

                match (then_term, else_term) {
                    (true, true) => return (out, true),
                    (true, false) => *env = else_env,
                    (false, true) => *env = then_env,
                    (false, false) => {
                        let locals = state.promotable.clone();
                        merge_envs_into_phis(
                            func,
                            &locals,
                            &[(then_env, 0), (else_env, 1)],
                            env,
                            &mut out,
                        );
                    }
                }
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                let mut visited = IndexSet::new();
                let selector = mem2reg_resolve_inst(func, selector, env, state, &mut visited);
                let pre_env = env.clone();
                let mut new_cases = Vec::with_capacity(cases.len());
                let mut fallthrough_envs: Vec<(IndexMap<u32, InstId>, u32)> = Vec::new();

                for (case_idx, (k, case_body)) in cases.into_iter().enumerate() {
                    let mut case_env = pre_env.clone();
                    let mut case_breaks: Vec<IndexMap<u32, InstId>> = Vec::new();
                    let mut switch_cf = ControlFlowMergeCtx {
                        continue_envs: cf_ctx.continue_envs.as_deref_mut(),
                        break_envs: Some(&mut case_breaks),
                    };
                    let (new_b, term) =
                        mem2reg_stmts(func, case_body, &mut case_env, state, &mut switch_cf);
                    new_cases.push((k, new_b));
                    if !term {
                        fallthrough_envs.push((case_env, case_idx as u32));
                    }
                    for b_env in case_breaks {
                        fallthrough_envs.push((b_env, case_idx as u32));
                    }
                }

                let default_tag = new_cases.len() as u32;
                let mut default_env = pre_env;
                let mut default_breaks: Vec<IndexMap<u32, InstId>> = Vec::new();
                let mut default_cf = ControlFlowMergeCtx {
                    continue_envs: cf_ctx.continue_envs.as_deref_mut(),
                    break_envs: Some(&mut default_breaks),
                };
                let (new_default, def_term) =
                    mem2reg_stmts(func, default, &mut default_env, state, &mut default_cf);
                if !def_term {
                    fallthrough_envs.push((default_env, default_tag));
                }
                for b_env in default_breaks {
                    fallthrough_envs.push((b_env, default_tag));
                }

                out.push(StructuredStmt::Switch {
                    selector,
                    cases: new_cases,
                    default: new_default,
                });

                if fallthrough_envs.is_empty() {
                    return (out, true);
                }
                if fallthrough_envs.len() == 1 {
                    *env = fallthrough_envs.remove(0).0;
                } else {
                    let locals = state.promotable.clone();
                    merge_envs_into_phis(func, &locals, &fallthrough_envs, env, &mut out);
                }
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                let induction_init = induction_init.map(|ind| {
                    let mut visited = IndexSet::new();
                    mem2reg_resolve_inst(func, ind, env, state, &mut visited)
                });

                let mut written = IndexSet::new();
                collect_written_promotable_locals(func, &body, state, &mut written);
                collect_written_promotable_locals(func, &continuing, state, &mut written);

                let pre_env = env.clone();
                let mut header_phis: IndexMap<u32, InstId> = IndexMap::new();
                let mut header_evals: Vec<StructuredStmt> = Vec::with_capacity(written.len());
                for &local_idx in &written {
                    let ty = func.local_vars[local_idx as usize].ty.clone();
                    let phi = func.emit_typed(
                        ty,
                        InstData::Phi {
                            incoming: vec![(pre_env[&local_idx], 0)],
                        },
                    );
                    header_phis.insert(local_idx, phi);
                    env.insert(local_idx, phi);
                    header_evals.push(StructuredStmt::Eval(phi));
                }

                let mut inner_loop = LoopMergeState::default();
                let mut body_cf = ControlFlowMergeCtx {
                    continue_envs: Some(&mut inner_loop.continue_envs),
                    break_envs: Some(&mut inner_loop.break_envs),
                };
                let (mut new_body, body_term) = mem2reg_stmts(func, body, env, state, &mut body_cf);

                // Collect all edges reaching the loop `continuing` block: normal body fallthrough
                // plus every `Continue` statement targeting this loop.
                let mut cont_entry_envs: Vec<(IndexMap<u32, InstId>, u32)> = Vec::new();
                if !body_term {
                    cont_entry_envs.push((env.clone(), 0));
                }
                for cont_env in inner_loop.continue_envs {
                    let tag = cont_entry_envs.len() as u32;
                    cont_entry_envs.push((cont_env, tag));
                }

                let mut cont_phi_stmts = Vec::new();
                let (mut cont_env, has_continuing_edge) = if cont_entry_envs.is_empty() {
                    (pre_env.clone(), false)
                } else if cont_entry_envs.len() == 1 {
                    (cont_entry_envs.remove(0).0, true)
                } else {
                    let mut merged = pre_env.clone();
                    merge_envs_into_phis(
                        func,
                        &written,
                        &cont_entry_envs,
                        &mut merged,
                        &mut cont_phi_stmts,
                    );
                    (merged, true)
                };

                let mut cont_cf = ControlFlowMergeCtx {
                    continue_envs: None,
                    break_envs: Some(&mut inner_loop.break_envs),
                };
                let (mut new_continuing, cont_term) =
                    mem2reg_stmts(func, continuing, &mut cont_env, state, &mut cont_cf);

                if !cont_phi_stmts.is_empty() {
                    cont_phi_stmts.extend(new_continuing);
                    new_continuing = cont_phi_stmts;
                }

                // Patch loop header Phis with their back-edge values from `continuing`.
                for (&local_idx, &phi_id) in &header_phis {
                    if has_continuing_edge && !cont_term {
                        func.insts[phi_id].data = InstData::Phi {
                            incoming: vec![(pre_env[&local_idx], 0), (cont_env[&local_idx], 1)],
                        };
                    } else {
                        func.insts[phi_id].data = InstData::Phi {
                            incoming: vec![(pre_env[&local_idx], 0)],
                        };
                    }
                }

                if !header_evals.is_empty() {
                    header_evals.extend(new_body);
                    new_body = header_evals;
                }

                out.push(StructuredStmt::Loop {
                    body: new_body,
                    continuing: new_continuing,
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });

                // Merge `Break` edges into post-loop SSA values.
                if inner_loop.break_envs.is_empty() {
                    // Preserve header Phi bindings for any code following the loop.
                    for (&local_idx, &phi_id) in &header_phis {
                        env.insert(local_idx, phi_id);
                    }
                } else if inner_loop.break_envs.len() == 1 {
                    *env = inner_loop.break_envs.remove(0);
                } else {
                    let tagged_breaks: Vec<(IndexMap<u32, InstId>, u32)> = inner_loop
                        .break_envs
                        .into_iter()
                        .enumerate()
                        .map(|(i, b_env)| (b_env, i as u32))
                        .collect();
                    merge_envs_into_phis(func, &written, &tagged_breaks, env, &mut out);
                }
            }
            StructuredStmt::Continue => {
                if let Some(ce) = cf_ctx.continue_envs.as_deref_mut() {
                    ce.push(env.clone());
                }
                out.push(StructuredStmt::Continue);
                return (out, true);
            }
            StructuredStmt::Break => {
                if let Some(be) = cf_ctx.break_envs.as_deref_mut() {
                    be.push(env.clone());
                }
                out.push(StructuredStmt::Break);
                return (out, true);
            }
            StructuredStmt::Return(opt_val) => {
                let mapped = opt_val.map(|v| {
                    let mut visited = IndexSet::new();
                    mem2reg_resolve_inst(func, v, env, state, &mut visited)
                });
                out.push(StructuredStmt::Return(mapped));
                return (out, true);
            }
        }
    }
    (out, false)
}

/// Propagates trivial single-input/identical-input `Phi`s and identity
/// `VectorShuffle`/`CompositeConstruct` copies to a fixed point.
pub struct CopyPropPass;

impl IrPass for CopyPropPass {
    fn name(&self) -> &'static str {
        "copy-prop"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut changed = true;
        while changed {
            changed = false;
            let mut remap: IndexMap<InstId, InstId> = IndexMap::new();
            for (id, inst) in func.insts.iter() {
                let candidate = match &inst.data {
                    InstData::Phi { incoming } => {
                        let mut unique: Option<InstId> = None;
                        let mut trivial = true;
                        for &(v, _) in incoming {
                            if v == id {
                                continue;
                            }
                            match unique {
                                None => unique = Some(v),
                                Some(prev) if prev == v => {}
                                Some(_) => {
                                    trivial = false;
                                    break;
                                }
                            }
                        }
                        if trivial { unique } else { None }
                    }
                    InstData::VectorShuffle {
                        lhs,
                        rhs,
                        components,
                    } => {
                        let lhs_len = vector_len_of(func, *lhs);
                        let rhs_len = vector_len_of(func, *rhs);
                        if let Some(l_len) = lhs_len
                            && components.len() == usize::from(l_len)
                            && components.iter().enumerate().all(|(i, &c)| c == i as u32)
                        {
                            Some(*lhs)
                        } else if let (Some(l_len), Some(r_len)) = (lhs_len, rhs_len)
                            && components.len() == usize::from(r_len)
                            && components
                                .iter()
                                .enumerate()
                                .all(|(i, &c)| c == u32::from(l_len) + i as u32)
                        {
                            Some(*rhs)
                        } else {
                            None
                        }
                    }
                    InstData::CompositeConstruct { elems } => {
                        identity_reconstruct_source(func, id, elems)
                    }
                    InstData::MakeVec2(a, b) => identity_reconstruct_source(func, id, &[*a, *b]),
                    InstData::MakeVec3(a, b, c) => {
                        identity_reconstruct_source(func, id, &[*a, *b, *c])
                    }
                    InstData::MakeVec4(a, b, c, d) => {
                        identity_reconstruct_source(func, id, &[*a, *b, *c, *d])
                    }
                    _ => None,
                };
                if let Some(target) = candidate
                    && target != id
                {
                    remap.insert(id, target);
                }
            }

            if !remap.is_empty() {
                let all_ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
                for id in all_ids {
                    if remap.contains_key(&id) {
                        func.insts[id].data = InstData::Nop;
                        func.insts[id].non_uniform = false;
                    } else {
                        let updated = remap_inst_data(&func.insts[id].data, &remap);
                        func.insts[id].data = updated;
                    }
                }
                let body = std::mem::take(&mut func.body);
                func.body = remap_stmts_ids(body, &remap);
                changed = true;
            }
        }
    }
}

fn vector_len_of(func: &IrFunction, id: InstId) -> Option<u8> {
    match &func.insts[id].ir_ty {
        IrType::Vector { len, .. } => return Some(*len),
        IrType::Void => {}
        _ => return None,
    }
    match &func.insts[id].data {
        InstData::MakeVec2(_, _) => Some(2),
        InstData::MakeVec3(_, _, _) => Some(3),
        InstData::MakeVec4(_, _, _, _) => Some(4),
        InstData::CompositeConstruct { elems } if (2..=4).contains(&elems.len()) => {
            Some(elems.len() as u8)
        }
        _ => None,
    }
}

fn identity_reconstruct_source(
    func: &IrFunction,
    construct_id: InstId,
    elems: &[InstId],
) -> Option<InstId> {
    if elems.is_empty() {
        return None;
    }
    let mut source: Option<InstId> = None;
    for (idx, &elem_id) in elems.iter().enumerate() {
        let (src, comp) = match &func.insts[elem_id].data {
            InstData::ExtractComponent { vec, index } => (*vec, *index),
            InstData::CompositeExtract { composite, indices } if indices.len() == 1 => {
                (*composite, indices[0])
            }
            _ => return None,
        };
        if comp != idx as u32 {
            return None;
        }
        match source {
            None => source = Some(src),
            Some(prev) if prev == src => {}
            Some(_) => return None,
        }
    }
    let src = source?;
    let src_ty = &func.insts[src].ir_ty;
    let dst_ty = &func.insts[construct_id].ir_ty;
    if *src_ty != IrType::Void && *dst_ty != IrType::Void {
        return (src_ty == dst_ty).then_some(src);
    }
    let is_non_vec_aggregate = |ty: &IrType| {
        matches!(
            ty,
            IrType::Struct(_) | IrType::Array { .. } | IrType::Matrix { .. }
        )
    };
    if is_non_vec_aggregate(src_ty) || is_non_vec_aggregate(dst_ty) {
        return None;
    }
    let matches_len = vector_len_of(func, src).is_some_and(|l| usize::from(l) == elems.len());
    matches_len.then_some(src)
}

/// Hoists loop-invariant pure instructions whose operands are all defined outside the loop
/// into statements immediately preceding `StructuredStmt::Loop`.
pub struct LicmPass;

impl IrPass for LicmPass {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run(&self, func: &mut IrFunction) {
        let body = std::mem::take(&mut func.body);
        func.body = licm_stmts(func, body);
    }
}

fn is_licm_hoistable(func: &IrFunction, id: InstId) -> bool {
    if func.insts[id].has_side_effects() {
        return false;
    }
    !matches!(
        func.insts[id].data,
        InstData::Nop
            | InstData::Phi { .. }
            | InstData::LocalLoad { .. }
            | InstData::WorkgroupLoad { .. }
            | InstData::BdaLoad { .. }
            | InstData::DescriptorLoad { .. }
            | InstData::ResourceLoad { .. }
            | InstData::ImageSample { .. }
            | InstData::ImageSampleImplicitLod { .. }
            | InstData::ImageSampleGrad { .. }
            | InstData::ImageSampleDrefExplicitLod { .. }
            | InstData::ImageGather { .. }
            | InstData::DPdx(_)
            | InstData::DPdy(_)
            | InstData::Fwidth(_)
            | InstData::RayQueryCommittedType(_)
            | InstData::RayQueryCandidateType(_)
            | InstData::RayQueryCommittedT(_)
            | InstData::RayQueryCandidateT(_)
            | InstData::RayQueryCommittedInstanceId(_)
            | InstData::RayQueryCommittedInstanceCustomIndex(_)
            | InstData::RayQueryCommittedPrimitiveIndex(_)
            | InstData::RayQueryCommittedGeometryIndex(_)
            | InstData::RayQueryCommittedObjectToWorld(_)
            | InstData::RayQueryCommittedWorldToObject(_)
            | InstData::RayQueryCommittedFrontFace(_)
            | InstData::RayQueryCommittedBarycentrics(_)
            | InstData::LoadStageVar { .. }
            | InstData::HitObjectIsHit(_)
            | InstData::HitObjectGetRayTMax(_)
            | InstData::HitObjectGetSbtRecordIndex(_)
            | InstData::RayQueryTriangleVertexPositions { .. }
    )
}

fn depends_on_any(
    func: &IrFunction,
    id: InstId,
    loop_defined: &IndexSet<InstId>,
    visited: &mut IndexSet<InstId>,
) -> bool {
    if loop_defined.contains(&id) {
        return true;
    }
    if !visited.insert(id) {
        return false;
    }
    if !is_licm_hoistable(func, id) {
        return true;
    }
    func.insts[id]
        .data
        .operands()
        .into_iter()
        .any(|op| depends_on_any(func, op, loop_defined, visited))
}

fn licm_stmts(func: &IrFunction, stmts: Vec<StructuredStmt>) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => out.push(StructuredStmt::If {
                cond,
                then_body: licm_stmts(func, then_body),
                else_body: licm_stmts(func, else_body),
            }),
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => out.push(StructuredStmt::Switch {
                selector,
                cases: cases
                    .into_iter()
                    .map(|(k, b)| (k, licm_stmts(func, b)))
                    .collect(),
                default: licm_stmts(func, default),
            }),
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                // Optimize inner loops first so their hoisted statements can be hoisted further.
                let mut body = licm_stmts(func, body);
                let continuing = licm_stmts(func, continuing);

                let mut loop_defined = IndexSet::new();
                collect_eval_ids(&body, &mut loop_defined);
                collect_eval_ids(&continuing, &mut loop_defined);
                if let Some(ind) = induction_init {
                    loop_defined.insert(ind);
                }

                let mut hoisted: Vec<StructuredStmt> = Vec::new();
                let mut changed = true;
                while changed {
                    changed = false;
                    let mut remaining = Vec::with_capacity(body.len());
                    for s in body {
                        if let StructuredStmt::Eval(id) = s
                            && is_licm_hoistable(func, id)
                        {
                            let invariant = func.insts[id].data.operands().into_iter().all(|op| {
                                let mut visited = IndexSet::new();
                                !depends_on_any(func, op, &loop_defined, &mut visited)
                            });
                            if invariant {
                                loop_defined.shift_remove(&id);
                                hoisted.push(StructuredStmt::Eval(id));
                                changed = true;
                                continue;
                            }
                        }
                        remaining.push(s);
                    }
                    body = remaining;
                }

                out.extend(hoisted);
                out.push(StructuredStmt::Loop {
                    body,
                    continuing,
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });
            }
            other => out.push(other),
        }
    }
    out
}

/// Eliminates dead `LocalStore`s overwritten by a subsequent `LocalStore` to the same
/// `LocalAddr` in the same block with no intervening load, call, atomic, or barrier.
pub struct DeadStoreElimPass;

impl IrPass for DeadStoreElimPass {
    fn name(&self) -> &'static str {
        "dse"
    }

    fn run(&self, func: &mut IrFunction) {
        let body = std::mem::take(&mut func.body);
        func.body = dse_stmts(func, body);
    }
}

fn dse_observe_inst_tree(
    func: &IrFunction,
    id: InstId,
    pending: &mut IndexMap<u32, (usize, InstId)>,
    visited: &mut IndexSet<InstId>,
) {
    if pending.is_empty() || !visited.insert(id) {
        return;
    }
    match &func.insts[id].data {
        InstData::LocalLoad { ptr } => {
            if let InstData::LocalAddr { local_index } = func.insts[*ptr].data {
                pending.shift_remove(&local_index);
            } else {
                pending.clear();
            }
        }
        InstData::FunctionCall { .. }
        | InstData::WorkgroupBarrier
        | InstData::MemoryBarrier { .. }
        | InstData::AtomicAdd { .. }
        | InstData::AtomicMin { .. }
        | InstData::AtomicMax { .. }
        | InstData::AtomicCas { .. } => {
            pending.clear();
        }
        other => {
            for op in other.operands() {
                dse_observe_inst_tree(func, op, pending, visited);
            }
        }
    }
}

fn inst_tree_has_side_effects(
    func: &IrFunction,
    id: InstId,
    visited: &mut IndexSet<InstId>,
) -> bool {
    if !visited.insert(id) {
        return false;
    }
    if func.insts[id].has_side_effects() {
        return true;
    }
    if matches!(func.insts[id].data, InstData::Phi { .. }) {
        return false;
    }
    func.insts[id]
        .data
        .operands()
        .into_iter()
        .any(|op| inst_tree_has_side_effects(func, op, visited))
}

fn dse_stmts(func: &mut IrFunction, stmts: Vec<StructuredStmt>) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut pending: IndexMap<u32, (usize, InstId)> = IndexMap::new();
    let mut dead_indices: IndexSet<usize> = IndexSet::new();

    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                if let InstData::LocalStore { ptr, value } = func.insts[id].data {
                    let mut visited = IndexSet::new();
                    dse_observe_inst_tree(func, value, &mut pending, &mut visited);
                    if let InstData::LocalAddr { local_index } = func.insts[ptr].data {
                        if let Some((prev_out_idx, prev_store_id)) =
                            pending.insert(local_index, (out.len(), id))
                        {
                            let prev_value = match func.insts[prev_store_id].data {
                                InstData::LocalStore { value: v, .. } => Some(v),
                                _ => None,
                            };
                            let has_side_effects = prev_value.is_some_and(|v| {
                                let mut se_visited = IndexSet::new();
                                inst_tree_has_side_effects(func, v, &mut se_visited)
                            });
                            if let (true, Some(pv)) = (has_side_effects, prev_value) {
                                out[prev_out_idx] = StructuredStmt::Eval(pv);
                            } else {
                                dead_indices.insert(prev_out_idx);
                            }
                            func.insts[prev_store_id].data = InstData::Nop;
                            func.insts[prev_store_id].non_uniform = false;
                        }
                    } else {
                        pending.clear();
                    }
                    out.push(StructuredStmt::Eval(id));
                } else {
                    let mut visited = IndexSet::new();
                    dse_observe_inst_tree(func, id, &mut pending, &mut visited);
                    out.push(StructuredStmt::Eval(id));
                }
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                pending.clear();
                out.push(StructuredStmt::If {
                    cond,
                    then_body: dse_stmts(func, then_body),
                    else_body: dse_stmts(func, else_body),
                });
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                pending.clear();
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: cases
                        .into_iter()
                        .map(|(k, b)| (k, dse_stmts(func, b)))
                        .collect(),
                    default: dse_stmts(func, default),
                });
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                pending.clear();
                out.push(StructuredStmt::Loop {
                    body: dse_stmts(func, body),
                    continuing: dse_stmts(func, continuing),
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });
            }
            other @ (StructuredStmt::Return(_)
            | StructuredStmt::Break
            | StructuredStmt::Continue) => {
                pending.clear();
                out.push(other);
            }
        }
    }

    if dead_indices.is_empty() {
        out
    } else {
        out.into_iter()
            .enumerate()
            .filter_map(|(i, s)| (!dead_indices.contains(&i)).then_some(s))
            .collect()
    }
}

/// Module-level selective inliner that inlines small non-recursive helper [`IrFunction`]s
/// (up to [`MAX_INLINE_BODY_INSTS`] live instructions) at `FunctionCall` sites while keeping
/// larger helpers as real `FunctionCall`s.
pub struct SelectiveInlinePass;

impl SelectiveInlinePass {
    pub const fn name(&self) -> &'static str {
        "selective-inline"
    }

    pub fn run_module(&self, module: &mut IrModule) {
        const MAX_ROUNDS: usize = 4;
        for _ in 0..MAX_ROUNDS {
            let call_graph = build_call_graph(module);
            let eligible: IndexMap<SmolStr, IrFunction> = module
                .functions
                .iter()
                .filter(|f| is_inline_eligible(f, &call_graph))
                .map(|f| (f.name.clone(), f.clone()))
                .collect();

            if eligible.is_empty() {
                break;
            }

            let mut any_inlined = false;
            for func in &mut module.functions {
                if inline_calls_in_function(func, &eligible) {
                    any_inlined = true;
                }
            }
            if !any_inlined {
                break;
            }
        }
    }
}

fn build_call_graph(module: &IrModule) -> IndexMap<SmolStr, IndexSet<SmolStr>> {
    let mut graph: IndexMap<SmolStr, IndexSet<SmolStr>> = IndexMap::new();
    for func in &module.functions {
        let mut callees = IndexSet::new();
        for (_, inst) in func.insts.iter() {
            if let InstData::FunctionCall { callee, .. } = &inst.data {
                callees.insert(callee.clone());
            }
        }
        graph.insert(func.name.clone(), callees);
    }
    graph
}

fn is_recursive(name: &str, graph: &IndexMap<SmolStr, IndexSet<SmolStr>>) -> bool {
    let mut visited: IndexSet<SmolStr> = IndexSet::new();
    let mut stack: Vec<SmolStr> = Vec::new();
    if let Some(direct) = graph.get(name) {
        for callee in direct {
            stack.push(callee.clone());
        }
    }
    while let Some(cur) = stack.pop() {
        if cur.as_str() == name {
            return true;
        }
        if !visited.insert(cur.clone()) {
            continue;
        }
        if let Some(nexts) = graph.get(&cur) {
            for n in nexts {
                stack.push(n.clone());
            }
        }
    }
    false
}

fn has_nested_return(stmts: &[StructuredStmt], is_top_level: bool) -> bool {
    for (idx, stmt) in stmts.iter().enumerate() {
        match stmt {
            StructuredStmt::Return(_) => {
                if !is_top_level || idx + 1 != stmts.len() {
                    return true;
                }
            }
            StructuredStmt::If {
                then_body,
                else_body,
                ..
            } => {
                if has_nested_return(then_body, false) || has_nested_return(else_body, false) {
                    return true;
                }
            }
            StructuredStmt::Switch { cases, default, .. } => {
                if cases.iter().any(|(_, b)| has_nested_return(b, false))
                    || has_nested_return(default, false)
                {
                    return true;
                }
            }
            StructuredStmt::Loop {
                body, continuing, ..
            } => {
                if has_nested_return(body, false) || has_nested_return(continuing, false) {
                    return true;
                }
            }
            StructuredStmt::Eval(_) | StructuredStmt::Break | StructuredStmt::Continue => {}
        }
    }
    false
}

fn is_inline_eligible(
    func: &IrFunction,
    call_graph: &IndexMap<SmolStr, IndexSet<SmolStr>>,
) -> bool {
    if func.stage.is_some() {
        return false;
    }
    let live_inst_count = func
        .insts
        .iter()
        .filter(|(_, inst)| !matches!(inst.data, InstData::Nop))
        .count();
    if live_inst_count > MAX_INLINE_BODY_INSTS {
        return false;
    }
    if is_recursive(&func.name, call_graph) {
        return false;
    }
    if has_nested_return(&func.body, true) {
        return false;
    }
    true
}

fn inline_calls_in_function(
    caller: &mut IrFunction,
    eligible: &IndexMap<SmolStr, IrFunction>,
) -> bool {
    let mut call_remap: IndexMap<InstId, InstId> = IndexMap::new();
    let mut inlined_any = false;
    let body = std::mem::take(&mut caller.body);
    let new_body = inline_stmts(caller, body, eligible, &mut call_remap, &mut inlined_any);

    if inlined_any && !call_remap.is_empty() {
        let all_ids: Vec<InstId> = caller.insts.iter().map(|(id, _)| id).collect();
        for id in all_ids {
            if call_remap.contains_key(&id) {
                caller.insts[id].data = InstData::Nop;
                caller.insts[id].non_uniform = false;
            } else {
                let updated = remap_inst_data(&caller.insts[id].data, &call_remap);
                caller.insts[id].data = updated;
            }
        }
        caller.body = remap_stmts_ids(new_body, &call_remap);
    } else {
        caller.body = new_body;
    }
    inlined_any
}

fn max_ray_query_id_plus_one(func: &IrFunction) -> u32 {
    let mut next_id = 0u32;
    for (_, inst) in func.insts.iter() {
        let rq = match inst.data {
            InstData::RayQueryInitialize { ray_query_id, .. }
            | InstData::RayQueryGenerateIntersection { ray_query_id, .. } => Some(ray_query_id),
            InstData::RayQueryProceed(id)
            | InstData::RayQueryCommittedType(id)
            | InstData::RayQueryCandidateType(id)
            | InstData::RayQueryConfirmIntersection(id)
            | InstData::RayQueryTerminate(id)
            | InstData::RayQueryCommittedT(id)
            | InstData::RayQueryCandidateT(id)
            | InstData::RayQueryCommittedInstanceId(id)
            | InstData::RayQueryCommittedInstanceCustomIndex(id)
            | InstData::RayQueryCommittedPrimitiveIndex(id)
            | InstData::RayQueryCommittedGeometryIndex(id)
            | InstData::RayQueryCommittedObjectToWorld(id)
            | InstData::RayQueryCommittedWorldToObject(id)
            | InstData::RayQueryCommittedFrontFace(id)
            | InstData::RayQueryCommittedBarycentrics(id) => Some(id),
            _ => None,
        };
        if let Some(id) = rq {
            next_id = next_id.max(id.saturating_add(1));
        }
    }
    next_id
}

fn can_merge_callee_resources(caller: &IrFunction, callee: &IrFunction) -> bool {
    for cd in &callee.descriptors {
        if let Some(existing) = caller
            .descriptors
            .iter()
            .find(|d| d.set == cd.set && d.binding == cd.binding)
            && existing.kind != cd.kind
        {
            return false;
        }
    }
    for cw in &callee.workgroup_vars {
        if let Some(existing) = caller.workgroup_vars.iter().find(|w| w.name == cw.name)
            && existing.effective_type() != cw.effective_type()
        {
            return false;
        }
    }
    for cio in &callee.stage_io {
        if let Some(existing) = caller
            .stage_io
            .iter()
            .find(|io| io.location == cio.location && io.is_output == cio.is_output)
            && existing.ty != cio.ty
        {
            return false;
        }
    }
    if let (Some(caller_pc), Some(callee_pc)) = (&caller.push_constants, &callee.push_constants) {
        let mut max_end = caller_pc.total_size;
        for cf in &callee_pc.fields {
            let exact_match = caller_pc.fields.iter().any(|f| {
                f.byte_offset == cf.byte_offset
                    && f.byte_size == cf.byte_size
                    && f.scalar == cf.scalar
            });
            if exact_match {
                continue;
            }
            let cf_end = cf.byte_offset.saturating_add(cf.byte_size);
            let overlaps = caller_pc.fields.iter().any(|f| {
                let f_end = f.byte_offset.saturating_add(f.byte_size);
                cf.byte_offset < f_end && f.byte_offset < cf_end
            });
            if overlaps || cf_end > 128 {
                return false;
            }
            max_end = max_end.max(cf_end);
        }
        if max_end > 128 {
            return false;
        }
    }
    true
}

struct InlineResourceRemap {
    local_offset: u32,
    desc_remap: Vec<u32>,
    wg_remap: Vec<u32>,
    stage_io_remap: Vec<u32>,
    pc_field_remap: Vec<u32>,
    rq_offset: u32,
}

fn merge_callee_resources(caller: &mut IrFunction, callee: &IrFunction) -> InlineResourceRemap {
    let local_offset = caller.local_vars.len() as u32;
    for lv in &callee.local_vars {
        caller.local_vars.push(lv.clone());
    }

    let mut desc_remap = Vec::with_capacity(callee.descriptors.len());
    for cd in &callee.descriptors {
        if let Some(idx) = caller
            .descriptors
            .iter()
            .position(|d| d.set == cd.set && d.binding == cd.binding)
        {
            desc_remap.push(idx as u32);
        } else {
            let idx = caller.descriptors.len() as u32;
            caller.descriptors.push(cd.clone());
            desc_remap.push(idx);
        }
    }

    let mut wg_remap = Vec::with_capacity(callee.workgroup_vars.len());
    for cw in &callee.workgroup_vars {
        if let Some(idx) = caller
            .workgroup_vars
            .iter()
            .position(|w| w.name == cw.name && w.effective_type() == cw.effective_type())
        {
            wg_remap.push(idx as u32);
        } else {
            let idx = caller.workgroup_vars.len() as u32;
            caller.workgroup_vars.push(cw.clone());
            wg_remap.push(idx);
        }
    }

    let mut stage_io_remap = Vec::with_capacity(callee.stage_io.len());
    for cio in &callee.stage_io {
        if let Some(idx) = caller
            .stage_io
            .iter()
            .position(|io| io.location == cio.location && io.is_output == cio.is_output)
        {
            stage_io_remap.push(idx as u32);
        } else {
            let idx = caller.stage_io.len() as u32;
            caller.stage_io.push(cio.clone());
            stage_io_remap.push(idx);
        }
    }

    let mut pc_field_remap = Vec::new();
    if let Some(callee_pc) = &callee.push_constants {
        if let Some(caller_pc) = &mut caller.push_constants {
            pc_field_remap.reserve(callee_pc.fields.len());
            for cf in &callee_pc.fields {
                if let Some(idx) = caller_pc.fields.iter().position(|f| {
                    f.byte_offset == cf.byte_offset
                        && f.byte_size == cf.byte_size
                        && f.scalar == cf.scalar
                }) {
                    pc_field_remap.push(idx as u32);
                } else {
                    let idx = caller_pc.fields.len() as u32;
                    caller_pc.fields.push(cf.clone());
                    caller_pc.total_size = caller_pc
                        .total_size
                        .max(cf.byte_offset.saturating_add(cf.byte_size));
                    pc_field_remap.push(idx);
                }
            }
        } else {
            pc_field_remap = (0..callee_pc.fields.len() as u32).collect();
            caller.push_constants = Some(callee_pc.clone());
        }
    }

    let rq_offset = max_ray_query_id_plus_one(caller);

    InlineResourceRemap {
        local_offset,
        desc_remap,
        wg_remap,
        stage_io_remap,
        pc_field_remap,
        rq_offset,
    }
}

fn remap_inlined_resource_indices(data: &mut InstData, res: &InlineResourceRemap) {
    match data {
        InstData::LocalAddr { local_index } => {
            *local_index += res.local_offset;
        }
        InstData::DescriptorAccessChain {
            descriptor_index, ..
        }
        | InstData::ImageSample {
            descriptor_index, ..
        }
        | InstData::ImageSampleImplicitLod {
            descriptor_index, ..
        }
        | InstData::ImageSampleGrad {
            descriptor_index, ..
        }
        | InstData::ImageSampleDrefExplicitLod {
            descriptor_index, ..
        }
        | InstData::ImageGather {
            descriptor_index, ..
        }
        | InstData::ImageQuerySize {
            descriptor_index, ..
        }
        | InstData::ImageQueryLevels { descriptor_index }
        | InstData::ImageTexelPointer {
            descriptor_index, ..
        }
        | InstData::ImageStore {
            descriptor_index, ..
        }
        | InstData::ResourceAccessChain {
            descriptor_index, ..
        }
        | InstData::DescriptorLoad {
            descriptor_index, ..
        } => {
            if let Some(&mapped) = res.desc_remap.get(*descriptor_index as usize) {
                *descriptor_index = mapped;
            }
        }
        InstData::WorkgroupAddr { var_index, .. }
        | InstData::WorkgroupLoad { var_index, .. }
        | InstData::WorkgroupStore { var_index, .. } => {
            if let Some(&mapped) = res.wg_remap.get(*var_index as usize) {
                *var_index = mapped;
            }
        }
        InstData::LoadStageInput { var_index } | InstData::StoreStageOutput { var_index, .. } => {
            if let Some(&mapped) = res.stage_io_remap.get(*var_index as usize) {
                *var_index = mapped;
            }
        }
        InstData::LoadPushConstant { field_index } => {
            if let Some(&mapped) = res.pc_field_remap.get(*field_index as usize) {
                *field_index = mapped;
            }
        }
        InstData::RayQueryInitialize {
            ray_query_id,
            tlas_descriptor_index,
            ..
        } => {
            *ray_query_id += res.rq_offset;
            if let Some(&mapped) = res.desc_remap.get(*tlas_descriptor_index as usize) {
                *tlas_descriptor_index = mapped;
            }
        }
        InstData::TraceRayKhr {
            tlas_descriptor_index,
            ..
        } => {
            if let Some(&mapped) = res.desc_map.get(tlas_descriptor_index) {
                *tlas_descriptor_index = mapped;
            }
        }
        InstData::HitObjectTraceRay {
            tlas_descriptor_index,
            ..
        } => {
            if let Some(&mapped) = res.desc_map.get(tlas_descriptor_index) {
                *tlas_descriptor_index = mapped;
            }
        }
        InstData::RayQueryTriangleVertexPositions { ray_query_id, .. } => {
            *ray_query_id += res.rq_offset;
        }
        InstData::RayQueryGenerateIntersection { ray_query_id, .. } => {
            *ray_query_id += res.rq_offset;
        }
        InstData::RayQueryProceed(id)
        | InstData::RayQueryCommittedType(id)
        | InstData::RayQueryCandidateType(id)
        | InstData::RayQueryConfirmIntersection(id)
        | InstData::RayQueryTerminate(id)
        | InstData::RayQueryCommittedT(id)
        | InstData::RayQueryCandidateT(id)
        | InstData::RayQueryCommittedInstanceId(id)
        | InstData::RayQueryCommittedInstanceCustomIndex(id)
        | InstData::RayQueryCommittedPrimitiveIndex(id)
        | InstData::RayQueryCommittedGeometryIndex(id)
        | InstData::RayQueryCommittedObjectToWorld(id)
        | InstData::RayQueryCommittedWorldToObject(id)
        | InstData::RayQueryCommittedFrontFace(id)
        | InstData::RayQueryCommittedBarycentrics(id) => {
            *id += res.rq_offset;
        }
        _ => {}
    }
}

fn inline_call_site(
    caller: &mut IrFunction,
    call_id: InstId,
    callee: &IrFunction,
    args: &[InstId],
    call_remap: &mut IndexMap<InstId, InstId>,
    prefix_stmts: &mut Vec<StructuredStmt>,
) {
    let res_remap = merge_callee_resources(caller, callee);
    caller.uses_bda |= callee.uses_bda;
    caller.uses_subgroups |= callee.uses_subgroups;
    caller.uses_quad_control |= callee.uses_quad_control;
    caller.required_caps |= callee.required_caps;

    let mut inst_remap: IndexMap<InstId, InstId> = IndexMap::new();
    // Keep fresh IDs strictly above every callee ID: `remap_inst_data` uses transitive
    // `remap_id`, which would otherwise cascade through overlapping source/target IDs.
    for _ in caller.insts.len()..callee.insts.len() {
        caller.emit_typed(IrType::Void, InstData::Nop);
    }
    // Pass 1: allocate placeholder slots for every non-Param callee instruction so forward
    // references (e.g. loop Phis) resolve cleanly.
    for (cid, cinst) in callee.insts.iter() {
        if let InstData::Param { index } = cinst.data
            && let Some(&arg_id) = args.get(index as usize)
        {
            inst_remap.insert(cid, arg_id);
            continue;
        }
        let fresh = caller.emit_typed(cinst.ir_ty.clone(), InstData::Nop);
        caller.insts[fresh].ty = cinst.ty;
        caller.insts[fresh].non_uniform = cinst.non_uniform;
        inst_remap.insert(cid, fresh);
    }

    // Pass 2: populate remapped instruction payloads with merged resource indices.
    for (cid, cinst) in callee.insts.iter() {
        if matches!(cinst.data, InstData::Param { .. }) {
            continue;
        }
        let target = inst_remap[&cid];
        let mut data = remap_inst_data(&cinst.data, &inst_remap);
        remap_inlined_resource_indices(&mut data, &res_remap);
        caller.insts[target].data = data;
    }

    let mut cloned_body = remap_stmts_ids(callee.body.clone(), &inst_remap);
    let ret_val = match cloned_body.last() {
        Some(StructuredStmt::Return(val)) => {
            let v = *val;
            cloned_body.pop();
            v
        }
        _ => None,
    };
    prefix_stmts.extend(cloned_body);

    if let Some(rv) = ret_val {
        call_remap.insert(call_id, rv);
    }
    caller.insts[call_id].data = InstData::Nop;
    caller.insts[call_id].non_uniform = false;
}

fn inline_stmts(
    caller: &mut IrFunction,
    stmts: Vec<StructuredStmt>,
    eligible: &IndexMap<SmolStr, IrFunction>,
    call_remap: &mut IndexMap<InstId, InstId>,
    inlined_any: &mut bool,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                // Only inline `FunctionCall`s that appear as explicit `StructuredStmt::Eval`
                // statements, avoiding operand-tree/Phi recursion and SSA dominance violations
                // when a call result is referenced across sibling branches.
                if let InstData::FunctionCall { callee, args } = caller.insts[id].data.clone()
                    && callee != caller.name
                    && let Some(callee_fn) = eligible.get(&callee)
                    && can_merge_callee_resources(caller, callee_fn)
                {
                    let remapped_args: Vec<InstId> =
                        args.iter().map(|&a| remap_id(a, call_remap)).collect();
                    inline_call_site(caller, id, callee_fn, &remapped_args, call_remap, &mut out);
                    *inlined_any = true;
                } else {
                    out.push(StructuredStmt::Eval(id));
                }
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                out.push(StructuredStmt::If {
                    cond,
                    then_body: inline_stmts(caller, then_body, eligible, call_remap, inlined_any),
                    else_body: inline_stmts(caller, else_body, eligible, call_remap, inlined_any),
                });
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: cases
                        .into_iter()
                        .map(|(k, b)| {
                            (
                                k,
                                inline_stmts(caller, b, eligible, call_remap, inlined_any),
                            )
                        })
                        .collect(),
                    default: inline_stmts(caller, default, eligible, call_remap, inlined_any),
                });
            }
            StructuredStmt::Loop {
                body,
                continuing,
                unroll,
                trip_count,
                induction_init,
                induction_step,
            } => {
                out.push(StructuredStmt::Loop {
                    body: inline_stmts(caller, body, eligible, call_remap, inlined_any),
                    continuing: inline_stmts(caller, continuing, eligible, call_remap, inlined_any),
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });
            }
            other @ (StructuredStmt::Return(_)
            | StructuredStmt::Break
            | StructuredStmt::Continue) => out.push(other),
        }
    }
    out
}

/// Executes the mandatory VSL 2.0 Rust SSA legalization pipeline (zero runtime `spirv-opt`).
pub struct PassPipeline {
    pre_inline_passes: Vec<Box<dyn IrPass>>,
    inline_pass: Option<SelectiveInlinePass>,
    passes: Vec<Box<dyn IrPass>>,
}

impl Default for PassPipeline {
    fn default() -> Self {
        Self {
            pre_inline_passes: vec![Box::new(SroaPass), Box::new(Mem2RegPass)],
            inline_pass: Some(SelectiveInlinePass),
            passes: vec![
                Box::new(SroaPass),
                Box::new(Mem2RegPass),
                Box::new(UnrollLoopsPass),
                Box::new(LowerBdaOffsetsPass),
                Box::new(ConstFoldPass),
                Box::new(SroaPass),
                Box::new(Mem2RegPass),
                Box::new(CopyPropPass),
                Box::new(SimplifyAndGvnPass),
                Box::new(LicmPass),
                Box::new(DeadStoreElimPass),
                Box::new(PropagateNonUniformPass),
                Box::new(DeadCodeEliminationPass),
            ],
        }
    }
}

impl PassPipeline {
    pub fn run_module(&self, module: &mut IrModule) {
        for func in &mut module.functions {
            for pass in &self.pre_inline_passes {
                pass.run(func);
            }
        }
        if let Some(inliner) = &self.inline_pass {
            inliner.run_module(module);
        }
        for func in &mut module.functions {
            for pass in &self.passes {
                pass.run(func);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::MemoryAccess;

    #[test]
    fn propagates_nonuniform_onto_access_chain_and_load() {
        let mut func = IrFunction::new("bindless_test");
        let raw_idx = func.emit(None, InstData::ConstU32(3));
        let nu_idx = func.emit(None, InstData::NonUniform(raw_idx));
        let ptr = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: nu_idx,
            },
        );
        let load = func.emit(None, InstData::ResourceLoad { ptr });
        func.body.push(StructuredStmt::Return(Some(load)));

        PropagateNonUniformPass.run(&mut func);
        assert!(func.insts[nu_idx].non_uniform);
        assert!(func.insts[ptr].non_uniform);
        assert!(func.insts[load].non_uniform);
    }

    #[test]
    fn lowers_bda_offset_standalone_and_as_operand_preserving_exact_align() {
        let mut func = IrFunction::new("bda_test");
        let addr = func.emit(None, InstData::ConstU64(0x1000));
        let base = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: addr,
                align: 16,
            },
        );
        let idx = func.emit(None, InstData::ConstI64(4));
        let off = func.emit(
            None,
            InstData::BdaOffset {
                base_ptr: base,
                index_i64: idx,
                stride_bytes: 24,
                align: 8,
            },
        );
        let load = func.emit(
            None,
            InstData::BdaLoad {
                ptr: off,
                access: MemoryAccess {
                    align: 8,
                    non_writable: true,
                    restrict: false,
                },
            },
        );
        // Only `load` is in `func.body`, not `off`.
        func.body.push(StructuredStmt::Return(Some(load)));

        LowerBdaOffsetsPass.run(&mut func);
        assert!(func.uses_bda);
        let InstData::ConvertUToPtr {
            addr_u64: target_u64,
            align: 8,
        } = func.insts[off].data
        else {
            panic!("expected lowered ConvertUToPtr with align 8");
        };
        let InstData::IAdd64(_, byte_off) = func.insts[target_u64].data else {
            panic!("expected IAdd64");
        };
        let InstData::IMul64(casted_idx, _) = func.insts[byte_off].data else {
            panic!("expected IMul64");
        };
        assert_eq!(func.insts[casted_idx].data, InstData::CastI64ToU64(idx));

        // ConstFoldPass must fold CastI64ToU64(ConstI64(4)) * ConstU64(24) -> ConstU64(96).
        ConstFoldPass.run(&mut func);
        assert_eq!(func.insts[byte_off].data, InstData::ConstU64(96));
    }

    #[test]
    fn lower_bda_offsets_ignores_dead_arena_entries() {
        let mut func = IrFunction::new("dead_bda_test");
        let addr = func.emit(None, InstData::ConstU64(0x1000));
        let base = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: addr,
                align: 16,
            },
        );
        let idx = func.emit(None, InstData::ConstI64(2));
        let dead_off = func.emit(
            None,
            InstData::BdaOffset {
                base_ptr: base,
                index_i64: idx,
                stride_bytes: 16,
                align: 16,
            },
        );
        func.body.push(StructuredStmt::Return(None));

        LowerBdaOffsetsPass.run(&mut func);
        assert!(!func.uses_bda);
        assert!(matches!(
            func.insts[dead_off].data,
            InstData::BdaOffset { .. }
        ));
    }

    #[test]
    fn const_fold_reduces_32bit_64bit_f32_cmp_select_and_const_if() {
        let mut func = IrFunction::new("fold_test");
        let a = func.emit(None, InstData::ConstI32(10));
        let b = func.emit(None, InstData::ConstI32(4));
        let sum = func.emit(None, InstData::IAdd32(a, b));
        let diff = func.emit(None, InstData::ISub32(sum, b));
        let prod = func.emit(None, InstData::IMul32(diff, b));
        let quot = func.emit(None, InstData::IDiv32(prod, b));
        let rem = func.emit(None, InstData::IRem32(a, b));

        let mask_lhs = func.emit(None, InstData::ConstU32(0xFF0F));
        let mask_rhs = func.emit(None, InstData::ConstU32(0x00FF));
        let masked = func.emit(None, InstData::BitAnd32(mask_lhs, mask_rhs));
        let ored = func.emit(None, InstData::BitOr32(mask_lhs, mask_rhs));
        let xored = func.emit(None, InstData::BitXor32(mask_lhs, mask_rhs));
        let shift_amt = func.emit(None, InstData::ConstU32(4));
        let shl = func.emit(None, InstData::Shl32(masked, shift_amt));
        let shr = func.emit(None, InstData::Shr32(shl, shift_amt));
        let bnot = func.emit(None, InstData::BitNot32(masked));

        let base64 = func.emit(None, InstData::ConstU64(64));
        let stride64 = func.emit(None, InstData::ConstU64(16));
        let idx_u64 = func.emit(None, InstData::ConstU64(3));
        let mul_u64 = func.emit(None, InstData::IMul64(idx_u64, stride64));
        let add_u64 = func.emit(None, InstData::IAdd64(base64, mul_u64));

        // Mixed signedness must NOT fold.
        let signed_idx = func.emit(None, InstData::ConstI64(3));
        let mixed_mul = func.emit(None, InstData::IMul64(signed_idx, stride64));
        let cast_u64 = func.emit(None, InstData::CastI64ToU64(signed_idx));
        let cast_mul = func.emit(None, InstData::IMul64(cast_u64, stride64));

        // Finite f32 folding.
        let fa = func.emit(None, InstData::const_f32(1.5));
        let fb = func.emit(None, InstData::const_f32(2.0));
        let fsum = func.emit(None, InstData::FAdd32(fa, fb));
        let float_diff = func.emit(None, InstData::FSub32(fsum, fa));
        let fmul = func.emit(None, InstData::FMul32(fsum, fb));
        let fdiv = func.emit(None, InstData::FDiv32(fmul, fb));

        // Comparisons, logical ops, and Select.
        let cmp_less = func.emit(None, InstData::ICmpLt(b, a));
        let cmp_less_eq = func.emit(None, InstData::ICmpLe(a, a));
        let cmp_greater = func.emit(None, InstData::ICmpGt(a, b));
        let cmp_greater_eq = func.emit(None, InstData::ICmpGe(b, a));
        let cmp_not_eq = func.emit(None, InstData::ICmpNe(a, b));
        let cmp_equal = func.emit(None, InstData::ICmpEq(diff, a));
        let ucmp_lt = func.emit(None, InstData::UCmpLt(mask_rhs, mask_lhs));
        let land = func.emit(None, InstData::LogicalAnd(cmp_less, cmp_equal));
        let lor = func.emit(None, InstData::LogicalOr(cmp_greater_eq, cmp_greater));
        let lnot = func.emit(None, InstData::LogicalNot(cmp_greater_eq));
        let sel = func.emit(
            None,
            InstData::Select {
                cond: cmp_less,
                on_true: prod,
                on_false: b,
            },
        );

        let barrier = func.emit(None, InstData::WorkgroupBarrier);
        func.body.push(StructuredStmt::If {
            cond: cmp_equal,
            then_body: vec![StructuredStmt::Eval(barrier)],
            else_body: Vec::new(),
        });

        ConstFoldPass.run(&mut func);

        assert_eq!(func.insts[sum].data, InstData::ConstI32(14));
        assert_eq!(func.insts[diff].data, InstData::ConstI32(10));
        assert_eq!(func.insts[prod].data, InstData::ConstI32(40));
        assert_eq!(func.insts[quot].data, InstData::ConstI32(10));
        assert_eq!(func.insts[rem].data, InstData::ConstI32(2));
        assert_eq!(func.insts[masked].data, InstData::ConstU32(0x000F));
        assert_eq!(func.insts[ored].data, InstData::ConstU32(0xFFFF));
        assert_eq!(func.insts[xored].data, InstData::ConstU32(0xFFF0));
        assert_eq!(func.insts[shl].data, InstData::ConstU32(0x00F0));
        assert_eq!(func.insts[shr].data, InstData::ConstU32(0x000F));
        assert_eq!(func.insts[bnot].data, InstData::ConstU32(!0x000F_u32));
        assert_eq!(func.insts[mul_u64].data, InstData::ConstU64(48));
        assert_eq!(func.insts[add_u64].data, InstData::ConstU64(112));
        assert!(matches!(func.insts[mixed_mul].data, InstData::IMul64(_, _)));
        assert_eq!(func.insts[cast_u64].data, InstData::ConstU64(3));
        assert_eq!(func.insts[cast_mul].data, InstData::ConstU64(48));
        assert_eq!(func.insts[fsum].data, InstData::const_f32(3.5));
        assert_eq!(func.insts[float_diff].data, InstData::const_f32(2.0));
        assert_eq!(func.insts[fmul].data, InstData::const_f32(7.0));
        assert_eq!(func.insts[fdiv].data, InstData::const_f32(3.5));
        assert_eq!(func.insts[cmp_less].data, InstData::ConstBool(true));
        assert_eq!(func.insts[cmp_less_eq].data, InstData::ConstBool(true));
        assert_eq!(func.insts[cmp_greater].data, InstData::ConstBool(true));
        assert_eq!(func.insts[cmp_greater_eq].data, InstData::ConstBool(false));
        assert_eq!(func.insts[cmp_not_eq].data, InstData::ConstBool(true));
        assert_eq!(func.insts[cmp_equal].data, InstData::ConstBool(true));
        assert_eq!(func.insts[ucmp_lt].data, InstData::ConstBool(true));
        assert_eq!(func.insts[land].data, InstData::ConstBool(true));
        assert_eq!(func.insts[lor].data, InstData::ConstBool(true));
        assert_eq!(func.insts[lnot].data, InstData::ConstBool(true));
        assert_eq!(func.insts[sel].data, InstData::ConstI32(40));
        assert_eq!(func.body, vec![StructuredStmt::Eval(barrier)]);
    }

    #[test]
    fn unroll_loops_clones_transitive_operands_and_steps_induction_var() {
        let mut func = IrFunction::new("unroll_test");
        let barrier = func.emit(None, InstData::WorkgroupBarrier);
        let ind = func.emit(None, InstData::ConstI64(0));
        let stride = func.emit(None, InstData::ConstI64(16));
        let offset = func.emit(None, InstData::IMul64(ind, stride));
        let addr = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: offset,
                align: 16,
            },
        );
        let val = func.emit(None, InstData::ConstU32(1));
        let store = func.emit(
            None,
            InstData::BdaStore {
                ptr: addr,
                value: val,
                access: MemoryAccess {
                    align: 16,
                    non_writable: false,
                    restrict: false,
                },
            },
        );

        func.body.push(StructuredStmt::Loop {
            body: vec![StructuredStmt::Eval(store)],
            continuing: vec![StructuredStmt::Eval(barrier)],
            unroll: true,
            trip_count: Some(3),
            induction_init: Some(ind),
            induction_step: 1,
        });
        func.body.push(StructuredStmt::Loop {
            body: vec![StructuredStmt::Eval(barrier)],
            continuing: Vec::new(),
            unroll: true,
            trip_count: Some(MAX_UNROLL_ITERATIONS + 1),
            induction_init: None,
            induction_step: 0,
        });

        UnrollLoopsPass.run(&mut func);

        // First loop unrolled to 3 * (1 body + 1 continuing) = 6 Eval statements,
        // followed by the oversized loop left intact.
        assert_eq!(func.body.len(), 7);
        let store_ids: Vec<InstId> = [0, 2, 4]
            .into_iter()
            .map(|idx| match func.body[idx] {
                StructuredStmt::Eval(id) => id,
                ref other => panic!("expected Eval after unroll, got {other:?}"),
            })
            .collect();
        let unique_stores: IndexSet<InstId> = store_ids.iter().copied().collect();
        assert_eq!(unique_stores.len(), 3);

        // Verify each unrolled store received a distinct transitive pointer chain and stepped index.
        let mut ind_values = Vec::new();
        for sid in store_ids {
            let InstData::BdaStore { ptr, .. } = func.insts[sid].data else {
                panic!("expected BdaStore");
            };
            let InstData::ConvertUToPtr { addr_u64, .. } = func.insts[ptr].data else {
                panic!("expected ConvertUToPtr");
            };
            let InstData::IMul64(lhs, _) = func.insts[addr_u64].data else {
                panic!("expected IMul64");
            };
            let InstData::ConstI64(v) = func.insts[lhs].data else {
                panic!("expected stepped ConstI64");
            };
            ind_values.push(v);
        }
        assert_eq!(ind_values, vec![0, 1, 2]);
        assert!(matches!(
            func.body[6],
            StructuredStmt::Loop {
                unroll: true,
                trip_count: Some(65),
                ..
            }
        ));
    }

    #[test]
    fn unroll_loops_preserves_external_loads_and_phis_defined_before_loop() {
        let mut func = IrFunction::new("unroll_external_test");
        let base_addr = func.emit(None, InstData::ConstU64(0x2000));
        let base_ptr = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: base_addr,
                align: 16,
            },
        );
        let ext_bda_load = func.emit(
            None,
            InstData::BdaLoad {
                ptr: base_ptr,
                access: MemoryAccess {
                    align: 16,
                    non_writable: true,
                    restrict: false,
                },
            },
        );
        let res_idx = func.emit(None, InstData::ConstU32(0));
        let res_ptr = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: res_idx,
            },
        );
        let ext_res_load = func.emit(None, InstData::ResourceLoad { ptr: res_ptr });
        let ext_phi = func.emit(
            None,
            InstData::Phi {
                incoming: vec![(ext_bda_load, 0), (ext_res_load, 1)],
            },
        );

        let ind = func.emit(None, InstData::ConstU32(0));
        let sum = func.emit(None, InstData::IAdd32(ext_phi, ind));
        let wg_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: Some(ind),
                value: sum,
            },
        );

        func.body.push(StructuredStmt::Eval(ext_bda_load));
        func.body.push(StructuredStmt::Loop {
            body: vec![StructuredStmt::Eval(wg_store)],
            continuing: Vec::new(),
            unroll: true,
            trip_count: Some(3),
            induction_init: Some(ind),
            induction_step: 1,
        });

        UnrollLoopsPass.run(&mut func);

        // External loads and Phi must NOT have been cloned (still exactly 1 of each in arena).
        let bda_load_count = func
            .insts
            .iter()
            .filter(|(_, inst)| matches!(inst.data, InstData::BdaLoad { .. }))
            .count();
        let res_load_count = func
            .insts
            .iter()
            .filter(|(_, inst)| matches!(inst.data, InstData::ResourceLoad { .. }))
            .count();
        let phi_count = func
            .insts
            .iter()
            .filter(|(_, inst)| matches!(inst.data, InstData::Phi { .. }))
            .count();
        assert_eq!(bda_load_count, 1);
        assert_eq!(res_load_count, 1);
        assert_eq!(phi_count, 1);

        // All 3 unrolled WorkgroupStore instructions must reference the original ext_phi via sum.
        assert_eq!(func.body.len(), 4);
        for (expected_ind, stmt) in func.body[1..].iter().enumerate() {
            let StructuredStmt::Eval(store_id) = *stmt else {
                panic!("expected Eval(WorkgroupStore)");
            };
            let InstData::WorkgroupStore {
                index: Some(idx_id),
                value,
                ..
            } = func.insts[store_id].data
            else {
                panic!("expected WorkgroupStore");
            };
            assert_eq!(
                func.insts[idx_id].data,
                InstData::ConstU32(expected_ind as u32)
            );
            let InstData::IAdd32(lhs, rhs) = func.insts[value].data else {
                panic!("expected IAdd32");
            };
            assert_eq!(lhs, ext_phi);
            assert_eq!(
                func.insts[rhs].data,
                InstData::ConstU32(expected_ind as u32)
            );
        }
    }

    #[test]
    fn dce_clears_dead_arena_entries_to_nop_and_recomputes_feature_flags() {
        let mut func = IrFunction::new("dce_test");
        func.uses_bda = true;
        func.uses_subgroups = true;
        let dead_u64 = func.emit(None, InstData::ConstU64(42));
        let dead_ptr = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: dead_u64,
                align: 16,
            },
        );
        let dead_idx = func.emit(None, InstData::ConstU32(7));
        let dead_nu = func.emit(None, InstData::NonUniform(dead_idx));
        func.insts[dead_nu].non_uniform = true;
        func.body.push(StructuredStmt::Eval(dead_ptr));
        func.body.push(StructuredStmt::Eval(dead_nu));
        func.body.push(StructuredStmt::Return(None));

        DeadCodeEliminationPass.run(&mut func);

        assert_eq!(func.body, vec![StructuredStmt::Return(None)]);
        assert_eq!(func.insts[dead_u64].data, InstData::Nop);
        assert_eq!(func.insts[dead_ptr].data, InstData::Nop);
        assert_eq!(func.insts[dead_nu].data, InstData::Nop);
        assert!(!func.insts[dead_nu].non_uniform);
        assert!(!func.uses_bda);
        assert!(!func.uses_subgroups);
    }

    #[test]
    fn unroll_loops_updates_loop_carried_phi_across_iterations() {
        let mut func = IrFunction::new("unroll_phi_test");
        let init = func.emit(None, InstData::ConstU32(0));
        let step = func.emit(None, InstData::ConstU32(5));
        let phi = func.emit(
            None,
            InstData::Phi {
                incoming: Vec::new(),
            },
        );
        let next = func.emit(None, InstData::IAdd32(phi, step));
        func.insts[phi].data = InstData::Phi {
            incoming: vec![(init, 0), (next, 1)],
        };
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: next,
            },
        );

        func.body.push(StructuredStmt::Loop {
            body: vec![StructuredStmt::Eval(store)],
            continuing: Vec::new(),
            unroll: true,
            trip_count: Some(3),
            induction_init: None,
            induction_step: 0,
        });
        func.body.push(StructuredStmt::Return(Some(phi)));

        UnrollLoopsPass.run(&mut func);
        ConstFoldPass.run(&mut func);

        assert_eq!(func.body.len(), 4);
        let mut stored_vals = Vec::new();
        for stmt in &func.body[..3] {
            let StructuredStmt::Eval(store_id) = *stmt else {
                panic!("expected Eval(WorkgroupStore)");
            };
            let InstData::WorkgroupStore { value, .. } = func.insts[store_id].data else {
                panic!("expected WorkgroupStore");
            };
            let InstData::ConstU32(v) = func.insts[value].data else {
                panic!("expected folded ConstU32, got {:?}", func.insts[value].data);
            };
            stored_vals.push(v);
        }
        assert_eq!(stored_vals, vec![5, 10, 15]);
        let StructuredStmt::Return(Some(ret_id)) = func.body[3] else {
            panic!("expected Return(Some(_))");
        };
        assert_eq!(func.insts[ret_id].data, InstData::ConstU32(15));
    }

    #[test]
    fn const_fold_handles_cast_to_u32_and_operands_covers_new_variants() {
        let mut func = IrFunction::new("cast_u32_test");
        let u64_val = func.emit(None, InstData::ConstU64(0x1_0000_002A));
        let f32_val = func.emit(None, InstData::const_f32(7.5));
        let bool_val = func.emit(None, InstData::ConstBool(true));
        let cast_u64 = func.emit(None, InstData::CastToU32(u64_val));
        let cast_float = func.emit(None, InstData::CastToU32(f32_val));
        let cast_bool = func.emit(None, InstData::CastToU32(bool_val));

        let idx = func.emit(None, InstData::ConstU32(2));
        let wg_addr = func.emit(
            None,
            InstData::WorkgroupAddr {
                var_index: 0,
                index: Some(idx),
            },
        );
        let coord = func.emit(None, InstData::ConstI32(0));
        let lod = func.emit(None, InstData::ConstI32(0));
        let sample = func.emit(
            None,
            InstData::ImageSample {
                descriptor_index: 0,
                image_inst: None,
                coord,
                lod,
                is_fetch: true,
            },
        );

        assert_eq!(func.insts[cast_u64].data.operands(), vec![u64_val]);
        assert_eq!(func.insts[wg_addr].data.operands(), vec![idx]);
        assert_eq!(func.insts[sample].data.operands(), vec![coord, lod]);

        ConstFoldPass.run(&mut func);

        assert_eq!(func.insts[cast_u64].data, InstData::ConstU32(42));
        assert_eq!(func.insts[cast_float].data, InstData::ConstU32(7));
        assert_eq!(func.insts[cast_bool].data, InstData::ConstU32(1));
    }

    #[test]
    fn nonuniform_image_sample_chain_stays_live_through_dce() {
        let mut func = IrFunction::new("bindless_sample_dce");
        let raw_idx = func.emit(None, InstData::ConstU32(2));
        let nu_idx = func.emit(None, InstData::NonUniform(raw_idx));
        let chain = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: nu_idx,
            },
        );
        let load = func.emit(None, InstData::ResourceLoad { ptr: chain });
        let coord = func.emit(None, InstData::ConstI32(0));
        let image_lod = func.emit(None, InstData::ConstI32(0));
        let sample = func.emit(
            None,
            InstData::ImageSample {
                descriptor_index: 0,
                image_inst: Some(load),
                coord,
                lod: image_lod,
                is_fetch: true,
            },
        );
        let store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 1,
                coord,
                value: sample,
            },
        );
        func.body.push(StructuredStmt::Eval(store));

        PropagateNonUniformPass.run(&mut func);
        DeadCodeEliminationPass.run(&mut func);

        assert!(func.insts[nu_idx].non_uniform);
        assert!(func.insts[chain].non_uniform);
        assert!(func.insts[load].non_uniform);
        assert!(func.insts[sample].non_uniform);
        assert_ne!(func.insts[nu_idx].data, InstData::Nop);
        assert_ne!(func.insts[chain].data, InstData::Nop);
        assert_ne!(func.insts[load].data, InstData::Nop);
        assert_ne!(func.insts[sample].data, InstData::Nop);
    }

    #[test]
    fn const_fold_if_resolves_indirect_phi_and_preserves_other_control_flow_phis() {
        let mut func = IrFunction::new("fold_indirect_phi");
        let cond_true = func.emit(None, InstData::ConstBool(true));
        let then_val = func.emit(None, InstData::ConstU32(11));
        let else_val = func.emit(None, InstData::ConstU32(22));
        let if_phi = func.emit(
            None,
            InstData::Phi {
                incoming: vec![(then_val, 0), (else_val, 1)],
            },
        );
        let store_folded = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: if_phi,
            },
        );

        // Non-folded dynamic If whose Phi must remain untouched.
        let dyn_cond = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let dyn_a = func.emit(None, InstData::ConstU32(33));
        let dyn_b = func.emit(None, InstData::ConstU32(44));
        let dyn_phi = func.emit(
            None,
            InstData::Phi {
                incoming: vec![(dyn_a, 0), (dyn_b, 1)],
            },
        );

        func.body.push(StructuredStmt::If {
            cond: cond_true,
            then_body: vec![StructuredStmt::Eval(then_val)],
            else_body: vec![StructuredStmt::Eval(else_val)],
        });
        func.body.push(StructuredStmt::Eval(if_phi));
        func.body.push(StructuredStmt::Eval(store_folded));
        func.body.push(StructuredStmt::If {
            cond: dyn_cond,
            then_body: vec![StructuredStmt::Eval(dyn_a)],
            else_body: vec![StructuredStmt::Eval(dyn_b)],
        });
        func.body.push(StructuredStmt::Return(Some(dyn_phi)));

        ConstFoldPass.run(&mut func);

        assert_eq!(func.insts[if_phi].data, InstData::Nop);
        let InstData::WorkgroupStore { value, .. } = func.insts[store_folded].data else {
            panic!("expected WorkgroupStore");
        };
        assert_eq!(value, then_val);
        assert!(matches!(func.insts[dyn_phi].data, InstData::Phi { .. }));
    }

    #[test]
    fn make_vec2_propagates_nonuniform_and_preserves_live_operands_through_dce() {
        let mut func = IrFunction::new("make_vec2_passes");
        let x = func.emit(None, InstData::ConstU32(4));
        let y_raw = func.emit(None, InstData::ConstU32(9));
        let y_nu = func.emit(None, InstData::NonUniform(y_raw));
        let vec = func.emit(None, InstData::MakeVec2(x, y_nu));
        let dead_vec = func.emit(None, InstData::MakeVec2(x, x));
        let val = func.emit(None, InstData::const_f32(1.0));
        let store = func.emit(
            None,
            InstData::ImageStore {
                descriptor_index: 0,
                coord: vec,
                value: val,
            },
        );
        func.body.push(StructuredStmt::Eval(dead_vec));
        func.body.push(StructuredStmt::Eval(store));

        assert_eq!(func.insts[vec].data.operands(), vec![x, y_nu]);

        ConstFoldPass.run(&mut func);
        PropagateNonUniformPass.run(&mut func);
        assert!(func.insts[vec].non_uniform);
        assert!(!func.insts[dead_vec].non_uniform);

        DeadCodeEliminationPass.run(&mut func);
        assert!(matches!(func.insts[vec].data, InstData::MakeVec2(_, _)));
        assert!(matches!(func.insts[dead_vec].data, InstData::Nop));
        assert!(matches!(func.insts[x].data, InstData::ConstU32(4)));
        assert!(matches!(func.insts[y_nu].data, InstData::NonUniform(_)));
    }

    #[test]
    fn simplify_and_gvn_eliminates_zero_offset_bda_and_deduplicates_pure_loads() {
        let mut func = IrFunction::new("gvn_bda");
        let base_addr = func.emit(None, InstData::ConstU64(0x1000));
        let base_ptr = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: base_addr,
                align: 16,
            },
        );
        let zero = func.emit(None, InstData::ConstU64(0));
        let p2u_a = func.emit(None, InstData::ConvertPtrToU(base_ptr));
        let add0 = func.emit(None, InstData::IAdd64(p2u_a, zero));
        let roundtrip = func.emit(
            None,
            InstData::ConvertUToPtr {
                addr_u64: add0,
                align: 16,
            },
        );
        let gid_x1 = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: crate::ssa::BuiltinKind::GlobalInvocationId,
                component: Some(0),
            },
        );
        let gid_x2 = func.emit(
            None,
            InstData::LoadBuiltin {
                kind: crate::ssa::BuiltinKind::GlobalInvocationId,
                component: Some(0),
            },
        );
        let sum = func.emit(None, InstData::IAdd32(gid_x1, gid_x2));
        let store = func.emit(
            None,
            InstData::BdaStore {
                ptr: roundtrip,
                value: sum,
                access: MemoryAccess {
                    align: 16,
                    non_writable: false,
                    restrict: false,
                },
            },
        );
        func.body.push(StructuredStmt::Eval(base_ptr));
        func.body.push(StructuredStmt::Eval(store));

        SimplifyAndGvnPass.run(&mut func);
        DeadCodeEliminationPass.run(&mut func);

        let InstData::BdaStore { ptr, value, .. } = func.insts[store].data else {
            panic!("expected BdaStore");
        };
        assert_eq!(
            ptr, base_ptr,
            "0-offset BDA roundtrip must collapse to base_ptr"
        );
        let InstData::IAdd32(lhs, rhs) = func.insts[value].data else {
            panic!("expected IAdd32");
        };
        assert_eq!(lhs, rhs, "duplicate LoadBuiltin must be GVN-deduplicated");
    }

    #[test]
    fn dce_prunes_empty_conditionals_and_unreachable_tail_after_return() {
        let mut func = IrFunction::new("dce_prune");
        let cond = func.emit(None, InstData::ConstBool(true));
        let dead_val = func.emit(None, InstData::ConstU32(42));
        let dead_after_ret = func.emit(None, InstData::WorkgroupBarrier);
        func.body.push(StructuredStmt::If {
            cond,
            then_body: vec![StructuredStmt::Eval(dead_val)],
            else_body: Vec::new(),
        });
        func.body.push(StructuredStmt::Return(None));
        func.body.push(StructuredStmt::Eval(dead_after_ret));

        DeadCodeEliminationPass.run(&mut func);
        assert_eq!(func.body.len(), 1);
        assert!(matches!(func.body[0], StructuredStmt::Return(None)));
        assert_eq!(func.insts[dead_after_ret].data, InstData::Nop);
    }

    #[test]
    fn sroa_splits_local_struct_and_small_array_for_mem2reg() {
        let mut func = IrFunction::new("sroa_test");
        let struct_ty = IrType::Struct(crate::ssa::IrStructLayout {
            name: SmolStr::new_static("Pair"),
            fields: vec![
                crate::ssa::IrStructFieldLayout {
                    name: SmolStr::new_static("a"),
                    byte_offset: 0,
                    byte_size: 4,
                    ty: IrType::Scalar(ScalarIrTy::F32),
                },
                crate::ssa::IrStructFieldLayout {
                    name: SmolStr::new_static("b"),
                    byte_offset: 4,
                    byte_size: 4,
                    ty: IrType::Scalar(ScalarIrTy::F32),
                },
            ],
            size: 8,
            align: 4,
            layout: vsl_hir::LayoutKind::Scalar,
        });
        func.local_vars.push(IrLocalVar {
            name: SmolStr::new_static("p"),
            ty: struct_ty,
        });
        func.local_vars.push(IrLocalVar {
            name: SmolStr::new_static("arr"),
            ty: IrType::Array {
                element: Box::new(IrType::Scalar(ScalarIrTy::U32)),
                len: 2,
            },
        });

        let p_addr = func.emit(None, InstData::LocalAddr { local_index: 0 });
        let idx0 = func.emit(None, InstData::ConstU32(0));
        let idx1 = func.emit(None, InstData::ConstU32(1));
        let p_a = func.emit_typed(
            IrType::Pointer {
                pointee: Box::new(IrType::Scalar(ScalarIrTy::F32)),
                align: 4,
                storage: IrStorageClass::Function,
            },
            InstData::AccessChain {
                base_ptr: p_addr,
                indices: vec![idx0],
            },
        );
        let p_b = func.emit_typed(
            IrType::Pointer {
                pointee: Box::new(IrType::Scalar(ScalarIrTy::F32)),
                align: 4,
                storage: IrStorageClass::Function,
            },
            InstData::AccessChain {
                base_ptr: p_addr,
                indices: vec![idx1],
            },
        );
        let c1 = func.emit(None, InstData::const_f32(3.0));
        let c2 = func.emit(None, InstData::const_f32(4.0));
        let st_a = func.emit(
            None,
            InstData::LocalStore {
                ptr: p_a,
                value: c1,
            },
        );
        let st_b = func.emit(
            None,
            InstData::LocalStore {
                ptr: p_b,
                value: c2,
            },
        );
        let ld_a = func.emit(None, InstData::LocalLoad { ptr: p_a });
        let ld_b = func.emit(None, InstData::LocalLoad { ptr: p_b });
        let sum = func.emit(None, InstData::FAdd32(ld_a, ld_b));

        let arr_addr = func.emit(None, InstData::LocalAddr { local_index: 1 });
        let arr_1 = func.emit_typed(
            IrType::Pointer {
                pointee: Box::new(IrType::Scalar(ScalarIrTy::U32)),
                align: 4,
                storage: IrStorageClass::Function,
            },
            InstData::AccessChain {
                base_ptr: arr_addr,
                indices: vec![idx1],
            },
        );
        let st_arr = func.emit(
            None,
            InstData::LocalStore {
                ptr: arr_1,
                value: idx1,
            },
        );

        func.body.push(StructuredStmt::Eval(st_a));
        func.body.push(StructuredStmt::Eval(st_b));
        func.body.push(StructuredStmt::Eval(st_arr));
        func.body.push(StructuredStmt::Return(Some(sum)));

        SroaPass.run(&mut func);
        assert!(matches!(func.insts[p_a].data, InstData::LocalAddr { .. }));
        assert!(matches!(func.insts[p_b].data, InstData::LocalAddr { .. }));
        assert!(matches!(func.insts[arr_1].data, InstData::LocalAddr { .. }));

        Mem2RegPass.run(&mut func);
        ConstFoldPass.run(&mut func);
        let Some(StructuredStmt::Return(Some(ret_id))) = func.body.last() else {
            panic!("expected Return");
        };
        assert_eq!(func.insts[*ret_id].data, InstData::const_f32(7.0));
    }

    #[test]
    fn mem2reg_promotes_locals_across_continue_inside_nested_if_in_loop() {
        let mut func = IrFunction::new("mem2reg_loop_continue");
        func.local_vars.push(IrLocalVar {
            name: SmolStr::new_static("x"),
            ty: IrType::Scalar(ScalarIrTy::U32),
        });
        let x_addr = func.emit(None, InstData::LocalAddr { local_index: 0 });
        let zero = func.emit(None, InstData::ConstU32(0));
        let one = func.emit(None, InstData::ConstU32(1));
        let two = func.emit(None, InstData::ConstU32(2));
        let init_store = func.emit(
            None,
            InstData::LocalStore {
                ptr: x_addr,
                value: zero,
            },
        );

        let cond_skip = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let cond_done = func.emit(None, InstData::LoadPushConstant { field_index: 1 });

        let ld_in_if = func.emit(None, InstData::LocalLoad { ptr: x_addr });
        let inc_1 = func.emit(None, InstData::IAdd32(ld_in_if, one));
        let st_in_if = func.emit(
            None,
            InstData::LocalStore {
                ptr: x_addr,
                value: inc_1,
            },
        );

        let ld_body = func.emit(None, InstData::LocalLoad { ptr: x_addr });
        let inc_2 = func.emit(None, InstData::IAdd32(ld_body, two));
        let st_body = func.emit(
            None,
            InstData::LocalStore {
                ptr: x_addr,
                value: inc_2,
            },
        );

        let ld_ret = func.emit(None, InstData::LocalLoad { ptr: x_addr });

        func.body.push(StructuredStmt::Eval(init_store));
        func.body.push(StructuredStmt::Loop {
            body: vec![
                StructuredStmt::If {
                    cond: cond_skip,
                    then_body: vec![StructuredStmt::Eval(st_in_if), StructuredStmt::Continue],
                    else_body: Vec::new(),
                },
                StructuredStmt::Eval(st_body),
                StructuredStmt::If {
                    cond: cond_done,
                    then_body: vec![StructuredStmt::Break],
                    else_body: Vec::new(),
                },
            ],
            continuing: Vec::new(),
            unroll: false,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });
        func.body.push(StructuredStmt::Return(Some(ld_ret)));

        Mem2RegPass.run(&mut func);

        // No LocalLoad or LocalStore should remain.
        assert!(func.insts.iter().all(|(_, inst)| !matches!(
            inst.data,
            InstData::LocalLoad { .. } | InstData::LocalStore { .. }
        )));

        // The loop continuing block must hold a merge Phi combining both `inc_2` (fallthrough)
        // and `inc_1` (the `continue` inside `if`), and the loop header Phi must take that
        // continuing Phi as its back-edge.
        let StructuredStmt::Loop {
            body: loop_body,
            continuing,
            ..
        } = &func.body[0]
        else {
            panic!("expected Loop at body[0]");
        };
        let Some(&StructuredStmt::Eval(header_phi)) = loop_body.first() else {
            panic!("expected header Phi Eval at start of loop body");
        };
        let Some(&StructuredStmt::Eval(cont_phi)) = continuing.first() else {
            panic!("expected continuing Phi Eval capturing continue edge");
        };
        let InstData::Phi {
            incoming: cont_incoming,
        } = &func.insts[cont_phi].data
        else {
            panic!("expected Phi in continuing");
        };
        assert_eq!(cont_incoming.len(), 2);
        assert!(cont_incoming.iter().any(|&(v, _)| v == inc_1));
        assert!(cont_incoming.iter().any(|&(v, _)| v == inc_2));

        let InstData::Phi {
            incoming: header_incoming,
        } = &func.insts[header_phi].data
        else {
            panic!("expected header Phi");
        };
        assert_eq!(header_incoming, &vec![(zero, 0), (cont_phi, 1)]);

        // Post-loop return must receive `inc_2` from the `Break` edge.
        assert_eq!(func.body[1], StructuredStmt::Return(Some(inc_2)));
    }

    #[test]
    fn licm_hoists_loop_invariant_pure_chain_before_loop() {
        let mut func = IrFunction::new("licm_test");
        let pc0 = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let two = func.emit(None, InstData::ConstU32(2));
        let inv1 = func.emit(None, InstData::IMul32(pc0, two));
        let inv2 = func.emit(None, InstData::IAdd32(inv1, two));
        let ind = func.emit(None, InstData::ConstU32(0));
        let variant = func.emit(None, InstData::IAdd32(inv2, ind));
        let store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: Some(ind),
                value: variant,
            },
        );

        func.body.push(StructuredStmt::Loop {
            body: vec![
                StructuredStmt::Eval(inv1),
                StructuredStmt::Eval(inv2),
                StructuredStmt::Eval(variant),
                StructuredStmt::Eval(store),
            ],
            continuing: Vec::new(),
            unroll: false,
            trip_count: None,
            induction_init: Some(ind),
            induction_step: 1,
        });

        LicmPass.run(&mut func);

        assert_eq!(func.body.len(), 3);
        assert_eq!(func.body[0], StructuredStmt::Eval(inv1));
        assert_eq!(func.body[1], StructuredStmt::Eval(inv2));
        let StructuredStmt::Loop { body, .. } = &func.body[2] else {
            panic!("expected Loop after hoisted statements");
        };
        assert_eq!(
            body,
            &vec![StructuredStmt::Eval(variant), StructuredStmt::Eval(store)]
        );
    }

    #[test]
    fn copy_prop_eliminates_trivial_phi_and_identity_shuffle_and_reconstruct() {
        let mut func = IrFunction::new("copy_prop_test");
        let x = func.emit(None, InstData::const_f32(1.0));
        let y = func.emit(None, InstData::const_f32(2.0));
        let v = func.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            InstData::MakeVec2(x, y),
        );
        let trivial_phi = func.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            InstData::Phi {
                incoming: vec![(v, 0), (v, 1)],
            },
        );
        let id_shuf = func.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            InstData::VectorShuffle {
                lhs: trivial_phi,
                rhs: trivial_phi,
                components: vec![0, 1],
            },
        );
        let ex0 = func.emit(
            None,
            InstData::ExtractComponent {
                vec: id_shuf,
                index: 0,
            },
        );
        let ex1 = func.emit(
            None,
            InstData::ExtractComponent {
                vec: id_shuf,
                index: 1,
            },
        );
        let rebuilt = func.emit_typed(
            IrType::Vector {
                scalar: ScalarIrTy::F32,
                len: 2,
            },
            InstData::CompositeConstruct {
                elems: vec![ex0, ex1],
            },
        );
        func.body.push(StructuredStmt::Return(Some(rebuilt)));

        CopyPropPass.run(&mut func);

        assert_eq!(func.body, vec![StructuredStmt::Return(Some(v))]);
        assert_eq!(func.insts[trivial_phi].data, InstData::Nop);
        assert_eq!(func.insts[id_shuf].data, InstData::Nop);
        assert_eq!(func.insts[rebuilt].data, InstData::Nop);
    }

    #[test]
    fn dse_removes_overwritten_local_store_but_keeps_observed_store() {
        let mut func = IrFunction::new("dse_test");
        func.local_vars.push(IrLocalVar {
            name: SmolStr::new_static("a"),
            ty: IrType::Scalar(ScalarIrTy::U32),
        });
        let addr = func.emit(None, InstData::LocalAddr { local_index: 0 });
        let c1 = func.emit(None, InstData::ConstU32(1));
        let c2 = func.emit(None, InstData::ConstU32(2));
        let c3 = func.emit(None, InstData::ConstU32(3));
        let st1 = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: c1,
            },
        );
        let st2 = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: c2,
            },
        );
        let ld = func.emit(None, InstData::LocalLoad { ptr: addr });
        let st3 = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: c3,
            },
        );

        func.body.push(StructuredStmt::Eval(st1));
        func.body.push(StructuredStmt::Eval(st2));
        func.body.push(StructuredStmt::Eval(ld));
        func.body.push(StructuredStmt::Eval(st3));
        func.body.push(StructuredStmt::Return(Some(ld)));

        DeadStoreElimPass.run(&mut func);

        assert_eq!(func.insts[st1].data, InstData::Nop);
        assert!(matches!(func.insts[st2].data, InstData::LocalStore { .. }));
        assert!(matches!(func.insts[st3].data, InstData::LocalStore { .. }));
        assert_eq!(func.body.len(), 4);
    }

    #[test]
    fn const_fold_handles_f16_fconvert_and_vector_arithmetic_and_shuffle() {
        let mut func = IrFunction::new("f16_vec_fold");
        let f32_val = func.emit(None, InstData::const_f32(1.5));
        let to_f16 = func.emit(
            None,
            InstData::FConvert {
                value: f32_val,
                target: ScalarIrTy::F16,
            },
        );
        let back_f32 = func.emit(
            None,
            InstData::FConvert {
                value: to_f16,
                target: ScalarIrTy::F32,
            },
        );
        let half_two = func.emit(None, InstData::ConstF16(f32_to_f16_bits(2.0)));
        let half_sum = func.emit(None, InstData::FAdd32(to_f16, half_two));

        let vx = func.emit(None, InstData::const_f32(2.0));
        let vy = func.emit(None, InstData::const_f32(3.0));
        let va = func.emit(None, InstData::MakeVec2(vx, vy));
        let scale = func.emit(None, InstData::const_f32(4.0));
        let scaled = func.emit(
            None,
            InstData::VectorTimesScalar {
                vec: va,
                scalar: scale,
            },
        );
        let swapped = func.emit(
            None,
            InstData::VectorShuffle {
                lhs: scaled,
                rhs: scaled,
                components: vec![1, 0],
            },
        );
        let comp0 = func.emit(
            None,
            InstData::ExtractComponent {
                vec: swapped,
                index: 0,
            },
        );

        ConstFoldPass.run(&mut func);

        assert_eq!(func.insts[to_f16].data, InstData::ConstF16(0x3E00));
        assert_eq!(func.insts[back_f32].data, InstData::const_f32(1.5));
        assert_eq!(
            func.insts[half_sum].data,
            InstData::ConstF16(f32_to_f16_bits(3.5))
        );
        assert_eq!(func.insts[comp0].data, InstData::const_f32(12.0));
    }

    #[test]
    fn unroll_loops_respects_max_unroll_body_insts_cap() {
        let mut func = IrFunction::new("unroll_body_cap");
        let mut large_body = Vec::with_capacity(MAX_UNROLL_BODY_INSTS + 2);
        for _ in 0..=MAX_UNROLL_BODY_INSTS {
            let b = func.emit(None, InstData::WorkgroupBarrier);
            large_body.push(StructuredStmt::Eval(b));
        }
        func.body.push(StructuredStmt::Loop {
            body: large_body,
            continuing: Vec::new(),
            unroll: true,
            trip_count: Some(4),
            induction_init: None,
            induction_step: 0,
        });

        UnrollLoopsPass.run(&mut func);

        assert_eq!(func.body.len(), 1);
        assert!(matches!(
            func.body[0],
            StructuredStmt::Loop {
                unroll: true,
                trip_count: Some(4),
                ..
            }
        ));
    }

    #[test]
    fn selective_inline_inlines_small_helper_and_keeps_large_helper_call() {
        // Small helper: add_one(x) -> x + 1
        let mut small_helper = IrFunction::new("add_one");
        small_helper.params.push(crate::ssa::IrParam {
            name: SmolStr::new_static("x"),
            ty: IrType::Scalar(ScalarIrTy::U32),
            is_inout: false,
            is_out: false,
        });
        small_helper.return_ty = IrType::Scalar(ScalarIrTy::U32);
        let p0 = small_helper.emit(None, InstData::Param { index: 0 });
        let one = small_helper.emit(None, InstData::ConstU32(1));
        let sum = small_helper.emit(None, InstData::IAdd32(p0, one));
        small_helper.body.push(StructuredStmt::Return(Some(sum)));

        // Large helper: > MAX_INLINE_BODY_INSTS instructions
        let mut large_helper = IrFunction::new("big_helper");
        large_helper.return_ty = IrType::Scalar(ScalarIrTy::U32);
        let mut acc = large_helper.emit(None, InstData::ConstU32(0));
        let step = large_helper.emit(None, InstData::ConstU32(1));
        for _ in 0..=MAX_INLINE_BODY_INSTS {
            acc = large_helper.emit(None, InstData::IAdd32(acc, step));
            large_helper.body.push(StructuredStmt::Eval(acc));
        }
        large_helper.body.push(StructuredStmt::Return(Some(acc)));

        // Entry caller calls both helpers
        let mut main_fn = IrFunction::new("main");
        main_fn.stage = Some(vsl_mc::ShaderStage::Compute);
        let arg = main_fn.emit(None, InstData::ConstU32(41));
        let small_call = main_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::FunctionCall {
                callee: SmolStr::new_static("add_one"),
                args: vec![arg],
            },
        );
        let big_call = main_fn.emit_typed(
            IrType::Scalar(ScalarIrTy::U32),
            InstData::FunctionCall {
                callee: SmolStr::new_static("big_helper"),
                args: Vec::new(),
            },
        );
        let total = main_fn.emit(None, InstData::IAdd32(small_call, big_call));
        main_fn.body.push(StructuredStmt::Eval(small_call));
        main_fn.body.push(StructuredStmt::Return(Some(total)));

        let mut module = IrModule {
            name: SmolStr::new_static("inline_mod"),
            functions: vec![small_helper, large_helper, main_fn],
        };

        SelectiveInlinePass.run_module(&mut module);
        ConstFoldPass.run(&mut module.functions[2]);

        let entry = &module.functions[2];
        // `add_one` call must have been inlined and folded to 42; `big_helper` call must remain.
        assert_eq!(entry.insts[small_call].data, InstData::Nop);
        assert!(matches!(
            &entry.insts[big_call].data,
            InstData::FunctionCall { callee, .. } if callee == "big_helper"
        ));
        let StructuredStmt::Return(Some(ret_id)) = entry.body[0] else {
            panic!("expected Return");
        };
        let InstData::IAdd32(lhs, rhs) = entry.insts[ret_id].data else {
            panic!("expected IAdd32");
        };
        assert_eq!(entry.insts[lhs].data, InstData::ConstU32(42));
        assert_eq!(rhs, big_call);
    }

    #[test]
    fn dce_preserves_empty_branch_if_and_cond_when_merge_phi_is_live() {
        let mut func = IrFunction::new("dce_mem2reg_phi");
        func.local_vars.push(IrLocalVar {
            name: SmolStr::new_static("x"),
            ty: IrType::Scalar(ScalarIrTy::U32),
        });
        let addr = func.emit(None, InstData::LocalAddr { local_index: 0 });
        let cond_raw = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let zero = func.emit(None, InstData::ConstU32(0));
        let cond = func.emit(None, InstData::ICmpNe(cond_raw, zero));
        let c1 = func.emit(None, InstData::ConstU32(10));
        let c2 = func.emit(None, InstData::ConstU32(20));
        let st_then = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: c1,
            },
        );
        let st_else = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: c2,
            },
        );
        let ld = func.emit(None, InstData::LocalLoad { ptr: addr });

        func.body.push(StructuredStmt::If {
            cond,
            then_body: vec![StructuredStmt::Eval(st_then)],
            else_body: vec![StructuredStmt::Eval(st_else)],
        });
        func.body.push(StructuredStmt::Return(Some(ld)));

        Mem2RegPass.run(&mut func);
        DeadCodeEliminationPass.run(&mut func);

        // `Mem2Reg` empties both branches and emits a merge `Eval(Phi)`; `DCE` must keep
        // the `If` and its `cond` alive rather than pruning the `If` or wiping `cond` to `Nop`.
        assert!(matches!(
            func.body.first(),
            Some(StructuredStmt::If { cond: if_cond, .. }) if *if_cond == cond
        ));
        assert!(matches!(func.insts[cond].data, InstData::ICmpNe(_, _)));
        assert!(func.body.iter().any(|stmt| matches!(
            stmt,
            StructuredStmt::Eval(id) if matches!(func.insts[*id].data, InstData::Phi { .. })
        )));
    }

    #[test]
    fn propagate_nonuniform_flows_through_bitand_iadd_and_select_cond() {
        let mut func = IrFunction::new("nu_arith_select");
        let raw = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let nu = func.emit(None, InstData::NonUniform(raw));
        let mask = func.emit(None, InstData::ConstU32(0xFF));
        let masked = func.emit(None, InstData::BitAnd32(nu, mask));
        let one = func.emit(None, InstData::ConstU32(1));
        let added = func.emit(None, InstData::IAdd32(masked, one));
        let zero = func.emit(None, InstData::ConstU32(0));
        let cond = func.emit(None, InstData::ICmpNe(added, zero));
        let c10 = func.emit(None, InstData::ConstU32(10));
        let c20 = func.emit(None, InstData::ConstU32(20));
        let sel = func.emit(
            None,
            InstData::Select {
                cond,
                on_true: c10,
                on_false: c20,
            },
        );
        let ptr = func.emit(
            None,
            InstData::ResourceAccessChain {
                descriptor_index: 0,
                index: sel,
            },
        );
        let load = func.emit(None, InstData::ResourceLoad { ptr });
        func.body.push(StructuredStmt::Return(Some(load)));

        PropagateNonUniformPass.run(&mut func);

        assert!(func.insts[masked].non_uniform);
        assert!(func.insts[added].non_uniform);
        assert!(func.insts[cond].non_uniform);
        assert!(func.insts[sel].non_uniform);
        assert!(func.insts[ptr].non_uniform);
        assert!(func.insts[load].non_uniform);
    }

    #[test]
    fn mem2reg_switch_break_inside_loop_merges_at_switch_not_loop() {
        let mut func = IrFunction::new("mem2reg_switch_in_loop");
        func.local_vars.push(IrLocalVar {
            name: SmolStr::new_static("acc"),
            ty: IrType::Scalar(ScalarIrTy::U32),
        });
        let addr = func.emit(None, InstData::LocalAddr { local_index: 0 });
        let init = func.emit(None, InstData::ConstU32(0));
        let st_init = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: init,
            },
        );
        let sel = func.emit(None, InstData::LoadPushConstant { field_index: 0 });
        let c1 = func.emit(None, InstData::ConstU32(1));
        let c2 = func.emit(None, InstData::ConstU32(2));
        let st_case0 = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: c1,
            },
        );
        let st_def = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: c2,
            },
        );
        let ld_after_switch = func.emit(None, InstData::LocalLoad { ptr: addr });
        let wg_store = func.emit(
            None,
            InstData::WorkgroupStore {
                var_index: 0,
                index: None,
                value: ld_after_switch,
            },
        );
        let ld_after_loop = func.emit(None, InstData::LocalLoad { ptr: addr });

        func.body.push(StructuredStmt::Eval(st_init));
        func.body.push(StructuredStmt::Loop {
            body: vec![
                StructuredStmt::Switch {
                    selector: sel,
                    cases: vec![(
                        0,
                        vec![StructuredStmt::Eval(st_case0), StructuredStmt::Break],
                    )],
                    default: vec![StructuredStmt::Eval(st_def), StructuredStmt::Break],
                },
                StructuredStmt::Eval(wg_store),
                StructuredStmt::Break,
            ],
            continuing: Vec::new(),
            unroll: false,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });
        func.body.push(StructuredStmt::Return(Some(ld_after_loop)));

        Mem2RegPass.run(&mut func);

        // `Switch` breaks must merge into a `Phi` inside the loop (feeding `wg_store`),
        // not escape directly into the enclosing `Loop`'s `break_envs`.
        let InstData::WorkgroupStore { value, .. } = func.insts[wg_store].data else {
            panic!("expected WorkgroupStore");
        };
        let InstData::Phi { incoming } = &func.insts[value].data else {
            panic!(
                "expected Switch merge Phi, got {:?}",
                func.insts[value].data
            );
        };
        assert_eq!(incoming.len(), 2);
        assert!(incoming.iter().any(|&(v, _)| v == c1));
        assert!(incoming.iter().any(|&(v, _)| v == c2));
        assert_eq!(func.body.last(), Some(&StructuredStmt::Return(Some(value))));
    }

    #[test]
    fn licm_does_not_hoist_local_load_dependent_arithmetic() {
        let mut func = IrFunction::new("licm_local_load");
        func.local_vars.push(IrLocalVar {
            name: SmolStr::new_static("x"),
            ty: IrType::Scalar(ScalarIrTy::U32),
        });
        let addr = func.emit(None, InstData::LocalAddr { local_index: 0 });
        let one = func.emit(None, InstData::ConstU32(1));
        let ld = func.emit(None, InstData::LocalLoad { ptr: addr });
        let add = func.emit(None, InstData::IAdd32(ld, one));
        let st = func.emit(
            None,
            InstData::LocalStore {
                ptr: addr,
                value: add,
            },
        );

        // `ld` is not emitted as a standalone `Eval`, only referenced inside `add`.
        func.body.push(StructuredStmt::Loop {
            body: vec![StructuredStmt::Eval(add), StructuredStmt::Eval(st)],
            continuing: Vec::new(),
            unroll: false,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });

        LicmPass.run(&mut func);

        // `add` must stay inside the loop because its operand tree contains `LocalLoad`.
        assert_eq!(func.body.len(), 1);
        let StructuredStmt::Loop { body, .. } = &func.body[0] else {
            panic!("expected Loop to remain at index 0 with no hoisted statements");
        };
        assert_eq!(
            body,
            &vec![StructuredStmt::Eval(add), StructuredStmt::Eval(st)]
        );
    }
}
