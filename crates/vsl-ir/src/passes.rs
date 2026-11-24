use crate::ssa::{InstData, InstId, IrFunction, IrModule, StructuredStmt};
use indexmap::{IndexMap, IndexSet};

/// Upper bound on loop iterations that [`UnrollLoopsPass`] will expand to prevent code-size blowup.
pub const MAX_UNROLL_ITERATIONS: u32 = 64;

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
                let folded = match func.insts[id].data {
                    InstData::IAdd32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x.wrapping_add(*y)))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x.wrapping_add(*y)))
                        }
                        _ => None,
                    },
                    InstData::ISub32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x.wrapping_sub(*y)))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x.wrapping_sub(*y)))
                        }
                        _ => None,
                    },
                    InstData::IMul32(a, b) => match (&func.insts[a].data, &func.insts[b].data) {
                        (InstData::ConstI32(x), InstData::ConstI32(y)) => {
                            Some(InstData::ConstI32(x.wrapping_mul(*y)))
                        }
                        (InstData::ConstU32(x), InstData::ConstU32(y)) => {
                            Some(InstData::ConstU32(x.wrapping_mul(*y)))
                        }
                        _ => None,
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
                    InstData::FAdd32(a, b) => fold_f32_binop(func, a, b, |x, y| x + y),
                    InstData::FSub32(a, b) => fold_f32_binop(func, a, b, |x, y| x - y),
                    InstData::FMul32(a, b) => fold_f32_binop(func, a, b, |x, y| x * y),
                    InstData::FDiv32(a, b) => fold_f32_binop(func, a, b, |x, y| x / y),
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
                    InstData::ConstBool(_)
                    | InstData::ConstI32(_)
                    | InstData::ConstU32(_)
                    | InstData::ConstF32(_)
                    | InstData::ConstI64(_)
                    | InstData::ConstU64(_)
                    | InstData::Phi { .. }
                    | InstData::BdaOffset { .. }
                    | InstData::ConvertPtrToU(_)
                    | InstData::ConvertUToPtr { .. }
                    | InstData::BdaLoad { .. }
                    | InstData::BdaStore { .. }
                    | InstData::ImageSample { .. }
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
                    | InstData::SubgroupBallot(_)
                    | InstData::SubgroupBallotBitCount(_)
                    | InstData::SubgroupBroadcastFirst(_)
                    | InstData::SpecConstant { .. }
                    | InstData::DescriptorLoad { .. }
                    | InstData::GlslStd450 { .. }
                    | InstData::CastToI32(_)
                    | InstData::CastToF32(_)
                    | InstData::CastToI64(_)
                    | InstData::CastToU64(_)
                    | InstData::MakeVec3(_, _, _)
                    | InstData::MakeVec4(_, _, _, _)
                    | InstData::ExtractComponent { .. }
                    | InstData::LoadBuiltin { .. }
                    | InstData::LoadPushConstant { .. }
                    | InstData::RayQueryInitialize { .. }
                    | InstData::RayQueryProceed(_)
                    | InstData::RayQueryCommittedType(_)
                    | InstData::RayQueryCommittedInstanceCustomIndex(_)
                    | InstData::RayQueryCommittedBarycentrics(_)
                    | InstData::MakeVec2(_, _)
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

fn fold_f32_cmp(
    func: &IrFunction,
    a: InstId,
    b: InstId,
    op: impl FnOnce(f32, f32) -> bool,
) -> Option<InstData> {
    if let (InstData::ConstF32(xb), InstData::ConstF32(yb)) =
        (&func.insts[a].data, &func.insts[b].data)
    {
        let x = f32::from_bits(*xb);
        let y = f32::from_bits(*yb);
        if x.is_finite() && y.is_finite() {
            return Some(InstData::ConstBool(op(x, y)));
        }
    }
    None
}

fn fold_f32_binop(
    func: &IrFunction,
    a: InstId,
    b: InstId,
    op: impl FnOnce(f32, f32) -> f32,
) -> Option<InstData> {
    if let (InstData::ConstF32(x_bits), InstData::ConstF32(y_bits)) =
        (&func.insts[a].data, &func.insts[b].data)
    {
        let x = f32::from_bits(*x_bits);
        let y = f32::from_bits(*y_bits);
        let res = op(x, y);
        (x.is_finite() && y.is_finite() && res.is_finite()).then(|| InstData::const_f32(res))
    } else {
        None
    }
}

const fn is_const_inst(data: &InstData) -> bool {
    matches!(
        data,
        InstData::ConstBool(_)
            | InstData::ConstI32(_)
            | InstData::ConstU32(_)
            | InstData::ConstF32(_)
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

fn collect_phis_in_inst_tree(
    func: &IrFunction,
    id: InstId,
    visited: &mut IndexSet<InstId>,
    out: &mut Vec<InstId>,
) {
    if !visited.insert(id) {
        return;
    }
    if matches!(func.insts[id].data, InstData::Phi { .. }) {
        out.push(id);
        return;
    }
    for op in func.insts[id].data.operands() {
        collect_phis_in_inst_tree(func, op, visited, out);
    }
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

fn phi_belongs_to_folded_if(
    func: &IrFunction,
    phi_id: InstId,
    branch_evals: &IndexSet<InstId>,
    prior_non_folded_evals: &IndexSet<InstId>,
) -> bool {
    let InstData::Phi { incoming } = &func.insts[phi_id].data else {
        return false;
    };
    let from_prior = incoming
        .iter()
        .any(|&(v, _)| prior_non_folded_evals.contains(&v));
    let from_this = incoming.iter().any(|&(v, _)| branch_evals.contains(&v));
    !from_prior || from_this
}

fn fold_const_ifs(
    func: &IrFunction,
    stmts: Vec<StructuredStmt>,
    folded_any: &mut bool,
    phi_remap: &mut IndexMap<InstId, InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut prior_non_folded_evals: IndexSet<InstId> = IndexSet::new();
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
                    let mut branch_evals = IndexSet::new();
                    collect_eval_ids(&then_body, &mut branch_evals);
                    collect_eval_ids(&else_body, &mut branch_evals);
                    let taken = if flag { then_body } else { else_body };
                    out.extend(fold_const_ifs(func, taken, folded_any, phi_remap));

                    // Resolve standalone merge-block `Eval(Phi)` statements immediately after the
                    // folded `If`.
                    while let Some(StructuredStmt::Eval(phi_id)) = iter.as_slice().first() {
                        let phi_id = *phi_id;
                        if let Some(selected) = phi_selected_for_const_if(func, phi_id, taken_tag)
                            && phi_belongs_to_folded_if(
                                func,
                                phi_id,
                                &branch_evals,
                                &prior_non_folded_evals,
                            )
                        {
                            let _ = iter.next();
                            phi_remap.insert(phi_id, selected);
                        } else {
                            break;
                        }
                    }

                    // Also scan subsequent statements' operand trees before the next control-flow
                    // node for merge `Phi`s referenced indirectly without a standalone `Eval`.
                    let mut visited = IndexSet::new();
                    let mut immediate_phis = Vec::new();
                    for rem_stmt in iter.as_slice() {
                        match rem_stmt {
                            StructuredStmt::Eval(id) | StructuredStmt::Return(Some(id)) => {
                                collect_phis_in_inst_tree(
                                    func,
                                    *id,
                                    &mut visited,
                                    &mut immediate_phis,
                                );
                            }
                            StructuredStmt::If {
                                cond: next_cond, ..
                            } => {
                                collect_phis_in_inst_tree(
                                    func,
                                    *next_cond,
                                    &mut visited,
                                    &mut immediate_phis,
                                );
                                break;
                            }
                            StructuredStmt::Switch { selector, .. } => {
                                collect_phis_in_inst_tree(
                                    func,
                                    *selector,
                                    &mut visited,
                                    &mut immediate_phis,
                                );
                                break;
                            }
                            StructuredStmt::Loop { induction_init, .. } => {
                                if let Some(ind_id) = induction_init {
                                    collect_phis_in_inst_tree(
                                        func,
                                        *ind_id,
                                        &mut visited,
                                        &mut immediate_phis,
                                    );
                                }
                                break;
                            }
                            StructuredStmt::Break
                            | StructuredStmt::Continue
                            | StructuredStmt::Return(None) => break,
                        }
                    }
                    for phi_id in immediate_phis {
                        if !phi_remap.contains_key(&phi_id)
                            && let Some(selected) =
                                phi_selected_for_const_if(func, phi_id, taken_tag)
                            && phi_belongs_to_folded_if(
                                func,
                                phi_id,
                                &branch_evals,
                                &prior_non_folded_evals,
                            )
                        {
                            phi_remap.insert(phi_id, selected);
                        }
                    }

                    // Resolve any `Phi` later in the block whose incoming operands were explicitly
                    // evaluated inside the folded `then_body`/`else_body`.
                    if !branch_evals.is_empty() {
                        let mut later_refs = IndexSet::new();
                        collect_stmt_transitive_refs(func, iter.as_slice(), &mut later_refs);
                        for phi_id in later_refs {
                            if !phi_remap.contains_key(&phi_id)
                                && let Some(selected) =
                                    phi_selected_for_const_if(func, phi_id, taken_tag)
                                && let InstData::Phi { incoming } = &func.insts[phi_id].data
                                && incoming.iter().any(|&(v, _)| branch_evals.contains(&v))
                            {
                                phi_remap.insert(phi_id, selected);
                            }
                        }
                    }
                } else {
                    collect_eval_ids(&then_body, &mut prior_non_folded_evals);
                    collect_eval_ids(&else_body, &mut prior_non_folded_evals);
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
                for (_, case_body) in &cases {
                    collect_eval_ids(case_body, &mut prior_non_folded_evals);
                }
                collect_eval_ids(&default, &mut prior_non_folded_evals);
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
                collect_eval_ids(&body, &mut prior_non_folded_evals);
                collect_eval_ids(&continuing, &mut prior_non_folded_evals);
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
        func.body = unroll_stmts(func, body, &ancestor_outside);
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

    let mut defined_in_loop: IndexSet<InstId> = IndexSet::new();
    for id in postorder {
        if Some(id) == induction_init {
            continue;
        }
        if direct_evals.contains(&id) {
            defined_in_loop.insert(id);
            continue;
        }
        let data = &func.insts[id].data;
        let is_external_sensitive = matches!(
            data,
            InstData::BdaLoad { .. }
                | InstData::ResourceLoad { .. }
                | InstData::WorkgroupLoad { .. }
                | InstData::ImageSample { .. }
                | InstData::Phi { .. }
        );
        if is_external_sensitive {
            // An external load must keep its single SSA definition even when its address
            // operand is loop-derived. Only explicit `Eval` inside the loop clones it.
            continue;
        }
        let depends_on_loop = data
            .operands()
            .into_iter()
            .any(|op| Some(op) == induction_init || defined_in_loop.contains(&op));
        if depends_on_loop {
            defined_in_loop.insert(id);
            continue;
        }
        if !outside_refs.contains(&id) && !func.insts[id].has_side_effects() {
            defined_in_loop.insert(id);
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
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut post_remap: IndexMap<InstId, InstId> = IndexMap::new();
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
                    then_body: unroll_stmts(func, then_body, &then_outside),
                    else_body: unroll_stmts(func, else_body, &else_outside),
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
                    .map(|((k, b), outside)| (k, unroll_stmts(func, b, &outside)))
                    .collect();
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: new_cases,
                    default: unroll_stmts(func, default, &default_outside),
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

                let body = unroll_stmts(func, body, &body_outside);
                let continuing = unroll_stmts(func, continuing, &cont_outside);
                if unroll
                    && let Some(trips) = trip_count
                    && trips <= MAX_UNROLL_ITERATIONS
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
                                    &mut remap,
                                );
                                out.extend(cloned);
                            }
                        }
                    } else {
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
        out = remap_stmts_ids(out, &post_remap);
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
        InstData::ConvertPtrToU(_) | InstData::CastI64ToU64(_) => {
            let delta_const = func.emit(None, InstData::ConstU64(delta.cast_unsigned()));
            func.emit(ty, InstData::IAdd64(ind_id, delta_const))
        }
        InstData::IAdd64(_, _) | InstData::IMul64(_, _) => {
            let delta_const = func.emit(None, InstData::ConstI64(delta));
            func.emit(ty, InstData::IAdd64(ind_id, delta_const))
        }
        _ => {
            let delta_const = func.emit(None, InstData::ConstI32(delta as i32));
            func.emit(ty, InstData::IAdd32(ind_id, delta_const))
        }
    }
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
    for op in original.data.operands() {
        let _ = clone_inst_in_loop(func, op, defined_in_loop, remap);
    }
    let remapped_data = remap_inst_data(&original.data, remap);
    let new_id = func.emit(original.ty, remapped_data);
    func.insts[new_id].non_uniform = original.non_uniform;
    remap.insert(id, new_id);
    new_id
}

/// Clones structured statements for an unrolled loop iteration, allocating fresh SSA
/// instruction IDs only for instructions in `defined_in_loop`.
fn clone_stmts_with_remap(
    func: &mut IrFunction,
    stmts: &[StructuredStmt],
    defined_in_loop: &IndexSet<InstId>,
    remap: &mut IndexMap<InstId, InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
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
                    then_body: clone_stmts_with_remap(func, then_body, defined_in_loop, remap),
                    else_body: clone_stmts_with_remap(func, else_body, defined_in_loop, remap),
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
                        .map(|(k, b)| (*k, clone_stmts_with_remap(func, b, defined_in_loop, remap)))
                        .collect(),
                    default: clone_stmts_with_remap(func, default, defined_in_loop, remap),
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
                    body: clone_stmts_with_remap(func, body, defined_in_loop, remap),
                    continuing: clone_stmts_with_remap(func, continuing, defined_in_loop, remap),
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
        InstData::ExtractComponent { vec, index } => InstData::ExtractComponent {
            vec: remap_id(*vec, remap),
            index: *index,
        },
        InstData::Nop => InstData::Nop,
    }
}

/// Lowers reachable `BdaOffset` instructions (standalone statements and transitive operands
/// reachable from `func.body`) into explicit unsigned 64-bit integer arithmetic
/// (`OpConvertPtrToU` + `CastI64ToU64` + `OpIMul` + `OpIAdd` + `OpConvertUToPtr`) so VSL 2.0
/// never emits `OpPtrAccessChain` (avoiding NVIDIA's +4 GB negative-offset driver bug).
pub struct LowerBdaOffsetsPass;

impl IrPass for LowerBdaOffsetsPass {
    fn name(&self) -> &'static str {
        "lower-bda-offsets"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut lowered_any = false;
        let mut visited: IndexSet<InstId> = IndexSet::new();
        let body = std::mem::take(&mut func.body);
        func.body = rewrite_stmts_bda(func, body, &mut visited, &mut lowered_any);

        if lowered_any {
            func.uses_bda = true;
        }
    }
}

fn lower_bda_in_inst(
    func: &mut IrFunction,
    id: InstId,
    is_root_eval: bool,
    visited: &mut IndexSet<InstId>,
    lowered_any: &mut bool,
    prefix_evals: &mut Vec<StructuredStmt>,
) {
    if !visited.insert(id) {
        return;
    }

    let operands = func.insts[id].data.operands();
    for op in operands {
        lower_bda_in_inst(func, op, false, visited, lowered_any, prefix_evals);
    }

    if let InstData::BdaOffset {
        base_ptr,
        index_i64,
        stride_bytes,
        align,
    } = func.insts[id].data
    {
        let ty = func.insts[id].ty;
        let base_u64 = func.emit(None, InstData::ConvertPtrToU(base_ptr));
        let index_u64 = func.emit(None, InstData::CastI64ToU64(index_i64));
        let stride_const = func.emit(None, InstData::ConstU64(stride_bytes));
        let byte_off = func.emit(None, InstData::IMul64(index_u64, stride_const));
        let target_u64 = func.emit(None, InstData::IAdd64(base_u64, byte_off));
        func.insts[id].data = InstData::ConvertUToPtr {
            addr_u64: target_u64,
            align,
        };
        func.insts[id].ty = ty;

        prefix_evals.push(StructuredStmt::Eval(base_u64));
        prefix_evals.push(StructuredStmt::Eval(index_u64));
        prefix_evals.push(StructuredStmt::Eval(stride_const));
        prefix_evals.push(StructuredStmt::Eval(byte_off));
        prefix_evals.push(StructuredStmt::Eval(target_u64));
        if !is_root_eval {
            prefix_evals.push(StructuredStmt::Eval(id));
        }
        *lowered_any = true;
    }
}

fn rewrite_stmts_bda(
    func: &mut IrFunction,
    stmts: Vec<StructuredStmt>,
    visited: &mut IndexSet<InstId>,
    lowered_any: &mut bool,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                lower_bda_in_inst(func, id, true, visited, lowered_any, &mut out);
                out.push(StructuredStmt::Eval(id));
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                lower_bda_in_inst(func, cond, false, visited, lowered_any, &mut out);
                out.push(StructuredStmt::If {
                    cond,
                    then_body: rewrite_stmts_bda(func, then_body, visited, lowered_any),
                    else_body: rewrite_stmts_bda(func, else_body, visited, lowered_any),
                });
            }
            StructuredStmt::Switch {
                selector,
                cases,
                default,
            } => {
                lower_bda_in_inst(func, selector, false, visited, lowered_any, &mut out);
                out.push(StructuredStmt::Switch {
                    selector,
                    cases: cases
                        .into_iter()
                        .map(|(k, b)| (k, rewrite_stmts_bda(func, b, visited, lowered_any)))
                        .collect(),
                    default: rewrite_stmts_bda(func, default, visited, lowered_any),
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
                    lower_bda_in_inst(func, ind_id, false, visited, lowered_any, &mut out);
                }
                out.push(StructuredStmt::Loop {
                    body: rewrite_stmts_bda(func, body, visited, lowered_any),
                    continuing: rewrite_stmts_bda(func, continuing, visited, lowered_any),
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step,
                });
            }
            StructuredStmt::Return(opt_val) => {
                if let Some(val_id) = opt_val {
                    lower_bda_in_inst(func, val_id, false, visited, lowered_any, &mut out);
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

impl IrPass for PropagateNonUniformPass {
    fn name(&self) -> &'static str {
        "propagate-nonuniform"
    }

    fn run(&self, func: &mut IrFunction) {
        let mut changed = true;
        while changed {
            changed = false;
            let ids: Vec<InstId> = func.insts.iter().map(|(id, _)| id).collect();
            for id in ids {
                if func.insts[id].non_uniform {
                    continue;
                }
                let tainted = match &func.insts[id].data {
                    InstData::NonUniform(_) => true,
                    InstData::CastToU32(v)
                    | InstData::CastToI32(v)
                    | InstData::CastToF32(v)
                    | InstData::CastToI64(v)
                    | InstData::CastToU64(v)
                    | InstData::ExtractComponent { vec: v, .. } => func.insts[*v].non_uniform,
                    InstData::MakeVec3(a, b, c) => {
                        func.insts[*a].non_uniform
                            || func.insts[*b].non_uniform
                            || func.insts[*c].non_uniform
                    }
                    InstData::MakeVec4(a, b, c, d) => {
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
                    InstData::ImageSample { image_inst, .. } => {
                        image_inst.is_some_and(|img| func.insts[img].non_uniform)
                    }
                    InstData::Select {
                        on_true, on_false, ..
                    } => func.insts[*on_true].non_uniform || func.insts[*on_false].non_uniform,
                    InstData::Phi { incoming } => {
                        incoming.iter().any(|&(v, _)| func.insts[v].non_uniform)
                    }
                    InstData::MakeVec2(a, b) => {
                        func.insts[*a].non_uniform || func.insts[*b].non_uniform
                    }
                    InstData::ConstBool(_)
                    | InstData::ConstI32(_)
                    | InstData::ConstU32(_)
                    | InstData::ConstF32(_)
                    | InstData::ConstI64(_)
                    | InstData::ConstU64(_)
                    | InstData::IAdd32(_, _)
                    | InstData::ISub32(_, _)
                    | InstData::IMul32(_, _)
                    | InstData::IDiv32(_, _)
                    | InstData::IRem32(_, _)
                    | InstData::FAdd32(_, _)
                    | InstData::FSub32(_, _)
                    | InstData::FMul32(_, _)
                    | InstData::FDiv32(_, _)
                    | InstData::ICmpEq(_, _)
                    | InstData::ICmpNe(_, _)
                    | InstData::ICmpLt(_, _)
                    | InstData::ICmpLe(_, _)
                    | InstData::ICmpGt(_, _)
                    | InstData::ICmpGe(_, _)
                    | InstData::UCmpLt(_, _)
                    | InstData::FCmpEq(_, _)
                    | InstData::FCmpNe(_, _)
                    | InstData::FCmpLt(_, _)
                    | InstData::FCmpLe(_, _)
                    | InstData::FCmpGt(_, _)
                    | InstData::FCmpGe(_, _)
                    | InstData::LogicalAnd(_, _)
                    | InstData::LogicalOr(_, _)
                    | InstData::LogicalNot(_)
                    | InstData::BitAnd32(_, _)
                    | InstData::BitOr32(_, _)
                    | InstData::BitXor32(_, _)
                    | InstData::BitNot32(_)
                    | InstData::Shl32(_, _)
                    | InstData::Shr32(_, _)
                    | InstData::CastI64ToU64(_)
                    | InstData::BdaOffset { .. }
                    | InstData::ConvertPtrToU(_)
                    | InstData::ConvertUToPtr { .. }
                    | InstData::IMul64(_, _)
                    | InstData::IAdd64(_, _)
                    | InstData::BdaLoad { .. }
                    | InstData::BdaStore { .. }
                    | InstData::ImageStore { .. }
                    | InstData::WorkgroupAddr { .. }
                    | InstData::WorkgroupLoad { .. }
                    | InstData::WorkgroupStore { .. }
                    | InstData::AtomicAdd { .. }
                    | InstData::AtomicMin { .. }
                    | InstData::AtomicMax { .. }
                    | InstData::AtomicCas { .. }
                    | InstData::WorkgroupBarrier
                    | InstData::SubgroupBallot(_)
                    | InstData::SubgroupBallotBitCount(_)
                    | InstData::SubgroupBroadcastFirst(_)
                    | InstData::SpecConstant { .. }
                    | InstData::DescriptorLoad { .. }
                    | InstData::LoadBuiltin { .. }
                    | InstData::LoadPushConstant { .. }
                    | InstData::RayQueryInitialize { .. }
                    | InstData::RayQueryProceed(_)
                    | InstData::RayQueryCommittedType(_)
                    | InstData::RayQueryCommittedInstanceCustomIndex(_)
                    | InstData::RayQueryCommittedBarycentrics(_)
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
    ConvertUToPtr(InstId, u32),
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
        let mut gvn_remap: IndexMap<InstId, InstId> = IndexMap::new();
        let mut visited: IndexSet<InstId> = IndexSet::new();
        let body = std::mem::take(&mut func.body);
        let new_body = gvn_stmts(func, body, &mut scopes, &mut gvn_remap, &mut visited);
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
            if is_one_32(func, *a) {
                Some(*b)
            } else if is_one_32(func, *b) || is_zero_32(func, *a) {
                Some(*a)
            } else if is_zero_32(func, *b) {
                Some(*b)
            } else {
                None
            }
        }
        InstData::IDiv32(a, b) if is_one_32(func, *b) => Some(*a),
        InstData::BitAnd32(a, b) => {
            if a == b || is_zero_32(func, *a) {
                Some(*a)
            } else if is_zero_32(func, *b) {
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
            if is_one_64(func, *a) {
                Some(*b)
            } else if is_one_64(func, *b) || is_zero_64(func, *a) {
                Some(*a)
            } else if is_zero_64(func, *b) {
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
            if let InstData::ConvertPtrToU(orig_ptr) = func.insts[*addr_u64].data {
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

fn gvn_key_for(data: &InstData) -> Option<GvnKey> {
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
        InstData::ConvertUToPtr { addr_u64, align } => Some(GvnKey::ConvertUToPtr(addr_u64, align)),
        InstData::LoadBuiltin { kind, component } => Some(GvnKey::LoadBuiltin(kind, component)),
        InstData::LoadPushConstant { field_index } => Some(GvnKey::LoadPushConstant(field_index)),
        InstData::MakeVec2(a, b) => Some(GvnKey::MakeVec2(a, b)),
        _ => None,
    }
}

fn lookup_gvn(scopes: &[IndexMap<GvnKey, InstId>], key: &GvnKey) -> Option<InstId> {
    for scope in scopes.iter().rev() {
        if let Some(&id) = scope.get(key) {
            return Some(id);
        }
    }
    None
}

fn gvn_visit_inst(
    func: &mut IrFunction,
    id: InstId,
    scopes: &mut [IndexMap<GvnKey, InstId>],
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
        let _ = gvn_visit_inst(func, op, scopes, remap, visited);
    }
    let remapped = remap_inst_data(&func.insts[id].data, remap);
    func.insts[id].data = remapped;

    // Never deduplicate NonUniform-tainted nodes with uniform nodes.
    if func.insts[id].non_uniform {
        return id;
    }
    if let Some(key) = gvn_key_for(&func.insts[id].data) {
        if let Some(leader) = lookup_gvn(scopes, &key) {
            remap.insert(id, leader);
            return leader;
        }
        if let Some(current) = scopes.last_mut() {
            current.insert(key, id);
        }
    }
    id
}

fn gvn_stmts(
    func: &mut IrFunction,
    stmts: Vec<StructuredStmt>,
    scopes: &mut Vec<IndexMap<GvnKey, InstId>>,
    remap: &mut IndexMap<InstId, InstId>,
    visited: &mut IndexSet<InstId>,
) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                let leader = gvn_visit_inst(func, id, scopes, remap, visited);
                out.push(StructuredStmt::Eval(leader));
            }
            StructuredStmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond = gvn_visit_inst(func, cond, scopes, remap, visited);
                scopes.push(IndexMap::new());
                let then_body = gvn_stmts(func, then_body, scopes, remap, visited);
                scopes.pop();
                scopes.push(IndexMap::new());
                let else_body = gvn_stmts(func, else_body, scopes, remap, visited);
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
                let selector = gvn_visit_inst(func, selector, scopes, remap, visited);
                let mut new_cases = Vec::with_capacity(cases.len());
                for (k, body) in cases {
                    scopes.push(IndexMap::new());
                    let b = gvn_stmts(func, body, scopes, remap, visited);
                    scopes.pop();
                    new_cases.push((k, b));
                }
                scopes.push(IndexMap::new());
                let default = gvn_stmts(func, default, scopes, remap, visited);
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
                let induction_init =
                    induction_init.map(|id| gvn_visit_inst(func, id, scopes, remap, visited));
                scopes.push(IndexMap::new());
                let body = gvn_stmts(func, body, scopes, remap, visited);
                scopes.pop();
                // Scope `continuing` separately so instructions emitted after a conditional
                // `Continue` in `body` never leak into `continuing`'s GVN table.
                scopes.push(IndexMap::new());
                let continuing = gvn_stmts(func, continuing, scopes, remap, visited);
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
                let opt = opt.map(|id| gvn_visit_inst(func, id, scopes, remap, visited));
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

        let _ = collect_roots(&func.body, func, &mut live, &mut worklist);

        while let Some(id) = worklist.pop() {
            for operand in func.insts[id].data.operands() {
                if live.insert(operand) {
                    worklist.push(operand);
                }
            }
        }

        let body = std::mem::take(&mut func.body);
        func.body = prune_stmts(body, &live);

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
                    | InstData::SubgroupBroadcastFirst(_)
            )
        });
    }
}

fn collect_roots(
    stmts: &[StructuredStmt],
    func: &IrFunction,
    live: &mut IndexSet<InstId>,
    worklist: &mut Vec<InstId>,
) -> bool {
    let mut has_live = false;
    for stmt in stmts {
        match stmt {
            StructuredStmt::Eval(id) => {
                if func.insts[*id].has_side_effects() {
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
                if then_live || else_live {
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
                if any_live {
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

fn prune_stmts(stmts: Vec<StructuredStmt>, live: &IndexSet<InstId>) -> Vec<StructuredStmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
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
                let then_body = prune_stmts(then_body, live);
                let else_body = prune_stmts(else_body, live);
                if !then_body.is_empty() || !else_body.is_empty() {
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
                    .map(|(k, b)| (k, prune_stmts(b, live)))
                    .collect();
                let default = prune_stmts(default, live);
                if cases.iter().any(|(_, b)| !b.is_empty()) || !default.is_empty() {
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
                body: prune_stmts(body, live),
                continuing: prune_stmts(continuing, live),
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

/// Executes the mandatory VSL 2.0 Rust SSA legalization pipeline (zero runtime `spirv-opt`).
pub struct PassPipeline {
    passes: Vec<Box<dyn IrPass>>,
}

impl Default for PassPipeline {
    fn default() -> Self {
        Self {
            passes: vec![
                Box::new(UnrollLoopsPass),
                Box::new(LowerBdaOffsetsPass),
                Box::new(ConstFoldPass),
                Box::new(SimplifyAndGvnPass),
                Box::new(PropagateNonUniformPass),
                Box::new(DeadCodeEliminationPass),
            ],
        }
    }
}

impl PassPipeline {
    pub fn run_module(&self, module: &mut IrModule) {
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
}
