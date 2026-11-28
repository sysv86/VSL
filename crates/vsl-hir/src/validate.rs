//! Compile-time SPIR-V legality validation delegating directly to the unified HIR lowering pass.

use crate::lower::lower_unit_for_validation;
use indexmap::IndexMap;
use smol_str::SmolStr;
use vsl_diagnostics::DiagnosticBag;
use vsl_mc::{AbiProfile, HardwareCaps, HookStage, MAX_PUSH_CONSTANT_BYTES, ShaderStage};
use vsl_syntax::ast::SourceUnit;

/// Summary of validated items, entry points, hooks, and required capabilities for a [`SourceUnit`].
#[derive(Debug, Default, Clone)]
pub struct ValidationSummary {
    pub entry_points: Vec<(SmolStr, ShaderStage, Option<[u32; 3]>)>,
    pub hooks: Vec<(SmolStr, HookStage)>,
    pub push_constant_structs: IndexMap<SmolStr, u32>,
    pub required_caps: HardwareCaps,
}

/// Validates compile-time SPIR-V legality rules against a parsed [`SourceUnit`].
///
/// Delegates directly to the unified HIR lowering pipeline (without tier pruning and with
/// all device capabilities enabled so per-function capability gates and SPIR-V legality rules
/// are checked identically with zero drift).
pub fn validate_unit(unit: &SourceUnit) -> (ValidationSummary, DiagnosticBag) {
    validate_unit_with_abi(unit, AbiProfile::Hyaloclast)
}

/// Validates all tiers using the descriptor ABI selected by the caller.
pub fn validate_unit_with_abi(
    unit: &SourceUnit,
    abi: AbiProfile,
) -> (ValidationSummary, DiagnosticBag) {
    let (hir, diags) = lower_unit_for_validation(unit, abi);
    let mut summary = ValidationSummary::default();

    for (_, st) in hir.structs.iter() {
        summary.required_caps |= st.required_caps;
        if st.is_push_constant
            && let Some(ref layout) = st.computed_layout
            && layout.size <= MAX_PUSH_CONSTANT_BYTES
        {
            summary
                .push_constant_structs
                .insert(st.name.clone(), layout.size);
        }
    }

    for (_, func) in hir.functions.iter() {
        summary.required_caps |= func.required_caps;
        if let Some(stage) = func.stage {
            summary
                .entry_points
                .push((func.name.clone(), stage, func.workgroup_size));
        }
        if let Some(hook) = func.hook {
            summary.hooks.push((func.name.clone(), hook));
        }
    }

    (summary, diags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vsl_diagnostics::FileId;
    use vsl_syntax::parse_source;

    #[test]
    fn rejects_rayquery_inside_struct() {
        let src = r"
            struct Bad {
                rq: RayQuery,
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = validate_unit(&parsed.unit);
        assert!(diags.has_errors());
        assert_eq!(diags.as_slice()[0].code, "E0210");
    }

    #[test]
    fn rejects_oversized_push_constant() {
        let src = r"
            #[push_constant, layout(scalar)]
            pub struct HugePush {
                pub a: mat4,
                pub b: mat4,
                pub c: vec4,
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = validate_unit(&parsed.unit);
        assert!(diags.has_errors());
        assert_eq!(diags.as_slice()[0].code, "E0220");
    }

    #[test]
    fn enforces_float16_cap_inside_if_branches() {
        let src = r"
            pub fn bad_if(cond: bool) {
                if cond {
                    let x: f16 = 1.0f16;
                }
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = validate_unit(&parsed.unit);
        assert!(diags.has_errors());
        assert_eq!(diags.as_slice()[0].code, "E0232");
    }
}
