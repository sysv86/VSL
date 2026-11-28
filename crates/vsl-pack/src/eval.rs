use crate::ir::{
    BufferDecl, HookRegistration, PackMetadata, PassBindingRef, PassDispatch, PassKind, PassMeta,
    PassNode, PassRef, PipelineIr, PipelineMode, PresetIr, ResourceTextureDecl, SettingIr,
    SettingValueIr, TargetDecl,
};
use crate::schedule::is_write_binding_key;
use indexmap::IndexMap;
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, DiagnosticBag, Span};
use vsl_mc::{
    BuiltinResourceId, CanonicalPass, HookStage, ImageFormat, PACK_RESOURCE_SLOTS, RenderTier,
};
use vsl_syntax::ast::{
    BinaryOp, Block, CallArg, Expr, Item, PackDecl, PipelineDecl, PipelineModeAst, PresetsDecl,
    SettingKind, SettingTypeAst, SettingsDecl, SourceUnit, Stmt, TypeArg, TypeExpr, UnaryOp,
};

/// Hard execution ceiling for deterministic `pack.vsl` evaluation (Spec §3.3).
pub const MAX_EVAL_STEPS: usize = 100_000;

/// Maximum iterations permitted for any loop inside `pack.vsl` (Spec §3.3).
pub const MAX_LOOP_ITERS: usize = 256;

/// Active runtime configuration passed to `pipeline extend/custom(g, cfg)`.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub tier: RenderTier,
    pub param_overrides: IndexMap<SmolStr, SettingValueIr>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            tier: RenderTier::Tier2RayQuery,
            param_overrides: IndexMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum HandleValue {
    GraphBuilder,
    Config,
    Target { name: SmolStr, temporal: bool },
    TargetWrite { name: SmolStr },
    TargetHistory { name: SmolStr },
    Buffer { name: SmolStr },
    BufferWrite { name: SmolStr },
    BuiltinTarget(BuiltinResourceId),
    ResourceTexture { slot: u32 },
    PassIndex { idx: usize, is_mip: bool },
    Bool(bool),
    Int(i64),
    Float(f32),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    EnumVariant(SmolStr),
    Unit,
}

#[derive(Debug, Clone)]
enum ControlFlow {
    Normal(HandleValue),
    Break,
    Continue,
    Return,
}

/// Evaluates a parsed `pack.vsl` [`SourceUnit`] inside a deterministic zero-IO sandbox.
pub fn evaluate_pack_vsl(unit: &SourceUnit, cfg: &EvalConfig) -> (PipelineIr, DiagnosticBag) {
    let mut ir = PipelineIr::default();
    let mut diags = DiagnosticBag::new();

    // Pass 1a: collect pack metadata and settings first so preset validation is order-independent.
    for item in &unit.items {
        match item {
            Item::Pack(decl) => eval_pack_decl(decl, &mut ir.metadata, &mut diags),
            Item::Settings(decl) => eval_settings_decl(decl, &mut ir.settings, &mut diags),
            _ => {}
        }
    }

    // Pass 1b: collect presets and validate every assigned key against declared settings.
    for item in &unit.items {
        if let Item::Presets(decl) = item {
            eval_presets_decl(decl, &ir.settings, &mut ir.presets, &mut diags);
        }
    }

    if cfg.tier.as_u8() < ir.metadata.tier_min || cfg.tier.as_u8() > ir.metadata.tier_max {
        diags.push(Diagnostic::error(
            "E0423",
            format!(
                "pack `{}` supports tiers {}..={}, but tier {} was requested",
                ir.metadata.name,
                ir.metadata.tier_min,
                ir.metadata.tier_max,
                cfg.tier.as_u8()
            ),
            Span::synthetic(),
        ));
        return (ir, diags);
    }

    // Pass 2: evaluate `pipeline extend/custom(...)`.
    for item in &unit.items {
        if let Item::Pipeline(decl) = item {
            ir.mode = match decl.mode {
                PipelineModeAst::Extend => PipelineMode::Extend,
                PipelineModeAst::Custom => PipelineMode::Custom,
            };
            let mut sandbox = Sandbox {
                cfg,
                ir: &mut ir,
                diags: &mut diags,
                env: IndexMap::new(),
                steps: 0,
                aborted: false,
            };
            sandbox.eval_pipeline(decl);
        }
    }

    (ir, diags)
}

fn eval_pack_decl(decl: &PackDecl, meta: &mut PackMetadata, diags: &mut DiagnosticBag) {
    meta.id = decl.id.clone();
    meta.tier_min = 0;
    meta.tier_max = 3;

    for field in &decl.fields {
        match field.key.name.as_str() {
            "name" => {
                if let Expr::StringLit(s, _) = &field.value {
                    meta.name = s.clone();
                }
            }
            "version" => {
                if let Expr::StringLit(s, _) = &field.value {
                    meta.version = s.clone();
                }
            }
            "author" => {
                if let Expr::StringLit(s, _) = &field.value {
                    meta.author = s.clone();
                }
            }
            "description" => {
                if let Expr::StringLit(s, _) = &field.value {
                    meta.description = s.clone();
                }
            }
            "extends" => match &field.value {
                Expr::NoneLit(_) => meta.extends = None,
                Expr::StringLit(s, _) => meta.extends = Some(s.clone()),
                _ => diags.push(Diagnostic::error(
                    "E0401",
                    "`extends` must be `none` or a pack ID string literal",
                    field.span,
                )),
            },
            "tiers" => {
                if let Some((lo, hi)) = eval_tier_range(&field.value, diags) {
                    meta.tier_min = lo;
                    meta.tier_max = hi;
                }
            }
            "capabilities" => {
                if let Expr::ArrayLit { elements, .. } = &field.value {
                    for el in elements {
                        if let Expr::Path { path, .. } = el {
                            meta.capabilities.push(SmolStr::new(path.joined()));
                        }
                    }
                }
            }
            other => {
                diags.push(Diagnostic::error(
                    "E0419",
                    format!("unknown field `{other}` in `pack` declaration"),
                    field.span,
                ));
            }
        }
    }
}

fn eval_settings_decl(
    decl: &SettingsDecl,
    out: &mut IndexMap<SmolStr, SettingIr>,
    diags: &mut DiagnosticBag,
) {
    for category in &decl.categories {
        for entry in &category.entries {
            let mut label = entry.name.name.clone();
            let mut description = SmolStr::new_static("");
            let mut range = None;
            let mut step = None;
            let mut tiers = (0u8, 3u8);

            for field in &entry.metadata {
                match field.key.name.as_str() {
                    "label" => {
                        if let Expr::StringLit(s, _) = &field.value {
                            label = s.clone();
                        }
                    }
                    "description" => {
                        if let Expr::StringLit(s, _) = &field.value {
                            description = s.clone();
                        }
                    }
                    "range" => {
                        range = eval_numeric_range(&field.value, diags);
                    }
                    "step" => {
                        step = eval_const_f64(&field.value);
                    }
                    "tiers" => {
                        if let Some(t) = eval_tier_range(&field.value, diags) {
                            tiers = t;
                        }
                    }
                    _ => {}
                }
            }

            let enum_variants = match &entry.ty {
                SettingTypeAst::Enum { variants, .. } => {
                    variants.iter().map(|v| v.name.clone()).collect()
                }
                SettingTypeAst::Named(_) => Vec::new(),
            };

            let parsed_default = eval_setting_value(&entry.default_value);
            let default_value = if let Some(raw_val) = parsed_default {
                let val = widen_setting_value_for_type(&entry.ty, raw_val);
                validate_setting_type_and_range(
                    &entry.name.name,
                    &entry.ty,
                    &val,
                    range,
                    entry.default_value.span(),
                    diags,
                );
                val
            } else {
                diags.push(Diagnostic::error(
                    "E0420",
                    format!(
                        "setting `{}` has an invalid default value expression",
                        entry.name.name
                    ),
                    entry.default_value.span(),
                ));
                SettingValueIr::Int(0)
            };

            out.insert(
                entry.name.name.clone(),
                SettingIr {
                    name: entry.name.name.clone(),
                    category: category.name.clone(),
                    is_spec: matches!(entry.kind, SettingKind::Spec),
                    label,
                    description,
                    default_value,
                    range,
                    step,
                    tiers,
                    enum_variants,
                },
            );
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn widen_setting_value_for_type(ty: &SettingTypeAst, val: SettingValueIr) -> SettingValueIr {
    if let SettingTypeAst::Named(ident) = ty
        && matches!(ident.name.as_str(), "f32" | "f64" | "float")
        && let SettingValueIr::Int(i) = val
    {
        return SettingValueIr::Float(i as f32);
    }
    val
}

#[allow(clippy::cast_precision_loss)]
fn validate_setting_type_and_range(
    name: &str,
    ty: &SettingTypeAst,
    val: &SettingValueIr,
    range: Option<(f64, f64)>,
    span: Span,
    diags: &mut DiagnosticBag,
) {
    let type_ok = match ty {
        SettingTypeAst::Named(ident) => match ident.name.as_str() {
            "bool" => matches!(val, SettingValueIr::Bool(_)),
            "i32" | "i64" | "u32" | "u64" | "int" => matches!(val, SettingValueIr::Int(_)),
            "f32" | "f64" | "float" => {
                matches!(val, SettingValueIr::Float(_) | SettingValueIr::Int(_))
            }
            "color_rgb" | "vec3" => matches!(val, SettingValueIr::ColorRgb(_)),
            _ => true,
        },
        SettingTypeAst::Enum { variants, .. } => match val {
            SettingValueIr::EnumVariant(v) => variants.iter().any(|var| var.name == *v),
            _ => false,
        },
    };

    if !type_ok {
        diags.push(Diagnostic::error(
            "E0420",
            format!("setting `{name}` default value does not match its declared type"),
            span,
        ));
        return;
    }

    if let Some((lo, hi)) = range {
        let numeric = match val {
            SettingValueIr::Int(i) => Some(*i as f64),
            SettingValueIr::Float(f) => Some(f64::from(*f)),
            _ => None,
        };
        if let Some(num) = numeric
            && !(lo..=hi).contains(&num)
        {
            diags.push(Diagnostic::error(
                "E0420",
                format!(
                    "setting `{name}` default value {num} is outside declared range {lo}..={hi}"
                ),
                span,
            ));
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn validate_preset_override(
    preset_name: &str,
    setting: &SettingIr,
    val: &SettingValueIr,
    span: Span,
    diags: &mut DiagnosticBag,
) {
    let type_ok = match (&setting.default_value, val) {
        (SettingValueIr::Bool(_), SettingValueIr::Bool(_))
        | (SettingValueIr::Int(_), SettingValueIr::Int(_))
        | (SettingValueIr::Float(_), SettingValueIr::Float(_) | SettingValueIr::Int(_))
        | (SettingValueIr::ColorRgb(_), SettingValueIr::ColorRgb(_)) => true,
        (SettingValueIr::EnumVariant(_), SettingValueIr::EnumVariant(v)) => {
            setting.enum_variants.is_empty() || setting.enum_variants.contains(v)
        }
        _ => false,
    };

    if !type_ok {
        diags.push(Diagnostic::error(
            "E0420",
            format!(
                "preset `{preset_name}` assigns a value of mismatched type to setting `{}`",
                setting.name
            ),
            span,
        ));
        return;
    }

    if let Some((lo, hi)) = setting.range {
        let numeric = match val {
            SettingValueIr::Int(i) => Some(*i as f64),
            SettingValueIr::Float(f) => Some(f64::from(*f)),
            _ => None,
        };
        if let Some(num) = numeric
            && !(lo..=hi).contains(&num)
        {
            diags.push(Diagnostic::error(
                "E0420",
                format!(
                    "preset `{preset_name}` assigns out-of-range value {num} to setting `{}` (expected {lo}..={hi})",
                    setting.name
                ),
                span,
            ));
        }
    }
}

fn eval_presets_decl(
    decl: &PresetsDecl,
    settings: &IndexMap<SmolStr, SettingIr>,
    out: &mut Vec<PresetIr>,
    diags: &mut DiagnosticBag,
) {
    for preset in &decl.presets {
        let mut label = preset.name.name.clone();
        let mut description = SmolStr::new_static("");
        let mut order = i32::try_from(out.len()).unwrap_or(i32::MAX);
        let mut tiers = (0u8, 3u8);
        for field in &preset.metadata {
            match field.key.name.as_str() {
                "label" => {
                    if let Expr::StringLit(s, _) = &field.value {
                        label = s.clone();
                    }
                }
                "description" => {
                    if let Expr::StringLit(s, _) = &field.value {
                        description = s.clone();
                    }
                }
                "order" => {
                    if let Some(v) = eval_const_i64(&field.value) {
                        order = i32::try_from(v).unwrap_or(order);
                    }
                }
                "tiers" => {
                    if let Some(t) = eval_tier_range(&field.value, diags) {
                        tiers = t;
                    }
                }
                _ => {}
            }
        }
        let mut overrides = IndexMap::new();
        for (k, v) in &preset.assignments {
            let mut parsed_val = eval_setting_value(v);
            if let Some(setting) = settings.get(&k.name) {
                if let Some(ref mut val) = parsed_val {
                    #[allow(clippy::cast_precision_loss)]
                    if matches!(setting.default_value, SettingValueIr::Float(_))
                        && let SettingValueIr::Int(i) = *val
                    {
                        *val = SettingValueIr::Float(i as f32);
                    }
                    validate_preset_override(&preset.name.name, setting, val, v.span(), diags);
                } else {
                    diags.push(Diagnostic::error(
                        "E0420",
                        format!(
                            "preset `{}` assigns an invalid value expression to setting `{}`",
                            preset.name.name, k.name
                        ),
                        v.span(),
                    ));
                }
            } else {
                diags.push(Diagnostic::error(
                    "E0416",
                    format!(
                        "preset `{}` assigns unknown setting key `{}`",
                        preset.name.name, k.name
                    ),
                    k.span,
                ));
            }
            overrides.insert(k.name.clone(), parsed_val.unwrap_or(SettingValueIr::Int(0)));
        }
        out.push(PresetIr {
            name: preset.name.name.clone(),
            label,
            description,
            order,
            tiers,
            overrides,
        });
    }
}

#[allow(clippy::float_cmp)]
fn float_eq(a: f32, b: f32) -> bool {
    a == b
}

#[allow(clippy::float_cmp)]
fn f64_eq(a: f64, b: f64) -> bool {
    a == b
}

#[allow(clippy::cast_precision_loss)]
fn handle_values_equal(a: &HandleValue, b: &HandleValue) -> bool {
    match (a, b) {
        (HandleValue::Int(x), HandleValue::Int(y)) => x == y,
        (HandleValue::Float(x), HandleValue::Float(y)) => float_eq(*x, *y),
        (HandleValue::Int(x), HandleValue::Float(y)) => f64_eq(*x as f64, f64::from(*y)),
        (HandleValue::Float(x), HandleValue::Int(y)) => f64_eq(f64::from(*x), *y as f64),
        (HandleValue::Bool(x), HandleValue::Bool(y)) => x == y,
        (HandleValue::EnumVariant(x), HandleValue::EnumVariant(y)) => x == y,
        _ => false,
    }
}

struct Sandbox<'a> {
    cfg: &'a EvalConfig,
    ir: &'a mut PipelineIr,
    diags: &'a mut DiagnosticBag,
    env: IndexMap<SmolStr, HandleValue>,
    steps: usize,
    aborted: bool,
}

impl Sandbox<'_> {
    fn invalid_value(&mut self, message: impl Into<String>, span: Span) -> HandleValue {
        self.diags.push(Diagnostic::error("E0430", message, span));
        HandleValue::Unit
    }

    fn tick(&mut self, span: Span) -> bool {
        if self.aborted {
            return false;
        }
        self.steps += 1;
        if self.steps > MAX_EVAL_STEPS {
            self.aborted = true;
            self.diags.push(Diagnostic::error(
                "E0490",
                format!(
                    "`pack.vsl` evaluation exceeded deterministic limit of {MAX_EVAL_STEPS} steps"
                ),
                span,
            ));
            return false;
        }
        true
    }

    fn eval_pipeline(&mut self, decl: &PipelineDecl) {
        // Seed pipeline parameters (`g: mut mc::GraphBuilder`, `cfg: mc::PipelineConfig`) so
        // member accesses and method calls on them do not trigger undefined-variable diagnostics.
        for (idx, param) in decl.params.iter().enumerate() {
            let role = param.ty.as_ref().and_then(|ty| match ty {
                TypeExpr::Named { path, .. } => {
                    let joined = path.joined();
                    if joined.contains("GraphBuilder") {
                        Some(HandleValue::GraphBuilder)
                    } else if joined.contains("PipelineConfig") {
                        Some(HandleValue::Config)
                    } else {
                        None
                    }
                }
                TypeExpr::Array { .. } => None,
            });
            let val = role.unwrap_or(if idx == 0 {
                HandleValue::GraphBuilder
            } else {
                HandleValue::Config
            });
            self.env.insert(param.name.name.clone(), val);
        }
        if !self.env.contains_key("g") {
            self.env
                .insert(SmolStr::new_static("g"), HandleValue::GraphBuilder);
        }
        if !self.env.contains_key("cfg") {
            self.env
                .insert(SmolStr::new_static("cfg"), HandleValue::Config);
        }

        let _ = self.eval_block(&decl.body);
    }

    fn eval_block(&mut self, block: &Block) -> ControlFlow {
        let saved_keys: indexmap::IndexSet<SmolStr> = self.env.keys().cloned().collect();
        let mut shadowed: Vec<(SmolStr, HandleValue)> = Vec::new();
        let result = self.eval_block_scoped(block, &saved_keys, &mut shadowed);
        self.restore_scope(&saved_keys, shadowed);
        result
    }

    fn restore_scope(
        &mut self,
        saved_keys: &indexmap::IndexSet<SmolStr>,
        shadowed: Vec<(SmolStr, HandleValue)>,
    ) {
        self.env.retain(|k, _| saved_keys.contains(k));
        for (k, prev) in shadowed {
            self.env.insert(k, prev);
        }
    }

    fn eval_block_scoped(
        &mut self,
        block: &Block,
        saved_keys: &indexmap::IndexSet<SmolStr>,
        shadowed: &mut Vec<(SmolStr, HandleValue)>,
    ) -> ControlFlow {
        for stmt in &block.stmts {
            if !self.tick(block.span) {
                return ControlFlow::Return;
            }
            match stmt {
                Stmt::Let { name, init, .. } => {
                    let val = if let Some(expr) = init {
                        match self.eval_expr(expr) {
                            ControlFlow::Normal(val) => val,
                            other => return other,
                        }
                    } else {
                        HandleValue::Unit
                    };
                    if let Some(prev) = self.env.insert(name.name.clone(), val)
                        && saved_keys.contains(&name.name)
                        && !shadowed.iter().any(|(k, _)| k == &name.name)
                    {
                        shadowed.push((name.name.clone(), prev));
                    }
                }
                Stmt::Return { value, .. } => {
                    if let Some(expr) = value {
                        return match self.eval_expr(expr) {
                            ControlFlow::Normal(_) => ControlFlow::Return,
                            other => other,
                        };
                    }
                    return ControlFlow::Return;
                }
                Stmt::Break(_) => return ControlFlow::Break,
                Stmt::Continue(_) => return ControlFlow::Continue,
                Stmt::For {
                    binding,
                    iter,
                    body,
                    span,
                    ..
                } => {
                    if let Expr::Binary {
                        op: op @ (BinaryOp::Range | BinaryOp::RangeInclusive),
                        lhs,
                        rhs,
                        ..
                    } = iter
                    {
                        let start = match self.eval_expr(lhs) {
                            ControlFlow::Normal(HandleValue::Int(v)) => Some(v),
                            ControlFlow::Normal(_) => {
                                self.diags.push(Diagnostic::error(
                                    "E0492",
                                    "`pack.vsl` `for` range start bound must evaluate to an integer",
                                    lhs.span(),
                                ));
                                None
                            }
                            other => return other,
                        };
                        let end = match self.eval_expr(rhs) {
                            ControlFlow::Normal(HandleValue::Int(v)) => Some(v),
                            ControlFlow::Normal(_) => {
                                self.diags.push(Diagnostic::error(
                                    "E0492",
                                    "`pack.vsl` `for` range end bound must evaluate to an integer",
                                    rhs.span(),
                                ));
                                None
                            }
                            other => return other,
                        };
                        if let (Some(start), Some(end)) = (start, end) {
                            let raw_count = if matches!(op, BinaryOp::RangeInclusive) {
                                end.saturating_sub(start).saturating_add(1).max(0)
                            } else {
                                end.saturating_sub(start).max(0)
                            };
                            let count = usize::try_from(raw_count).unwrap_or(usize::MAX);
                            if count > MAX_LOOP_ITERS {
                                self.diags.push(Diagnostic::error(
                                    "E0491",
                                    format!(
                                        "`pack.vsl` loop exceeds maximum {MAX_LOOP_ITERS} iterations"
                                    ),
                                    *span,
                                ));
                            } else {
                                let prev_binding = self.env.get(&binding.name).cloned();
                                let mut early_return: Option<ControlFlow> = None;
                                for idx in 0..count {
                                    if !self.tick(*span) {
                                        early_return = Some(ControlFlow::Return);
                                        break;
                                    }
                                    let offset = i64::try_from(idx).unwrap_or(0);
                                    self.env.insert(
                                        binding.name.clone(),
                                        HandleValue::Int(start.saturating_add(offset)),
                                    );
                                    match self.eval_block(body) {
                                        ControlFlow::Normal(_) | ControlFlow::Continue => {}
                                        ControlFlow::Break => break,
                                        ret @ ControlFlow::Return => {
                                            early_return = Some(ret);
                                            break;
                                        }
                                    }
                                }
                                if let Some(prev) = prev_binding {
                                    self.env.insert(binding.name.clone(), prev);
                                } else {
                                    self.env.shift_remove(&binding.name);
                                }
                                if let Some(cf) = early_return {
                                    return cf;
                                }
                            }
                        }
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0492",
                            "`pack.vsl` `for` requires an integer range iterator",
                            *span,
                        ));
                    }
                }
                Stmt::While {
                    cond, body, span, ..
                } => {
                    let mut iters = 0usize;
                    loop {
                        if !self.tick(*span) {
                            return ControlFlow::Return;
                        }
                        let cond_val = match self.eval_expr(cond) {
                            ControlFlow::Normal(v) => v,
                            other => return other,
                        };
                        match cond_val {
                            HandleValue::Bool(true) => {}
                            HandleValue::Bool(false) => break,
                            _ => {
                                self.invalid_value(
                                    "while condition must evaluate to bool",
                                    cond.span(),
                                );
                                break;
                            }
                        }
                        iters += 1;
                        if iters > MAX_LOOP_ITERS {
                            self.diags.push(Diagnostic::error(
                                "E0491",
                                format!(
                                    "`pack.vsl` loop exceeds maximum {MAX_LOOP_ITERS} iterations"
                                ),
                                *span,
                            ));
                            break;
                        }
                        match self.eval_block(body) {
                            ControlFlow::Normal(_) | ControlFlow::Continue => {}
                            ControlFlow::Break => break,
                            ret @ ControlFlow::Return => return ret,
                        }
                    }
                }
                Stmt::Loop { body, span, .. } => {
                    let mut iters = 0usize;
                    loop {
                        if !self.tick(*span) {
                            return ControlFlow::Return;
                        }
                        iters += 1;
                        if iters > MAX_LOOP_ITERS {
                            self.diags.push(Diagnostic::error(
                                "E0491",
                                format!(
                                    "`pack.vsl` loop exceeds maximum {MAX_LOOP_ITERS} iterations"
                                ),
                                *span,
                            ));
                            break;
                        }
                        match self.eval_block(body) {
                            ControlFlow::Normal(_) | ControlFlow::Continue => {}
                            ControlFlow::Break => break,
                            ret @ ControlFlow::Return => return ret,
                        }
                    }
                }
                Stmt::Switch {
                    selector,
                    cases,
                    default,
                    ..
                } => {
                    let sel_val = match self.eval_expr(selector) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    let mut matched = false;
                    for case in cases {
                        let case_val = match self.eval_expr(&case.value) {
                            ControlFlow::Normal(v) => v,
                            other => return other,
                        };
                        if handle_values_equal(&sel_val, &case_val) {
                            matched = true;
                            match self.eval_block(&case.body) {
                                ControlFlow::Normal(_) | ControlFlow::Break => {}
                                other => return other,
                            }
                            break;
                        }
                    }
                    if !matched && let Some(def_block) = default {
                        match self.eval_block(def_block) {
                            ControlFlow::Normal(_) | ControlFlow::Break => {}
                            other => return other,
                        }
                    }
                }
                Stmt::Expr { expr, .. } => match self.eval_expr(expr) {
                    ControlFlow::Normal(_) => {}
                    other => return other,
                },
            }
        }
        if let Some(ref tail) = block.tail_expr {
            return self.eval_expr(tail);
        }
        ControlFlow::Normal(HandleValue::Unit)
    }

    fn eval_expr(&mut self, expr: &Expr) -> ControlFlow {
        if !self.tick(expr.span()) {
            return ControlFlow::Return;
        }
        match expr {
            Expr::BoolLit(b, _) => ControlFlow::Normal(HandleValue::Bool(*b)),
            Expr::IntLit(_, _) => ControlFlow::Normal(match eval_const_i64(expr) {
                Some(value) => HandleValue::Int(value),
                None => self.invalid_value(
                    "integer literal exceeds the evaluator's signed 64-bit range",
                    expr.span(),
                ),
            }),
            Expr::FloatLit(_, _) => {
                ControlFlow::Normal(match eval_const_f32(expr).filter(|v| v.is_finite()) {
                    Some(value) => HandleValue::Float(value),
                    None => self.invalid_value(
                        "floating-point literal must be finite and representable",
                        expr.span(),
                    ),
                })
            }
            Expr::Path { path, span, .. } => {
                if path.segments.len() == 1 {
                    let ident = &path.segments[0].name;
                    if let Some(val) = self.env.get(ident) {
                        return ControlFlow::Normal(val.clone());
                    }
                    self.diags.push(Diagnostic::error(
                        "E0418",
                        format!("undefined variable `{ident}` in `pack.vsl`"),
                        *span,
                    ));
                }
                ControlFlow::Normal(HandleValue::Unit)
            }
            Expr::Field {
                receiver, field, ..
            } => {
                if let Expr::Path { path, .. } = receiver.as_ref()
                    && matches!(path.joined().as_str(), "settings" | "params")
                    && !self.env.contains_key(path.joined().as_str())
                {
                    return ControlFlow::Normal(
                        self.lookup_setting_handle(&field.name, field.span),
                    );
                }
                let recv_val = match self.eval_expr(receiver) {
                    ControlFlow::Normal(v) => v,
                    other => return other,
                };
                let val = match recv_val {
                    HandleValue::Config => match field.name.as_str() {
                        "tier" => HandleValue::Int(i64::from(self.cfg.tier.as_u8())),
                        "params" | "settings" => HandleValue::Config,
                        _ => self.lookup_setting_handle(&field.name, field.span),
                    },
                    HandleValue::Vec3([x, y, z]) => match field.name.as_str() {
                        "x" | "r" => HandleValue::Float(x),
                        "y" | "g" => HandleValue::Float(y),
                        "z" | "b" => HandleValue::Float(z),
                        _ => HandleValue::Unit,
                    },
                    HandleValue::Vec4([x, y, z, w]) => match field.name.as_str() {
                        "x" | "r" => HandleValue::Float(x),
                        "y" | "g" => HandleValue::Float(y),
                        "z" | "b" => HandleValue::Float(z),
                        "w" | "a" => HandleValue::Float(w),
                        _ => HandleValue::Unit,
                    },
                    _ => HandleValue::Unit,
                };
                ControlFlow::Normal(val)
            }
            Expr::Unary { op, operand, .. } => {
                let inner = match self.eval_expr(operand) {
                    ControlFlow::Normal(v) => v,
                    other => return other,
                };
                let val = match (op, inner) {
                    (UnaryOp::Not, HandleValue::Bool(b)) => HandleValue::Bool(!b),
                    (UnaryOp::Neg, HandleValue::Int(i)) => HandleValue::Int(i.saturating_neg()),
                    (UnaryOp::Neg, HandleValue::Float(f)) => HandleValue::Float(-f),
                    (UnaryOp::BitNot, HandleValue::Int(i)) => HandleValue::Int(!i),
                    _ => HandleValue::Unit,
                };
                ControlFlow::Normal(val)
            }
            Expr::Binary { op, lhs, rhs, span } => self.eval_binary(*op, lhs, rhs, *span),
            Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let c = match self.eval_expr(cond) {
                    ControlFlow::Normal(v) => v,
                    other => return other,
                };
                match c {
                    HandleValue::Bool(true) => self.eval_block(then_branch),
                    HandleValue::Bool(false) => match else_branch {
                        Some(eb) => self.eval_expr(eb),
                        None => ControlFlow::Normal(HandleValue::Unit),
                    },
                    _ => ControlFlow::Normal(
                        self.invalid_value("if condition must evaluate to bool", cond.span()),
                    ),
                }
            }
            Expr::Block(block) => self.eval_block(block),
            Expr::Call { callee, args, .. } => self.eval_call(callee, args),
            Expr::MethodCall {
                receiver,
                method,
                turbofish,
                args,
                span,
            } => self.eval_method_call(receiver, &method.name, turbofish, args, *span),
            _ => ControlFlow::Normal(HandleValue::Unit),
        }
    }

    fn eval_binary(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr, span: Span) -> ControlFlow {
        match op {
            BinaryOp::And => match self.eval_expr(lhs) {
                ControlFlow::Normal(HandleValue::Bool(false)) => {
                    ControlFlow::Normal(HandleValue::Bool(false))
                }
                ControlFlow::Normal(HandleValue::Bool(true)) => match self.eval_expr(rhs) {
                    ControlFlow::Normal(v @ HandleValue::Bool(_)) => ControlFlow::Normal(v),
                    ControlFlow::Normal(_) => ControlFlow::Normal(
                        self.invalid_value("logical and requires boolean operands", rhs.span()),
                    ),
                    other => other,
                },
                ControlFlow::Normal(_) => ControlFlow::Normal(
                    self.invalid_value("logical and requires boolean operands", lhs.span()),
                ),
                other => other,
            },
            BinaryOp::Or => match self.eval_expr(lhs) {
                ControlFlow::Normal(HandleValue::Bool(true)) => {
                    ControlFlow::Normal(HandleValue::Bool(true))
                }
                ControlFlow::Normal(HandleValue::Bool(false)) => match self.eval_expr(rhs) {
                    ControlFlow::Normal(v @ HandleValue::Bool(_)) => ControlFlow::Normal(v),
                    ControlFlow::Normal(_) => ControlFlow::Normal(
                        self.invalid_value("logical or requires boolean operands", rhs.span()),
                    ),
                    other => other,
                },
                ControlFlow::Normal(_) => ControlFlow::Normal(
                    self.invalid_value("logical or requires boolean operands", lhs.span()),
                ),
                other => other,
            },
            BinaryOp::Assign
            | BinaryOp::AddAssign
            | BinaryOp::SubAssign
            | BinaryOp::MulAssign
            | BinaryOp::DivAssign
            | BinaryOp::RemAssign
            | BinaryOp::BitAndAssign
            | BinaryOp::BitOrAssign
            | BinaryOp::BitXorAssign => {
                let r = match self.eval_expr(rhs) {
                    ControlFlow::Normal(v) => v,
                    other => return other,
                };
                if let Expr::Path { path, .. } = lhs
                    && path.segments.len() == 1
                {
                    let name = &path.segments[0].name;
                    let Some(existing) = self.env.get(name).cloned() else {
                        self.diags.push(Diagnostic::error(
                            "E0418",
                            format!("undefined variable `{name}` in `pack.vsl`"),
                            span,
                        ));
                        return ControlFlow::Normal(HandleValue::Unit);
                    };
                    let updated = match op {
                        BinaryOp::Assign => r,
                        BinaryOp::AddAssign => apply_arith(BinaryOp::Add, existing, r),
                        BinaryOp::SubAssign => apply_arith(BinaryOp::Sub, existing, r),
                        BinaryOp::MulAssign => apply_arith(BinaryOp::Mul, existing, r),
                        BinaryOp::DivAssign => apply_arith(BinaryOp::Div, existing, r),
                        BinaryOp::RemAssign => apply_arith(BinaryOp::Rem, existing, r),
                        BinaryOp::BitAndAssign => apply_bitwise(BinaryOp::BitAnd, existing, r),
                        BinaryOp::BitOrAssign => apply_bitwise(BinaryOp::BitOr, existing, r),
                        BinaryOp::BitXorAssign => apply_bitwise(BinaryOp::BitXor, existing, r),
                        _ => HandleValue::Unit,
                    };
                    if matches!(updated, HandleValue::Unit) {
                        self.invalid_value(format!("invalid assignment operands for {op:?}"), span);
                    } else {
                        self.env.insert(name.clone(), updated);
                    }
                } else {
                    self.invalid_value("assignment requires a local variable", lhs.span());
                }
                ControlFlow::Normal(HandleValue::Unit)
            }
            _ => {
                let l = match self.eval_expr(lhs) {
                    ControlFlow::Normal(v) => v,
                    other => return other,
                };
                let r = match self.eval_expr(rhs) {
                    ControlFlow::Normal(v) => v,
                    other => return other,
                };
                #[allow(clippy::cast_precision_loss)]
                let val = match op {
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem => apply_arith(op, l, r),
                    BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::Shl
                    | BinaryOp::Shr => apply_bitwise(op, l, r),
                    BinaryOp::Ge => match (l, r) {
                        (HandleValue::Int(a), HandleValue::Int(b)) => HandleValue::Bool(a >= b),
                        (HandleValue::Float(a), HandleValue::Float(b)) => HandleValue::Bool(a >= b),
                        (HandleValue::Int(a), HandleValue::Float(b)) => {
                            HandleValue::Bool((a as f64) >= f64::from(b))
                        }
                        (HandleValue::Float(a), HandleValue::Int(b)) => {
                            HandleValue::Bool(f64::from(a) >= (b as f64))
                        }
                        _ => HandleValue::Unit,
                    },
                    BinaryOp::Gt => match (l, r) {
                        (HandleValue::Int(a), HandleValue::Int(b)) => HandleValue::Bool(a > b),
                        (HandleValue::Float(a), HandleValue::Float(b)) => HandleValue::Bool(a > b),
                        (HandleValue::Int(a), HandleValue::Float(b)) => {
                            HandleValue::Bool((a as f64) > f64::from(b))
                        }
                        (HandleValue::Float(a), HandleValue::Int(b)) => {
                            HandleValue::Bool(f64::from(a) > (b as f64))
                        }
                        _ => HandleValue::Unit,
                    },
                    BinaryOp::Le => match (l, r) {
                        (HandleValue::Int(a), HandleValue::Int(b)) => HandleValue::Bool(a <= b),
                        (HandleValue::Float(a), HandleValue::Float(b)) => HandleValue::Bool(a <= b),
                        (HandleValue::Int(a), HandleValue::Float(b)) => {
                            HandleValue::Bool((a as f64) <= f64::from(b))
                        }
                        (HandleValue::Float(a), HandleValue::Int(b)) => {
                            HandleValue::Bool(f64::from(a) <= (b as f64))
                        }
                        _ => HandleValue::Unit,
                    },
                    BinaryOp::Lt => match (l, r) {
                        (HandleValue::Int(a), HandleValue::Int(b)) => HandleValue::Bool(a < b),
                        (HandleValue::Float(a), HandleValue::Float(b)) => HandleValue::Bool(a < b),
                        (HandleValue::Int(a), HandleValue::Float(b)) => {
                            HandleValue::Bool((a as f64) < f64::from(b))
                        }
                        (HandleValue::Float(a), HandleValue::Int(b)) => {
                            HandleValue::Bool(f64::from(a) < (b as f64))
                        }
                        _ => HandleValue::Unit,
                    },
                    BinaryOp::Eq => HandleValue::Bool(handle_values_equal(&l, &r)),
                    BinaryOp::Ne => HandleValue::Bool(!handle_values_equal(&l, &r)),
                    _ => HandleValue::Unit,
                };
                if matches!(val, HandleValue::Unit) {
                    ControlFlow::Normal(
                        self.invalid_value(format!("invalid operands or result for {op:?}"), span),
                    )
                } else {
                    ControlFlow::Normal(val)
                }
            }
        }
    }

    fn lookup_setting_handle(&mut self, name: &str, span: Span) -> HandleValue {
        if let Some(setting) = self.ir.settings.get(name)
            && (self.cfg.tier.as_u8() < setting.tiers.0 || self.cfg.tier.as_u8() > setting.tiers.1)
        {
            self.diags.push(Diagnostic::error(
                "E0424",
                format!(
                    "setting `{name}` is only available on tiers {}..={}",
                    setting.tiers.0, setting.tiers.1
                ),
                span,
            ));
            return HandleValue::Unit;
        }
        let val = self
            .cfg
            .param_overrides
            .get(name)
            .or_else(|| self.ir.settings.get(name).map(|s| &s.default_value));
        match val {
            Some(SettingValueIr::Bool(b)) => HandleValue::Bool(*b),
            Some(SettingValueIr::Int(i)) => HandleValue::Int(*i),
            Some(SettingValueIr::Float(f)) => HandleValue::Float(*f),
            Some(SettingValueIr::ColorRgb(rgb)) => HandleValue::Vec3(*rgb),
            Some(SettingValueIr::EnumVariant(s)) => HandleValue::EnumVariant(s.clone()),
            None => {
                self.diags.push(Diagnostic::error(
                    "E0416",
                    format!("unknown pack setting `{name}`"),
                    span,
                ));
                HandleValue::Unit
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn eval_call(&mut self, callee: &Expr, args: &[CallArg]) -> ControlFlow {
        let Expr::Path { path, .. } = callee else {
            return ControlFlow::Normal(HandleValue::Unit);
        };
        match path.joined().as_str() {
            "vec4" => {
                let mut components = Vec::with_capacity(args.len());
                for arg in args {
                    let val = match self.eval_expr(&arg.value) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    match val {
                        HandleValue::Float(f) => components.push(f),
                        HandleValue::Int(i) => components.push(i as f32),
                        _ => return ControlFlow::Normal(HandleValue::Unit),
                    }
                }
                match components.as_slice() {
                    [v] => ControlFlow::Normal(HandleValue::Vec4([*v, *v, *v, *v])),
                    [x, y, z, w] => ControlFlow::Normal(HandleValue::Vec4([*x, *y, *z, *w])),
                    _ => ControlFlow::Normal(HandleValue::Unit),
                }
            }
            "vec3" => {
                let mut components = Vec::with_capacity(args.len());
                for arg in args {
                    let val = match self.eval_expr(&arg.value) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    match val {
                        HandleValue::Float(f) => components.push(f),
                        HandleValue::Int(i) => components.push(i as f32),
                        _ => return ControlFlow::Normal(HandleValue::Unit),
                    }
                }
                match components.as_slice() {
                    [v] => ControlFlow::Normal(HandleValue::Vec3([*v, *v, *v])),
                    [x, y, z] => ControlFlow::Normal(HandleValue::Vec3([*x, *y, *z])),
                    _ => ControlFlow::Normal(HandleValue::Unit),
                }
            }
            "write" => {
                if let Some(arg) = args.first() {
                    let inner = match self.eval_expr(&arg.value) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    match inner {
                        HandleValue::Target { name, .. } | HandleValue::TargetWrite { name } => {
                            return ControlFlow::Normal(HandleValue::TargetWrite { name });
                        }
                        HandleValue::Buffer { name } | HandleValue::BufferWrite { name } => {
                            return ControlFlow::Normal(HandleValue::BufferWrite { name });
                        }
                        _ => {
                            self.diags.push(Diagnostic::error(
                                "E0426",
                                "`write(...)` requires a custom target or buffer handle",
                                arg.span,
                            ));
                        }
                    }
                } else {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "`write(...)` requires a custom target or buffer handle",
                        callee.span(),
                    ));
                }
                ControlFlow::Normal(HandleValue::Unit)
            }
            other => {
                self.diags.push(Diagnostic::error(
                    "E0426",
                    format!("unknown function `{other}` in `pack.vsl`"),
                    callee.span(),
                ));
                ControlFlow::Normal(HandleValue::Unit)
            }
        }
    }

    fn eval_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        turbofish: &[TypeArg],
        args: &[CallArg],
        span: Span,
    ) -> ControlFlow {
        let recv_val = match self.eval_expr(receiver) {
            ControlFlow::Normal(v) => v,
            other => return other,
        };
        // Every handle kind has a fixed method surface; a typo on any of them is a silent
        // no-op unless it is diagnosed here.
        let handle_like = matches!(
            &recv_val,
            HandleValue::GraphBuilder
                | HandleValue::Config
                | HandleValue::Target { .. }
                | HandleValue::TargetWrite { .. }
                | HandleValue::TargetHistory { .. }
                | HandleValue::Buffer { .. }
                | HandleValue::BufferWrite { .. }
                | HandleValue::BuiltinTarget(_)
                | HandleValue::ResourceTexture { .. }
                | HandleValue::PassIndex { .. }
        );
        let result = match method {
            "param" | "spec" => {
                if let HandleValue::Config = recv_val {
                    if let Some(Expr::Path { path, span, .. }) = args.first().map(|a| &a.value) {
                        self.lookup_setting_handle(&path.joined(), *span)
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0425",
                            "`param`/`spec` requires a setting name",
                            span,
                        ));
                        HandleValue::Unit
                    }
                } else if handle_like {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        format!("method `{method}` is only valid on a pack config receiver"),
                        span,
                    ));
                    HandleValue::Unit
                } else {
                    recv_val
                }
            }
            "hook" => {
                if !matches!(recv_val, HandleValue::GraphBuilder) {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `hook` is only valid on a `GraphBuilder` receiver",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                }
                let stage_name = args
                    .first()
                    .and_then(|a| match &a.value {
                        Expr::Path { path, .. } => Some(path.joined()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let fn_path = args
                    .get(1)
                    .and_then(|a| match &a.value {
                        Expr::Path { path, .. } => Some(SmolStr::new(path.joined())),
                        _ => None,
                    })
                    .unwrap_or_default();
                if let Some(stage) = HookStage::from_name(&stage_name) {
                    self.ir.hooks.push(HookRegistration {
                        stage,
                        target_fn: fn_path,
                    });
                } else {
                    self.diags.push(Diagnostic::error(
                        "E0410",
                        format!("unknown hook stage `{stage_name}` in `g.hook(...)`"),
                        span,
                    ));
                }
                HandleValue::Unit
            }
            "resource_texture" => {
                if !matches!(recv_val, HandleValue::GraphBuilder) {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `resource_texture` is only valid on a `GraphBuilder` receiver",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                }
                let slot = self.ir.resource_textures.len() as u32;
                if slot >= PACK_RESOURCE_SLOTS {
                    self.diags.push(Diagnostic::error(
                        "E0411",
                        format!(
                            "pack texture assets exceed Set 3 limit of {PACK_RESOURCE_SLOTS} slots"
                        ),
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                }
                let Some(Expr::StringLit(path, _)) = args.first().map(|a| &a.value) else {
                    self.diags.push(Diagnostic::error(
                        "E0427",
                        "`resource_texture` requires a string literal path",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                let path = path.clone();
                let format = match args.get(1).map(|a| &a.value) {
                    Some(Expr::Path {
                        path: fmt_path,
                        span: fmt_span,
                        ..
                    }) => {
                        let fmt_name = fmt_path.joined();
                        if let Some(fmt) = ImageFormat::from_name(&fmt_name) {
                            fmt
                        } else {
                            self.diags.push(Diagnostic::error(
                                "E0417",
                                format!(
                                    "unknown image format `{fmt_name}` in `g.resource_texture(...)`"
                                ),
                                *fmt_span,
                            ));
                            ImageFormat::Rgba8
                        }
                    }
                    Some(other) => {
                        self.diags.push(Diagnostic::error(
                            "E0417",
                            "invalid image format expression in `g.resource_texture(...)`",
                            other.span(),
                        ));
                        ImageFormat::Rgba8
                    }
                    None => ImageFormat::Rgba8,
                };
                let filter = arg_ident_or(args.get(2), "linear");
                let wrap = arg_ident_or(args.get(3), "clamp");
                let name = SmolStr::new(
                    path.rsplit(['/', '\\'])
                        .next()
                        .unwrap_or(path.as_str())
                        .split_once('.')
                        .map_or(path.as_str(), |(stem, _)| stem),
                );
                self.ir.resource_textures.push(ResourceTextureDecl {
                    slot,
                    name,
                    path,
                    format,
                    filter,
                    wrap,
                });
                HandleValue::ResourceTexture { slot }
            }
            "target" | "target_array" => {
                if !matches!(recv_val, HandleValue::GraphBuilder) {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        format!("method `{method}` is only valid on a `GraphBuilder` receiver"),
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                }
                let format = match turbofish.first() {
                    Some(TypeArg::Type(TypeExpr::Named {
                        path: fmt_path,
                        span: fmt_span,
                        ..
                    })) => {
                        let fmt_name = fmt_path.joined();
                        if let Some(fmt) = ImageFormat::from_name(&fmt_name) {
                            fmt
                        } else {
                            self.diags.push(Diagnostic::error(
                                "E0417",
                                format!(
                                    "unknown image format `{fmt_name}` in `g.{method}::<...>(...)`"
                                ),
                                *fmt_span,
                            ));
                            ImageFormat::Rgba16f
                        }
                    }
                    Some(_) => {
                        self.diags.push(Diagnostic::error(
                            "E0417",
                            format!(
                                "invalid image format type argument in `g.{method}::<...>(...)`"
                            ),
                            span,
                        ));
                        ImageFormat::Rgba16f
                    }
                    None => ImageFormat::Rgba16f,
                };
                let Some(Expr::StringLit(name, _)) = args.first().map(|a| &a.value) else {
                    return ControlFlow::Normal(
                        self.invalid_value("target requires a string literal name", span),
                    );
                };
                let name = name.clone();
                let mut scale = 1.0f32;
                let mut layers = 1u32;
                let mut mip_levels = 1u32;
                let mut temporal = false;
                let mut clear_color = None;
                #[allow(clippy::cast_precision_loss)]
                for arg in args.iter().skip(1) {
                    if let Some(ref key) = arg.name {
                        let evaluated = match self.eval_expr(&arg.value) {
                            ControlFlow::Normal(v) => v,
                            other => return other,
                        };
                        match key.name.as_str() {
                            "scale" => match evaluated {
                                HandleValue::Float(f) if f.is_finite() && f > 0.0 => scale = f,
                                HandleValue::Int(i) if i > 0 => scale = i as f32,
                                _ => {
                                    self.invalid_value(
                                        "target scale must be finite and positive",
                                        arg.span,
                                    );
                                }
                            },
                            "layers" => match evaluated {
                                HandleValue::Int(v)
                                    if v > 0
                                        && let Ok(value) = u32::try_from(v) =>
                                {
                                    layers = value;
                                }
                                _ => {
                                    self.invalid_value(
                                        "target layers must be a positive u32 integer",
                                        arg.span,
                                    );
                                }
                            },
                            "mip_levels" => match evaluated {
                                HandleValue::Int(v)
                                    if v > 0
                                        && let Ok(value) = u32::try_from(v) =>
                                {
                                    mip_levels = value;
                                }
                                _ => {
                                    self.invalid_value(
                                        "target mip_levels must be a positive u32 integer",
                                        arg.span,
                                    );
                                }
                            },
                            "temporal" => {
                                if let HandleValue::Bool(b) = evaluated {
                                    temporal = b;
                                } else {
                                    self.invalid_value("target temporal must be boolean", arg.span);
                                }
                            }
                            "clear_color" => {
                                if let HandleValue::Vec4(c) = evaluated {
                                    clear_color = Some(c);
                                } else {
                                    self.invalid_value("target clear_color must be vec4", arg.span);
                                }
                            }
                            other => {
                                self.diags.push(Diagnostic::error(
                                    "E0429",
                                    format!(
                                        "unknown named argument `{other}` in `g.{method}(...)`"
                                    ),
                                    arg.span,
                                ));
                            }
                        }
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0429",
                            format!("unexpected positional argument in `g.{method}(...)`"),
                            arg.span,
                        ));
                    }
                }
                if let Some(existing) = self.ir.targets.get(&name) {
                    self.diags.push(Diagnostic::error(
                        "E0421",
                        format!("duplicate render target `{name}` in `pack.vsl`"),
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Target {
                        name,
                        temporal: existing.temporal,
                    });
                }
                self.ir.targets.insert(
                    name.clone(),
                    TargetDecl {
                        name: name.clone(),
                        format,
                        scale,
                        layers,
                        mip_levels,
                        temporal,
                        clear_color,
                    },
                );
                HandleValue::Target { name, temporal }
            }
            "buffer" => {
                if !matches!(recv_val, HandleValue::GraphBuilder) {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `buffer` is only valid on a `GraphBuilder` receiver",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                }
                let mut element_type = match turbofish.first() {
                    Some(TypeArg::Type(TypeExpr::Named { path, .. })) => {
                        SmolStr::new(path.joined())
                    }
                    _ => SmolStr::default(),
                };
                let Some(Expr::StringLit(name, _)) = args.first().map(|a| &a.value) else {
                    self.diags.push(Diagnostic::error(
                        "E0427",
                        "`buffer` requires a string literal buffer name",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                let name = name.clone();
                let mut count = None;
                let mut size_bytes = None;
                let mut indirect = false;
                for arg in args.iter().skip(1) {
                    if let Some(ref key) = arg.name {
                        match key.name.as_str() {
                            "count" => {
                                let evaluated = match self.eval_expr(&arg.value) {
                                    ControlFlow::Normal(v) => v,
                                    other => return other,
                                };
                                if let HandleValue::Int(v) = evaluated
                                    && v > 0
                                {
                                    count = u64::try_from(v).ok();
                                } else {
                                    self.diags.push(Diagnostic::error(
                                        "E0428",
                                        "`buffer` `count` argument must be a positive integer",
                                        arg.span,
                                    ));
                                }
                            }
                            "size_bytes" | "size" => {
                                let evaluated = match self.eval_expr(&arg.value) {
                                    ControlFlow::Normal(v) => v,
                                    other => return other,
                                };
                                if let HandleValue::Int(v) = evaluated
                                    && v > 0
                                {
                                    size_bytes = u64::try_from(v).ok();
                                } else {
                                    self.diags.push(Diagnostic::error(
                                        "E0428",
                                        "`buffer` `size_bytes` argument must be a positive integer",
                                        arg.span,
                                    ));
                                }
                            }
                            "indirect" => {
                                let evaluated = match self.eval_expr(&arg.value) {
                                    ControlFlow::Normal(v) => v,
                                    other => return other,
                                };
                                if let HandleValue::Bool(b) = evaluated {
                                    indirect = b;
                                }
                            }
                            "element_type" => match &arg.value {
                                Expr::StringLit(s, _) => element_type = s.clone(),
                                Expr::Path { path, .. } => {
                                    element_type = SmolStr::new(path.joined());
                                }
                                _ => {}
                            },
                            other => {
                                self.diags.push(Diagnostic::error(
                                    "E0429",
                                    format!("unknown named argument `{other}` in `g.buffer(...)`"),
                                    arg.span,
                                ));
                            }
                        }
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0429",
                            "unexpected positional argument in `g.buffer(...)`",
                            arg.span,
                        ));
                    }
                }
                if self.ir.buffers.contains_key(&name) {
                    self.diags.push(Diagnostic::error(
                        "E0421",
                        format!("duplicate storage buffer `{name}` in `pack.vsl`"),
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Buffer { name });
                }
                self.ir.buffers.insert(
                    name.clone(),
                    BufferDecl {
                        name: name.clone(),
                        element_type,
                        count,
                        size_bytes,
                        indirect,
                    },
                );
                HandleValue::Buffer { name }
            }
            "write" => match recv_val {
                HandleValue::Target { name, .. } | HandleValue::TargetWrite { name } => {
                    HandleValue::TargetWrite { name }
                }
                HandleValue::Buffer { name } | HandleValue::BufferWrite { name } => {
                    HandleValue::BufferWrite { name }
                }
                _ => {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `write` is only valid on a custom target or buffer handle",
                        span,
                    ));
                    HandleValue::Unit
                }
            },
            "history" => match recv_val {
                HandleValue::Target {
                    name,
                    temporal: true,
                } => HandleValue::TargetHistory { name },
                HandleValue::Target {
                    name,
                    temporal: false,
                } => {
                    self.diags.push(Diagnostic::error(
                        "E0412",
                        format!("target `{name}` was not declared with `temporal = true`"),
                        span,
                    ));
                    HandleValue::Unit
                }
                _ => {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `history` is only valid on a temporal custom target handle",
                        span,
                    ));
                    HandleValue::Unit
                }
            },
            "builtin_target" => {
                if !matches!(recv_val, HandleValue::GraphBuilder) {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `builtin_target` is only valid on a `GraphBuilder` receiver",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                }
                if let Some(Expr::Path { path, .. }) = args.first().map(|a| &a.value) {
                    let tail = path.segments.last().map_or("", |seg| seg.name.as_str());
                    if let Some(res) = BuiltinResourceId::from_name(tail) {
                        return ControlFlow::Normal(HandleValue::BuiltinTarget(res));
                    }
                    self.diags.push(Diagnostic::error(
                        "E0413",
                        format!("unknown Hyaloclast builtin target `{}`", path.joined()),
                        span,
                    ));
                }
                HandleValue::Unit
            }
            "compute_pass" | "raster_pass" | "transfer_pass" | "mip_pass" => {
                if !matches!(recv_val, HandleValue::GraphBuilder) {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        format!("method `{method}` is only valid on a `GraphBuilder` receiver"),
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                }
                let kind = match method {
                    "compute_pass" => PassKind::Compute,
                    "raster_pass" => PassKind::Raster,
                    _ => PassKind::Transfer,
                };
                let is_mip = method == "mip_pass";
                let name = match args.first().map(|a| &a.value) {
                    Some(Expr::StringLit(s, _)) => s.clone(),
                    _ => SmolStr::new_static("unnamed_pass"),
                };
                let entry_fn = match args.get(1).map(|a| &a.value) {
                    Some(Expr::Path { path, .. }) => {
                        let is_local_handle = kind == PassKind::Transfer
                            && path.segments.len() == 1
                            && self.env.contains_key(&path.segments[0].name);
                        if is_local_handle {
                            SmolStr::default()
                        } else {
                            SmolStr::new(path.joined())
                        }
                    }
                    _ => SmolStr::default(),
                };
                let idx = self.ir.passes.len();
                self.ir.passes.push(PassNode {
                    name,
                    kind,
                    entry_fn,
                    replaces: None,
                    after: Vec::new(),
                    before: Vec::new(),
                    bindings: IndexMap::new(),
                    meta: PassMeta::default(),
                    dispatch_target: None,
                    dispatch: None,
                });
                HandleValue::PassIndex { idx, is_mip }
            }
            "replaces" => {
                let HandleValue::PassIndex { idx, is_mip } = recv_val else {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `replaces` is only valid on a pass handle",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                if let Some(Expr::Path { path, .. }) = args.first().map(|a| &a.value) {
                    let tail = path.segments.last().map_or("", |seg| seg.name.as_str());
                    if let Some(canonical) = CanonicalPass::from_name(tail) {
                        self.ir.passes[idx].replaces = Some(canonical);
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0414",
                            format!("unknown canonical Hyaloclast pass `{}`", path.joined()),
                            span,
                        ));
                    }
                }
                return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
            }
            "after" | "before" => {
                let HandleValue::PassIndex { idx, is_mip } = recv_val else {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        format!("method `{method}` is only valid on a pass handle"),
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                if let Some(arg) = args.first() {
                    let pass_ref = match &arg.value {
                        Expr::StringLit(s, _) => CanonicalPass::from_name(s).map_or_else(
                            || Some(PassRef::Custom(s.clone())),
                            |cp| Some(PassRef::Canonical(cp)),
                        ),
                        Expr::Path { path, .. } => {
                            let tail = path.segments.last().map_or("", |seg| seg.name.as_str());
                            if path.segments.len() > 1 {
                                if let Some(cp) = CanonicalPass::from_name(tail) {
                                    Some(PassRef::Canonical(cp))
                                } else {
                                    self.diags.push(Diagnostic::error(
                                        "E0414",
                                        format!(
                                            "unknown canonical Hyaloclast pass `{}`",
                                            path.joined()
                                        ),
                                        span,
                                    ));
                                    None
                                }
                            } else if let Some(HandleValue::PassIndex { idx: other_idx, .. }) =
                                self.env.get(tail)
                            {
                                Some(PassRef::Custom(self.ir.passes[*other_idx].name.clone()))
                            } else if let Some(cp) = CanonicalPass::from_name(tail) {
                                Some(PassRef::Canonical(cp))
                            } else {
                                Some(PassRef::Custom(SmolStr::new(tail)))
                            }
                        }
                        _ => None,
                    };
                    if let Some(pref) = pass_ref {
                        let pass = &mut self.ir.passes[idx];
                        if method == "after" {
                            pass.after.push(pref);
                        } else {
                            pass.before.push(pref);
                        }
                    }
                }
                return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
            }
            "bind" => {
                let HandleValue::PassIndex { idx, is_mip } = recv_val else {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `bind` is only valid on a pass handle",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                for arg in args {
                    let Some(ref key) = arg.name else {
                        self.diags.push(Diagnostic::error(
                            "E0429",
                            "`bind(...)` requires named `key = resource` arguments",
                            arg.span,
                        ));
                        continue;
                    };
                    let val = match self.eval_expr(&arg.value) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    let binding = match val {
                        HandleValue::Target { name, .. } => Some(PassBindingRef::CustomTarget {
                            name,
                            history: false,
                            is_write: is_mip || is_write_binding_key(key.name.as_str()),
                        }),
                        HandleValue::TargetWrite { name } => Some(PassBindingRef::CustomTarget {
                            name,
                            history: false,
                            is_write: true,
                        }),
                        HandleValue::TargetHistory { name } => Some(PassBindingRef::CustomTarget {
                            name,
                            history: true,
                            is_write: false,
                        }),
                        HandleValue::Buffer { name } => Some(PassBindingRef::CustomBuffer {
                            name,
                            is_write: is_write_binding_key(key.name.as_str()),
                        }),
                        HandleValue::BufferWrite { name } => Some(PassBindingRef::CustomBuffer {
                            name,
                            is_write: true,
                        }),
                        HandleValue::BuiltinTarget(resource) => {
                            Some(PassBindingRef::BuiltinTarget { resource })
                        }
                        HandleValue::ResourceTexture { slot } => {
                            Some(PassBindingRef::ResourceTexture { slot })
                        }
                        _ => {
                            self.diags.push(Diagnostic::error(
                                "E0415",
                                format!("unresolved resource handle for binding `{}`", key.name),
                                arg.span,
                            ));
                            None
                        }
                    };
                    if let Some(b) = binding {
                        self.ir.passes[idx].bindings.insert(key.name.clone(), b);
                    }
                }
                return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
            }
            "dispatch_for" => {
                let HandleValue::PassIndex { idx, is_mip } = recv_val else {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `dispatch_for` is only valid on a pass handle",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                if let Some(arg) = args.first() {
                    let val = match self.eval_expr(&arg.value) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    if let HandleValue::Target { name, .. } | HandleValue::TargetWrite { name } =
                        val
                    {
                        self.ir.passes[idx].dispatch_target = Some(name.clone());
                        self.ir.passes[idx].dispatch = Some(PassDispatch::ForTarget(name));
                    } else {
                        self.diags.push(Diagnostic::error(
                            "E0415",
                            "unresolved custom target handle passed to `dispatch_for(...)`",
                            arg.span,
                        ));
                    }
                }
                return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
            }
            "dispatch" => {
                let HandleValue::PassIndex { idx, is_mip } = recv_val else {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `dispatch` is only valid on a pass handle",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                if args.len() != 3 {
                    self.diags.push(Diagnostic::error(
                        "E0428",
                        "`dispatch(x, y, z)` requires exactly 3 positive integer arguments",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
                }
                let mut dims = [0u32; 3];
                let mut valid = true;
                for (dim_idx, arg) in args.iter().enumerate() {
                    let val = match self.eval_expr(&arg.value) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    match val {
                        HandleValue::Int(v) if v > 0 => {
                            if let Ok(d) = u32::try_from(v) {
                                dims[dim_idx] = d;
                            } else {
                                valid = false;
                                self.diags.push(Diagnostic::error(
                                    "E0428",
                                    format!(
                                        "`dispatch(x, y, z)` argument {} exceeds `u32::MAX`",
                                        dim_idx + 1
                                    ),
                                    arg.span,
                                ));
                            }
                        }
                        _ => {
                            valid = false;
                            self.diags.push(Diagnostic::error(
                                "E0428",
                                format!(
                                    "`dispatch(x, y, z)` argument {} must be a positive integer",
                                    dim_idx + 1
                                ),
                                arg.span,
                            ));
                        }
                    }
                }
                if valid {
                    self.ir.passes[idx].dispatch_target = None;
                    self.ir.passes[idx].dispatch = Some(PassDispatch::Explicit(dims));
                }
                return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
            }
            "dispatch_indirect" => {
                let HandleValue::PassIndex { idx, is_mip } = recv_val else {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        "method `dispatch_indirect` is only valid on a pass handle",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::Unit);
                };
                let Some(arg) = args.first() else {
                    self.diags.push(Diagnostic::error(
                        "E0428",
                        "`dispatch_indirect(...)` requires a declared buffer handle or buffer name",
                        span,
                    ));
                    return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
                };
                let resolved = if let Expr::StringLit(s, _) = &arg.value {
                    if s.trim().is_empty() || !self.ir.buffers.contains_key(s.as_str()) {
                        None
                    } else {
                        Some(s.clone())
                    }
                } else {
                    let val = match self.eval_expr(&arg.value) {
                        ControlFlow::Normal(v) => v,
                        other => return other,
                    };
                    match val {
                        HandleValue::Buffer { name } | HandleValue::BufferWrite { name } => {
                            Some(name)
                        }
                        _ => None,
                    }
                };
                if let Some(name) = resolved {
                    self.ir.passes[idx].dispatch_target = None;
                    self.ir.passes[idx].dispatch = Some(PassDispatch::Indirect(name));
                } else {
                    self.diags.push(Diagnostic::error(
                        "E0415",
                        "unresolved or undeclared buffer passed to `dispatch_indirect(...)`",
                        arg.span,
                    ));
                }
                return ControlFlow::Normal(HandleValue::PassIndex { idx, is_mip });
            }
            _ => {
                if handle_like {
                    self.diags.push(Diagnostic::error(
                        "E0426",
                        format!("unknown method `{method}` on pack handle"),
                        span,
                    ));
                    HandleValue::Unit
                } else {
                    recv_val
                }
            }
        };
        ControlFlow::Normal(result)
    }
}

#[allow(clippy::cast_precision_loss)]
fn apply_arith(op: BinaryOp, lhs: HandleValue, rhs: HandleValue) -> HandleValue {
    match (op, lhs, rhs) {
        (BinaryOp::Add, HandleValue::Int(a), HandleValue::Int(b)) => {
            a.checked_add(b).map_or(HandleValue::Unit, HandleValue::Int)
        }
        (BinaryOp::Sub, HandleValue::Int(a), HandleValue::Int(b)) => {
            a.checked_sub(b).map_or(HandleValue::Unit, HandleValue::Int)
        }
        (BinaryOp::Mul, HandleValue::Int(a), HandleValue::Int(b)) => {
            a.checked_mul(b).map_or(HandleValue::Unit, HandleValue::Int)
        }
        (BinaryOp::Div, HandleValue::Int(a), HandleValue::Int(b)) => {
            a.checked_div(b).map_or(HandleValue::Unit, HandleValue::Int)
        }
        (BinaryOp::Rem, HandleValue::Int(a), HandleValue::Int(b)) => {
            a.checked_rem(b).map_or(HandleValue::Unit, HandleValue::Int)
        }
        (op, HandleValue::Int(a), HandleValue::Float(b)) => {
            apply_arith(op, HandleValue::Float(a as f32), HandleValue::Float(b))
        }
        (op, HandleValue::Float(a), HandleValue::Int(b)) => {
            apply_arith(op, HandleValue::Float(a), HandleValue::Float(b as f32))
        }
        (BinaryOp::Add, HandleValue::Float(a), HandleValue::Float(b)) => finite_float(a + b),
        (BinaryOp::Sub, HandleValue::Float(a), HandleValue::Float(b)) => finite_float(a - b),
        (BinaryOp::Mul, HandleValue::Float(a), HandleValue::Float(b)) => finite_float(a * b),
        (BinaryOp::Div, HandleValue::Float(a), HandleValue::Float(b)) => finite_float(a / b),
        (BinaryOp::Rem, HandleValue::Float(a), HandleValue::Float(b)) => finite_float(a % b),
        _ => HandleValue::Unit,
    }
}

fn finite_float(value: f32) -> HandleValue {
    if value.is_finite() {
        HandleValue::Float(value)
    } else {
        HandleValue::Unit
    }
}

fn apply_bitwise(op: BinaryOp, lhs: HandleValue, rhs: HandleValue) -> HandleValue {
    match (op, lhs, rhs) {
        (BinaryOp::BitAnd, HandleValue::Int(a), HandleValue::Int(b)) => HandleValue::Int(a & b),
        (BinaryOp::BitOr, HandleValue::Int(a), HandleValue::Int(b)) => HandleValue::Int(a | b),
        (BinaryOp::BitXor, HandleValue::Int(a), HandleValue::Int(b)) => HandleValue::Int(a ^ b),
        (BinaryOp::Shl, HandleValue::Int(a), HandleValue::Int(b)) => u32::try_from(b)
            .ok()
            .and_then(|shift| a.checked_shl(shift))
            .map_or(HandleValue::Unit, HandleValue::Int),
        (BinaryOp::Shr, HandleValue::Int(a), HandleValue::Int(b)) => u32::try_from(b)
            .ok()
            .and_then(|shift| a.checked_shr(shift))
            .map_or(HandleValue::Unit, HandleValue::Int),
        _ => HandleValue::Unit,
    }
}

fn arg_ident_or(arg: Option<&CallArg>, fallback: &'static str) -> SmolStr {
    match arg.map(|a| &a.value) {
        Some(Expr::Path { path, .. }) => SmolStr::new(path.joined()),
        _ => SmolStr::new_static(fallback),
    }
}

fn eval_tier_range(expr: &Expr, diags: &mut DiagnosticBag) -> Option<(u8, u8)> {
    if let Expr::IntLit(_, _) = expr {
        let single = u8::try_from(eval_const_i64(expr)?.clamp(0, 3)).ok()?;
        return Some((single, single));
    }
    let Expr::Binary {
        op: op @ (BinaryOp::RangeInclusive | BinaryOp::Range),
        lhs,
        rhs,
        ..
    } = expr
    else {
        return None;
    };
    let raw_lo = eval_const_i64(lhs)?;
    let raw_hi = eval_const_i64(rhs)?;
    let lo = raw_lo.clamp(0, 3);
    let hi = if matches!(op, BinaryOp::Range) {
        raw_hi.saturating_sub(1).clamp(0, 3)
    } else {
        raw_hi.clamp(0, 3)
    };
    // Clamping can hide an empty exclusive range (`0..0` would otherwise clamp to `0..=0`).
    let empty_exclusive = matches!(op, BinaryOp::Range) && raw_hi <= raw_lo;
    if lo > hi || empty_exclusive {
        let sep = if matches!(op, BinaryOp::Range) {
            ".."
        } else {
            "..="
        };
        diags.push(Diagnostic::error(
            "E0422",
            format!("invalid tier range `{raw_lo}{sep}{raw_hi}`"),
            expr.span(),
        ));
        return None;
    }
    Some((u8::try_from(lo).unwrap_or(0), u8::try_from(hi).unwrap_or(3)))
}

fn eval_numeric_range(expr: &Expr, diags: &mut DiagnosticBag) -> Option<(f64, f64)> {
    let Expr::Binary {
        op: BinaryOp::RangeInclusive | BinaryOp::Range,
        lhs,
        rhs,
        ..
    } = expr
    else {
        return None;
    };
    let lo = eval_const_f64(lhs)?;
    let hi = eval_const_f64(rhs)?;
    if lo > hi {
        diags.push(Diagnostic::error(
            "E0420",
            format!("invalid numeric range `{lo}..={hi}`: lower bound exceeds upper bound"),
            expr.span(),
        ));
        return None;
    }
    Some((lo, hi))
}

fn strip_int_suffix(raw: &str) -> &str {
    for suffix in ["u64", "i64", "u32", "i32", "u", "i"] {
        if let Some(stripped) = raw.strip_suffix(suffix) {
            return stripped;
        }
    }
    raw
}

fn eval_const_i64(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::IntLit(raw, _) => {
            let clean = strip_int_suffix(raw).replace('_', "");
            if let Some(hex) = clean
                .strip_prefix("0x")
                .or_else(|| clean.strip_prefix("0X"))
            {
                i64::from_str_radix(hex, 16).ok()
            } else if let Some(bin) = clean
                .strip_prefix("0b")
                .or_else(|| clean.strip_prefix("0B"))
            {
                i64::from_str_radix(bin, 2).ok()
            } else if let Some(oct) = clean
                .strip_prefix("0o")
                .or_else(|| clean.strip_prefix("0O"))
            {
                i64::from_str_radix(oct, 8).ok()
            } else {
                clean.parse::<i64>().ok()
            }
        }
        Expr::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => eval_const_i64(operand).map(i64::saturating_neg),
        _ => None,
    }
}

#[allow(clippy::cast_precision_loss)]
fn eval_const_f32(expr: &Expr) -> Option<f32> {
    match expr {
        Expr::FloatLit(raw, _) => raw
            .trim_end_matches("f32")
            .trim_end_matches("f16")
            .trim_end_matches('f')
            .replace('_', "")
            .parse::<f32>()
            .ok(),
        Expr::IntLit(_, _) => eval_const_i64(expr).map(|v| v as f32),
        Expr::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => eval_const_f32(operand).map(|v| -v),
        _ => None,
    }
}

#[allow(clippy::cast_precision_loss)]
fn eval_const_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::FloatLit(raw, _) => raw
            .trim_end_matches("f32")
            .trim_end_matches("f16")
            .trim_end_matches('f')
            .replace('_', "")
            .parse::<f64>()
            .ok(),
        Expr::IntLit(_, _) => eval_const_i64(expr).map(|v| v as f64),
        Expr::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => eval_const_f64(operand).map(|v| -v),
        _ => None,
    }
}

fn eval_setting_value(expr: &Expr) -> Option<SettingValueIr> {
    match expr {
        Expr::BoolLit(b, _) => Some(SettingValueIr::Bool(*b)),
        Expr::IntLit(_, _) => eval_const_i64(expr).map(SettingValueIr::Int),
        Expr::FloatLit(_, _) => eval_const_f32(expr).map(SettingValueIr::Float),
        Expr::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => match operand.as_ref() {
            Expr::IntLit(_, _) => eval_const_i64(expr).map(SettingValueIr::Int),
            Expr::FloatLit(_, _) => eval_const_f32(expr).map(SettingValueIr::Float),
            _ => None,
        },
        Expr::Path { path, .. } => Some(SettingValueIr::EnumVariant(SmolStr::new(path.joined()))),
        Expr::Call { callee, args, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref()
                && path.joined() == "vec3"
                && args.len() == 3
            {
                return Some(SettingValueIr::ColorRgb([
                    eval_const_f32(&args[0].value)?,
                    eval_const_f32(&args[1].value)?,
                    eval_const_f32(&args[2].value)?,
                ]));
            }
            None
        }
        _ => None,
    }
}
