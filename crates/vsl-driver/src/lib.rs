//! End-to-end VSL 2.0 compilation driver coordinating frontend, multi-file `ModuleGraph`,
//! HIR lowering + single-source reflection, structured SSA lowering & passes, direct SPIR-V 1.6
//! emission, Stage A GLSL 460 oracle + Raster G-Buffer hook bridge, L1/L2 BLAKE3 caching, and
//! `pack.vsl` DAG scheduling.

use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, FileId, SourceDatabase, Span};
use vsl_glsl::{emit_glsl_460, emit_raster_gbuffer_hook_slot};
use vsl_hir::module::{L1ModuleHash, ModuleGraph, compute_l2_artifact_hash};
use vsl_hir::{
    HirFunction, HirModule, LayoutKind, PackSchema, ScalarKind, Type, ValidationSummary,
    lower_unit_with_schema, validate_unit,
};
use vsl_ir::{
    AtomicOrder, AtomicScope, BuiltinKind, GlslStd450Op, InstData, InstId, IrDescriptorVar,
    IrFunction, IrModule, IrPushConstantBlock, IrPushConstantField, IrWorkgroupVar, MemoryAccess,
    PassPipeline, ScalarIrTy, StructuredStmt,
};
use vsl_mc::{
    CompiledShaderUnit, DescriptorAccess, DescriptorBinding, DescriptorKind, HardwareCaps,
    HookAbiRoute, HookStage, ImageFormat, RenderTier, ShaderStage, TextureDim,
};
use vsl_pack::{
    EvalConfig, PipelineIr, SchedulePlan, SettingValueIr, evaluate_pack_vsl, validate_and_schedule,
};
use vsl_spirv::emit_module_spirv;
use vsl_syntax::ast::{
    Attribute, BinaryOp, Block, Expr, FnDecl, Item, Stmt, TypeArg, TypeExpr, UnaryOp,
};
use vsl_syntax::{ParseResult, parse_source};

/// Target compilation options passed by `vulkanshaders-core` or `vslc`.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub tier: RenderTier,
    pub active_caps: HardwareCaps,
    /// Active specialization-constant values keyed by setting name (encoded as raw `u32` bits).
    pub spec_values: IndexMap<SmolStr, u32>,
    /// Optional `pack.vsl` schema providing pack-global `spec_id`s, BDA parameter byte offsets,
    /// and Set 3 resource texture slots.
    pub pack_schema: Option<PackSchema>,
    pub driver_uuid: [u8; 16],
}

impl CompileOptions {
    /// Returns tier-appropriate default hardware capabilities matching Hyaloclast's
    /// `device_merger.rs` baseline (`INT64_ATOMICS` enabled across all Vulkan 1.3 tiers,
    /// plus `RAY_QUERY` and `MAXIMAL_RECONVERGENCE` on Tier 2/3 hardware ray-tracing targets).
    #[must_use]
    pub fn for_tier(tier: RenderTier) -> Self {
        let active_caps = match tier {
            RenderTier::Tier0Raster | RenderTier::Tier1ComputeGi => HardwareCaps::INT64_ATOMICS,
            RenderTier::Tier2RayQuery | RenderTier::Tier3PathTracing => {
                HardwareCaps::INT64_ATOMICS
                    | HardwareCaps::RAY_QUERY
                    | HardwareCaps::MAXIMAL_RECONVERGENCE
            }
        };
        Self {
            tier,
            active_caps,
            spec_values: IndexMap::new(),
            pack_schema: None,
            driver_uuid: [0u8; 16],
        }
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::for_tier(RenderTier::Tier2RayQuery)
    }
}

/// Constructs a pack-wide [`PackSchema`] from an evaluated [`PipelineIr`] so `.vsl` shader
/// modules receive consistent `spec_id`s, std430-aligned `PackParamBuffer` byte offsets, and
/// Set 3 resource texture slots matching `pack.vsl`.
#[must_use]
pub fn pack_schema_from_pipeline(pipeline: &PipelineIr) -> PackSchema {
    let mut schema = PackSchema::default();
    let mut next_spec_id = 0u32;
    let mut param_offset = 0u32;

    for (name, setting) in &pipeline.settings {
        if setting.is_spec {
            schema.spec_ids.insert(name.clone(), next_spec_id);
            next_spec_id += 1;
        } else {
            let (size, align) = match setting.default_value {
                SettingValueIr::ColorRgb(_) => (12u32, 16u32),
                SettingValueIr::Bool(_)
                | SettingValueIr::Int(_)
                | SettingValueIr::Float(_)
                | SettingValueIr::EnumVariant(_) => (4u32, 4u32),
            };
            let aligned = param_offset.div_ceil(align) * align;
            schema.params.insert(name.clone(), (aligned, size));
            param_offset = aligned + size;
        }
    }

    for tex in &pipeline.resource_textures {
        let stem = tex
            .path
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(tex.path.as_str())
            .split('.')
            .next()
            .unwrap_or(tex.path.as_str());
        schema.resource_slots.insert(SmolStr::new(stem), tex.slot);
        schema.resource_slots.insert(tex.path.clone(), tex.slot);
        // v1 `[resources.<name>]` manifests name assets independently of their file path;
        // shaders reference them by that name, so register it as a schema key too.
        schema.resource_slots.insert(tex.name.clone(), tex.slot);
    }

    schema
}

/// Output of compiling a `.vsl` shader source module.
#[derive(Debug, Clone)]
pub struct ShaderCompileOutput {
    pub file: FileId,
    pub summary: ValidationSummary,
    pub hir: HirModule,
    pub units: Vec<CompiledShaderUnit>,
    /// Raster G-Buffer hooks lowered via `vsl-glsl::emit_raster_gbuffer_hook_slot`.
    pub hook_glsl_slots: Vec<(HookStage, String)>,
    pub diagnostics: Vec<Diagnostic>,
}

impl ShaderCompileOutput {
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// Output of compiling a multi-file VSL 2.0 project resolved through [`ModuleGraph`].
#[derive(Debug, Clone)]
pub struct ProjectCompileOutput {
    pub module_order: Vec<SmolStr>,
    pub l1_hashes: IndexMap<SmolStr, L1ModuleHash>,
    pub units: Vec<CompiledShaderUnit>,
    pub hook_glsl_slots: Vec<(HookStage, String)>,
    pub diagnostics: Vec<Diagnostic>,
}

impl ProjectCompileOutput {
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// Output of compiling a `.vsl` shader module through the Stage A GLSL 460 oracle backend.
#[derive(Debug, Clone)]
pub struct GlslCompileOutput {
    pub file: FileId,
    pub glsl: String,
    pub diagnostics: Vec<Diagnostic>,
}

impl GlslCompileOutput {
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// Output of evaluating and scheduling a `pack.vsl` manifest.
#[derive(Debug, Clone)]
pub struct PackEvalOutput {
    pub file: FileId,
    pub pipeline: PipelineIr,
    pub schedule: Option<SchedulePlan>,
    pub diagnostics: Vec<Diagnostic>,
}

impl PackEvalOutput {
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// Composite L1 cache key incorporating source/import hash, target tier, active hardware caps,
/// and `pack.vsl` schema hash so HIR caching is sound across configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct L1CacheKey {
    hash: L1ModuleHash,
    tier: RenderTier,
    active_caps: HardwareCaps,
    pack_schema_hash: [u8; 32],
}

/// Pure-Rust in-process VSL 2.0 compiler session with L1 (HIR) and L2 (SPIR-V) BLAKE3 caches.
#[derive(Debug, Default)]
pub struct VslCompiler {
    pub sources: SourceDatabase,
    l1_cache: IndexMap<L1CacheKey, HirModule>,
    l2_cache: IndexMap<[u8; 32], CompiledShaderUnit>,
}

impl VslCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn l1_cache_len(&self) -> usize {
        self.l1_cache.len()
    }

    pub fn l2_cache_len(&self) -> usize {
        self.l2_cache.len()
    }

    /// Parses, lowers to HIR (with early tier/cap pruning and single-source-of-truth reflection),
    /// lowers entry points and native hooks to structured SSA IR -> SPIR-V 1.6, lowers raster
    /// G-Buffer hooks through the GLSL hook bridge, and caches results under L1/L2 BLAKE3 keys.
    pub fn compile_shader_source(
        &mut self,
        file_name: impl Into<String>,
        source: &str,
        options: &CompileOptions,
    ) -> ShaderCompileOutput {
        let l1_hash = L1ModuleHash(*blake3::hash(source.as_bytes()).as_bytes());
        let file = self.sources.add_file(file_name, source);
        let parsed = parse_source(file, source);
        self.compile_shader_with_parsed(file, parsed, l1_hash, options)
    }

    /// Resolves and compiles a multi-file VSL 2.0 project in topological dependency order using
    /// [`ModuleGraph`], parsing each file once and propagating imported interface hashes into
    /// L1 and L2 cache keys.
    pub fn compile_project(
        &mut self,
        files: &[(String, String)],
        options: &CompileOptions,
    ) -> ProjectCompileOutput {
        let mut graph = ModuleGraph::new();
        let mut by_module: IndexMap<SmolStr, (FileId, ParseResult, String)> = IndexMap::new();
        let mut diagnostics = Vec::new();

        for (idx, (file_name, source)) in files.iter().enumerate() {
            let file_id = self.sources.add_file(file_name.clone(), source.clone());
            let parsed = parse_source(file_id, source);

            let mod_name = parsed.unit.module_decl.as_ref().map_or_else(
                || SmolStr::new(format!("anon_{idx}")),
                |m| SmolStr::new(m.path.joined()),
            );
            let imports: Vec<SmolStr> = parsed
                .unit
                .imports
                .iter()
                .map(|imp| SmolStr::new(imp.prefix.joined()))
                .collect();
            graph.register_module(mod_name.clone(), source, imports);
            by_module.insert(mod_name, (file_id, parsed, source.clone()));
        }

        let module_order = match graph.topological_order() {
            Ok(order) => order,
            Err(err) => {
                for (_, (_, parsed, _)) in by_module {
                    diagnostics.extend(parsed.diagnostics);
                }
                diagnostics.push(Diagnostic::error(
                    "E0500",
                    err.to_string(),
                    Span::synthetic(),
                ));
                return ProjectCompileOutput {
                    module_order: Vec::new(),
                    l1_hashes: IndexMap::new(),
                    units: Vec::new(),
                    hook_glsl_slots: Vec::new(),
                    diagnostics,
                };
            }
        };

        let l1_hashes = match graph.compute_l1_hashes() {
            Ok(hashes) => hashes,
            Err(err) => {
                for (_, (_, parsed, _)) in by_module {
                    diagnostics.extend(parsed.diagnostics);
                }
                diagnostics.push(Diagnostic::error(
                    "E0500",
                    err.to_string(),
                    Span::synthetic(),
                ));
                return ProjectCompileOutput {
                    module_order,
                    l1_hashes: IndexMap::new(),
                    units: Vec::new(),
                    hook_glsl_slots: Vec::new(),
                    diagnostics,
                };
            }
        };

        let mut units = Vec::new();
        let mut hook_glsl_slots = Vec::new();
        let mut exported_items: IndexMap<SmolStr, Vec<Item>> = IndexMap::new();

        for mod_name in &module_order {
            if let Some((file_id, mut parsed, source)) = by_module.shift_remove(mod_name) {
                // Collect `pub` structs, consts, and helper functions from imported project modules
                // so cross-module references resolve during HIR lowering and IR function inlining.
                let mut imported_items: Vec<Item> = Vec::new();
                for imp in &parsed.unit.imports {
                    let prefix = SmolStr::new(imp.prefix.joined());
                    if let Some(items) = exported_items.get(&prefix) {
                        for item in items {
                            let already_present = match item {
                                Item::Struct(s) => parsed
                                    .unit
                                    .items
                                    .iter()
                                    .chain(imported_items.iter())
                                    .any(|existing| {
                                        matches!(existing, Item::Struct(e) if e.name.name == s.name.name)
                                    }),
                                Item::Const(c) => parsed
                                    .unit
                                    .items
                                    .iter()
                                    .chain(imported_items.iter())
                                    .any(|existing| {
                                        matches!(existing, Item::Const(e) if e.name.name == c.name.name)
                                    }),
                                Item::Function(f) => parsed
                                    .unit
                                    .items
                                    .iter()
                                    .chain(imported_items.iter())
                                    .any(|existing| {
                                        matches!(existing, Item::Function(e) if e.name.name == f.name.name)
                                    }),
                                _ => true,
                            };
                            if !already_present {
                                imported_items.push(item.clone());
                            }
                        }
                    }
                }

                // Export this module's own `pub` structs, consts, and non-entry functions.
                let mut exports = Vec::new();
                for item in &parsed.unit.items {
                    match item {
                        Item::Struct(s) if s.is_pub => exports.push(Item::Struct(s.clone())),
                        Item::Const(c) if c.is_pub => exports.push(Item::Const(c.clone())),
                        Item::Function(f)
                            if f.is_pub
                                && !f.attrs.iter().any(|a| {
                                    a.entries
                                        .iter()
                                        .any(|e| e.name.name == "stage" || e.name.name == "hook")
                                }) =>
                        {
                            exports.push(Item::Function(f.clone()));
                        }
                        _ => {}
                    }
                }
                exported_items.insert(mod_name.clone(), exports);

                if !imported_items.is_empty() {
                    imported_items.append(&mut parsed.unit.items);
                    parsed.unit.items = imported_items;
                }

                let l1_hash = l1_hashes
                    .get(mod_name)
                    .copied()
                    .unwrap_or_else(|| L1ModuleHash(*blake3::hash(source.as_bytes()).as_bytes()));
                let out = self.compile_shader_with_parsed(file_id, parsed, l1_hash, options);
                diagnostics.extend(out.diagnostics);
                units.extend(out.units);
                hook_glsl_slots.extend(out.hook_glsl_slots);
            }
        }

        ProjectCompileOutput {
            module_order,
            l1_hashes,
            units,
            hook_glsl_slots,
            diagnostics,
        }
    }

    fn compile_shader_with_parsed(
        &mut self,
        file: FileId,
        parsed: ParseResult,
        l1_hash: L1ModuleHash,
        options: &CompileOptions,
    ) -> ShaderCompileOutput {
        let mut diagnostics = parsed.diagnostics;
        let has_parse_errors = diagnostics.iter().any(Diagnostic::is_error);

        let pack_schema_hash = hash_pack_schema(options.pack_schema.as_ref());
        let l1_key = L1CacheKey {
            hash: l1_hash,
            tier: options.tier,
            active_caps: options.active_caps,
            pack_schema_hash,
        };

        let hir = if !has_parse_errors && let Some(cached_hir) = self.l1_cache.get(&l1_key) {
            cached_hir.clone()
        } else {
            let (lowered_hir, hir_diags) = lower_unit_with_schema(
                &parsed.unit,
                options.tier,
                options.active_caps,
                options.pack_schema.as_ref(),
            );
            diagnostics.extend(hir_diags.into_vec());
            if !diagnostics.iter().any(Diagnostic::is_error) {
                self.l1_cache.insert(l1_key, lowered_hir.clone());
            }
            lowered_hir
        };

        let summary = build_summary_from_hir(&hir);

        if diagnostics.iter().any(Diagnostic::is_error) {
            return ShaderCompileOutput {
                file,
                summary,
                hir,
                units: Vec::new(),
                hook_glsl_slots: Vec::new(),
                diagnostics,
            };
        }

        let pipeline = PassPipeline::default();
        let mut units = Vec::new();
        let mut hook_glsl_slots = Vec::new();

        for (_, func) in hir.functions.iter() {
            // Raster G-Buffer hooks lower via the GLSL bridge (Spec §4.1).
            if let Some(hook_stage) = func.hook
                && hook_stage.abi_route() == HookAbiRoute::RasterGbufferGlslBridge
            {
                if let Some(ast_fn) = find_ast_function(&parsed.unit.items, &func.name) {
                    match emit_raster_gbuffer_hook_slot(hook_stage, ast_fn) {
                        Ok(glsl) => hook_glsl_slots.push((hook_stage, glsl)),
                        Err(msg) => diagnostics.push(Diagnostic::error("E0510", msg, func.span)),
                    }
                }
                continue;
            }

            // Native entry points (`@stage(...)`) and native SPIR-V hooks compile to SPIR-V 1.6.
            // Native compute hooks without an explicit `@workgroup_size` default to `[8, 8, 1]`.
            let (stage, effective_wg_size) = if let Some(s) = func.stage {
                (s, func.workgroup_size)
            } else if func
                .hook
                .is_some_and(|h| h.abi_route() == HookAbiRoute::NativeSpirv)
            {
                (
                    ShaderStage::Compute,
                    func.workgroup_size.or(Some([8, 8, 1])),
                )
            } else {
                continue;
            };

            let used_spec_pairs: Vec<(u32, u32)> = func
                .reflection
                .used_specs
                .iter()
                .map(|s| {
                    let raw = options.spec_values.get(&s.name).copied().unwrap_or(0);
                    (s.spec_id, raw)
                })
                .collect();
            let cache_key = compute_l2_artifact_hash(
                l1_hash,
                func.name.as_str(),
                options.tier,
                options.active_caps,
                &used_spec_pairs,
                &pack_schema_hash,
                &options.driver_uuid,
            )
            .0;

            if let Some(cached) = self.l2_cache.get(&cache_key) {
                units.push(cached.clone());
                continue;
            }

            let (ir_fn, unified_descriptors) =
                lower_hir_function_to_ir(&hir, func, &parsed.unit.items, stage, effective_wg_size);
            let mut ir_mod = IrModule {
                name: hir.module_name.clone(),
                functions: vec![ir_fn],
            };
            pipeline.run_module(&mut ir_mod);

            let spirv = emit_module_spirv(&ir_mod);
            let unit = CompiledShaderUnit {
                entry_point: func.name.clone(),
                stage,
                spirv,
                descriptors: unified_descriptors,
                push_constants: func.reflection.push_constants.clone(),
                push_constant_size: func.reflection.push_constant_size,
                workgroup_size: effective_wg_size,
                required_caps: func.required_caps,
                used_specs: func.reflection.used_specs.clone(),
                used_params: func.reflection.used_params.clone(),
                debug_channels: func.reflection.debug_channels.clone(),
                cache_key,
            };
            self.l2_cache.insert(cache_key, unit.clone());
            units.push(unit);
        }

        ShaderCompileOutput {
            file,
            summary,
            hir,
            units,
            hook_glsl_slots,
            diagnostics,
        }
    }

    /// Parses, validates, and emits Stage A GLSL 460 oracle source for differential testing.
    pub fn compile_shader_to_glsl(
        &mut self,
        file_name: impl Into<String>,
        source: &str,
    ) -> GlslCompileOutput {
        let file = self.sources.add_file(file_name, source);
        let parsed = parse_source(file, source);
        let mut diagnostics = parsed.diagnostics;

        let (_, val_diags) = validate_unit(&parsed.unit);
        diagnostics.extend(val_diags.into_vec());

        let glsl = if diagnostics.iter().any(Diagnostic::is_error) {
            String::new()
        } else {
            emit_glsl_460(&parsed.unit)
        };

        GlslCompileOutput {
            file,
            glsl,
            diagnostics,
        }
    }

    /// Parses, evaluates, and topologically schedules a `pack.vsl` file into a validated
    /// [`PipelineIr`] DAG and [`SchedulePlan`].
    pub fn evaluate_pack_source(
        &mut self,
        file_name: impl Into<String>,
        source: &str,
        cfg: &EvalConfig,
    ) -> PackEvalOutput {
        let file = self.sources.add_file(file_name, source);
        let parsed = parse_source(file, source);
        let mut diagnostics = parsed.diagnostics;

        let (pipeline, eval_diags) = evaluate_pack_vsl(&parsed.unit, cfg);
        diagnostics.extend(eval_diags.into_vec());

        let schedule = if diagnostics.iter().any(Diagnostic::is_error) {
            None
        } else {
            match validate_and_schedule(&pipeline) {
                Ok(plan) => Some(plan),
                Err(errors) => {
                    for msg in errors {
                        diagnostics.push(Diagnostic::error("E0450", msg, Span::new(file, 0, 0)));
                    }
                    None
                }
            }
        };

        PackEvalOutput {
            file,
            pipeline,
            schedule,
            diagnostics,
        }
    }
}

fn hash_pack_schema(schema: Option<&PackSchema>) -> [u8; 32] {
    let Some(schema) = schema else {
        return [0u8; 32];
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vsl2-pack-schema-v1");

    let mut params: Vec<_> = schema.params.iter().collect();
    params.sort_by_key(|(a, _)| *a);
    hasher.update(&(params.len() as u64).to_le_bytes());
    for (name, (off, sz)) in params {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&off.to_le_bytes());
        hasher.update(&sz.to_le_bytes());
    }

    let mut specs: Vec<_> = schema.spec_ids.iter().collect();
    specs.sort_by_key(|(a, _)| *a);
    hasher.update(&(specs.len() as u64).to_le_bytes());
    for (name, id) in specs {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&id.to_le_bytes());
    }

    let mut slots: Vec<_> = schema.resource_slots.iter().collect();
    slots.sort_by_key(|(a, _)| *a);
    hasher.update(&(slots.len() as u64).to_le_bytes());
    for (name, slot) in slots {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&slot.to_le_bytes());
    }

    *hasher.finalize().as_bytes()
}

fn build_summary_from_hir(hir: &HirModule) -> ValidationSummary {
    let mut summary = ValidationSummary::default();
    for (_, st) in hir.structs.iter() {
        if st.is_push_constant
            && let Some(ref layout) = st.computed_layout
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
    summary
}

fn find_ast_function<'a>(items: &'a [Item], name: &str) -> Option<&'a vsl_syntax::ast::FnDecl> {
    items.iter().find_map(|item| match item {
        Item::Function(f) if f.name.name == name => Some(f),
        _ => None,
    })
}

fn parse_builtin_kind(name: &str) -> Option<BuiltinKind> {
    match name {
        "global_invocation_id" => Some(BuiltinKind::GlobalInvocationId),
        "local_invocation_id" => Some(BuiltinKind::LocalInvocationId),
        "local_invocation_index" => Some(BuiltinKind::LocalInvocationIndex),
        "workgroup_id" => Some(BuiltinKind::WorkgroupId),
        "num_workgroups" => Some(BuiltinKind::NumWorkgroups),
        "vertex_index" => Some(BuiltinKind::VertexIndex),
        "instance_index" => Some(BuiltinKind::InstanceIndex),
        "frag_coord" => Some(BuiltinKind::FragCoord),
        _ => None,
    }
}

fn parse_swizzle_byte(b: u8) -> Option<u32> {
    match b {
        b'x' | b'r' => Some(0),
        b'y' | b'g' => Some(1),
        b'z' | b'b' => Some(2),
        b'w' | b'a' => Some(3),
        _ => None,
    }
}

fn parse_swizzle_components(field: &str) -> Option<(u32, Option<u32>)> {
    let bytes = field.as_bytes();
    let c0 = parse_swizzle_byte(*bytes.first()?)?;
    let c1 = bytes.get(1).copied().and_then(parse_swizzle_byte);
    Some((c0, c1))
}

/// Lowers a validated [`HirFunction`] and its reflection metadata into a structured SSA
/// [`IrFunction`], unifying and sorting `debug_channels` into both SPIR-V descriptors and
/// reflection.
fn lower_hir_function_to_ir(
    hir: &HirModule,
    func: &HirFunction,
    module_items: &[Item],
    stage: ShaderStage,
    workgroup_size: Option<[u32; 3]>,
) -> (IrFunction, Vec<DescriptorBinding>) {
    let mut ir_fn = IrFunction::new(func.name.clone());
    ir_fn.stage = Some(stage);
    ir_fn.workgroup_size = workgroup_size;
    ir_fn.required_caps = func.required_caps;
    ir_fn.uses_bda = func.reflection.uses_bda;
    ir_fn.uses_subgroups = func.reflection.uses_subgroups;

    // Combine regular descriptors and debug-channel storage images into one sorted, authoritative list.
    let mut unified_descriptors = func.reflection.descriptors.clone();
    for dbg in &func.reflection.debug_channels {
        let already_present = unified_descriptors
            .iter()
            .any(|d| d.set == dbg.set && d.binding == dbg.binding);
        if !already_present {
            unified_descriptors.push(DescriptorBinding {
                name: SmolStr::new(format!("debug_{}", dbg.channel_name)),
                set: dbg.set,
                binding: dbg.binding,
                kind: DescriptorKind::StorageImage,
                dim: Some(TextureDim::Dim2D),
                sample_scalar: None,
                image_format: Some(ImageFormat::Rgba16f),
                access: DescriptorAccess::WriteOnly,
                is_unsized: false,
                count: None,
            });
        }
    }
    unified_descriptors.sort_by_key(|d| (d.set, d.binding));

    ir_fn.descriptors = unified_descriptors
        .iter()
        .map(|d| IrDescriptorVar {
            name: d.name.clone(),
            set: d.set,
            binding: d.binding,
            kind: d.kind,
            dim: d.dim,
            sample_scalar: d.sample_scalar,
            image_format: d.image_format,
            access: d.access,
            is_unsized: d.is_unsized,
            count: d.count,
        })
        .collect();

    // Lower module workgroup variables into `StorageClass::Workgroup` IR declarations (Compute only).
    if stage == ShaderStage::Compute {
        for wg in &hir.workgroup_vars {
            let (element_is_u32, array_len) = match hir.types.get(wg.ty) {
                Type::Array { element, len } => (
                    matches!(hir.types.get(*element), Type::Scalar(ScalarKind::U32)),
                    Some(*len),
                ),
                Type::Scalar(ScalarKind::U32) => (true, None),
                _ => (false, None),
            };
            ir_fn.workgroup_vars.push(IrWorkgroupVar {
                name: wg.name.clone(),
                element_is_u32,
                array_len,
            });
        }
    }

    if func.reflection.push_constant_size > 0 {
        let pc_struct_fields = find_push_constant_struct_fields(hir, func);
        ir_fn.push_constants = Some(IrPushConstantBlock {
            name: SmolStr::new(format!("{}_push", func.name)),
            fields: func
                .reflection
                .push_constants
                .iter()
                .map(|f| {
                    let scalar = pc_struct_fields
                        .get(&f.name)
                        .copied()
                        .unwrap_or(ScalarIrTy::U32);
                    IrPushConstantField {
                        name: f.name.clone(),
                        byte_offset: f.byte_offset,
                        byte_size: f.byte_size,
                        scalar,
                    }
                })
                .collect(),
            total_size: func.reflection.push_constant_size,
        });
    }

    let mut ctx = LowerCtx::new(hir, func, module_items, &mut ir_fn);
    if let Some(ref block) = func.body_ast {
        let mut stmts = std::mem::take(&mut ctx.entry_preamble);
        stmts.extend(ctx.lower_block(block));
        ir_fn.body = stmts;
    }

    (ir_fn, unified_descriptors)
}

fn scalar_kind_to_ir(s: ScalarKind) -> ScalarIrTy {
    match s {
        ScalarKind::Bool => ScalarIrTy::Bool,
        ScalarKind::I32 => ScalarIrTy::I32,
        ScalarKind::U32 => ScalarIrTy::U32,
        ScalarKind::I64 => ScalarIrTy::I64,
        ScalarKind::U64 => ScalarIrTy::U64,
        ScalarKind::F32 => ScalarIrTy::F32,
        ScalarKind::F16 => ScalarIrTy::F16,
    }
}

fn type_id_to_scalar_ir(hir: &HirModule, ty_id: vsl_hir::TypeId) -> ScalarIrTy {
    match hir.types.get(ty_id) {
        Type::Scalar(s) | Type::Vector { scalar: s, .. } | Type::Matrix { scalar: s, .. } => {
            scalar_kind_to_ir(*s)
        }
        Type::Array { element, .. } => type_id_to_scalar_ir(hir, *element),
        Type::Pointer { .. } => ScalarIrTy::U64,
        _ => ScalarIrTy::U32,
    }
}

fn find_push_constant_struct_fields(
    hir: &HirModule,
    func: &HirFunction,
) -> IndexMap<SmolStr, ScalarIrTy> {
    let mut out = IndexMap::new();
    if let Some(ty_id) = func
        .params
        .iter()
        .find(|p| p.is_push_constant)
        .and_then(|p| p.ty)
        && let Type::Struct { fields, .. } = hir.types.get(ty_id)
    {
        for (name, field_ty) in fields {
            out.insert(name.clone(), type_id_to_scalar_ir(hir, *field_ty));
        }
    }
    if out.is_empty()
        && let Some((_, s)) = hir.structs.iter().find(|(_, s)| s.is_push_constant)
    {
        for field in &s.fields {
            out.insert(field.name.clone(), type_id_to_scalar_ir(hir, field.ty));
        }
    }
    out
}

struct LowerCtx<'a> {
    ir_fn: &'a mut IrFunction,
    entry_preamble: Vec<StructuredStmt>,
    locals: IndexMap<SmolStr, InstId>,
    ptr_meta: IndexMap<SmolStr, (u64, u32)>,
    descriptor_indices: IndexMap<SmolStr, u32>,
    storage_image_descriptors: IndexMap<SmolStr, u32>,
    tlas_descriptor_index: Option<u32>,
    workgroup_indices: IndexMap<SmolStr, u32>,
    builtin_params: IndexMap<SmolStr, BuiltinKind>,
    push_constant_param: Option<SmolStr>,
    push_constant_fields: IndexMap<SmolStr, u32>,
    ray_query_handles: IndexMap<SmolStr, u32>,
    loop_break_stack: Vec<LoopBreakCapture>,
    module_consts: IndexMap<SmolStr, Expr>,
    helper_fns: IndexMap<SmolStr, FnDecl>,
    spec_ids: IndexMap<SmolStr, (u32, ScalarIrTy)>,
    param_offsets: IndexMap<SmolStr, u32>,
    inline_depth: usize,
}

struct LoopBreakCapture {
    tracked_locals: Vec<SmolStr>,
    next_break_tag: u32,
    incoming_by_local: IndexMap<SmolStr, Vec<(InstId, u32)>>,
}

impl<'a> LowerCtx<'a> {
    fn new(
        hir: &HirModule,
        func: &HirFunction,
        module_items: &[Item],
        ir_fn: &'a mut IrFunction,
    ) -> Self {
        let mut entry_preamble = Vec::new();
        let mut locals = IndexMap::new();
        let mut ptr_meta = IndexMap::new();
        let mut descriptor_indices = IndexMap::new();
        let mut storage_image_descriptors = IndexMap::new();
        let mut tlas_descriptor_index = None;
        let mut workgroup_indices = IndexMap::new();
        let mut builtin_params = IndexMap::new();
        let mut push_constant_param = None;
        let mut push_constant_fields = IndexMap::new();

        for (idx, desc) in ir_fn.descriptors.iter().enumerate() {
            let idx_u32 = idx as u32;
            descriptor_indices.insert(desc.name.clone(), idx_u32);
            if desc.kind == DescriptorKind::StorageImage {
                storage_image_descriptors.insert(desc.name.clone(), idx_u32);
            } else if desc.kind == DescriptorKind::AccelerationStructure {
                tlas_descriptor_index = Some(idx_u32);
            }
        }

        for (idx, wg) in ir_fn.workgroup_vars.iter().enumerate() {
            workgroup_indices.insert(wg.name.clone(), idx as u32);
        }

        for (idx, field) in func.reflection.push_constants.iter().enumerate() {
            push_constant_fields.insert(field.name.clone(), idx as u32);
        }

        for param in &func.params {
            if let Some(ref builtin_name) = param.builtin
                && let Some(kind) = parse_builtin_kind(builtin_name.as_str())
            {
                builtin_params.insert(param.name.clone(), kind);
                let val = ir_fn.emit(
                    param.ty,
                    InstData::LoadBuiltin {
                        kind,
                        component: None,
                    },
                );
                entry_preamble.push(StructuredStmt::Eval(val));
                locals.insert(param.name.clone(), val);
                continue;
            }

            if param.is_push_constant {
                push_constant_param = Some(param.name.clone());
                let val = ir_fn.emit(param.ty, InstData::LoadPushConstant { field_index: 0 });
                entry_preamble.push(StructuredStmt::Eval(val));
                locals.insert(param.name.clone(), val);
                continue;
            }

            if let Some(ty_id) = param.ty {
                match *hir.types.get(ty_id) {
                    Type::Pointer { pointee, align, .. } => {
                        let stride = hir
                            .types
                            .size_and_align(pointee, LayoutKind::Std430)
                            .map_or(u64::from(align), |(sz, _)| u64::from(sz.max(4)));
                        ptr_meta.insert(param.name.clone(), (stride, align));
                        let raw = ir_fn.emit(None, InstData::ConstU64(0));
                        let ptr_val = ir_fn.emit(
                            Some(ty_id),
                            InstData::ConvertUToPtr {
                                addr_u64: raw,
                                align,
                            },
                        );
                        locals.insert(param.name.clone(), ptr_val);
                    }
                    Type::Scalar(ScalarKind::F32 | ScalarKind::F16)
                    | Type::Vector {
                        scalar: ScalarKind::F32 | ScalarKind::F16,
                        ..
                    } => {
                        let val = ir_fn.emit(Some(ty_id), InstData::const_f32(0.0));
                        locals.insert(param.name.clone(), val);
                    }
                    Type::Scalar(ScalarKind::U64) => {
                        let val = ir_fn.emit(Some(ty_id), InstData::ConstU64(0));
                        locals.insert(param.name.clone(), val);
                    }
                    Type::Scalar(ScalarKind::I64) => {
                        let val = ir_fn.emit(Some(ty_id), InstData::ConstI64(0));
                        locals.insert(param.name.clone(), val);
                    }
                    Type::Scalar(ScalarKind::I32)
                    | Type::Vector {
                        scalar: ScalarKind::I32,
                        ..
                    } => {
                        let val = ir_fn.emit(Some(ty_id), InstData::ConstI32(0));
                        locals.insert(param.name.clone(), val);
                    }
                    Type::Scalar(ScalarKind::Bool) => {
                        let val = ir_fn.emit(Some(ty_id), InstData::ConstBool(false));
                        locals.insert(param.name.clone(), val);
                    }
                    _ => {
                        let val = ir_fn.emit(Some(ty_id), InstData::ConstU32(0));
                        locals.insert(param.name.clone(), val);
                    }
                }
            }
        }

        Self {
            ir_fn,
            entry_preamble,
            locals,
            ptr_meta,
            descriptor_indices,
            storage_image_descriptors,
            tlas_descriptor_index,
            workgroup_indices,
            builtin_params,
            push_constant_param,
            push_constant_fields,
            ray_query_handles: IndexMap::new(),
            loop_break_stack: Vec::new(),
            module_consts: module_items
                .iter()
                .filter_map(|item| match item {
                    Item::Const(c) => Some((c.name.name.clone(), c.value.clone())),
                    _ => None,
                })
                .collect(),
            helper_fns: module_items
                .iter()
                .filter_map(|item| match item {
                    Item::Function(f) if f.body.is_some() && f.name.name != func.name => {
                        Some((f.name.name.clone(), f.clone()))
                    }
                    _ => None,
                })
                .collect(),
            spec_ids: func
                .reflection
                .used_specs
                .iter()
                .map(|s| (s.name.clone(), (s.spec_id, ScalarIrTy::U32)))
                .collect(),
            param_offsets: func
                .reflection
                .used_params
                .iter()
                .map(|p| (p.name.clone(), p.byte_offset))
                .collect(),
            inline_depth: 0,
        }
    }

    fn ray_query_id_for_receiver(&mut self, receiver: &Expr) -> u32 {
        let key = match receiver {
            Expr::Path { path, .. } => SmolStr::new(path.joined()),
            _ => SmolStr::new_static("__default_rq"),
        };
        if let Some(&id) = self.ray_query_handles.get(&key) {
            return id;
        }
        let next_id = self.ray_query_handles.len() as u32;
        self.ray_query_handles.insert(key, next_id);
        next_id
    }

    fn is_float_inst(&self, id: InstId) -> bool {
        let mut visited = IndexSet::new();
        self.is_float_inst_inner(id, &mut visited)
    }

    fn is_float_inst_inner(&self, id: InstId, visited: &mut IndexSet<InstId>) -> bool {
        if !visited.insert(id) {
            return false;
        }
        match &self.ir_fn.insts[id].data {
            InstData::ConstF32(_)
            | InstData::FAdd32(_, _)
            | InstData::FSub32(_, _)
            | InstData::FMul32(_, _)
            | InstData::FDiv32(_, _)
            | InstData::CastToF32(_)
            | InstData::ImageSample { .. }
            | InstData::RayQueryCommittedBarycentrics(_)
            | InstData::LoadBuiltin {
                kind: BuiltinKind::FragCoord,
                ..
            } => true,
            InstData::WorkgroupLoad { var_index, .. } => self
                .ir_fn
                .workgroup_vars
                .get(*var_index as usize)
                .is_some_and(|wg| !wg.element_is_u32),
            InstData::LoadPushConstant { field_index } => self
                .ir_fn
                .push_constants
                .as_ref()
                .and_then(|pc| pc.fields.get(*field_index as usize))
                .is_some_and(|f| matches!(f.scalar, ScalarIrTy::F32 | ScalarIrTy::F16)),
            InstData::GlslStd450 { op, .. } => !matches!(
                op,
                GlslStd450Op::SAbs
                    | GlslStd450Op::SMin
                    | GlslStd450Op::SMax
                    | GlslStd450Op::SClamp
                    | GlslStd450Op::UMin
                    | GlslStd450Op::UMax
                    | GlslStd450Op::UClamp
            ),
            InstData::SpecConstant { scalar, .. } | InstData::DescriptorLoad { scalar, .. } => {
                matches!(scalar, ScalarIrTy::F32 | ScalarIrTy::F16)
            }
            InstData::ExtractComponent { vec, .. } => self.is_float_inst_inner(*vec, visited),
            InstData::MakeVec2(a, b) => {
                self.is_float_inst_inner(*a, visited) || self.is_float_inst_inner(*b, visited)
            }
            InstData::MakeVec3(a, b, c) => {
                self.is_float_inst_inner(*a, visited)
                    || self.is_float_inst_inner(*b, visited)
                    || self.is_float_inst_inner(*c, visited)
            }
            InstData::MakeVec4(a, b, c, d) => {
                self.is_float_inst_inner(*a, visited)
                    || self.is_float_inst_inner(*b, visited)
                    || self.is_float_inst_inner(*c, visited)
                    || self.is_float_inst_inner(*d, visited)
            }
            InstData::SubgroupBroadcastFirst(inner) | InstData::NonUniform(inner) => {
                self.is_float_inst_inner(*inner, visited)
            }
            InstData::Select {
                on_true, on_false, ..
            } => {
                self.is_float_inst_inner(*on_true, visited)
                    || self.is_float_inst_inner(*on_false, visited)
            }
            InstData::Phi { incoming } => incoming
                .iter()
                .any(|&(v, _)| self.is_float_inst_inner(v, visited)),
            _ => false,
        }
    }

    fn is_64bit_inst(&self, id: InstId) -> bool {
        let mut visited = IndexSet::new();
        self.is_64bit_inst_inner(id, &mut visited)
    }

    fn is_64bit_inst_inner(&self, id: InstId, visited: &mut IndexSet<InstId>) -> bool {
        if !visited.insert(id) {
            return false;
        }
        match &self.ir_fn.insts[id].data {
            InstData::ConstU64(_)
            | InstData::ConstI64(_)
            | InstData::IAdd64(_, _)
            | InstData::IMul64(_, _)
            | InstData::CastI64ToU64(_)
            | InstData::CastToI64(_)
            | InstData::CastToU64(_)
            | InstData::ConvertPtrToU(_) => true,
            InstData::LoadPushConstant { field_index } => self
                .ir_fn
                .push_constants
                .as_ref()
                .and_then(|pc| pc.fields.get(*field_index as usize))
                .is_some_and(|f| matches!(f.scalar, ScalarIrTy::I64 | ScalarIrTy::U64)),
            InstData::SpecConstant { scalar, .. } | InstData::DescriptorLoad { scalar, .. } => {
                matches!(scalar, ScalarIrTy::I64 | ScalarIrTy::U64)
            }
            InstData::SubgroupBroadcastFirst(inner) | InstData::NonUniform(inner) => {
                self.is_64bit_inst_inner(*inner, visited)
            }
            InstData::Select {
                on_true, on_false, ..
            } => {
                self.is_64bit_inst_inner(*on_true, visited)
                    || self.is_64bit_inst_inner(*on_false, visited)
            }
            InstData::Phi { incoming } => incoming
                .iter()
                .any(|&(v, _)| self.is_64bit_inst_inner(v, visited)),
            _ => false,
        }
    }

    fn is_u32_inst(&self, id: InstId) -> bool {
        let mut visited = IndexSet::new();
        self.is_u32_inst_inner(id, &mut visited)
    }

    fn is_u32_inst_inner(&self, id: InstId, visited: &mut IndexSet<InstId>) -> bool {
        if !visited.insert(id) {
            return false;
        }
        match &self.ir_fn.insts[id].data {
            InstData::ConstU32(_)
            | InstData::CastToU32(_)
            | InstData::AtomicAdd { .. }
            | InstData::AtomicMin { .. }
            | InstData::AtomicMax { .. }
            | InstData::AtomicCas { .. }
            | InstData::SubgroupBallotBitCount(_)
            | InstData::RayQueryCommittedType(_)
            | InstData::RayQueryCommittedInstanceCustomIndex(_) => true,
            InstData::GlslStd450 { op, .. } => {
                matches!(
                    op,
                    GlslStd450Op::UMin | GlslStd450Op::UMax | GlslStd450Op::UClamp
                )
            }
            InstData::SpecConstant { scalar, .. } | InstData::DescriptorLoad { scalar, .. } => {
                matches!(scalar, ScalarIrTy::U32 | ScalarIrTy::Bool)
            }
            InstData::LoadBuiltin { kind, .. } => *kind != BuiltinKind::FragCoord,
            InstData::WorkgroupLoad { var_index, .. } => self
                .ir_fn
                .workgroup_vars
                .get(*var_index as usize)
                .is_some_and(|wg| wg.element_is_u32),
            InstData::LoadPushConstant { field_index } => self
                .ir_fn
                .push_constants
                .as_ref()
                .and_then(|pc| pc.fields.get(*field_index as usize))
                .is_some_and(|f| matches!(f.scalar, ScalarIrTy::U32 | ScalarIrTy::Bool)),
            InstData::SubgroupBroadcastFirst(inner) | InstData::NonUniform(inner) => {
                self.is_u32_inst_inner(*inner, visited)
            }
            InstData::Select {
                on_true, on_false, ..
            } => {
                self.is_u32_inst_inner(*on_true, visited)
                    && self.is_u32_inst_inner(*on_false, visited)
            }
            InstData::Phi { incoming } => {
                !incoming.is_empty()
                    && incoming
                        .iter()
                        .all(|&(v, _)| v == id || self.is_u32_inst_inner(v, visited))
            }
            _ => false,
        }
    }

    fn is_const_inst(&self, id: InstId) -> bool {
        matches!(
            self.ir_fn.insts[id].data,
            InstData::ConstBool(_)
                | InstData::ConstI32(_)
                | InstData::ConstU32(_)
                | InstData::ConstF32(_)
                | InstData::ConstI64(_)
                | InstData::ConstU64(_)
        )
    }

    fn lower_block(&mut self, block: &Block) -> Vec<StructuredStmt> {
        let (mut stmts, tail) = self.lower_block_value(block);
        if let Some(val) = tail {
            stmts.push(StructuredStmt::Eval(val));
        }
        stmts
    }

    fn lower_block_value(&mut self, block: &Block) -> (Vec<StructuredStmt>, Option<InstId>) {
        let mut out = Vec::new();
        for stmt in &block.stmts {
            self.lower_stmt(stmt, &mut out);
        }
        let tail = block
            .tail_expr
            .as_ref()
            .map(|e| self.lower_expr(e, &mut out));
        (out, tail)
    }

    /// Lowers a structured `if` / `else` with branch-isolated `self.locals`, emitting merge-block
    /// `InstData::Phi` nodes for any pre-existing local mutated inside either branch as well as
    /// for an expression-level `if` result.
    fn lower_if_branches(
        &mut self,
        cond: &Expr,
        then_branch: &Block,
        else_branch: Option<&Expr>,
        want_value: bool,
        out: &mut Vec<StructuredStmt>,
    ) -> Option<InstId> {
        let cond_val = self.lower_expr(cond, out);
        let saved_locals = self.locals.clone();

        let (mut then_body, then_val) = self.lower_block_value(then_branch);
        if !want_value && let Some(v) = then_val {
            then_body.push(StructuredStmt::Eval(v));
        }
        let then_locals = self.locals.clone();

        self.locals.clone_from(&saved_locals);
        let (mut else_body, else_val) = match else_branch {
            Some(Expr::Block(b)) => self.lower_block_value(b),
            Some(other) => {
                let mut eb = Vec::new();
                let v = self.lower_expr(other, &mut eb);
                (eb, Some(v))
            }
            None => (Vec::new(), None),
        };
        if !want_value && let Some(v) = else_val {
            else_body.push(StructuredStmt::Eval(v));
        }
        let else_locals = self.locals.clone();
        self.locals.clone_from(&saved_locals);

        let then_terminates = block_ends_with_terminator(&then_body);
        let else_terminates = block_ends_with_terminator(&else_body);

        out.push(StructuredStmt::If {
            cond: cond_val,
            then_body,
            else_body,
        });

        // Merge mutated pre-existing locals across surviving branches via `OpPhi`.
        if !(then_terminates && else_terminates) {
            for (name, &orig_id) in &saved_locals {
                let t_id = then_locals.get(name).copied().unwrap_or(orig_id);
                let e_id = else_locals.get(name).copied().unwrap_or(orig_id);
                if t_id != e_id {
                    if then_terminates {
                        self.locals.insert(name.clone(), e_id);
                    } else if else_terminates {
                        self.locals.insert(name.clone(), t_id);
                    } else {
                        let phi = self.ir_fn.emit(
                            None,
                            InstData::Phi {
                                incoming: vec![(t_id, 0), (e_id, 1)],
                            },
                        );
                        out.push(StructuredStmt::Eval(phi));
                        self.locals.insert(name.clone(), phi);
                    }
                } else if t_id != orig_id {
                    self.locals.insert(name.clone(), t_id);
                }
            }
        }

        if want_value && let (Some(on_true), Some(on_false)) = (then_val, else_val) {
            if then_terminates && !else_terminates {
                return Some(on_false);
            }
            if else_terminates && !then_terminates {
                return Some(on_true);
            }
            let phi = self.ir_fn.emit(
                None,
                InstData::Phi {
                    incoming: vec![(on_true, 0), (on_false, 1)],
                },
            );
            out.push(StructuredStmt::Eval(phi));
            return Some(phi);
        }

        None
    }

    /// Lowers a `while` or `loop` statement with loop-header `OpPhi` nodes for any pre-existing
    /// local variable mutated inside the loop body, preserving SSA dominance into the merge block.
    fn lower_while_or_loop(
        &mut self,
        attrs: &[Attribute],
        cond: Option<&Expr>,
        body: &Block,
        out: &mut Vec<StructuredStmt>,
    ) {
        let unroll = attrs
            .iter()
            .any(|a| a.entries.iter().any(|e| e.name.name == "unroll"));
        let mut mutated = IndexSet::new();
        collect_assigned_locals_block(body, &mut mutated);

        let saved_locals = self.locals.clone();
        let mut loop_phis: Vec<(SmolStr, InstId, InstId)> = Vec::new();
        for name in mutated {
            if self.workgroup_indices.contains_key(&name) {
                continue;
            }
            if let Some(&init_val) = saved_locals.get(&name) {
                if !self.is_const_inst(init_val) {
                    out.push(StructuredStmt::Eval(init_val));
                }
                let phi_id = self.ir_fn.emit(
                    None,
                    InstData::Phi {
                        incoming: vec![(init_val, 0)],
                    },
                );
                self.locals.insert(name.clone(), phi_id);
                loop_phis.push((name, phi_id, init_val));
            }
        }

        let tracked_locals: Vec<SmolStr> = loop_phis.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_break_stack.push(LoopBreakCapture {
            tracked_locals,
            next_break_tag: 0,
            incoming_by_local: IndexMap::new(),
        });

        let mut loop_body = Vec::new();
        if let Some(cond_expr) = cond {
            let cond_val = self.lower_expr(cond_expr, &mut loop_body);
            self.record_loop_break_snapshot();
            loop_body.push(StructuredStmt::If {
                cond: cond_val,
                then_body: Vec::new(),
                else_body: vec![StructuredStmt::Break],
            });
        }
        let mut inner = self.lower_block(body);
        loop_body.append(&mut inner);

        let break_capture = self.loop_break_stack.pop();

        let mut continuing = Vec::new();
        for (name, phi_id, init_val) in &loop_phis {
            let back_val = self.locals.get(name).copied().unwrap_or(*phi_id);
            if let InstData::Phi { ref mut incoming } = self.ir_fn.insts[*phi_id].data {
                incoming.clear();
                incoming.push((*init_val, 0));
                incoming.push((back_val, 1));
            }
            continuing.push(StructuredStmt::Eval(back_val));
        }

        out.push(StructuredStmt::Loop {
            body: loop_body,
            continuing,
            unroll,
            trip_count: None,
            induction_init: None,
            induction_step: 0,
        });

        self.locals.clone_from(&saved_locals);
        for (name, phi_id, _) in loop_phis {
            let merged = break_capture
                .as_ref()
                .and_then(|cap| cap.incoming_by_local.get(&name))
                .filter(|pairs| !pairs.is_empty() && pairs.iter().any(|&(v, _)| v != phi_id))
                .map_or(phi_id, |pairs| {
                    let merge_phi = self.ir_fn.emit(
                        None,
                        InstData::Phi {
                            incoming: pairs.clone(),
                        },
                    );
                    out.push(StructuredStmt::Eval(merge_phi));
                    merge_phi
                });
            self.locals.insert(name, merged);
        }
    }

    fn record_loop_break_snapshot(&mut self) {
        if let Some(frame) = self.loop_break_stack.last_mut() {
            let tag = frame.next_break_tag;
            frame.next_break_tag += 1;
            for name in &frame.tracked_locals {
                if let Some(&val) = self.locals.get(name) {
                    frame
                        .incoming_by_local
                        .entry(name.clone())
                        .or_default()
                        .push((val, tag));
                }
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt, out: &mut Vec<StructuredStmt>) {
        match stmt {
            Stmt::Let { name, init, .. } => {
                if let Some(expr) = init {
                    let val = self.lower_expr(expr, out);
                    if !self.is_const_inst(val) {
                        out.push(StructuredStmt::Eval(val));
                    }
                    self.locals.insert(name.name.clone(), val);
                } else {
                    let zero = self.ir_fn.emit(None, InstData::ConstU32(0));
                    self.locals.insert(name.name.clone(), zero);
                }
            }
            Stmt::Return { value, .. } => {
                let ret_val = value.as_ref().map(|e| self.lower_expr(e, out));
                out.push(StructuredStmt::Return(ret_val));
            }
            Stmt::Break(_) => {
                self.record_loop_break_snapshot();
                out.push(StructuredStmt::Break);
            }
            Stmt::Continue(_) => out.push(StructuredStmt::Continue),
            Stmt::For {
                attrs,
                binding,
                iter,
                body,
                ..
            } => {
                let unroll = attrs
                    .iter()
                    .any(|a| a.entries.iter().any(|e| e.name.name == "unroll"));
                let saved_locals = self.locals.clone();
                let (range_expr, step_expr) = unwrap_step_by(iter);
                let step_const = step_expr.and_then(parse_const_i64_expr).unwrap_or(1).max(1);
                let (trip_count, induction_init) =
                    extract_const_range_bounds(range_expr, step_const, self.ir_fn);

                // Hoist loop-carried header Phis for any pre-existing outer local mutated inside
                // the `for` body (skipped for statically unrolled loops that flatten into
                // straight-line statements) so SSA dominance holds into the post-loop merge block.
                let mut mutated = IndexSet::new();
                if !(unroll && trip_count.is_some()) {
                    collect_assigned_locals_block(body, &mut mutated);
                }
                let mut outer_phis: Vec<(SmolStr, InstId, InstId)> = Vec::new();
                for name in mutated {
                    if name == binding.name || self.workgroup_indices.contains_key(&name) {
                        continue;
                    }
                    if let Some(&init_val) = saved_locals.get(&name) {
                        if !self.is_const_inst(init_val) {
                            out.push(StructuredStmt::Eval(init_val));
                        }
                        let phi_id = self.ir_fn.emit(
                            None,
                            InstData::Phi {
                                incoming: vec![(init_val, 0)],
                            },
                        );
                        self.locals.insert(name.clone(), phi_id);
                        outer_phis.push((name, phi_id, init_val));
                    }
                }

                let tracked_locals: Vec<SmolStr> =
                    outer_phis.iter().map(|(n, _, _)| n.clone()).collect();
                self.loop_break_stack.push(LoopBreakCapture {
                    tracked_locals,
                    next_break_tag: 0,
                    incoming_by_local: IndexMap::new(),
                });

                let (mut loop_body, mut continuing) = if unroll && trip_count.is_some() {
                    if let Some(init_id) = induction_init {
                        self.locals.insert(binding.name.clone(), init_id);
                    }
                    (self.lower_block(body), Vec::new())
                } else if let Expr::Binary {
                    op: op @ (BinaryOp::Range | BinaryOp::RangeInclusive),
                    lhs,
                    rhs,
                    ..
                } = range_expr
                {
                    let start_val = induction_init.unwrap_or_else(|| self.lower_expr(lhs, out));
                    let end_val = self.lower_expr(rhs, out);
                    let step_val = match step_expr {
                        Some(se) => self.lower_expr(se, out),
                        None => self.ir_fn.emit(None, InstData::ConstI32(1)),
                    };
                    for v in [start_val, end_val, step_val] {
                        if !self.is_const_inst(v) {
                            out.push(StructuredStmt::Eval(v));
                        }
                    }

                    // Create a loop-carried header Phi `(start_val @ preheader, next @ continue)`.
                    let induction_phi = self.ir_fn.emit(
                        None,
                        InstData::Phi {
                            incoming: vec![(start_val, 0)],
                        },
                    );
                    self.locals.insert(binding.name.clone(), induction_phi);

                    let mut body_stmts = Vec::new();
                    let cond = if matches!(op, BinaryOp::RangeInclusive) {
                        self.ir_fn
                            .emit(None, InstData::ICmpLe(induction_phi, end_val))
                    } else {
                        self.ir_fn
                            .emit(None, InstData::ICmpLt(induction_phi, end_val))
                    };
                    self.record_loop_break_snapshot();
                    body_stmts.push(StructuredStmt::If {
                        cond,
                        then_body: Vec::new(),
                        else_body: vec![StructuredStmt::Break],
                    });
                    body_stmts.extend(self.lower_block(body));

                    let next = if self.is_64bit_inst(start_val) || self.is_64bit_inst(step_val) {
                        self.ir_fn
                            .emit(None, InstData::IAdd64(induction_phi, step_val))
                    } else {
                        self.ir_fn
                            .emit(None, InstData::IAdd32(induction_phi, step_val))
                    };
                    if let InstData::Phi { ref mut incoming } = self.ir_fn.insts[induction_phi].data
                    {
                        incoming.push((next, 1));
                    }
                    (body_stmts, vec![StructuredStmt::Eval(next)])
                } else {
                    let iter_val = self.lower_expr(iter, out);
                    self.locals.insert(binding.name.clone(), iter_val);
                    let mut body_stmts = self.lower_block(body);
                    // Ensure non-range loops have a structured exit edge.
                    let done = self.ir_fn.emit(None, InstData::ConstBool(true));
                    self.record_loop_break_snapshot();
                    body_stmts.push(StructuredStmt::If {
                        cond: done,
                        then_body: vec![StructuredStmt::Break],
                        else_body: Vec::new(),
                    });
                    (body_stmts, Vec::new())
                };

                let break_capture = self.loop_break_stack.pop();

                for (name, phi_id, init_val) in &outer_phis {
                    let back_val = self.locals.get(name).copied().unwrap_or(*phi_id);
                    if let InstData::Phi { ref mut incoming } = self.ir_fn.insts[*phi_id].data {
                        incoming.clear();
                        incoming.push((*init_val, 0));
                        incoming.push((back_val, 1));
                    }
                    continuing.push(StructuredStmt::Eval(back_val));
                }

                if loop_body.is_empty() && !unroll {
                    loop_body.push(StructuredStmt::Break);
                }

                out.push(StructuredStmt::Loop {
                    body: loop_body,
                    continuing,
                    unroll,
                    trip_count,
                    induction_init,
                    induction_step: step_const,
                });

                self.locals.clone_from(&saved_locals);
                for (name, phi_id, _) in outer_phis {
                    let merged = break_capture
                        .as_ref()
                        .and_then(|cap| cap.incoming_by_local.get(&name))
                        .filter(|pairs| {
                            !pairs.is_empty() && pairs.iter().any(|&(v, _)| v != phi_id)
                        })
                        .map_or(phi_id, |pairs| {
                            let merge_phi = self.ir_fn.emit(
                                None,
                                InstData::Phi {
                                    incoming: pairs.clone(),
                                },
                            );
                            out.push(StructuredStmt::Eval(merge_phi));
                            merge_phi
                        });
                    self.locals.insert(name, merged);
                }
            }
            Stmt::While {
                attrs, cond, body, ..
            } => {
                self.lower_while_or_loop(attrs, Some(cond), body, out);
            }
            Stmt::Loop { attrs, body, .. } => {
                self.lower_while_or_loop(attrs, None, body, out);
            }
            Stmt::Switch {
                selector,
                cases,
                default,
                ..
            } => {
                let sel_val = self.lower_expr(selector, out);
                let saved_locals = self.locals.clone();
                let mut arm_locals: Vec<(u32, IndexMap<SmolStr, InstId>, bool)> =
                    Vec::with_capacity(cases.len() + 1);
                let mut lowered_cases = Vec::with_capacity(cases.len());

                for (idx, c) in cases.iter().enumerate() {
                    self.locals.clone_from(&saved_locals);
                    let key = parse_const_i64_expr(&c.value).unwrap_or(0);
                    let body_stmts = self.lower_block(&c.body);
                    let terminates = block_ends_with_terminator(&body_stmts);
                    arm_locals.push((idx as u32, self.locals.clone(), terminates));
                    lowered_cases.push((key, body_stmts));
                }

                self.locals.clone_from(&saved_locals);
                let lowered_default = default
                    .as_ref()
                    .map_or_else(Vec::new, |b| self.lower_block(b));
                let default_terminates = block_ends_with_terminator(&lowered_default);
                arm_locals.push((cases.len() as u32, self.locals.clone(), default_terminates));
                self.locals.clone_from(&saved_locals);

                out.push(StructuredStmt::Switch {
                    selector: sel_val,
                    cases: lowered_cases,
                    default: lowered_default,
                });

                // Emit merge-block `OpPhi` for any pre-existing local mutated in surviving arms.
                let surviving: Vec<&(u32, IndexMap<SmolStr, InstId>, bool)> =
                    arm_locals.iter().filter(|(_, _, term)| !*term).collect();
                if !surviving.is_empty() {
                    for (name, &orig_id) in &saved_locals {
                        let incoming: Vec<(InstId, u32)> = surviving
                            .iter()
                            .map(|(tag, map, _)| (map.get(name).copied().unwrap_or(orig_id), *tag))
                            .collect();
                        let first_val = incoming[0].0;
                        let all_same = incoming.iter().all(|&(v, _)| v == first_val);
                        if all_same {
                            if first_val != orig_id {
                                self.locals.insert(name.clone(), first_val);
                            }
                        } else {
                            let phi = self.ir_fn.emit(None, InstData::Phi { incoming });
                            out.push(StructuredStmt::Eval(phi));
                            self.locals.insert(name.clone(), phi);
                        }
                    }
                }
            }
            Stmt::Expr { expr, .. } => {
                if let Expr::If {
                    cond,
                    then_branch,
                    else_branch,
                    ..
                } = expr
                {
                    let _ = self.lower_if_branches(
                        cond,
                        then_branch,
                        else_branch.as_deref(),
                        false,
                        out,
                    );
                } else {
                    let val = self.lower_expr(expr, out);
                    out.push(StructuredStmt::Eval(val));
                }
            }
        }
    }

    fn lower_binary_op(&mut self, op: BinaryOp, l: InstId, r: InstId) -> InstId {
        let is_float = self.is_float_inst(l) || self.is_float_inst(r);
        let is_64 = self.is_64bit_inst(l) || self.is_64bit_inst(r);
        let is_u32 = self.is_u32_inst(l) && self.is_u32_inst(r);

        match op {
            BinaryOp::Add | BinaryOp::AddAssign => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FAdd32(l, r))
                } else if is_64 {
                    self.ir_fn.emit(None, InstData::IAdd64(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::IAdd32(l, r))
                }
            }
            BinaryOp::Sub | BinaryOp::SubAssign => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FSub32(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::ISub32(l, r))
                }
            }
            BinaryOp::Mul | BinaryOp::MulAssign => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FMul32(l, r))
                } else if is_64 {
                    self.ir_fn.emit(None, InstData::IMul64(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::IMul32(l, r))
                }
            }
            BinaryOp::Div | BinaryOp::DivAssign => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FDiv32(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::IDiv32(l, r))
                }
            }
            BinaryOp::Rem | BinaryOp::RemAssign => self.ir_fn.emit(None, InstData::IRem32(l, r)),
            BinaryOp::BitAnd | BinaryOp::BitAndAssign => {
                self.ir_fn.emit(None, InstData::BitAnd32(l, r))
            }
            BinaryOp::BitOr | BinaryOp::BitOrAssign => {
                self.ir_fn.emit(None, InstData::BitOr32(l, r))
            }
            BinaryOp::BitXor | BinaryOp::BitXorAssign => {
                self.ir_fn.emit(None, InstData::BitXor32(l, r))
            }
            BinaryOp::Shl => self.ir_fn.emit(None, InstData::Shl32(l, r)),
            BinaryOp::Shr => self.ir_fn.emit(None, InstData::Shr32(l, r)),
            BinaryOp::Eq => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FCmpEq(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::ICmpEq(l, r))
                }
            }
            BinaryOp::Ne => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FCmpNe(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::ICmpNe(l, r))
                }
            }
            BinaryOp::Lt => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FCmpLt(l, r))
                } else if is_u32 {
                    self.ir_fn.emit(None, InstData::UCmpLt(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::ICmpLt(l, r))
                }
            }
            BinaryOp::Le => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FCmpLe(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::ICmpLe(l, r))
                }
            }
            BinaryOp::Gt => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FCmpGt(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::ICmpGt(l, r))
                }
            }
            BinaryOp::Ge => {
                if is_float {
                    self.ir_fn.emit(None, InstData::FCmpGe(l, r))
                } else {
                    self.ir_fn.emit(None, InstData::ICmpGe(l, r))
                }
            }
            BinaryOp::And => self.ir_fn.emit(None, InstData::LogicalAnd(l, r)),
            BinaryOp::Or => self.ir_fn.emit(None, InstData::LogicalOr(l, r)),
            BinaryOp::Assign | BinaryOp::Range | BinaryOp::RangeInclusive => r,
        }
    }

    fn lower_expr(&mut self, expr: &Expr, out: &mut Vec<StructuredStmt>) -> InstId {
        match expr {
            Expr::BoolLit(b, _) => self.ir_fn.emit(None, InstData::ConstBool(*b)),
            Expr::IntLit(raw, _) => lower_int_literal(raw, self.ir_fn),
            Expr::FloatLit(raw, _) => {
                let val = raw
                    .trim_end_matches("f32")
                    .trim_end_matches("f16")
                    .trim_end_matches('f')
                    .replace('_', "")
                    .parse::<f32>()
                    .unwrap_or(0.0);
                self.ir_fn.emit(None, InstData::const_f32(val))
            }
            Expr::Path { path, .. } => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].name;
                    if let Some(&id) = self.locals.get(name) {
                        return id;
                    }
                    if let Some(&var_index) = self.workgroup_indices.get(name) {
                        return self.ir_fn.emit(
                            None,
                            InstData::WorkgroupLoad {
                                var_index,
                                index: None,
                            },
                        );
                    }
                    if let Some(const_expr) = self.module_consts.get(name).cloned() {
                        return self.lower_expr(&const_expr, out);
                    }
                }
                let joined = path.joined();
                if let Some(spec_name) = joined
                    .strip_prefix("mc::spec::")
                    .or_else(|| joined.strip_prefix("spec::"))
                {
                    let (spec_id, scalar) = self
                        .spec_ids
                        .get(spec_name)
                        .copied()
                        .unwrap_or((0, ScalarIrTy::U32));
                    return self.ir_fn.emit(
                        None,
                        InstData::SpecConstant {
                            spec_id,
                            scalar,
                            default_bits: 0,
                        },
                    );
                }
                match joined.as_str() {
                    "mc::HitType::Triangle" | "mc::RayFlags::OPAQUE" => {
                        return self.ir_fn.emit(None, InstData::ConstU32(1));
                    }
                    "mc::HitType::None" => return self.ir_fn.emit(None, InstData::ConstU32(0)),
                    _ => {}
                }
                self.ir_fn.emit(None, InstData::ConstU32(0))
            }
            Expr::Unary { op, operand, .. } => {
                // `&workgroup_var` and `&workgroup_var[idx]` produce a `StorageClass::Workgroup`
                // pointer (`WorkgroupAddr`) rather than loading the scalar value.
                if matches!(op, UnaryOp::AddrOf) {
                    match operand.as_ref() {
                        Expr::Path { path, .. }
                            if path.segments.len() == 1
                                && self.workgroup_indices.contains_key(&path.segments[0].name) =>
                        {
                            let var_index = self.workgroup_indices[&path.segments[0].name];
                            return self.ir_fn.emit(
                                None,
                                InstData::WorkgroupAddr {
                                    var_index,
                                    index: None,
                                },
                            );
                        }
                        Expr::Index {
                            receiver, index, ..
                        } => {
                            if let Expr::Path { path, .. } = receiver.as_ref()
                                && path.segments.len() == 1
                                && let Some(&var_index) =
                                    self.workgroup_indices.get(&path.segments[0].name)
                            {
                                let idx_val = self.lower_expr(index, out);
                                return self.ir_fn.emit(
                                    None,
                                    InstData::WorkgroupAddr {
                                        var_index,
                                        index: Some(idx_val),
                                    },
                                );
                            }
                        }
                        _ => {}
                    }
                }

                let val = self.lower_expr(operand, out);
                match op {
                    UnaryOp::Neg => {
                        if self.is_float_inst(val) {
                            let zero = self.ir_fn.emit(None, InstData::const_f32(0.0));
                            self.ir_fn.emit(None, InstData::FSub32(zero, val))
                        } else {
                            let zero = self.ir_fn.emit(None, InstData::ConstI32(0));
                            self.ir_fn.emit(None, InstData::ISub32(zero, val))
                        }
                    }
                    UnaryOp::Not => self.ir_fn.emit(None, InstData::LogicalNot(val)),
                    UnaryOp::BitNot => self.ir_fn.emit(None, InstData::BitNot32(val)),
                    UnaryOp::AddrOf => val,
                }
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                if matches!(
                    op,
                    BinaryOp::Assign
                        | BinaryOp::AddAssign
                        | BinaryOp::SubAssign
                        | BinaryOp::MulAssign
                        | BinaryOp::DivAssign
                        | BinaryOp::RemAssign
                        | BinaryOp::BitAndAssign
                        | BinaryOp::BitOrAssign
                        | BinaryOp::BitXorAssign
                ) {
                    let r = self.lower_expr(rhs, out);
                    let final_val = if *op == BinaryOp::Assign {
                        r
                    } else {
                        let l = self.lower_expr(lhs, out);
                        self.lower_binary_op(*op, l, r)
                    };

                    match lhs.as_ref() {
                        Expr::Path { path, .. } if path.segments.len() == 1 => {
                            let name = &path.segments[0].name;
                            if let Some(&var_index) = self.workgroup_indices.get(name) {
                                let store = self.ir_fn.emit(
                                    None,
                                    InstData::WorkgroupStore {
                                        var_index,
                                        index: None,
                                        value: final_val,
                                    },
                                );
                                out.push(StructuredStmt::Eval(store));
                            } else {
                                self.locals.insert(name.clone(), final_val);
                            }
                        }
                        Expr::Index {
                            receiver, index, ..
                        } => {
                            if let Expr::Path { path, .. } = receiver.as_ref()
                                && path.segments.len() == 1
                                && let Some(&var_index) =
                                    self.workgroup_indices.get(&path.segments[0].name)
                            {
                                let idx_val = self.lower_expr(index, out);
                                let store = self.ir_fn.emit(
                                    None,
                                    InstData::WorkgroupStore {
                                        var_index,
                                        index: Some(idx_val),
                                        value: final_val,
                                    },
                                );
                                out.push(StructuredStmt::Eval(store));
                            }
                        }
                        _ => {}
                    }
                    return final_val;
                }

                let l = self.lower_expr(lhs, out);
                let r = self.lower_expr(rhs, out);
                self.lower_binary_op(*op, l, r)
            }
            Expr::Call { callee, args, .. } => {
                if let Expr::Path {
                    path, turbofish, ..
                } = callee.as_ref()
                {
                    let joined = path.joined();
                    match joined.as_str() {
                        "u32" => {
                            let inner = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            return self.ir_fn.emit(None, InstData::CastToU32(inner));
                        }
                        "i32" => {
                            let inner = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstI32(0)),
                            };
                            return self.ir_fn.emit(None, InstData::CastToI32(inner));
                        }
                        "f32" => {
                            let inner = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::const_f32(0.0)),
                            };
                            return self.ir_fn.emit(None, InstData::CastToF32(inner));
                        }
                        "i64" => {
                            let inner = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstI64(0)),
                            };
                            return self.ir_fn.emit(None, InstData::CastToI64(inner));
                        }
                        "u64" => {
                            let inner = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU64(0)),
                            };
                            return self.ir_fn.emit(None, InstData::CastToU64(inner));
                        }
                        "ivec2" | "uvec2" | "vec2" if args.len() == 2 => {
                            let x = self.lower_expr(&args[0].value, out);
                            let y = self.lower_expr(&args[1].value, out);
                            return self.ir_fn.emit(None, InstData::MakeVec2(x, y));
                        }
                        "ivec3" | "uvec3" | "vec3" if args.len() == 3 => {
                            let x = self.lower_expr(&args[0].value, out);
                            let y = self.lower_expr(&args[1].value, out);
                            let z = self.lower_expr(&args[2].value, out);
                            return self.ir_fn.emit(None, InstData::MakeVec3(x, y, z));
                        }
                        "ivec4" | "uvec4" | "vec4" if args.len() == 4 => {
                            let x = self.lower_expr(&args[0].value, out);
                            let y = self.lower_expr(&args[1].value, out);
                            let z = self.lower_expr(&args[2].value, out);
                            let w = self.lower_expr(&args[3].value, out);
                            return self.ir_fn.emit(None, InstData::MakeVec4(x, y, z, w));
                        }
                        "mc::spec" | "spec" => {
                            let name = match args.first().map(|a| &a.value) {
                                Some(Expr::StringLit(s, _)) => s.clone(),
                                Some(Expr::Path { path: p, .. }) => SmolStr::new(p.joined()),
                                _ => SmolStr::new_static("spec"),
                            };
                            let (spec_id, scalar) = self
                                .spec_ids
                                .get(&name)
                                .copied()
                                .unwrap_or((0, ScalarIrTy::U32));
                            return self.ir_fn.emit(
                                None,
                                InstData::SpecConstant {
                                    spec_id,
                                    scalar,
                                    default_bits: 0,
                                },
                            );
                        }
                        "mc::param" | "param" => {
                            let name = match args.first().map(|a| &a.value) {
                                Some(Expr::StringLit(s, _)) => s.clone(),
                                Some(Expr::Path { path: p, .. }) => SmolStr::new(p.joined()),
                                _ => SmolStr::new_static("param"),
                            };
                            let byte_offset = self.param_offsets.get(&name).copied().unwrap_or(0);
                            let Some(&world_desc) = self.descriptor_indices.get("world") else {
                                return self.ir_fn.emit(None, InstData::const_f32(0.0));
                            };
                            let base_u64 = self.ir_fn.emit(
                                None,
                                InstData::DescriptorLoad {
                                    descriptor_index: world_desc,
                                    word_index: 0,
                                    scalar: ScalarIrTy::U64,
                                },
                            );
                            let off_u64 = self
                                .ir_fn
                                .emit(None, InstData::ConstU64(u64::from(byte_offset)));
                            let addr = self.ir_fn.emit(None, InstData::IAdd64(base_u64, off_u64));
                            let ptr = self.ir_fn.emit(
                                None,
                                InstData::ConvertUToPtr {
                                    addr_u64: addr,
                                    align: 4,
                                },
                            );
                            self.ir_fn.uses_bda = true;
                            return self.ir_fn.emit(
                                None,
                                InstData::BdaLoad {
                                    ptr,
                                    access: MemoryAccess {
                                        align: 4,
                                        non_writable: true,
                                        restrict: false,
                                    },
                                },
                            );
                        }
                        "sin" | "cos" | "sqrt" | "abs" | "floor" | "ceil" | "fract" | "exp"
                        | "log" | "pow" | "min" | "max" | "clamp" | "mix" | "fma" | "dot"
                        | "length" | "cross" | "normalize" | "reflect" => {
                            let lowered_args: Vec<InstId> = args
                                .iter()
                                .map(|a| self.lower_expr(&a.value, out))
                                .collect();
                            let any_float = lowered_args.iter().any(|&a| self.is_float_inst(a));
                            let all_u32 = !lowered_args.is_empty()
                                && lowered_args.iter().all(|&a| self.is_u32_inst(a));
                            let op = match joined.as_str() {
                                "sin" => GlslStd450Op::Sin,
                                "cos" => GlslStd450Op::Cos,
                                "sqrt" => GlslStd450Op::Sqrt,
                                "floor" => GlslStd450Op::Floor,
                                "ceil" => GlslStd450Op::Ceil,
                                "fract" => GlslStd450Op::Fract,
                                "exp" => GlslStd450Op::Exp,
                                "log" => GlslStd450Op::Log,
                                "pow" => GlslStd450Op::Pow,
                                "mix" => GlslStd450Op::FMix,
                                "fma" => GlslStd450Op::Fma,
                                "dot" => GlslStd450Op::Dot,
                                "length" => GlslStd450Op::Length,
                                "cross" => GlslStd450Op::Cross,
                                "normalize" => GlslStd450Op::Normalize,
                                "reflect" => GlslStd450Op::Reflect,
                                "abs" => {
                                    if any_float {
                                        GlslStd450Op::FAbs
                                    } else {
                                        GlslStd450Op::SAbs
                                    }
                                }
                                "min" => {
                                    if any_float {
                                        GlslStd450Op::FMin
                                    } else if all_u32 {
                                        GlslStd450Op::UMin
                                    } else {
                                        GlslStd450Op::SMin
                                    }
                                }
                                "max" => {
                                    if any_float {
                                        GlslStd450Op::FMax
                                    } else if all_u32 {
                                        GlslStd450Op::UMax
                                    } else {
                                        GlslStd450Op::SMax
                                    }
                                }
                                "clamp" => {
                                    if any_float {
                                        GlslStd450Op::FClamp
                                    } else if all_u32 {
                                        GlslStd450Op::UClamp
                                    } else {
                                        GlslStd450Op::SClamp
                                    }
                                }
                                _ => unreachable!(),
                            };
                            return self.ir_fn.emit(
                                None,
                                InstData::GlslStd450 {
                                    op,
                                    args: lowered_args,
                                },
                            );
                        }
                        "subgroup_ballot_bit_count"
                        | "std::subgroup::subgroup_ballot_bit_count" => {
                            let ballot = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            self.ir_fn.uses_subgroups = true;
                            let id = self
                                .ir_fn
                                .emit(None, InstData::SubgroupBallotBitCount(ballot));
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "subgroup_broadcast_first" | "std::subgroup::subgroup_broadcast_first" => {
                            let val = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            self.ir_fn.uses_subgroups = true;
                            let id = self.ir_fn.emit(None, InstData::SubgroupBroadcastFirst(val));
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "atomic_min" | "std::sync::atomic_min" => {
                            let (scope, order) = parse_atomic_turbofish(turbofish);
                            let ptr = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU64(0)),
                            };
                            let val = match args.get(1) {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            let id = self.ir_fn.emit(
                                None,
                                InstData::AtomicMin {
                                    ptr,
                                    value: val,
                                    scope,
                                    order,
                                },
                            );
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "atomic_max" | "std::sync::atomic_max" => {
                            let (scope, order) = parse_atomic_turbofish(turbofish);
                            let ptr = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU64(0)),
                            };
                            let val = match args.get(1) {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            let id = self.ir_fn.emit(
                                None,
                                InstData::AtomicMax {
                                    ptr,
                                    value: val,
                                    scope,
                                    order,
                                },
                            );
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "atomic_cas"
                        | "atomic_compare_exchange"
                        | "std::sync::atomic_cas"
                        | "std::sync::atomic_compare_exchange" => {
                            let (scope, order) = parse_atomic_turbofish(turbofish);
                            let ptr = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU64(0)),
                            };
                            let expected = match args.get(1) {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            let desired = match args.get(2) {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            let id = self.ir_fn.emit(
                                None,
                                InstData::AtomicCas {
                                    ptr,
                                    expected,
                                    desired,
                                    scope,
                                    order,
                                },
                            );
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "workgroup_barrier" | "std::sync::workgroup_barrier" => {
                            let id = self.ir_fn.emit(None, InstData::WorkgroupBarrier);
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "nonuniform" => {
                            let inner = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                            };
                            return self.ir_fn.emit(None, InstData::NonUniform(inner));
                        }
                        "subgroup_ballot" | "std::subgroup::subgroup_ballot" => {
                            let pred = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstBool(true)),
                            };
                            self.ir_fn.uses_subgroups = true;
                            let id = self.ir_fn.emit(None, InstData::SubgroupBallot(pred));
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "atomic_add" | "std::sync::atomic_add" => {
                            let (scope, order) = parse_atomic_turbofish(turbofish);
                            let ptr = match args.first() {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU64(0)),
                            };
                            let val = match args.get(1) {
                                Some(a) => self.lower_expr(&a.value, out),
                                None => self.ir_fn.emit(None, InstData::ConstU32(1)),
                            };
                            let id = self.ir_fn.emit(
                                None,
                                InstData::AtomicAdd {
                                    ptr,
                                    value: val,
                                    scope,
                                    order,
                                },
                            );
                            out.push(StructuredStmt::Eval(id));
                            return id;
                        }
                        "mc::debug_write" | "debug_write" => {
                            if let Some(Expr::StringLit(channel, _)) =
                                args.first().map(|a| &a.value)
                            {
                                let coord = match args.get(1) {
                                    Some(a) => self.lower_expr(&a.value, out),
                                    None => self.ir_fn.emit(None, InstData::ConstI32(0)),
                                };
                                let value = match args.get(2) {
                                    Some(a) => self.lower_expr(&a.value, out),
                                    None => self.ir_fn.emit(None, InstData::const_f32(0.0)),
                                };
                                let desc_name = SmolStr::new(format!("debug_{channel}"));
                                if let Some(&descriptor_index) =
                                    self.storage_image_descriptors.get(&desc_name)
                                {
                                    let id = self.ir_fn.emit(
                                        None,
                                        InstData::ImageStore {
                                            descriptor_index,
                                            coord,
                                            value,
                                        },
                                    );
                                    out.push(StructuredStmt::Eval(id));
                                    return id;
                                }
                                return value;
                            }
                        }
                        "mc::sample_sky" => {
                            for arg in args {
                                let _ = self.lower_expr(&arg.value, out);
                            }
                            return self.ir_fn.emit(None, InstData::const_f32(0.5));
                        }
                        _ => {
                            let short_name = path.segments.last().map_or("", |s| s.name.as_str());
                            if self.inline_depth < 8
                                && let Some(helper) = self.helper_fns.get(short_name).cloned()
                                && let Some(ref body) = helper.body
                                && !block_has_non_tail_return(body)
                            {
                                let lowered_args: Vec<InstId> = args
                                    .iter()
                                    .map(|a| self.lower_expr(&a.value, out))
                                    .collect();
                                let saved_locals = std::mem::take(&mut self.locals);
                                self.inline_depth += 1;
                                for (param, val) in helper.params.iter().zip(lowered_args) {
                                    self.locals.insert(param.name.name.clone(), val);
                                }
                                let (mut inlined_stmts, tail) = self.lower_block_value(body);
                                let mut ret_val = tail;
                                // Convert a trailing `Return(Some(v))` inside the inlined helper
                                // into the inlined expression result rather than returning from
                                // the caller entry point.
                                if let Some(StructuredStmt::Return(opt)) = inlined_stmts.last() {
                                    if ret_val.is_none() {
                                        ret_val = *opt;
                                    }
                                    inlined_stmts.pop();
                                }
                                out.append(&mut inlined_stmts);
                                self.inline_depth -= 1;
                                self.locals = saved_locals;
                                return ret_val.unwrap_or_else(|| {
                                    self.ir_fn.emit(None, InstData::ConstU32(0))
                                });
                            }
                        }
                    }
                }
                let mut lowered = Vec::with_capacity(args.len());
                for arg in args {
                    lowered.push(self.lower_expr(&arg.value, out));
                }
                // Preserve primary non-constant dataflow (e.g. `vec4(color, 1.0)` -> `color`)
                // so downstream `ImageStore` / `BdaStore` keeps upstream loads live through DCE.
                lowered
                    .iter()
                    .copied()
                    .find(|&id| !self.is_const_inst(id))
                    .or_else(|| lowered.first().copied())
                    .unwrap_or_else(|| self.ir_fn.emit(None, InstData::ConstU32(0)))
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                // `RayQuery` methods lower to native `SPV_KHR_ray_query` IR instructions.
                match method.name.as_str() {
                    "initialize" if args.len() >= 7 => {
                        let tlas_descriptor_index =
                            args.first()
                                .and_then(|a| match &a.value {
                                    Expr::Path { path, .. } => path.segments.last().and_then(|s| {
                                        self.descriptor_indices.get(&s.name).copied()
                                    }),
                                    _ => None,
                                })
                                .or(self.tlas_descriptor_index)
                                .unwrap_or(0);
                        let flags = self.lower_expr(&args[1].value, out);
                        let cull_mask = self.lower_expr(&args[2].value, out);
                        let origin = self.lower_expr(&args[3].value, out);
                        let t_min = self.lower_expr(&args[4].value, out);
                        let dir = self.lower_expr(&args[5].value, out);
                        let t_max = self.lower_expr(&args[6].value, out);
                        let ray_query_id = self.ray_query_id_for_receiver(receiver);
                        let id = self.ir_fn.emit(
                            None,
                            InstData::RayQueryInitialize {
                                ray_query_id,
                                tlas_descriptor_index,
                                flags,
                                cull_mask,
                                origin,
                                t_min,
                                dir,
                                t_max,
                            },
                        );
                        out.push(StructuredStmt::Eval(id));
                        return id;
                    }
                    "proceed" => {
                        let ray_query_id = self.ray_query_id_for_receiver(receiver);
                        let id = self
                            .ir_fn
                            .emit(None, InstData::RayQueryProceed(ray_query_id));
                        out.push(StructuredStmt::Eval(id));
                        return id;
                    }
                    "committed_type" => {
                        let ray_query_id = self.ray_query_id_for_receiver(receiver);
                        return self
                            .ir_fn
                            .emit(None, InstData::RayQueryCommittedType(ray_query_id));
                    }
                    "committed_instance_custom_index" => {
                        let ray_query_id = self.ray_query_id_for_receiver(receiver);
                        return self.ir_fn.emit(
                            None,
                            InstData::RayQueryCommittedInstanceCustomIndex(ray_query_id),
                        );
                    }
                    "committed_barycentrics" => {
                        let ray_query_id = self.ray_query_id_for_receiver(receiver);
                        return self
                            .ir_fn
                            .emit(None, InstData::RayQueryCommittedBarycentrics(ray_query_id));
                    }
                    _ => {}
                }

                // Storage-image `.store(coord, value)` emits a side-effecting `ImageStore`.
                if method.name == "store"
                    && let Expr::Path { path, .. } = receiver.as_ref()
                    && let Some(last_seg) = path.segments.last()
                    && let Some(&descriptor_index) =
                        self.storage_image_descriptors.get(&last_seg.name)
                {
                    let coord = match args.first() {
                        Some(a) => self.lower_expr(&a.value, out),
                        None => self.ir_fn.emit(None, InstData::ConstI32(0)),
                    };
                    let value = match args.get(1) {
                        Some(a) => self.lower_expr(&a.value, out),
                        None => self.ir_fn.emit(None, InstData::const_f32(0.0)),
                    };
                    let id = self.ir_fn.emit(
                        None,
                        InstData::ImageStore {
                            descriptor_index,
                            coord,
                            value,
                        },
                    );
                    out.push(StructuredStmt::Eval(id));
                    return id;
                }

                // Texture `.sample(...)`, `.sample_lod(...)`, and `.fetch(...)` lower to
                // `ImageSample` returning an `f32` value (preserving `ResourceLoad` + `NonUniform`
                // on bindless arrays through `image_inst: Some(load)`).
                if matches!(method.name.as_str(), "sample" | "sample_lod" | "fetch") {
                    let is_fetch = method.name == "fetch";
                    let coord = match args.first() {
                        Some(a) => self.lower_expr(&a.value, out),
                        None => self.ir_fn.emit(None, InstData::ConstI32(0)),
                    };
                    let lod = match args.get(1) {
                        Some(a) => self.lower_expr(&a.value, out),
                        None => self.ir_fn.emit(None, InstData::const_f32(0.0)),
                    };
                    match receiver.as_ref() {
                        Expr::Path { path, .. } => {
                            let key = path.segments.last().map_or("", |s| s.name.as_str());
                            if let Some(&descriptor_index) = self.descriptor_indices.get(key) {
                                return self.ir_fn.emit(
                                    None,
                                    InstData::ImageSample {
                                        descriptor_index,
                                        image_inst: None,
                                        coord,
                                        lod,
                                        is_fetch,
                                    },
                                );
                            }
                        }
                        Expr::Index {
                            receiver: arr_recv,
                            index,
                            ..
                        } => {
                            let idx_val = self.lower_expr(index, out);
                            if let Expr::Path { path, .. } = arr_recv.as_ref() {
                                let key = path.segments.last().map_or("", |s| s.name.as_str());
                                if let Some(&descriptor_index) = self.descriptor_indices.get(key) {
                                    let chain = self.ir_fn.emit(
                                        None,
                                        InstData::ResourceAccessChain {
                                            descriptor_index,
                                            index: idx_val,
                                        },
                                    );
                                    let image_load = self
                                        .ir_fn
                                        .emit(None, InstData::ResourceLoad { ptr: chain });
                                    return self.ir_fn.emit(
                                        None,
                                        InstData::ImageSample {
                                            descriptor_index,
                                            image_inst: Some(image_load),
                                            coord,
                                            lod,
                                            is_fetch,
                                        },
                                    );
                                }
                            }
                            return self.ir_fn.emit(None, InstData::const_f32(0.0));
                        }
                        _ => {}
                    }
                    let _ = self.lower_expr(receiver, out);
                    return self.ir_fn.emit(None, InstData::const_f32(0.0));
                }

                let recv = self.lower_expr(receiver, out);
                match method.name.as_str() {
                    "offset" => {
                        let idx = match args.first() {
                            Some(a) => self.lower_expr(&a.value, out),
                            None => self.ir_fn.emit(None, InstData::ConstI64(0)),
                        };
                        let (stride_bytes, align) = match receiver.as_ref() {
                            Expr::Path { path, .. } if path.segments.len() == 1 => self
                                .ptr_meta
                                .get(&path.segments[0].name)
                                .copied()
                                .unwrap_or((4, 4)),
                            _ => (4, 4),
                        };
                        self.ir_fn.uses_bda = true;
                        let off = self.ir_fn.emit(
                            None,
                            InstData::BdaOffset {
                                base_ptr: recv,
                                index_i64: idx,
                                stride_bytes,
                                align,
                            },
                        );
                        out.push(StructuredStmt::Eval(off));
                        off
                    }
                    "store" => {
                        let val = match args.last() {
                            Some(a) => self.lower_expr(&a.value, out),
                            None => self.ir_fn.emit(None, InstData::ConstU32(0)),
                        };
                        if self.ir_fn.uses_bda {
                            let id = self.ir_fn.emit(
                                None,
                                InstData::BdaStore {
                                    ptr: recv,
                                    value: val,
                                    access: MemoryAccess {
                                        align: 4,
                                        non_writable: false,
                                        restrict: false,
                                    },
                                },
                            );
                            out.push(StructuredStmt::Eval(id));
                            id
                        } else {
                            val
                        }
                    }
                    _ => {
                        for arg in args {
                            let _ = self.lower_expr(&arg.value, out);
                        }
                        recv
                    }
                }
            }
            Expr::Index {
                receiver, index, ..
            } => {
                let idx_val = self.lower_expr(index, out);
                if let Expr::Path { path, .. } = receiver.as_ref() {
                    let key = path.segments.last().map_or("", |s| s.name.as_str());
                    if let Some(&desc_idx) = self.descriptor_indices.get(key) {
                        let chain = self.ir_fn.emit(
                            None,
                            InstData::ResourceAccessChain {
                                descriptor_index: desc_idx,
                                index: idx_val,
                            },
                        );
                        let load = self.ir_fn.emit(None, InstData::ResourceLoad { ptr: chain });
                        out.push(StructuredStmt::Eval(chain));
                        out.push(StructuredStmt::Eval(load));
                        return load;
                    }
                    if path.segments.len() == 1
                        && let Some(&var_index) = self.workgroup_indices.get(key)
                    {
                        return self.ir_fn.emit(
                            None,
                            InstData::WorkgroupLoad {
                                var_index,
                                index: Some(idx_val),
                            },
                        );
                    }
                }
                let _ = self.lower_expr(receiver, out);
                idx_val
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => self
                .lower_if_branches(cond, then_branch, else_branch.as_deref(), true, out)
                .unwrap_or_else(|| self.ir_fn.emit(None, InstData::ConstU32(0))),
            Expr::Block(b) => {
                let (mut stmts, tail) = self.lower_block_value(b);
                out.append(&mut stmts);
                tail.unwrap_or_else(|| self.ir_fn.emit(None, InstData::ConstU32(0)))
            }
            Expr::Field {
                receiver, field, ..
            } => {
                if let Expr::Path { path, .. } = receiver.as_ref()
                    && path.segments.len() == 1
                {
                    let base = &path.segments[0].name;
                    if let Some(&kind) = self.builtin_params.get(base)
                        && let Some((c0, c1)) = parse_swizzle_components(field.name.as_str())
                    {
                        if let Some(comp1) = c1 {
                            let x = self.ir_fn.emit(
                                None,
                                InstData::LoadBuiltin {
                                    kind,
                                    component: Some(c0),
                                },
                            );
                            let y = self.ir_fn.emit(
                                None,
                                InstData::LoadBuiltin {
                                    kind,
                                    component: Some(comp1),
                                },
                            );
                            return self.ir_fn.emit(None, InstData::MakeVec2(x, y));
                        }
                        return self.ir_fn.emit(
                            None,
                            InstData::LoadBuiltin {
                                kind,
                                component: Some(c0),
                            },
                        );
                    }
                    if self.push_constant_param.as_ref() == Some(base)
                        && let Some(&field_index) = self.push_constant_fields.get(&field.name)
                    {
                        return self
                            .ir_fn
                            .emit(None, InstData::LoadPushConstant { field_index });
                    }
                }
                if let Expr::Path { path, .. } = receiver.as_ref()
                    && let Some(last) = path.segments.last()
                    && let Some(&descriptor_index) = self.descriptor_indices.get(&last.name)
                {
                    let word_index = 0u32;
                    let scalar = if field.name.as_str() == "pack_params_bda" {
                        ScalarIrTy::U64
                    } else {
                        ScalarIrTy::U32
                    };
                    return self.ir_fn.emit(
                        None,
                        InstData::DescriptorLoad {
                            descriptor_index,
                            word_index,
                            scalar,
                        },
                    );
                }
                let recv_id = self.lower_expr(receiver, out);
                if let Some((comp, None)) = parse_swizzle_components(field.name.as_str())
                    && matches!(
                        self.ir_fn.insts[recv_id].data,
                        InstData::MakeVec2(_, _)
                            | InstData::MakeVec3(_, _, _)
                            | InstData::MakeVec4(_, _, _, _)
                    )
                {
                    return self.ir_fn.emit(
                        None,
                        InstData::ExtractComponent {
                            vec: recv_id,
                            index: comp,
                        },
                    );
                }
                recv_id
            }
            _ => self.ir_fn.emit(None, InstData::ConstU32(0)),
        }
    }
}

fn stmt_contains_return(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return { .. } => true,
        Stmt::For { body, .. } | Stmt::While { body, .. } | Stmt::Loop { body, .. } => {
            body.stmts.iter().any(stmt_contains_return)
        }
        Stmt::Switch { cases, default, .. } => {
            cases
                .iter()
                .any(|c| c.body.stmts.iter().any(stmt_contains_return))
                || default
                    .as_ref()
                    .is_some_and(|d| d.stmts.iter().any(stmt_contains_return))
        }
        Stmt::Expr { expr, .. } => expr_contains_return(expr),
        _ => false,
    }
}

fn expr_contains_return(expr: &Expr) -> bool {
    match expr {
        Expr::Block(b) => b.stmts.iter().any(stmt_contains_return),
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            then_branch.stmts.iter().any(stmt_contains_return)
                || else_branch.as_deref().is_some_and(expr_contains_return)
        }
        _ => false,
    }
}

fn block_has_non_tail_return(block: &Block) -> bool {
    let last_idx = block.stmts.len().saturating_sub(1);
    block.stmts.iter().enumerate().any(|(idx, s)| {
        if idx == last_idx && matches!(s, Stmt::Return { .. }) {
            false
        } else {
            stmt_contains_return(s)
        }
    })
}

fn collect_assigned_locals_block(block: &Block, out: &mut IndexSet<SmolStr>) {
    for stmt in &block.stmts {
        collect_assigned_locals_stmt(stmt, out);
    }
    if let Some(ref tail) = block.tail_expr {
        collect_assigned_locals_expr(tail, out);
    }
}

fn collect_assigned_locals_stmt(stmt: &Stmt, out: &mut IndexSet<SmolStr>) {
    match stmt {
        Stmt::Let { init: Some(e), .. } | Stmt::Return { value: Some(e), .. } => {
            collect_assigned_locals_expr(e, out);
        }
        Stmt::For { iter, body, .. } => {
            collect_assigned_locals_expr(iter, out);
            collect_assigned_locals_block(body, out);
        }
        Stmt::While { cond, body, .. } => {
            collect_assigned_locals_expr(cond, out);
            collect_assigned_locals_block(body, out);
        }
        Stmt::Loop { body, .. } => collect_assigned_locals_block(body, out),
        Stmt::Switch {
            selector,
            cases,
            default,
            ..
        } => {
            collect_assigned_locals_expr(selector, out);
            for c in cases {
                collect_assigned_locals_block(&c.body, out);
            }
            if let Some(d) = default {
                collect_assigned_locals_block(d, out);
            }
        }
        Stmt::Expr { expr, .. } => collect_assigned_locals_expr(expr, out),
        Stmt::Let { init: None, .. }
        | Stmt::Return { value: None, .. }
        | Stmt::Break(_)
        | Stmt::Continue(_) => {}
    }
}

fn collect_assigned_locals_expr(expr: &Expr, out: &mut IndexSet<SmolStr>) {
    match expr {
        Expr::Binary { op, lhs, rhs, .. } => {
            if matches!(
                op,
                BinaryOp::Assign
                    | BinaryOp::AddAssign
                    | BinaryOp::SubAssign
                    | BinaryOp::MulAssign
                    | BinaryOp::DivAssign
                    | BinaryOp::RemAssign
                    | BinaryOp::BitAndAssign
                    | BinaryOp::BitOrAssign
                    | BinaryOp::BitXorAssign
            ) && let Expr::Path { path, .. } = lhs.as_ref()
                && path.segments.len() == 1
            {
                out.insert(path.segments[0].name.clone());
            }
            collect_assigned_locals_expr(lhs, out);
            collect_assigned_locals_expr(rhs, out);
        }
        Expr::Unary { operand, .. }
        | Expr::Field {
            receiver: operand, ..
        } => {
            collect_assigned_locals_expr(operand, out);
        }
        Expr::Call { callee, args, .. } => {
            collect_assigned_locals_expr(callee, out);
            for a in args {
                collect_assigned_locals_expr(&a.value, out);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            collect_assigned_locals_expr(receiver, out);
            for a in args {
                collect_assigned_locals_expr(&a.value, out);
            }
        }
        Expr::Index {
            receiver, index, ..
        } => {
            collect_assigned_locals_expr(receiver, out);
            collect_assigned_locals_expr(index, out);
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_assigned_locals_expr(cond, out);
            collect_assigned_locals_block(then_branch, out);
            if let Some(eb) = else_branch {
                collect_assigned_locals_expr(eb, out);
            }
        }
        Expr::Block(b) => collect_assigned_locals_block(b, out),
        Expr::ArrayLit { elements, .. } => {
            for e in elements {
                collect_assigned_locals_expr(e, out);
            }
        }
        Expr::StructLit { fields, .. } => {
            for (_, e) in fields {
                collect_assigned_locals_expr(e, out);
            }
        }
        _ => {}
    }
}

fn block_ends_with_terminator(stmts: &[StructuredStmt]) -> bool {
    matches!(
        stmts.last(),
        Some(StructuredStmt::Return(_) | StructuredStmt::Break | StructuredStmt::Continue)
    )
}

fn unwrap_step_by(expr: &Expr) -> (&Expr, Option<&Expr>) {
    if let Expr::MethodCall {
        receiver,
        method,
        args,
        ..
    } = expr
        && method.name == "step_by"
    {
        return (receiver.as_ref(), args.first().map(|a| &a.value));
    }
    (expr, None)
}

fn parse_atomic_turbofish(turbofish: &[TypeArg]) -> (AtomicScope, AtomicOrder) {
    let mut scope = AtomicScope::Device;
    let mut order = AtomicOrder::Relaxed;
    for arg in turbofish {
        if let TypeArg::Type(TypeExpr::Named { path, .. }) = arg {
            let joined = path.joined();
            match joined.as_str() {
                "Scope::Device" | "Device" => scope = AtomicScope::Device,
                "Scope::Workgroup" | "Workgroup" => scope = AtomicScope::Workgroup,
                "Scope::Subgroup" | "Subgroup" => scope = AtomicScope::Subgroup,
                "Order::Relaxed" | "Relaxed" => order = AtomicOrder::Relaxed,
                "Order::Acquire" | "Acquire" => order = AtomicOrder::Acquire,
                "Order::Release" | "Release" => order = AtomicOrder::Release,
                "Order::AcqRel" | "AcqRel" => order = AtomicOrder::AcqRel,
                _ => {}
            }
        }
    }
    (scope, order)
}

fn parse_radix_u64(clean: &str) -> Option<u64> {
    if let Some(hex) = clean
        .strip_prefix("0x")
        .or_else(|| clean.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else if let Some(bin) = clean
        .strip_prefix("0b")
        .or_else(|| clean.strip_prefix("0B"))
    {
        u64::from_str_radix(bin, 2).ok()
    } else if let Some(oct) = clean
        .strip_prefix("0o")
        .or_else(|| clean.strip_prefix("0O"))
    {
        u64::from_str_radix(oct, 8).ok()
    } else {
        clean.parse::<u64>().ok()
    }
}

fn lower_int_literal(raw: &str, ir_fn: &mut IrFunction) -> InstId {
    let is_u64 = raw.ends_with("u64");
    let is_signed_i64 = raw.ends_with("i64");
    let is_u32 = raw.ends_with("u32") || (raw.ends_with('u') && !is_u64);
    let clean = raw
        .trim_end_matches("u64")
        .trim_end_matches("i64")
        .trim_end_matches("u32")
        .trim_end_matches("i32")
        .trim_end_matches(['u', 'i'])
        .replace('_', "");

    let parsed = parse_radix_u64(&clean).unwrap_or(0);

    if is_u64 {
        ir_fn.emit(None, InstData::ConstU64(parsed))
    } else if is_signed_i64 {
        ir_fn.emit(None, InstData::ConstI64(parsed.cast_signed()))
    } else if is_u32 {
        ir_fn.emit(None, InstData::ConstU32(parsed as u32))
    } else {
        ir_fn.emit(None, InstData::ConstI32((parsed as u32).cast_signed()))
    }
}

fn parse_const_i64_expr(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::IntLit(raw, _) => {
            let clean = raw
                .trim_end_matches("u64")
                .trim_end_matches("i64")
                .trim_end_matches("u32")
                .trim_end_matches("i32")
                .trim_end_matches(['u', 'i'])
                .replace('_', "");
            parse_radix_u64(&clean).map(u64::cast_signed)
        }
        Expr::Unary {
            op: UnaryOp::Neg,
            operand,
            ..
        } => parse_const_i64_expr(operand).map(i64::saturating_neg),
        _ => None,
    }
}

fn extract_const_range_bounds(
    expr: &Expr,
    step: i64,
    ir_fn: &mut IrFunction,
) -> (Option<u32>, Option<InstId>) {
    let Expr::Binary {
        op: op @ (BinaryOp::Range | BinaryOp::RangeInclusive),
        lhs,
        rhs,
        ..
    } = expr
    else {
        return (None, None);
    };
    let (Some(start), Some(end)) = (parse_const_i64_expr(lhs), parse_const_i64_expr(rhs)) else {
        return (None, None);
    };
    let span_len = if matches!(op, BinaryOp::RangeInclusive) {
        end.saturating_sub(start).saturating_add(1).max(0)
    } else {
        end.saturating_sub(start).max(0)
    };
    let safe_step = step.max(1);
    let count = if span_len == 0 {
        0
    } else {
        (span_len - 1) / safe_step + 1
    };
    let trip = u32::try_from(count).ok();
    let init = ir_fn.emit(None, InstData::ConstI32(start as i32));
    (trip, Some(init))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use vsl_mc::BINDLESS_IMAGE_COUNT;

    const REFLECTIONS_VSL: &str = include_str!("../../../examples/sunlit_vanilla/reflections.vsl");
    const GI_CACHE_VSL: &str = include_str!("../../../examples/sunlit_vanilla/gi_cache.vsl");
    const RESTIR_VSL: &str = include_str!("../../../examples/sunlit_vanilla/restir.vsl");
    const TILED_LIGHTING_VSL: &str =
        include_str!("../../../examples/sunlit_vanilla/tiled_lighting.vsl");
    const PACK_VSL: &str = include_str!("../../../examples/sunlit_vanilla/pack.vsl");

    /// Scans a SPIR-V word stream for `OpDecorate <id> <decoration>`.
    fn spirv_has_decoration(words: &[u32], decoration: u32) -> bool {
        let mut idx = 5usize;
        while idx < words.len() {
            let word = words[idx];
            let wc = (word >> 16) as usize;
            let op = word & 0xFFFF;
            if wc == 0 || idx + wc > words.len() {
                break;
            }
            // OpDecorate = 71: [header, target_id, decoration, ...]
            if op == 71 && wc >= 3 && words[idx + 2] == decoration {
                return true;
            }
            idx += wc;
        }
        false
    }

    /// Runs `spirv-val --target-env vulkan1.3` when `spirv-val` is installed on `PATH`.
    fn validate_spirv_with_spirv_val(label: &str, words: &[u32]) {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for &w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let path =
            std::env::temp_dir().join(format!("vsl_driver_{label}_{}.spv", std::process::id()));
        std::fs::write(&path, &bytes).expect("write temp spirv");

        let status = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&path)
            .output();
        let _ = std::fs::remove_file(&path);

        match status {
            Ok(out) => {
                assert!(
                    out.status.success(),
                    "spirv-val failed for `{label}`:\nstdout: {}\nstderr: {}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr),
                );
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => panic!("failed to invoke spirv-val: {err}"),
        }
    }

    #[test]
    fn compiles_reflections_with_nonuniform_rayquery_and_spirv_val() {
        let mut compiler = VslCompiler::new();
        let opts = CompileOptions::for_tier(RenderTier::Tier2RayQuery);
        let out = compiler.compile_shader_source("reflections.vsl", REFLECTIONS_VSL, &opts);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        let unit = &out.units[0];
        assert_eq!(unit.entry_point, "trace_glossy_reflections");
        assert_eq!(unit.spirv[0], 0x0723_0203);
        assert_eq!(unit.debug_channels.len(), 1);
        assert!(
            unit.descriptors
                .iter()
                .any(|d| d.name == "debug_custom_reflections")
        );
        let bindless = unit
            .descriptors
            .iter()
            .find(|d| d.set == 2 && d.binding == 0)
            .expect("missing Set 2 bindless descriptor");
        assert_eq!(bindless.kind, DescriptorKind::CombinedImageSampler);
        assert!(!bindless.is_unsized);
        assert_eq!(bindless.count, Some(BINDLESS_IMAGE_COUNT));
        // AGENTS.md §4.4: NonUniform decoration (SPIR-V Decoration::NonUniform = 5300) must survive DCE.
        assert!(
            spirv_has_decoration(&unit.spirv, 5300),
            "expected OpDecorate NonUniform in reflections.vsl SPIR-V"
        );
        // GlobalInvocationId BuiltIn decoration (Decoration::BuiltIn = 11) must be emitted.
        assert!(
            spirv_has_decoration(&unit.spirv, 11),
            "expected OpDecorate BuiltIn in reflections.vsl SPIR-V"
        );
        for pair in unit.descriptors.windows(2) {
            assert!(
                (pair[0].set, pair[0].binding) <= (pair[1].set, pair[1].binding),
                "descriptors out of order: {:?}",
                unit.descriptors
            );
        }
        validate_spirv_with_spirv_val("reflections", &unit.spirv);

        assert_eq!(compiler.l1_cache_len(), 1);
        assert_eq!(compiler.l2_cache_len(), 1);
        let out2 = compiler.compile_shader_source("reflections.vsl", REFLECTIONS_VSL, &opts);
        assert!(!out2.has_errors());
        assert_eq!(compiler.l1_cache_len(), 1);
        assert_eq!(compiler.l2_cache_len(), 1);
    }

    #[test]
    fn compiles_tiled_lighting_with_workgroup_atomic_and_spirv_val() {
        let mut compiler = VslCompiler::new();
        let out = compiler.compile_shader_source(
            "tiled_lighting.vsl",
            TILED_LIGHTING_VSL,
            &CompileOptions::for_tier(RenderTier::Tier2RayQuery),
        );
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        assert_eq!(out.units[0].entry_point, "cull_tile_lights");
        assert_eq!(out.units[0].spirv[0], 0x0723_0203);
        validate_spirv_with_spirv_val("tiled_lighting", &out.units[0].spirv);
    }

    #[test]
    fn compiles_push_constant_builtin_and_branch_mutation_phi_with_spirv_val() {
        let mut compiler = VslCompiler::new();
        let src = r"
            #[push_constant]
            struct Params {
                mode: u32,
                scale: f32,
            }

            @stage(compute(8, 8, 1))
            pub fn branch_mutate(
                #[builtin(global_invocation_id)] gid: uvec3,
                #[push_constant] pc: Params,
                out_img: RWImage2D<rgba16f>,
            ) {
                let mut acc = gid.x;
                if pc.mode == 1u {
                    acc = acc + 4u;
                } else {
                    acc = acc + 2u;
                }
                out_img.store(gid.xy, acc);
            }
        ";
        let out =
            compiler.compile_shader_source("branch_mutate.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("branch_mutate", &out.units[0].spirv);
    }

    #[test]
    fn compiles_while_loop_local_mutation_with_dominating_phi_and_spirv_val() {
        let mut compiler = VslCompiler::new();
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn while_mutate(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let mut x = gid.x;
                while x < 10u {
                    x = x + 1u;
                }
                out_img.store(gid.xy, x);
            }
        ";
        let out =
            compiler.compile_shader_source("while_mutate.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("while_mutate", &out.units[0].spirv);
    }

    #[test]
    fn validates_gi_cache_on_tier1_and_restir_on_tier2_with_default_tier_caps() {
        let mut compiler = VslCompiler::new();
        let gi_out = compiler.compile_shader_source(
            "gi_cache.vsl",
            GI_CACHE_VSL,
            &CompileOptions::for_tier(RenderTier::Tier1ComputeGi),
        );
        assert!(!gi_out.has_errors(), "{:?}", gi_out.diagnostics);
        assert!(!gi_out.hir.functions.is_empty());
        assert_eq!(gi_out.hir.pruned_functions, 0);
        assert_eq!(gi_out.units, Vec::<vsl_mc::CompiledShaderUnit>::new());

        let restir_out = compiler.compile_shader_source(
            "restir.vsl",
            RESTIR_VSL,
            &CompileOptions::for_tier(RenderTier::Tier2RayQuery),
        );
        assert!(!restir_out.has_errors(), "{:?}", restir_out.diagnostics);
        assert!(!restir_out.hir.functions.is_empty());
        assert_eq!(restir_out.hir.pruned_functions, 0);
        assert_eq!(restir_out.units, Vec::<vsl_mc::CompiledShaderUnit>::new());
    }

    #[test]
    fn builds_pack_schema_from_evaluated_pipeline() {
        let mut compiler = VslCompiler::new();
        let out = compiler.evaluate_pack_source("pack.vsl", PACK_VSL, &EvalConfig::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let schema = pack_schema_from_pipeline(&out.pipeline);
        assert!(!schema.spec_ids.is_empty());
        assert!(!schema.params.is_empty());
        assert!(!schema.resource_slots.is_empty());
    }

    #[test]
    fn pack_schema_registers_v1_resource_names() {
        let mut pipeline = PipelineIr::default();
        pipeline
            .resource_textures
            .push(vsl_pack::ResourceTextureDecl {
                slot: 0,
                name: SmolStr::new_static("blue_noise"),
                path: SmolStr::new_static("textures/blue.png"),
                format: ImageFormat::Rgba8,
                filter: SmolStr::new_static("nearest"),
                wrap: SmolStr::new_static("repeat"),
            });
        let schema = pack_schema_from_pipeline(&pipeline);
        assert_eq!(schema.resource_slots.get("blue_noise"), Some(&0));
        assert_eq!(schema.resource_slots.get("blue"), Some(&0));
        assert_eq!(schema.resource_slots.get("textures/blue.png"), Some(&0));
    }

    #[test]
    fn native_spirv_hook_receives_default_compute_workgroup_size_and_validates() {
        let mut compiler = VslCompiler::new();
        let src = r"
            @hook(sky)
            pub fn stylized_sky(dir: vec3) -> vec3 {
                dir
            }
        ";
        let out = compiler.compile_shader_source("sky.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        assert_eq!(out.units[0].stage, ShaderStage::Compute);
        assert_eq!(out.units[0].workgroup_size, Some([8, 8, 1]));
        validate_spirv_with_spirv_val("hook_sky", &out.units[0].spirv);
    }

    #[test]
    fn compile_project_parses_each_file_once() {
        let mut compiler = VslCompiler::new();
        let files = vec![
            (
                "lib.vsl".to_owned(),
                "module pack::lib; pub fn helper(x: i32) -> i32 { x + 1 }".to_owned(),
            ),
            (
                "main.vsl".to_owned(),
                "module pack::main; import pack::lib::{helper}; @stage(compute(8, 8, 1)) pub fn run() { let y = helper(1); }".to_owned(),
            ),
        ];
        let out = compiler.compile_project(&files, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.module_order.len(), 2);
        assert_eq!(out.units.len(), 1);
    }

    #[test]
    fn compiles_raster_gbuffer_hook_via_glsl_bridge() {
        let mut compiler = VslCompiler::new();
        let src = r"
            @hook(vertex_displace)
            pub fn waving_grass(inout pos: vec3, time: f32) {
                pos.y += sin(time) * mc::param(wind_strength);
            }
        ";
        let out = compiler.compile_shader_source("grass.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.hook_glsl_slots.len(), 1);
        assert_eq!(out.hook_glsl_slots[0].0, HookStage::VertexDisplace);
        assert!(
            out.hook_glsl_slots[0]
                .1
                .contains("vs_dispatch_hook_vertex_displace")
        );
    }

    #[test]
    fn prunes_tier2_entry_point_when_compiling_for_tier0() {
        let mut compiler = VslCompiler::new();
        let opts = CompileOptions::for_tier(RenderTier::Tier0Raster);
        let out = compiler.compile_shader_source("reflections.vsl", REFLECTIONS_VSL, &opts);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.hir.pruned_functions, 1);
        assert_eq!(out.units, Vec::<vsl_mc::CompiledShaderUnit>::new());
    }

    #[test]
    fn compiles_tiled_lighting_to_glsl_460_oracle() {
        let mut compiler = VslCompiler::new();
        let out = compiler.compile_shader_to_glsl("tiled_lighting.vsl", TILED_LIGHTING_VSL);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(out.glsl.contains("#version 460 core"));
        assert!(
            out.glsl
                .contains("layout(push_constant, scalar) uniform AtrousPush")
        );
    }

    #[test]
    fn evaluates_and_schedules_sunlit_pack_vsl() {
        let mut compiler = VslCompiler::new();
        let out = compiler.evaluate_pack_source("pack.vsl", PACK_VSL, &EvalConfig::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let plan = out.schedule.expect("valid topological schedule");
        assert_ne!(plan.ordered_passes, Vec::<smol_str::SmolStr>::new());
    }
}
