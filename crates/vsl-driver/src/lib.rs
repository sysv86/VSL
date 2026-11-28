//! End-to-end VSL 2.0 compilation driver coordinating frontend, multi-file `ModuleGraph` +
//! visibility-checked `ModuleExports`, HIR lowering + single-source reflection, typed structured
//! SSA IR lowering & passes, direct SPIR-V 1.6 emission, native `@hook` chain dispatch,
//! schema-driven Raster G-Buffer GLSL hook bridge, L1/L2 BLAKE3 in-memory + disk caching,
//! parallel entry-point compilation, and `pack.vsl` DAG scheduling & cross-validation.

mod cache;
mod hooks;
mod lower_ir;
mod modules;

use crate::cache::{L1CacheKey, hash_pack_schema, load_l2_from_disk, save_l2_to_disk};
use crate::hooks::{build_hook_chain_unit, compile_hook_chain_sources};
use crate::lower_ir::{collect_reachable_helpers, lower_entry_module_to_ir};
use crate::modules::{ModuleExports, module_graph_dependency_path, resolve_module_imports};
use indexmap::{IndexMap, IndexSet};
use smol_str::SmolStr;
use std::path::PathBuf;
use vsl_diagnostics::{Diagnostic, FileId, SourceDatabase, Span};
use vsl_glsl::{
    BindlessAbi, EmitOptions as GlslEmitOptions, emit_glsl_460_with_options,
    emit_raster_gbuffer_hook_slot_with_schema,
};
use vsl_hir::module::{L1ModuleHash, ModuleGraph, compute_l2_artifact_hash};
use vsl_hir::{
    HirFunction, HirModule, PackSchema, ValidationSummary, lower_unit_with_schema_and_abi,
    validate_unit,
};
use vsl_ir::PassPipeline;
pub use vsl_mc::AbiProfile;
use vsl_mc::{
    CompiledShaderUnit, HardwareCaps, HookAbiRoute, HookStage, PackParamType, RenderTier,
    ShaderStage, SpecConstantDefault,
};
use vsl_pack::{
    EvalConfig, PipelineIr, SchedulePlan, SettingValueIr, evaluate_pack_vsl, validate_and_schedule,
    validate_pass_bindings_against_shader,
};
use vsl_spirv::emit_module_spirv;
use vsl_syntax::ast::{FnDecl, Item};
use vsl_syntax::{ParseResult, parse_source};

/// Target compilation options passed by `vulkanshaders-core` or `vslc`.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub tier: RenderTier,
    pub active_caps: HardwareCaps,
    /// Target descriptor/ABI profile (`Hyaloclast` by default, or `Standalone` for generic Vulkan 1.3+).
    pub abi: AbiProfile,
    /// Active specialization-constant values keyed by setting name (encoded as raw `u32` bits).
    pub spec_values: IndexMap<SmolStr, u32>,
    /// Optional `pack.vsl` schema providing pack-global `spec_id`s, typed defaults, BDA parameter
    /// byte offsets/types, and Set 3 resource texture slots.
    pub pack_schema: Option<PackSchema>,
    pub driver_uuid: [u8; 16],
    /// Optional directory for persistent disk L2 `CompiledShaderUnit` artifacts.
    pub l2_cache_dir: Option<PathBuf>,
}

impl CompileOptions {
    /// Returns tier-appropriate default hardware capabilities matching Hyaloclast's
    /// `device_merger.rs` baseline (`INT64_ATOMICS` enabled across all Vulkan 1.3 tiers,
    /// plus `RAY_QUERY` and `MAXIMAL_RECONVERGENCE` on Tier 2/3 hardware ray-tracing targets).
    #[must_use]
    pub fn for_tier(tier: RenderTier) -> Self {
        let active_caps = match tier {
            RenderTier::Tier0Raster | RenderTier::Tier1ComputeGi => HardwareCaps::INT64_ATOMICS,
            RenderTier::Tier2RayQuery => {
                HardwareCaps::INT64_ATOMICS
                    | HardwareCaps::RAY_QUERY
                    | HardwareCaps::MAXIMAL_RECONVERGENCE
            }
            RenderTier::Tier3PathTracing => {
                HardwareCaps::INT64_ATOMICS
                    | HardwareCaps::RAY_QUERY
                    | HardwareCaps::MAXIMAL_RECONVERGENCE
                    | HardwareCaps::RAY_TRACING_PIPELINE
                    | HardwareCaps::MESH_SHADER
            }
        };
        Self {
            tier,
            active_caps,
            abi: AbiProfile::Hyaloclast,
            spec_values: IndexMap::new(),
            pack_schema: None,
            driver_uuid: [0u8; 16],
            l2_cache_dir: None,
        }
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::for_tier(RenderTier::Tier2RayQuery)
    }
}

/// Converts a `pack.vsl` [`SettingValueIr`] into its canonical [`PackParamType`] and default
/// 32-bit specialization bit pattern.
fn setting_value_type_and_bits(
    value: &SettingValueIr,
    enum_variants: &[SmolStr],
) -> (PackParamType, u32) {
    match value {
        SettingValueIr::Bool(b) => (PackParamType::Bool, u32::from(*b)),
        SettingValueIr::Int(i) => (PackParamType::I32, (*i as i32).cast_unsigned()),
        SettingValueIr::Float(f) => (PackParamType::F32, f.to_bits()),
        SettingValueIr::ColorRgb(_) => (PackParamType::Vec3, 0),
        SettingValueIr::EnumVariant(variant) => {
            let idx = enum_variants.iter().position(|v| v == variant).unwrap_or(0) as u32;
            (PackParamType::Enum, idx)
        }
    }
}

/// Constructs a pack-wide [`PackSchema`] from an evaluated [`PipelineIr`] so `.vsl` shader
/// modules receive consistent `spec_id`s, typed [`SpecConstantDefault`]s, std430-aligned
/// `PackParamBuffer` byte offsets + [`PackParamType`]s, and Set 3 resource texture slots matching
/// `pack.vsl`.
#[must_use]
pub fn pack_schema_from_pipeline(pipeline: &PipelineIr) -> PackSchema {
    let mut schema = PackSchema::default();
    let mut next_spec_id = 0u32;
    let mut param_offset = 0u32;

    for (name, setting) in &pipeline.settings {
        let (param_ty, default_bits) =
            setting_value_type_and_bits(&setting.default_value, &setting.enum_variants);
        if setting.is_spec {
            schema.spec_ids.insert(name.clone(), next_spec_id);
            schema.spec_defaults.insert(
                name.clone(),
                SpecConstantDefault {
                    spec_id: next_spec_id,
                    ty: param_ty,
                    default_bits,
                },
            );
            next_spec_id += 1;
        } else {
            let size = param_ty.byte_size();
            let align = param_ty.std430_align();
            let aligned = param_offset.div_ceil(align) * align;
            schema.params.insert(name.clone(), (aligned, size));
            schema.param_types.insert(name.clone(), param_ty);
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
        if !tex.name.is_empty() {
            schema.resource_slots.insert(tex.name.clone(), tex.slot);
        }
    }

    schema
}

/// Cross-validates every render-graph pass in `pipeline` whose `entry_fn` matches one of the
/// compiled shader `units`, calling [`vsl_pack::validate_pass_bindings_against_shader`].
///
/// Returns an empty vector when all matched passes satisfy their shader descriptor ABIs.
#[must_use]
pub fn validate_pack_shader_bindings(
    pipeline: &PipelineIr,
    units: &[CompiledShaderUnit],
) -> Vec<String> {
    let mut errors = Vec::new();
    for pass in &pipeline.passes {
        if pass.entry_fn.trim().is_empty() {
            continue;
        }
        let short_entry = pass
            .entry_fn
            .rsplit("::")
            .next()
            .unwrap_or(pass.entry_fn.as_str());
        if let Some(unit) = units.iter().find(|u| {
            u.entry_point.as_str() == pass.entry_fn.as_str()
                || u.entry_point.as_str() == short_entry
        }) {
            errors.extend(validate_pass_bindings_against_shader(pass, unit, pipeline));
        }
    }
    errors
}

/// Output of compiling a `.vsl` shader source module.
#[derive(Debug, Clone)]
pub struct ShaderCompileOutput {
    pub file: FileId,
    pub summary: ValidationSummary,
    pub hir: HirModule,
    pub units: Vec<CompiledShaderUnit>,
    /// Raster G-Buffer hooks lowered via `vsl-glsl::emit_raster_gbuffer_hook_slot_with_schema`.
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

/// Pure-Rust in-process VSL 2.0 compiler session with L1 (HIR) and L2 (SPIR-V) BLAKE3 caches
/// and optional persistent disk L2 cache directory.
#[derive(Debug, Default)]
pub struct VslCompiler {
    pub sources: SourceDatabase,
    l1_cache: IndexMap<L1CacheKey, HirModule>,
    l2_cache: IndexMap<[u8; 32], CompiledShaderUnit>,
    l2_cache_dir: Option<PathBuf>,
}

impl VslCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a compiler session configured with a persistent disk L2 artifact cache directory.
    #[must_use]
    pub fn with_l2_cache_dir(dir: PathBuf) -> Self {
        Self {
            sources: SourceDatabase::default(),
            l1_cache: IndexMap::new(),
            l2_cache: IndexMap::new(),
            l2_cache_dir: Some(dir),
        }
    }

    pub fn l1_cache_len(&self) -> usize {
        self.l1_cache.len()
    }

    pub fn l2_cache_len(&self) -> usize {
        self.l2_cache.len()
    }

    /// Parses, lowers to HIR (with early tier/cap pruning and single-source-of-truth reflection),
    /// lowers entry points and native hooks to typed structured SSA IR -> SPIR-V 1.6 (in parallel
    /// across entry points using `std::thread::scope`), lowers raster G-Buffer hooks through the
    /// schema-driven GLSL hook bridge, and caches results under L1/L2 BLAKE3 keys.
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

    /// Compiles an ordered chain of native SPIR-V `@hook` functions for `stage` into a single
    /// dispatch entry point `vs_hook_chain_<stage>` (with alpha-over compositing for
    /// `HookStage::Clouds`).
    pub fn compile_hook_chain(
        &mut self,
        stage: HookStage,
        hook_functions: &[(&str, &str)],
        options: &CompileOptions,
    ) -> Result<CompiledShaderUnit, Vec<Diagnostic>> {
        compile_hook_chain_sources(&mut self.sources, stage, hook_functions, options)
    }

    /// Resolves and compiles a multi-file VSL 2.0 project in topological dependency order using
    /// [`ModuleGraph`] and per-module [`ModuleExports`] symbol tables (enforcing `pub` visibility
    /// with `E0502`, exporting `Struct`/`Const`/`Function`/`Trait`/`Impl`, and reporting name
    /// collisions with `E0503`).
    pub fn compile_project(
        &mut self,
        files: &[(String, String)],
        options: &CompileOptions,
    ) -> ProjectCompileOutput {
        let mut parsed_modules: Vec<(SmolStr, FileId, ParseResult, String)> =
            Vec::with_capacity(files.len());
        let mut known_modules: IndexSet<SmolStr> = IndexSet::with_capacity(files.len());

        for (idx, (file_name, source)) in files.iter().enumerate() {
            let file_id = self.sources.add_file(file_name.clone(), source.clone());
            let parsed = parse_source(file_id, source);
            let mod_name = parsed.unit.module_decl.as_ref().map_or_else(
                || SmolStr::new(format!("anon_{idx}")),
                |m| SmolStr::new(m.path.joined()),
            );
            known_modules.insert(mod_name.clone());
            parsed_modules.push((mod_name, file_id, parsed, source.clone()));
        }

        let mut graph = ModuleGraph::new();
        let mut by_module: IndexMap<SmolStr, (FileId, ParseResult, String)> =
            IndexMap::with_capacity(parsed_modules.len());
        let mut diagnostics = Vec::new();

        for (mod_name, file_id, parsed, source) in parsed_modules {
            if by_module.contains_key(&mod_name) {
                let span = parsed
                    .unit
                    .module_decl
                    .as_ref()
                    .map_or_else(|| Span::new(file_id, 0, 0), |m| m.span);
                diagnostics.push(Diagnostic::error(
                    "E0503",
                    format!("duplicate module declaration `{mod_name}` in project"),
                    span,
                ));
                continue;
            }
            let imports: Vec<SmolStr> = parsed
                .unit
                .imports
                .iter()
                .map(|imp| module_graph_dependency_path(imp, &known_modules))
                .collect();
            graph.register_module(mod_name.clone(), &source, imports);
            by_module.insert(mod_name, (file_id, parsed, source));
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
        let mut exports_by_module: IndexMap<SmolStr, ModuleExports> = IndexMap::new();
        let mut native_hooks_by_stage: IndexMap<
            HookStage,
            Vec<(L1ModuleHash, HirModule, HirFunction, Vec<Item>)>,
        > = IndexMap::new();

        for mod_name in &module_order {
            if let Some((file_id, mut parsed, source)) = by_module.shift_remove(mod_name) {
                // Collect this module's own `pub` + private declarations before prepending imports,
                // then attach resolved upstream items for downstream transitive closure (C4).
                let mut exports = ModuleExports::from_unit(&parsed.unit);

                let mut import_diags = Vec::new();
                let mut imported_items =
                    resolve_module_imports(&parsed.unit, &exports_by_module, &mut import_diags);
                let has_import_errors = import_diags.iter().any(Diagnostic::is_error);
                diagnostics.extend(import_diags);

                exports.record_upstream_items(&imported_items);
                exports_by_module.insert(mod_name.clone(), exports);

                if !imported_items.is_empty() {
                    imported_items.append(&mut parsed.unit.items);
                    parsed.unit.items = imported_items;
                }

                let module_items_snapshot = parsed.unit.items.clone();
                let l1_hash = l1_hashes
                    .get(mod_name)
                    .copied()
                    .unwrap_or_else(|| L1ModuleHash(*blake3::hash(source.as_bytes()).as_bytes()));
                let out = self.compile_shader_with_parsed_inner(
                    file_id,
                    parsed,
                    l1_hash,
                    options,
                    !has_import_errors,
                );

                for (_, func) in out.hir.functions.iter() {
                    if let Some(hook_stage) = func.hook
                        && hook_stage.abi_route() == HookAbiRoute::NativeSpirv
                    {
                        native_hooks_by_stage.entry(hook_stage).or_default().push((
                            l1_hash,
                            out.hir.clone(),
                            func.clone(),
                            module_items_snapshot.clone(),
                        ));
                    }
                }

                diagnostics.extend(out.diagnostics);
                units.extend(out.units);
                hook_glsl_slots.extend(out.hook_glsl_slots);
            }
        }

        // Synthesize combined native `@hook` chain dispatch units whenever multiple native hooks
        // register for the same `HookStage` across project modules.
        if !diagnostics.iter().any(Diagnostic::is_error) {
            for (stage, hook_entries) in native_hooks_by_stage {
                if hook_entries.len() >= 2 {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"vsl2-project-hook-chain-v1");
                    hasher.update(stage.as_str().as_bytes());
                    let mut lowered_entries = Vec::with_capacity(hook_entries.len());
                    for (mod_l1, hir, func, items) in hook_entries {
                        hasher.update(&mod_l1.0);
                        hasher.update(&(hir.module_name.len() as u64).to_le_bytes());
                        hasher.update(hir.module_name.as_bytes());
                        hasher.update(&(func.name.len() as u64).to_le_bytes());
                        hasher.update(func.name.as_bytes());
                        lowered_entries.push((hir, func, items));
                    }
                    let chain_l1 = L1ModuleHash(*hasher.finalize().as_bytes());
                    if let Some(chain_unit) = build_hook_chain_unit(
                        stage,
                        &lowered_entries,
                        chain_l1,
                        options,
                        &mut diagnostics,
                    ) {
                        units.push(chain_unit);
                    }
                }
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
        self.compile_shader_with_parsed_inner(file, parsed, l1_hash, options, true)
    }

    fn compile_shader_with_parsed_inner(
        &mut self,
        file: FileId,
        parsed: ParseResult,
        l1_hash: L1ModuleHash,
        options: &CompileOptions,
        allow_l1_cache: bool,
    ) -> ShaderCompileOutput {
        let mut diagnostics = parsed.diagnostics;
        let has_parse_errors = diagnostics.iter().any(Diagnostic::is_error);

        let mut pack_schema_hash = hash_pack_schema(options.pack_schema.as_ref());
        pack_schema_hash[31] ^= options.abi.as_u8();
        let l1_key = L1CacheKey {
            hash: l1_hash,
            tier: options.tier,
            active_caps: options.active_caps,
            abi: options.abi,
            pack_schema_hash,
        };

        let hir = if allow_l1_cache
            && !has_parse_errors
            && let Some(cached_hir) = self.l1_cache.get(&l1_key)
        {
            cached_hir.clone()
        } else {
            let (lowered_hir, hir_diags) = lower_unit_with_schema_and_abi(
                &parsed.unit,
                options.tier,
                options.active_caps,
                options.pack_schema.as_ref(),
                options.abi,
            );
            diagnostics.extend(hir_diags.into_vec());
            if allow_l1_cache && !diagnostics.iter().any(Diagnostic::is_error) {
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

        let mut hook_glsl_slots = Vec::new();
        let empty_param_types = IndexMap::new();
        let schema_param_types = options
            .pack_schema
            .as_ref()
            .map_or(&empty_param_types, |s| &s.param_types);

        let disk_cache_dir = options
            .l2_cache_dir
            .as_deref()
            .or(self.l2_cache_dir.as_deref());

        // Collect all active native entry points and resolve L2 in-memory / disk cache hits.
        let mut slots: Vec<Option<CompiledShaderUnit>> = Vec::new();
        let mut pending: Vec<(usize, &HirFunction, ShaderStage, Option<[u32; 3]>, [u8; 32])> =
            Vec::new();

        for (_, func) in hir.functions.iter() {
            // Raster G-Buffer hooks lower via the schema-driven GLSL hook bridge (Item 25).
            if let Some(hook_stage) = func.hook
                && hook_stage.abi_route() == HookAbiRoute::RasterGbufferGlslBridge
            {
                if let Some(ast_fn) = find_ast_function(&parsed.unit.items, &func.name) {
                    match emit_raster_gbuffer_hook_slot_with_schema(
                        hook_stage,
                        ast_fn,
                        schema_param_types,
                    ) {
                        Ok(glsl) => hook_glsl_slots.push((hook_stage, glsl)),
                        Err(msg) => diagnostics.push(Diagnostic::error("E0510", msg, func.span)),
                    }
                }
                continue;
            }

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

            let mut seen_spec_names: IndexSet<SmolStr> = IndexSet::new();
            let mut used_spec_pairs: Vec<(u32, u32)> = Vec::new();
            for s in &func.reflection.used_specs {
                if seen_spec_names.insert(s.name.clone()) {
                    let raw = options.spec_values.get(&s.name).copied().unwrap_or(0);
                    used_spec_pairs.push((s.spec_id, raw));
                }
            }
            for helper in collect_reachable_helpers(&hir, func) {
                for s in &helper.reflection.used_specs {
                    if seen_spec_names.insert(s.name.clone()) {
                        let raw = options.spec_values.get(&s.name).copied().unwrap_or(0);
                        used_spec_pairs.push((s.spec_id, raw));
                    }
                }
            }
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

            let entry_idx = slots.len();
            if let Some(cached) = self.l2_cache.get(&cache_key) {
                slots.push(Some(cached.clone()));
                continue;
            }
            if let Some(dir) = disk_cache_dir
                && let Some(disk_unit) = load_l2_from_disk(dir, &cache_key)
            {
                self.l2_cache.insert(cache_key, disk_unit.clone());
                slots.push(Some(disk_unit));
                continue;
            }

            slots.push(None);
            pending.push((entry_idx, func, stage, effective_wg_size, cache_key));
        }

        // Compile cache-missed entry points (in parallel via `std::thread::scope` when 2+ exist),
        // placing compiled units back into `slots` by original `entry_idx` for deterministic order.
        let compiled_results: Vec<
            Result<(usize, [u8; 32], CompiledShaderUnit, Vec<Diagnostic>), Diagnostic>,
        > = if pending.len() <= 1 {
            pending
                .into_iter()
                .map(|(entry_idx, func, stage, wg_size, cache_key)| {
                    let (unit, diags) = compile_single_entry_unit(
                        &hir,
                        func,
                        &parsed.unit.items,
                        options.pack_schema.as_ref(),
                        stage,
                        wg_size,
                        cache_key,
                    );
                    Ok((entry_idx, cache_key, unit, diags))
                })
                .collect()
        } else {
            let hir_ref = &hir;
            let items_ref = parsed.unit.items.as_slice();
            let schema_ref = options.pack_schema.as_ref();
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(pending.len());
                for (entry_idx, func, stage, wg_size, cache_key) in pending {
                    let fn_name = func.name.clone();
                    let fn_span = func.span;
                    let handle = scope.spawn(move || {
                        let (unit, diags) = compile_single_entry_unit(
                            hir_ref, func, items_ref, schema_ref, stage, wg_size, cache_key,
                        );
                        (entry_idx, cache_key, unit, diags)
                    });
                    handles.push((fn_name, fn_span, handle));
                }
                handles
                    .into_iter()
                    .map(|(fn_name, fn_span, handle)| match handle.join() {
                        Ok(res) => Ok(res),
                        Err(payload) => {
                            let reason = payload
                                .downcast_ref::<&str>()
                                .copied()
                                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                                .unwrap_or("unknown panic payload");
                            Err(Diagnostic::error(
                                "E0599",
                                format!(
                                    "internal compiler error: worker thread panicked while compiling entry point `{fn_name}`: {reason}"
                                ),
                                fn_span,
                            ))
                        }
                    })
                    .collect()
            })
        };

        for res in compiled_results {
            match res {
                Ok((entry_idx, cache_key, unit, entry_diags)) => {
                    let has_entry_err = entry_diags.iter().any(Diagnostic::is_error);
                    diagnostics.extend(entry_diags);
                    if !has_entry_err {
                        self.l2_cache.insert(cache_key, unit.clone());
                        if let Some(dir) = disk_cache_dir {
                            save_l2_to_disk(dir, &cache_key, &unit);
                        }
                        slots[entry_idx] = Some(unit);
                    }
                }
                Err(ice_diag) => diagnostics.push(ice_diag),
            }
        }

        let units: Vec<CompiledShaderUnit> = slots.into_iter().flatten().collect();

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
        self.compile_shader_to_glsl_with_abi(file_name, source, AbiProfile::Hyaloclast)
    }

    /// Parses, validates, and emits Stage A GLSL 460 oracle source for the requested [`AbiProfile`].
    pub fn compile_shader_to_glsl_with_abi(
        &mut self,
        file_name: impl Into<String>,
        source: &str,
        abi: AbiProfile,
    ) -> GlslCompileOutput {
        let file = self.sources.add_file(file_name, source);
        let parsed = parse_source(file, source);
        let mut diagnostics = parsed.diagnostics;

        let (_, val_diags) = validate_unit(&parsed.unit);
        diagnostics.extend(val_diags.into_vec());

        let glsl = if diagnostics.iter().any(Diagnostic::is_error) {
            String::new()
        } else {
            let bindless_abi = match abi {
                AbiProfile::Hyaloclast => BindlessAbi::Hyaloclast4096,
                AbiProfile::Standalone => BindlessAbi::UpstreamUnsized,
            };
            emit_glsl_460_with_options(
                &parsed.unit,
                GlslEmitOptions {
                    route_params_to_vs_motion: false,
                    bindless_abi,
                    abi,
                },
            )
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

fn compile_single_entry_unit(
    hir: &HirModule,
    func: &HirFunction,
    module_items: &[Item],
    pack_schema: Option<&PackSchema>,
    stage: ShaderStage,
    workgroup_size: Option<[u32; 3]>,
    cache_key: [u8; 32],
) -> (CompiledShaderUnit, Vec<Diagnostic>) {
    let mut lowered =
        lower_entry_module_to_ir(hir, func, module_items, pack_schema, stage, workgroup_size);
    let pipeline = PassPipeline::default();
    pipeline.run_module(&mut lowered.ir_module);

    // Merge transitive helper reflection (`required_caps`, `used_specs`, `used_params`,
    // `debug_channels`, `push_constants`) and reconcile IR flags (`uses_bda`, `uses_subgroups`,
    // `uses_quad_control`) before SPIR-V emission and `CompiledShaderUnit` construction.
    let mut required_caps = func.required_caps;
    let mut used_specs = func.reflection.used_specs.clone();
    let mut seen_specs: IndexSet<SmolStr> = used_specs.iter().map(|s| s.name.clone()).collect();
    let mut used_params = func.reflection.used_params.clone();
    let mut seen_params: IndexSet<SmolStr> = used_params.iter().map(|p| p.name.clone()).collect();
    let mut debug_channels = func.reflection.debug_channels.clone();
    let mut seen_debug: IndexSet<SmolStr> = debug_channels
        .iter()
        .map(|d| d.channel_name.clone())
        .collect();
    let mut push_constants = func.reflection.push_constants.clone();
    let mut push_constant_size = func.reflection.push_constant_size;

    for helper in collect_reachable_helpers(hir, func) {
        required_caps |= helper.required_caps;
        for spec in &helper.reflection.used_specs {
            if seen_specs.insert(spec.name.clone()) {
                used_specs.push(spec.clone());
            }
        }
        for param in &helper.reflection.used_params {
            if seen_params.insert(param.name.clone()) {
                used_params.push(param.clone());
            }
        }
        for dbg in &helper.reflection.debug_channels {
            if seen_debug.insert(dbg.channel_name.clone()) {
                debug_channels.push(dbg.clone());
            }
        }
        if push_constant_size == 0 && helper.reflection.push_constant_size > 0 {
            push_constant_size = helper.reflection.push_constant_size;
            push_constants.clone_from(&helper.reflection.push_constants);
        }
    }

    let mut any_bda = false;
    let mut any_subgroups = false;
    let mut any_quad_control = false;
    for ir_fn in &lowered.ir_module.functions {
        required_caps |= ir_fn.required_caps;
        any_bda |= ir_fn.uses_bda;
        any_subgroups |= ir_fn.uses_subgroups;
        any_quad_control |= ir_fn.uses_quad_control;
    }
    if any_quad_control {
        required_caps |= HardwareCaps::QUAD_CONTROL;
    }
    if let Some(entry_ir) = lowered.ir_module.functions.first_mut() {
        entry_ir.uses_bda |= any_bda;
        entry_ir.uses_subgroups |= any_subgroups;
        entry_ir.uses_quad_control |= any_quad_control;
        entry_ir.required_caps |= required_caps;
    }

    let spirv = emit_module_spirv(&lowered.ir_module);

    let unit = CompiledShaderUnit {
        entry_point: func.name.clone(),
        stage,
        spirv,
        descriptors: lowered.unified_descriptors,
        push_constants,
        push_constant_size,
        workgroup_size,
        required_caps,
        used_specs,
        used_params,
        debug_channels,
        cache_key,
    };
    (unit, lowered.diagnostics)
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

fn find_ast_function<'a>(items: &'a [Item], name: &str) -> Option<&'a FnDecl> {
    items.iter().find_map(|item| match item {
        Item::Function(f) if f.name.name == name => Some(f),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use vsl_hir::lower_unit_with_schema;
    use vsl_ir::{GlslStd450Op, InstData, IrType, ScalarIrTy};
    use vsl_mc::{BINDLESS_IMAGE_COUNT, DescriptorKind, ImageFormat};
    use vsl_pack::{PassBindingRef, PassKind, PassMeta, PassNode, TargetDecl};

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
        assert!(
            spirv_has_decoration(&unit.spirv, 5300),
            "expected OpDecorate NonUniform in reflections.vsl SPIR-V"
        );
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
        assert_eq!(gi_out.units, Vec::<CompiledShaderUnit>::new());

        let restir_out = compiler.compile_shader_source(
            "restir.vsl",
            RESTIR_VSL,
            &CompileOptions::for_tier(RenderTier::Tier2RayQuery),
        );
        assert!(!restir_out.has_errors(), "{:?}", restir_out.diagnostics);
        assert!(!restir_out.hir.functions.is_empty());
        assert_eq!(restir_out.hir.pruned_functions, 0);
        assert_eq!(restir_out.units, Vec::<CompiledShaderUnit>::new());
    }

    #[test]
    fn builds_pack_schema_from_evaluated_pipeline() {
        let mut compiler = VslCompiler::new();
        let out = compiler.evaluate_pack_source("pack.vsl", PACK_VSL, &EvalConfig::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let schema = pack_schema_from_pipeline(&out.pipeline);
        assert!(!schema.spec_ids.is_empty());
        assert_eq!(schema.spec_ids.len(), schema.spec_defaults.len());
        assert!(!schema.params.is_empty());
        assert_eq!(schema.params.len(), schema.param_types.len());
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
        assert_eq!(out.units, Vec::<CompiledShaderUnit>::new());
    }

    #[test]
    fn compiles_tiled_lighting_to_glsl_460_oracle() {
        let mut compiler = VslCompiler::new();
        let out = compiler.compile_shader_to_glsl("tiled_lighting.vsl", TILED_LIGHTING_VSL);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(out.glsl.contains("#version 460 core"));
        assert!(
            out.glsl
                .contains("layout(set = 0, binding = 0, std140) uniform WorldUBO")
        );
        assert!(out.glsl.contains("shared uint s_tile_count;"));
    }

    #[test]
    fn evaluates_and_schedules_sunlit_pack_vsl() {
        let mut compiler = VslCompiler::new();
        let out = compiler.evaluate_pack_source("pack.vsl", PACK_VSL, &EvalConfig::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let plan = out.schedule.expect("valid topological schedule");
        assert_ne!(plan.ordered_passes, Vec::<SmolStr>::new());
    }

    // -------------------------------------------------------------------------
    // Stage 1-5 Verification Tests
    // -------------------------------------------------------------------------

    #[test]
    fn module_symbol_table_exports_traits_impls_and_reports_e0502_e0503() {
        // 1. `pub` trait + struct + impl exported across modules.
        let mut compiler = VslCompiler::new();
        let files_ok = vec![
            (
                "types.vsl".to_owned(),
                r"
                module pack::types;
                pub trait Weightable {
                    fn weight(self) -> f32;
                }
                pub struct Sample {
                    pub value: f32,
                }
                impl Weightable for Sample {
                    pub fn weight(self) -> f32 {
                        self.value * 2.0
                    }
                }
                fn private_only() -> f32 { 1.0 }
                "
                .to_owned(),
            ),
            (
                "consumer.vsl".to_owned(),
                r"
                module pack::consumer;
                import pack::types::{Weightable, Sample};
                @stage(compute(8, 8, 1))
                pub fn run(out_img: RWImage2D<rgba16f>) {
                    let s = Sample { value: 0.5 };
                    let w = s.weight();
                    out_img.store(ivec2(0, 0), vec4(w, w, w, 1.0));
                }
                "
                .to_owned(),
            ),
        ];
        let out_ok = compiler.compile_project(&files_ok, &CompileOptions::default());
        assert!(!out_ok.has_errors(), "{:?}", out_ok.diagnostics);
        assert_eq!(out_ok.units.len(), 1);

        // 2. Private symbol explicit import triggers E0502.
        let files_priv = vec![
            (
                "lib.vsl".to_owned(),
                "module pack::lib; fn secret() -> i32 { 42 }".to_owned(),
            ),
            (
                "main.vsl".to_owned(),
                "module pack::main; import pack::lib::{secret}; @stage(compute(8, 8, 1)) pub fn run() { let _x = secret(); }".to_owned(),
            ),
        ];
        let out_priv = compiler.compile_project(&files_priv, &CompileOptions::default());
        assert!(
            out_priv.diagnostics.iter().any(|d| d.code == "E0502"),
            "expected E0502 for private import, got {:?}",
            out_priv.diagnostics
        );

        // 3. Name collision between two imported modules triggers E0503.
        let files_collision = vec![
            (
                "a.vsl".to_owned(),
                "module pack::a; pub fn dup(x: i32) -> i32 { x + 1 }".to_owned(),
            ),
            (
                "b.vsl".to_owned(),
                "module pack::b; pub fn dup(x: i32) -> i32 { x + 2 }".to_owned(),
            ),
            (
                "main.vsl".to_owned(),
                r"
                module pack::main;
                import pack::a::{dup};
                import pack::b::{dup};
                @stage(compute(8, 8, 1))
                pub fn run() { let _y = dup(1); }
                "
                .to_owned(),
            ),
        ];
        let out_coll = compiler.compile_project(&files_collision, &CompileOptions::default());
        assert!(
            out_coll.diagnostics.iter().any(|d| d.code == "E0503"),
            "expected E0503 for symbol collision, got {:?}",
            out_coll.diagnostics
        );
    }

    #[test]
    fn lowers_mc_builtins_subgroup_exclusive_and_typed_vec3_param_without_stubs() {
        let mut schema = PackSchema::default();
        schema
            .params
            .insert(SmolStr::new_static("sky_tint"), (0, 12));
        schema
            .param_types
            .insert(SmolStr::new_static("sky_tint"), PackParamType::Vec3);

        let src = r"
            import mc::{world};
            import std::subgroup::{subgroup_ballot, subgroup_ballot_exclusive_bit_count};

            @stage(compute(8, 8, 1))
            pub fn test_builtins(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let uv = vec2(0.5, 0.5);
                let wp = mc::reconstruct_world_pos(uv, 0.25);
                let vdir = mc::view_dir(wp);
                let sky = mc::sample_sky(vdir);
                let tint = mc::param(sky_tint);
                let ballot = subgroup_ballot(gid.x < 4u);
                let lane_off = subgroup_ballot_exclusive_bit_count(ballot);
                let col = sky * tint + vec3(f32(lane_off));
                out_img.store(ivec2(gid.xy), vec4(col, 1.0));
            }
        ";

        let file = FileId(0);
        let parsed = parse_source(file, src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (hir, hir_diags) = lower_unit_with_schema(
            &parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::INT64_ATOMICS,
            Some(&schema),
        );
        assert!(!hir_diags.has_errors(), "{:?}", hir_diags.as_slice());

        let (_, entry_fn) = hir.entry_points().next().expect("entry point");
        let lowered = lower_entry_module_to_ir(
            &hir,
            entry_fn,
            &parsed.unit.items,
            Some(&schema),
            ShaderStage::Compute,
            Some([8, 8, 1]),
        );
        assert!(
            lowered.diagnostics.is_empty(),
            "unexpected lowering diagnostics: {:?}",
            lowered.diagnostics
        );

        let ir_fn = &lowered.ir_module.functions[0];
        let mut has_mat_mul = false;
        let mut has_normalize = false;
        let mut has_fmix = false;
        let mut has_excl_bit_count = false;
        let mut bda_f32_loads = 0usize;

        for (_, inst) in ir_fn.insts.iter() {
            match &inst.data {
                InstData::MatrixTimesVector { .. } => has_mat_mul = true,
                InstData::GlslStd450 {
                    op: GlslStd450Op::Normalize,
                    ..
                } => has_normalize = true,
                InstData::GlslStd450 {
                    op: GlslStd450Op::FMix,
                    ..
                } => has_fmix = true,
                InstData::SubgroupBallotExclusiveBitCount(_) => has_excl_bit_count = true,
                InstData::BdaLoad { .. } if inst.ir_ty == IrType::Scalar(ScalarIrTy::F32) => {
                    bda_f32_loads += 1;
                }
                _ => {}
            }
        }

        assert!(
            has_mat_mul,
            "expected MatrixTimesVector for mc::reconstruct_world_pos"
        );
        assert!(
            has_normalize,
            "expected GlslStd450Op::Normalize for mc::view_dir"
        );
        assert!(has_fmix, "expected GlslStd450Op::FMix for mc::sample_sky");
        assert!(
            has_excl_bit_count,
            "expected SubgroupBallotExclusiveBitCount instruction"
        );
        assert_eq!(
            bda_f32_loads, 3,
            "expected 3 typed f32 BDA loads for vec3 mc::param(sky_tint)"
        );
    }

    #[test]
    fn compiles_clouds_hook_chain_with_alpha_over_dispatch() {
        let mut compiler = VslCompiler::new();
        let layer_a = r"
            @hook(clouds)
            pub fn cumulus_layer(dir: vec3) -> vec4 {
                vec4(0.9, 0.92, 0.95, 0.5)
            }
        ";
        let layer_b = r"
            @hook(clouds)
            pub fn cirrus_layer(dir: vec3) -> vec4 {
                vec4(1.0, 0.98, 0.95, 0.25)
            }
        ";
        let unit = compiler
            .compile_hook_chain(
                HookStage::Clouds,
                &[("cumulus_layer", layer_a), ("cirrus_layer", layer_b)],
                &CompileOptions::default(),
            )
            .expect("valid clouds hook chain");
        assert_eq!(unit.entry_point, "vs_hook_chain_clouds");
        assert_eq!(unit.stage, ShaderStage::Compute);
        assert_eq!(unit.workgroup_size, Some([8, 8, 1]));
        assert_eq!(unit.spirv[0], 0x0723_0203);
    }

    #[test]
    fn pack_schema_preserves_param_types_and_spec_defaults() {
        let mut pipeline = PipelineIr::default();
        pipeline.settings.insert(
            SmolStr::new_static("enable_rt"),
            vsl_pack::SettingIr {
                name: SmolStr::new_static("enable_rt"),
                category: SmolStr::new_static("RT"),
                is_spec: true,
                label: SmolStr::new_static("Enable RT"),
                description: SmolStr::default(),
                default_value: SettingValueIr::Bool(true),
                range: None,
                step: None,
                tiers: (2, 3),
                enum_variants: Vec::new(),
            },
        );
        pipeline.settings.insert(
            SmolStr::new_static("tone_map"),
            vsl_pack::SettingIr {
                name: SmolStr::new_static("tone_map"),
                category: SmolStr::new_static("Post"),
                is_spec: true,
                label: SmolStr::new_static("Tone Map"),
                description: SmolStr::default(),
                default_value: SettingValueIr::EnumVariant(SmolStr::new_static("aces")),
                range: None,
                step: None,
                tiers: (0, 3),
                enum_variants: vec![
                    SmolStr::new_static("agx"),
                    SmolStr::new_static("aces"),
                    SmolStr::new_static("neutral"),
                ],
            },
        );
        pipeline.settings.insert(
            SmolStr::new_static("exposure"),
            vsl_pack::SettingIr {
                name: SmolStr::new_static("exposure"),
                category: SmolStr::new_static("Post"),
                is_spec: false,
                label: SmolStr::new_static("Exposure"),
                description: SmolStr::default(),
                default_value: SettingValueIr::Float(1.5),
                range: None,
                step: None,
                tiers: (0, 3),
                enum_variants: Vec::new(),
            },
        );
        pipeline.settings.insert(
            SmolStr::new_static("sun_tint"),
            vsl_pack::SettingIr {
                name: SmolStr::new_static("sun_tint"),
                category: SmolStr::new_static("Sky"),
                is_spec: false,
                label: SmolStr::new_static("Sun Tint"),
                description: SmolStr::default(),
                default_value: SettingValueIr::ColorRgb([1.0, 0.9, 0.8]),
                range: None,
                step: None,
                tiers: (0, 3),
                enum_variants: Vec::new(),
            },
        );

        let schema = pack_schema_from_pipeline(&pipeline);
        assert_eq!(
            schema.spec_defaults.get("enable_rt"),
            Some(&SpecConstantDefault {
                spec_id: 0,
                ty: PackParamType::Bool,
                default_bits: 1,
            })
        );
        assert_eq!(
            schema.spec_defaults.get("tone_map"),
            Some(&SpecConstantDefault {
                spec_id: 1,
                ty: PackParamType::Enum,
                default_bits: 1,
            })
        );
        assert_eq!(
            schema.param_types.get("exposure"),
            Some(&PackParamType::F32)
        );
        assert_eq!(
            schema.param_types.get("sun_tint"),
            Some(&PackParamType::Vec3)
        );
        // `exposure` at 0 (4B), `sun_tint` aligned to 16B (12B).
        assert_eq!(schema.params.get("exposure"), Some(&(0, 4)));
        assert_eq!(schema.params.get("sun_tint"), Some(&(16, 12)));
    }

    #[test]
    fn validates_pack_shader_bindings_against_compiled_units() {
        let mut compiler = VslCompiler::new();
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn fog_pass(out_fog: RWImage2D<rgba16f>) {
                out_fog.store(ivec2(0, 0), vec4(1.0, 1.0, 1.0, 1.0));
            }
        ";
        let out = compiler.compile_shader_source("fog.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);

        let mut pipeline = PipelineIr::default();
        pipeline.targets.insert(
            SmolStr::new_static("fog_tex"),
            TargetDecl {
                name: SmolStr::new_static("fog_tex"),
                format: ImageFormat::Rgba16f,
                scale: 1.0,
                layers: 1,
                mip_levels: 1,
                temporal: false,
                clear_color: None,
            },
        );
        let mut good_bindings = IndexMap::new();
        good_bindings.insert(
            SmolStr::new_static("out_fog"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("fog_tex"),
                history: false,
                is_write: true,
            },
        );
        pipeline.passes.push(PassNode {
            name: SmolStr::new_static("fog"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::fog::fog_pass"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings: good_bindings,
            meta: PassMeta::default(),
            dispatch_target: Some(SmolStr::new_static("fog_tex")),
            dispatch: None,
        });

        let ok_errs = validate_pack_shader_bindings(&pipeline, &out.units);
        assert!(ok_errs.is_empty(), "expected no errors, got: {ok_errs:?}");

        // Clear bindings -> must report missing `out_fog` binding.
        pipeline.passes[0].bindings.clear();
        let missing_errs = validate_pack_shader_bindings(&pipeline, &out.units);
        assert!(
            !missing_errs.is_empty(),
            "expected binding validation error when `out_fog` is omitted"
        );
    }

    #[test]
    fn persists_and_loads_l2_disk_cache_across_compiler_instances() {
        let cache_dir = std::env::temp_dir().join(format!(
            "vsl_driver_l2_disk_cache_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&cache_dir);

        let src = r"
            @stage(compute(8, 8, 1))
            pub fn cached_cs(out_img: RWImage2D<rgba16f>) {
                out_img.store(ivec2(0, 0), vec4(0.25, 0.5, 0.75, 1.0));
            }
        ";

        let opts = CompileOptions {
            l2_cache_dir: Some(cache_dir.clone()),
            ..CompileOptions::default()
        };

        let mut c1 = VslCompiler::with_l2_cache_dir(cache_dir.clone());
        let out1 = c1.compile_shader_source("cached.vsl", src, &opts);
        assert!(!out1.has_errors(), "{:?}", out1.diagnostics);
        assert_eq!(out1.units.len(), 1);
        let expected_path = cache::l2_cache_file_path(&cache_dir, &out1.units[0].cache_key);
        assert!(
            expected_path.exists(),
            "expected L2 disk cache file at {expected_path:?}"
        );

        // Second fresh compiler instance loads directly from disk L2 cache.
        let mut c2 = VslCompiler::with_l2_cache_dir(cache_dir.clone());
        let out2 = c2.compile_shader_source("cached.vsl", src, &opts);
        assert!(!out2.has_errors(), "{:?}", out2.diagnostics);
        assert_eq!(out2.units.len(), 1);
        assert_eq!(out2.units[0], out1.units[0]);
        assert_eq!(c2.l2_cache_len(), 1);

        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[test]
    fn compiles_multiple_entry_points_in_parallel_preserving_deterministic_order() {
        let mut compiler = VslCompiler::new();
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn entry_alpha(out_a: RWImage2D<rgba16f>) {
                out_a.store(ivec2(0, 0), vec4(1.0, 0.0, 0.0, 1.0));
            }

            @stage(compute(8, 8, 1))
            pub fn entry_beta(out_b: RWImage2D<rgba16f>) {
                out_b.store(ivec2(0, 0), vec4(0.0, 1.0, 0.0, 1.0));
            }

            @stage(compute(8, 8, 1))
            pub fn entry_gamma(out_c: RWImage2D<rgba16f>) {
                out_c.store(ivec2(0, 0), vec4(0.0, 0.0, 1.0, 1.0));
            }
        ";
        let out = compiler.compile_shader_source("multi.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let names: Vec<&str> = out.units.iter().map(|u| u.entry_point.as_str()).collect();
        assert_eq!(names, vec!["entry_alpha", "entry_beta", "entry_gamma"]);
        assert_eq!(compiler.l2_cache_len(), 3);
    }

    #[test]
    fn lowers_short_circuit_and_or_with_structured_if_guards() {
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn short_circuit_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let cond_and = gid.x < 16u && gid.y < 16u;
                let cond_or = gid.x == 0u || gid.y == 0u;
                let v = if cond_and && cond_or { 1.0 } else { 0.0 };
                out_img.store(ivec2(gid.xy), vec4(v, v, v, 1.0));
            }
        ";
        let file = FileId(0);
        let parsed = parse_source(file, src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (hir, hir_diags) = lower_unit_with_schema(
            &parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::INT64_ATOMICS,
            None,
        );
        assert!(!hir_diags.has_errors(), "{:?}", hir_diags.as_slice());
        let (_, entry_fn) = hir.entry_points().next().expect("entry point");
        let lowered = lower_entry_module_to_ir(
            &hir,
            entry_fn,
            &parsed.unit.items,
            None,
            ShaderStage::Compute,
            Some([8, 8, 1]),
        );
        assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);

        let ir_fn = &lowered.ir_module.functions[0];
        let if_count = ir_fn
            .body
            .iter()
            .filter(|s| matches!(s, vsl_ir::StructuredStmt::If { .. }))
            .count();
        assert!(
            if_count >= 3,
            "expected structured `If` guards for short-circuit `&&`/`||`, got {if_count}"
        );
        let has_eager_logical = ir_fn.insts.iter().any(|(_, inst)| {
            matches!(
                inst.data,
                InstData::LogicalAnd(_, _) | InstData::LogicalOr(_, _)
            )
        });
        assert!(
            !has_eager_logical,
            "short-circuit `&&`/`||` should lower to `If` rather than eager LogicalAnd/LogicalOr"
        );

        let mut compiler = VslCompiler::new();
        let out =
            compiler.compile_shader_source("short_circuit.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("short_circuit", &out.units[0].spirv);
    }

    #[test]
    fn lowers_unrolled_u32_for_loop_with_canonical_exit_and_continuing() {
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn unroll_u32_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let mut acc = 0u;
                #[unroll]
                for i in 0u..4u {
                    acc += i + gid.x;
                }
                let f = f32(acc);
                out_img.store(ivec2(gid.xy), vec4(f, f, f, 1.0));
            }
        ";
        let file = FileId(0);
        let parsed = parse_source(file, src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (hir, hir_diags) = lower_unit_with_schema(
            &parsed.unit,
            RenderTier::Tier2RayQuery,
            HardwareCaps::INT64_ATOMICS,
            None,
        );
        assert!(!hir_diags.has_errors(), "{:?}", hir_diags.as_slice());
        let (_, entry_fn) = hir.entry_points().next().expect("entry point");
        let lowered = lower_entry_module_to_ir(
            &hir,
            entry_fn,
            &parsed.unit.items,
            None,
            ShaderStage::Compute,
            Some([8, 8, 1]),
        );
        assert!(lowered.diagnostics.is_empty(), "{:?}", lowered.diagnostics);

        let ir_fn = &lowered.ir_module.functions[0];
        let loop_stmt = ir_fn
            .body
            .iter()
            .find_map(|s| match s {
                vsl_ir::StructuredStmt::Loop {
                    body,
                    continuing,
                    unroll,
                    trip_count,
                    induction_init,
                    ..
                } => Some((body, continuing, *unroll, *trip_count, *induction_init)),
                _ => None,
            })
            .expect("expected StructuredStmt::Loop");
        assert!(loop_stmt.2, "expected unroll flag to be set");
        assert_eq!(loop_stmt.3, Some(4));
        assert!(
            !loop_stmt.1.is_empty(),
            "expected canonical continuing increment even with #[unroll]"
        );
        assert!(
            matches!(
                loop_stmt.0.first(),
                Some(vsl_ir::StructuredStmt::If { else_body, .. })
                    if matches!(else_body.as_slice(), [vsl_ir::StructuredStmt::Break])
            ),
            "expected canonical exit guard `If {{ else: Break }}` in loop body"
        );
        let init_id = loop_stmt.4.expect("expected induction_init");
        assert_eq!(
            ir_fn.insts[init_id].ir_ty,
            IrType::Scalar(ScalarIrTy::U32),
            "expected u32 induction init type for `0u..4u`"
        );

        let mut compiler = VslCompiler::new();
        let out = compiler.compile_shader_source("unroll_u32.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("unroll_u32", &out.units[0].spirv);
    }

    #[test]
    fn links_transitive_private_and_upstream_module_helpers() {
        let mut compiler = VslCompiler::new();
        let files = vec![
            (
                "base.vsl".to_owned(),
                r"
                module pack::base;
                const SCALE: f32 = 2.0;
                fn private_leaf(x: f32) -> f32 {
                    x * SCALE
                }
                pub fn base_eval(x: f32) -> f32 {
                    private_leaf(x) + 1.0
                }
                "
                .to_owned(),
            ),
            (
                "mid.vsl".to_owned(),
                r"
                module pack::mid;
                import pack::base::{base_eval};
                fn private_mid(x: f32) -> f32 {
                    base_eval(x) * 0.5
                }
                pub fn mid_eval(x: f32) -> f32 {
                    private_mid(x)
                }
                "
                .to_owned(),
            ),
            (
                "main.vsl".to_owned(),
                r"
                module pack::main;
                import pack::mid::{mid_eval};
                @stage(compute(8, 8, 1))
                pub fn run(out_img: RWImage2D<rgba16f>) {
                    let v = mid_eval(3.0);
                    out_img.store(ivec2(0, 0), vec4(v, v, v, 1.0));
                }
                "
                .to_owned(),
            ),
        ];
        let out = compiler.compile_project(&files, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("transitive_modules", &out.units[0].spirv);
    }

    #[test]
    fn project_hook_chain_l2_cache_key_changes_when_hook_source_changes() {
        let mut compiler = VslCompiler::new();
        let files_v1 = vec![
            (
                "clouds_a.vsl".to_owned(),
                r"
                module pack::clouds_a;
                @hook(clouds)
                pub fn layer_a(dir: vec3) -> vec4 {
                    vec4(0.8, 0.85, 0.9, 0.4)
                }
                "
                .to_owned(),
            ),
            (
                "clouds_b.vsl".to_owned(),
                r"
                module pack::clouds_b;
                @hook(clouds)
                pub fn layer_b(dir: vec3) -> vec4 {
                    vec4(1.0, 0.98, 0.95, 0.2)
                }
                "
                .to_owned(),
            ),
        ];
        let files_v2 = vec![
            (
                "clouds_a.vsl".to_owned(),
                r"
                module pack::clouds_a;
                @hook(clouds)
                pub fn layer_a(dir: vec3) -> vec4 {
                    vec4(0.5, 0.6, 0.7, 0.5)
                }
                "
                .to_owned(),
            ),
            files_v1[1].clone(),
        ];

        let out1 = compiler.compile_project(&files_v1, &CompileOptions::default());
        assert!(!out1.has_errors(), "{:?}", out1.diagnostics);
        let chain1 = out1
            .units
            .iter()
            .find(|u| u.entry_point == "vs_hook_chain_clouds")
            .expect("expected synthesized clouds hook chain unit");

        let out2 = compiler.compile_project(&files_v2, &CompileOptions::default());
        assert!(!out2.has_errors(), "{:?}", out2.diagnostics);
        let chain2 = out2
            .units
            .iter()
            .find(|u| u.entry_point == "vs_hook_chain_clouds")
            .expect("expected synthesized clouds hook chain unit");

        assert_ne!(
            chain1.cache_key, chain2.cache_key,
            "hook chain L2 cache_key must change when any hook module source changes"
        );
    }

    #[test]
    fn user_rwbuffer_struct_layout_and_field_mutation_pass_spirv_val() {
        let src = r"
            module test::ssbo_struct;
            import std::sync::{Scope, Order, atomic_add};

            #[layout(std430)]
            struct Particle {
                pos: vec4,
                vel: vec3,
                mass: f32,
                count: u32,
                xform: mat4,
            }

            @stage(compute(64, 1, 1))
            pub fn step_particles(
                #[builtin(global_invocation_id)] gid: uvec3,
                in_buf: Buffer<Particle>,
                out_buf: RWBuffer<Particle>,
            ) {
                let i = gid.x;
                let p = in_buf[i].pos;
                let m = in_buf[i].xform;
                let next_pos = m * p + vec4(in_buf[i].vel * in_buf[i].mass, 0.0);
                out_buf[i].pos = next_pos;
                out_buf[i].mass = in_buf[i].mass + 1.0;
                out_buf[i].count = in_buf[i].count + 1u;
                atomic_add::<Scope::Device, Order::Relaxed>(&out_buf[i].count, 1u);
            }
        ";
        let mut compiler = VslCompiler::new();
        let out =
            compiler.compile_shader_source("ssbo_struct.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("user_rwbuffer_struct", &out.units[0].spirv);
    }

    #[test]
    fn gi_cache_accumulate_bda_struct_field_atomics_passes_spirv_val() {
        let gi_cache_src = include_str!("../../../examples/sunlit_vanilla/gi_cache.vsl");
        let entry_src = r"
            module pack::gi_entry;
            import pack::gi_cache::{CacheCell, cache_accumulate};

            struct PushData {
                keys_addr: ptr_mut<u64, align = 8>,
                cells_addr: ptr_mut<CacheCell, align = 16>,
                mask: u32,
            }

            #[cap(int64_atomics)]
            @stage(compute(8, 8, 1))
            pub fn accumulate_entry(
                #[builtin(global_invocation_id)] gid: uvec3,
                #[push_constant] pc: PushData,
            ) {
                let key = u64(gid.x) + (u64(gid.y) << 16u64) + 1u64;
                let ok = cache_accumulate(
                    pc.keys_addr,
                    pc.cells_addr,
                    pc.mask,
                    key,
                    vec3(0.25, 0.5, 0.75),
                );
                if ok {
                    let c = pc.cells_addr.offset(i64(gid.x & pc.mask));
                    let _r = c.rad_r;
                }
            }
        ";
        let mut compiler = VslCompiler::new();
        let files = vec![
            ("gi_cache.vsl".to_owned(), gi_cache_src.to_owned()),
            ("gi_entry.vsl".to_owned(), entry_src.to_owned()),
        ];
        let out = compiler.compile_project(&files, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("gi_cache_accumulate", &out.units[0].spirv);
    }

    #[test]
    fn bda_struct_direct_field_read_write_and_nested_field_pass_spirv_val() {
        let src = r"
            module test::bda_fields;
            import std::sync::{Scope, Order, atomic_add};

            #[layout(std430)]
            struct Inner {
                counter: u32,
                weight: f32,
            }

            #[layout(std430)]
            struct Outer {
                header: vec4,
                inner: Inner,
            }

            struct PushData {
                ptr: ptr_mut<Outer, align = 16>,
            }

            @stage(compute(8, 8, 1))
            pub fn test_bda(
                #[builtin(global_invocation_id)] gid: uvec3,
                #[push_constant] pc: PushData,
            ) {
                let o = pc.ptr.offset(i64(gid.x));
                let h = o.header;
                o.header = h + vec4(1.0);
                o.inner.weight += 0.5;
                atomic_add::<Scope::Device, Order::Relaxed>(&o.inner.counter, 1u);
            }
        ";
        let mut compiler = VslCompiler::new();
        let out = compiler.compile_shader_source("bda_fields.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("bda_struct_fields", &out.units[0].spirv);
    }

    #[test]
    fn pub_image_param_helper_cross_module_project_passes_spirv_val() {
        let helper_src = r"
            module pack::tex_helpers;

            pub fn sample_albedo(tex: Texture2D<f32>, samp: Sampler, uv: vec2) -> vec4 {
                tex.sample_lod(samp, uv, 0.0)
            }
        ";
        let entry_src = r"
            module pack::main_pass;
            import pack::tex_helpers::sample_albedo;

            @stage(compute(8, 8, 1))
            pub fn main_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                albedo: Texture2D<f32>,
                samp: Sampler,
                out_img: RWImage2D<rgba16f>,
            ) {
                let uv = vec2(gid.xy) / 64.0;
                let col = sample_albedo(albedo, samp, uv);
                out_img.store(ivec2(gid.xy), col);
            }
        ";
        let mut compiler = VslCompiler::new();
        let files = vec![
            ("tex_helpers.vsl".to_owned(), helper_src.to_owned()),
            ("main_pass.vsl".to_owned(), entry_src.to_owned()),
        ];
        let out = compiler.compile_project(&files, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("pub_image_helper", &out.units[0].spirv);
    }

    #[test]
    fn shadow_dref_sampling_op_type_image_depth_passes_spirv_val() {
        let src = r"
            module test::shadow_dref;

            @stage(compute(8, 8, 1))
            pub fn shadow_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                shadow_map: Texture2D<f32>,
                shadow_samp: SamplerShadow,
                out_img: RWImage2D<rgba16f>,
            ) {
                let uv = vec2(gid.xy) / 64.0;
                let vis = shadow_map.sample_compare_lod(shadow_samp, uv, 0.5, 0.0);
                out_img.store(ivec2(gid.xy), vec4(vis, vis, vis, 1.0));
            }
        ";
        let mut compiler = VslCompiler::new();
        let out =
            compiler.compile_shader_source("shadow_dref.vsl", src, &CompileOptions::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        validate_spirv_with_spirv_val("shadow_dref", &out.units[0].spirv);
    }

    #[test]
    fn standalone_abi_compute_explicit_bindings_and_no_world_ubo() {
        let src = r"
            module standalone::compute;

            #[layout(std430)]
            struct Item {
                val: vec4,
            }

            @stage(compute(32, 1, 1))
            pub fn cs_main(
                #[builtin(global_invocation_id)] gid: uvec3,
                #[binding(set = 0, binding = 0)] src_buf: Buffer<Item>,
                #[binding(set = 1, binding = 2)] dst_buf: RWBuffer<Item>,
                auto_img: RWImage2D<rgba16f>,
            ) {
                let v = src_buf[gid.x].val;
                dst_buf[gid.x].val = v * 2.0;
                auto_img.store(ivec2(gid.xy), v);
            }
        ";
        let mut compiler = VslCompiler::new();
        let mut opts = CompileOptions::for_tier(RenderTier::Tier0Raster);
        opts.abi = AbiProfile::Standalone;
        let out = compiler.compile_shader_source("standalone_cs.vsl", src, &opts);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        let unit = &out.units[0];
        assert!(
            !unit.descriptors.iter().any(|d| d.name == "world"),
            "Standalone mode must not auto-inject WorldUBO: {:?}",
            unit.descriptors
        );
        assert_eq!(unit.descriptors.len(), 3);
        assert_eq!(
            (unit.descriptors[0].set, unit.descriptors[0].binding),
            (0, 0)
        );
        assert_eq!(
            (unit.descriptors[1].set, unit.descriptors[1].binding),
            (0, 1)
        );
        assert_eq!(
            (unit.descriptors[2].set, unit.descriptors[2].binding),
            (1, 2)
        );
        validate_spirv_with_spirv_val("standalone_compute", &unit.spirv);
    }

    #[test]
    fn standalone_abi_unsized_bindless_texture_array_passes_spirv_val() {
        let src = r"
            module standalone::bindless;

            @stage(compute(8, 8, 1))
            pub fn bindless_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                #[binding(set = 0, binding = 0)] textures: Texture2D<f32>[],
                #[binding(set = 0, binding = 1)] samp: Sampler,
                #[binding(set = 0, binding = 2)] out_img: RWImage2D<rgba16f>,
            ) {
                let idx = nonuniform(gid.x & 15u);
                let col = textures[idx].sample_lod(samp, vec2(0.5, 0.5), 0.0);
                out_img.store(ivec2(gid.xy), col);
            }
        ";
        let mut compiler = VslCompiler::new();
        let mut opts = CompileOptions::for_tier(RenderTier::Tier0Raster);
        opts.abi = AbiProfile::Standalone;
        let out = compiler.compile_shader_source("standalone_bindless.vsl", src, &opts);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 1);
        let unit = &out.units[0];
        assert!(unit.descriptors[0].is_unsized);
        assert_eq!(unit.descriptors[0].count, None);
        assert_eq!(unit.descriptors[0].kind, DescriptorKind::SampledImage);
        validate_spirv_with_spirv_val("standalone_bindless", &unit.spirv);
    }

    #[test]
    fn standalone_abi_vertex_and_fragment_without_mc_pass_spirv_val() {
        let src = r"
            module standalone::raster;

            struct PushMvp {
                mvp: mat4,
                tint: vec4,
            }

            @stage(vertex)
            pub fn vs_main(
                #[location(0)] in_pos: vec3,
                #[location(1)] in_uv: vec2,
                #[push_constant] pc: PushMvp,
                #[builtin(position)] out clip_pos: vec4,
                #[location(0)] out v_uv: vec2,
            ) {
                clip_pos = pc.mvp * vec4(in_pos, 1.0);
                v_uv = in_uv;
            }

            @stage(fragment)
            pub fn fs_main(
                #[location(0)] v_uv: vec2,
                #[binding(set = 0, binding = 0)] albedo: Texture2D<f32>,
                #[binding(set = 0, binding = 1)] samp: Sampler,
                #[push_constant] pc: PushMvp,
                #[location(0)] out frag_color: vec4,
            ) {
                frag_color = albedo.sample(samp, v_uv) * pc.tint;
            }
        ";
        let mut compiler = VslCompiler::new();
        let mut opts = CompileOptions::for_tier(RenderTier::Tier0Raster);
        opts.abi = AbiProfile::Standalone;
        let out = compiler.compile_shader_source("standalone_raster.vsl", src, &opts);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert_eq!(out.units.len(), 2);
        for unit in &out.units {
            assert!(
                !unit.descriptors.iter().any(|d| d.name == "world"),
                "Standalone raster shader must not inject WorldUBO: {:?}",
                unit.descriptors
            );
            validate_spirv_with_spirv_val(&unit.entry_point, &unit.spirv);
        }
    }

    #[test]
    fn sunlit_vanilla_examples_all_compile_and_pass_spirv_val() {
        let pack_src = include_str!("../../../examples/sunlit_vanilla/pack.vsl");
        let pack_parsed = parse_source(FileId(0), pack_src);
        assert!(!pack_parsed.has_errors(), "{:?}", pack_parsed.diagnostics);
        let (pipeline, eval_diags) = vsl_pack::evaluate_pack_vsl(
            &pack_parsed.unit,
            &vsl_pack::EvalConfig {
                tier: RenderTier::Tier2RayQuery,
                ..vsl_pack::EvalConfig::default()
            },
        );
        assert!(!eval_diags.has_errors(), "{eval_diags:?}");
        let schema = pack_schema_from_pipeline(&pipeline);

        let mut opts = CompileOptions::for_tier(RenderTier::Tier2RayQuery);
        opts.pack_schema = Some(schema);

        let files = vec![
            (
                "gi_cache.vsl".to_owned(),
                include_str!("../../../examples/sunlit_vanilla/gi_cache.vsl").to_owned(),
            ),
            (
                "tiled_lighting.vsl".to_owned(),
                include_str!("../../../examples/sunlit_vanilla/tiled_lighting.vsl").to_owned(),
            ),
            (
                "reflections.vsl".to_owned(),
                include_str!("../../../examples/sunlit_vanilla/reflections.vsl").to_owned(),
            ),
            (
                "restir.vsl".to_owned(),
                include_str!("../../../examples/sunlit_vanilla/restir.vsl").to_owned(),
            ),
            (
                "fog.vsl".to_owned(),
                include_str!("../../../examples/sunlit_vanilla/fog.vsl").to_owned(),
            ),
            (
                "hooks_clouds.vsl".to_owned(),
                include_str!("../../../examples/sunlit_vanilla/hooks_clouds.vsl").to_owned(),
            ),
            (
                "hooks_water.vsl".to_owned(),
                include_str!("../../../examples/sunlit_vanilla/hooks_water.vsl").to_owned(),
            ),
        ];

        let mut compiler = VslCompiler::new();
        let out = compiler.compile_project(&files, &opts);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        assert!(!out.units.is_empty(), "expected compiled shader units");
        for unit in &out.units {
            validate_spirv_with_spirv_val(&unit.entry_point, &unit.spirv);
        }
    }
}
