use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use vsl_diagnostics::render_diagnostics;
use vsl_driver::{
    AbiProfile, CompileOptions, VslCompiler, pack_schema_from_pipeline,
    validate_pack_shader_bindings,
};
use vsl_mc::{HardwareCaps, RenderTier};
use vsl_pack::{
    EvalConfig, PipelineIr, SchedulePlan, SettingIr, SettingValueIr, desugar_pack_toml,
    validate_and_schedule,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliAbiProfile {
    Hyaloclast,
    Standalone,
}

impl From<CliAbiProfile> for AbiProfile {
    fn from(value: CliAbiProfile) -> Self {
        match value {
            CliAbiProfile::Hyaloclast => Self::Hyaloclast,
            CliAbiProfile::Standalone => Self::Standalone,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "vslc",
    version,
    about = "VSL 2.0 shader compiler, reflection dumper, and pack.vsl evaluator"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse, validate, and lower a `.vsl` shader module against VSL 2.0 SPIR-V legality rules.
    Check {
        /// Path to the `.vsl` source file.
        file: PathBuf,
        /// Optional `pack.vsl` or `pack.toml` manifest supplying `PackSchema` (params, specs, Set 3 slots).
        #[arg(long)]
        pack: Option<PathBuf>,
        /// Active Hyaloclast hardware tier (`0..=3`).
        #[arg(long, default_value_t = 2)]
        tier: u8,
        /// Target descriptor ABI profile (`hyaloclast` or `standalone`).
        #[arg(long, value_enum, default_value_t = CliAbiProfile::Hyaloclast)]
        abi: CliAbiProfile,
        /// Enable additional optional Vulkan hardware capability (repeatable: `float16`,
        /// `int64_atomics`, `float_atomics`, `ray_query`, `maximal_reconvergence`, `quad_control`).
        #[arg(long = "cap")]
        caps: Vec<String>,
    },
    /// Emit single-source-of-truth `CompiledShaderUnit` reflection metadata as JSON.
    Reflect {
        /// Path to the `.vsl` source file.
        file: PathBuf,
        /// Optional `pack.vsl` or `pack.toml` manifest supplying `PackSchema` (params, specs, Set 3 slots).
        #[arg(long)]
        pack: Option<PathBuf>,
        /// Active Hyaloclast hardware tier (`0..=3`).
        #[arg(long, default_value_t = 2)]
        tier: u8,
        /// Target descriptor ABI profile (`hyaloclast` or `standalone`).
        #[arg(long, value_enum, default_value_t = CliAbiProfile::Hyaloclast)]
        abi: CliAbiProfile,
        /// Enable additional optional Vulkan hardware capability (repeatable).
        #[arg(long = "cap")]
        caps: Vec<String>,
    },
    /// Evaluate and topologically schedule a `pack.vsl` or legacy `pack.toml` manifest.
    PackEval {
        /// Path to `pack.vsl` or `pack.toml`.
        file: PathBuf,
        /// Active Hyaloclast hardware tier (`0..=3`).
        #[arg(long, default_value_t = 2)]
        tier: u8,
        /// Optional preset name to apply before evaluating `pipeline`.
        #[arg(long)]
        preset: Option<String>,
        /// Parameter overrides in `key=value` format (repeatable; supports `bool`, `int`, `float`,
        /// `vec3(r,g,b)` / `r,g,b`, and enum variants).
        #[arg(long = "param")]
        params: Vec<String>,
        /// Emit formatted JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Build an entire shader pack directory (or `pack.vsl`/`pack.toml`), compiling all `.vsl`
    /// modules, cross-validating render-graph bindings, and writing `.spv` binaries + manifest.
    BuildPack {
        /// Path to a pack directory containing `pack.vsl`/`pack.toml`, or directly to the manifest file.
        path: PathBuf,
        /// Active Hyaloclast hardware tier (`0..=3`).
        #[arg(long, default_value_t = 2)]
        tier: u8,
        /// Target descriptor ABI profile (`hyaloclast` or `standalone`).
        #[arg(long, value_enum, default_value_t = CliAbiProfile::Hyaloclast)]
        abi: CliAbiProfile,
        /// Optional preset name to apply before evaluating `pipeline`.
        #[arg(long)]
        preset: Option<String>,
        /// Parameter overrides in `key=value` format (repeatable).
        #[arg(long = "param")]
        params: Vec<String>,
        /// Enable additional optional Vulkan hardware capability (repeatable).
        #[arg(long = "cap")]
        caps: Vec<String>,
        /// Output directory for compiled `.spv` binaries, GLSL hook slots, and `manifest.json`.
        #[arg(short = 'o', long = "out-dir")]
        out_dir: PathBuf,
        /// Emit the full build manifest JSON to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Compile a `.vsl` module and write SPIR-V 1.6 binaries for all (or a selected) entry point.
    EmitSpv {
        /// Path to the `.vsl` source file.
        file: PathBuf,
        /// Optional `pack.vsl` or `pack.toml` manifest supplying `PackSchema` (params, specs, Set 3 slots).
        #[arg(long)]
        pack: Option<PathBuf>,
        /// Output `.spv` file path (or output directory when emitting multiple entry points).
        #[arg(short, long)]
        out: PathBuf,
        /// Optional entry-point name filter.
        #[arg(long)]
        entry: Option<String>,
        /// Active Hyaloclast hardware tier (`0..=3`).
        #[arg(long, default_value_t = 2)]
        tier: u8,
        /// Target descriptor ABI profile (`hyaloclast` or `standalone`).
        #[arg(long, value_enum, default_value_t = CliAbiProfile::Hyaloclast)]
        abi: CliAbiProfile,
        /// Enable additional optional Vulkan hardware capability (repeatable).
        #[arg(long = "cap")]
        caps: Vec<String>,
    },
    /// Compile a `.vsl` module through the Stage A GLSL 460 oracle backend.
    EmitGlsl {
        /// Path to the `.vsl` source file.
        file: PathBuf,
        /// Optional output `.glsl` file path (prints to stdout when omitted).
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Target descriptor ABI profile (`hyaloclast` or `standalone`).
        #[arg(long, value_enum, default_value_t = CliAbiProfile::Hyaloclast)]
        abi: CliAbiProfile,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(io::stderr)
        .init();

    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("vslc error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, Box<dyn std::error::Error>> {
    match cli.command {
        Command::Check {
            file,
            pack,
            tier,
            abi,
            caps,
        } => {
            let source = fs::read_to_string(&file)?;
            let render_tier = parse_tier(tier)?;
            let mut opts = CompileOptions::for_tier(render_tier);
            opts.abi = abi.into();
            apply_cli_caps(&mut opts, &caps)?;
            let mut compiler = VslCompiler::new();
            if let Some(ref pack_path) = pack
                && !apply_pack_schema(&mut compiler, pack_path, render_tier, &mut opts)?
            {
                return Ok(ExitCode::FAILURE);
            }
            let output = compiler.compile_shader_source(file.display().to_string(), &source, &opts);
            render_diagnostics(&compiler.sources, &output.diagnostics, &mut io::stderr())?;
            if output.has_errors() {
                return Ok(ExitCode::FAILURE);
            }
            println!(
                "ok: {} active entry point(s), {} pruned, {} hook(s) ({} GLSL bridge), {} push-constant struct(s)",
                output.units.len(),
                output.hir.pruned_functions,
                output.summary.hooks.len(),
                output.hook_glsl_slots.len(),
                output.summary.push_constant_structs.len(),
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Reflect {
            file,
            pack,
            tier,
            abi,
            caps,
        } => {
            let source = fs::read_to_string(&file)?;
            let render_tier = parse_tier(tier)?;
            let mut opts = CompileOptions::for_tier(render_tier);
            opts.abi = abi.into();
            apply_cli_caps(&mut opts, &caps)?;
            let mut compiler = VslCompiler::new();
            if let Some(ref pack_path) = pack
                && !apply_pack_schema(&mut compiler, pack_path, render_tier, &mut opts)?
            {
                return Ok(ExitCode::FAILURE);
            }
            let output = compiler.compile_shader_source(file.display().to_string(), &source, &opts);
            render_diagnostics(&compiler.sources, &output.diagnostics, &mut io::stderr())?;
            if output.has_errors() {
                return Ok(ExitCode::FAILURE);
            }
            println!("{}", serde_json::to_string_pretty(&output.units)?);
            Ok(ExitCode::SUCCESS)
        }
        Command::PackEval {
            file,
            tier,
            preset,
            params,
            json,
        } => {
            let render_tier = parse_tier(tier)?;
            let parsed_params = parse_cli_params(&params)?;
            let Some((pipeline, schedule)) =
                evaluate_pack_manifest(&file, render_tier, preset.as_deref(), &parsed_params)?
            else {
                return Ok(ExitCode::FAILURE);
            };

            if json {
                let payload = serde_json::json!({
                    "pipeline": pipeline,
                    "schedule": schedule,
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                let kind = if is_toml_path(&file) {
                    "pack.toml"
                } else {
                    "pack"
                };
                println!(
                    "{kind} `{}` ({}) — mode={:?}, hooks={}, targets={}, passes={}, scheduled={}",
                    pipeline.metadata.id,
                    pipeline.metadata.version,
                    pipeline.mode,
                    pipeline.hooks.len(),
                    pipeline.targets.len(),
                    pipeline.passes.len(),
                    schedule.ordered_passes.len(),
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::BuildPack {
            path,
            tier,
            abi,
            preset,
            params,
            caps,
            out_dir,
            json,
        } => {
            let render_tier = parse_tier(tier)?;
            let parsed_params = parse_cli_params(&params)?;
            let (pack_dir, manifest_path) = resolve_pack_manifest_path(&path)?;

            let Some((pipeline, schedule)) = evaluate_pack_manifest(
                &manifest_path,
                render_tier,
                preset.as_deref(),
                &parsed_params,
            )?
            else {
                return Ok(ExitCode::FAILURE);
            };

            let shader_paths = collect_pack_shader_files(&pack_dir, &manifest_path)?;
            let mut project_files: Vec<(String, String)> = Vec::with_capacity(shader_paths.len());
            for shader_path in &shader_paths {
                let display_name = shader_path
                    .strip_prefix(&pack_dir)
                    .unwrap_or(shader_path)
                    .display()
                    .to_string();
                let source = fs::read_to_string(shader_path)?;
                project_files.push((display_name, source));
            }

            let mut opts = CompileOptions::for_tier(render_tier);
            opts.abi = abi.into();
            apply_cli_caps(&mut opts, &caps)?;
            opts.pack_schema = Some(pack_schema_from_pipeline(&pipeline));

            let mut compiler = VslCompiler::new();
            let project_out = compiler.compile_project(&project_files, &opts);
            render_diagnostics(
                &compiler.sources,
                &project_out.diagnostics,
                &mut io::stderr(),
            )?;
            if project_out.has_errors() {
                return Ok(ExitCode::FAILURE);
            }

            let binding_errors = validate_pack_shader_bindings(&pipeline, &project_out.units);
            if !binding_errors.is_empty() {
                for err in &binding_errors {
                    eprintln!("vslc binding error: {err}");
                }
                return Ok(ExitCode::FAILURE);
            }

            fs::create_dir_all(&out_dir)?;

            let mut shader_entries = Vec::with_capacity(project_out.units.len());
            for unit in &project_out.units {
                let spv_name = format!("{}.spv", unit.entry_point);
                let spv_path = out_dir.join(&spv_name);
                let mut bytes = Vec::with_capacity(unit.spirv.len() * 4);
                for word in &unit.spirv {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                fs::write(&spv_path, bytes)?;
                shader_entries.push(serde_json::json!({
                    "entry_point": unit.entry_point,
                    "stage": unit.stage,
                    "spv_file": spv_name,
                    "spirv_words": unit.spirv.len(),
                    "descriptors": unit.descriptors,
                    "push_constants": unit.push_constants,
                    "push_constant_size": unit.push_constant_size,
                    "workgroup_size": unit.workgroup_size,
                    "required_caps": unit.required_caps,
                    "used_specs": unit.used_specs,
                    "used_params": unit.used_params,
                    "debug_channels": unit.debug_channels,
                    "ray_interfaces": unit.ray_interfaces,
                    "cache_key": unit.cache_key,
                }));
            }

            let mut glsl_entries = Vec::with_capacity(project_out.hook_glsl_slots.len());
            for (stage, glsl) in &project_out.hook_glsl_slots {
                let glsl_name = format!("hook_{}.glsl", stage.as_str());
                let glsl_path = out_dir.join(&glsl_name);
                fs::write(&glsl_path, glsl)?;
                glsl_entries.push(serde_json::json!({
                    "hook_stage": stage.as_str(),
                    "glsl_file": glsl_name,
                }));
            }

            let manifest = serde_json::json!({
                "tier": render_tier.as_u8(),
                "pipeline": pipeline,
                "schedule": schedule,
                "module_order": project_out.module_order,
                "shaders": shader_entries,
                "hook_glsl_slots": glsl_entries,
            });
            let manifest_json = serde_json::to_string_pretty(&manifest)?;
            let manifest_out_path = out_dir.join("manifest.json");
            fs::write(&manifest_out_path, &manifest_json)?;

            if json {
                println!("{manifest_json}");
            } else {
                println!(
                    "built pack `{}` ({}) — {} module(s), {} SPIR-V unit(s), {} GLSL hook slot(s), {} scheduled pass(es) -> {}",
                    pipeline.metadata.id,
                    pipeline.metadata.version,
                    project_out.module_order.len(),
                    project_out.units.len(),
                    project_out.hook_glsl_slots.len(),
                    schedule.ordered_passes.len(),
                    out_dir.display(),
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::EmitSpv {
            file,
            pack,
            out,
            entry,
            tier,
            abi,
            caps,
        } => {
            let source = fs::read_to_string(&file)?;
            let render_tier = parse_tier(tier)?;
            let mut opts = CompileOptions::for_tier(render_tier);
            opts.abi = abi.into();
            apply_cli_caps(&mut opts, &caps)?;
            let mut compiler = VslCompiler::new();
            if let Some(ref pack_path) = pack
                && !apply_pack_schema(&mut compiler, pack_path, render_tier, &mut opts)?
            {
                return Ok(ExitCode::FAILURE);
            }
            let output = compiler.compile_shader_source(file.display().to_string(), &source, &opts);
            render_diagnostics(&compiler.sources, &output.diagnostics, &mut io::stderr())?;
            if output.has_errors() {
                return Ok(ExitCode::FAILURE);
            }

            let selected: Vec<_> = output
                .units
                .iter()
                .filter(|u| {
                    entry
                        .as_deref()
                        .is_none_or(|wanted| u.entry_point.as_str() == wanted)
                })
                .collect();

            if selected.is_empty() {
                return Err("no matching active `@stage(...)` entry points found".into());
            }

            let treat_as_dir = out.is_dir() || (selected.len() > 1 && out.extension().is_none());
            if treat_as_dir {
                fs::create_dir_all(&out)?;
            }

            let multi = selected.len() > 1 || treat_as_dir;
            for unit in selected {
                let target_path = if treat_as_dir {
                    out.join(format!("{}.spv", unit.entry_point))
                } else if multi {
                    let stem = out
                        .file_stem()
                        .and_then(std::ffi::OsStr::to_str)
                        .unwrap_or("shader");
                    out.with_file_name(format!("{stem}_{}.spv", unit.entry_point))
                } else {
                    out.clone()
                };

                if let Some(parent) = target_path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    fs::create_dir_all(parent)?;
                }

                let mut bytes = Vec::with_capacity(unit.spirv.len() * 4);
                for word in &unit.spirv {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                fs::write(&target_path, bytes)?;
                println!(
                    "wrote {} words ({} bytes) for entry `{}` ({} descriptors) -> {}",
                    unit.spirv.len(),
                    unit.spirv.len() * 4,
                    unit.entry_point,
                    unit.descriptors.len(),
                    target_path.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::EmitGlsl { file, out, abi } => {
            let source = fs::read_to_string(&file)?;
            let mut compiler = VslCompiler::new();
            let output = compiler.compile_shader_to_glsl_with_abi(
                file.display().to_string(),
                &source,
                abi.into(),
            );
            render_diagnostics(&compiler.sources, &output.diagnostics, &mut io::stderr())?;
            if output.has_errors() {
                return Ok(ExitCode::FAILURE);
            }
            if let Some(path) = out {
                fs::write(&path, &output.glsl)?;
                println!("wrote GLSL 460 oracle -> {}", path.display());
            } else {
                print!("{}", output.glsl);
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn is_toml_path(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
}

fn parse_cli_params(params: &[String]) -> Result<Vec<(String, SettingValueIr)>, String> {
    let mut parsed_params: Vec<(String, SettingValueIr)> = Vec::with_capacity(params.len());
    for raw in params {
        let (key, val_str) = raw
            .split_once('=')
            .ok_or_else(|| format!("invalid `--param` `{raw}`: expected `key=value`"))?;
        parsed_params.push((
            key.trim().to_owned(),
            parse_cli_setting_value(val_str.trim())?,
        ));
    }
    Ok(parsed_params)
}

fn resolve_pack_manifest_path(path: &Path) -> Result<(PathBuf, PathBuf), String> {
    if path.is_dir() {
        let vsl_manifest = path.join("pack.vsl");
        if vsl_manifest.is_file() {
            return Ok((path.to_path_buf(), vsl_manifest));
        }
        let toml_manifest = path.join("pack.toml");
        if toml_manifest.is_file() {
            return Ok((path.to_path_buf(), toml_manifest));
        }
        return Err(format!(
            "no `pack.vsl` or `pack.toml` found in directory `{}`",
            path.display()
        ));
    }
    if !path.is_file() {
        return Err(format!("pack path `{}` does not exist", path.display()));
    }
    let pack_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    Ok((pack_dir, path.to_path_buf()))
}

fn collect_pack_shader_files(
    pack_dir: &Path,
    manifest_path: &Path,
) -> Result<Vec<PathBuf>, io::Error> {
    let manifest_canon = fs::canonicalize(manifest_path).ok();
    let mut out = Vec::new();
    collect_vsl_files_recursive(pack_dir, manifest_canon.as_deref(), &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_vsl_files_recursive(
    dir: &Path,
    skip_manifest: Option<&Path>,
    out: &mut Vec<PathBuf>,
) -> Result<(), io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_vsl_files_recursive(&entry_path, skip_manifest, out)?;
            continue;
        }
        let is_vsl = entry_path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|ext| ext.eq_ignore_ascii_case("vsl"));
        if !is_vsl {
            continue;
        }
        let is_manifest_name = entry_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| name.eq_ignore_ascii_case("pack.vsl"));
        if is_manifest_name {
            continue;
        }
        if let Some(skip) = skip_manifest
            && fs::canonicalize(&entry_path).ok().as_deref() == Some(skip)
        {
            continue;
        }
        out.push(entry_path);
    }
    Ok(())
}

/// Evaluates and schedules a `pack.vsl` or `pack.toml` manifest with strict upfront validation
/// of pack tier bounds, `--preset`, and `--param` overrides (including `validate_setting_tier`)
/// before running the final `pipeline` evaluation, inserting into `compiler.sources` at most once.
fn evaluate_pack_manifest(
    file: &Path,
    render_tier: RenderTier,
    preset: Option<&str>,
    parsed_params: &[(String, SettingValueIr)],
) -> Result<Option<(PipelineIr, SchedulePlan)>, Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;

    if is_toml_path(file) {
        let mut pipeline = desugar_pack_toml(&source)?;
        apply_toml_overrides(&mut pipeline, render_tier, preset, parsed_params)?;
        let schedule = validate_and_schedule(&pipeline).map_err(|errs| errs.join("; "))?;
        return Ok(Some((pipeline, schedule)));
    }

    let file_display = file.display().to_string();
    let probe_cfg = EvalConfig {
        tier: render_tier,
        ..EvalConfig::default()
    };

    // Probe settings, presets, and pack metadata in an isolated compiler instance so the final
    // compiler session never inserts duplicate entries into `compiler.sources`.
    let mut probe_compiler = VslCompiler::new();
    let probe = probe_compiler.evaluate_pack_source(file_display.clone(), &source, &probe_cfg);
    if probe.has_errors() {
        render_diagnostics(
            &probe_compiler.sources,
            &probe.diagnostics,
            &mut io::stderr(),
        )?;
        return Ok(None);
    }

    if preset.is_none() && parsed_params.is_empty() {
        render_diagnostics(
            &probe_compiler.sources,
            &probe.diagnostics,
            &mut io::stderr(),
        )?;
        let schedule = probe
            .schedule
            .ok_or_else(|| "pack evaluation succeeded without producing a schedule".to_owned())?;
        return Ok(Some((probe.pipeline, schedule)));
    }

    let mut cfg = EvalConfig {
        tier: render_tier,
        ..EvalConfig::default()
    };
    validate_and_collect_pack_overrides(
        &probe.pipeline,
        render_tier,
        preset,
        parsed_params,
        &mut cfg,
    )?;

    let mut compiler = VslCompiler::new();
    let output = compiler.evaluate_pack_source(file_display, &source, &cfg);
    render_diagnostics(&compiler.sources, &output.diagnostics, &mut io::stderr())?;
    if output.has_errors() {
        return Ok(None);
    }
    let schedule = output
        .schedule
        .ok_or_else(|| "pack evaluation succeeded without producing a schedule".to_owned())?;
    Ok(Some((output.pipeline, schedule)))
}

fn validate_and_collect_pack_overrides(
    pipeline: &PipelineIr,
    render_tier: RenderTier,
    preset: Option<&str>,
    params: &[(String, SettingValueIr)],
    cfg: &mut EvalConfig,
) -> Result<(), String> {
    let active_u8 = render_tier.as_u8();
    let meta = &pipeline.metadata;
    if active_u8 < meta.tier_min || active_u8 > meta.tier_max {
        return Err(format!(
            "pack `{}` supports tiers {}..={}, but tier {} was requested",
            meta.name, meta.tier_min, meta.tier_max, active_u8
        ));
    }

    if let Some(preset_name) = preset {
        let found = pipeline
            .presets
            .iter()
            .find(|p| p.name == preset_name)
            .ok_or_else(|| format!("unknown preset `{preset_name}`"))?;
        let (min_tier, max_tier) = found.tiers;
        if !(min_tier..=max_tier).contains(&active_u8) {
            return Err(format!(
                "preset `{preset_name}` is restricted to tiers {min_tier}..={max_tier} (active tier is {active_u8})"
            ));
        }
        for (k, v) in &found.overrides {
            let Some(setting) = pipeline.settings.get(k.as_str()) else {
                return Err(format!(
                    "preset `{preset_name}` references unknown setting `{k}`"
                ));
            };
            validate_setting_tier(setting, render_tier)?;
            let coerced = validate_setting_override(k.as_str(), setting, v)?;
            cfg.param_overrides.insert(k.clone(), coerced);
        }
    }

    for (key, val) in params {
        let Some(setting) = pipeline.settings.get(key.as_str()) else {
            return Err(format!(
                "unknown `--param` key `{key}` (not declared in `settings`)"
            ));
        };
        validate_setting_tier(setting, render_tier)?;
        let coerced = validate_setting_override(key, setting, val)?;
        cfg.param_overrides.insert(key.as_str().into(), coerced);
    }

    Ok(())
}

fn apply_pack_schema(
    compiler: &mut VslCompiler,
    pack_path: &Path,
    tier: RenderTier,
    opts: &mut CompileOptions,
) -> Result<bool, Box<dyn std::error::Error>> {
    let pack_source = fs::read_to_string(pack_path)?;
    if is_toml_path(pack_path) {
        let pipeline = desugar_pack_toml(&pack_source)?;
        opts.pack_schema = Some(pack_schema_from_pipeline(&pipeline));
        return Ok(true);
    }
    let cfg = EvalConfig {
        tier,
        ..EvalConfig::default()
    };
    let pack_out =
        compiler.evaluate_pack_source(pack_path.display().to_string(), &pack_source, &cfg);
    render_diagnostics(&compiler.sources, &pack_out.diagnostics, &mut io::stderr())?;
    if pack_out.has_errors() {
        return Ok(false);
    }
    opts.pack_schema = Some(pack_schema_from_pipeline(&pack_out.pipeline));
    Ok(true)
}

fn parse_tier(tier: u8) -> Result<RenderTier, String> {
    RenderTier::from_u8(tier).ok_or_else(|| format!("invalid tier `{tier}`: expected 0..=3"))
}

fn apply_cli_caps(opts: &mut CompileOptions, caps: &[String]) -> Result<(), String> {
    for raw in caps {
        let name = raw.trim();
        let cap = HardwareCaps::from_cap_name(name).ok_or_else(|| {
            format!(
                "unknown `--cap` `{name}`: expected one of float16, int64_atomics, float_atomics, ray_query, maximal_reconvergence, quad_control"
            )
        })?;
        opts.active_caps |= cap;
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn validate_setting_override(
    key: &str,
    setting: &SettingIr,
    val: &SettingValueIr,
) -> Result<SettingValueIr, String> {
    let coerced = match (&setting.default_value, val) {
        (SettingValueIr::Bool(_), SettingValueIr::Bool(b)) => SettingValueIr::Bool(*b),
        (SettingValueIr::Int(_), SettingValueIr::Int(i)) => SettingValueIr::Int(*i),
        (SettingValueIr::Float(_), SettingValueIr::Float(f)) => SettingValueIr::Float(*f),
        (SettingValueIr::Float(_), SettingValueIr::Int(i)) => SettingValueIr::Float(*i as f32),
        (SettingValueIr::ColorRgb(_), SettingValueIr::ColorRgb(c)) => SettingValueIr::ColorRgb(*c),
        (SettingValueIr::EnumVariant(_), SettingValueIr::EnumVariant(v)) => {
            if !setting.enum_variants.is_empty() && !setting.enum_variants.contains(v) {
                return Err(format!(
                    "invalid `--param` value `{v}` for enum setting `{key}` (expected one of: {:?})",
                    setting.enum_variants
                ));
            }
            SettingValueIr::EnumVariant(v.clone())
        }
        _ => {
            return Err(format!(
                "type mismatch for `--param` `{key}`: value `{val:?}` does not match setting default `{:?}`",
                setting.default_value
            ));
        }
    };

    if let Some((min, max)) = setting.range {
        let numeric = match coerced {
            SettingValueIr::Int(i) => Some(i as f64),
            SettingValueIr::Float(f) => Some(f64::from(f)),
            _ => None,
        };
        if let Some(n) = numeric
            && (n < min || n > max)
        {
            return Err(format!(
                "`--param` `{key}` value {n} is outside declared range [{min}, {max}]"
            ));
        }
    }

    Ok(coerced)
}

fn validate_setting_tier(setting: &SettingIr, render_tier: RenderTier) -> Result<(), String> {
    let active = render_tier.as_u8();
    if active < setting.tiers.0 || active > setting.tiers.1 {
        return Err(format!(
            "setting `{}` is only available on tiers {}..={}",
            setting.name, setting.tiers.0, setting.tiers.1
        ));
    }
    Ok(())
}

fn apply_toml_overrides(
    pipeline: &mut PipelineIr,
    render_tier: RenderTier,
    preset: Option<&str>,
    params: &[(String, SettingValueIr)],
) -> Result<(), String> {
    let active_u8 = render_tier.as_u8();
    let meta = &pipeline.metadata;
    if active_u8 < meta.tier_min || active_u8 > meta.tier_max {
        return Err(format!(
            "pack `{}` supports tiers {}..={}, but tier {} was requested",
            meta.name, meta.tier_min, meta.tier_max, active_u8
        ));
    }

    if let Some(preset_name) = preset {
        let found = pipeline
            .presets
            .iter()
            .find(|p| p.name == preset_name)
            .cloned()
            .ok_or_else(|| format!("unknown preset `{preset_name}`"))?;
        let (min_tier, max_tier) = found.tiers;
        if !(min_tier..=max_tier).contains(&active_u8) {
            return Err(format!(
                "preset `{preset_name}` is restricted to tiers {min_tier}..={max_tier} (active tier is {active_u8})"
            ));
        }
        for (k, v) in found.overrides {
            if let Some(setting) = pipeline.settings.get_mut(&k) {
                validate_setting_tier(setting, render_tier)?;
                let coerced = validate_setting_override(&k, setting, &v)?;
                setting.default_value = coerced;
            }
        }
    }

    for (key, val) in params {
        let Some(setting) = pipeline.settings.get_mut(key.as_str()) else {
            return Err(format!(
                "unknown `--param` key `{key}` (not declared in `settings`)"
            ));
        };
        validate_setting_tier(setting, render_tier)?;
        let coerced = validate_setting_override(key, setting, val)?;
        setting.default_value = coerced;
    }

    Ok(())
}

fn parse_cli_setting_value(raw: &str) -> Result<SettingValueIr, String> {
    if raw == "true" {
        return Ok(SettingValueIr::Bool(true));
    }
    if raw == "false" {
        return Ok(SettingValueIr::Bool(false));
    }
    if let Some(inner) = raw.strip_prefix("vec3(").and_then(|s| s.strip_suffix(')')) {
        return parse_vec3_components(inner, raw);
    }
    if raw.contains(',') {
        return parse_vec3_components(raw, raw);
    }
    if let Ok(i) = raw.parse::<i64>() {
        return Ok(SettingValueIr::Int(i));
    }
    if let Ok(f) = raw.parse::<f32>() {
        return Ok(SettingValueIr::Float(f));
    }
    Ok(SettingValueIr::EnumVariant(raw.into()))
}

fn parse_vec3_components(inner: &str, raw: &str) -> Result<SettingValueIr, String> {
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "invalid vec3 `--param` value `{raw}`: expected 3 comma-separated floats"
        ));
    }
    let r = parts[0]
        .parse::<f32>()
        .map_err(|_| format!("invalid float `{}` in `{raw}`", parts[0]))?;
    let g = parts[1]
        .parse::<f32>()
        .map_err(|_| format!("invalid float `{}` in `{raw}`", parts[1]))?;
    let b = parts[2]
        .parse::<f32>()
        .map_err(|_| format!("invalid float `{}` in `{raw}`", parts[2]))?;
    Ok(SettingValueIr::ColorRgb([r, g, b]))
}
