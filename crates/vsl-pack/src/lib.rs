//! Deterministic `pack.vsl` sandbox evaluator, `PipelineIR` DAG, and `pack.toml` v2 desugarer.

pub mod eval;
pub mod ir;
pub mod schedule;
pub mod toml_desugar;

pub use eval::{EvalConfig, MAX_EVAL_STEPS, MAX_LOOP_ITERS, evaluate_pack_vsl};
pub use ir::{
    HookRegistration, PackMetadata, PassBindingRef, PassDispatch, PassKind, PassMeta, PassNode,
    PassRef, PipelineIr, PipelineMode, PresetIr, ResourceTextureDecl, SchedulePlan, SettingIr,
    SettingValueIr, TargetDecl,
};
pub use schedule::{validate_and_schedule, validate_pass_bindings_against_shader};
pub use toml_desugar::{TomlDesugarError, desugar_pack_toml};

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use smol_str::SmolStr;
    use vsl_diagnostics::FileId;
    use vsl_mc::{BuiltinResourceId, CanonicalPass, ImageFormat, RenderTier};
    use vsl_syntax::parse_source;

    const SUNLIT_PACK_VSL: &str = include_str!("../../../examples/sunlit_vanilla/pack.vsl");

    #[test]
    fn evaluates_and_schedules_sunlit_vanilla_pack() {
        let parsed = parse_source(FileId(0), SUNLIT_PACK_VSL);
        assert!(
            !parsed.has_errors(),
            "unexpected parse diagnostics: {:?}",
            parsed.diagnostics
        );

        let mut overrides = IndexMap::new();
        overrides.insert(
            SmolStr::new_static("custom_rayquery_refl"),
            SettingValueIr::Bool(true),
        );
        let cfg = EvalConfig {
            tier: RenderTier::Tier2RayQuery,
            param_overrides: overrides,
        };

        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &cfg);
        assert!(
            !diags.has_errors(),
            "unexpected eval diagnostics: {:?}",
            diags.as_slice()
        );

        // Verify extended SettingIr metadata.
        let sun_intensity = ir
            .settings
            .get("sun_intensity")
            .expect("sun_intensity setting");
        assert_eq!(sun_intensity.range, Some((0.5, 8.0)));
        assert_eq!(sun_intensity.step, Some(0.1));
        assert_eq!(sun_intensity.tiers, (0, 3));
        assert_eq!(sun_intensity.enum_variants, Vec::<SmolStr>::new());

        let pt_bounces = ir
            .settings
            .get("pt_max_bounces")
            .expect("pt_max_bounces setting");
        assert!(pt_bounces.is_spec);
        assert_eq!(pt_bounces.range, Some((1.0, 4.0)));
        assert_eq!(pt_bounces.step, Some(1.0));
        assert_eq!(pt_bounces.tiers, (3, 3));

        let tone_mapping = ir
            .settings
            .get("tone_mapping")
            .expect("tone_mapping setting");
        assert_eq!(
            tone_mapping.enum_variants,
            vec![
                SmolStr::new_static("agx"),
                SmolStr::new_static("aces"),
                SmolStr::new_static("pbr_neutral"),
            ]
        );

        // Verify extended PresetIr metadata.
        assert_eq!(ir.presets.len(), 4);
        assert_eq!(ir.presets[0].tiers, (0, 3));
        assert_eq!(ir.presets[3].name, "ultra");
        assert_eq!(ir.presets[3].tiers, (2, 3));

        // Verify deterministic topological scheduling.
        let plan = validate_and_schedule(&ir).expect("valid sunlit_vanilla schedule");
        assert_eq!(
            plan.replaced_canonical,
            vec![CanonicalPass::VolumetricFogPass]
        );
        assert!(
            !plan
                .ordered_passes
                .iter()
                .any(|p| p == CanonicalPass::VolumetricFogPass.as_str())
        );

        let pos = |name: &str| {
            plan.ordered_passes
                .iter()
                .position(|p| p == name)
                .unwrap_or_else(|| panic!("pass `{name}` missing from schedule"))
        };
        assert!(pos("gbuffer_mrt_pass") < pos("custom_volumetric_fog"));
        assert!(pos("custom_volumetric_fog") < pos("lighting_raster_compute_gi"));
        assert!(pos("gbuffer_mrt_pass") < pos("custom_refl_trace"));
        assert!(pos("custom_refl_trace") < pos("denoiser_svgf_nrd"));
    }

    #[test]
    fn emits_e0416_for_unknown_preset_setting_key() {
        let src = r#"
            pack "test-pack" { name: "Test" }
            settings {
                category "General" {
                    param exposure: f32 = 1.0 { label: "Exposure" }
                }
            }
            presets {
                preset broken (order = 0, label: "Broken") { nonexistent_key = 42; }
            }
            pipeline extend(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {}
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());

        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(
            diags.as_slice().iter().any(|d| d.code == "E0416"),
            "expected E0416 diagnostic, got: {:?}",
            diags.as_slice()
        );
    }

    #[test]
    fn emits_e0417_for_unknown_image_format() {
        let src = r#"
            pack "test-pack" { name: "Test" }
            pipeline extend(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                let _tex = g.resource_texture("noise.png", rgba999, linear, repeat);
                let _t1 = g.target::<bogus_fmt>("bad_target");
                let _t2 = g.target_array::<also_bogus>("bad_array", layers = 4);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());

        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        let e0417_count = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0417")
            .count();
        assert_eq!(
            e0417_count,
            3,
            "expected 3 E0417 diagnostics, got: {:?}",
            diags.as_slice()
        );
    }

    #[test]
    fn emits_e0418_e0419_and_e0420_diagnostics() {
        let src = r#"
            pack "diag-pack" {
                name: "Diag Pack",
                unknown_meta_field: "oops",
            }
            settings {
                category "General" {
                    param bad_type: f32 = true { label: "Bad Type" }
                    param out_of_range: i32 = 10 { label: "Out", range: 1..=4 }
                    param valid_i: i32 = 2 { label: "Valid", range: 1..=4 }
                }
            }
            presets {
                preset bad (order = 0, label: "Bad") {
                    valid_i = 9;
                }
            }
            pipeline extend(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                let _x = nonexistent_var + 1;
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        let codes: Vec<&str> = diags.as_slice().iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E0419"), "expected E0419, got {codes:?}");
        assert!(codes.contains(&"E0418"), "expected E0418, got {codes:?}");
        assert_eq!(
            codes.iter().filter(|&&c| c == "E0420").count(),
            3,
            "expected 3 E0420 diagnostics, got {codes:?}"
        );
    }

    #[test]
    fn evaluates_while_loop_break_continue_return_arith_hex_and_spec() {
        let src = r#"
            pack "ctrl-pack" { name: "Control Flow" }
            settings {
                category "PT" {
                    spec bounces: i32 = 0x4 { label: "Bounces", range: 1..=8 }
                }
            }
            pipeline custom(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                let mut i = 0;
                let mut created = 0;
                while i < cfg.spec(bounces) {
                    i += 1;
                    if i % 2 == 0 { continue; }
                    created += 1;
                }
                let mut j = 0;
                loop {
                    j = j + 1;
                    if j == 2 { break; }
                }
                if created == 2 && j == 2 {
                    let t = g.target::<rgba16f>("t0", layers = 0x2);
                    g.compute_pass("p1", pack::a::run).bind(out_t = t).dispatch_for(t);
                    return;
                }
                g.compute_pass("unreachable_pass", pack::b::run);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());
        assert_eq!(ir.passes.len(), 1);
        assert_eq!(ir.passes[0].name, "p1");
        assert_eq!(ir.targets["t0"].layers, 2);
    }

    #[test]
    fn evaluates_dynamic_target_args_int_to_float_widen_for_scope_bitwise_and_cfg_fields() {
        let src = r#"
            pack "dyn-pack" { name: "Dynamic Eval" }
            settings {
                category "General" {
                    param fog_scale: f32 = 1 { label: "Fog Scale", range: 0.25..=2.0 }
                    param fog_layers: i32 = 4 { label: "Fog Layers", range: 1..=32 }
                    param fog_temporal: bool = true { label: "Fog Temporal" }
                }
            }
            presets {
                preset med (order = 0, label: "Med") {
                    fog_scale = 2;
                }
            }
            pipeline custom(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                let outer = 99;
                let mut mask = 0;
                for outer in 0..4 {
                    mask |= 1 << outer;
                }
                let clear_alpha = if (mask & 0xF) == 15 && outer == 99 && cfg.fog_scale == 1 {
                    1.0
                } else {
                    0.0
                };
                let mips = (1 << 2) >> 1;
                let dyn_layers = cfg.param(fog_layers) * 2;
                let t = g.target_array::<rgba16f>(
                    "dyn_fog",
                    scale = cfg.param(fog_scale) * 0.5,
                    layers = dyn_layers,
                    mip_levels = mips,
                    temporal = cfg.fog_temporal,
                    clear_color = vec4(0.1, 0.2, 0.3, clear_alpha),
                );
                g.compute_pass("fog_pass", pack::fog::run)
                    .bind(out_fog = t)
                    .dispatch_for(t);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        assert_eq!(
            ir.settings["fog_scale"].default_value,
            SettingValueIr::Float(1.0)
        );
        assert_eq!(
            ir.presets[0].overrides["fog_scale"],
            SettingValueIr::Float(2.0)
        );
        let t = &ir.targets["dyn_fog"];
        assert!((t.scale - 0.5).abs() < 1e-6);
        assert_eq!(t.layers, 8);
        assert_eq!(t.mip_levels, 2);
        assert!(t.temporal);
        assert_eq!(t.clear_color, Some([0.1, 0.2, 0.3, 1.0]));
    }

    #[test]
    fn orders_custom_passes_and_infers_write_read_target_edges() {
        let mut ir = PipelineIr {
            mode: PipelineMode::Custom,
            ..PipelineIr::default()
        };
        ir.targets.insert(
            SmolStr::new_static("buf_a"),
            TargetDecl {
                name: SmolStr::new_static("buf_a"),
                format: ImageFormat::Rgba16f,
                scale: 1.0,
                layers: 1,
                mip_levels: 1,
                temporal: false,
                clear_color: None,
            },
        );
        // Declared in reverse execution order to prove both custom `.after(...)` and write->read
        // inference (including explicit `is_write: true` without `out_` prefix).
        let mut consumer_bindings = IndexMap::new();
        consumer_bindings.insert(
            SmolStr::new_static("in_a"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("buf_a"),
                history: false,
                is_write: false,
            },
        );
        let mut producer_bindings = IndexMap::new();
        producer_bindings.insert(
            SmolStr::new_static("radiance_dst"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("buf_a"),
                history: false,
                is_write: true,
            },
        );

        ir.passes.push(PassNode {
            name: SmolStr::new_static("final_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::final::run"),
            replaces: None,
            after: vec![PassRef::Custom(SmolStr::new_static("consumer_pass"))],
            before: Vec::new(),
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: None,
        });
        ir.passes.push(PassNode {
            name: SmolStr::new_static("consumer_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::consumer::run"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings: consumer_bindings,
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: None,
        });
        ir.passes.push(PassNode {
            name: SmolStr::new_static("producer_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::producer::run"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings: producer_bindings,
            meta: PassMeta::default(),
            dispatch_target: Some(SmolStr::new_static("buf_a")),
            dispatch: None,
        });

        let plan = validate_and_schedule(&ir).expect("valid custom schedule");
        assert_eq!(
            plan.ordered_passes,
            vec![
                SmolStr::new_static("producer_pass"),
                SmolStr::new_static("consumer_pass"),
                SmolStr::new_static("final_pass"),
            ]
        );
    }

    #[test]
    fn anchors_replacing_pass_before_canonical_successor_when_only_after_is_specified() {
        let mut ir = PipelineIr {
            mode: PipelineMode::Extend,
            ..PipelineIr::default()
        };
        ir.passes.push(PassNode {
            name: SmolStr::new_static("custom_fog_only_after"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::fog::run"),
            replaces: Some(CanonicalPass::VolumetricFogPass),
            after: vec![PassRef::Canonical(CanonicalPass::GbufferMrtPass)],
            before: Vec::new(),
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: None,
        });

        let plan = validate_and_schedule(&ir).expect("valid anchored replace schedule");
        let pos = |name: &str| {
            plan.ordered_passes
                .iter()
                .position(|p| p == name)
                .unwrap_or_else(|| panic!("pass `{name}` missing from schedule"))
        };
        assert!(pos("gbuffer_mrt_pass") < pos("custom_fog_only_after"));
        assert!(pos("custom_fog_only_after") < pos("lighting_raster_compute_gi"));
    }

    #[test]
    fn validates_binding_targets_temporal_history_and_empty_entry_fn() {
        let mut ir = PipelineIr {
            mode: PipelineMode::Custom,
            ..PipelineIr::default()
        };
        ir.targets.insert(
            SmolStr::new_static("non_temporal"),
            TargetDecl {
                name: SmolStr::new_static("non_temporal"),
                format: ImageFormat::Rgba16f,
                scale: 1.0,
                layers: 1,
                mip_levels: 1,
                temporal: false,
                clear_color: None,
            },
        );
        let mut bindings = IndexMap::new();
        bindings.insert(
            SmolStr::new_static("missing"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("no_such_target"),
                history: false,
                is_write: false,
            },
        );
        bindings.insert(
            SmolStr::new_static("bad_hist"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("non_temporal"),
                history: true,
                is_write: false,
            },
        );
        ir.passes.push(PassNode {
            name: SmolStr::new_static("broken_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static(""),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings,
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: None,
        });

        let errs = validate_and_schedule(&ir).expect_err("expected validation errors");
        assert_eq!(errs.len(), 3, "expected 3 errors, got: {errs:?}");
    }

    #[test]
    fn detects_schedule_cycle_cleanly() {
        let mut ir = PipelineIr {
            mode: PipelineMode::Extend,
            ..PipelineIr::default()
        };
        ir.passes.push(PassNode {
            name: SmolStr::new_static("cyclic_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::cyclic::run"),
            replaces: None,
            after: vec![PassRef::Canonical(CanonicalPass::LightingRasterComputeGi)],
            before: vec![PassRef::Canonical(CanonicalPass::GbufferMrtPass)],
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: None,
        });

        let err = validate_and_schedule(&ir).expect_err("expected cycle error");
        assert!(
            err.iter().any(|msg| msg.contains("cycle")),
            "expected cycle diagnostic, got: {err:?}"
        );
    }

    #[test]
    fn rejects_duplicate_pass_duplicate_replace_and_missing_dispatch_target() {
        let mut ir = PipelineIr {
            mode: PipelineMode::Extend,
            ..PipelineIr::default()
        };
        ir.passes.push(PassNode {
            name: SmolStr::new_static("dup_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::a::run"),
            replaces: Some(CanonicalPass::VolumetricFogPass),
            after: Vec::new(),
            before: Vec::new(),
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: Some(SmolStr::new_static("missing_target")),
            dispatch: None,
        });
        ir.passes.push(PassNode {
            name: SmolStr::new_static("dup_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::b::run"),
            replaces: Some(CanonicalPass::VolumetricFogPass),
            after: Vec::new(),
            before: Vec::new(),
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: None,
        });

        let errs = validate_and_schedule(&ir).expect_err("expected validation errors");
        assert_eq!(errs.len(), 3, "expected 3 validation errors, got: {errs:?}");
    }

    #[test]
    fn desugars_full_pack_toml_and_validates_presets_and_ranges() {
        let toml_src = r#"
[pack]
id = "legacy-test"
name = "Legacy Test"
version = "2.0.0"

[pipeline]
mode = "extend"

[settings.exposure]
category = "Post"
label = "Exposure"
default = 1.5
range = [0.2, 4.0]
step = 0.1

[presets.high]
label = "High"
order = 1
[presets.high.overrides]
exposure = 2.0

[[resources]]
path = "resources/noise.png"
format = "rgba8"
filter = "nearest"
wrap = "repeat"

[targets.fog_vol]
format = "rgba16f"
scale = 0.25
temporal = true

[[passes]]
name = "custom_fog"
kind = "compute"
shader = "pack::fog::run"
after = ["gbuffer_mrt_pass"]
dispatch = "fog_vol"
[passes.bindings]
prev_fog = "fog_vol.history"
out_fog = "fog_vol"
depth = "RES_DEPTH"
noise = "resource:0"
"#;
        let ir = desugar_pack_toml(toml_src).expect("valid pack.toml");
        assert_eq!(ir.settings.len(), 1);
        assert_eq!(ir.presets.len(), 1);
        assert_eq!(ir.resource_textures.len(), 1);
        assert_eq!(ir.passes.len(), 1);
        assert_eq!(ir.passes[0].dispatch_target.as_deref(), Some("fog_vol"));
        assert_eq!(
            ir.passes[0].bindings.get("depth"),
            Some(&PassBindingRef::BuiltinTarget {
                resource: BuiltinResourceId::ResMcMainDepth
            })
        );
        assert_eq!(
            ir.passes[0].bindings.get("noise"),
            Some(&PassBindingRef::ResourceTexture { slot: 0 })
        );
        assert_eq!(
            ir.passes[0].bindings.get("prev_fog"),
            Some(&PassBindingRef::CustomTarget {
                name: SmolStr::new_static("fog_vol"),
                history: true,
                is_write: false,
            })
        );
        assert_eq!(
            ir.passes[0].bindings.get("out_fog"),
            Some(&PassBindingRef::CustomTarget {
                name: SmolStr::new_static("fog_vol"),
                history: false,
                is_write: true,
            })
        );
        assert_eq!(ir.targets["fog_vol"].format, ImageFormat::Rgba16f);

        let bad_toml = r#"
[pack]
id = "bad"
name = "Bad"
[targets.bad_t]
format = "rgba999"
"#;
        let err = desugar_pack_toml(bad_toml).expect_err("expected UnknownImageFormat");
        assert!(matches!(err, TomlDesugarError::UnknownImageFormat(ref s) if s == "rgba999"));

        let unknown_preset_key = r#"
[pack]
id = "bad-preset"
name = "Bad Preset"
[settings.exposure]
default = 1.0
range = [0.2, 4.0]
[presets.broken.overrides]
nonexistent = 2.0
"#;
        let err = desugar_pack_toml(unknown_preset_key).expect_err("expected UnknownPresetSetting");
        assert!(matches!(
            err,
            TomlDesugarError::UnknownPresetSetting { ref key, .. } if key == "nonexistent"
        ));

        let out_of_range_preset = r#"
[pack]
id = "bad-range"
name = "Bad Range"
[settings.exposure]
default = 1.0
range = [0.2, 4.0]
[presets.broken.overrides]
exposure = 9.5
"#;
        let err =
            desugar_pack_toml(out_of_range_preset).expect_err("expected PresetValueOutOfRange");
        assert!(matches!(
            err,
            TomlDesugarError::PresetValueOutOfRange { ref key, .. } if key == "exposure"
        ));
    }

    #[test]
    fn emits_e0416_for_unknown_setting_in_cfg_param_and_cfg_field() {
        let src = r#"
            pack "typo-pack" { name: "Typo Pack" }
            settings {
                category "General" {
                    param exposure: f32 = 1.0 { label: "Exposure" }
                }
            }
            pipeline extend(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                let _a = cfg.param(exposur_typo);
                let _b = cfg.another_typo;
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        let e0416: Vec<_> = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0416")
            .collect();
        assert_eq!(
            e0416.len(),
            2,
            "expected 2 E0416 diagnostics, got: {:?}",
            diags.as_slice()
        );
        assert!(e0416[0].message.contains("exposur_typo"));
        assert!(e0416[1].message.contains("another_typo"));
    }

    #[test]
    fn supports_explicit_write_binding_syntax_without_out_prefix() {
        let src = r#"
            pack "write-bind-pack" { name: "Write Bind Pack" }
            pipeline custom(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                let buf_a = g.target::<rgba16f>("buf_a");
                let buf_b = g.target::<rgba16f>("buf_b");

                g.compute_pass("producer_fn", pack::a::run)
                    .bind(my_attachment = write(buf_a))
                    .dispatch_for(buf_a);

                g.compute_pass("producer_method", pack::b::run)
                    .bind(in_a = buf_a, color_slot = buf_b.write())
                    .dispatch_for(buf_b);

                g.compute_pass("consumer", pack::c::run)
                    .bind(src_b = buf_b)
                    .dispatch_for(buf_b);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());

        assert_eq!(
            ir.passes[0].bindings.get("my_attachment"),
            Some(&PassBindingRef::CustomTarget {
                name: SmolStr::new_static("buf_a"),
                history: false,
                is_write: true,
            })
        );
        assert_eq!(
            ir.passes[1].bindings.get("color_slot"),
            Some(&PassBindingRef::CustomTarget {
                name: SmolStr::new_static("buf_b"),
                history: false,
                is_write: true,
            })
        );
        assert_eq!(
            ir.passes[2].bindings.get("src_b"),
            Some(&PassBindingRef::CustomTarget {
                name: SmolStr::new_static("buf_b"),
                history: false,
                is_write: false,
            })
        );

        let plan = validate_and_schedule(&ir).expect("valid write-read schedule");
        assert_eq!(
            plan.ordered_passes,
            vec![
                SmolStr::new_static("producer_fn"),
                SmolStr::new_static("producer_method"),
                SmolStr::new_static("consumer"),
            ]
        );
    }

    #[test]
    fn widens_int_default_to_float_in_pack_toml_when_range_or_step_is_float() {
        let toml_src = r#"
[pack]
id = "widen-test"
name = "Widen Test"

[settings.exposure]
default = 1
range = [0.2, 4.0]

[settings.gamma]
default = 2
step = 0.25

[presets.bright.overrides]
exposure = 1.5
gamma = 2.25
"#;
        let ir = desugar_pack_toml(toml_src).expect("valid widened pack.toml");
        assert_eq!(
            ir.settings["exposure"].default_value,
            SettingValueIr::Float(1.0)
        );
        assert_eq!(
            ir.settings["gamma"].default_value,
            SettingValueIr::Float(2.0)
        );
        assert_eq!(
            ir.presets[0].overrides["exposure"],
            SettingValueIr::Float(1.5)
        );
        assert_eq!(
            ir.presets[0].overrides["gamma"],
            SettingValueIr::Float(2.25)
        );
    }

    #[test]
    fn lexical_let_scopes_do_not_leak_or_clobber_outer_bindings() {
        let src = r#"
            pack "scope-test" { name: "Scope Test", version: "1.0.0" }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {
                let x = g.target::<rgba16f>("outer");
                if true {
                    let x = g.target::<rgba16f>("inner");
                    let _inner = x;
                }
                g.compute_pass("use_outer", main_cs)
                    .after(mc::passes::gbuffer_mrt_pass)
                    .bind(out = x.write());
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());
        assert!(ir.targets.contains_key("inner"));
        assert!(ir.targets.contains_key("outer"));
        let pass = ir
            .passes
            .iter()
            .find(|p| p.name == "use_outer")
            .expect("use_outer pass");
        assert!(matches!(
            pass.bindings.get("out"),
            Some(PassBindingRef::CustomTarget { name, .. }) if name == "outer"
        ));
    }

    #[test]
    fn duplicate_target_declaration_emits_diagnostic() {
        let src = r#"
            pack "dup-target" { name: "Dup Target", version: "1.0.0" }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {
                let a = g.target::<rgba16f>("shared");
                let b = g.target::<r16f>("shared");
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(
            diags.as_slice().iter().any(|d| d.code == "E0421"),
            "expected E0421 duplicate target error, got: {:?}",
            diags.as_slice()
        );
        assert_eq!(ir.targets["shared"].format, vsl_mc::ImageFormat::Rgba16f);
    }

    #[test]
    fn write_after_read_with_explicit_after_does_not_create_false_cycle() {
        let src = r#"
            pack "war-test" { name: "WAR Test", version: "1.0.0" }
            pipeline custom(g: mut mc::GraphBuilder, cfg: PackConfig) {
                let t = g.target::<rgba16f>("t");
                let a = g.compute_pass("pass_a", a_main).bind(out = t.write());
                let b = g.compute_pass("pass_b", b_main).after(a).bind(inp = t);
                let _c = g.compute_pass("pass_c", c_main).after(b).bind(out = t.write());
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());
        let plan = validate_and_schedule(&ir).expect("acyclic write-after-read schedule");
        let names: Vec<&str> = plan.ordered_passes.iter().map(SmolStr::as_str).collect();
        assert_eq!(names, vec!["pass_a", "pass_b", "pass_c"]);
    }

    #[test]
    fn desugars_hyaloclast_v1_manifest_without_semantic_loss() {
        let src = r#"
            [pack]
            id = "v1-test"
            name = "V1 Test"
            version = "1.0.0"

            [capabilities]
            tiers = [0, 3]
            pbr_labpbr = true

            [pipeline]
            mode = "extend"

            [settings.foo]
            type = "float"
            default = 1.0
            min = 0.0
            max = 2.0
            runtime = true
            min_tier = 1
            max_tier = 3

            [settings.enum_setting]
            type = "enum"
            default = "a"
            options = ["a", "b"]
            runtime = true

            [presets.low]
            label = "Low"
            order = 0

            [presets.low.values]
            foo = 0.5
            enum_setting = "b"

            [resources.blue_noise]
            path = "blue.png"
            format = "RGBA8_UNORM"
            filter = "nearest"
            wrap = "repeat"

            [targets.t]
            format = "rgba16f"
            scale = 0.5

            [[passes]]
            id = "p"
            kind = "compute"
            shader = "p_main"
            replaces = "volumetric_fog_pass"
            reads = ["mc_main_depth"]
            writes = ["t"]
            scale = 0.5
            min_tier = 0
            max_tier = 2
        "#;
        let ir = desugar_pack_toml(src).expect("v1 manifest must deserialize");
        assert_eq!(ir.metadata.tier_min, 0);
        assert_eq!(ir.metadata.tier_max, 3);
        assert!(ir.metadata.capabilities.iter().any(|c| c == "pbr_labpbr"));

        let foo = &ir.settings["foo"];
        assert_eq!(foo.range, Some((0.0, 2.0)));
        assert_eq!(foo.tiers, (1, 3));
        assert!(!foo.is_spec);
        assert_eq!(ir.settings["enum_setting"].enum_variants, vec!["a", "b"]);

        let low = &ir.presets[0];
        assert_eq!(low.overrides["foo"], SettingValueIr::Float(0.5));
        assert_eq!(
            low.overrides["enum_setting"],
            SettingValueIr::EnumVariant("b".into())
        );

        let resource = &ir.resource_textures[0];
        assert_eq!(resource.name, "blue_noise");
        assert_eq!(resource.format, ImageFormat::Rgba8);

        let pass = &ir.passes[0];
        assert_eq!(pass.name, "p");
        assert_eq!(pass.meta.reads, vec!["mc_main_depth"]);
        assert_eq!(pass.meta.writes, vec!["t"]);
        assert_eq!(pass.meta.scale, Some(0.5));
        assert_eq!(pass.meta.tiers, Some((0, 2)));
    }

    #[test]
    fn reports_unknown_graph_builder_methods() {
        let src = r#"
            pack "unknown-method" { name: "U", version: "1.0.0" }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {
                g.comput_pass("oops", main_cs);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(diags.as_slice().iter().any(|d| d.code == "E0426"));
    }

    #[test]
    fn cfg_param_works_for_any_pipeline_config_binding() {
        let src = r#"
            pack "cfg-param" { name: "C", version: "1.0.0" }
            settings { category "General" { param exposure: f32 = 1.0; } }
            pipeline extend(g: mut mc::GraphBuilder, config: mc::PipelineConfig) {
                let _v = config.param(exposure);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());
    }

    #[test]
    fn rejects_non_range_for_iterators() {
        let src = r#"
            pack "bad-for" { name: "B", version: "1.0.0" }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {
                for i in 1 { let _x = i; }
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(diags.as_slice().iter().any(|d| d.code == "E0492"));
    }

    #[test]
    fn rejects_resource_texture_without_path() {
        let src = r#"
            pack "bad-resource" { name: "B", version: "1.0.0" }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {
                let _r = g.resource_texture();
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(diags.as_slice().iter().any(|d| d.code == "E0427"));
    }

    #[test]
    fn enforces_pack_and_setting_tier_ranges() {
        let pack_src = r#"
            pack "tier-gated" { name: "T", version: "1.0.0", tiers: 2..=3 }
            settings { category "General" { param late: f32 = 1.0 { tiers: 3..=3, } } }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {}
        "#;
        let parsed = parse_source(FileId(0), pack_src);
        assert!(!parsed.has_errors());
        let cfg = EvalConfig {
            tier: RenderTier::Tier0Raster,
            ..EvalConfig::default()
        };
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &cfg);
        assert!(diags.as_slice().iter().any(|d| d.code == "E0423"));
    }

    #[test]
    fn rejects_reversed_tier_ranges() {
        let src = r#"
            pack "bad-tiers" { name: "B", version: "1.0.0", tiers: 3..=1 }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {}
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(diags.as_slice().iter().any(|d| d.code == "E0422"));
    }

    #[test]
    fn rejects_empty_exclusive_tier_range() {
        let src = r#"
            pack "empty-tiers" { name: "E", version: "1.0.0", tiers: 0..0 }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {}
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(diags.as_slice().iter().any(|d| d.code == "E0422"));
    }

    #[test]
    fn rejects_tier_gated_setting_access() {
        let src = r#"
            pack "tier-setting" { name: "T", version: "1.0.0", tiers: 0..=3 }
            settings { category "General" { param late: f32 = 1.0 { tiers: 3..=3, } } }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {
                let _v = cfg.param(late);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        // Default tier is 2, allowed by the pack but excluded by the setting.
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(diags.as_slice().iter().any(|d| d.code == "E0424"));
    }

    #[test]
    fn reports_unknown_methods_on_all_pack_handles() {
        let src = r#"
            pack "unknown-handle-method" { name: "U", version: "1.0.0" }
            pipeline extend(g: mut mc::GraphBuilder, cfg: PackConfig) {
                let t = g.target::<rgba16f>("t");
                t.writ();
                g.param(t);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        let count = diags
            .as_slice()
            .iter()
            .filter(|d| d.code == "E0426")
            .count();
        assert_eq!(
            count,
            2,
            "expected E0426 for target and config handles: {:?}",
            diags.as_slice()
        );
    }

    #[test]
    fn defaults_one_sided_v1_bounds() {
        let src = r#"
            [pack]
            id = "one-sided"
            name = "One Sided"

            [settings.gated]
            type = "float"
            default = 1.5
            min = 1.0

            [[passes]]
            id = "p"
            shader = "p_main"
            min_tier = 1
        "#;
        let ir = desugar_pack_toml(src).expect("v1 manifest must deserialize");
        assert_eq!(ir.settings["gated"].range, Some((1.0, 3.0)));
        assert_eq!(ir.passes[0].meta.tiers, Some((1, 3)));
    }

    #[test]
    fn evaluates_and_schedules_transfer_and_mip_passes_with_dispatch_modes() {
        let src = r#"
            pack "transfer-dispatch" { name: "Transfer & Dispatch", version: "1.0.0" }
            pipeline custom(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                let src_tex = g.target::<rgba16f>("src_tex", mip_levels = 4);
                let dst_tex = g.target::<rgba16f>("dst_tex");
                let indirect_buf = g.target::<r32ui>("indirect_buf");

                let p_explicit = g.compute_pass("fill_src", pack::fill::run)
                    .bind(out_src = src_tex.write())
                    .dispatch(16, 16, 1);

                let p_mip = g.mip_pass("gen_mips")
                    .after(p_explicit)
                    .bind(in_src = src_tex);

                let p_transfer = g.transfer_pass("copy_to_dst")
                    .after(p_mip)
                    .bind(in_src = src_tex, out_dst = dst_tex.write())
                    .dispatch_for(dst_tex);

                let _p_custom_transfer = g.transfer_pass("custom_blit", pack::blit::run)
                    .after(p_transfer)
                    .bind(in_dst = dst_tex)
                    .dispatch_indirect(indirect_buf);

                let _p_indirect_str = g.compute_pass("indirect_str", pack::ind::run)
                    .after(p_transfer)
                    .dispatch_indirect("args_ssbo");
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (ir, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        assert!(!diags.has_errors(), "{:?}", diags.as_slice());
        assert_eq!(ir.passes.len(), 5);

        assert_eq!(ir.passes[0].kind, PassKind::Compute);
        assert_eq!(ir.passes[0].dispatch_target, None);
        assert_eq!(ir.passes[0].dispatch, Some(PassDispatch::Explicit([16, 16, 1])));

        assert_eq!(ir.passes[1].kind, PassKind::Transfer);
        assert!(ir.passes[1].entry_fn.is_empty());

        assert_eq!(ir.passes[2].kind, PassKind::Transfer);
        assert!(ir.passes[2].entry_fn.is_empty());
        assert_eq!(ir.passes[2].dispatch_target.as_deref(), Some("dst_tex"));
        assert_eq!(
            ir.passes[2].dispatch,
            Some(PassDispatch::ForTarget(SmolStr::new_static("dst_tex")))
        );

        assert_eq!(ir.passes[3].kind, PassKind::Transfer);
        assert_eq!(ir.passes[3].entry_fn, "pack::blit::run");
        assert_eq!(
            ir.passes[3].dispatch,
            Some(PassDispatch::Indirect(SmolStr::new_static("indirect_buf")))
        );

        assert_eq!(
            ir.passes[4].dispatch,
            Some(PassDispatch::Indirect(SmolStr::new_static("args_ssbo")))
        );

        let plan =
            validate_and_schedule(&ir).expect("transfer passes with empty entry_fn should schedule");
        let pos = |name: &str| {
            plan.ordered_passes
                .iter()
                .position(|p| p == name)
                .unwrap_or_else(|| panic!("pass `{name}` missing from schedule"))
        };
        assert!(pos("fill_src") < pos("gen_mips"));
        assert!(pos("gen_mips") < pos("copy_to_dst"));
        assert!(pos("copy_to_dst") < pos("custom_blit"));
    }

    #[test]
    fn rejects_invalid_dispatch_arguments_in_eval_and_schedule() {
        let src = r#"
            pack "bad-dispatch" { name: "Bad Dispatch", version: "1.0.0" }
            pipeline custom(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                g.compute_pass("p1", pack::a::run).dispatch(0, 8, 1);
                g.compute_pass("p2", pack::b::run).dispatch(8, 8);
                g.compute_pass("p3", pack::c::run).dispatch_indirect("");
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let (_, diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
        let codes: Vec<&str> = diags.as_slice().iter().map(|d| d.code.as_str()).collect();
        assert_eq!(
            codes.iter().filter(|&&c| c == "E0428").count(),
            2,
            "expected 2 E0428 diagnostics, got: {codes:?}"
        );
        assert!(
            codes.contains(&"E0415"),
            "expected E0415 for empty indirect dispatch, got: {codes:?}"
        );

        let mut ir = PipelineIr {
            mode: PipelineMode::Custom,
            ..PipelineIr::default()
        };
        ir.passes.push(PassNode {
            name: SmolStr::new_static("zero_xyz"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::a::run"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: Some(PassDispatch::Explicit([4, 0, 1])),
        });
        ir.passes.push(PassNode {
            name: SmolStr::new_static("empty_indirect"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::b::run"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: Some(PassDispatch::Indirect(SmolStr::new_static("   "))),
        });
        ir.passes.push(PassNode {
            name: SmolStr::new_static("missing_for_target"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::c::run"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings: IndexMap::new(),
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: Some(PassDispatch::ForTarget(SmolStr::new_static("no_target"))),
        });

        let errs = validate_and_schedule(&ir).expect_err("expected dispatch validation errors");
        assert_eq!(errs.len(), 3, "expected 3 errors, got: {errs:?}");
    }

    #[test]
    fn desugars_pack_toml_dispatch_variants_and_transfer_pass() {
        let toml_src = r#"
[pack]
id = "dispatch-toml"
name = "Dispatch TOML"

[pipeline]
mode = "custom"

[targets.vol]
format = "rgba16f"

[[passes]]
name = "p_for_target"
kind = "compute"
shader = "pack::a::run"
dispatch = "vol"

[[passes]]
name = "p_explicit"
kind = "compute"
shader = "pack::b::run"
dispatch_xyz = [8, 4, 2]

[[passes]]
name = "p_indirect"
kind = "compute"
shader = "pack::c::run"
dispatch_indirect = "indirect_args"

[[passes]]
name = "p_transfer"
kind = "transfer"
"#;
        let ir = desugar_pack_toml(toml_src).expect("valid pack.toml with dispatch variants");
        assert_eq!(ir.passes.len(), 4);
        assert_eq!(ir.passes[0].dispatch_target.as_deref(), Some("vol"));
        assert_eq!(
            ir.passes[0].dispatch,
            Some(PassDispatch::ForTarget(SmolStr::new_static("vol")))
        );
        assert_eq!(
            ir.passes[1].dispatch,
            Some(PassDispatch::Explicit([8, 4, 2]))
        );
        assert_eq!(
            ir.passes[2].dispatch,
            Some(PassDispatch::Indirect(SmolStr::new_static("indirect_args")))
        );
        assert_eq!(ir.passes[3].kind, PassKind::Transfer);
        assert!(ir.passes[3].entry_fn.is_empty());

        let plan = validate_and_schedule(&ir).expect("valid schedule");
        assert_eq!(plan.ordered_passes.len(), 4);
    }

    fn make_test_shader_unit(
        entry_point: &str,
        descriptors: Vec<vsl_mc::DescriptorBinding>,
    ) -> vsl_mc::CompiledShaderUnit {
        vsl_mc::CompiledShaderUnit {
            entry_point: SmolStr::new(entry_point),
            stage: vsl_mc::ShaderStage::Compute,
            spirv: Vec::new(),
            descriptors,
            push_constants: Vec::new(),
            push_constant_size: 0,
            workgroup_size: Some([8, 8, 1]),
            required_caps: vsl_mc::HardwareCaps::empty(),
            used_specs: Vec::new(),
            used_params: Vec::new(),
            debug_channels: Vec::new(),
            cache_key: [0u8; 32],
        }
    }

    fn make_desc(
        name: &str,
        set: u32,
        binding: u32,
        kind: vsl_mc::DescriptorKind,
        dim: Option<vsl_mc::TextureDim>,
        image_format: Option<ImageFormat>,
        access: vsl_mc::DescriptorAccess,
    ) -> vsl_mc::DescriptorBinding {
        vsl_mc::DescriptorBinding {
            name: SmolStr::new(name),
            set,
            binding,
            kind,
            dim,
            sample_scalar: None,
            image_format,
            access,
            is_unsized: false,
            count: None,
        }
    }

    #[test]
    fn validates_matching_pass_bindings_against_compiled_shader_unit() {
        use vsl_mc::{
            DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS, DESCRIPTOR_SET_SCENE_GLOBALS,
            DescriptorAccess, DescriptorKind, TextureDim,
        };

        let mut ir = PipelineIr {
            mode: PipelineMode::Custom,
            ..PipelineIr::default()
        };
        ir.targets.insert(
            SmolStr::new_static("color_in"),
            TargetDecl {
                name: SmolStr::new_static("color_in"),
                format: ImageFormat::Rgba16f,
                scale: 1.0,
                layers: 1,
                mip_levels: 1,
                temporal: false,
                clear_color: None,
            },
        );
        ir.targets.insert(
            SmolStr::new_static("fog_arr"),
            TargetDecl {
                name: SmolStr::new_static("fog_arr"),
                format: ImageFormat::Rgba16f,
                scale: 0.5,
                layers: 8,
                mip_levels: 1,
                temporal: false,
                clear_color: None,
            },
        );

        let mut bindings = IndexMap::new();
        bindings.insert(
            SmolStr::new_static("in_color"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("color_in"),
                history: false,
                is_write: false,
            },
        );
        bindings.insert(
            SmolStr::new_static("out_fog"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("fog_arr"),
                history: false,
                is_write: true,
            },
        );
        bindings.insert(
            SmolStr::new_static("depth"),
            PassBindingRef::BuiltinTarget {
                resource: BuiltinResourceId::ResMcMainDepth,
            },
        );
        bindings.insert(
            SmolStr::new_static("tlas"),
            PassBindingRef::BuiltinTarget {
                resource: BuiltinResourceId::ResTlas,
            },
        );
        bindings.insert(
            SmolStr::new_static("lights"),
            PassBindingRef::BuiltinTarget {
                resource: BuiltinResourceId::ResEmissiveLights,
            },
        );
        bindings.insert(
            SmolStr::new_static("blue_noise"),
            PassBindingRef::ResourceTexture { slot: 2 },
        );

        let pass = PassNode {
            name: SmolStr::new_static("fog_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::fog::run"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings,
            meta: PassMeta::default(),
            dispatch_target: Some(SmolStr::new_static("fog_arr")),
            dispatch: Some(PassDispatch::ForTarget(SmolStr::new_static("fog_arr"))),
        };

        let unit = make_test_shader_unit(
            "pack::fog::run",
            vec![
                // Engine-managed Set 0 binding 0 and Set 1 scene global must be ignored.
                make_desc(
                    "world_ubo",
                    DESCRIPTOR_SET_PER_PASS,
                    0,
                    DescriptorKind::UniformBuffer,
                    None,
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "scene_tlas",
                    DESCRIPTOR_SET_SCENE_GLOBALS,
                    0,
                    DescriptorKind::AccelerationStructure,
                    None,
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "in_color",
                    DESCRIPTOR_SET_PER_PASS,
                    1,
                    DescriptorKind::SampledImage,
                    Some(TextureDim::Dim2D),
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "out_fog",
                    DESCRIPTOR_SET_PER_PASS,
                    2,
                    DescriptorKind::StorageImage,
                    Some(TextureDim::Dim2DArray),
                    Some(ImageFormat::Rgba16f),
                    DescriptorAccess::WriteOnly,
                ),
                make_desc(
                    "depth",
                    DESCRIPTOR_SET_PER_PASS,
                    3,
                    DescriptorKind::SampledImage,
                    Some(TextureDim::Dim2D),
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "tlas",
                    DESCRIPTOR_SET_PER_PASS,
                    4,
                    DescriptorKind::AccelerationStructure,
                    None,
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "lights",
                    DESCRIPTOR_SET_PER_PASS,
                    5,
                    DescriptorKind::StorageBuffer,
                    None,
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "blue_noise",
                    DESCRIPTOR_SET_PACK_RESOURCES,
                    2,
                    DescriptorKind::CombinedImageSampler,
                    Some(TextureDim::Dim2D),
                    None,
                    DescriptorAccess::ReadOnly,
                ),
            ],
        );

        let errs = validate_pass_bindings_against_shader(&pass, &unit, &ir);
        assert!(errs.is_empty(), "expected no errors, got: {errs:?}");
    }

    #[test]
    fn reports_format_access_dim_kind_missing_and_unknown_shader_binding_errors() {
        use vsl_mc::{
            DESCRIPTOR_SET_PACK_RESOURCES, DESCRIPTOR_SET_PER_PASS, DescriptorAccess,
            DescriptorKind, TextureDim,
        };

        let mut ir = PipelineIr {
            mode: PipelineMode::Custom,
            ..PipelineIr::default()
        };
        ir.targets.insert(
            SmolStr::new_static("t2d"),
            TargetDecl {
                name: SmolStr::new_static("t2d"),
                format: ImageFormat::Rgba8,
                scale: 1.0,
                layers: 1,
                mip_levels: 1,
                temporal: false,
                clear_color: None,
            },
        );
        ir.targets.insert(
            SmolStr::new_static("tarr"),
            TargetDecl {
                name: SmolStr::new_static("tarr"),
                format: ImageFormat::Rgba16f,
                scale: 1.0,
                layers: 4,
                mip_levels: 1,
                temporal: false,
                clear_color: None,
            },
        );

        let mut bindings = IndexMap::new();
        // Format mismatch (Rgba8 vs Rgba16f) + access mismatch (is_write: false vs WriteOnly)
        // + dim mismatch (single-layer vs Dim2DArray).
        bindings.insert(
            SmolStr::new_static("bad_storage"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("t2d"),
                history: false,
                is_write: false,
            },
        );
        // Access mismatch (is_write: true vs ReadOnly) + dim mismatch (4 layers vs Dim2D).
        bindings.insert(
            SmolStr::new_static("bad_read"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("tarr"),
                history: false,
                is_write: true,
            },
        );
        // DescriptorKind mismatch (CustomTarget -> StorageBuffer).
        bindings.insert(
            SmolStr::new_static("bad_kind"),
            PassBindingRef::CustomTarget {
                name: SmolStr::new_static("t2d"),
                history: false,
                is_write: false,
            },
        );
        // ResourceTexture wrong slot (slot 1 vs Set 3 binding 3).
        bindings.insert(
            SmolStr::new_static("bad_res"),
            PassBindingRef::ResourceTexture { slot: 1 },
        );
        // Unknown binding not in shader descriptors.
        bindings.insert(
            SmolStr::new_static("extra_binding"),
            PassBindingRef::BuiltinTarget {
                resource: BuiltinResourceId::ResGbufferAlbedo,
            },
        );

        let pass = PassNode {
            name: SmolStr::new_static("broken_shader_pass"),
            kind: PassKind::Compute,
            entry_fn: SmolStr::new_static("pack::broken::run"),
            replaces: None,
            after: Vec::new(),
            before: Vec::new(),
            bindings,
            meta: PassMeta::default(),
            dispatch_target: None,
            dispatch: None,
        };

        let unit = make_test_shader_unit(
            "pack::broken::run",
            vec![
                make_desc(
                    "bad_storage",
                    DESCRIPTOR_SET_PER_PASS,
                    1,
                    DescriptorKind::StorageImage,
                    Some(TextureDim::Dim2DArray),
                    Some(ImageFormat::Rgba16f),
                    DescriptorAccess::WriteOnly,
                ),
                make_desc(
                    "bad_read",
                    DESCRIPTOR_SET_PER_PASS,
                    2,
                    DescriptorKind::SampledImage,
                    Some(TextureDim::Dim2D),
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "bad_kind",
                    DESCRIPTOR_SET_PER_PASS,
                    3,
                    DescriptorKind::StorageBuffer,
                    None,
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                // Missing binding in `pass.bindings`.
                make_desc(
                    "required_missing",
                    DESCRIPTOR_SET_PER_PASS,
                    4,
                    DescriptorKind::SampledImage,
                    Some(TextureDim::Dim2D),
                    None,
                    DescriptorAccess::ReadOnly,
                ),
                make_desc(
                    "bad_res",
                    DESCRIPTOR_SET_PACK_RESOURCES,
                    3,
                    DescriptorKind::SampledImage,
                    Some(TextureDim::Dim2D),
                    None,
                    DescriptorAccess::ReadOnly,
                ),
            ],
        );

        let errs = validate_pass_bindings_against_shader(&pass, &unit, &ir);
        assert!(
            errs.iter().any(|e| e.contains("required_missing")),
            "expected missing binding error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("extra_binding")),
            "expected unknown binding error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("rgba8") && e.contains("rgba16f")),
            "expected ImageFormat mismatch error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("WriteOnly")),
            "expected WriteOnly access error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("ReadOnly")),
            "expected ReadOnly access error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("single-layer")),
            "expected single-layer TextureDim error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("multi-layer")),
            "expected multi-layer TextureDim error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("StorageBuffer")),
            "expected DescriptorKind error, got: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("slot 1") && e.contains("binding 3")),
            "expected ResourceTexture slot mismatch error, got: {errs:?}"
        );
    }
}
