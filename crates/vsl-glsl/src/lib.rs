//! Stage A GLSL 460 oracle backend and Hyaloclast raster G-Buffer hook bridge for VSL 2.0.

pub mod emit;
pub mod hook_bridge;

pub use emit::emit_glsl_460;
pub use hook_bridge::{emit_raster_gbuffer_hook_slot, emit_raster_gbuffer_hook_slot_with_types};

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use smol_str::SmolStr;
    use vsl_diagnostics::FileId;
    use vsl_mc::HookStage;
    use vsl_syntax::ast::Item;
    use vsl_syntax::parse_source;

    const TILED_LIGHTING_VSL: &str =
        include_str!("../../../examples/sunlit_vanilla/tiled_lighting.vsl");
    const GI_CACHE_VSL: &str = include_str!("../../../examples/sunlit_vanilla/gi_cache.vsl");
    const REFLECTIONS_VSL: &str = include_str!("../../../examples/sunlit_vanilla/reflections.vsl");
    const RESTIR_VSL: &str = include_str!("../../../examples/sunlit_vanilla/restir.vsl");

    #[test]
    fn emits_compute_stage_push_constant_and_shared_workgroup_vars() {
        let parsed = parse_source(FileId(0), TILED_LIGHTING_VSL);
        assert!(
            !parsed.has_errors(),
            "unexpected parse errors: {:?}",
            parsed.diagnostics
        );

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(glsl.contains("#version 460 core"));
        assert!(glsl.contains("#extension GL_EXT_scalar_block_layout : require"));
        assert!(glsl.contains("layout(push_constant, scalar) uniform AtrousPush"));
        assert!(glsl.contains("shared uint s_tile_lights[64];"));
        assert!(glsl.contains("shared uint s_tile_count;"));
        assert!(glsl.contains("layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;"));
        assert!(glsl.contains("void main()"));
        assert!(glsl.contains("uint lid = gl_LocalInvocationIndex;"));
        assert!(glsl.contains("uvec3 gid = gl_GlobalInvocationID;"));
        assert!(glsl.contains("barrier();"));
        assert!(glsl.contains("imageStore(out_lit,"));
        // Scoped local type inference for `min(s_tile_count, 64u)` and separated `texture2D`.
        assert!(glsl.contains("uint total = min(s_tile_count, 64u);"));
        assert!(glsl.contains("layout(set = 0, binding = 1) uniform texture2D gbuffer_depth;"));
        assert!(glsl.contains("layout(set = 0, binding = 2, rgba16f) uniform image2D out_lit;"));
        assert!(!glsl.contains("vsl_default_sampler"));
        assert!(!glsl.contains("binding = 15"));
        assert!(
            glsl.contains("layout(set = 1, binding = 3, scalar) readonly buffer EmissiveLightSSBO")
        );
    }

    #[test]
    fn emits_bda_buffer_reference2_arithmetic_and_value_dereferences() {
        let parsed = parse_source(FileId(0), GI_CACHE_VSL);
        assert!(
            !parsed.has_errors(),
            "unexpected parse errors: {:?}",
            parsed.diagnostics
        );

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(glsl.contains("#extension GL_EXT_buffer_reference : require"));
        assert!(glsl.contains("#extension GL_EXT_buffer_reference2 : require"));
        assert!(glsl.contains("#extension GL_EXT_shader_atomic_int64 : require"));
        assert!(glsl.contains("struct CacheCell"));
        assert!(glsl.contains(
            "layout(buffer_reference, scalar, buffer_reference_align = 8) buffer VslPtrRw_uint64_t_a8"
        ));
        assert!(glsl.contains(
            "layout(buffer_reference, scalar, buffer_reference_align = 16) buffer VslPtrRw_CacheCell_a16"
        ));
        assert!(glsl.contains(
            "VslPtrRw_uint64_t_a8 key_ptr = VslPtrRw_uint64_t_a8(uint64_t(keys) + uint64_t(int64_t(slot)) * 8ul);"
        ));
        assert!(
            glsl.contains("uint64_t prev = atomicCompSwap(key_ptr.value, EMPTY_KEY, hash_key);")
        );
        assert!(glsl.contains(
            "VslPtrRw_CacheCell_a16 cell = VslPtrRw_CacheCell_a16(uint64_t(cells) + uint64_t(int64_t(slot)) * 16ul);"
        ));
        assert!(glsl.contains("atomicAdd(cell.value.rad_r, q.x);"));
    }

    #[test]
    fn hoists_stateful_if_expr_and_emits_mc_globals_in_reflections() {
        let parsed = parse_source(FileId(0), REFLECTIONS_VSL);
        assert!(
            !parsed.has_errors(),
            "unexpected parse errors: {:?}",
            parsed.diagnostics
        );

        let glsl = emit_glsl_460(&parsed.unit);
        // Exact Hyaloclast 4-set Vulkan ABI.
        assert!(glsl.contains("layout(set = 0, binding = 0, scalar) uniform WorldUBO"));
        assert!(glsl.contains("layout(set = 0, binding = 1) uniform texture2D gbuffer_normal;"));
        assert!(glsl.contains("layout(set = 0, binding = 2) uniform texture2D gbuffer_depth;"));
        assert!(glsl.contains("layout(set = 0, binding = 3, rgba16f) uniform image2D out_refl;"));
        assert!(glsl.contains(
            "layout(set = 0, binding = 4, rgba16f) writeonly uniform image2D vsl_debug_custom_reflections;"
        ));
        assert!(
            glsl.contains("layout(set = 0, binding = 14) uniform sampler vsl_default_sampler;")
        );
        assert!(!glsl.contains("binding = 15"));
        assert!(
            glsl.contains("layout(set = 1, binding = 0) uniform accelerationStructureEXT tlas;")
        );
        assert!(
            glsl.contains("layout(set = 1, binding = 1, scalar) readonly buffer InstanceMetaSSBO")
        );
        assert!(glsl.contains("layout(set = 2, binding = 0) uniform texture2D bindless[];"));

        // Helper forward declarations for `mc::*` utilities.
        assert!(glsl.contains("vec3 mc_reconstruct_world_pos(ivec2 coord, texture2D depth_tex);"));
        assert!(glsl.contains("vec3 mc_view_dir(ivec2 coord);"));
        assert!(glsl.contains("vec3 mc_sample_sky(vec3 dir);"));

        // `mc::debug_write` lowered to `imageStore` with zero string literals in GLSL.
        assert!(
            glsl.contains("imageStore(vsl_debug_custom_reflections, coord, vec4(color, 1.0));")
        );
        assert!(!glsl.contains("\"custom_reflections\""));

        assert!(glsl.contains("textureLod(sampler2D(bindless[nonuniformEXT(inst.albedo_slot)], vsl_default_sampler), uv, 0.0).rgb"));
        assert!(glsl.contains("vec3 color;"));
        assert!(glsl.contains("InstanceMeta inst = mc_instances["));
        assert!(glsl.contains("vec2 uv = rayQueryGetIntersectionBarycentricsEXT(rq, true);"));
    }

    #[test]
    fn emits_trait_and_impl_methods_in_restir() {
        let parsed = parse_source(FileId(0), RESTIR_VSL);
        assert!(
            !parsed.has_errors(),
            "unexpected parse errors: {:?}",
            parsed.diagnostics
        );

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(glsl.contains("struct VslDefaultTarget {\n    vec3 pos;\n};"));
        assert!(glsl.contains("VslDefaultTarget _vsl_sample;"));
        assert!(glsl.contains("self._vsl_sample = candidate;"));
        assert!(glsl.contains("target_pdf(self._vsl_sample, world_pos, normal)"));
        assert!(glsl.contains("// trait ResampleTarget"));
        assert!(
            glsl.contains("float target_pdf(VslDefaultTarget self, vec3 world_pos, vec3 normal);")
        );
        assert!(glsl.contains(
            "bool Reservoir_update(inout Reservoir self, in VslDefaultTarget candidate,"
        ));
        assert!(glsl.contains("void Reservoir_finalize(inout Reservoir self,"));
        assert!(!glsl.contains("    S sample;"));
        assert!(!glsl.contains("in S candidate"));
        assert!(glsl.contains("atomicAdd(counter.value, count)"));
    }

    #[test]
    fn skips_push_constant_param_in_set0_bindings_and_parses_hex_workgroup() {
        let src = r"
            #[push_constant, layout(scalar)]
            pub struct Pc { pub exposure: f32 }

            @stage(compute(0x8u, 0x8u, 1u))
            pub fn main_cs(
                #[push_constant] pc: Pc,
                in_tex: Texture2D<f32>,
                out_img: RWImage2D<rgba16f>,
            ) {
                let c = in_tex.sample(vec2(0.5, 0.5));
                out_img.store(ivec2(0, 0), c * pc.exposure);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(glsl.contains("layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;"));
        assert!(glsl.contains("layout(set = 0, binding = 0, scalar) uniform WorldUBO"));
        assert!(glsl.contains("layout(set = 0, binding = 1) uniform texture2D in_tex;"));
        assert!(glsl.contains("layout(set = 0, binding = 2, rgba16f) uniform image2D out_img;"));
        assert!(
            glsl.contains("layout(set = 0, binding = 14) uniform sampler vsl_default_sampler;")
        );
        assert!(!glsl.contains("binding = 15"));
        assert!(!glsl.contains("uniform Pc pc;"));
    }

    #[test]
    fn renames_imported_world_and_hoists_nested_stateful_if_in_call_args() {
        let src = r"
            import mc::{world};

            @stage(compute(8, 8, 1))
            pub fn nested_hoist_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let t = world.time;
                out_img.store(
                    ivec2(gid.xy),
                    vec4(
                        if t > 0.5 {
                            let boost = t * 2.0;
                            vec3(boost, 0.0, 0.0)
                        } else {
                            vec3(0.0, 1.0, 0.0)
                        },
                        1.0,
                    ),
                );
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(glsl.contains("layout(set = 0, binding = 0, scalar) uniform WorldUBO"));
        assert!(glsl.contains("float t = mc_world.time;"));
        assert!(glsl.contains("vec3 _vsl_tmp_0;"));
        assert!(glsl.contains("float boost = (t * 2.0);"));
        assert!(glsl.contains("imageStore(out_img, ivec2(gid.xy), vec4(_vsl_tmp_0, 1.0));"));
    }

    #[test]
    fn emits_raster_gbuffer_hook_slots_and_routes_unified_params_to_vs_motion() {
        let src = r"
            @hook(vertex_displace)
            pub fn waving_foliage(inout world_pos: vec3, time: f32) {
                let a = mc::param(wind_strength);
                let b = mc::params.gust_speed;
                let c = params.leaf_scale;
                let d = pack::params.sway_bias;
                let enabled = mc::param(enable_wind);
                let steps: i32 = params.wave_octaves;
                let tint = params.foliage_tint.rgb;
                if enabled && steps > 0 {
                    world_pos.x += sin(time * b) * a * c + d + tint.x;
                }
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let func = parsed
            .unit
            .items
            .iter()
            .find_map(|item| match item {
                Item::Function(f) => Some(f),
                _ => None,
            })
            .expect("function item");

        let vd = emit_raster_gbuffer_hook_slot(HookStage::VertexDisplace, func)
            .expect("vertex_displace slot");
        assert!(vd.contains("#extension GL_EXT_scalar_block_layout : require"));
        assert!(vd.contains("struct VsPackParams"));
        assert!(vd.contains("layout(set = 0, binding = 0, scalar) uniform VsMotionBlock"));
        assert!(vd.contains("} VsMotion;"));
        assert!(vd.contains("bool enable_wind;"));
        assert!(vd.contains("int wave_octaves;"));
        assert!(vd.contains("vec3 foliage_tint;"));
        assert!(vd.contains("float wind_strength;"));
        assert!(vd.contains("void vs_dispatch_hook_vertex_displace("));
        assert!(vd.contains("VsMotion.packParams.wind_strength"));
        assert!(vd.contains("VsMotion.packParams.gust_speed"));
        assert!(vd.contains("VsMotion.packParams.leaf_scale"));
        assert!(vd.contains("VsMotion.packParams.sway_bias"));
        assert!(vd.contains("waving_foliage(world_pos, time);"));

        // Explicit param_types override helper.
        let mut overrides = IndexMap::new();
        overrides.insert(SmolStr::new_static("wind_strength"), "vec3".to_owned());
        let custom =
            emit_raster_gbuffer_hook_slot_with_types(HookStage::VertexDisplace, func, &overrides)
                .expect("vertex_displace slot with overrides");
        assert!(custom.contains("vec3 wind_strength;"));

        let surf = emit_raster_gbuffer_hook_slot(HookStage::Surface, func).expect("surface slot");
        assert!(surf.contains("void vs_dispatch_hook_surface("));

        let gbuf =
            emit_raster_gbuffer_hook_slot(HookStage::GbufferPack, func).expect("gbuffer_pack slot");
        assert!(gbuf.contains("void vs_dispatch_hook_gbuffer_pack("));
    }

    #[test]
    fn rejects_non_raster_hooks_in_gbuffer_bridge() {
        let src = r"
            @hook(clouds)
            pub fn custom_clouds(dir: vec3) -> vec4 {
                return vec4(dir, 1.0);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors());

        let func = parsed
            .unit
            .items
            .iter()
            .find_map(|item| match item {
                Item::Function(f) => Some(f),
                _ => None,
            })
            .expect("function item");

        let err = emit_raster_gbuffer_hook_slot(HookStage::Clouds, func)
            .expect_err("clouds hook must be rejected by raster G-Buffer bridge");
        assert!(err.contains("clouds"));
    }

    #[test]
    fn escapes_glsl_reserved_keywords_without_touching_texture_methods() {
        let parsed = parse_source(FileId(0), RESTIR_VSL);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let restir_glsl = emit_glsl_460(&parsed.unit);
        assert!(restir_glsl.contains("VslDefaultTarget _vsl_sample;"));
        assert!(restir_glsl.contains("self._vsl_sample = candidate;"));
        assert!(!restir_glsl.contains("VslDefaultTarget sample;"));

        let src = r"
            pub struct FilterState {
                pub sample: f32,
                pub input: vec3,
                pub output: vec3,
            }

            @stage(compute(8, 8, 1))
            pub fn keyword_test(
                in_tex: Texture2D<f32>,
                out_img: RWImage2D<rgba16f>,
            ) {
                let sample = in_tex.sample(vec2(0.5, 0.5)).r;
                let filter = in_tex.sample_lod(vec2(0.5, 0.5), 0.0).rgb;
                let st = FilterState { sample: sample, input: filter, output: filter };
                out_img.store(ivec2(0, 0), vec4(st.output * st.sample, 1.0));
            }
        ";
        let parsed_kw = parse_source(FileId(0), src);
        assert!(!parsed_kw.has_errors(), "{:?}", parsed_kw.diagnostics);
        let glsl = emit_glsl_460(&parsed_kw.unit);
        assert!(glsl.contains("float _vsl_sample;"));
        assert!(glsl.contains("vec3 _vsl_input;"));
        assert!(glsl.contains("vec3 _vsl_output;"));
        assert!(glsl.contains(
            "float _vsl_sample = texture(sampler2D(in_tex, vsl_default_sampler), vec2(0.5, 0.5)).r;"
        ));
        assert!(glsl.contains("vec3 _vsl_filter = textureLod(sampler2D(in_tex, vsl_default_sampler), vec2(0.5, 0.5), 0.0).rgb;"));
        assert!(glsl.contains("FilterState(_vsl_sample, _vsl_filter, _vsl_filter)"));
        assert!(glsl.contains("(st._vsl_output * st._vsl_sample)"));
    }

    #[test]
    fn converts_binary_and_octal_integer_literals_to_hex() {
        let src = r"
            @stage(compute(1, 1, 1))
            pub fn lit_cs(out_img: RWImage2D<rgba16f>) {
                let a = 0b1010_0101u;
                let b = 0B1111;
                let c = 0o755;
                let d = 0O77u32;
                let e = 0b1000u64;
                out_img.store(ivec2(0, 0), vec4(f32(a + d), f32(b + c), f32(e), 1.0));
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let glsl = emit_glsl_460(&parsed.unit);
        assert!(glsl.contains("uint a = 0xA5u;"), "got:\n{glsl}");
        assert!(glsl.contains("int b = 0xF;"), "got:\n{glsl}");
        assert!(glsl.contains("int c = 0x1ED;"), "got:\n{glsl}");
        assert!(glsl.contains("uint d = 0x3Fu;"), "got:\n{glsl}");
        assert!(glsl.contains("uint64_t e = 0x8ul;"), "got:\n{glsl}");
        assert!(!glsl.contains("0b"));
        assert!(!glsl.contains("0B"));
        assert!(!glsl.contains("0o"));
        assert!(!glsl.contains("0O"));
    }

    #[test]
    fn assigns_ascending_set0_debug_channel_bindings_matching_hir() {
        let src = r#"
            @stage(compute(8, 8, 1))
            pub fn multi_debug(
                #[builtin(global_invocation_id)] gid: uvec3,
                tex_a: Texture2D<f32>,
                out_img: RWImage2D<rgba16f>,
            ) {
                let coord = ivec2(gid.xy);
                let col = tex_a.fetch(coord, 0).rgb;
                out_img.store(coord, vec4(col, 1.0));
                mc::debug_write("albedo_dbg", coord, col);
                mc::debug_write("normal_dbg", coord, col);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let glsl = emit_glsl_460(&parsed.unit);
        assert!(glsl.contains("layout(set = 0, binding = 1) uniform texture2D tex_a;"));
        assert!(glsl.contains("layout(set = 0, binding = 2, rgba16f) uniform image2D out_img;"));
        assert!(glsl.contains(
            "layout(set = 0, binding = 3, rgba16f) writeonly uniform image2D vsl_debug_albedo_dbg;"
        ));
        assert!(glsl.contains(
            "layout(set = 0, binding = 4, rgba16f) writeonly uniform image2D vsl_debug_normal_dbg;"
        ));
        assert!(
            glsl.contains("layout(set = 0, binding = 14) uniform sampler vsl_default_sampler;")
        );
        assert!(!glsl.contains("binding = 15"));
    }

    #[test]
    fn infers_vec3_pack_params_from_mul_and_div_in_hook_bridge() {
        let src = r"
            @hook(surface)
            pub fn tint_surface(inout albedo: vec3) {
                let lit = mc::param(sun_tint) * vec3(1.0);
                let inv = mc::param(inv_tint) / vec3(2.0);
                albedo *= mc::param(mul_tint);
                albedo /= mc::param(div_tint);
                albedo += lit + inv;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let func = parsed
            .unit
            .items
            .iter()
            .find_map(|item| match item {
                Item::Function(f) => Some(f),
                _ => None,
            })
            .expect("function item");

        let glsl = emit_raster_gbuffer_hook_slot(HookStage::Surface, func).expect("surface slot");
        assert!(glsl.contains("vec3 sun_tint;"), "got:\n{glsl}");
        assert!(glsl.contains("vec3 inv_tint;"), "got:\n{glsl}");
        assert!(glsl.contains("vec3 mul_tint;"), "got:\n{glsl}");
        assert!(glsl.contains("vec3 div_tint;"), "got:\n{glsl}");
    }

    #[test]
    fn avoids_set0_binding_collision_for_default_sampler_with_14_pass_resources() {
        let src = r"
            @stage(compute(8, 8, 1))
            pub fn many_resources_cs(
                t1: Texture2D<f32>,
                t2: Texture2D<f32>,
                t3: Texture2D<f32>,
                t4: Texture2D<f32>,
                t5: Texture2D<f32>,
                t6: Texture2D<f32>,
                t7: Texture2D<f32>,
                t8: Texture2D<f32>,
                t9: Texture2D<f32>,
                t10: Texture2D<f32>,
                t11: Texture2D<f32>,
                t12: Texture2D<f32>,
                t13: Texture2D<f32>,
                out_img: RWImage2D<rgba16f>,
            ) {
                let c = t1.sample(vec2(0.5, 0.5));
                out_img.store(ivec2(0, 0), c);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(
            glsl.contains("layout(set = 0, binding = 1) uniform texture2D t1;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 13) uniform texture2D t13;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 14, rgba16f) uniform image2D out_img;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 15) uniform sampler vsl_default_sampler;"),
            "got:\n{glsl}"
        );
        assert!(
            !glsl.contains("layout(set = 0, binding = 14) uniform sampler vsl_default_sampler;"),
            "got:\n{glsl}"
        );

        let mut seen_bindings = std::collections::BTreeSet::new();
        for line in glsl.lines() {
            if let Some(rest) = line.strip_prefix("layout(set = 0, binding = ") {
                let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
                let binding: u32 = digits.parse().expect("valid binding number");
                assert!(
                    seen_bindings.insert(binding),
                    "duplicate Set 0 binding {binding} in:\n{glsl}"
                );
            }
        }
        assert!(seen_bindings.contains(&0), "missing ABI WorldUBO binding");
        assert_eq!(seen_bindings.len(), 16, "got:\n{glsl}");
    }

    #[test]
    fn no_return_emitted_in_void_or_loop_and_supports_buffer_impl_float_atomics() {
        let src = r"
            struct Reservoir {
                weight_sum: f32,
            }

            impl Reservoir {
                pub fn update(mut self, w: f32) -> f32 {
                    self.weight_sum = self.weight_sum + w;
                    self.weight_sum
                }
            }

            @stage(compute(8, 8, 1))
            #[cap(float_atomics)]
            pub fn test_cs(
                in_buf: Buffer<f32>,
                out_buf: RWBuffer<f32>,
            ) {
                let mut res = Reservoir { weight_sum: 0.0 };
                for i in 0u..4u {
                    if i < 2u {
                        let _ = res.update(in_buf[i]);
                    }
                }
                if res.weight_sum > 0.0 {
                    out_buf[0] = res.weight_sum;
                }
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(
            glsl.contains("#extension GL_EXT_shader_atomic_float : require"),
            "missing float_atomics extension:
{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 1, scalar) readonly buffer VslBuf_in_buf"),
            "missing readonly Buffer SSBO:
{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 2, scalar) buffer VslBuf_out_buf"),
            "missing RWBuffer SSBO:
{glsl}"
        );
        assert!(
            glsl.contains("Reservoir_update(res,"),
            "expected mangled impl method call:
{glsl}"
        );
        let main_start = glsl.find("void main()").expect("void main()");
        let main_body = &glsl[main_start..];
        assert!(
            !main_body.contains("return "),
            "void main() must never emit a value return:
{main_body}"
        );
    }

    #[test]
    fn sunlit_vanilla_examples_emit_no_value_return_in_void_functions() {
        let tiled_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/sunlit_vanilla/tiled_lighting.vsl");
        let tiled_src = std::fs::read_to_string(&tiled_path).expect("tiled_lighting.vsl");
        let tiled_parsed = parse_source(FileId(0), &tiled_src);
        assert!(!tiled_parsed.has_errors(), "{:?}", tiled_parsed.diagnostics);
        let tiled_glsl = emit_glsl_460(&tiled_parsed.unit);
        let main_start = tiled_glsl.find("void main()").expect("void main()");
        let main_body = &tiled_glsl[main_start..];
        assert!(
            !main_body.contains("return "),
            "tiled_lighting.vsl void main() must not emit value return:
{main_body}"
        );

        let restir_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/sunlit_vanilla/restir.vsl");
        let restir_src = std::fs::read_to_string(&restir_path).expect("restir.vsl");
        let restir_parsed = parse_source(FileId(0), &restir_src);
        assert!(
            !restir_parsed.has_errors(),
            "{:?}",
            restir_parsed.diagnostics
        );
        let restir_glsl = emit_glsl_460(&restir_parsed.unit);
        let fn_start = restir_glsl
            .find("void append_active_pixels(")
            .expect("void append_active_pixels");
        let fn_body = &restir_glsl[fn_start..];
        assert!(
            !fn_body.contains("return "),
            "restir.vsl void append_active_pixels() must not emit value return:
{fn_body}"
        );
    }
}
