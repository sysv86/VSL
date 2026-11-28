//! Stage A GLSL 460 oracle backend and Hyaloclast raster G-Buffer hook bridge for VSL 2.0.

pub mod emit;
pub mod hook_bridge;

pub use emit::{
    BindlessAbi, EmitOptions, emit_glsl_460, emit_glsl_460_entry_with_options,
    emit_glsl_460_with_options,
};
pub use hook_bridge::{
    emit_raster_gbuffer_hook_slot, emit_raster_gbuffer_hook_slot_from_unit,
    emit_raster_gbuffer_hook_slot_with_schema, emit_raster_gbuffer_hook_slot_with_types,
};

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use smol_str::SmolStr;
    use vsl_diagnostics::FileId;
    use vsl_mc::{HookStage, PackParamType};
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
        assert!(
            glsl.contains("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require")
        );
        assert!(glsl.contains("layout(set = 0, binding = 0, std140) uniform WorldUBO"));
        assert!(glsl.contains("mc_world.cur_view_proj"));
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
            glsl.contains("layout(set = 1, binding = 3, std430) readonly buffer EmissiveLightSSBO")
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
            "layout(buffer_reference, std430, buffer_reference_align = 16) buffer VslPtrRw_CacheCell_a16"
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
        assert!(glsl.contains("layout(set = 0, binding = 0, std140) uniform WorldUBO"));
        assert!(glsl.contains("layout(set = 0, binding = 1) uniform texture2D gbuffer_normal;"));
        assert!(glsl.contains("layout(set = 0, binding = 2) uniform texture2D gbuffer_depth;"));
        assert!(glsl.contains("layout(set = 0, binding = 3, rgba16f) uniform image2D out_refl;"));
        assert!(glsl.contains(
            "layout(set = 0, binding = 4, rgba16f) writeonly uniform image2D vsl_debug_custom_reflections;"
        ));
        assert!(!glsl.contains("vsl_default_sampler"));
        assert!(!glsl.contains("binding = 15"));
        assert!(
            glsl.contains("layout(set = 1, binding = 0) uniform accelerationStructureEXT tlas;")
        );
        assert!(
            glsl.contains("layout(set = 1, binding = 1, std430) readonly buffer InstanceMetaSSBO")
        );
        assert!(glsl.contains("layout(set = 2, binding = 0) uniform sampler2D bindless[4096];"));

        // Helper forward declarations for `mc::*` utilities.
        assert!(glsl.contains("vec3 mc_reconstruct_world_pos(ivec2 coord, texture2D depth_tex);"));
        assert!(glsl.contains("vec3 mc_view_dir(ivec2 coord);"));
        assert!(glsl.contains("vec3 mc_sample_sky(vec3 dir);"));

        // `mc::debug_write` lowered to `imageStore` with zero string literals in GLSL.
        assert!(
            glsl.contains("imageStore(vsl_debug_custom_reflections, coord, vec4(color, 1.0));")
        );
        assert!(!glsl.contains("\"custom_reflections\""));

        assert!(glsl.contains("textureLod(bindless[nonuniformEXT(inst.texture_id)], uv, 0.0).rgb"));
        assert!(glsl.contains("vec3 color;"));
        assert!(glsl.contains("InstanceMeta inst = mc_instances["));
        assert!(glsl.contains("vec2 uv = rayQueryGetIntersectionBarycentricsEXT(rq, true);"));
    }

    const POSITION_FETCH_VSL: &str = r"
        #[cap(ray_query, position_fetch, omm, omm_khr)]
        @stage(compute(1, 1, 1))
        fn query_main(
            #[binding(set = 0, binding = 0)] tlas: AccelerationStructure,
            #[binding(set = 0, binding = 1)] result_buf: RWBuffer<f32>,
        ) {
            let mut rq: RayQuery;
            let flags = RayFlags::FORCE_OPACITY_MICROMAP_2_STATE;
            rq.initialize(tlas, flags, 255u, vec3(0.0, 0.0, 0.0), 0.001, vec3(0.0, 0.0, 1.0), 100.0);
            while rq.proceed() {
                if rq.candidate_type() == 0u {
                    let candidate = rq.candidate_triangle_vertex_positions();
                    let first = candidate[0u];
                    result_buf[0u] = first.x;
                    rq.confirm_intersection();
                }
            }
            if rq.committed_type() == mc::HitType::Triangle {
                let committed: [vec3; 3] = rq.committed_triangle_vertex_positions();
                let second = committed[1u];
                let third = rq.committed_triangle_vertex_positions()[2u];
                result_buf[1u] = second.x + third.y;
            }
        }
    ";

    fn emit_standalone_test_shader(src: &str) -> String {
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        emit_glsl_460_with_options(
            &parsed.unit,
            EmitOptions {
                abi: vsl_mc::AbiProfile::Standalone,
                ..EmitOptions::default()
            },
        )
    }

    #[test]
    fn dispatches_ray_query_methods_only_for_ray_query_receivers() {
        let glsl = emit_standalone_test_shader(
            r"
            struct Cursor { value: f32 }
            impl Cursor {
                fn initialize(self) -> f32 { self.value }
                fn proceed(self) -> f32 { self.value }
                fn committed_type(self) -> vec2 { vec2(self.value, 0.0) }
                fn committed_triangle_vertex_positions(self) -> u32 { 7u }
            }
            #[cap(ray_query)]
            @stage(compute(1, 1, 1))
            fn query_main(
                #[binding(set = 0, binding = 0)] tlas: AccelerationStructure,
                #[binding(set = 0, binding = 1)] result_buf: RWBuffer<f32>,
            ) {
                let cursor = Cursor { value: 1.0 };
                let init = cursor.initialize();
                let step = cursor.proceed();
                let kind = cursor.committed_type();
                let vertices = cursor.committed_triangle_vertex_positions();
                let mut rq: RayQuery;
                rq.initialize(tlas, mc::RayFlags::OPAQUE, 255u, vec3(0.0, 0.0, 0.0), 0.001, vec3(0.0, 0.0, 1.0), 100.0);
                while rq.proceed() {}
                let hit = rq.committed_type();
                result_buf[0u] = init + step + kind.x + f32(vertices) + f32(hit);
            }
        ",
        );
        assert!(
            glsl.contains("float init = Cursor_initialize(cursor);"),
            "{glsl}"
        );
        assert!(
            glsl.contains("float step = Cursor_proceed(cursor);"),
            "{glsl}"
        );
        assert!(
            glsl.contains("vec2 kind = Cursor_committed_type(cursor);"),
            "{glsl}"
        );
        assert!(
            glsl.contains("uint vertices = Cursor_committed_triangle_vertex_positions(cursor);"),
            "{glsl}"
        );
        assert!(glsl.contains("while (rayQueryProceedEXT(rq))"), "{glsl}");
        assert!(
            glsl.contains("uint hit = rayQueryGetIntersectionTypeEXT(rq, true);"),
            "{glsl}"
        );
        assert!(
            !glsl.contains("GL_EXT_ray_tracing_position_fetch"),
            "{glsl}"
        );
        validate_glsl_if_available("receiver_dispatch", &glsl, false);
    }

    #[test]
    fn emits_ray_query_position_fetch_arrays_and_opacity_micromap_modes() {
        let glsl = emit_standalone_test_shader(POSITION_FETCH_VSL);
        for expected in [
            "#extension GL_EXT_ray_query : require",
            "#extension GL_EXT_ray_tracing_position_fetch : require",
            "#extension GL_EXT_opacity_micromap : require",
            "#extension GL_EXT_opacity_micromap_ray_query_mode : require",
            "const bool gl_EnableOpacityMicromapEXT = true;",
            "rayQueryGetIntersectionTriangleVertexPositionsEXT(q, true, positions);",
            "rayQueryGetIntersectionTriangleVertexPositionsEXT(q, false, positions);",
            "uint flags = gl_RayFlagsForceOpacityMicromap2StateEXT;",
            "vec3[3] candidate = _vsl_ray_query_candidate_triangle_vertex_positions(rq);",
            "vec3 first = candidate[0u];",
            "vec3 committed[3] = _vsl_ray_query_committed_triangle_vertex_positions(rq);",
            "vec3 second = committed[1u];",
            "vec3 third = _vsl_ray_query_committed_triangle_vertex_positions(rq)[2u];",
        ] {
            assert!(glsl.contains(expected), "missing {expected}:\n{glsl}");
        }
        assert_eq!(
            glsl.matches("const bool gl_EnableOpacityMicromapEXT = true;")
                .count(),
            1
        );
        let aliases = POSITION_FETCH_VSL.replace(
            "position_fetch, omm, omm_khr",
            "ray_tracing_position_fetch, opacity_micromap, opacity_micromap_khr",
        );
        assert_eq!(glsl, emit_standalone_test_shader(&aliases));
    }

    #[test]
    fn keeps_legacy_and_khr_opacity_micromap_capabilities_independent() {
        let legacy = emit_standalone_test_shader(&POSITION_FETCH_VSL.replace(", omm_khr", ""));
        assert!(legacy.contains("#extension GL_EXT_opacity_micromap : require"));
        assert!(!legacy.contains("GL_EXT_opacity_micromap_ray_query_mode"));
        assert!(!legacy.contains("gl_EnableOpacityMicromapEXT"));
        let khr = emit_standalone_test_shader(&POSITION_FETCH_VSL.replace(", omm,", ",").replace(
            "RayFlags::FORCE_OPACITY_MICROMAP_2_STATE",
            "mc::RayFlags::OPAQUE",
        ));
        assert!(!khr.contains("#extension GL_EXT_opacity_micromap : require"));
        assert!(khr.contains("#extension GL_EXT_opacity_micromap_ray_query_mode : require"));
        assert!(khr.contains("const bool gl_EnableOpacityMicromapEXT = true;"));
    }

    fn validate_glsl_if_available(label: &str, glsl: &str, check_omm_mode: bool) {
        validate_glsl_stage(label, glsl, "comp", check_omm_mode);
    }

    fn validate_glsl_stage(label: &str, glsl: &str, stage: &str, check_omm_mode: bool) {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let path =
            std::env::temp_dir().join(format!("vsl_glsl_{label}_{}.spv", std::process::id()));
        let child = Command::new("glslangValidator")
            .args([
                "--stdin",
                "-S",
                stage,
                "-V",
                "--target-env",
                "vulkan1.3",
                "-o",
            ])
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound && !check_omm_mode => return,
            Err(err) => panic!("failed to invoke glslangValidator: {err}"),
        };
        child
            .stdin
            .take()
            .expect("glslang stdin")
            .write_all(glsl.as_bytes())
            .expect("write GLSL");
        let compiled = child.wait_with_output().expect("wait for glslang");
        assert!(
            compiled.status.success(),
            "glslang failed: {}\n{}\n{glsl}",
            String::from_utf8_lossy(&compiled.stdout),
            String::from_utf8_lossy(&compiled.stderr)
        );
        let validated = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&path)
            .output();
        let disassembled = check_omm_mode.then(|| {
            Command::new("spirv-dis")
                .arg(&path)
                .output()
                .expect("spirv-dis")
        });
        std::fs::remove_file(&path).expect("remove temporary SPIR-V");
        match validated {
            Ok(out) => assert!(
                out.status.success(),
                "spirv-val: {}",
                String::from_utf8_lossy(&out.stderr)
            ),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound && !check_omm_mode => {}
            Err(err) => panic!("failed to invoke spirv-val: {err}"),
        }
        if let Some(disassembled) = disassembled {
            assert!(disassembled.status.success());
            let assembly = String::from_utf8_lossy(&disassembled.stdout);
            for expected in [
                "OpRayQueryGetIntersectionTriangleVertexPositionsKHR",
                "RayTracingOpacityMicromapExecutionModeKHR",
                "OpacityMicromapIdKHR",
                "OpConstantTrue",
            ] {
                assert!(
                    assembly.contains(expected),
                    "missing {expected}:\n{assembly}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires glslang and SPIRV-Tools with 2026 KHR opacity micromap support"]
    fn validates_generated_ray_query_extensions_with_khronos_tools() {
        let glsl = emit_standalone_test_shader(POSITION_FETCH_VSL);
        validate_glsl_if_available("ray_query_extensions", &glsl, true);
    }

    fn standalone_options() -> EmitOptions {
        EmitOptions {
            abi: vsl_mc::AbiProfile::Standalone,
            ..EmitOptions::default()
        }
    }

    fn selected_glsl(src: &str, entry: &str) -> Result<String, String> {
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        emit_glsl_460_entry_with_options(&parsed.unit, entry, standalone_options())
    }

    const RT_STAGE_SOURCE: &str = r"
        @raygen fn launch(
            #[binding(set=0, binding=0)] scene: AccelerationStructure,
            #[binding(set=0, binding=1)] image: RWImage2D<rgba16f>,
            #[ray_payload(3)] inout color: vec4,
            #[callable_data(2)] inout data: vec4,
            #[builtin(launch_id)] pixel: uvec3,
        ) {
            color = vec4(1.0);
            trace_ray(scene, 0u, 255u, 0u, 0u, 0u, vec3(0.0), 0.01, vec3(0.0, 0.0, 1.0), 32.0, color);
            data = color;
            execute_callable(0u, data);
            image.store(ivec2(pixel.xy), data);
        }
        @stage(closest_hit) fn hit(
            #[incoming_ray_payload(3)] inout color: vec4,
            #[hit_attribute] uv: vec2,
            #[builtin(ray_tmax)] distance: f32,
            #[builtin(instance_custom_index)] instance: u32,
        ) { color = vec4(uv.x, uv.y, distance + f32(instance), 1.0); }
        @miss fn miss_color(#[incoming_ray_payload(3)] inout color: vec4) { color = vec4(0.0); }
        @any_hit fn filter_hit(#[incoming_ray_payload(3)] inout color: vec4) {
            if color.x < 0.0 { ignore_intersection(); } else { terminate_ray(); }
        }
        @intersection fn intersect(#[hit_attribute] out attribute_value: vec2) {
            attribute_value = vec2(0.25);
            let accepted = report_intersection(1.0, 0u);
        }
        @callable fn shade(#[incoming_callable_data(2)] inout data: vec4) { data = data.zyxw; }
    ";

    #[test]
    fn emits_single_rt_entry_with_native_interfaces_and_shader_calls() {
        for (entry, stage, expected) in [
            (
                "launch",
                "rgen",
                "layout(location = 3) rayPayloadEXT vec4 color;",
            ),
            (
                "hit",
                "rchit",
                "layout(location = 3) rayPayloadInEXT vec4 color;",
            ),
            (
                "miss_color",
                "rmiss",
                "layout(location = 3) rayPayloadInEXT vec4 color;",
            ),
            ("filter_hit", "rahit", "ignoreIntersectionEXT;"),
            (
                "intersect",
                "rint",
                "bool accepted = reportIntersectionEXT(1.0, 0u);",
            ),
            (
                "shade",
                "rcall",
                "layout(location = 2) callableDataInEXT vec4 data;",
            ),
        ] {
            let glsl = selected_glsl(RT_STAGE_SOURCE, entry).unwrap();
            assert_eq!(glsl.matches("void main()").count(), 1, "{glsl}");
            assert!(glsl.contains(expected), "missing {expected}:\n{glsl}");
            assert!(glsl.contains("#extension GL_EXT_ray_tracing : require"));
            if entry == "launch" {
                assert!(glsl.contains("32.0, 3);"), "{glsl}");
                assert!(glsl.contains("executeCallableEXT(0u, 2);"), "{glsl}");
            }
            validate_glsl_stage(entry, &glsl, stage, false);
        }
    }

    #[test]
    fn rejects_ambiguous_entries_unsupported_stages_and_bad_rt_interface() {
        let glsl = emit_standalone_test_shader(RT_STAGE_SOURCE);
        assert!(glsl.contains("#error Multiple entry points"));
        assert!(selected_glsl(RT_STAGE_SOURCE, "absent").is_err());
        assert!(selected_glsl("@mesh fn mesh_main() {}", "mesh_main").is_err());
        assert!(
            selected_glsl(
                "@raygen fn bad(#[incoming_ray_payload(0)] p: vec4) {}",
                "bad"
            )
            .is_err()
        );
        assert!(
            selected_glsl(
                "@raygen fn bad(#[ray_payload(0)] p: vec4) { execute_callable(0u, p); }",
                "bad"
            )
            .is_err()
        );
        assert!(selected_glsl("@raygen fn bad() { let hit: HitObject; }", "bad").is_err());
        assert!(selected_glsl("#[cap(ser_nv, ser_ext)] @raygen fn bad() {}", "bad").is_err());
    }

    #[test]
    fn emits_shader_record_block_and_position_fetch_builtin() {
        let glsl = selected_glsl(
            r"
            #[layout(std430)] struct Coefficients { gain: vec4, bias: vec4 }
            @raygen fn read_record(
                #[binding(set=0,binding=0)] image: RWImage2D<rgba16f>,
                #[shader_record] record: Coefficients,
            ) { image.store(ivec2(0,0), record.gain + record.bias); }
        ",
            "read_record",
        )
        .unwrap();
        assert!(
            glsl.contains("layout(shaderRecordEXT, std430) buffer"),
            "{glsl}"
        );
        validate_glsl_stage("shader_record", &glsl, "rgen", false);
        let glsl = selected_glsl(
            r"
            #[cap(position_fetch)] @closest_hit fn fetch_vertices(
                #[incoming_ray_payload(0)] inout color: vec4,
                #[builtin(hit_triangle_vertex_positions)] positions: [vec3; 3],
            ) { color = vec4(positions[0u], 1.0); }
        ",
            "fetch_vertices",
        )
        .unwrap();
        assert!(glsl.contains("gl_HitTriangleVertexPositionsEXT"));
        validate_glsl_stage("hit_positions", &glsl, "rchit", false);
    }

    #[test]
    #[ignore = "requires glslang and SPIRV-Tools with NV and EXT SER support"]
    fn validates_both_hit_object_profiles() {
        for (cap, suffix) in [("ser_nv", "NV"), ("ser_ext", "EXT")] {
            let src = format!(
                r"
                #[cap({cap})] @raygen fn deferred(
                    #[binding(set=0,binding=0)] scene: AccelerationStructure,
                    #[binding(set=0,binding=1)] image: RWImage2D<rgba16f>,
                    #[ray_payload(4)] inout color: vec4,
                ) {{
                    let hit: HitObject;
                    hit.record_empty();
                    color = vec4(0.0);
                    hit.trace_ray(scene, 0u, 255u, 0u, 0u, 0u, vec3(0.0), 0.01, vec3(0.0,0.0,1.0), 32.0, color);
                    reorder_thread_with_hit_object(hit, 1u, 2u);
                    if hit.is_hit() {{
                        let distance = hit.ray_tmax();
                        let record = hit.sbt_record_index();
                        image.store(ivec2(0,0), vec4(distance + f32(record)));
                    }}
                    hit.execute_shader(color);
                    reorder_thread_with_hint(0u, 1u);
                    image.store(ivec2(1,0), color);
                }}
            "
            );
            let glsl = selected_glsl(&src, "deferred").unwrap();
            assert!(
                glsl.contains(&format!("hitObjectTraceRay{suffix}(hit,")),
                "{glsl}"
            );
            assert!(
                glsl.contains(&format!("hitObjectExecuteShader{suffix}(hit, 4);")),
                "{glsl}"
            );
            validate_glsl_stage(cap, &glsl, "rgen", false);
        }
    }

    #[test]
    fn emits_all_ray_query_getters_with_native_result_types() {
        let glsl = emit_standalone_test_shader(
            r"
            #[cap(ray_query)] @stage(compute(1,1,1)) fn query(
                #[binding(set=0,binding=0)] scene: AccelerationStructure,
                #[binding(set=0,binding=1)] image: RWImage2D<rgba16f>,
            ) {
                let mut rq: RayQuery;
                rq.initialize(scene, 0u, 255u, vec3(0.0), 0.01, vec3(0.0,0.0,1.0), 32.0);
                let flags = rq.ray_flags();
                let tmin = rq.ray_tmin();
                let origin = rq.world_ray_origin();
                let direction = rq.world_ray_direction();
                while rq.proceed() {
                    let index = rq.candidate_instance_id();
                    let custom = rq.candidate_instance_custom_index();
                    let geom = rq.candidate_geometry_index();
                    let prim = rq.candidate_primitive_index();
                    let offset = rq.candidate_sbt_record_offset();
                    let object_origin = rq.candidate_object_ray_origin();
                    let object_direction = rq.candidate_object_ray_direction();
                    let object_to_world = rq.candidate_object_to_world();
                    let world_to_object = rq.candidate_world_to_object();
                    let column = object_to_world[0u];
                    if rq.candidate_type() == 0u {
                        let front = rq.candidate_front_face();
                        let uv = rq.candidate_barycentrics();
                        let t = rq.candidate_t();
                        if front { image.store(ivec2(0,0), vec4(uv, t, 1.0)); }
                        rq.confirm_intersection();
                    } else {
                        let opaque = rq.candidate_aabb_opaque();
                        if opaque { rq.generate_intersection(1.0); }
                    }
                }
                if rq.committed_type() == 1u {
                    let index = rq.committed_instance_id();
                    let custom = rq.committed_instance_custom_index();
                    let geom = rq.committed_geometry_index();
                    let prim = rq.committed_primitive_index();
                    let offset = rq.committed_sbt_record_offset();
                    let object_origin = rq.committed_object_ray_origin();
                    let object_direction = rq.committed_object_ray_direction();
                    let object_to_world = rq.committed_object_to_world();
                    let world_to_object = rq.committed_world_to_object();
                    let column = world_to_object[0u];
                    let front = rq.committed_front_face();
                    let uv = rq.committed_barycentrics();
                    let t = rq.committed_t();
                    if front { image.store(ivec2(1,0), vec4(uv, t, 1.0)); }
                }
                image.store(ivec2(2,0), vec4(origin + direction, tmin + f32(flags)));
            }
        ",
        );
        for expected in [
            "uint flags = rayQueryGetRayFlagsEXT(rq);",
            "float tmin = rayQueryGetRayTMinEXT(rq);",
            "vec3 origin = rayQueryGetWorldRayOriginEXT(rq);",
            "vec3 direction = rayQueryGetWorldRayDirectionEXT(rq);",
            "uint index = uint(rayQueryGetIntersectionInstanceIdEXT(rq, false));",
            "uint offset = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, true);",
            "mat4x3 object_to_world = rayQueryGetIntersectionObjectToWorldEXT(rq, false);",
            "vec3 column = object_to_world[0u];",
            "bool opaque = rayQueryGetIntersectionCandidateAABBOpaqueEXT(rq);",
            "vec2 uv = rayQueryGetIntersectionBarycentricsEXT(rq, false);",
        ] {
            assert!(glsl.contains(expected), "missing {expected}:\n{glsl}");
        }
        validate_glsl_if_available("all_query_getters", &glsl, false);
        assert!(selected_glsl("#[cap(ray_query)] @stage(compute(1,1,1)) fn bad() { let rq: RayQuery; rq.proceed(1u); }", "bad").is_err());
        assert!(selected_glsl("#[cap(ray_query)] @stage(compute(1,1,1)) fn bad() { let rq: RayQuery; rq.unknown(); }", "bad").is_err());
    }

    #[test]
    fn preserves_lazy_branches_and_while_condition_preludes() {
        let glsl = selected_glsl(r"
            @stage(compute(1,1,1)) fn control(
                #[binding(set=0,binding=0)] result: RWBuffer<u32>,
            ) {
                let mut count = 0u;
                let enabled = false;
                let active = enabled && { count = count + 1u; count > 0u };
                let alternative = enabled || { count = count + 2u; count > 0u };
                let value: u32 = if active { { count = count + 4u; count } } else if { count = count + 8u; alternative } { 2u } else { 3u };
                while { count = count + 1u; count < 20u } {
                    if count == 15u { continue; }
                }
                result[0u] = count + value;
            }
        ", "control").unwrap();
        let and_guard = glsl.find("if (_vsl_tmp_").unwrap();
        let and_increment = glsl.find("count = (count + 1u);").unwrap();
        assert!(and_guard < and_increment, "{glsl}");
        assert!(glsl.contains("if (!_vsl_tmp_"), "{glsl}");
        assert!(glsl.contains("} else {"), "{glsl}");
        let loop_start = glsl.find("while (true)").unwrap();
        assert!(
            glsl[loop_start..].contains("count = (count + 1u);"),
            "{glsl}"
        );
        validate_glsl_if_available("lazy_control", &glsl, false);
    }

    #[test]
    fn rejects_invalid_rt_locations_and_mixed_shader_record_layouts() {
        for source in [
            "@raygen fn bad(#[ray_payload(-1)] p: vec4) {}",
            "@raygen fn bad(#[ray_payload(0, 1)] p: vec4) {}",
            "@raygen fn bad(#[ray_payload(0)] a: vec4, #[ray_payload(0)] b: vec4) {}",
            "@raygen fn bad(#[shader_record] p: f32) {}",
            "struct Record { value: bool } @raygen fn bad(#[shader_record] r: Record) {}",
            "#[layout(std140)] struct Inner { value: f32 } struct Record { inner: Inner } @raygen fn bad(#[shader_record] r: Record) {}",
            "@raygen fn bad(#[ray_payload(0)] p: vec4, #[binding(set=0,binding=0)] scene: AccelerationStructure) { { let p = vec4(0.0); trace_ray(scene, 0u, 255u, 0u, 0u, 0u, vec3(0.0), 0.01, vec3(0.0), 32.0, p); } }",
        ] {
            assert!(selected_glsl(source, "bad").is_err(), "{source}");
        }
        let glsl = selected_glsl(
            r"
            struct Record { scale: f32, direction: vec3, weights: [f32; 2] }
            @raygen fn record_default(#[shader_record] record: Record) {
                let value = record.direction;
            }
        ",
            "record_default",
        )
        .unwrap();
        assert!(
            glsl.contains("layout(shaderRecordEXT, std430) buffer"),
            "{glsl}"
        );
        validate_glsl_stage("record_default", &glsl, "rgen", false);
    }

    #[test]
    fn writes_graphics_outputs_before_early_return() {
        let glsl = selected_glsl(
            r"
            @stage(vertex) fn vertex_main(#[builtin(position)] out position: vec4) {
                position = vec4(1.0);
                if position.x > 0.0 { return; }
                position = vec4(0.0);
            }
        ",
            "vertex_main",
        )
        .unwrap();
        let early_return = glsl.find("return;").unwrap();
        assert!(
            glsl[..early_return].contains("gl_Position = position;"),
            "{glsl}"
        );
        validate_glsl_stage("early_return", &glsl, "vert", false);
    }

    #[test]
    fn reserves_explicit_bindings_before_automatic_and_debug_resources() {
        let glsl = selected_glsl(
            r#"
            @stage(compute(1,1,1)) fn bindings(
                automatic: RWBuffer<u32>,
                #[binding(set=0,binding=0)] explicit_zero: RWBuffer<u32>,
                #[binding(set=0,binding=2)] explicit_two: RWBuffer<u32>,
                #[pack_resource(7)] texture: Texture2D<f32>,
            ) {
                automatic[0u] = explicit_zero[0u] + explicit_two[0u];
                mc::debug_write("diagnostic", ivec2(0,0), vec4(1.0));
            }
        "#,
            "bindings",
        )
        .unwrap();
        assert!(
            glsl.contains("layout(set = 0, binding = 1, scalar) buffer VslBuf_automatic"),
            "{glsl}"
        );
        assert!(glsl.contains("layout(set = 0, binding = 3, rgba16f) writeonly uniform image2D vsl_debug_diagnostic"), "{glsl}");
        assert!(
            glsl.contains("layout(set = 3, binding = 7) uniform texture2D texture;"),
            "{glsl}"
        );
        validate_glsl_if_available("binding_plan", &glsl, false);
        assert!(selected_glsl("@stage(compute(1,1,1)) fn bad(#[binding(set=0,binding=0)] a: RWBuffer<u32>, #[binding(set=0,binding=0)] b: RWBuffer<u32>) {}", "bad").is_err());
    }

    #[test]
    fn evaluates_range_bounds_once_and_preserves_array_dimension_order() {
        let glsl = selected_glsl(
            r"
            @stage(compute(1,1,1)) fn ranges(#[binding(set=0,binding=0)] result: RWBuffer<u32>) {
                let mut limit = 4u;
                let mut stride = 1u;
                let mut sum = 0u;
                for i in (0u..limit).step_by(stride) {
                    limit = 0u;
                    stride = 4u;
                    sum = sum + i;
                }
                let values: [[u32; 3]; 2];
                values[1u][2u] = sum;
                let row = values[1u];
                result[0u] = row[2u];
            }
        ",
            "ranges",
        )
        .unwrap();
        assert!(glsl.contains("uint values[2][3];"), "{glsl}");
        assert!(glsl.contains("uint[3] row = values[1u];"), "{glsl}");
        let loop_start = glsl.find("for (uint i").unwrap();
        assert!(glsl[..loop_start].contains("uint(limit)"), "{glsl}");
        assert!(glsl[..loop_start].contains("uint(stride)"), "{glsl}");
        validate_glsl_if_available("range_array", &glsl, false);
    }

    #[test]
    fn emits_trait_and_generic_template_metadata_in_standalone_restir() {
        let parsed = parse_source(FileId(0), RESTIR_VSL);
        assert!(
            !parsed.has_errors(),
            "unexpected parse errors: {:?}",
            parsed.diagnostics
        );

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(!glsl.contains("VslDefaultTarget"), "got:\n{glsl}");
        assert!(glsl.contains("// trait ResampleTarget"));
        assert!(glsl.contains(
            "// generic struct Reservoir<S: ResampleTarget> (no concrete instantiations in this unit)"
        ));
        assert!(glsl.contains(
            "// generic impl<S: ResampleTarget> Reservoir<S> (no concrete instantiations in this unit)"
        ));
        assert!(!glsl.contains("struct Reservoir {"));
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
        assert!(glsl.contains("layout(set = 0, binding = 0, std140) uniform WorldUBO"));
        assert!(glsl.contains("} pc;"));
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
        assert!(glsl.contains("layout(set = 0, binding = 0, std140) uniform WorldUBO"));
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
        assert!(vd.contains("struct VsPackParams"));
        assert!(vd.contains("layout(set = 0, binding = 15, std430) readonly buffer VsMotionBlock"));
        assert!(!vd.contains("binding = 0"));
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
        assert!(!restir_glsl.contains("VslDefaultTarget"));

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
        assert!(glsl.contains("texelFetch(tex_a, coord, 0)"));
        assert!(!glsl.contains("vsl_default_sampler"));
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

    #[test]
    fn monomorphizes_generic_struct_and_impl_methods_without_default_target() {
        let src = r"
            pub trait ResampleTarget {
                fn target_pdf(self, world_pos: vec3, normal: vec3) -> f32;
            }

            #[layout(std430)]
            pub struct LightSample {
                pub pos: vec3,
                pub radiance: vec3,
            }

            impl ResampleTarget for LightSample {
                pub fn target_pdf(self, world_pos: vec3, normal: vec3) -> f32 {
                    let d = normalize(self.pos - world_pos);
                    max(dot(d, normal), 0.0) * length(self.radiance)
                }
            }

            #[layout(std430)]
            pub struct Reservoir<S: ResampleTarget> {
                pub sample: S,
                pub w_sum: f32,
                pub m: u32,
                pub w: f32,
            }

            impl<S: ResampleTarget> Reservoir<S> {
                pub fn update(inout self, candidate: S, weight: f32, rng_u: f32) -> bool {
                    self.w_sum += weight;
                    self.m += 1u;
                    if rng_u * self.w_sum < weight {
                        self.sample = candidate;
                        return true;
                    }
                    return false;
                }

                pub fn finalize(inout self, world_pos: vec3, normal: vec3) {
                    let p_hat = self.sample.target_pdf(world_pos, normal);
                    self.w = if p_hat > 1e-6 && self.m > 0u {
                        self.w_sum / (f32(self.m) * p_hat)
                    } else {
                        0.0
                    };
                }
            }

            @stage(compute(8, 8, 1))
            pub fn restir_cs(out_img: RWImage2D<rgba16f>) {
                let s = LightSample { pos: vec3(0.0, 10.0, 0.0), radiance: vec3(1.0) };
                let mut res: Reservoir<LightSample> = Reservoir {
                    sample: s,
                    w_sum: 0.0,
                    m: 0u,
                    w: 0.0,
                };
                let _ = res.update(s, 1.0, 0.25);
                res.finalize(vec3(0.0), vec3(0.0, 1.0, 0.0));
                out_img.store(ivec2(0, 0), vec4(res.sample.radiance * res.w, 1.0));
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(!glsl.contains("VslDefaultTarget"), "got:\n{glsl}");
        assert!(!glsl.contains("struct Reservoir {"), "got:\n{glsl}");
        assert!(
            glsl.contains("struct Reservoir__LightSample {"),
            "got:\n{glsl}"
        );
        assert!(glsl.contains("LightSample _vsl_sample;"), "got:\n{glsl}");
        assert!(
            glsl.contains("float LightSample_target_pdf(in LightSample self, in vec3 world_pos, in vec3 normal)"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("bool Reservoir__LightSample_update(inout Reservoir__LightSample self, in LightSample candidate,"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("void Reservoir__LightSample_finalize(inout Reservoir__LightSample self, in vec3 world_pos, in vec3 normal)"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("LightSample_target_pdf(self._vsl_sample, world_pos, normal)"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("Reservoir__LightSample res = Reservoir__LightSample(s, 0.0, 0u, 0.0);"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("Reservoir__LightSample_update(res, s, 1.0, 0.25)"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("Reservoir__LightSample_finalize(res, vec3(0.0), vec3(0.0, 1.0, 0.0));"),
            "got:\n{glsl}"
        );
    }

    #[test]
    fn emits_set2_bindless_combined_sampler_array_without_default_sampler() {
        let src = r"
            import mc::{bindless};

            @stage(compute(8, 8, 1))
            pub fn bindless_only_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let slot = gid.x;
                let uv = vec2(0.5, 0.5);
                let a = bindless[nonuniform(slot)].sample_lod(uv, 0.0);
                let b = bindless[nonuniform(slot)].sample(uv);
                out_img.store(ivec2(gid.xy), a + b);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(
            glsl.contains("layout(set = 2, binding = 0) uniform sampler2D bindless[4096];"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("textureLod(bindless[nonuniformEXT(slot)], uv, 0.0)"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("texture(bindless[nonuniformEXT(slot)], uv)"),
            "got:\n{glsl}"
        );
        assert!(!glsl.contains("vsl_default_sampler"), "got:\n{glsl}");
        assert!(!glsl.contains("binding = 14"), "got:\n{glsl}");
    }

    #[test]
    fn emits_schema_driven_vs_pack_params_in_hook_bridge() {
        let src = r"
            @hook(surface)
            pub fn stylized_surface(inout albedo: vec3) {
                let tint = mc::param(sky_tint);
                let enabled = mc::param(enable_fog);
                let mode = params.fog_mode;
                let strength = params.fog_strength;
                let budget = params.frame_budget;
                if enabled && mode > 0 {
                    albedo = albedo * tint * strength + vec3(f32(budget));
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

        let mut schema = IndexMap::new();
        schema.insert(SmolStr::new_static("sky_tint"), PackParamType::Vec3);
        schema.insert(SmolStr::new_static("enable_fog"), PackParamType::Bool);
        schema.insert(SmolStr::new_static("fog_mode"), PackParamType::Enum);
        schema.insert(SmolStr::new_static("fog_strength"), PackParamType::F32);
        schema.insert(SmolStr::new_static("frame_budget"), PackParamType::U32);

        let glsl = emit_raster_gbuffer_hook_slot_with_schema(HookStage::Surface, func, &schema)
            .expect("surface schema slot");
        assert!(glsl.contains("vec3 sky_tint;"), "got:\n{glsl}");
        assert!(glsl.contains("bool enable_fog;"), "got:\n{glsl}");
        assert!(glsl.contains("int fog_mode;"), "got:\n{glsl}");
        assert!(glsl.contains("float fog_strength;"), "got:\n{glsl}");
        assert!(glsl.contains("uint frame_budget;"), "got:\n{glsl}");
        assert!(
            glsl.contains("vec3 tint = VsMotion.packParams.sky_tint;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("bool enabled = VsMotion.packParams.enable_fog;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("int mode = VsMotion.packParams.fog_mode;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("float strength = VsMotion.packParams.fog_strength;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("uint budget = VsMotion.packParams.frame_budget;"),
            "got:\n{glsl}"
        );
    }

    #[test]
    fn emits_canonical_vsl_mc_abi_layouts_including_sharc_and_restir() {
        let src = r"
            import mc::{world, sharc_config, lights, restir_reservoir};

            @stage(compute(8, 8, 1))
            pub fn abi_check_cs(out_img: RWImage2D<rgba16f>) {
                let p = world.pack_params_bda;
                let cap = mc::sharc_config.capacity;
                let n = mc::lights.count;
                let r = mc::restir_reservoir.count;
                out_img.store(ivec2(0, 0), vec4(f32(p), f32(cap), f32(n), f32(r)));
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(
            glsl.contains("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 0, std140) uniform WorldUBO"),
            "got:\n{glsl}"
        );
        assert!(glsl.contains("uint64_t pack_params_bda;"), "got:\n{glsl}");
        assert!(glsl.contains("float wind_phase;"), "got:\n{glsl}");
        assert!(glsl.contains("uint pack_params_bda_pad;"), "got:\n{glsl}");
        assert!(
            glsl.contains("layout(set = 1, binding = 2, std140) uniform SharcConfigUBO"),
            "got:\n{glsl}"
        );
        assert!(glsl.contains("uint64_t hash_table_bda;"), "got:\n{glsl}");
        assert!(
            glsl.contains("layout(set = 1, binding = 3, std430) readonly buffer EmissiveLightSSBO"),
            "got:\n{glsl}"
        );
        assert!(glsl.contains("vec3 position;"), "got:\n{glsl}");
        assert!(glsl.contains("float radius;"), "got:\n{glsl}");
        assert!(glsl.contains("uint _pad0;"), "got:\n{glsl}");
        assert!(
            glsl.contains("layout(set = 1, binding = 4, std430) buffer RestirReservoirSSBO"),
            "got:\n{glsl}"
        );
        assert!(glsl.contains("vec3 sample_pos;"), "got:\n{glsl}");
        assert!(glsl.contains("float weight_sum;"), "got:\n{glsl}");
    }

    #[test]
    fn computes_layout_aware_std430_and_std140_bda_strides() {
        let src = r"
            #[layout(std430)]
            pub struct PaddedStd430 {
                pub a: f32,
                pub b: vec3,
            }

            #[layout(std140)]
            pub struct ArrayStd140 {
                pub vals: [f32; 2],
            }

            #[layout(std430)]
            pub struct PtrHolder {
                pub p: ptr<u32, align = 4>,
                pub tag: f32,
            }

            @stage(compute(1, 1, 1))
            pub fn stride_cs(
                s_ptr: ptr<PaddedStd430, align = 16>,
                a_ptr: ptr<ArrayStd140, align = 16>,
                h_ptr: ptr<PtrHolder, align = 8>,
                out_buf: RWBuffer<PaddedStd430>,
            ) {
                let s1 = s_ptr.offset(1);
                let a1 = a_ptr.offset(1);
                let h1 = h_ptr.offset(1);
                out_buf[0] = s1.load();
                let _ = a1.load();
                let _ = h1.load();
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(
            glsl.contains(
                "layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer VslPtrRo_PaddedStd430_a16"
            ),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains(
                "layout(buffer_reference, std140, buffer_reference_align = 16) readonly buffer VslPtrRo_ArrayStd140_a16"
            ),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 4, std430) buffer VslBuf_out_buf"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("VslPtrRo_PaddedStd430_a16(uint64_t(s_ptr) + uint64_t(1) * 32ul)"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("VslPtrRo_ArrayStd140_a16(uint64_t(a_ptr) + uint64_t(1) * 32ul)"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("VslPtrRo_PtrHolder_a8(uint64_t(h_ptr) + uint64_t(1) * 16ul)"),
            "got:\n{glsl}"
        );
    }

    #[test]
    fn supports_upstream_unsized_bindless_abi_toggle() {
        let src = r"
            import mc::{bindless};

            @stage(compute(8, 8, 1))
            pub fn unsized_bindless_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                out_img: RWImage2D<rgba16f>,
            ) {
                let slot = gid.x;
                let uv = vec2(0.5, 0.5);
                let c = bindless[nonuniform(slot)].sample_lod(uv, 0.0);
                out_img.store(ivec2(gid.xy), c);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let opts = EmitOptions {
            bindless_abi: BindlessAbi::UpstreamUnsized,
            ..EmitOptions::default()
        };
        let glsl = emit_glsl_460_with_options(&parsed.unit, opts);
        assert!(
            glsl.contains("layout(set = 2, binding = 0) uniform texture2D bindless[];"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 14) uniform sampler vsl_default_sampler;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains(
                "textureLod(sampler2D(bindless[nonuniformEXT(slot)], vsl_default_sampler), uv, 0.0)"
            ),
            "got:\n{glsl}"
        );
    }

    #[test]
    fn emits_pack_resource_in_set3_and_named_push_constant_instance() {
        let src = r"
            #[push_constant, layout(std430)]
            pub struct TonePush {
                pub exposure: f32,
            }

            @stage(compute(8, 8, 1))
            pub fn tonemap_cs(
                #[push_constant] push: TonePush,
                #[pack_resource] lut: Texture2D<f32>,
                in_tex: Texture2D<f32>,
                out_img: RWImage2D<rgba16f>,
            ) {
                let c = in_tex.fetch(ivec2(0, 0), 0);
                let l = lut.fetch(ivec2(0, 0), 0);
                out_img.store(ivec2(0, 0), c * l * push.exposure);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let glsl = emit_glsl_460(&parsed.unit);
        assert!(
            glsl.contains("layout(push_constant, std430) uniform TonePush"),
            "got:\n{glsl}"
        );
        assert!(glsl.contains("} push;"), "got:\n{glsl}");
        assert!(
            glsl.contains("layout(set = 3, binding = 0) uniform texture2D lut;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 1) uniform texture2D in_tex;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 2, rgba16f) uniform image2D out_img;"),
            "got:\n{glsl}"
        );
        assert!(!glsl.contains("vsl_default_sampler"), "got:\n{glsl}");
    }

    #[test]
    fn emits_vertex_and_fragment_stages_with_locations_and_graphics_builtins() {
        let vs_src = r"
            @stage(vertex)
            pub fn main_vs(
                #[builtin(vertex_index)] vid: i32,
                #[builtin(instance_index)] iid: i32,
                #[location(0)] in_pos: vec3,
                #[location(0)] out v_uv: vec2,
                #[builtin(position)] out clip_pos: vec4,
            ) {
                v_uv = vec2(f32(vid), f32(iid));
                clip_pos = vec4(in_pos, 1.0);
            }
        ";
        let vs_parsed = parse_source(FileId(0), vs_src);
        assert!(!vs_parsed.has_errors(), "{:?}", vs_parsed.diagnostics);
        let vs_glsl = emit_glsl_460(&vs_parsed.unit);
        assert!(
            vs_glsl.contains("layout(location = 0) in vec3 in_pos;"),
            "got:\n{vs_glsl}"
        );
        assert!(
            vs_glsl.contains("layout(location = 0) out vec2 v_uv;"),
            "got:\n{vs_glsl}"
        );
        assert!(vs_glsl.contains("void main()"), "got:\n{vs_glsl}");
        assert!(
            vs_glsl.contains("int vid = gl_VertexIndex;"),
            "got:\n{vs_glsl}"
        );
        assert!(
            vs_glsl.contains("int iid = gl_InstanceIndex;"),
            "got:\n{vs_glsl}"
        );
        assert!(
            vs_glsl.contains("gl_Position = clip_pos;"),
            "got:\n{vs_glsl}"
        );

        let fs_src = r"
            @stage(fragment)
            pub fn main_fs(
                #[builtin(frag_coord)] fc: vec4,
                #[builtin(front_facing)] ff: bool,
                #[builtin(point_coord)] pc: vec2,
                #[location(0)] v_uv: vec2,
                #[location(0)] out frag_color: vec4,
            ) {
                let s = if ff { 1.0 } else { 0.5 };
                frag_color = vec4(v_uv + pc, fc.z * s, 1.0);
            }
        ";
        let fs_parsed = parse_source(FileId(1), fs_src);
        assert!(!fs_parsed.has_errors(), "{:?}", fs_parsed.diagnostics);
        let fs_glsl = emit_glsl_460(&fs_parsed.unit);
        assert!(
            fs_glsl.contains("layout(location = 0) in vec2 v_uv;"),
            "got:\n{fs_glsl}"
        );
        assert!(
            fs_glsl.contains("layout(location = 0) out vec4 frag_color;"),
            "got:\n{fs_glsl}"
        );
        assert!(fs_glsl.contains("void main()"), "got:\n{fs_glsl}");
        assert!(
            fs_glsl.contains("vec4 fc = gl_FragCoord;"),
            "got:\n{fs_glsl}"
        );
        assert!(
            fs_glsl.contains("bool ff = gl_FrontFacing;"),
            "got:\n{fs_glsl}"
        );
        assert!(
            fs_glsl.contains("vec2 pc = gl_PointCoord;"),
            "got:\n{fs_glsl}"
        );
    }

    #[test]
    fn emits_ordered_schema_and_transitive_helpers_in_hook_bridge() {
        let src = r"
            const SCALE: f32 = 2.0;

            struct TintData {
                pub weight: f32,
            }

            fn compute_tint(w: f32) -> TintData {
                TintData { weight: w * SCALE }
            }

            @hook(surface)
            pub fn sample(inout albedo: vec3) {
                let td = compute_tint(mc::param(second_param));
                albedo = albedo * td.weight;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let func = parsed
            .unit
            .items
            .iter()
            .find_map(|item| match item {
                Item::Function(f) if f.name.name == "sample" => Some(f),
                _ => None,
            })
            .expect("hook function");

        let mut schema = IndexMap::new();
        schema.insert(SmolStr::new_static("first_param"), PackParamType::Vec3);
        schema.insert(SmolStr::new_static("second_param"), PackParamType::F32);

        let glsl = emit_raster_gbuffer_hook_slot_from_unit(
            HookStage::Surface,
            &parsed.unit,
            func,
            &schema,
        )
        .expect("hook slot from unit");

        let first_idx = glsl
            .find("vec3 first_param;")
            .expect("first_param in schema");
        let second_idx = glsl
            .find("float second_param;")
            .expect("second_param in schema");
        assert!(
            first_idx < second_idx,
            "schema field order must be preserved:\n{glsl}"
        );
        assert!(glsl.contains("const float SCALE = 2.0;"), "got:\n{glsl}");
        assert!(glsl.contains("struct TintData"), "got:\n{glsl}");
        assert!(glsl.contains("TintData compute_tint("), "got:\n{glsl}");
        assert!(glsl.contains("void _vsl_sample("), "got:\n{glsl}");
        assert!(glsl.contains("_vsl_sample(albedo);"), "got:\n{glsl}");
        assert!(
            glsl.contains("layout(set = 0, binding = 15, std430) readonly buffer VsMotionBlock"),
            "got:\n{glsl}"
        );
    }

    #[test]
    fn standalone_abi_omits_world_ubo_and_uses_explicit_bindings() {
        let src = r"
            module standalone::glsl;

            #[stage(compute(8, 8, 1))]
            pub fn main_cs(
                #[builtin(global_invocation_id)] gid: uvec3,
                #[binding(set = 1, binding = 3)] in_tex: Texture2D<f32>,
                #[binding(set = 1, binding = 4)] samp: Sampler,
                out_img: RWImage2D<rgba16f>,
            ) {
                let c = in_tex.sample_lod(samp, vec2(gid.xy), 0.0);
                out_img.store(ivec2(gid.xy), c);
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let opts = EmitOptions {
            abi: vsl_mc::AbiProfile::Standalone,
            bindless_abi: BindlessAbi::UpstreamUnsized,
            ..EmitOptions::default()
        };
        let glsl = emit_glsl_460_with_options(&parsed.unit, opts);
        assert!(!glsl.contains("WorldUBO"), "got:\n{glsl}");
        assert!(
            glsl.contains("layout(set = 1, binding = 3) uniform texture2D in_tex;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 1, binding = 4) uniform sampler samp;"),
            "got:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(set = 0, binding = 0, rgba16f) uniform image2D out_img;"),
            "got:\n{glsl}"
        );
    }
}
