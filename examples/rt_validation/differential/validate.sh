#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
cd "$root"
out="${1:-$(mktemp -d "${TMPDIR:-/tmp}/vsl-rt-differential.XXXXXX")}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
slang="${SLANGC:-slangc}"
command -v "$slang" >/dev/null
command -v glslangValidator >/dev/null
command -v spirv-val >/dev/null
cargo build -p vsl-cli --release
compiler="${VSL_COMPILER:-$root/target/release/vslc}"

for name in ray_query position_fetch omm_khr omm_ext; do
    flags=(--abi standalone --tier 3)
    case "$name" in
        position_fetch) flags+=(--cap position_fetch) ;;
        omm_khr) flags+=(--cap omm_khr) ;;
        omm_ext) flags+=(--cap omm) ;;
    esac
    "$compiler" emit-spv "$here/$name.vsl" "${flags[@]}" --out "$out/$name.vsl.spv"
    "$compiler" emit-glsl "$here/$name.vsl" --abi standalone --out "$out/$name.comp"
    glslangValidator -V --target-env vulkan1.3 -S comp "$out/$name.comp" -o "$out/$name.glsl.spv"
done
"$slang" "$here/ray_query.slang" -entry main -target spirv -profile spirv_1_6 -o "$out/ray_query.slang.spv"

mkdir -p "$out/pipeline-vsl" "$out/pipeline-slang" "$out/aggregate-pipeline"
"$compiler" emit-spv "$here/pipeline.vsl" --abi standalone --tier 3 --out "$out/pipeline-vsl"
"$compiler" emit-spv "$here/aggregate_pipeline.vsl" --abi standalone --tier 3 --out "$out/aggregate-pipeline"
for entry in rgen rchit rmiss; do
    "$slang" "$here/pipeline.slang" -entry "$entry" -target spirv -profile spirv_1_6 \
        -fvk-use-entrypoint-name -o "$out/pipeline-slang/$entry.spv"
done

# Validate only artifacts produced by this invocation, even in a reused directory.
binaries=()
for name in ray_query position_fetch omm_khr omm_ext; do
    binaries+=("$out/$name.vsl.spv" "$out/$name.glsl.spv")
done
binaries+=("$out/ray_query.slang.spv")
for entry in rgen rchit rmiss; do
    binaries+=("$out/pipeline-vsl/$entry.spv" "$out/pipeline-slang/$entry.spv" "$out/aggregate-pipeline/$entry.spv")
done
for binary in "${binaries[@]}"; do
    spirv-val --target-env vulkan1.3 "$binary"
    spirv-val --target-env vulkan1.4 "$binary"
done
printf 'Validated %d differential SPIR-V modules. Artifacts: %s\n' "${#binaries[@]}" "$out"
