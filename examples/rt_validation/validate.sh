#!/usr/bin/env bash
# Compile standalone RT fixtures through native SPIR-V and selected-entry GLSL.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
if [[ -n ${1:-} ]]; then
    mkdir -p -- "$1"
    out=$(cd -- "$1" && pwd)
else
    out=$(mktemp -d "${TMPDIR:-/tmp}/vsl-rt-validation.XXXXXX")
fi
if [[ -z ${VSL_CLI:-} ]]; then
    cargo build -p vsl-cli --bin vslc
fi
vslc=${VSL_CLI:-target/debug/vslc}
validator=${SPIRV_VAL:-spirv-val}
glslang=${GLSLANG_VALIDATOR:-glslangValidator}
count=0
for tool in "$vslc" "$validator" "$glslang" python3; do
    command -v "$tool" >/dev/null || { printf "Missing required tool: %s\n" "$tool" >&2; exit 1; }
done
validate_module() {
    local name=$1
    shift
    local source="examples/rt_validation/$name.vsl"
    local dir="$out/$name"
    mkdir -p "$dir"
    "$vslc" check "$source" --abi standalone "$@"
    "$vslc" emit-spv "$source" --abi standalone "$@" -o "$dir"
    "$vslc" reflect "$source" --abi standalone "$@" > "$dir/reflection.json"
    # Validate only the entry points in the newly produced reflection; stale
    # binaries in a reused output directory must not count as current results.
    python3 - "$dir/reflection.json" <<'PY' > "$dir/entries.txt"
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    units = json.load(source)
if not units:
    raise SystemExit("No active entry points")
for unit in units:
    stages = {"compute": "comp", "ray_gen": "rgen", "closest_hit": "rchit", "any_hit": "rahit", "miss": "rmiss", "intersection": "rint", "callable": "rcall"}
    print(unit["entry_point"], stages[unit["stage"]], sep="\t")
PY
    local entry stage env
    while IFS=$'\t' read -r entry stage; do
        "$vslc" emit-glsl "$source" --abi standalone --entry "$entry" -o "$dir/$entry.glsl"
        for env in vulkan1.3 vulkan1.4; do
            "$validator" --target-env "$env" "$dir/$entry.spv"
            "$glslang" -V --target-env "$env" -S "$stage" "$dir/$entry.glsl" -o "$dir/$entry.glsl.$env.spv"
            "$validator" --target-env "$env" "$dir/$entry.glsl.$env.spv"
        done
        printf 'Validated %s/%s (native and GLSL; Vulkan 1.3/1.4)\n' "$name" "$entry"
        count=$((count + 1))
    done < "$dir/entries.txt"
}
validate_module ray_query --tier 2 --cap ray_query
validate_module pipeline --tier 3
validate_module callable --tier 3
validate_module shader_record --tier 3
validate_module ser_nv --tier 3 --cap ser_nv
validate_module ser_ext --tier 3 --cap ser_ext
validate_module omm_khr --tier 2 --cap ray_query --cap omm_khr --cap position_fetch
if [[ $count -ne 10 ]]; then
    printf 'Expected 10 entry points, validated %s\n' "$count" >&2
    exit 1
fi
printf 'Validated %s entry points. Artifacts: %s\n' "$count" "$out"
