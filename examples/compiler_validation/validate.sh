#!/usr/bin/env bash
# Validate native and supported GLSL paths without running GPU workloads.
set -euo pipefail
ulimit -c 0
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
if [[ -n ${1:-} ]]; then
    mkdir -p -- "$1"
    out=$(mktemp -d "$(cd -- "$1" && pwd)/compiler-validation.XXXXXX")
else
    out=$(mktemp -d "${TMPDIR:-/tmp}/vsl-compiler-validation.XXXXXX")
fi
if [[ -z ${VSL_CLI:-} ]]; then
    cargo build -p vsl-cli --bin vslc
fi
vslc=${VSL_CLI:-target/debug/vslc}
validator=${SPIRV_VAL:-spirv-val}
glslang=${GLSLANG_VALIDATOR:-glslangValidator}
for tool in "$vslc" "$validator" "$glslang" python3; do
    command -v "$tool" >/dev/null || { printf 'Missing required tool: %s\n' "$tool" >&2; exit 1; }
done
native_count=0
glsl_count=0
validate_spv() {
    local file=$1 env
    for env in vulkan1.3 vulkan1.4; do
        "$validator" --target-env "$env" "$file"
    done
}
validate_glsl() {
    local source=$1 entry=$2 stage=$3 abi=$4 dir=$5 env
    "$vslc" emit-glsl "$source" --abi "$abi" --entry "$entry" -o "$dir/$entry.glsl"
    for env in vulkan1.3 vulkan1.4; do
        "$glslang" -V --target-env "$env" -S "$stage" "$dir/$entry.glsl" -o "$dir/$entry.glsl.$env.spv"
        "$validator" --target-env "$env" "$dir/$entry.glsl.$env.spv"
    done
    glsl_count=$((glsl_count + 1))
}
validate_module() {
    local module=$1 expected=$2 glsl=$3
    local source="examples/compiler_validation/$module.vsl"
    local dir="$out/$module"
    mkdir -p "$dir"
    "$vslc" check "$source" --abi standalone --tier 2
    "$vslc" emit-spv "$source" --abi standalone --tier 2 -o "$dir"
    "$vslc" reflect "$source" --abi standalone --tier 2 > "$dir/reflection.json"
    python3 - "$dir" "$expected" <<'PY' > "$dir/entries.txt"
import json
from pathlib import Path
import struct
import sys
root = Path(sys.argv[1])
units = json.loads((root / "reflection.json").read_text())
expected = set(sys.argv[2].split(","))
assert len(units) == len(expected), units
assert {u["entry_point"] for u in units} == expected, units
stages = {"compute": "comp", "vertex": "vert", "fragment": "frag"}
for u in units:
    entry = u["entry_point"]
    words = u["spirv"]
    assert (root / f"{entry}.spv").read_bytes() == struct.pack(f"<{len(words)}I", *words), entry
    print(entry, stages[u["stage"]], sep="\t")
PY
    local entry stage
    while IFS=$'\t' read -r entry stage; do
        validate_spv "$dir/$entry.spv"
        native_count=$((native_count + 1))
        if [[ $glsl == yes || ${STRICT_GLSL:-0} == 1 ]]; then
            validate_glsl "$source" "$entry" "$stage" standalone "$dir"
        else
            printf 'GLSL NOT RUN for %s/%s (free generic function monomorphization is unsupported; STRICT_GLSL=1 to require it)\n' "$module" "$entry"
        fi
    done < "$dir/entries.txt"
}
validate_module generics generic_values no
validate_module mutable mutable_values yes
validate_module resources filter_image yes
validate_module control control_flow yes
validate_module graphics vertex_main,fragment_main yes
validate_module regressions/scalar_inout mutable_values yes
validate_module regressions/aggregate_inout mutable_values yes
validate_module regressions/push_constant_params filter_image yes
validate_module regressions/loop_continue control_flow yes

pack=examples/compiler_validation/pack
"$vslc" pack-eval "$pack/pack.vsl" --tier 2 --json > "$out/pack-default.json"
"$vslc" pack-eval "$pack/pack.vsl" --tier 2 --preset minimal --json > "$out/pack-minimal.json"
"$vslc" pack-eval "$pack/pack.vsl" --tier 2 --preset minimal --param include_copy=true --json > "$out/pack-override.json"
"$vslc" build-pack "$pack" --tier 2 -o "$out/pack"
python3 - "$out" <<'PY' > "$out/pack/entries.txt"
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
def load(name):
    return json.loads((root / name).read_text())
for name in ("pack-default.json", "pack-override.json", "pack/manifest.json"):
    schedule = load(name)["schedule"]
    assert schedule["ordered_passes"] == ["fill", "copy"], schedule
    assert schedule["barriers"] == [{"src_pass": "fill", "dst_pass": "copy", "resource": "source", "kind": "raw"}], schedule
minimal = load("pack-minimal.json")["schedule"]
assert minimal["ordered_passes"] == ["fill"] and not minimal["barriers"], minimal
shaders = load("pack/manifest.json")["shaders"]
assert len(shaders) == 2 and {s["entry_point"] for s in shaders} == {"fill_image", "copy_image"}, shaders
for shader in shaders:
    assert shader["stage"] == "compute"
    print(shader["entry_point"], shader["spv_file"], sep="\t")
PY
while IFS=$'\t' read -r entry file; do
    validate_spv "$out/pack/$file"
    native_count=$((native_count + 1))
    validate_glsl "$pack/shaders.vsl" "$entry" comp hyaloclast "$out/pack"
done < "$out/pack/entries.txt"
if [[ $native_count != 12 ]]; then
    printf 'Expected 12 native entries, validated %s\n' "$native_count" >&2
    exit 1
fi
expected_glsl=11
if [[ ${STRICT_GLSL:-0} == 1 ]]; then
    expected_glsl=12
fi
if [[ $glsl_count != "$expected_glsl" ]]; then
    printf 'Expected %s GLSL entries, validated %s\n' "$expected_glsl" "$glsl_count" >&2
    exit 1
fi
printf 'Validated %s native and %s GLSL entries on both Vulkan environments; pack schedule/presets checked. Artifacts: %s\n' "$native_count" "$glsl_count" "$out"
