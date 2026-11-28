#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
out="${1:-$(mktemp -d "${TMPDIR:-/tmp}/vsl-rt-validation.XXXXXX")}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

command -v spirv-val >/dev/null
cargo build -p vsl-cli --release
compiler="${VSL_COMPILER:-$root/target/release/vslc}"

count=0
for name in ray_query pipeline callable shader_record ser_nv ser_ext omm_khr; do
    source="$root/examples/rt_validation/$name.vsl"
    destination="$out/$name"
    mkdir -p "$destination"
    flags=(--abi standalone --tier 3)
    case "$name" in
        ser_ext) flags+=(--cap ser_ext) ;;
        omm_khr) flags+=(--cap omm_khr --cap position_fetch) ;;
    esac
    "$compiler" check "$source" "${flags[@]}"
    "$compiler" emit-spv "$source" "${flags[@]}" --out "$destination"
    "$compiler" reflect "$source" "${flags[@]}" > "$destination/reflection.json"
    while IFS= read -r -d '' binary; do
        spirv-val --target-env vulkan1.3 "$binary"
        spirv-val --target-env vulkan1.4 "$binary"
        count=$((count + 1))
    done < <(find "$destination" -name '*.spv' -type f -print0)
done
if (( count != 10 )); then
    printf 'Expected 10 SPIR-V modules; found %d\n' "$count" >&2
    exit 1
fi
printf 'Validated %d modules against Vulkan 1.3 and 1.4.\nArtifacts: %s\n' "$count" "$out"
