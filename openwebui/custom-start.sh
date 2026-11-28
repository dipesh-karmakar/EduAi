#!/bin/bash
set -eu

env_file=/app/backend/open_webui/env.py

if grep -Fq "    WEBUI_NAME += ' (Open WebUI)'" "$env_file"; then
    sed -i "/^if WEBUI_NAME != 'Open WebUI':$/,+1d" "$env_file"
elif grep -Fq "WEBUI_NAME += ' (Open WebUI)'" "$env_file"; then
    echo "Unexpected WEBUI_NAME suffix implementation in $env_file" >&2
    exit 1
fi

nodes_dir=/app/build/_app/immutable/nodes
logo_bundle=$(grep -rl --include='*.js' -E 'id="logo" crossorigin="anonymous" class=" w-(6 rounded-full|36|72)"' "$nodes_dir" || true)

if [ -z "$logo_bundle" ] || [ "$(printf '%s\n' "$logo_bundle" | wc -l)" -ne 1 ]; then
    echo "Could not find the expected auth logo element in $nodes_dir" >&2
    exit 1
fi

sed -i \
    -e 's/class=" w-6 rounded-full"/class=" w-72"/' \
    -e 's/class=" w-36"/class=" w-72"/' \
    "$logo_bundle"

if ! grep -R -q -F '.w-72{width:' /app/build/_app/immutable/assets; then
    echo "Could not find the 288px width utility in the frontend CSS" >&2
    exit 1
fi

chunks_dir=/app/build/_app/immutable/chunks
title_bundle=$(grep -rl --include='*.js' -F 'text-2xl @sm:text-2xl line-clamp-1 flex items-center' "$chunks_dir" || true)

if [ -z "$title_bundle" ] || [ "$(printf '%s\n' "$title_bundle" | wc -l)" -ne 1 ]; then
    echo "Could not find the expected chat placeholder bundle in $chunks_dir" >&2
    exit 1
fi

python3 - "$title_bundle" <<'PY'
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
content = bundle.read_text()
model_name_binding = 'T(yr,e(Qe))'
product_name = 'T(yr,"Pablo1.9.8.2")'

if content.count(model_name_binding) == 1:
    bundle.write_text(content.replace(model_name_binding, product_name, 1))
elif product_name not in content:
    raise SystemExit("Could not find the expected model heading text binding")
PY

response_name_binding='v(X,e(Te)?ia(e(Te),r().language):e(n).modelName||e(n).model)'
response_product_binding='v(X,"Pablo1.9.8.2")'
response_bundle=$(
    {
        grep -rl --include='*.js' -F "$response_name_binding" "$chunks_dir" || true
        grep -rl --include='*.js' -F "$response_product_binding" "$chunks_dir" || true
    } | sort -u
)

if [ -z "$response_bundle" ] || [ "$(printf '%s\n' "$response_bundle" | wc -l)" -ne 1 ]; then
    echo "Could not find the expected response model-name binding in $chunks_dir" >&2
    exit 1
fi

python3 - "$response_bundle" <<'PY'
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
content = bundle.read_text()
model_name_binding = 'v(X,e(Te)?ia(e(Te),r().language):e(n).modelName||e(n).model)'
product_name = 'v(X,"Pablo1.9.8.2")'

if content.count(model_name_binding) == 1:
    bundle.write_text(content.replace(model_name_binding, product_name, 1))
elif product_name not in content:
    raise SystemExit("Could not find the expected response model-name binding")
PY

exec bash start.sh
