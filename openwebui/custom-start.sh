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

exec bash start.sh
