#!/bin/sh
# Linux port of with_comfy_env.bat: run a command (default: an interactive
# shell) inside the ComfyUI virtualenv, starting in the poi-nodes folder.
#
# ComfyUI root: $COMFYUI_DIR, else three levels up from this file (the layout
# ComfyUI/custom_nodes/poi-nodes/tests/ used on Windows), else ~/src/ComfyUI.
# The venv is <root>/.venv (uv) or <root>/venv.
#
# Usage: tests/with_comfy_env.sh [command [args...]]
set -eu

here=$(cd "$(dirname "$0")" && pwd)
nodes_dir=$(dirname "$here")

if [ -n "${COMFYUI_DIR:-}" ]; then
    root=$COMFYUI_DIR
elif [ -d "$here/../../../.venv" ] || [ -d "$here/../../../venv" ]; then
    root=$(cd "$here/../../.." && pwd)
else
    root=$HOME/src/ComfyUI
fi

if [ -f "$root/.venv/bin/activate" ]; then
    venv=$root/.venv
elif [ -f "$root/venv/bin/activate" ]; then
    venv=$root/venv
else
    echo "Error: no virtualenv under $root (set COMFYUI_DIR, create it with: uv venv)" >&2
    exit 1
fi

echo "Activating ComfyUI virtual environment: $venv" >&2
# shellcheck disable=SC1091
. "$venv/bin/activate"
cd "$nodes_dir"
if [ "$#" -eq 0 ]; then
    echo "Starting a shell in $nodes_dir" >&2
    exec "${SHELL:-/bin/sh}"
fi
exec "$@"
