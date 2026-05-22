#!/bin/bash
# Source this file to activate the project python env. Idempotent and safe to
# call from every sbatch script.
#
#   source slurm/setup_env.sh
#
# Strategy (FASRC's module system has no python/3.11 — projects pinned to 3.11
# need uv-managed Python):
#   1. ensure `uv` is on PATH (install to ~/.local/bin if missing)
#   2. `uv python install <version>` (idempotent)
#   3. `uv sync` (creates .venv if needed)
#   4. activate .venv
#
# DO NOT `set -e` here: file is sourced, early exit would kill the caller.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || return 1

# Read python version from pyproject.toml's requires-python (defaults to 3.11).
# Override by exporting PYTHON_VERSION before sourcing.
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# IMPORTANT for FASRC: redirect uv's wheel cache off of $HOME. FASRC home dirs
# are ~95 GB; a single torch+transformers project can push 5+ GB into
# ~/.cache/uv and tip you over the quota mid-install. Default puts the cache
# adjacent to the project on netscratch, shared across sibling projects.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$PROJECT_ROOT")/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
echo "[setup_env] UV_CACHE_DIR=$UV_CACHE_DIR"

# --- 1. uv ------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[setup_env] installing uv to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[[ -d "$HOME/.cargo/bin" ]] && export PATH="$HOME/.cargo/bin:$PATH"
[[ -d "$HOME/.local/bin" ]] && export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo "[setup_env] ERROR: uv install failed; check curl / network" >&2
    return 1
fi
echo "[setup_env] uv: $(uv --version) at $(command -v uv)"

# --- 2. uv-managed Python ---------------------------------------------------
uv python install "$PYTHON_VERSION" || {
    echo "[setup_env] ERROR: uv python install $PYTHON_VERSION failed" >&2
    return 1
}

# --- 3. sync ----------------------------------------------------------------
if ! uv sync --frozen 2>/dev/null; then
    echo "[setup_env] uv sync --frozen failed (lock drift?); trying uv sync"
    uv sync || {
        echo "[setup_env] ERROR: uv sync failed" >&2
        return 1
    }
fi

# --- 4. activate ------------------------------------------------------------
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[setup_env] python: $(python --version) at $(which python)"
