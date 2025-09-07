#!/usr/bin/env bash
set -euo pipefail

# Remote installer for Claude Code Local Context
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/FarhanAliRaza/claude-context-local/main/scripts/install.sh | bash

REPO_URL="https://github.com/FarhanAliRaza/claude-context-local"
PROJECT_DIR="${HOME}/.local/share/claude-context-local"
STORAGE_DIR="${HOME}/.claude_code_search"
MODEL_NAME="google/embeddinggemma-300m"

print() { printf "%b\n" "$1"; }
hr() { print "\n==================================================\n"; }

hr; print "Installing Claude Context Local"; hr

# 1) Ensure git is available
if ! command -v git >/dev/null 2>&1; then
  print "ERROR: git is required. Please install git and re-run."; exit 1
fi

# 2) Ensure uv is available (prefer uv as per project policy)
if ! command -v uv >/dev/null 2>&1; then
  print "uv not found. Installing uv..."
  # Attempt installing uv
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if ! command -v uv >/dev/null 2>&1; then
    print "ERROR: uv installation failed or not found in PATH."; exit 1
  fi
fi

# 3) Clone or update repository
mkdir -p "${PROJECT_DIR}"
if [[ -d "${PROJECT_DIR}/.git" ]]; then
  print "Found existing installation at ${PROJECT_DIR}"
  
  # Check if there are uncommitted changes
  if ! git -C "${PROJECT_DIR}" diff-index --quiet HEAD -- 2>/dev/null; then
    print "WARNING: You have uncommitted changes in ${PROJECT_DIR}"
    printf "Options:\n  [u] Update anyway (stash changes)\n  [k] Keep current version\n  [d] Delete and reinstall\nChoice [u/k/d]: "
    read -r choice
    case "${choice}" in
      k|K) print "Keeping current installation. Skipping git update."; SKIP_UPDATE=1 ;;
      d|D) 
        print "Removing ${PROJECT_DIR} for clean reinstall..."
        rm -rf "${PROJECT_DIR}"
        git clone "${REPO_URL}" "${PROJECT_DIR}"
        ;;
      u|U|*)
        print "Stashing changes and updating..."
        git -C "${PROJECT_DIR}" stash push -m "Auto-stash before installer update $(date)"
        git -C "${PROJECT_DIR}" remote set-url origin "${REPO_URL}"
        git -C "${PROJECT_DIR}" fetch --tags --prune
        git -C "${PROJECT_DIR}" pull --ff-only
        print "Your changes are stashed. Run 'git stash pop' in ${PROJECT_DIR} to restore them."
        ;;
    esac
  else
    print "Updating repository..."
    git -C "${PROJECT_DIR}" remote set-url origin "${REPO_URL}"
    git -C "${PROJECT_DIR}" fetch --tags --prune
    git -C "${PROJECT_DIR}" pull --ff-only
  fi
else
  print "Cloning ${REPO_URL} to ${PROJECT_DIR}"
  git clone "${REPO_URL}" "${PROJECT_DIR}"
fi

# 4) Install Python dependencies
if [[ "${SKIP_UPDATE:-0}" != "1" ]]; then
  print "Installing Python dependencies with uv"
  (cd "${PROJECT_DIR}" && uv sync)
else
  print "Skipping dependency update (keeping current version)"
fi

# 5) Prefer FAISS GPU wheels on NVIDIA machines
print "Checking for NVIDIA GPU to install FAISS GPU wheels (optional)"
(
  set +e
  GPU_DETECTED=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_DETECTED=1
  fi
  if [[ "${GPU_DETECTED}" -eq 1 ]]; then
    echo "NVIDIA GPU detected. Attempting to install faiss-gpu wheels."
    # Try CUDA 12 wheels first, then CUDA 11, fallback to CPU
    if (cd "${PROJECT_DIR}" && uv add faiss-gpu-cu12) >/dev/null 2>&1; then
      echo "Installed faiss-gpu-cu12"
      (cd "${PROJECT_DIR}" && uv remove faiss-cpu) >/dev/null 2>&1 || true
    elif (cd "${PROJECT_DIR}" && uv add faiss-gpu-cu11) >/dev/null 2>&1; then
      echo "Installed faiss-gpu-cu11"
      (cd "${PROJECT_DIR}" && uv remove faiss-cpu) >/dev/null 2>&1 || true
    else
      echo "Could not install faiss-gpu wheels. Keeping CPU build."
    fi
  else
    echo "No NVIDIA GPU detected. Using faiss-cpu (default)."
  fi
  set -e
)

# 6) Download model to storage dir
print "Downloading embedding model to ${STORAGE_DIR}"
mkdir -p "${STORAGE_DIR}"
(cd "${PROJECT_DIR}" && uv run scripts/download_model_standalone.py --storage-dir "${STORAGE_DIR}" --model "${MODEL_NAME}" -v)

hr; print "Install complete"; hr
cat <<EOF
Project location : ${PROJECT_DIR}
Storage location : ${STORAGE_DIR} (embeddings preserved across updates)

Add MCP server to Claude Code (stdio mode):
  claude mcp add code-search --scope user -- uv run --directory ${PROJECT_DIR} python mcp_server/server.py

Then in Claude Code, run:
  index this codebase for indexing

Notes:
- Embeddings are stored in ${STORAGE_DIR} and preserved across updates
- Only ${PROJECT_DIR} is updated; your indexed projects remain intact
- To update later, re-run this installer
EOF


