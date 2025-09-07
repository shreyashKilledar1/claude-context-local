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
IS_UPDATE=0
if [[ -d "${PROJECT_DIR}/.git" ]]; then
  print "Found existing installation at ${PROJECT_DIR}"
  IS_UPDATE=1
  
  # Check if there are uncommitted changes
  if ! git -C "${PROJECT_DIR}" diff-index --quiet HEAD -- 2>/dev/null; then
    print "WARNING: You have uncommitted changes in ${PROJECT_DIR}"
    
    # When piped (curl | bash), auto-select update
    if [ -t 0 ]; then
      # Interactive terminal - ask user
      printf "Options:\n  [u] Update anyway (stash changes)\n  [k] Keep current version\n  [d] Delete and reinstall\nChoice [u/k/d]: "
      read -r choice
    else
      # Piped input - default to update
      print "Auto-selecting: Update anyway (stash changes)"
      choice="u"
    fi
    case "${choice}" in
      k|K) print "Keeping current installation. Skipping git update."; SKIP_UPDATE=1 ;;
      d|D) 
        print "Removing ${PROJECT_DIR} for clean reinstall..."
        rm -rf "${PROJECT_DIR}"
        git clone "${REPO_URL}" "${PROJECT_DIR}"
        IS_UPDATE=0  # Treat as fresh install
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
  IS_UPDATE=0
fi

# 4) Install Python dependencies
if [[ "${SKIP_UPDATE:-0}" != "1" ]]; then
  print "Installing Python dependencies with uv"
  (cd "${PROJECT_DIR}" && uv sync)
else
  print "Skipping dependency update (keeping current version)"
fi

# 5) Prefer FAISS GPU wheels on NVIDIA machines (skip in CI/piped mode)
if [ -t 0 ] && [[ "${SKIP_GPU:-0}" != "1" ]]; then
  print "Checking for NVIDIA GPU to install FAISS GPU wheels (optional)"
  (
    set +e
    GPU_DETECTED=0
    if command -v nvidia-smi >/dev/null 2>&1; then
      GPU_DETECTED=1
    fi
    if [[ "${GPU_DETECTED}" -eq 1 ]]; then
      echo "NVIDIA GPU detected. Attempting to install faiss-gpu wheels (30s timeout)..."
      # Try CUDA 12 wheels first, then CUDA 11, fallback to CPU with timeout
      if timeout 30 bash -c "cd '${PROJECT_DIR}' && uv add faiss-gpu-cu12" >/dev/null 2>&1; then
        echo "Installed faiss-gpu-cu12"
        (cd "${PROJECT_DIR}" && uv remove faiss-cpu) >/dev/null 2>&1 || true
      elif timeout 30 bash -c "cd '${PROJECT_DIR}' && uv add faiss-gpu-cu11" >/dev/null 2>&1; then
        echo "Installed faiss-gpu-cu11"
        (cd "${PROJECT_DIR}" && uv remove faiss-cpu) >/dev/null 2>&1 || true
      else
        echo "Could not install faiss-gpu wheels (timeout or failed). Keeping CPU build."
      fi
    else
      echo "No NVIDIA GPU detected. Using faiss-cpu (default)."
    fi
    set -e
  )
else
  print "Skipping GPU detection (piped mode or SKIP_GPU=1). Using faiss-cpu."
fi

# 6) Download model to storage dir
print "Downloading embedding model to ${STORAGE_DIR}"
mkdir -p "${STORAGE_DIR}"
(cd "${PROJECT_DIR}" && uv run scripts/download_model_standalone.py --storage-dir "${STORAGE_DIR}" --model "${MODEL_NAME}" -v)

# Colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

if [[ "${IS_UPDATE}" -eq 1 ]]; then
  hr; printf "${GREEN}${BOLD}✅ Update complete!${NC}\n"; hr
  
  printf "${BLUE}📍 Locations:${NC}\n"
  printf "  Project: ${PROJECT_DIR}\n"
  printf "  Storage: ${STORAGE_DIR} ${GREEN}(embeddings preserved ✓)${NC}\n\n"
  
  printf "${RED}${BOLD}🔄 IMPORTANT: Re-register MCP server after updates${NC}\n"
  printf "${YELLOW}Run these commands:${NC}\n\n"
  
  printf "${BOLD}1) Remove old server:${NC}\n"
  printf "   ${BLUE}claude mcp remove code-search${NC}\n\n"
  
  printf "${BOLD}2) Add updated server:${NC}\n"
  printf "   ${BLUE}claude mcp add code-search --scope user -- uv run --directory ${PROJECT_DIR} python mcp_server/server.py${NC}\n\n"
  
  printf "${BOLD}3) Verify connection:${NC}\n"
  printf "   ${BLUE}claude mcp list${NC}\n"
  printf "   ${GREEN}Look for: code-search ... ✓ Connected${NC}\n\n"
  
  printf "${BOLD}4) Then in Claude Code:${NC}\n"
  printf "   ${BLUE}index this codebase${NC}\n\n"
  
  printf "${YELLOW}💡 Notes:${NC}\n"
  printf "- Your embeddings and indexed projects are preserved\n"
  printf "- Only the code was updated; your data remains intact\n"
else
  hr; printf "${GREEN}${BOLD}✅ Install complete!${NC}\n"; hr
  
  printf "${BLUE}📍 Locations:${NC}\n"
  printf "  Project: ${PROJECT_DIR}\n"
  printf "  Storage: ${STORAGE_DIR}\n\n"
  
  printf "${BOLD}Next steps:${NC}\n\n"
  
  printf "${BOLD}1) Add MCP server to Claude Code:${NC}\n"
  printf "   ${BLUE}claude mcp add code-search --scope user -- uv run --directory ${PROJECT_DIR} python mcp_server/server.py${NC}\n\n"
  
  printf "${BOLD}2) Verify connection:${NC}\n"
  printf "   ${BLUE}claude mcp list${NC}\n"
  printf "   ${GREEN}Look for: code-search ... ✓ Connected${NC}\n\n"
  
  printf "${BOLD}3) Then in Claude Code:${NC}\n"
  printf "   ${BLUE}index this codebase${NC}\n\n"
  
  printf "${YELLOW}💡 Notes:${NC}\n"
  printf "- To update later, re-run this installer\n"
  printf "- Your embeddings will be stored in ${STORAGE_DIR}\n"
fi


