#!/usr/bin/env bash
# kontexta-cli installer. One line, from anywhere:
#   curl -fsSL https://raw.githubusercontent.com/safiyu/kontexta-cli/main/install.sh | bash
# Add `-s -- --yes` after `bash` to skip the per-dependency questions.
#
# Does: clone (or update) the repo, create a venv, pip install, run `kxtc setup` (Kontexta, LlamaShift),
# then link `kxtc` and `kontexta-cli` into ~/.local/bin.
# Env: KXTC_INSTALL_DIR (default ~/.kontexta-cli/src), KXTC_REPO, KXTC_BRANCH (default main), PYTHON, BIN_DIR.

# Everything lives in a function so bash reads the whole script before running it.
# That lets us hand stdin back to the terminal when the script itself arrived over a pipe.
main() {
  set -euo pipefail

  local repo="${KXTC_REPO:-https://github.com/safiyu/kontexta-cli.git}"
  local branch="${KXTC_BRANCH:-main}"
  local python="${PYTHON:-python3}"
  local bin_dir="${BIN_DIR:-$HOME/.local/bin}"
  local src_dir script_dir=""

  if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi

  command -v "$python" >/dev/null || { echo "Python 3.10+ is required (not found: $python)" >&2; return 1; }
  "$python" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python 3.10+ is required" >&2; return 1; }

  if [ -n "$script_dir" ] && [ -f "$script_dir/pyproject.toml" ]; then
    src_dir="$script_dir"                      # run from a checkout
  else
    command -v git >/dev/null || { echo "git is required to fetch kontexta-cli" >&2; return 1; }
    src_dir="${KXTC_INSTALL_DIR:-$HOME/.kontexta-cli/src}"
    if [ -d "$src_dir/.git" ]; then
      echo "Updating $src_dir"
      git -C "$src_dir" fetch --quiet origin "$branch"
      git -C "$src_dir" checkout --quiet "$branch"
      git -C "$src_dir" merge --ff-only --quiet "origin/$branch"
    else
      echo "Cloning $repo into $src_dir"
      mkdir -p "$(dirname "$src_dir")"
      git clone --quiet --branch "$branch" "$repo" "$src_dir"
    fi
  fi
  cd "$src_dir"

  [ -d .venv ] || "$python" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -e .

  # When piped from curl, stdin is the script. Use the terminal for prompts, or fall back to --yes.
  local setup_args=("$@")
  if [ ! -t 0 ]; then
    if { : < /dev/tty; } 2>/dev/null; then
      exec < /dev/tty
    else
      setup_args+=(--yes)
    fi
  fi
  .venv/bin/kxtc setup "${setup_args[@]}" || echo "Some dependencies need attention; see the report below." >&2

  mkdir -p "$bin_dir"
  ln -sf "$src_dir/.venv/bin/kxtc" "$bin_dir/kxtc"
  ln -sf "$src_dir/.venv/bin/kontexta-cli" "$bin_dir/kontexta-cli"

  echo
  .venv/bin/kxtc doctor || true
  echo
  echo "Installed kxtc and kontexta-cli in $bin_dir"
  case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) echo "Add it to your PATH:  export PATH=\"$bin_dir:\$PATH\"" ;;
  esac
}

main "$@"
