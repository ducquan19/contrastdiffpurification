# Sourced by every script in experiments/slurm/: storage locations and the Python environment.
#
# Server rule: code, data, environments and caches (including hidden ones such as .cache) live
# under /media/<user> or /media02/<user>, never under /home/<user>.
#
# Override before sourcing if needed:
#   CDP_WORK     storage root               (default: /media/$USER; use /media02/$USER if that is yours)
#   CDP_ENV      virtual environment path   (default: $CDP_WORK/envs/cdp)
#   CDP_MODULES  environment modules to load after `module purge`, e.g. the module that provides
#                the Python the venv was created from (default: none, i.e. the system Python)

: "${CDP_WORK:=/media/$USER}"
: "${CDP_ENV:=$CDP_WORK/envs/cdp}"
: "${CDP_MODULES:=}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$REPO_ROOT" in
  /media/*|/media02/*) ;;
  *) echo "error: the repository must live under /media/<user> or /media02/<user> (it is in $REPO_ROOT)" >&2
     return 1 2>/dev/null || exit 1 ;;
esac

# data/, pretrained/ and outputs/ default to folders inside the repository, which is already on
# /media; only the caches that would otherwise go to /home need redirecting.
export XDG_CACHE_HOME="$CDP_WORK/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONUNBUFFERED=1

cdp_module_purge() {
  # `module` is a shell function that batch shells may not define
  if type module >/dev/null 2>&1; then
    module purge
    if [ -n "$CDP_MODULES" ]; then
      # shellcheck disable=SC2086
      module load $CDP_MODULES
    fi
  fi
}

cdp_activate() {
  if [ ! -f "$CDP_ENV/bin/activate" ]; then
    echo "error: no virtual environment at $CDP_ENV; run experiments/slurm/setup_env.sh first" >&2
    return 1
  fi
  # activate scripts of older Pythons read unset variables, so relax `set -u` meanwhile
  local had_nounset=0
  [[ $- == *u* ]] && had_nounset=1
  set +u
  # shellcheck disable=SC1091
  source "$CDP_ENV/bin/activate"
  (( had_nounset )) && set -u
  return 0
}
