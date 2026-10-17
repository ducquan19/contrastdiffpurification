# Sourced by every script in experiments/slurm/: storage locations and the Python environment.
#
# Server rule: code, data, environments and caches (including hidden ones such as .cache) live
# under /media/<user> or /media02/<user>, never under /home/<user>.
#
# Override before sourcing if needed:
#   CDP_WORK     storage root               (default: /media02/$USER if the repository is under
#                                            /media02, else /media/$USER)
#   CDP_ENV      virtual environment path   (default: $CDP_WORK/envs/cdp)
#   CDP_MODULES  environment modules to load after `module purge`, e.g. the module that provides
#                the Python the venv was created from (default: none, i.e. the system Python)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$REPO_ROOT" in
  /media/*|/media02/*) ;;
  *) echo "error: the repository must live under /media/<user> or /media02/<user> (it is in $REPO_ROOT)" >&2
     return 1 2>/dev/null || exit 1 ;;
esac

# default storage root: the /media tree the repository itself lives in
if [ -z "${CDP_WORK:-}" ]; then
  case "$REPO_ROOT" in
    /media02/*) CDP_WORK="/media02/$USER" ;;
    *)          CDP_WORK="/media/$USER" ;;
  esac
fi
: "${CDP_ENV:=$CDP_WORK/envs/cdp}"
: "${CDP_MODULES:=}"
export CDP_WORK CDP_ENV   # Slurm jobs submitted from this shell inherit the same locations

# data/, pretrained/ and outputs/ default to folders inside the repository, which is already on
# /media; only the caches that would otherwise go to /home need redirecting.
export XDG_CACHE_HOME="$CDP_WORK/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export HF_HOME="$XDG_CACHE_HOME/huggingface"

# Some tools ignore XDG_CACHE_HOME and write to ~ directly (gdown: ~/.cache/gdown/cookies.txt,
# used by RobustBench downloads). On accounts whose /home is not writable, point HOME at
# CDP_WORK so those writes land on /media as well.
if ! { mkdir -p "$HOME/.cache" && [ -w "$HOME/.cache" ]; } 2>/dev/null; then
  export CDP_ORIG_HOME="${CDP_ORIG_HOME:-$HOME}"
  export HOME="$CDP_WORK"
fi
mkdir -p "$XDG_CACHE_HOME" "$HOME/.cache/gdown" 2>/dev/null || true
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
