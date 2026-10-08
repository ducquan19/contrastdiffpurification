# Sourced by every script in experiments/slurm/: storage locations and the conda environment.
#
# Server rule: code, data, environments and caches (including hidden ones such as .cache) live
# under /media/<user> or /media02/<user>, never under /home/<user>.
#
# Override before sourcing if needed:
#   CDP_WORK  storage root           (default: /media/$USER; use /media02/$USER if that is yours)
#   CDP_ENV   conda environment path (default: $CDP_WORK/envs/cdp)

: "${CDP_WORK:=/media/$USER}"
: "${CDP_ENV:=$CDP_WORK/envs/cdp}"

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
export CONDA_PKGS_DIRS="$CDP_WORK/.conda/pkgs"
export PYTHONUNBUFFERED=1

cdp_conda_init() {
  # batch shells are non-interactive, so load conda's shell function explicitly;
  # .bashrc and conda's hooks often read unset variables, so relax `set -u` meanwhile
  local had_nounset=0
  [[ $- == *u* ]] && had_nounset=1
  set +u
  if ! command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
  fi
  eval "$(conda shell.bash hook)"
  (( had_nounset )) && set -u
  return 0
}

cdp_activate() {
  cdp_conda_init
  local had_nounset=0
  [[ $- == *u* ]] && had_nounset=1
  set +u
  conda activate "$CDP_ENV"
  (( had_nounset )) && set -u
  return 0
}

cdp_module_purge() {
  # `module` is a shell function that batch shells may not define
  if type module >/dev/null 2>&1; then module purge; fi
}
