# Source this before running anything locally: keeps all caches/temp files on D:
# (this laptop's C: drive is almost full).  Usage:  source scripts/env.sh
export TICKETLENS_CACHE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -W)/.cache"
export TMP="$TICKETLENS_CACHE/tmp" TEMP="$TICKETLENS_CACHE/tmp" TMPDIR="$TICKETLENS_CACHE/tmp"
export PIP_CACHE_DIR="$TICKETLENS_CACHE/pip"
export HF_HOME="$TICKETLENS_CACHE/hf"
export PLAYWRIGHT_BROWSERS_PATH="$TICKETLENS_CACHE/pw"
export PYTHONUTF8=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 TOKENIZERS_PARALLELISM=false
