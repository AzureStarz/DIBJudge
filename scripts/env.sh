#!/usr/bin/env bash
# Shared, public-safe environment bootstrap for DIBJudge shell/SLURM scripts.
# Override paths with environment variables instead of editing scripts:
#   DIBJUDGE_ROOT=/path/to/DIBJudge
#   DIBJUDGE_CONDA_INIT=/path/to/conda.sh
#   DIBJUDGE_CONDA_ENV=dibjudge
#   DIBJUDGE_ENV_FILE=/path/to/private_env.sh
#   DIBJUDGE_MODULES="cuda/12 gcc/11"

DIBJUDGE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIBJUDGE_ROOT="${DIBJUDGE_ROOT:-$(cd "${DIBJUDGE_SCRIPT_DIR}/.." && pwd)}"
cd "${DIBJUDGE_ROOT}"

if [[ -n "${DIBJUDGE_CONDA_INIT:-}" ]]; then
  # shellcheck disable=SC1090
  source "${DIBJUDGE_CONDA_INIT}"
elif command -v conda >/dev/null 2>&1; then
  __conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${__conda_base}" && -f "${__conda_base}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${__conda_base}/etc/profile.d/conda.sh"
  fi
fi

if [[ -n "${DIBJUDGE_CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda activate "${DIBJUDGE_CONDA_ENV}"
  elif [[ -f "${DIBJUDGE_CONDA_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${DIBJUDGE_CONDA_ENV}/bin/activate"
  else
    echo "[warn] DIBJUDGE_CONDA_ENV is set but conda/env activate script was not found: ${DIBJUDGE_CONDA_ENV}" >&2
  fi
fi

if [[ -n "${DIBJUDGE_ENV_FILE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${DIBJUDGE_ENV_FILE}"
fi

if [[ -n "${DIBJUDGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  for module_name in ${DIBJUDGE_MODULES}; do
    module load "${module_name}"
  done
fi
