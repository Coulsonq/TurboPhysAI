#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve user-provided relative paths against the directory where the launcher
# was started, before this helper changes into the OpenPI checkout.
OPENPI_LAUNCH_CWD="$(pwd -P)"
openpi_resolve_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${OPENPI_LAUNCH_CWD}" "${path}"
    fi
}

# Source this file from the model launcher; repository paths are derived from this package.
OPENPI_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TURBO_ROOT="$(cd "${OPENPI_MODEL_DIR}/../../../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${TURBO_ROOT}/.." && pwd -P)"
MODEL_REPO="$(cd "$(openpi_resolve_path "${OPENPI_MODEL_REPO:-${WORKSPACE_ROOT}/openpi}")" && pwd -P)"
DATA_ROOT="$(openpi_resolve_path "${OPENPI_DATA_ROOT:-/data/openpi/data/huggingface_vla/libero}")"
WEIGHT_PATH="$(openpi_resolve_path "${OPENPI_WEIGHT_PATH:-/data/ckpt/pi05_base_pytorch}")"
ASSETS_BASE_DIR="$(openpi_resolve_path "${OPENPI_ASSETS_BASE_DIR:-${OPENPI_MODEL_DIR}/assets}")"
OUTPUT_ROOT="$(openpi_resolve_path "${OPENPI_OUTPUT_ROOT:-${WORKSPACE_ROOT}/output}")"
CHECKPOINT_BASE_DIR="$(openpi_resolve_path "${OPENPI_CHECKPOINT_BASE_DIR:-${OUTPUT_ROOT}/checkpoints}")"
LOG_DIR="$(openpi_resolve_path "${OPENPI_LOG_DIR:-${OUTPUT_ROOT}/logs}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUNTIME_ROOT="$(openpi_resolve_path "${OPENPI_RUNTIME_ROOT:-/dev/shm/openpi_8gpu_runtime_${STAMP}_$$}")"
CACHE_ROOT="$(openpi_resolve_path "${OPENPI_CACHE_ROOT:-/dev/shm/openpi_8gpu_cache}")"
LOG_FILE="${LOG_DIR}/pi05_libero_8gpu_bs32_${STAMP}.log"
export OPENPI_DATA_HOME="$(openpi_resolve_path "${OPENPI_DATA_HOME:-${RUNTIME_ROOT}/openpi_data}")"

# Each torchrun rank is bound before Torch imports when openpi.numa is enabled.
unset OPENPI_TURBO_NUMA_PREBOUND

export HF_LEROBOT_HOME="${DATA_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${RUNTIME_ROOT}/pycache"
export PYTHONPATH="${TURBO_ROOT}:${OPENPI_MODEL_DIR}/compat:${MODEL_REPO}:${MODEL_REPO}/src:${MODEL_REPO}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p \
    "${CACHE_ROOT}/hf_datasets" \
    "${CACHE_ROOT}/inductor" \
    "${CACHE_ROOT}/triton" \
    "${RUNTIME_ROOT}/tmp" \
    "${RUNTIME_ROOT}/pycache" \
    "${OPENPI_DATA_HOME}/big_vision" \
    "${CHECKPOINT_BASE_DIR}" \
    "${LOG_DIR}"
install -m 0644 "${OPENPI_MODEL_DIR}/assets/openpi_runtime/big_vision/paligemma_tokenizer.model" \
    "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"

echo "[INFO] Turbo root        = ${TURBO_ROOT}"
echo "[INFO] model repo        = ${MODEL_REPO}"
echo "[INFO] integration       = ${OPENPI_MODEL_DIR}"
echo "[INFO] data              = ${DATA_ROOT}"
echo "[INFO] initial weights   = ${WEIGHT_PATH}"
echo "[INFO] cache/runtime     = ${CACHE_ROOT} / ${RUNTIME_ROOT}"
echo "[INFO] output root       = ${OUTPUT_ROOT}"
echo "[INFO] log               = ${LOG_FILE}"
cd "${MODEL_REPO}"
