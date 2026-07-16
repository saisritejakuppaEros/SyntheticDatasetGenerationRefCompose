#!/usr/bin/env bash
# Build DIFT scale+rotation canvases for v2 synthetic dataset.
#
# Usage:
#   GPU_ID=3 bash run_canvas_dift.sh
#   GPU_ID=3 LIMIT=10 bash run_canvas_dift.sh
#   GPU_ID=3 THEME=beach_day_007000 bash run_canvas_dift.sh
set -euo pipefail

ROOT="/mnt/data0/teja/research_multiref/synthetic_dataset_generation/v2"
CONDA_ROOT="${CONDA_ROOT:-/mnt/data0/anaconda_dir/miniconda3}"
DIFT_PY="${DIFT_PY:-${CONDA_ROOT}/envs/pixart_parth/bin/python}"
GPU_ID="${GPU_ID:-3}"
LIMIT="${LIMIT:-0}"
THEME="${THEME:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

export HF_HOME="${HF_HOME:-/mnt/data0/parth/hf_models_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFT_SD_MODEL="${DIFT_SD_MODEL:-/tmp/sd15_dift_local}"

# torch CUDA runtime libs shipped via nvidia-* pip wheels
NVRTC_LIB="${CONDA_ROOT}/envs/pixart_parth/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
CUDNN_LIB="${CONDA_ROOT}/envs/pixart_parth/lib/python3.10/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB="${CONDA_ROOT}/envs/pixart_parth/lib/python3.10/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="${NVRTC_LIB}:${CUDNN_LIB}:${CUBLAS_LIB}:${LD_LIBRARY_PATH:-}"

cd "${ROOT}"
args=(--gpu_id "${GPU_ID}" --sd_model "${DIFT_SD_MODEL}")
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip_existing)
else
  args+=(--no_skip_existing)
fi
if [[ -n "${THEME}" ]]; then
  args+=(--theme "${THEME}")
fi
if [[ "${LIMIT}" != "0" ]]; then
  args+=(--limit "${LIMIT}")
fi
args+=(--save_debug)

mkdir -p "${ROOT}/outputs/canvas_dift" "${ROOT}/logs"
LOG="${ROOT}/logs/canvas_dift_$(date -u +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${DIFT_PY}" -u canvas_dift_generation.py "${args[@]}" 2>&1 | tee "${LOG}"
