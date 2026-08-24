#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="tests/e2e/accuracy/minimax_h3/test_minimax_h3_i2va_ref2va_similarity_npu.py"
EXPECTED_NPU_COUNT=8
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: ./run_minimax_h3_npu_accuracy.sh [all|i2va|ref2va] [pytest arguments...]

Environment variables:
  ASCEND_RT_VISIBLE_DEVICES                         Visible NPUs; defaults to 0,1,2,3,4,5,6,7.
  PYTHON_BIN                                       Python executable; defaults to python3.
  MINIMAX_H3_MODEL_ROOT                            Optional local MiniMax-H3 repository root.
  VLLM_TEST_MINIMAX_H3_FL2VA_MODEL                 Optional local FL2VA model path.
  VLLM_TEST_MINIMAX_H3_REF2VA_MODEL                Optional local Ref2VA model path.
  VLLM_TEST_MINIMAX_H3_NPU_ATTENTION_BACKEND       Defaults to FLASH_ATTN.
  VLLM_TEST_MINIMAX_H3_NPU_SSIM_THRESHOLD          Defaults to the CUDA E2E threshold, 0.97.
  VLLM_TEST_MINIMAX_H3_NPU_PSNR_THRESHOLD          Defaults to the CUDA E2E threshold, 34.0.

Examples:
  ./run_minimax_h3_npu_accuracy.sh i2va
  MINIMAX_H3_MODEL_ROOT=/models/MiniMax-H3 ./run_minimax_h3_npu_accuracy.sh all
EOF
}

case "${1:-all}" in
  -h|--help)
    usage
    exit 0
    ;;
  all|i2va|ref2va)
    TEST_CASE="${1:-all}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    ;;
  *)
    echo "Unknown test case: $1" >&2
    usage >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"

for binary in "${PYTHON_BIN}" ffmpeg ffprobe; do
  if ! command -v "${binary}" >/dev/null 2>&1; then
    echo "Required command not found: ${binary}" >&2
    exit 1
  fi
done

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a NPU_DEVICES <<<"${ASCEND_RT_VISIBLE_DEVICES}"
if [[ ${#NPU_DEVICES[@]} -ne ${EXPECTED_NPU_COUNT} ]]; then
  echo "MiniMax-H3 NPU accuracy requires exactly ${EXPECTED_NPU_COUNT} visible NPUs;" >&2
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-3600}"
export VLLM_TEST_MINIMAX_H3_NPU_ATTENTION_BACKEND="${VLLM_TEST_MINIMAX_H3_NPU_ATTENTION_BACKEND:-FLASH_ATTN}"
# Keep this accuracy run on the dense MindIE-SD path. LaserAttention is an
# independent optimization and is not validated for H3 FL2VA/Ref2VA.
unset MINDIE_SD_FA_TYPE

if [[ -n "${MINIMAX_H3_MODEL_ROOT:-}" ]]; then
  export VLLM_TEST_MINIMAX_H3_FL2VA_MODEL="${VLLM_TEST_MINIMAX_H3_FL2VA_MODEL:-${MINIMAX_H3_MODEL_ROOT}/FL2VA}"
  export VLLM_TEST_MINIMAX_H3_REF2VA_MODEL="${VLLM_TEST_MINIMAX_H3_REF2VA_MODEL:-${MINIMAX_H3_MODEL_ROOT}/Ref2VA}"
fi

validate_model_path() {
  local label="$1"
  local model_path="$2"
  if [[ ! -f "${model_path}/model_index.json" ]]; then
    echo "${label} model_index.json not found under: ${model_path}" >&2
    exit 1
  fi
}

if [[ "${TEST_CASE}" != "ref2va" && -n "${VLLM_TEST_MINIMAX_H3_FL2VA_MODEL:-}" ]]; then
  validate_model_path "FL2VA" "${VLLM_TEST_MINIMAX_H3_FL2VA_MODEL}"
fi
if [[ "${TEST_CASE}" != "i2va" && -n "${VLLM_TEST_MINIMAX_H3_REF2VA_MODEL:-}" ]]; then
  validate_model_path "Ref2VA" "${VLLM_TEST_MINIMAX_H3_REF2VA_MODEL}"
fi

EXPECTED_NPU_COUNT="${EXPECTED_NPU_COUNT}" "${PYTHON_BIN}" -c '
import os
import torch
import torch_npu  # noqa: F401

expected = int(os.environ["EXPECTED_NPU_COUNT"])
if os.environ["VLLM_TEST_MINIMAX_H3_NPU_ATTENTION_BACKEND"].upper() == "FLASH_ATTN":
    import mindiesd  # noqa: F401
if not torch.npu.is_available():
    raise SystemExit("torch_npu is installed, but torch.npu.is_available() is false")
actual = torch.npu.device_count()
if actual != expected:
    raise SystemExit(f"Expected {expected} visible NPUs, found {actual}")
print(f"NPU preflight passed: {actual} devices")
'

PYTEST_FILTER=()
case "${TEST_CASE}" in
  i2va)
    PYTEST_FILTER=(-k i2va)
    ;;
  ref2va)
    PYTEST_FILTER=(-k ref2va)
    ;;
esac

echo "Running MiniMax-H3 ${TEST_CASE} NPU accuracy test"
echo "  devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  attention_backend=${VLLM_TEST_MINIMAX_H3_NPU_ATTENTION_BACKEND}"
echo "  artifacts=tests/e2e/accuracy/artifacts/minimax_h3_*_npu"

exec "${PYTHON_BIN}" -m pytest -s -v "${TEST_FILE}" \
  -m "full_model and npu and A3" \
  --run-level full_model \
  "${PYTEST_FILTER[@]}" \
  "$@"
