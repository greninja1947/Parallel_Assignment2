#!/bin/bash

set -euo pipefail

MODE="${1:-all}"

if [[ "$MODE" == "all" ]]; then
    MODES=("knn" "kmeans" "approx_knn")
elif [[ "$MODE" == "knn" || "$MODE" == "kmeans" || "$MODE" == "approx_knn" ]]; then
    MODES=("$MODE")
else
    echo "Usage: bash colab_run.sh <knn|kmeans|approx_knn|all>"
    exit 1
fi

if [[ ! -f "input.txt" ]]; then
    echo "input.txt not found in $(pwd)"
    exit 1
fi

SEQ_CXX="${SEQ_CXX:-g++}"
NVCC_BIN="${NVCC_BIN:-nvcc}"
SEQ_FLAGS=(-O2 -std=c++17)
NVCC_FLAGS=(-O3 -Xcompiler -fopenmp)

if [[ -n "${CUDA_ARCH:-}" ]]; then
    NVCC_FLAGS+=("-arch=${CUDA_ARCH}")
fi

echo "=== BUILD: seq ==="
"${SEQ_CXX}" "${SEQ_FLAGS[@]}" -o seq submission.cpp

echo "=== BUILD: a2 ==="
"${NVCC_BIN}" "${NVCC_FLAGS[@]}" -o a2 submission.cu

need_exact_knn=false
for current_mode in "${MODES[@]}"; do
    if [[ "${current_mode}" == "knn" || "${current_mode}" == "approx_knn" ]]; then
        need_exact_knn=true
    fi
done

if [[ "${need_exact_knn}" == true ]]; then
    echo "=== CPU BASELINE: knn ==="
    ./seq input.txt knn
fi

for current_mode in "${MODES[@]}"; do
    echo ""
    echo "=== MODE: ${current_mode} ==="

    if [[ "${current_mode}" == "kmeans" ]]; then
        echo "Running CPU baseline for kmeans..."
        ./seq input.txt kmeans
        baseline_file="kmeans_seq.txt"
    elif [[ "${current_mode}" == "knn" ]]; then
        baseline_file="knn_seq.txt"
    else
        baseline_file="knn_seq.txt"
    fi

    echo "Running CUDA executable..."
    ./a2 input.txt "${current_mode}"

    if [[ "${current_mode}" == "approx_knn" ]]; then
        echo "--- MAE vs exact knn baseline ---"
        python3 mae_loss.py "./approx_knn.txt" "${baseline_file}"
    else
        echo "--- Diff vs $(basename "${baseline_file}") ---"
        if diff -q "./${current_mode}.txt" "${baseline_file}" > /dev/null 2>&1; then
            echo "PASS: Exact match"
        else
            echo "Outputs differ. First differences:"
            diff "./${current_mode}.txt" "${baseline_file}" | head -20 || true
        fi
    fi
done
