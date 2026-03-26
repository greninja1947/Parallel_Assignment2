#!/bin/bash
# ============================================================
# HPC Test Runner for Assignment 2
# Usage: ./test <knn|kmeans|approx_knn|all>
#
# ONE-TIME SETUP (do this first):
#   ssh-keygen -t ed25519          # accept defaults, no passphrase
#   ssh-copy-id cs1230459@hpc.iitd.ac.in
#
# If ssh-copy-id fails, do it manually:
#   cat ~/.ssh/id_ed25519.pub | ssh cs1230459@hpc.iitd.ac.in "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
#
# Without SSH keys, the script still works but prompts for
# password once per run (via SSH ControlMaster).
# ============================================================

set -uo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: ./test.sh <knn|kmeans|approx_knn|all>"
    echo "       ./test.sh fetch <knn|kmeans|approx_knn>   (download results from completed job)"
    exit 1
fi

# Handle "fetch" subcommand — just download results, no resubmit
FETCH_ONLY=false
if [[ "$MODE" == "fetch" ]]; then
    FETCH_ONLY=true
    MODE="${2:-}"
    if [[ -z "$MODE" ]]; then
        echo "Usage: ./test.sh fetch <knn|kmeans|approx_knn>"
        exit 1
    fi
fi

# Validate mode
MODES=()
if [[ "$MODE" == "all" ]]; then
    MODES=("knn" "kmeans" "approx_knn")
else
    if [[ "$MODE" != "knn" && "$MODE" != "kmeans" && "$MODE" != "approx_knn" ]]; then
        echo "Error: mode must be knn, kmeans, approx_knn, or all"
        exit 1
    fi
    MODES=("$MODE")
fi

# ---- Configuration ----
HPC_USER="cs1230459"
HPC_HOST="hpc.iitd.ac.in"
HPC_DIR="Assignment_2"
PROJECT="col7880.cs1230459.course"
POLL_INTERVAL=5
TIMEOUT=900   # 15 min (includes queue wait)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check input.txt
if [[ ! -f "input.txt" ]]; then
    echo "Error: input.txt not found. Generate one with:"
    echo "  python3 dataset_generator.py 1000 10 20"
    exit 1
fi

# ---- SSH ControlMaster (single password prompt per session) ----
SSH_CTRL="/tmp/ssh_hpc_a2"
SSH="ssh -o ControlMaster=auto -o ControlPath=${SSH_CTRL} -o ControlPersist=300 -o ServerAliveInterval=30"
SCP="scp -o ControlPath=${SSH_CTRL}"

# bash -l forces login shell so PATH includes PBS tools (qsub etc.)
run_hpc() { ${SSH} ${HPC_USER}@${HPC_HOST} "bash -l -c '$*'"; }

echo "Connecting to HPC..."
run_hpc "mkdir -p ~/${HPC_DIR}" || { echo "SSH connection failed."; exit 1; }

if [[ "$FETCH_ONLY" == false ]]; then
    # Files to upload (add more source files here if needed)
    UPLOAD_FILES=(Makefile submission.cu input.txt)

    echo "Uploading: ${UPLOAD_FILES[*]}"
    ${SCP} "${UPLOAD_FILES[@]}" ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/
fi

for CURRENT_MODE in "${MODES[@]}"; do
    echo ""
    echo "=========================================="
    if [[ "$FETCH_ONLY" == true ]]; then
        echo "  Fetching: ${CURRENT_MODE}"
    else
        echo "  Testing: ${CURRENT_MODE}"
    fi
    echo "=========================================="

    if [[ "$FETCH_ONLY" == false ]]; then
        # Clean previous run markers
        run_hpc "rm -f ~/${HPC_DIR}/job.done ~/${HPC_DIR}/job.log ~/${HPC_DIR}/${CURRENT_MODE}.txt"

        # Create PBS script locally and upload
        PBS_FILE="/tmp/a2_job_${CURRENT_MODE}.pbs"
        cat > "${PBS_FILE}" << PBSEOF
#!/bin/bash
#PBS -N a2_${CURRENT_MODE}
#PBS -P ${PROJECT}
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:10:00
#PBS -j oe

cd \$HOME/${HPC_DIR}
module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

{
    echo "=== BUILD ==="
    make
    BUILD_RC=\$?
    if [ \$BUILD_RC -ne 0 ]; then
        echo "BUILD FAILED (exit \$BUILD_RC)"
    else
        echo "=== RUN: ./a2 input.txt ${CURRENT_MODE} ==="
        time ./a2 input.txt ${CURRENT_MODE}
        echo "=== EXIT: \$? ==="
    fi
} > job.log 2>&1

touch job.done
PBSEOF

        ${SCP} "${PBS_FILE}" ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/job.pbs
        rm -f "${PBS_FILE}"

        # Submit job
        echo "Submitting PBS job..."
        JOB_ID=$(run_hpc "cd ~/${HPC_DIR} && qsub job.pbs" 2>&1)
        if [[ $? -ne 0 ]]; then
            echo "qsub failed: ${JOB_ID}"
            continue
        fi
        echo "  Job ID: ${JOB_ID}"

        # Poll for completion
        SECONDS=0
        while ! run_hpc "test -f ~/${HPC_DIR}/job.done" 2>/dev/null; do
            if (( SECONDS > TIMEOUT )); then
                echo ""
                echo "TIMEOUT after ${TIMEOUT}s. Job may still be queued."
                echo "  Fetch results later:  ./test.sh fetch ${CURRENT_MODE}"
                echo "  Check manually: ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qstat -u ${HPC_USER}\"'"
                echo "  Kill job:       ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qdel ${JOB_ID}\"'"
                continue 2
            fi
            printf "\r  Waiting... %ds" $SECONDS
            sleep ${POLL_INTERVAL}
        done
        printf "\r  Completed in %ds            \n" $SECONDS
    fi

    # ---- Download results (runs for both normal and fetch mode) ----

    # Download and show log
    if ${SCP} ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/job.log "./job_${CURRENT_MODE}.log" 2>/dev/null; then
        echo ""
        echo "--- Build & Run Log ---"
        cat "./job_${CURRENT_MODE}.log"
        echo ""
    else
        echo "No job.log found on HPC. Job may not have completed yet."
        echo "  Check: ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qstat -u ${HPC_USER}\"'"
        continue
    fi

    # Download output file
    if ${SCP} ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/${CURRENT_MODE}.txt "./${CURRENT_MODE}.txt" 2>/dev/null; then
        LINES=$(wc -l < "./${CURRENT_MODE}.txt" | tr -d ' ')
        echo "${CURRENT_MODE}.txt downloaded (${LINES} lines)"
        echo "First 3 lines:"
        head -3 "./${CURRENT_MODE}.txt"
        echo "..."
    else
        echo "ERROR: ${CURRENT_MODE}.txt not found on HPC. Check log above."
        continue
    fi

    # Compare with sequential output
    # Approximate KNN should be evaluated against the exact KNN baseline.
    if [[ "${CURRENT_MODE}" == "approx_knn" ]]; then
        SEQ_FILE="./knn_seq.txt"
    else
        SEQ_FILE="./${CURRENT_MODE}_seq.txt"
    fi
    if [[ -f "$SEQ_FILE" ]]; then
        echo ""
        echo "--- Comparison with $(basename "$SEQ_FILE") ---"
        if diff -q "./${CURRENT_MODE}.txt" "$SEQ_FILE" > /dev/null 2>&1; then
            echo "PASS: Exact match!"
        else
            DIFF_LINES=$(diff "./${CURRENT_MODE}.txt" "$SEQ_FILE" | grep -c "^[<>]" || true)
            echo "DIFF: ${DIFF_LINES} lines differ. First differences:"
            diff "./${CURRENT_MODE}.txt" "$SEQ_FILE" | head -20
        fi

        # MAE loss
        if [[ -f "mae_loss.py" ]]; then
            echo ""
            echo "--- MAE Loss ---"
            python3 mae_loss.py "./${CURRENT_MODE}.txt" "$SEQ_FILE" 2>&1 || true
        fi
    else
        echo ""
        echo "(No ${CURRENT_MODE}_seq.txt found -- skipping comparison)"
    fi
done

echo ""
echo "Done!"
